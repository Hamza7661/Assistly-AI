import json
import logging
import re
import time
import secrets
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager

from pydantic import BaseModel
from .config import settings
from .services.context_service import ContextService
from .services.lead_service import LeadService
from .services.email_validation_service import EmailValidationService
from .services.phone_validation_service import PhoneValidationService
from .services.whatsapp_service import WhatsAppService
from .services.voice_agent_service import VoiceAgentService
from .services.rag_service import RAGService
from .services.conversation_state import FlowController, ConversationState
from .services.response_generator import ResponseGenerator
from .services.data_extractors import DataExtractor
from .utils.phone_utils import format_phone_number
from .utils.language_utils import detect_language, get_language_name_for_prompt
from .utils.response_strings import get_string

from twilio.twiml.voice_response import VoiceResponse


logger = logging.getLogger("assistly")
logging.basicConfig(level=logging.INFO)

# In-memory storage for WhatsApp conversations (session-based)
# Key: session_id, Value: {phone, history, email_state, phone_state, context, user_id, created_at, last_activity}
whatsapp_sessions: Dict[str, Dict[str, Any]] = {}

# Mapping: phone number -> current active session_id
phone_to_session: Dict[str, str] = {}

# Session timeout from environment variable (default: 5 minutes)
SESSION_TIMEOUT = settings.session_timeout_seconds

# Voice agent sessions
voice_agent_service = VoiceAgentService(settings)

def get_or_create_session(user_phone: str) -> tuple[str, bool]:
    """Get existing session or create new one. Returns (session_id, is_new)"""
    current_time = time.time()
    
    # Check if user has an active session
    if user_phone in phone_to_session:
        session_id = phone_to_session[user_phone]
        
        # Check if session is still valid (not expired)
        if session_id in whatsapp_sessions:
            session = whatsapp_sessions[session_id]
            last_activity = session.get("last_activity", 0)
            
            # If session expired, create new one
            if current_time - last_activity > SESSION_TIMEOUT:
                logger.info(f"Session {session_id} expired for {user_phone}, creating new session")
                # Clean up old session
                del whatsapp_sessions[session_id]
                del phone_to_session[user_phone]
            else:
                # Session is valid, update last activity
                session["last_activity"] = current_time
                return session_id, False
    
    # Create new session
    session_id = secrets.token_urlsafe(16)
    phone_to_session[user_phone] = session_id
    logger.info(f"Created new session {session_id} for {user_phone}")
    
    return session_id, True

def cleanup_expired_sessions():
    """Remove expired sessions to prevent memory leaks"""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, session in whatsapp_sessions.items():
        last_activity = session.get("last_activity", 0)
        if current_time - last_activity > SESSION_TIMEOUT:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        session = whatsapp_sessions[session_id]
        phone = session.get("phone")
        if phone and phone_to_session.get(phone) == session_id:
            del phone_to_session[phone]
        del whatsapp_sessions[session_id]
        logger.info(f"Cleaned up expired session {session_id}")
    
    return len(expired_sessions)

async def background_session_cleanup():
    """Background task to periodically clean up expired sessions"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            expired_count = cleanup_expired_sessions()
            if expired_count > 0:
                logger.info(f"Background cleanup: removed {expired_count} expired session(s)")
                logger.info(f"Active sessions: {len(whatsapp_sessions)}")
        except Exception as e:
            logger.error(f"Error in background session cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Start background cleanup task
    cleanup_task = asyncio.create_task(background_session_cleanup())
    logger.info("Started background session cleanup task")
    
    yield
    
    # Shutdown: Cancel background task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logger.info("Background session cleanup task cancelled")

app = FastAPI(title="Assistly AI Chatbot WS", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring performance."""
    from app.utils.cache_utils import get_cache_stats
    return {
        "status": "ok",
        "cache": get_cache_stats(),
        "sessions": {
            "active_whatsapp": len(whatsapp_sessions),
            "phone_mappings": len(phone_to_session)
        }
    }


class InvalidateSessionsBody(BaseModel):
    twilio_phone: str


@app.post("/api/v1/whatsapp/invalidate-sessions")
async def invalidate_whatsapp_sessions(
    body: InvalidateSessionsBody,
    x_invalidate_sessions_secret: Optional[str] = Header(default=None, alias="X-Invalidate-Sessions-Secret"),
) -> Dict[str, Any]:
    """
    Clear all WhatsApp sessions for a given Twilio number so the next message fetches fresh context
    (e.g. after switching which app 'uses this number'). Called by backend when setUsesTwilioNumber succeeds.
    Also clears cached translations for affected apps.
    """
    # Use same signing secret as third-party API authentication (TP_SIGN_SECRET)
    # Only require the secret when AI has it set in .env. When unset, all requests are allowed.
    if settings.tp_sign_secret:
        if x_invalidate_sessions_secret != settings.tp_sign_secret:
            raise HTTPException(status_code=401, detail="Invalid or missing secret")
    clean_phone = (body.twilio_phone or "").replace("whatsapp:", "").strip()
    if not clean_phone:
        raise HTTPException(status_code=400, detail="twilio_phone required")
    
    removed = []
    app_ids_to_invalidate = set()
    
    for session_id, session in list(whatsapp_sessions.items()):
        session_phone = (session.get("twilio_phone") or "").replace("whatsapp:", "").strip()
        if session_phone == clean_phone:
            # Track app_id for cache invalidation
            app_id = session.get("app_id")
            if app_id:
                app_ids_to_invalidate.add(str(app_id))
            
            phone = session.get("phone")
            if phone and phone_to_session.get(phone) == session_id:
                del phone_to_session[phone]
            del whatsapp_sessions[session_id]
            removed.append(session_id)
    
    # Invalidate cached translations and greetings for affected apps
    if app_ids_to_invalidate:
        from app.utils.cache_utils import invalidate_app_cache
        for app_id in app_ids_to_invalidate:
            invalidate_app_cache(app_id)
        logger.info("Invalidated cache for %d app(s): %s", len(app_ids_to_invalidate), app_ids_to_invalidate)
    
    logger.info("Invalidated WhatsApp sessions for twilio_phone=%s, removed %d session(s)", clean_phone, len(removed))
    return {"status": "ok", "twilio_phone": clean_phone, "removed_sessions": len(removed)}


def _maybe_parse_json(text: str) -> Optional[Dict]:
    """Parse JSON from text if it looks like JSON."""
    content = text.strip()
    if not (content.startswith("{") and content.endswith("}")):
        return None
    try:
        return json.loads(content)
    except Exception:
        return None


def _validate_email_verification(email_validation_state: Dict) -> bool:
    """Validate that email has been verified."""
    return email_validation_state.get("otp_verified", False)

def _validate_phone_verification(phone_validation_state: Dict) -> bool:
    """Validate that phone has been verified."""
    return phone_validation_state.get("otp_verified", False)

def _is_valid_email(email: str) -> bool:
    """Validate email format more strictly."""
    import re
    # More comprehensive email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False
    
    # Additional checks
    if len(email) > 254:  # RFC 5321 limit
        return False
    
    local_part, domain = email.split('@', 1)
    if len(local_part) > 64:  # RFC 5321 limit
        return False
    
    if domain.count('.') < 1:  # Must have at least one dot
        return False
    
    return True

def _extract_otp_from_text(text: str) -> str:
    """Extract 6-digit OTP code from user text."""
    import re
    # Look for 6-digit numbers in the text (including those starting with 0)
    # Try multiple patterns to catch different formats, but ensure they are standalone
    patterns = [
        r'\b\d{6}\b',  # 6 digits with word boundaries (standalone)
        r'(?:code|pin|otp|verification).*?(\d{6})',  # After keywords
        r'^(\d{6})$',  # Exactly 6 digits (entire text)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Return the first group if it exists, otherwise the full match
            return match.group(1) if match.groups() else match.group(0)
    
    return None

def _detect_retry_request(reply: str) -> Tuple[Optional[str], Optional[str]]:
    """Detect retry requests from GPT's special response phrases and extract email/phone"""
    logger.info(f"Checking GPT response for retry phrases: {reply}")
    
    # Check for SEND_EMAIL format
    if 'SEND_EMAIL:' in reply:
        import re
        email_match = re.search(r'SEND_EMAIL:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', reply)
        if email_match:
            email = email_match.group(1)
            logger.info(f"Detected SEND_EMAIL: {email}")
            return ('send_email', email)
    
    # Check for SEND_PHONE format
    if 'SEND_PHONE:' in reply:
        import re
        phone_match = re.search(r'SEND_PHONE:\s*([\d\s\+\-\(\)]{10,})', reply)
        if phone_match:
            phone = phone_match.group(1).strip()
            logger.info(f"Detected SEND_PHONE: {phone}")
            return ('send_phone', phone)
    
    # Check for CHANGE_EMAIL_REQUESTED format
    if 'CHANGE_EMAIL_REQUESTED:' in reply:
        import re
        email_match = re.search(r'CHANGE_EMAIL_REQUESTED:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', reply)
        if email_match:
            email = email_match.group(1)
            logger.info(f"Detected CHANGE_EMAIL_REQUESTED: {email}")
            return ('change_email', email)
        elif 'CHANGE_EMAIL_REQUESTED' in reply:
            logger.info("Detected CHANGE_EMAIL_REQUESTED (no email provided)")
            return ('change_email', None)
    
    # Check for CHANGE_PHONE_REQUESTED format
    if 'CHANGE_PHONE_REQUESTED:' in reply:
        import re
        phone_match = re.search(r'CHANGE_PHONE_REQUESTED:\s*([\d\s\+\-\(\)]{10,})', reply)
        if phone_match:
            phone = phone_match.group(1).strip()
            logger.info(f"Detected CHANGE_PHONE_REQUESTED: {phone}")
            return ('change_phone', phone)
        elif 'CHANGE_PHONE_REQUESTED' in reply:
            logger.info("Detected CHANGE_PHONE_REQUESTED (no phone provided)")
            return ('change_phone', None)
    
    if 'RETRY_OTP_REQUESTED' in reply:
        logger.info("Detected RETRY_OTP_REQUESTED")
        return ('resend_otp', None)
    
    logger.info("No retry phrases detected")
    return (None, None)

def _extract_email_from_text(text: str) -> str:
    import re
    # Look for email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

def _extract_phone_from_text(text: str) -> str:
    import re
    # Look for phone number patterns (digits with possible separators)
    phone_patterns = [
        r'\b\d{10,15}\b',  # 10-15 digits
        r'\+\d{10,15}\b',  # + followed by 10-15 digits
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US format
        r'\b\d{4}[-.\s]?\d{3}[-.\s]?\d{3}\b',  # Some international formats
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            # Clean the phone number (remove separators)
            phone = re.sub(r'[-.\s]', '', match.group(0))
            # Format using phone utils
            return format_phone_number(phone)
    
    return None

def _extract_buttons_from_response(response: str) -> Tuple[str, List[Dict[str, str]]]:
    """Extract buttons from response and return cleaned text + button data"""
    buttons = []
    button_pattern = r'<\s*button[^>]*>\s*([^<]+?)\s*</\s*button[^>]*>'
    button_matches = re.findall(button_pattern, response, re.IGNORECASE | re.DOTALL)
    
    seen_buttons = set()
    for button_text in button_matches:
        clean_text = button_text.strip().lstrip('>').strip()
        if clean_text and clean_text.lower() not in seen_buttons:
            seen_buttons.add(clean_text.lower())
            buttons.append({
                "id": f"button_{len(buttons) + 1}",
                "title": clean_text
            })
    
    cleaned_response = re.sub(button_pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
    cleaned_response = re.sub(r'<\s*button[^>]*>.*?</\s*button[^>]*>', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
    cleaned_response = re.sub(r'<\s*button[^>]*>.*?$', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response, buttons

def _convert_lead_types_to_whatsapp_buttons(lead_types: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert lead types to WhatsApp button format"""
    buttons = []
    for i, lead_type in enumerate(lead_types, 1):
        text = lead_type.get("text", str(lead_type))
        value = lead_type.get("value", text)
        buttons.append({
            "id": f"lead_type_{value}",
            "title": text
        })
    return buttons

def _get_root_workflow(workflows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the first active root workflow"""
    if not workflows:
        return None
    for workflow in workflows:
        if workflow.get("isRoot") and workflow.get("isActive"):
            return workflow
    return None

def _find_workflow_by_id(workflows: List[Dict[str, Any]], workflow_id: str) -> Optional[Dict[str, Any]]:
    """Find a workflow by its ID"""
    if not workflows or not workflow_id:
        return None
    for workflow in workflows:
        if workflow.get("_id") == workflow_id:
            return workflow
    return None

def _match_option_to_response(user_text: str, options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match user response to workflow option (simple text matching for now)"""
    if not options:
        return None
    
    user_text_lower = user_text.strip().lower()
    
    # First try exact match
    for opt in options:
        if opt.get("text", "").lower() == user_text_lower:
            return opt
    
    # Try partial match
    for opt in options:
        opt_text = opt.get("text", "").lower()
        if opt_text in user_text_lower or user_text_lower in opt_text:
            return opt
    
    return None

def _process_workflow_response(user_text: str, current_workflow: Dict[str, Any], workflows: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Process user response within a workflow
    Returns: (option_text, next_question, is_terminal)
    """
    options = current_workflow.get("options", [])
    matched_option = _match_option_to_response(user_text, options)
    
    if not matched_option:
        return (None, None, False)
    
    option_text = matched_option.get("text", "")
    is_terminal = matched_option.get("isTerminal", False)
    next_question_id = matched_option.get("nextQuestionId")
    
    if is_terminal:
        return (option_text, None, True)
    
    if next_question_id:
        next_workflow = _find_workflow_by_id(workflows, next_question_id)
        if next_workflow:
            return (option_text, next_workflow.get("question"), False)
    
    return (option_text, None, True)

def _convert_services_to_whatsapp_list(services: List[Any], service_plans: List[Any] = None) -> List[Dict[str, Any]]:
    """Convert services and service plans to WhatsApp list format"""
    sections = []
    items = []
    
    # Add regular services
    for i, service in enumerate(services, 1):
        if isinstance(service, dict):
            title = service.get("name", service.get("title", str(service)))
            description = service.get("description", "")
        else:
            title = str(service)
            description = ""
        
        items.append({
            "id": f"service_{i}",
            "title": title,
            "description": description
        })
    
    # Add service plans
    if service_plans:
        for i, plan in enumerate(service_plans, len(services) + 1):
            if isinstance(plan, dict):
                title = plan.get("question", plan.get("title", str(plan)))
                description = plan.get("description", "")
            else:
                title = str(plan)
                description = ""
            
            items.append({
                "id": f"service_plan_{i}",
                "title": title,
                "description": description
            })
    
    if items:
        sections.append({
            "title": "Services & Service Plans",
            "rows": items
        })
    
    return sections

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    app_id: Optional[str] = websocket.query_params.get("app_id")
    user_id: Optional[str] = websocket.query_params.get("user_id")

    # App-wise: accept app_id (embed widget) or legacy user_id
    if app_id:
        identifier = app_id
        fetch_by_app = True
    elif user_id:
        identifier = user_id
        fetch_by_app = False
    else:
        await websocket.send_json({"type": "error", "content": "Missing app_id or user_id in query params"})
        await websocket.close(code=1008)
        return

    context_service = ContextService(settings)
    lead_service = LeadService(settings)
    rag_service = RAGService(settings)
    email_validation_service = EmailValidationService(settings)
    
    # Initialize OpenAI client for phone formatting
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    phone_validation_service = PhoneValidationService(
        settings, 
        openai_client=openai_client,
        gpt_model=settings.gpt_model
    )

    try:
        if fetch_by_app:
            context = await context_service.fetch_context_by_app(identifier)
        else:
            context = await context_service.fetch_user_context(identifier)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch user context: %s", exc)
        await websocket.send_json({
            "type": "error",
            "content": "Unable to fetch user context. Please try again shortly.",
        })
        await websocket.close(code=1011)
        return

    # For OTP, leads, etc.: use user id from context (app owner when app_id was used)
    user_id = str(context.get("user", {}).get("id") or identifier)

    # Initialize production-grade components
    flow_controller = FlowController(context)
    flow_controller.set_whatsapp(False)
    
    response_generator = ResponseGenerator(settings, rag_service)
    response_generator.set_profession(str(context.get("profession") or "Business"))
    response_generator.set_channel("web")
    
    extractor = DataExtractor()
    conversation_history: List[Dict[str, str]] = []
    
    # Build RAG vector store
    rag_service.build_vector_store(context)
    
    flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)

    # Send greeting (from DB) + lead types as buttons as soon as widget opens; no first user message required
    initial_reply = await response_generator.generate_greeting(context, channel="web", first_message=None)
    conversation_history.append({"role": "assistant", "content": initial_reply})
    await websocket.send_json({"type": "bot", "content": initial_reply})

    try:
        while True:
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                user_text = parsed.get("content") or parsed.get("text") or data
            except json.JSONDecodeError:
                parsed = {}
                user_text = data

            # ── Handle file_upload messages from the chat widget ──────────────
            if isinstance(parsed, dict) and parsed.get("type") == "file_upload":
                filename = parsed.get("filename", "file")
                download_url = parsed.get("downloadUrl", "")
                content_type = parsed.get("contentType", "")

                # Acknowledge the file upload and encourage the workflow to continue
                file_ack_msg = (
                    f"📎 Thank you for uploading **{filename}**. "
                    f"Our team will review it."
                )
                if download_url:
                    file_ack_msg += (
                        f'\n<file url="{download_url}" name="{filename}">View / Download {filename}</file>'
                    )

                # Record in conversation history
                conversation_history.append({"role": "user", "content": f"[File uploaded: {filename}]"})
                conversation_history.append({"role": "assistant", "content": file_ack_msg})

                await websocket.send_json({"type": "bot", "content": file_ack_msg})

                # If in workflow, record a placeholder answer so we advance to the next question
                wm = flow_controller.workflow_manager
                if wm and wm.is_active and not wm.is_workflow_complete():
                    has_more = wm.record_answer(f"[File: {filename}]")
                    if has_more:
                        next_q = wm.get_current_question()
                        if next_q:
                            next_text = wm.format_question_with_options(next_q)
                            conversation_history.append({"role": "assistant", "content": next_text})
                            await websocket.send_json({"type": "bot", "content": next_text})
                    else:
                        # Workflow done – move state forward
                        from app.services.conversation_state import ConversationState
                        flow_controller.collected_data["workflowAnswers"] = wm.get_workflow_answers()
                        flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                        next_reply = await response_generator.generate_response(
                            flow_controller, "[File uploaded]", conversation_history, context
                        )
                        conversation_history.append({"role": "assistant", "content": next_reply})
                        await websocket.send_json({"type": "bot", "content": next_reply})
                continue
            # ─────────────────────────────────────────────────────────────────

            if not user_text or not str(user_text).strip():
                continue
            if str(user_text).strip().lower() in {"ping", "pong", "keepalive", "heartbeat"}:
                continue

            # Add user message to history and process (e.g. lead type button click → services/workflows)
            conversation_history.append({"role": "user", "content": user_text})

            # Detect response language from current message so bot replies in same language
            lang_code = detect_language(str(user_text))
            response_generator.set_response_language(get_language_name_for_prompt(lang_code))
            
            # Production-grade state machine flow (workflows are now handled by response_generator after service plan selection)
            current_state = flow_controller.state
            logger.info(f"Current state: {current_state.value}")
            
            # Handle OTP verification states using state machine
            if current_state == ConversationState.EMAIL_OTP_VERIFICATION:
                # First, check if user is requesting to change email or resend OTP
                # Let ResponseGenerator handle it first to detect change/resend requests
                temp_reply = await response_generator.generate_response(
                    flow_controller, user_text, conversation_history, context
                )
                
                # Check if it's a change/resend request
                retry_type, extracted_value = _detect_retry_request(temp_reply)
                if retry_type == 'resend_otp' and flow_controller.otp_state["email_sent"] and not flow_controller.otp_state["email_verified"]:
                    # Resend OTP to existing email
                    email = flow_controller.collected_data.get("leadEmail")
                    customer_name = flow_controller.collected_data.get("leadName", "Customer")
                    if email:
                        ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                        reply = get_string("otp_resend", lang_code, email) if ok else get_string("otp_resend_fail_email", lang_code)
                    else:
                        reply = get_string("no_email", lang_code)
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
                elif retry_type == 'change_email':
                    # Reset email validation state to allow new email
                    flow_controller.otp_state["email_sent"] = False
                    flow_controller.otp_state["email_verified"] = False
                    
                    if extracted_value:
                        # Response generator provided the new email
                        email = extracted_value
                        flow_controller.collected_data["leadEmail"] = email
                        customer_name = flow_controller.collected_data.get("leadName", "Customer")
                        
                        logger.info(f"WebSocket: Sending OTP to NEW email: {email}")
                        ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                        if ok:
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        reply = get_string("perfect_otp_sent_email", lang_code, email) if ok else get_string("found_email_cant_send", lang_code)
                    else:
                        reply = get_string("no_problem_email", lang_code)
                        flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
                else:
                    # Check if it's an OTP code
                    otp_code = extractor.extract_otp_code(user_text)
                    if otp_code:
                        email = flow_controller.collected_data.get("leadEmail")
                        if email:
                            ok, _ = await email_validation_service.verify_otp(user_id, email, otp_code)
                            if ok:
                                flow_controller.otp_state["email_verified"] = True
                                flow_controller.transition_to(flow_controller.get_next_state())
                                # Get AI response to continue
                                reply = await response_generator.generate_response(
                                    flow_controller, "Email verified", conversation_history, context
                                )
                            else:
                                reply = get_string("otp_wrong_code", lang_code)
                        else:
                            reply = get_string("otp_please_enter", lang_code)
                    else:
                        # Not an OTP and not a change/resend request - use ResponseGenerator's response
                        reply = temp_reply
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
            
            if current_state == ConversationState.PHONE_OTP_VERIFICATION:
                otp_code = extractor.extract_otp_code(user_text)
                if otp_code:
                    phone = flow_controller.collected_data.get("leadPhoneNumber")
                    if phone:
                        ok, _ = await phone_validation_service.verify_sms_otp(user_id, phone, otp_code)
                        if ok:
                            flow_controller.otp_state["phone_verified"] = True
                            flow_controller.transition_to(flow_controller.get_next_state())
                            # Generate JSON if all data collected (with summary, description, and history)
                            if flow_controller.can_generate_json():
                                json_str = await response_generator.generate_lead_json(flow_controller, conversation_history)
                                json_data = _maybe_parse_json(json_str)
                                if not json_data:
                                    # Fallback if JSON parsing fails
                                    json_data = flow_controller.get_json_data(conversation_history)
                                # Add appId if available (for app-scoped leads)
                                if app_id:
                                    json_data["appId"] = app_id
                                try:
                                    ok, _ = await lead_service.create_public_lead(user_id, json_data)
                                    final_msg = get_string("final_success", lang_code) if ok else get_string("final_fallback", lang_code)
                                except Exception:
                                    final_msg = get_string("final_fallback", lang_code)
                                
                                integration = context.get("integration") or {}
                                if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                                    review_url_ws = integration["googleReviewUrl"].strip()
                                    await websocket.send_json({
                                        "type": "review_prompt",
                                        "content": get_string("review_prompt", lang_code, review_url_ws),
                                        "reviewUrl": review_url_ws,
                                    })
                                await websocket.send_json({"type": "bot", "content": final_msg})
                                await websocket.close(code=1000)
                                break
                            else:
                                # Get AI response to continue
                                reply = await response_generator.generate_response(
                                    flow_controller, "Phone verified", conversation_history, context
                                )
                        else:
                            reply = get_string("otp_wrong_code", lang_code)
                    else:
                        reply = get_string("otp_please_enter", lang_code)
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
            
            # Generate response using state machine
            reply = await response_generator.generate_response(
                flow_controller, user_text, conversation_history, context
            )
            
            # Handle special responses from response generator
            # Check for answer + SEND_EMAIL/PHONE format (when user asks question while providing email/phone)
            if "|||SEND_EMAIL:" in reply:
                parts = reply.split("|||SEND_EMAIL:", 1)
                answer = parts[0].strip()
                email = parts[1].strip()
                
                # Send the answer first
                conversation_history.append({"role": "assistant", "content": answer})
                await websocket.send_json({"type": "bot", "content": answer})
                
                # Then handle email OTP sending
                flow_controller.collected_data["leadEmail"] = email
                flow_controller.otp_state["email_sent"] = True
                flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
                
                customer_name = flow_controller.collected_data.get("leadName", "Customer")
                logger.info(f"Sending OTP email to: {email}")
                ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                if ok:
                    flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    reply = get_string("otp_sent_email", lang_code, email)
                else:
                    reply = get_string("otp_send_fail_email", lang_code)
                
                conversation_history.append({"role": "assistant", "content": reply})
                await websocket.send_json({"type": "bot", "content": reply})
                continue
            
            elif "|||SEND_PHONE:" in reply:
                parts = reply.split("|||SEND_PHONE:", 1)
                answer = parts[0].strip()
                phone = parts[1].strip()
                # Format phone with GPT
                from app.utils.phone_utils import format_phone_number_with_gpt
                phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
                
                # Send the answer first
                conversation_history.append({"role": "assistant", "content": answer})
                await websocket.send_json({"type": "bot", "content": answer})
                
                # Then handle phone OTP sending
                flow_controller.collected_data["leadPhoneNumber"] = phone
                flow_controller.otp_state["phone_sent"] = True
                flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
                
                logger.info(f"Sending OTP SMS to: {phone}")
                ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
                if ok:
                    flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                    reply = get_string("otp_sent_phone", lang_code, phone)
                else:
                    reply = get_string("otp_send_fail_phone", lang_code)
                
                conversation_history.append({"role": "assistant", "content": reply})
                await websocket.send_json({"type": "bot", "content": reply})
                continue
            
            elif reply.startswith("SEND_EMAIL:"):
                email = reply.split(":", 1)[1].strip()
                flow_controller.collected_data["leadEmail"] = email
                flow_controller.otp_state["email_sent"] = True
                flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
                
                # Get customer name from collected data
                customer_name = flow_controller.collected_data.get("leadName", "Customer")
                
                logger.info(f"Sending OTP email to: {email}")
                ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                if ok:
                    flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    reply = get_string("otp_sent_email", lang_code, email)
                else:
                    reply = get_string("otp_send_fail_email", lang_code)
                
                conversation_history.append({"role": "assistant", "content": reply})
                await websocket.send_json({"type": "bot", "content": reply})
                continue
            
            elif reply.startswith("SEND_PHONE:"):
                phone = reply.split(":", 1)[1].strip()
                # Format phone with GPT
                from app.utils.phone_utils import format_phone_number_with_gpt
                phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
                flow_controller.collected_data["leadPhoneNumber"] = phone
                flow_controller.otp_state["phone_sent"] = True
                flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
                
                logger.info(f"Sending OTP SMS to: {phone}")
                ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
                if ok:
                    flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                    reply = get_string("otp_sent_phone", lang_code, phone)
                else:
                    reply = get_string("otp_send_fail_phone", lang_code)
                
                conversation_history.append({"role": "assistant", "content": reply})
                await websocket.send_json({"type": "bot", "content": reply})
                continue
            
            # Check for retry requests from response generator
            retry_type, extracted_value = _detect_retry_request(reply)
            if retry_type:
                if retry_type == 'resend_otp' and flow_controller.otp_state["phone_sent"] and not flow_controller.otp_state["phone_verified"]:
                    # Resend OTP to existing phone number
                    phone = flow_controller.collected_data.get("leadPhoneNumber")
                    if phone:
                        ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
                        reply = get_string("otp_resend", lang_code, phone) if ok else get_string("otp_resend_fail_phone", lang_code)
                    else:
                        reply = get_string("no_phone", lang_code)
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
                    
                elif retry_type == 'resend_otp' and flow_controller.otp_state["email_sent"] and not flow_controller.otp_state["email_verified"]:
                    # Resend OTP to existing email
                    email = flow_controller.collected_data.get("leadEmail")
                    customer_name = flow_controller.collected_data.get("leadName", "Customer")
                    if email:
                        ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                        reply = get_string("otp_resend", lang_code, email) if ok else get_string("otp_resend_fail_email", lang_code)
                    else:
                        reply = get_string("no_email", lang_code)
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
                    
                elif retry_type == 'change_phone':
                    # Reset phone validation state to allow new phone number
                    flow_controller.otp_state["phone_sent"] = False
                    flow_controller.otp_state["phone_verified"] = False
                    
                    if extracted_value:
                        # Response generator provided the new phone - format with GPT
                        from app.utils.phone_utils import format_phone_number_with_gpt
                        phone = await format_phone_number_with_gpt(extracted_value, openai_client, settings.gpt_model)
                        flow_controller.collected_data["leadPhoneNumber"] = phone
                        
                        logger.info(f"WebSocket: Sending OTP to NEW phone: {phone}")
                        ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
                        if ok:
                            flow_controller.otp_state["phone_sent"] = True
                            flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                        reply = get_string("perfect_otp_sent_phone", lang_code, phone) if ok else get_string("found_phone_cant_send", lang_code)
                    else:
                        reply = get_string("no_problem_phone", lang_code)
                        flow_controller.transition_to(ConversationState.PHONE_COLLECTION)
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
                    
                elif retry_type == 'change_email':
                    # Reset email validation state to allow new email
                    flow_controller.otp_state["email_sent"] = False
                    flow_controller.otp_state["email_verified"] = False
                    
                    if extracted_value:
                        # Response generator provided the new email
                        email = extracted_value
                        flow_controller.collected_data["leadEmail"] = email
                        customer_name = flow_controller.collected_data.get("leadName", "Customer")
                        
                        logger.info(f"WebSocket: Sending OTP to NEW email: {email}")
                        ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                        if ok:
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        reply = get_string("perfect_otp_sent_email", lang_code, email) if ok else get_string("found_email_cant_send", lang_code)
                    else:
                        reply = get_string("no_problem_email", lang_code)
                        flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
            
            # Check if JSON was generated (all data collected)
            parsed_json = _maybe_parse_json(reply)
            if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
                # Add appId if available (for app-scoped leads)
                if app_id:
                    parsed_json["appId"] = app_id
                # Create lead
                try:
                    ok, _ = await lead_service.create_public_lead(user_id, parsed_json)
                    final_msg = get_string("final_success", lang_code) if ok else get_string("final_fallback", lang_code)
                except Exception:
                    final_msg = get_string("final_fallback", lang_code)
                
                integration = context.get("integration") or {}
                if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                    review_url = integration["googleReviewUrl"].strip()
                    await websocket.send_json({
                        "type": "review_prompt",
                        "content": get_string("review_prompt", lang_code, review_url),
                        "reviewUrl": review_url,
                    })
                await websocket.send_json({"type": "bot", "content": final_msg})
                await websocket.close(code=1000)
                break
            
            # Regular conversation response
            conversation_history.append({"role": "assistant", "content": reply})
            await websocket.send_json({"type": "bot", "content": reply})
            
            # State transitions are handled in response_generator
            # Only update if we're still in the same state (no transition happened)
            # The response generator handles state transitions internally
            
            # Keep history manageable
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for user_id=%s", user_id)


@app.post("/webhook/voice-agent")
async def voice_agent_webhook(request: Request) -> Response:
    """Handle incoming Twilio voice calls and connect them to the Deepgram voice agent."""
    form = await request.form()
    call_sid = form.get("CallSid")
    from_number = form.get("From")
    to_number = form.get("To")

    if not call_sid or not from_number:
        logger.error("Voice agent webhook missing required params: CallSid=%s From=%s", call_sid, from_number)
        failure = VoiceResponse()
        failure.say("Sorry, something went wrong connecting your call.")
        return Response(content=str(failure), media_type="text/xml")

    logger.info("Voice agent webhook: call_sid=%s from=%s to=%s", call_sid, from_number, to_number)

    try:
        await voice_agent_service.start_session(call_sid, from_number, to_number)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unable to initialize voice agent session for %s: %s", call_sid, exc)
        failure_response = VoiceResponse()
        failure_response.say("Sorry, our assistant is unavailable at the moment. Please try again later.")
        return Response(content=str(failure_response), media_type="text/xml")

    twiml_response = VoiceResponse()
    twiml_response.say("Connecting you to our virtual assistant now.")

    stream_url = str(request.url_for("voice_agent_stream"))
    if stream_url.startswith("http://"):
        stream_url = stream_url.replace("http://", "ws://", 1)
    elif stream_url.startswith("https://"):
        stream_url = stream_url.replace("https://", "wss://", 1)
    separator = "&" if "?" in stream_url else "?"
    stream_url = f"{stream_url}{separator}call_sid={call_sid}"

    connect = twiml_response.connect()
    connect.stream(
        url=stream_url,
        name=call_sid
    )
    return Response(content=str(twiml_response), media_type="text/xml")


@app.websocket("/webhook/voice/stream", name="voice_agent_stream")
async def voice_agent_stream(websocket: WebSocket):
    await websocket.accept()
    query_params = dict(websocket.query_params)
    logger.info("Voice agent stream connection params: %s", query_params)

    buffered_messages: List[str] = []
    start_payload: Optional[Dict[str, Any]] = None
    call_sid_candidates: List[str] = []
    try:
        while start_payload is None:
            message = await websocket.receive_text()
            buffered_messages.append(message)
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Voice agent stream received non-JSON message before start: %s", message)
                continue

            event_type = payload.get("event")
            if event_type != "start":
                logger.warning(
                    "Voice agent stream waiting for start event; received %s",
                    event_type,
                )
                continue

            start_payload = payload
            start_info = start_payload.get("start", {})
            call_sid = start_info.get("callSid")
            parent_call_sid = start_info.get("parentCallSid")
            stream_name = start_info.get("name")

            call_sid_candidates = [
                call_sid,
                parent_call_sid,
                stream_name,
                query_params.get("call_sid"),
                query_params.get("CallSid"),
                query_params.get("name"),
            ]

            if not any(call_sid_candidates):
                logger.error("Voice agent stream start event missing identifiers: %s", start_payload)
                await websocket.close(code=4400, reason="Missing CallSid")
                return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Voice agent stream failed during initialization: %s", exc)
        try:
            await websocket.close(code=1011, reason="Failed to initialize stream")
        except Exception:  # noqa: BLE001
            pass
        return

    session = None
    for candidate in call_sid_candidates:
        if not candidate:
            continue
        session = voice_agent_service.get_session(candidate)
        if session:
            if candidate != session.call_sid:
                logger.info(
                    "Voice agent stream matched session %s via identifier %s",
                    session.call_sid,
                    candidate,
                )
            break

    if not session:
        logger.error(
            "Voice agent stream received unknown identifiers: %s",
            call_sid_candidates,
        )
        try:
            await websocket.close(code=4404, reason="Session not found")
        except Exception:  # noqa: BLE001
            pass
        return

    try:
        await voice_agent_service.attach_twilio_stream(session.call_sid, websocket, buffered_messages)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling Twilio stream for %s: %s", call_sid, exc)
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:  # noqa: BLE001
            pass


@app.post("/webhook/voice-agent/status")
async def voice_agent_status(request: Request) -> Response:
    """Clean up voice agent sessions based on Twilio status callbacks."""
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    logger.info("Voice agent status update: call_sid=%s status=%s", call_sid, call_status)
    if call_sid and call_status and call_status.lower() in {"completed", "failed", "busy", "no-answer", "canceled"}:
        await voice_agent_service.stop_session(call_sid)
    return Response(content="", media_type="text/xml")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """Handle incoming WhatsApp messages via Twilio webhook"""
    try:
        # Parse form data from Twilio webhook
        form_data = await request.form()
        
        whatsapp_service = WhatsAppService(settings)
        message_data = whatsapp_service.parse_webhook_data(dict(form_data))
        
        if not message_data.get("from"):
            logger.error("No 'from' field in WhatsApp webhook data")
            return Response(content=whatsapp_service.create_twiml_response("Error: Missing sender information"), media_type="text/xml")
        
        # Extract user_id and Twilio phone number
        user_phone = message_data["from"]
        twilio_phone = message_data["to"]  # The Twilio WhatsApp number that received the message
        
        # Initialize services
        context_service = ContextService(settings)
        lead_service = LeadService(settings)
        rag_service = RAGService(settings)
        email_validation_service = EmailValidationService(settings)
        
        # Initialize OpenAI client for phone formatting
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        phone_validation_service = PhoneValidationService(
            settings,
            openai_client=openai_client,
            gpt_model=settings.gpt_model
        )
        
        # Use the Twilio number from the webhook as the reply-from number
        whatsapp_service.whatsapp_from = twilio_phone
        
        # Get or create session for this user (cleanup runs in background)
        session_id, is_new_session = get_or_create_session(user_phone)
        
        if is_new_session:
            # First message from this user (or session expired) - fetch context and initialize
            logger.info(
                "WhatsApp: new session for twilio_phone=%s user_phone=%s, fetching context from backend",
                twilio_phone, user_phone,
            )
            try:
                context = await context_service.fetch_user_context_by_twilio(twilio_phone)
            except Exception as exc:
                logger.exception("Failed to fetch user context for Twilio number %s: %s", twilio_phone, exc)
                return Response(content=whatsapp_service.create_twiml_response("Sorry, I'm having trouble accessing your information. Please try again later."), media_type="text/xml")
            
            # Extract user_id and app_id from context for lead creation
            user_data = context.get("user", {})
            user_id = user_data.get("id")
            app_data = context.get("app", {})
            app_id = app_data.get("id") if app_data else None
            app_name = (app_data.get("name") or "") if app_data else ""
            logger.info(
                "WhatsApp: context from backend for number %s -> app_id=%s app_name=%s (lead types will be for this app)",
                twilio_phone, app_id, app_name,
            )
            
            if not user_id:
                logger.error(f"WhatsApp: No user_id found in context for Twilio number {twilio_phone}")
                logger.error(f"WhatsApp: Context keys available: {list(context.keys())}")
                logger.error(f"WhatsApp: User data: {user_data}")
                return Response(content=whatsapp_service.create_twiml_response("Sorry, I couldn't identify your account. Please contact support."), media_type="text/xml")
            
            logger.info(f"WhatsApp: Using user_id '{user_id}' and app_id '{app_id}' for Twilio number {twilio_phone}")
            
            # Initialize new session state
            current_time = time.time()
            whatsapp_sessions[session_id] = {
                "phone": user_phone,
                "twilio_phone": twilio_phone,  # Store app's Twilio number for multi-app support
                "history": [],
                "email_state": {
                    "email": None,
                    "otp_sent": False,
                    "otp_verified": False,
                    "customer_name": None
                },
                "phone_state": {
                    "phone": user_phone,  # Pre-fill with WhatsApp phone number
                    "otp_sent": False,
                    "otp_verified": True  # Already verified by WhatsApp
                },
                "context": context,
                "user_id": user_id,
                "app_id": app_id,  # Store app_id for lead creation
                "created_at": current_time,
                "last_activity": current_time,
                "is_whatsapp": True  # Flag to identify WhatsApp conversations
            }
            
            # Initialize production-grade components for WhatsApp
            flow_controller = FlowController(context)
            flow_controller.set_whatsapp(True)
            # Store WhatsApp phone number from Twilio (caller's number)
            flow_controller.update_collected_data("leadPhoneNumber", user_phone)
            flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)
            
            response_generator = ResponseGenerator(settings, rag_service)
            response_generator.set_profession(str(context.get("profession") or "Business"))
            response_generator.set_channel("whatsapp")
            
            # First message from user (e.g. "السلام علیکم") – detect language and greet in that language
            first_message = (message_data.get("body") or "").strip()
            
            # Measure greeting generation time for performance monitoring
            greeting_start = time.time()
            initial_reply = await response_generator.generate_greeting(
                context, channel="whatsapp", first_message=first_message or None
            )
            greeting_duration = time.time() - greeting_start
            logger.info(f"WhatsApp: Generated greeting in {greeting_duration:.3f}s for {user_phone}")
            
            # Build RAG vector store in background (non-blocking) to improve response time
            # The vector store will be ready for subsequent FAQ/knowledge queries
            import asyncio
            asyncio.create_task(asyncio.to_thread(rag_service.build_vector_store, context))
            whatsapp_sessions[session_id]["history"].append({"role": "user", "content": first_message or "(started)"})
            whatsapp_sessions[session_id]["history"].append({"role": "assistant", "content": initial_reply})
            whatsapp_sessions[session_id]["flow_controller"] = flow_controller
            whatsapp_sessions[session_id]["response_generator"] = response_generator
            
            # Extract buttons and send
            cleaned_reply, buttons = _extract_buttons_from_response(initial_reply)
            if buttons:
                button_text = "\n\n" + "\n".join([f"{i}. {btn['title']}" for i, btn in enumerate(buttons, 1)])
                full_message = cleaned_reply + button_text + "\n\nPlease reply with the number of your choice."
                await whatsapp_service.send_message(user_phone, full_message, from_phone=twilio_phone)
            else:
                await whatsapp_service.send_message(user_phone, initial_reply, from_phone=twilio_phone)
            
            return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        # Load existing session state
        session = whatsapp_sessions[session_id]
        conversation_history = session["history"]
        email_validation_state = session["email_state"]
        phone_validation_state = session["phone_state"]
        context = session["context"]
        user_id = session["user_id"]
        app_id = session.get("app_id")  # Get app_id for lead creation
        twilio_phone = session.get("twilio_phone", twilio_phone)  # Get app's Twilio number from session
        
        # Get flow controller and response generator from session (or create if missing)
        flow_controller = session.get("flow_controller")
        response_generator = session.get("response_generator")
        
        if not flow_controller or not response_generator:
            # Initialize if missing (for backward compatibility)
            flow_controller = FlowController(context)
            flow_controller.set_whatsapp(True)
            # Store WhatsApp phone number from Twilio (caller's number)
            flow_controller.update_collected_data("leadPhoneNumber", user_phone)
            response_generator = ResponseGenerator(settings, rag_service)
            response_generator.set_profession(str(context.get("profession") or "Business"))
            response_generator.set_channel("whatsapp")
            session["flow_controller"] = flow_controller
            session["response_generator"] = response_generator
        
        # Handle button/list responses
        user_text = message_data.get("body", "")
        button_id = message_data.get("button_id", "")
        list_id = message_data.get("list_id", "")
        
        if button_id:
            # Handle button response
            user_text = button_id
        elif list_id:
            # Handle list response
            user_text = list_id
        
        # Convert numbered responses to actual values for better context
        enhanced_user_text = user_text
        if user_text.strip().isdigit():
            number = int(user_text.strip())
            
            # Check conversation history to determine what type of selection this is
            # Look at the conversation flow to determine context
            
            # Check if we have a lead type selected (look for leadType in recent messages)
            has_lead_type = any("leadType:" in msg.get("content", "") for msg in conversation_history[-3:] if msg.get("role") == "user")
            
            # Check if we have a service selected (look for service mentions in recent messages) 
            has_service = any(any(keyword in msg.get("content", "").lower() for keyword in ["cosmetic", "general", "dentistry", "treatment"]) 
                            for msg in conversation_history[-3:] if msg.get("role") == "user")
            
            logger.info(f"WhatsApp: Selection context - has_lead_type: {has_lead_type}, has_service: {has_service}")
            
            if not has_lead_type:
                # First selection - must be lead type
                lead_types = context.get("lead_types", [])
                if 1 <= number <= len(lead_types):
                    selected = lead_types[number - 1]
                    lead_type_value = selected.get('value', '')
                    lead_type_text = selected.get('text', '')
                    enhanced_user_text = f"{number} - {lead_type_text} (leadType: {lead_type_value})"
                    logger.info(f"WhatsApp: User selected lead type #{number} -> value: '{lead_type_value}', text: '{lead_type_text}'")
                else:
                    logger.warning(f"WhatsApp: User selected invalid lead type number {number}, available: {len(lead_types)}")
                    
            elif has_lead_type and not has_service:
                # Second selection - must be service plan
                # Use the SAME filtered list shown to the user (by lead type's relevantServicePlans)
                service_plans = context.get("service_plans", context.get("treatment_plans", []))
                lead_types = context.get("lead_types", [])
                collected_lead_type = flow_controller.collected_data.get("leadType")
                filtered_names = ResponseGenerator._filter_services_by_lead_type(
                    service_plans, lead_types, collected_lead_type
                )
                if filtered_names is not None:
                    all_options = filtered_names
                    logger.info(f"WhatsApp: Filtered to {len(filtered_names)} service plans for lead type '{collected_lead_type}'")
                else:
                    all_options = [t.get("question", str(t)) for t in service_plans if isinstance(t, dict)]
                
                logger.info(f"WhatsApp: Detected service plan selection. Available options: {all_options}")
                
                if 1 <= number <= len(all_options):
                    service_name = all_options[number - 1]
                    enhanced_user_text = f"{number} - {service_name}"
                    logger.info(f"WhatsApp: User selected service plan #{number} -> '{service_name}'")
                else:
                    logger.warning(f"WhatsApp: User selected invalid service number {number}, available: {len(all_options)}")
            elif flow_controller.state == ConversationState.WORKFLOW_QUESTION:
                # Convert numbered response to the actual workflow option text
                wm = flow_controller.workflow_manager
                if wm and wm.is_active:
                    current_q = wm.get_current_question()
                    if current_q:
                        opts = current_q.get("options", []) or []
                        sorted_opts = sorted(opts, key=lambda o: o.get("order", 0))
                        if 1 <= number <= len(sorted_opts):
                            opt_text = sorted_opts[number - 1].get("text", "").strip()
                            enhanced_user_text = opt_text
                            logger.info(f"WhatsApp: Workflow option #{number} -> '{opt_text}'")
                        else:
                            logger.warning(f"WhatsApp: Workflow option #{number} out of range ({len(sorted_opts)} options)")
                    else:
                        logger.info(f"WhatsApp: In WORKFLOW_QUESTION state but no current question found")
                else:
                    logger.info(f"WhatsApp: In WORKFLOW_QUESTION state but workflow manager not active")
            else:
                logger.info(f"WhatsApp: Number {number} - context unclear, treating as raw input")
        
        # Process the message using production-grade state machine
        
        # Add user message to history (use enhanced text for better GPT understanding)
        conversation_history.append({"role": "user", "content": enhanced_user_text})
        
        # Update session history
        whatsapp_sessions[session_id]["history"] = conversation_history
        
        # Detect response language from raw user message (not enhanced) for accurate detection
        lang_code = detect_language(user_text)
        session["response_language_code"] = lang_code
        lang_name = get_language_name_for_prompt(lang_code)
        session["response_language"] = lang_name
        response_generator.set_response_language(lang_name)
        
        # Check if we need to handle email validation (similar to WebSocket flow)
        validate_email = context.get("integration", {}).get("validateEmail", True)
        logger.info(f"WhatsApp: Email validation check - validate_email: {validate_email}, otp_sent: {email_validation_state['otp_sent']}, history_length: {len(conversation_history)}")
        
        if validate_email and not email_validation_state["otp_sent"] and len(conversation_history) > 1:
            # Check if last bot message asked for email
            last_bot_message = next((msg for msg in reversed(conversation_history) if msg["role"] == "assistant"), None)
            logger.info(f"WhatsApp: Last bot message: {last_bot_message['content'] if last_bot_message else 'None'}")
            
            if last_bot_message and "email" in last_bot_message["content"].lower():
                logger.info(f"WhatsApp: Detected email collection phase, processing user input: {user_text}")
                
                # Use structured extraction (production-grade approach)
                extractor = DataExtractor()
                email = extractor.extract_email(user_text)
                
                if email and _is_valid_email(email):
                    logger.info(f"WhatsApp: Valid email detected, sending OTP to {email}")
                    # Get customer name from collected data or conversation history
                    customer_name = flow_controller.collected_data.get("leadName", "Customer")
                    if customer_name == "Customer":
                        # Try to extract from conversation history
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "user":
                                name = extractor.extract_name(msg.get("content", ""), context.get("lead_types", []))
                                if name:
                                    customer_name = name
                                    break
                    
                    email_validation_state["email"] = email
                    email_validation_state["customer_name"] = customer_name
                    
                    # Update flow controller
                    flow_controller.collected_data["leadEmail"] = email
                    
                    # Send OTP email
                    ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                    
                    if ok:
                        email_validation_state["otp_sent"] = True
                        flow_controller.otp_state["email_sent"] = True
                        flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    
                    reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email) if ok else get_string("otp_send_fail_email", session.get("response_language_code", "en"))
                    conversation_history.append({"role": "assistant", "content": reply})
                    await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                    logger.info(f"WhatsApp: Email OTP sent, returning early to prevent JSON generation")
                    return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
                else:
                    logger.info(f"WhatsApp: No valid email found, letting state machine handle naturally")
                    # Let state machine handle invalid/missing email naturally
                    pass
        
        # Check if we're in email OTP verification mode
        if email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
            logger.info(f"WhatsApp: In OTP verification mode, processing user input: {user_text}")
            
            # First, let ResponseGenerator check if it's a change/resend request
            temp_reply = await response_generator.generate_response(flow_controller, user_text, conversation_history, context)
            
            # Check if it's a change/resend request
            retry_type, extracted_value = _detect_retry_request(temp_reply)
            if retry_type == 'resend_otp' and email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                # Resend OTP to existing email
                customer_name = email_validation_state.get("customer_name", flow_controller.collected_data.get("leadName", "Customer"))
                ok, _ = await email_validation_service.send_otp_email(user_id, email_validation_state["email"], customer_name)
                reply = get_string("otp_resend", session.get("response_language_code", "en"), email_validation_state["email"]) if ok else get_string("otp_resend_fail_email", session.get("response_language_code", "en"))
                conversation_history.append({"role": "assistant", "content": reply})
                success, message = await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                if not success:
                    logger.error("Failed to send WhatsApp email retry message: %s", message)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            elif retry_type == 'change_email':
                # Reset email validation state to allow new email
                email_validation_state["otp_sent"] = False
                email_validation_state["otp_verified"] = False
                email_validation_state["email"] = None
                # Also update flow_controller
                flow_controller.otp_state["email_sent"] = False
                flow_controller.otp_state["email_verified"] = False
                
                if extracted_value:
                    # LangChain provided the new email
                    email = extracted_value
                    email_validation_state["email"] = email
                    email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                    flow_controller.collected_data["leadEmail"] = email
                    
                    # Try to get customer name from conversation history if not already in flow_controller
                    if email_validation_state["customer_name"] == "Customer":
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "user":
                                content = msg.get("content", "").lower()
                                if "name is" in content or "my name is" in content:
                                    import re
                                    name_match = re.search(r'(?:name is|my name is)\s+([A-Za-z\s]+)', content, re.IGNORECASE)
                                    if name_match:
                                        customer_name = name_match.group(1).strip()
                                        email_validation_state["customer_name"] = customer_name
                                        flow_controller.collected_data["leadName"] = customer_name
                                        break
                    
                    logger.info(f"WhatsApp: Sending OTP to NEW email: {email}")
                    ok, _ = await email_validation_service.send_otp_email(user_id, email, email_validation_state["customer_name"])
                    if ok:
                        email_validation_state["otp_sent"] = True
                        flow_controller.otp_state["email_sent"] = True
                        flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    reply = get_string("perfect_otp_sent_email", session.get("response_language_code", "en"), email) if ok else get_string("found_email_cant_send", session.get("response_language_code", "en"))
                else:
                    reply = get_string("no_problem_email", session.get("response_language_code", "en"))
                    flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                
                conversation_history.append({"role": "assistant", "content": reply})
                success, message = await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                if not success:
                    logger.error("Failed to send WhatsApp change email message: %s", message)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            else:
                # Check if it's an OTP code
                otp_code = _extract_otp_from_text(user_text)
                if otp_code:
                    logger.info(f"WhatsApp: Extracted OTP code: {otp_code}")
                    # Verify OTP
                    ok, message = await email_validation_service.verify_otp(
                        user_id, 
                        email_validation_state["email"], 
                        otp_code
                    )
                    logger.info(f"WhatsApp Email OTP verification result: {ok} - {message}")
                    if ok:
                        email_validation_state["otp_verified"] = True
                        # Update flow controller OTP state
                        flow_controller.otp_state["email_verified"] = True
                        flow_controller.transition_to(flow_controller.get_next_state())
                        # Get AI response to continue the flow
                        reply = await response_generator.generate_response(flow_controller, "Email verified", conversation_history, context)
                    
                    # Check if it's JSON (lead completion) - BEFORE sending
                    parsed_json = _maybe_parse_json(reply)
                    if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
                        # Ensure WhatsApp phone number is included (should already be in JSON from flow_controller)
                        # But add it if missing (fallback)
                        if "leadPhoneNumber" not in parsed_json or not parsed_json.get("leadPhoneNumber"):
                            parsed_json["leadPhoneNumber"] = user_phone
                        
                        # Add appId if available (for app-scoped WhatsApp leads)
                        if app_id:
                            parsed_json["appId"] = app_id
                        
                        # Create lead and send friendly message
                        try:
                            ok_lead, _ = await lead_service.create_public_lead(user_id, parsed_json)
                            if ok_lead:
                                final_msg = get_string("final_success", session.get("response_language_code", "en"))
                            else:
                                final_msg = get_string("final_fallback", session.get("response_language_code", "en"))
                        except Exception:
                            final_msg = get_string("final_fallback", session.get("response_language_code", "en"))
                        
                        integration = context.get("integration") or {}
                        if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                            review_url = integration["googleReviewUrl"].strip()
                            review_msg = get_string("review_prompt", session.get("response_language_code", "en"), review_url)
                            conversation_history.append({"role": "assistant", "content": review_msg})
                            await whatsapp_service.send_message(user_phone, review_msg, from_phone=twilio_phone)
                        conversation_history.append({"role": "assistant", "content": final_msg})
                        await whatsapp_service.send_message(user_phone, final_msg, from_phone=twilio_phone)
                        
                        # Clean up session after lead creation
                        del whatsapp_sessions[session_id]
                        if user_phone in phone_to_session and phone_to_session[user_phone] == session_id:
                            del phone_to_session[user_phone]
                        logger.info(f"WhatsApp: Lead created, session cleaned up")
                        return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
                    else:
                        # Not JSON, send the regular response
                        conversation_history.append({"role": "assistant", "content": reply})
                        await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                        return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
                else:
                    # Not an OTP code - use ResponseGenerator's response (which might be an error message or natural response)
                    reply = temp_reply
                    conversation_history.append({"role": "assistant", "content": reply})
                    await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                    return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        # Generate response using production-grade state machine
        reply = await response_generator.generate_response(flow_controller, enhanced_user_text, conversation_history, context)
        
        # Handle special responses from response generator (SEND_EMAIL, SEND_PHONE)
        # Check for answer + SEND_EMAIL/PHONE format (when user asks question while providing email/phone)
        if "|||SEND_EMAIL:" in reply:
            parts = reply.split("|||SEND_EMAIL:", 1)
            answer = parts[0].strip()
            email = parts[1].strip()
            
            # Send the answer first
            conversation_history.append({"role": "assistant", "content": answer})
            await whatsapp_service.send_message(user_phone, answer, from_phone=twilio_phone)
            
            # Then handle email OTP sending
            flow_controller.collected_data["leadEmail"] = email
            flow_controller.otp_state["email_sent"] = True
            flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
            
            customer_name = flow_controller.collected_data.get("leadName", "Customer")
            logger.info(f"WhatsApp: Sending OTP email to: {email}")
            ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
            if ok:
                flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                email_validation_state["otp_sent"] = True
                email_validation_state["email"] = email
                email_validation_state["customer_name"] = customer_name
            
            reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email) if ok else get_string("otp_send_fail_email", session.get("response_language_code", "en"))
            conversation_history.append({"role": "assistant", "content": reply})
            await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
            return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        elif "|||SEND_PHONE:" in reply:
            parts = reply.split("|||SEND_PHONE:", 1)
            answer = parts[0].strip()
            phone = parts[1].strip()
            # Format phone with GPT (handled inside send_sms_otp, but format here for storage)
            from app.utils.phone_utils import format_phone_number_with_gpt
            phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
            
            # Send the answer first
            conversation_history.append({"role": "assistant", "content": answer})
            await whatsapp_service.send_message(user_phone, answer, from_phone=twilio_phone)
            
            # Then handle phone OTP sending
            flow_controller.collected_data["leadPhoneNumber"] = phone
            flow_controller.otp_state["phone_sent"] = True
            flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
            
            logger.info(f"WhatsApp: Sending OTP SMS to: {phone}")
            ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
            if ok:
                flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                phone_validation_state["otp_sent"] = True
                phone_validation_state["phone"] = phone
            
            reply = get_string("otp_sent_phone", session.get("response_language_code", "en"), phone) if ok else get_string("otp_send_fail_phone", session.get("response_language_code", "en"))
            conversation_history.append({"role": "assistant", "content": reply})
            await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
            return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        elif reply.startswith("SEND_EMAIL:"):
            email = reply.split(":", 1)[1].strip()
            flow_controller.collected_data["leadEmail"] = email
            flow_controller.otp_state["email_sent"] = True
            flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
            
            customer_name = flow_controller.collected_data.get("leadName", "Customer")
            logger.info(f"WhatsApp: Sending OTP email to: {email}")
            ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
            if ok:
                flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                email_validation_state["otp_sent"] = True
                email_validation_state["email"] = email
                email_validation_state["customer_name"] = customer_name
            
            reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email) if ok else get_string("otp_send_fail_email", session.get("response_language_code", "en"))
            conversation_history.append({"role": "assistant", "content": reply})
            await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
            return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        elif reply.startswith("SEND_PHONE:"):
            # For WhatsApp, phone is already verified, so this shouldn't happen
            # But handle it just in case
            phone = reply.split(":", 1)[1].strip()
            # Format phone with GPT (handled inside send_sms_otp, but format here for storage)
            from app.utils.phone_utils import format_phone_number_with_gpt
            phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
            flow_controller.collected_data["leadPhoneNumber"] = phone
            flow_controller.otp_state["phone_sent"] = True
            flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
            
            logger.info(f"WhatsApp: Sending OTP SMS to: {phone}")
            ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
            if ok:
                flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                phone_validation_state["otp_sent"] = True
                phone_validation_state["phone"] = phone
            
            reply = get_string("otp_sent_phone", session.get("response_language_code", "en"), phone) if ok else get_string("otp_send_fail_phone", session.get("response_language_code", "en"))
            conversation_history.append({"role": "assistant", "content": reply})
            await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
            return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        # Check if JSON was generated (all data collected) - BEFORE checking retry requests
        parsed_json = _maybe_parse_json(reply)
        if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
            # Ensure WhatsApp phone number is included (should already be in JSON from flow_controller)
            # But add it if missing (fallback)
            if "leadPhoneNumber" not in parsed_json or not parsed_json.get("leadPhoneNumber"):
                parsed_json["leadPhoneNumber"] = user_phone
            
            # Check email verification only if enabled
            email_valid = True
            if context.get("integration", {}).get("validateEmail", True):
                email_valid = _validate_email_verification(email_validation_state)
            
            # Phone is already verified by WhatsApp
            phone_valid = True
            
            if email_valid and phone_valid:
                # Add appId if available (for app-scoped WhatsApp leads)
                if app_id:
                    parsed_json["appId"] = app_id
                # Create lead
                try:
                    ok, _ = await lead_service.create_public_lead(user_id, parsed_json)
                    if ok:
                        final_msg = get_string("final_success", session.get("response_language_code", "en"))
                    else:
                        final_msg = get_string("final_fallback", session.get("response_language_code", "en"))
                except Exception:
                    final_msg = get_string("final_fallback", session.get("response_language_code", "en"))
                
                integration = context.get("integration") or {}
                if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                    review_url = integration["googleReviewUrl"].strip()
                    review_msg = get_string("review_prompt", session.get("response_language_code", "en"), review_url)
                    await whatsapp_service.send_message(user_phone, review_msg, from_phone=twilio_phone)
                await whatsapp_service.send_message(user_phone, final_msg, from_phone=twilio_phone)
                
                # Clean up session after lead creation
                del whatsapp_sessions[session_id]
                if user_phone in phone_to_session and phone_to_session[user_phone] == session_id:
                    del phone_to_session[user_phone]
                
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        # Check for retry requests from GPT response
        retry_type, extracted_value = _detect_retry_request(reply)
        if retry_type:
            if retry_type == 'resend_otp' and phone_validation_state["otp_sent"] and not phone_validation_state["otp_verified"]:
                # Resend OTP to existing phone number
                ok, _ = await phone_validation_service.send_sms_otp(user_id, phone_validation_state["phone"])
                reply = get_string("otp_resend", session.get("response_language_code", "en"), phone_validation_state["phone"]) if ok else get_string("otp_resend_fail_phone", session.get("response_language_code", "en"))
                conversation_history.append({"role": "assistant", "content": reply})
                success, message = await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                if not success:
                    logger.error("Failed to send WhatsApp retry message: %s", message)
                # Return empty TwiML response (no automatic reply needed)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            elif retry_type == 'resend_otp' and email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                # Resend OTP to existing email
                customer_name = email_validation_state.get("customer_name", flow_controller.collected_data.get("leadName", "Customer"))
                ok, _ = await email_validation_service.send_otp_email(user_id, email_validation_state["email"], customer_name)
                reply = get_string("otp_resend", session.get("response_language_code", "en"), email_validation_state["email"]) if ok else get_string("otp_resend_fail_email", session.get("response_language_code", "en"))
                conversation_history.append({"role": "assistant", "content": reply})
                success, message = await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                if not success:
                    logger.error("Failed to send WhatsApp email retry message: %s", message)
                # Return empty TwiML response (no automatic reply needed)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            elif retry_type == 'change_phone':
                # Reset phone validation state to allow new phone number
                phone_validation_state["otp_sent"] = False
                phone_validation_state["otp_verified"] = False
                phone_validation_state["phone"] = None
                # Also update flow_controller
                flow_controller.otp_state["phone_sent"] = False
                flow_controller.otp_state["phone_verified"] = False
                
                if extracted_value:
                    # LangChain provided the new phone - format with GPT
                    from app.utils.phone_utils import format_phone_number_with_gpt
                    phone = await format_phone_number_with_gpt(extracted_value, openai_client, settings.gpt_model)
                    phone_validation_state["phone"] = phone
                    flow_controller.collected_data["leadPhoneNumber"] = phone
                    
                    logger.info(f"WhatsApp: Sending OTP to NEW phone: {phone}")
                    ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
                    if ok:
                        phone_validation_state["otp_sent"] = True
                        flow_controller.otp_state["phone_sent"] = True
                        flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                    reply = get_string("perfect_otp_sent_phone", session.get("response_language_code", "en"), phone) if ok else get_string("found_phone_cant_send", session.get("response_language_code", "en"))
                else:
                    reply = get_string("no_problem_phone", session.get("response_language_code", "en"))
                    flow_controller.transition_to(ConversationState.PHONE_COLLECTION)
                
                conversation_history.append({"role": "assistant", "content": reply})
                success, message = await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                if not success:
                    logger.error("Failed to send WhatsApp change phone message: %s", message)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            elif retry_type == 'change_email':
                # Reset email validation state to allow new email
                email_validation_state["otp_sent"] = False
                email_validation_state["otp_verified"] = False
                email_validation_state["email"] = None
                # Also update flow_controller
                flow_controller.otp_state["email_sent"] = False
                flow_controller.otp_state["email_verified"] = False
                
                if extracted_value:
                    # LangChain provided the new email
                    email = extracted_value
                    email_validation_state["email"] = email
                    email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                    flow_controller.collected_data["leadEmail"] = email
                    
                    # Try to get customer name from conversation history if not already in flow_controller
                    if email_validation_state["customer_name"] == "Customer":
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "user":
                                content = msg.get("content", "").lower()
                                if "name is" in content or "my name is" in content:
                                    import re
                                    name_match = re.search(r'(?:name is|my name is)\s+([A-Za-z\s]+)', content, re.IGNORECASE)
                                    if name_match:
                                        customer_name = name_match.group(1).strip()
                                        email_validation_state["customer_name"] = customer_name
                                        flow_controller.collected_data["leadName"] = customer_name
                                        break
                    
                    logger.info(f"WhatsApp: Sending OTP to NEW email: {email}")
                    ok, _ = await email_validation_service.send_otp_email(user_id, email, email_validation_state["customer_name"])
                    if ok:
                        email_validation_state["otp_sent"] = True
                        flow_controller.otp_state["email_sent"] = True
                        flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    reply = get_string("perfect_otp_sent_email", session.get("response_language_code", "en"), email) if ok else get_string("found_email_cant_send", session.get("response_language_code", "en"))
                else:
                    reply = get_string("no_problem_email", session.get("response_language_code", "en"))
                    flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                
                conversation_history.append({"role": "assistant", "content": reply})
                success, message = await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                if not success:
                    logger.error("Failed to send WhatsApp change email message: %s", message)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            elif retry_type == 'send_email' and extracted_value:
                # Check if email validation is enabled
                validate_email = context.get("integration", {}).get("validateEmail", True)
                if not validate_email:
                    # Email validation is disabled - store email and let AI respond naturally
                    email = extracted_value
                    email_validation_state["email"] = email
                    # Add email acknowledgment to conversation history (without the email itself to avoid re-triggering)
                    conversation_history.append({"role": "user", "content": "Email provided"})
                    # Get AI response to continue the flow
                    reply = await response_generator.generate_response(flow_controller, "Email provided", conversation_history, context)
                    conversation_history.append({"role": "assistant", "content": reply})
                    await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                    return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
                
                # Email validation is enabled - send OTP
                email = extracted_value
                email_validation_state["email"] = email
                email_validation_state["customer_name"] = "Customer"  # Default
                
                # Try to get customer name from conversation history
                for msg in reversed(conversation_history):
                    if msg.get("role") == "user":
                        content = msg.get("content", "").lower()
                        if "name is" in content or "my name is" in content:
                            import re
                            name_match = re.search(r'(?:name is|my name is)\s+([A-Za-z\s]+)', content, re.IGNORECASE)
                            if name_match:
                                email_validation_state["customer_name"] = name_match.group(1).strip()
                                break
                
                logger.info(f"WhatsApp: Sending OTP email to: {email}")
                ok, _ = await email_validation_service.send_otp_email(user_id, email, email_validation_state["customer_name"])
                if ok:
                    email_validation_state["otp_sent"] = True
                reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email) if ok else get_string("otp_send_fail_email", session.get("response_language_code", "en"))
                conversation_history.append({"role": "assistant", "content": reply})
                await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            elif retry_type == 'send_phone' and extracted_value:
                # Check if phone validation is enabled (for WhatsApp, phone is already verified, so skip)
                validate_phone = context.get("integration", {}).get("validatePhoneNumber", True)
                if not validate_phone:
                    # Phone validation is disabled - store phone and let AI respond naturally
                    # Format phone with GPT for consistent storage
                    from app.utils.phone_utils import format_phone_number_with_gpt
                    phone = await format_phone_number_with_gpt(extracted_value, openai_client, settings.gpt_model)
                    phone_validation_state["phone"] = phone
                    # Add phone acknowledgment to conversation history (without the phone itself to avoid re-triggering)
                    conversation_history.append({"role": "user", "content": "Phone number provided"})
                    # Get AI response to continue the flow
                    reply = await response_generator.generate_response(flow_controller, "Phone number provided", conversation_history, context)
                    conversation_history.append({"role": "assistant", "content": reply})
                    await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                    return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
                
                # Phone validation is enabled - send OTP (Note: For WhatsApp, this shouldn't happen as phone is already verified)
                # Format phone with GPT
                from app.utils.phone_utils import format_phone_number_with_gpt
                phone = await format_phone_number_with_gpt(extracted_value, openai_client, settings.gpt_model)
                phone_validation_state["phone"] = phone
                
                logger.info(f"WhatsApp: Sending OTP SMS to: {phone}")
                ok, _ = await phone_validation_service.send_sms_otp(user_id, phone)
                if ok:
                    phone_validation_state["otp_sent"] = True
                reply = get_string("otp_sent_phone", session.get("response_language_code", "en"), phone) if ok else get_string("otp_send_fail_phone", session.get("response_language_code", "en"))
                conversation_history.append({"role": "assistant", "content": reply})
                await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        # Check if response contains buttons
        cleaned_reply, buttons = _extract_buttons_from_response(reply)
        
        # Safety check: ensure no raw button tags are sent
        if '<button' in cleaned_reply.lower():
            logger.warning("Found remaining button tags in cleaned reply, removing them")
            cleaned_reply = re.sub(r'<button[^>]*>.*?</button[^>]*>', '', cleaned_reply, flags=re.IGNORECASE | re.DOTALL)
            cleaned_reply = re.sub(r'<button[^>]*>.*?$', '', cleaned_reply, flags=re.IGNORECASE | re.DOTALL)
            cleaned_reply = cleaned_reply.strip()
        
        # Send appropriate WhatsApp message
        if buttons:
            # For WhatsApp, convert buttons to numbered list instead of interactive buttons
            # (interactive buttons don't work well in Twilio sandbox)
            button_text = "\n\n" + "\n".join([f"{i}. {btn['title']}" for i, btn in enumerate(buttons, 1)])
            full_message = cleaned_reply + button_text + "\n\nPlease reply with the number of your choice."
            success, message = await whatsapp_service.send_message(user_phone, full_message, from_phone=twilio_phone)
            if not success:
                logger.error("Failed to send WhatsApp message: %s", message)
        else:
            # Send simple text message (ensure no button tags)
            success, message = await whatsapp_service.send_message(user_phone, cleaned_reply, from_phone=twilio_phone)
            if not success:
                logger.error("Failed to send WhatsApp message: %s", message)
        
        # Update conversation history with bot response
        conversation_history.append({"role": "assistant", "content": cleaned_reply})
        whatsapp_sessions[session_id]["history"] = conversation_history
        
        # Return empty TwiML response (no automatic reply needed)
        return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
    except Exception as e:
        logger.exception("Error processing WhatsApp webhook: %s", str(e))
        whatsapp_service = WhatsAppService(settings)
        return Response(content=whatsapp_service.create_twiml_response("Sorry, I encountered an error. Please try again."), media_type="text/xml")


@app.get("/webhook/whatsapp")
async def whatsapp_webhook_verification(request: Request):
    """Handle WhatsApp webhook verification (GET request)"""
    # Twilio may send GET requests for webhook verification
    return {"status": "ok", "message": "WhatsApp webhook endpoint is active"}

