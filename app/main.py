import json
import logging
import re
import time
import secrets
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager

from .config import settings
from .services.context_service import ContextService
from .services.lead_service import LeadService
from .services.gpt_service import GptService
from .services.email_validation_service import EmailValidationService
from .services.phone_validation_service import PhoneValidationService
from .services.whatsapp_service import WhatsAppService


logger = logging.getLogger("assistly")
logging.basicConfig(level=logging.INFO)

# In-memory storage for WhatsApp conversations (session-based)
# Key: session_id, Value: {phone, history, email_state, phone_state, context, user_id, created_at, last_activity}
whatsapp_sessions: Dict[str, Dict[str, Any]] = {}

# Mapping: phone number -> current active session_id
phone_to_session: Dict[str, str] = {}

# Session timeout in seconds (30 minutes)
SESSION_TIMEOUT = 30 * 60

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
    # Look for 6-digit numbers in the text
    otp_match = re.search(r'\b\d{6}\b', text)
    return otp_match.group(0) if otp_match else None

def _extract_phone_from_text(text: str) -> str:
    """Extract phone number from user text."""
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
            return phone
    
    return None

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

def _convert_services_to_whatsapp_list(services: List[Any], treatment_plans: List[Any] = None) -> List[Dict[str, Any]]:
    """Convert services and treatment plans to WhatsApp list format"""
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
    
    # Add treatment plans
    if treatment_plans:
        for i, plan in enumerate(treatment_plans, len(services) + 1):
            if isinstance(plan, dict):
                title = plan.get("question", plan.get("title", str(plan)))
                description = plan.get("description", "")
            else:
                title = str(plan)
                description = ""
            
            items.append({
                "id": f"treatment_{i}",
                "title": title,
                "description": description
            })
    
    if items:
        sections.append({
            "title": "Services & Treatments",
            "rows": items
        })
    
    return sections

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    user_id: Optional[str] = websocket.query_params.get("user_id")

    if not user_id:
        await websocket.send_json({"type": "error", "content": "Missing user_id in query params"})
        await websocket.close(code=1008)
        return

    context_service = ContextService(settings)
    lead_service = LeadService(settings)
    gpt_service = GptService(settings)
    email_validation_service = EmailValidationService(settings)
    phone_validation_service = PhoneValidationService(settings)

    try:
        context = await context_service.fetch_user_context(user_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch user context: %s", exc)
        await websocket.send_json({
            "type": "error",
            "content": "Unable to fetch user context. Please try again shortly.",
        })
        await websocket.close(code=1011)
        return

    # Initialize conversation
    gpt_service.set_profession(str(context.get("profession") or "Clinic"))
    conversation_history: List[Dict[str, str]] = []
    
    # Email validation state
    email_validation_state = {
        "email": None,
        "otp_sent": False,
        "otp_verified": False,
        "customer_name": None
    }
    
    # Phone validation state
    phone_validation_state = {
        "phone": None,
        "otp_sent": False,
        "otp_verified": False
    }
    
    # Send initial greeting
    initial_reply = await gpt_service.agent_greet(context, {})
    conversation_history.append({"role": "assistant", "content": initial_reply})
    await websocket.send_json({"type": "bot", "content": initial_reply})

    try:
        while True:
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                user_text = parsed.get("content") or parsed.get("text") or data
                user_country = parsed.get("country")  # Extract country from user message
            except json.JSONDecodeError:
                user_text = data
                user_country = None

            if not user_text or not str(user_text).strip():
                continue
            if str(user_text).strip().lower() in {"ping", "pong", "keepalive", "heartbeat"}:
                continue

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_text})
            
            # Check if we need to handle email validation
            if email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                # User is entering OTP code - try to extract 6-digit code from text
                otp_code = _extract_otp_from_text(user_text)
                if otp_code:
                    # Verify OTP
                    ok, message = await email_validation_service.verify_otp(
                        user_id, 
                        email_validation_state["email"], 
                        otp_code
                    )
                    logger.info(f"Email OTP verification result: {ok} - {message}")
                    if ok:
                        email_validation_state["otp_verified"] = True
                        # Send success message directly
                        success_msg = "Great! Your email has been verified. Now, could you please tell me your phone number?"
                        conversation_history.append({"role": "assistant", "content": success_msg})
                        await websocket.send_json({"type": "bot", "content": success_msg})
                        continue
                    else:
                        # Send error message directly
                        error_msg = "That code doesn't look right. Please check and try entering the 6-digit code again."
                        conversation_history.append({"role": "assistant", "content": error_msg})
                        await websocket.send_json({"type": "bot", "content": error_msg})
                        continue
                else:
                    # Let GPT handle non-OTP responses naturally
                    pass
            
            # Check if user provided email and we need to send OTP
            # Only intercept if we're in the email collection phase (last bot message asked for email)
            # AND if email validation is enabled in user context
            last_bot_message = next((msg for msg in reversed(conversation_history) if msg["role"] == "assistant"), None)
            validate_email = context.get("integration", {}).get("validateEmail", True)
            if validate_email and not email_validation_state["otp_sent"] and last_bot_message and "email" in last_bot_message["content"].lower():
                # Extract name and email in one GPT call
                extraction_prompt = (
                    f"Extract from conversation: 1) Customer name (respond with just name or 'Customer'), "
                    f"2) Email from '{user_text}' (respond with just email or 'NO_EMAIL'). "
                    f"Format: NAME|EMAIL. Conversation: {[msg['content'] for msg in conversation_history if msg['role'] == 'user']}"
                )
                
                extraction_response = await gpt_service.short_reply(conversation_history, extraction_prompt, context)
                
                if "|" in extraction_response:
                    customer_name, email = extraction_response.split("|", 1)
                    customer_name = customer_name.strip()
                    email = email.strip()
                    
                    if email != "NO_EMAIL" and "@" in email and _is_valid_email(email):
                        email_validation_state["email"] = email
                        email_validation_state["customer_name"] = customer_name
                        
                        # Send OTP email
                        ok, _ = await email_validation_service.send_otp_email(user_id, email, customer_name)
                        
                        reply = (f"Great! I've sent a 6-digit verification code to {email}. Please enter the code to verify your email." 
                                if ok else "Sorry, I couldn't send the verification email. Please check your email address and try again.")
                        conversation_history.append({"role": "assistant", "content": reply})
                        await websocket.send_json({"type": "bot", "content": reply})
                        if ok:
                            email_validation_state["otp_sent"] = True
                        continue
                    else:
                        # Let GPT handle invalid email naturally
                        pass
                else:
                    # Let GPT handle missing email naturally
                    pass
            
            # Check if we need to handle phone validation
            validate_phone = context.get("integration", {}).get("validatePhoneNumber", True)
            if validate_phone and not phone_validation_state["otp_sent"] and last_bot_message and "phone" in last_bot_message["content"].lower():
                # Extract phone number from user text
                phone = _extract_phone_from_text(user_text)
                if phone:
                    phone_validation_state["phone"] = phone
                    
                    # Get country code from context (fallback for non-prefixed numbers)
                    api_country_code = context.get("country", "US")
                    logger.info(f"Using API country code as fallback: '{api_country_code}', user country: '{user_country}' for phone: '{phone}'")
                    
                    # Send SMS OTP (country detection happens inside the service with priority: user_country > phone_prefix > api_country)
                    ok, _ = await phone_validation_service.send_sms_otp(user_id, phone, api_country_code, user_country)
                    
                    reply = (f"Great! I've sent a 6-digit verification code to {phone}. Please enter the code to verify your phone number." 
                            if ok else "Sorry, I couldn't send the verification SMS. Please check your phone number and try again.")
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    if ok:
                        phone_validation_state["otp_sent"] = True
                    continue
                else:
                    # Let GPT handle invalid phone naturally
                    pass
            
            # Check if we're in phone OTP verification mode
            if phone_validation_state["otp_sent"] and not phone_validation_state["otp_verified"]:
                # User is entering phone OTP code - try to extract 6-digit code from text
                otp_code = _extract_otp_from_text(user_text)
                if otp_code:
                    # Verify phone OTP (country detection happens inside the service with priority: user_country > phone_prefix > api_country)
                    api_country_code = context.get("country", "US")
                    logger.info(f"Verifying phone OTP for phone: '{phone_validation_state['phone']}' with API fallback country: '{api_country_code}', user country: '{user_country}'")
                    ok, message = await phone_validation_service.verify_sms_otp(
                        user_id, 
                        phone_validation_state["phone"], 
                        otp_code,
                        api_country_code,
                        user_country
                    )
                    logger.info(f"Phone OTP verification result: {ok} - {message}")
                    if ok:
                        phone_validation_state["otp_verified"] = True
                        
                        # Trigger GPT to generate JSON immediately after phone verification
                        json_trigger_msg = "Generate JSON now with all collected information."
                        conversation_history.append({"role": "user", "content": json_trigger_msg})
                        
                        # Get GPT response for JSON generation
                        reply = await gpt_service.agent_reply(conversation_history, json_trigger_msg, context, {})
                        
                        # Check if it's JSON (lead completion)
                        parsed_json = _maybe_parse_json(reply)
                        if parsed_json and isinstance(parsed_json, dict):
                            # Check email verification only if enabled
                            email_valid = True
                            if context.get("integration", {}).get("validateEmail", True):
                                email_valid = _validate_email_verification(email_validation_state)
                            
                            # Check phone verification only if enabled
                            phone_valid = True
                            if context.get("integration", {}).get("validatePhoneNumber", True):
                                phone_valid = _validate_phone_verification(phone_validation_state)
                            
                            if email_valid and phone_valid:
                                # Create lead
                                try:
                                    ok, _ = await lead_service.create_public_lead(user_id, parsed_json)
                                    if ok:
                                        final_msg = "Thanks! I have your details and someone will get back to you soon. Bye!"
                                    else:
                                        final_msg = "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!"
                                except Exception:
                                    final_msg = "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!"
                                
                                await websocket.send_json({"type": "bot", "content": final_msg})
                                await websocket.close(code=1000)
                                break
                        continue
                    else:
                        # Send error message directly
                        error_msg = "That code doesn't look right. Please check and try entering the 6-digit code again."
                        conversation_history.append({"role": "assistant", "content": error_msg})
                        await websocket.send_json({"type": "bot", "content": error_msg})
                        continue
                else:
                    # Let GPT handle non-OTP responses naturally
                    pass
            
            # Get GPT response
            reply = await gpt_service.agent_reply(conversation_history, user_text, context, {})
            
            # Check if it's JSON (lead completion)
            parsed_json = _maybe_parse_json(reply)
            if parsed_json and isinstance(parsed_json, dict):
                # Check email verification only if enabled
                email_valid = True
                if context.get("integration", {}).get("validateEmail", True):
                    email_valid = _validate_email_verification(email_validation_state)
                
                # Check phone verification only if enabled
                phone_valid = True
                if context.get("integration", {}).get("validatePhoneNumber", True):
                    phone_valid = _validate_phone_verification(phone_validation_state)
                
                if email_valid and phone_valid:
                    # Create lead
                    try:
                        ok, _ = await lead_service.create_public_lead(user_id, parsed_json)
                        if ok:
                            final_msg = "Thanks! I have your details and someone will get back to you soon. Bye!"
                        else:
                            final_msg = "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!"
                    except Exception:
                        final_msg = "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!"
                    
                    await websocket.send_json({"type": "bot", "content": final_msg})
                await websocket.close(code=1000)
                break
            else:
                # Regular conversation response
                conversation_history.append({"role": "assistant", "content": reply})
                await websocket.send_json({"type": "bot", "content": reply})
            
            # Keep history manageable
            conversation_history = conversation_history[-10:]

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for user_id=%s", user_id)


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
        gpt_service = GptService(settings)
        email_validation_service = EmailValidationService(settings)
        phone_validation_service = PhoneValidationService(settings)
        
        # Use the Twilio number from the webhook as the reply-from number
        whatsapp_service.whatsapp_from = twilio_phone
        
        # Get or create session for this user (cleanup runs in background)
        session_id, is_new_session = get_or_create_session(user_phone)
        
        if is_new_session:
            # First message from this user - fetch context and initialize
            try:
                context = await context_service.fetch_user_context_by_twilio(twilio_phone)
            except Exception as exc:
                logger.exception("Failed to fetch user context for Twilio number %s: %s", twilio_phone, exc)
                return Response(content=whatsapp_service.create_twiml_response("Sorry, I'm having trouble accessing your information. Please try again later."), media_type="text/xml")
            
            # Extract user_id from context for lead creation
            user_data = context.get("user", {})
            user_id = user_data.get("id")
            
            if not user_id:
                logger.error(f"WhatsApp: No user_id found in context for Twilio number {twilio_phone}")
                logger.error(f"WhatsApp: Context keys available: {list(context.keys())}")
                logger.error(f"WhatsApp: User data: {user_data}")
                return Response(content=whatsapp_service.create_twiml_response("Sorry, I couldn't identify your account. Please contact support."), media_type="text/xml")
            
            logger.info(f"WhatsApp: Using user_id '{user_id}' for Twilio number {twilio_phone}")
            
            # Initialize new session state
            current_time = time.time()
            whatsapp_sessions[session_id] = {
                "phone": user_phone,
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
                "created_at": current_time,
                "last_activity": current_time,
                "is_whatsapp": True  # Flag to identify WhatsApp conversations
            }
            
            # Set profession and send initial greeting
            gpt_service.set_profession(str(context.get("profession") or "Clinic"))
            initial_reply = await gpt_service.agent_greet(context, {}, is_whatsapp=True)
            whatsapp_sessions[session_id]["history"].append({"role": "assistant", "content": initial_reply})
            
            # Extract buttons and send
            cleaned_reply, buttons = gpt_service.extract_buttons_from_response(initial_reply)
            if buttons:
                button_text = "\n\n" + "\n".join([f"{i}. {btn['title']}" for i, btn in enumerate(buttons, 1)])
                full_message = cleaned_reply + button_text + "\n\nPlease reply with the number of your choice."
                await whatsapp_service.send_message(user_phone, full_message)
            else:
                await whatsapp_service.send_message(user_phone, initial_reply)
            
            return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        # Load existing session state
        session = whatsapp_sessions[session_id]
        conversation_history = session["history"]
        email_validation_state = session["email_state"]
        phone_validation_state = session["phone_state"]
        context = session["context"]
        user_id = session["user_id"]
        
        # Set profession
        gpt_service.set_profession(str(context.get("profession") or "Clinic"))
        
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
                # Second selection - must be service
                services = context.get("service_types", [])
                treatments = context.get("treatment_plans", [])
                all_options = services + [t.get("question", str(t)) for t in treatments if isinstance(t, dict)]
                
                logger.info(f"WhatsApp: Detected service selection. Available options: {all_options}")
                
                if 1 <= number <= len(all_options):
                    selected = all_options[number - 1]
                    service_name = selected.get("question", selected) if isinstance(selected, dict) else selected
                    enhanced_user_text = f"{number} - {service_name}"
                    logger.info(f"WhatsApp: User selected service #{number} -> '{service_name}'")
                else:
                    logger.warning(f"WhatsApp: User selected invalid service number {number}, available: {len(all_options)}")
            else:
                logger.info(f"WhatsApp: Number {number} - context unclear, treating as raw input")
        
        # Process the message similar to WebSocket flow
        gpt_service.set_profession(str(context.get("profession") or "Clinic"))
        
        # Add user message to history (use enhanced text for better GPT understanding)
        conversation_history.append({"role": "user", "content": enhanced_user_text})
        
        # Get GPT response (with WhatsApp flag)
        reply = await gpt_service.agent_reply(conversation_history, user_text, context, {}, is_whatsapp=True)
        
        # Check if it's JSON (lead completion)
        parsed_json = _maybe_parse_json(reply)
        if parsed_json and isinstance(parsed_json, dict):
            # Add WhatsApp phone number to the lead JSON
            parsed_json["leadPhoneNumber"] = user_phone
            
            # Check email verification only if enabled
            email_valid = True
            if context.get("integration", {}).get("validateEmail", True):
                email_valid = _validate_email_verification(email_validation_state)
            
            # Phone is already verified by WhatsApp (skip phone OTP)
            if email_valid:
                # Create lead
                try:
                    ok, _ = await lead_service.create_public_lead(user_id, parsed_json)
                    if ok:
                        final_msg = "Thanks! I have your details and someone will get back to you soon. Bye!"
                    else:
                        final_msg = "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!"
                except Exception:
                    final_msg = "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!"
                
                await whatsapp_service.send_message(user_phone, final_msg)
                
                # Clean up session after lead creation
                del whatsapp_sessions[session_id]
                if user_phone in phone_to_session and phone_to_session[user_phone] == session_id:
                    del phone_to_session[user_phone]
                
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
        # Check if response contains buttons
        cleaned_reply, buttons = gpt_service.extract_buttons_from_response(reply)
        
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
            success, message = await whatsapp_service.send_message(user_phone, full_message)
            if not success:
                logger.error("Failed to send WhatsApp message: %s", message)
        else:
            # Send simple text message (ensure no button tags)
            success, message = await whatsapp_service.send_message(user_phone, cleaned_reply)
            if not success:
                logger.error("Failed to send WhatsApp message: %s", message)
        
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
