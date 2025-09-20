import json
import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .services.context_service import ContextService
from .services.lead_service import LeadService
from .services.gpt_service import GptService
from .services.email_validation_service import EmailValidationService


logger = logging.getLogger("assistly")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Assistly AI Chatbot WS")
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

def _validate_required_fields(parsed: Dict) -> bool:
    """Validate that all required fields are present in the parsed JSON."""
    required_fields = ["leadType", "serviceType", "leadName", "leadEmail", "leadPhoneNumber"]
    return all(parsed.get(field) for field in required_fields)

def _validate_email_verification(email_validation_state: Dict) -> bool:
    """Validate that email has been verified."""
    return email_validation_state.get("otp_verified", False)

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
            except json.JSONDecodeError:
                user_text = data

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
                    if ok:
                        email_validation_state["otp_verified"] = True
                        # Let GPT handle the success message naturally
                        pass
                    else:
                        # Let GPT handle invalid OTP naturally
                        pass
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
            
            # Get GPT response
            reply = await gpt_service.agent_reply(conversation_history, user_text, context, {})
            
            # Check if it's JSON (lead completion)
            parsed_json = _maybe_parse_json(reply)
            if parsed_json and isinstance(parsed_json, dict):
                if _validate_required_fields(parsed_json) and _validate_email_verification(email_validation_state):
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
                    # Missing fields, ask for them
                    missing_fields = [field for field in ["leadType", "serviceType", "leadName", "leadEmail", "leadPhoneNumber"] if not parsed_json.get(field)]
                    missing_msg = f"I need a bit more information. Please provide: {', '.join(missing_fields)}"
                    conversation_history.append({"role": "assistant", "content": missing_msg})
                    await websocket.send_json({"type": "bot", "content": missing_msg})
            else:
                # Regular conversation response
                conversation_history.append({"role": "assistant", "content": reply})
                await websocket.send_json({"type": "bot", "content": reply})
            
            # Keep history manageable
            conversation_history = conversation_history[-10:]

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for user_id=%s", user_id)
