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
from .services.phone_validation_service import PhoneValidationService


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
                    
                    # Get country code from context
                    country_code = context.get("country", "US")
                    logger.info(f"Using country code from context: '{country_code}' for phone: '{phone}'")
                    
                    # Send SMS OTP
                    ok, _ = await phone_validation_service.send_sms_otp(user_id, phone, country_code)
                    
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
                    # Verify phone OTP
                    country_code = context.get("country", "US")
                    logger.info(f"Verifying phone OTP for phone: '{phone_validation_state['phone']}' with country: '{country_code}'")
                    ok, message = await phone_validation_service.verify_sms_otp(
                        user_id, 
                        phone_validation_state["phone"], 
                        otp_code,
                        country_code
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
