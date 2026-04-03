import json
import logging
import re
import time
import secrets
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict

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
from .services.twilio_messaging_service import TwilioMessagingService, CHANNEL_MESSENGER, CHANNEL_INSTAGRAM
from .services.instagram_graph_service import InstagramGraphService
from .services.messenger_graph_service import MessengerGraphService
from .services.voice_agent_service import VoiceAgentService
from .services.rag_service import RAGService
from .services.calendar_service import CalendarService
from .services.conversation_state import FlowController, ConversationState
from .services.response_generator import ResponseGenerator
from .services.data_extractors import DataExtractor
from .services.lead_type_resolver import LeadTypeResolutionMode, resolve_lead_type
from .utils.phone_utils import format_phone_number
from .utils.language_utils import detect_language, get_language_name_for_prompt
from .utils.response_strings import get_string

from twilio.twiml.voice_response import VoiceResponse


logger = logging.getLogger("assistly")
logging.basicConfig(level=logging.INFO)


# ─── TypedDicts for Messenger and Instagram sessions ────────────────────────

class EmailValidationState(TypedDict, total=False):
    """Email OTP validation state within a session."""
    email: Optional[str]
    otp_sent: bool
    otp_verified: bool
    customer_name: Optional[str]


class PhoneValidationState(TypedDict, total=False):
    """Phone OTP validation state within a session."""
    phone: Optional[str]
    otp_sent: bool
    otp_verified: bool


class MessengerSession(TypedDict, total=False):
    """Typed structure for in-memory Messenger sessions."""

    user_id: str  # PSID – conversation partner
    recipient_id: str  # Page ID
    history: List[Dict[str, str]]
    email_state: EmailValidationState
    phone_state: PhoneValidationState
    context: Dict[str, object]
    user_id_owner: str
    app_id: Optional[str]
    created_at: float
    last_activity: float
    flow_controller: Optional[FlowController]
    response_generator: Optional[ResponseGenerator]
    page_access_token: str
    response_language_code: str
    response_language: str


class InstagramSession(TypedDict, total=False):
    """Typed structure for in-memory Instagram sessions."""

    user_id: str  # IGSID – conversation partner
    recipient_id: str  # IG Business Account ID
    history: List[Dict[str, str]]
    email_state: EmailValidationState
    phone_state: PhoneValidationState
    context: Dict[str, object]
    user_id_owner: str
    app_id: Optional[str]
    created_at: float
    last_activity: float
    flow_controller: Optional[FlowController]
    response_generator: Optional[ResponseGenerator]
    instagram_access_token: str
    response_language_code: str
    response_language: str


# In-memory storage for WhatsApp conversations (session-based)
whatsapp_sessions: Dict[str, Dict[str, Any]] = {}

# Mapping: phone number -> current active session_id
phone_to_session: Dict[str, str] = {}

# Messenger: session_id -> session data; (page_id, user_id) -> session_id
messenger_sessions: Dict[str, MessengerSession] = {}
messenger_key_to_session: Dict[str, str] = {}

def get_or_create_messenger_session(page_id: str, user_id: str) -> Tuple[str, bool]:
    """Get or create session for Messenger. Returns (session_id, is_new)."""
    key = f"messenger_{page_id}_{user_id}"
    current_time = time.time()
    if key in messenger_key_to_session:
        session_id = messenger_key_to_session[key]
        if session_id in messenger_sessions:
            session = messenger_sessions[session_id]
            if current_time - session.get("last_activity", 0) > SESSION_TIMEOUT:
                del messenger_sessions[session_id]
                del messenger_key_to_session[key]
            else:
                session["last_activity"] = current_time
                return session_id, False
        else:
            del messenger_key_to_session[key]
    session_id = secrets.token_urlsafe(16)
    messenger_key_to_session[key] = session_id
    logger.info("Created new Messenger session %s for page=%s user=%s", session_id, page_id, user_id)
    return session_id, True

# Instagram: session_id -> session data; (sender_id, user_id) -> session_id
instagram_sessions: Dict[str, InstagramSession] = {}
instagram_key_to_session: Dict[str, str] = {}

def get_or_create_instagram_session(sender_id: str, user_id: str) -> Tuple[str, bool]:
    """Get or create session for Instagram. Returns (session_id, is_new)."""
    key = f"instagram_{sender_id}_{user_id}"
    current_time = time.time()
    if key in instagram_key_to_session:
        session_id = instagram_key_to_session[key]
        if session_id in instagram_sessions:
            session = instagram_sessions[session_id]
            if current_time - session.get("last_activity", 0) > SESSION_TIMEOUT:
                del instagram_sessions[session_id]
                del instagram_key_to_session[key]
            else:
                session["last_activity"] = current_time
                return session_id, False
        else:
            del instagram_key_to_session[key]
    session_id = secrets.token_urlsafe(16)
    instagram_key_to_session[key] = session_id
    logger.info("Created new Instagram session %s for sender=%s user=%s", session_id, sender_id, user_id)
    return session_id, True

# Session timeout from environment variable (default: 5 minutes)
SESSION_TIMEOUT = settings.session_timeout_seconds

# Conversation-style toggle refresh:
# Backend notifies this service when `integration.conversationStyle` changes.
# For existing Messenger/Instagram sessions we do NOT clear them immediately.
# Instead, the next incoming message will apply the new style only after
# the session has been idle for `idle_seconds`.
conversation_style_change_requests: Dict[str, Dict[str, Any]] = {}

# Web widget: resume interrupted chats (in-memory; best-effort across reconnects / page navigations)
WIDGET_CHAT_RESUME: Dict[str, Dict[str, Any]] = {}
WIDGET_RESUME_TTL_SEC = 86400 * 2
WIDGET_RESUME_MAX_KEYS = 4000

# Mid-flow lead-type switch is only meaningful after the user left the greeting / lead menu.
_LEAD_TYPE_SWITCH_ALLOWED_STATES = frozenset(
    s
    for s in ConversationState
    if s
    not in (
        ConversationState.GREETING,
        ConversationState.LEAD_TYPE_SELECTION,
        ConversationState.COMPLETE,
    )
)

# Voice agent sessions
voice_agent_service = VoiceAgentService(settings)


def _integration_google_review_url(integration: Any) -> Optional[str]:
    """Return Google review URL when enabled; tolerates string booleans from APIs."""
    if not isinstance(integration, dict):
        return None
    raw_en = integration.get("googleReviewEnabled")
    if isinstance(raw_en, str):
        enabled = raw_en.strip().lower() in ("1", "true", "yes", "on")
    else:
        enabled = bool(raw_en)
    if not enabled:
        return None
    url = str(integration.get("googleReviewUrl") or "").strip()
    return url or None


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


class ConversationStyleChangeBody(BaseModel):
    app_id: str
    conversation_style: bool
    idle_seconds: int = 120


@app.post("/api/v1/social/conversation-style/invalidate-sessions")
async def invalidate_conversation_style_sessions(
    body: ConversationStyleChangeBody,
    x_invalidate_sessions_secret: Optional[str] = Header(default=None, alias="X-Invalidate-Sessions-Secret"),
) -> Dict[str, Any]:
    """
    Mark existing Messenger/Instagram sessions for `app_id` as stale.

    We don't clear in-memory sessions immediately. When a session becomes idle
    for `idle_seconds`, the next incoming message will apply the new
    `conversationStyle` to the session context.
    """
    if settings.tp_sign_secret:
        if x_invalidate_sessions_secret != settings.tp_sign_secret:
            raise HTTPException(status_code=401, detail="Invalid or missing secret")

    clean_app_id = (body.app_id or "").strip()
    if not clean_app_id:
        raise HTTPException(status_code=400, detail="app_id required")

    idle_seconds = max(0, int(body.idle_seconds or 120))
    desired_conversation_style = bool(body.conversation_style)

    current = conversation_style_change_requests.get(clean_app_id)
    current_version = int(current.get("version", 0)) if isinstance(current, dict) else 0
    new_version = current_version + 1

    conversation_style_change_requests[clean_app_id] = {
        "version": new_version,
        "conversation_style": desired_conversation_style,
        "idle_seconds": idle_seconds,
        "requested_at": time.time(),
    }

    logger.info(
        "Marked social sessions stale after idle: app_id=%s version=%s conversation_style=%s idle_seconds=%s",
        clean_app_id,
        new_version,
        desired_conversation_style,
        idle_seconds,
    )
    return {"status": "ok", "app_id": clean_app_id, "version": new_version, "idle_seconds": idle_seconds}


def _maybe_parse_json(text: str) -> Optional[Dict]:
    """Parse JSON from text if it looks like JSON."""
    content = text.strip()
    if not (content.startswith("{") and content.endswith("}")):
        return None
    try:
        return json.loads(content)
    except Exception:
        return None


def _looks_like_internal_lead_payload(text: str) -> bool:
    """Detect internal lead payloads that should never be sent to end users."""
    if not text or not isinstance(text, str):
        return False

    parsed = _maybe_parse_json(text)
    if not isinstance(parsed, dict):
        return False

    payload_keys = {
        "leadType",
        "serviceType",
        "leadName",
        "leadEmail",
        "leadPhoneNumber",
        "history",
        "workflowAnswers",
        "appointmentSlot",
        "sourceChannel",
        "title",
    }
    return any(k in parsed for k in payload_keys)


async def _continue_after_otp_delivery_failed(
    flow_controller: FlowController,
    response_generator: ResponseGenerator,
    conversation_history: List[Dict[str, str]],
    context: Dict[str, Any],
    lang_code: str,
    kind: str,
) -> str:
    """
    When email/SMS OTP cannot be sent (provider/network), skip verification and advance the flow.
    Wrong codes entered by the user are handled separately (stay in verification).
    """
    logger.warning("OTP delivery failed; skipping verification and advancing flow (kind=%s)", kind)
    if kind == "email":
        flow_controller.skip_email_verification_after_send_failure()
        prefix = get_string("otp_unavailable_skip_email", lang_code)
    else:
        flow_controller.skip_phone_verification_after_send_failure()
        prefix = get_string("otp_unavailable_skip_phone", lang_code)
    next_prompt = await response_generator._generate_state_response(
        flow_controller.state, "", conversation_history, context
    )
    body = (next_prompt or "").strip()
    return f"{prefix}\n\n{body}" if body else prefix


async def _continue_after_otp_delivery_failed_with_session(
    flow_controller: FlowController,
    response_generator: ResponseGenerator,
    conversation_history: List[Dict[str, str]],
    context: Dict[str, Any],
    lang_code: str,
    kind: str,
    email_validation_state: Optional[Dict[str, Any]] = None,
    phone_validation_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Session-aware wrapper for WhatsApp/Messenger/Instagram handlers.
    Keeps per-channel OTP state in sync when delivery fails.
    """
    if kind == "email" and email_validation_state is not None:
        email_validation_state["otp_sent"] = False
        email_validation_state["otp_verified"] = True
    if kind == "phone" and phone_validation_state is not None:
        phone_validation_state["otp_sent"] = False
        phone_validation_state["otp_verified"] = True
    return await _continue_after_otp_delivery_failed(
        flow_controller,
        response_generator,
        conversation_history,
        context,
        lang_code,
        kind,
    )


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


def _is_availability_intent(text: str) -> bool:
    """Detect if user is asking for calendar availability or wants to add/book an appointment."""
    if not text or not isinstance(text, str):
        return False
    t = text.strip().lower()
    if len(t) > 200:  # Long messages are likely not a simple availability question
        return False
    availability_phrases = [
        # "When are you free?" style
        "when are you free",
        "when are we free",
        "are you free",
        "what times are available",
        "what slots are available",
        "available times",
        "available slots",
        "when can i book",
        "when can we meet",
        "do you have availability",
        "show me your availability",
        "your availability",
        "free slots",
        "free times",
        "open slots",
        "book a slot",
        "available this week",
        "available tomorrow",
        "free this week",
        # "Add / book / schedule appointment" and lead-type style
        "add an appointment",
        "add appointment",
        "schedule an appointment",
        "book an appointment",
        "make an appointment",
        "i'd like to schedule",
        "i would like to schedule",
        "want to schedule",
        "want to book",
        "schedule a meeting",
        "book a meeting",
        "set up a meeting",
        "set up an appointment",
        # Lead type options that are about scheduling (e.g. "1 - I would like to schedule a checkup (leadType: ...)")
        "schedule a routine",
        "schedule a property",
        "schedule a consultation",
        "schedule a viewing",
        "schedule a service",
        "schedule a checkup",
        "book a treatment",
        "book treatment",
    ]
    return any(p in t for p in availability_phrases)


def _extract_index_choice(text: str) -> Optional[int]:
    """Extract numeric choice from text like '2', '2. Fri', or '2 - 10:00 AM'."""
    if not text or not isinstance(text, str):
        return None
    match = re.match(r"^\s*(\d{1,2})\b", text.strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _format_slot_time_local(iso_str: str, iana_timezone: str) -> str:
    """Format a UTC ISO datetime to a 12-hour local time string using the given IANA timezone."""
    try:
        dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        dt_local = dt_utc
        if iana_timezone and iana_timezone != "UTC":
            try:
                from zoneinfo import ZoneInfo
                dt_local = dt_utc.astimezone(ZoneInfo(iana_timezone))
            except Exception:
                pass
        h = dt_local.hour
        m = dt_local.minute
        period = "AM" if h < 12 else "PM"
        display_h = h % 12 or 12
        return f"{display_h}:{m:02d} {period}"
    except Exception:
        return iso_str


def _get_tz_label(iana_timezone: str) -> str:
    """Return a readable timezone label like 'Asia/Karachi (GMT+5)'."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(iana_timezone)
        offset = datetime.now(tz).utcoffset()
        total_minutes = int(offset.total_seconds() / 60)
        sign = "+" if total_minutes >= 0 else "-"
        abs_h, abs_m = divmod(abs(total_minutes), 60)
        offset_str = f"GMT{sign}{abs_h}" if abs_m == 0 else f"GMT{sign}{abs_h}:{abs_m:02d}"
        return f"{iana_timezone} ({offset_str})"
    except Exception:
        return iana_timezone or "UTC"


def _calendar_tz_from_integration(integration: Dict[str, Any]) -> str:
    tz = (integration or {}).get("calendarTimezone") or "UTC"
    return str(tz) if tz else "UTC"


def _availability_window_from_tomorrow_utc(integration: Dict[str, Any], horizon_days: int = 14) -> Tuple[str, str]:
    """UTC ISO Z range for calendar API: start tomorrow 00:00 in the business calendar timezone."""
    from zoneinfo import ZoneInfo

    cal_tz = _calendar_tz_from_integration(integration)
    try:
        zi = ZoneInfo(cal_tz)
    except Exception:
        zi = ZoneInfo("UTC")
    now_local = datetime.now(zi)
    tomorrow = (now_local + timedelta(days=1)).date()
    start_local = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0, tzinfo=zi)
    end_local = start_local + timedelta(days=horizon_days)
    start_utc = start_local.astimezone(timezone.utc).replace(microsecond=0)
    end_utc = end_local.astimezone(timezone.utc).replace(microsecond=0)
    return (
        start_utc.isoformat().replace("+00:00", "Z"),
        end_utc.isoformat().replace("+00:00", "Z"),
    )


def _local_slot_date_key(iso_start: str, cal_tz: str) -> str:
    try:
        from zoneinfo import ZoneInfo

        dt_utc = datetime.fromisoformat(iso_start.replace("Z", "+00:00"))
        return dt_utc.astimezone(ZoneInfo(cal_tz)).strftime("%Y-%m-%d")
    except Exception:
        return (iso_start or "")[:10]


def _tomorrow_date_key_in_tz(cal_tz: str) -> str:
    from zoneinfo import ZoneInfo

    try:
        zi = ZoneInfo(cal_tz)
    except Exception:
        zi = ZoneInfo("UTC")
    return (datetime.now(zi) + timedelta(days=1)).strftime("%Y-%m-%d")


def _filter_slots_from_tomorrow_local(slots: List[Dict[str, Any]], cal_tz: str) -> List[Dict[str, Any]]:
    if not slots:
        return []
    cutoff = _tomorrow_date_key_in_tz(cal_tz)
    return [s for s in slots if _local_slot_date_key((s.get("start") or ""), cal_tz) >= cutoff]


def _normalize_service_title_for_plan_match(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def _find_post_booking_note(service_plans: List[Any], service_title: str) -> str:
    target = _normalize_service_title_for_plan_match(service_title)
    for sp in service_plans or []:
        if not isinstance(sp, dict):
            continue
        q = _normalize_service_title_for_plan_match(str(sp.get("question", "")))
        if q == target:
            raw = sp.get("postBookingNote")
            return (str(raw).strip() if raw is not None else "")
    return ""


def _html_post_booking_to_chat_text(html: str) -> str:
    from html import unescape

    s = html
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p>\s*", "\n", s, flags=re.I)
    s = re.sub(r"<p[^>]*>", "", s, flags=re.I)
    s = re.sub(r"<li[^>]*>", "\n• ", s, flags=re.I)
    s = re.sub(r"</li>", "", s, flags=re.I)
    s = re.sub(r"</(ul|ol)>", "\n", s, flags=re.I)
    s = re.sub(r"<(ul|ol)[^>]*>", "\n", s, flags=re.I)

    def _bold_tag(m: Any) -> str:
        inner = (m.group(1) or "").strip()
        return f"**{inner}**" if inner else ""

    s = re.sub(r"<strong[^>]*>([\s\S]*?)</strong>", _bold_tag, s, flags=re.I)
    s = re.sub(r"<b[^>]*>([\s\S]*?)</b>", _bold_tag, s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = unescape(s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines)


def _format_post_booking_note_for_chat(note: str) -> str:
    raw = (note or "").strip()
    if not raw:
        return ""
    if "<" in raw and ">" in raw:
        return _html_post_booking_to_chat_text(raw)
    lines_out: List[str] = []
    for ln in raw.splitlines():
        piece = ln.strip()
        if not piece:
            continue
        if re.match(r"^[-•*]\s*", piece):
            piece = re.sub(r"^[-•*]\s+", "", piece)
            lines_out.append(f"• {piece}")
        else:
            lines_out.append(f"• {piece}")
    return "\n".join(lines_out)


def _snapshot_flow_controller_for_resume(fc: FlowController) -> Dict[str, Any]:
    wm = fc.workflow_manager
    return {
        "state": fc.state.value,
        "collected_data": json.loads(json.dumps(fc.collected_data, default=str)),
        "otp_state": dict(fc.otp_state),
        "workflow": wm.export_state() if wm else None,
    }


def _restore_flow_controller_from_resume(fc: FlowController, snap: Optional[Dict[str, Any]]) -> None:
    if not snap or not isinstance(snap, dict):
        return
    st = snap.get("state")
    if st:
        try:
            fc.state = ConversationState(st)
        except Exception:
            pass
    cd = snap.get("collected_data")
    if isinstance(cd, dict):
        fc.collected_data = {**fc.collected_data, **cd}
    ot = snap.get("otp_state")
    if isinstance(ot, dict):
        fc.otp_state = {**fc.otp_state, **ot}
    wf_snap = snap.get("workflow")
    if wf_snap:
        from .services.workflow_manager import WorkflowManager

        wm = WorkflowManager(fc.context)
        wm.import_state(wf_snap)
        fc.workflow_manager = wm


def _prune_widget_resume_store() -> None:
    if len(WIDGET_CHAT_RESUME) <= WIDGET_RESUME_MAX_KEYS:
        return
    items = sorted(WIDGET_CHAT_RESUME.items(), key=lambda kv: kv[1].get("saved_at", 0))
    drop = max(1, len(items) // 2)
    for k, _ in items[:drop]:
        WIDGET_CHAT_RESUME.pop(k, None)


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

def _bot_requests_file_upload(text: str) -> bool:
    """Return True if the bot message is asking the user to upload a file/document."""
    lower = text.lower()
    request_verbs = ("upload", "send us", "please send", "attach", "provide", "share", "submit", "email us")
    file_nouns = ("file", "document", "photo", "image", "picture", "form", "certificate", "id", "proof", "receipt", "invoice", "attachment")
    has_verb = any(v in lower for v in request_verbs)
    has_noun = any(n in lower for n in file_nouns)
    return has_verb and has_noun


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    resume_key = (websocket.query_params.get("resume") or "").strip()
    widget_session_complete = False
    skip_history_replay = (websocket.query_params.get("skip_history_replay") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    app_id: Optional[str] = websocket.query_params.get("app_id")
    user_id: Optional[str] = websocket.query_params.get("user_id")
    country_hint_raw: Optional[str] = websocket.query_params.get("country")
    country_hint = (country_hint_raw or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", country_hint):
        country_hint = ""

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
    flow_controller.update_collected_data("sourceChannel", "web")
    
    extractor = DataExtractor()
    conversation_history: List[Dict[str, str]] = []
    # Calendar flow state (persists across messages in this WebSocket session)
    calendar_flow: Optional[str] = None
    calendar_days: List[Dict[str, Any]] = []
    calendar_slots: List[Dict[str, Any]] = []
    calendar_free_slots: List[Dict[str, Any]] = []
    calendar_selected_day: Optional[str] = None
    calendar_pending_slot: Dict[str, Any] = {}
    user_timezone: str = "UTC"

    # Build RAG off the event loop so the client gets the first message(s) quickly.
    # Blocking embeddings here previously stretched time-to-first-byte (~5s+), which
    # triggered proxy/browser WebSocket closes (1006) before the greeting was sent.
    rag_build_task = asyncio.create_task(
        asyncio.to_thread(rag_service.build_vector_store, context)
    )

    lead_id: Optional[str] = None
    last_snapshot: Dict[str, Any] = {}
    restored = False
    if resume_key and app_id:
        resume_blob = WIDGET_CHAT_RESUME.get(resume_key)
        if (
            isinstance(resume_blob, dict)
            and str(resume_blob.get("app_id") or "") == str(app_id)
            and (time.time() - float(resume_blob.get("saved_at", 0))) < WIDGET_RESUME_TTL_SEC
            and not resume_blob.get("terminal")
        ):
            restored = True
            conversation_history = list(resume_blob.get("conversation_history") or [])
            calendar_flow = resume_blob.get("calendar_flow")
            calendar_days = list(resume_blob.get("calendar_days") or [])
            calendar_slots = list(resume_blob.get("calendar_slots") or [])
            calendar_free_slots = list(resume_blob.get("calendar_free_slots") or [])
            calendar_selected_day = resume_blob.get("calendar_selected_day")
            calendar_pending_slot = dict(resume_blob.get("calendar_pending_slot") or {})
            user_timezone = str(resume_blob.get("user_timezone") or "UTC")
            _restore_flow_controller_from_resume(flow_controller, resume_blob.get("flow"))
            lid = resume_blob.get("lead_id")
            if lid:
                lead_id = str(lid)
                flow_controller.update_collected_data("leadId", lead_id)

    if not restored:
        flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)
        try:
            location_country = country_hint or (context.get("country") or "US")
            ok_lead, lead_resp = await lead_service.create_interaction_lead(
                app_id=app_id,
                user_id=user_id,
                location={"country": location_country, "countryCode": location_country},
                initial_interaction="widget_opened",
                source_channel="web",
                dedupe_window_hours=settings.lead_dedupe_window_hours,
            )
            if ok_lead and isinstance(lead_resp, dict):
                lead_obj = ((lead_resp.get("data") or {}).get("lead") or {})
                lead_id = str(lead_obj.get("_id") or "")
                if lead_id:
                    flow_controller.update_collected_data("leadId", lead_id)
        except Exception as lead_exc:
            logger.warning("WebSocket interaction lead creation failed: %s", lead_exc)

    # Greeting for new sessions; resumed sessions replay transcript unless the client
    # already has the thread (sessionStorage) and passes skip_history_replay=1.
    try:
        if restored and not skip_history_replay:
            for turn in conversation_history[-50:]:
                role = turn.get("role")
                content = str(turn.get("content") or "")
                if not content:
                    continue
                if role == "assistant":
                    await websocket.send_json({"type": "bot", "content": content})
                elif role == "user":
                    await websocket.send_json({"type": "user_replay", "content": content})
        elif restored and skip_history_replay:
            logger.info("WebSocket resume: skip_history_replay=1, not re-sending transcript")
        elif (not restored) and skip_history_replay:
            # Client still has the thread in sessionStorage (skip flag) but this worker has no
            # resume blob (TTL, cold start, or new resume id). Sending greeting again duplicates UI.
            logger.info(
                "WebSocket: skip_history_replay=1 without server resume — skipping greeting "
                "(client already showing transcript)"
            )
        else:
            initial_reply = await response_generator.generate_greeting(context, channel="web", first_message=None)
            conversation_history.append({"role": "assistant", "content": initial_reply})
            await websocket.send_json({"type": "bot", "content": initial_reply})
    except WebSocketDisconnect:
        logger.info(
            "Client disconnected before/during initial WebSocket messages (user_id=%s)",
            user_id,
        )
    except Exception as greeting_exc:
        logger.exception("Failed to generate greeting: %s", greeting_exc)
        try:
            await websocket.send_json(
                {"type": "error", "content": "Failed to load greeting. Please try again."}
            )
        except Exception:
            pass

    # Lazily await RAG on first user turn so the receive loop can run while embeddings
    # finish (important for skip_history_replay: client may send immediately with no prior bot frame).
    rag_ready = False

    async def ensure_rag_ready() -> None:
        nonlocal rag_ready
        if rag_ready:
            return
        try:
            await rag_build_task
        except Exception as rag_wait_exc:
            logger.warning("RAG build task failed (non-fatal): %s", rag_wait_exc)
        rag_ready = True

    try:
        while True:
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                user_text = parsed.get("content") or parsed.get("text") or data
            except json.JSONDecodeError:
                parsed = {}
                user_text = data

            # Capture user's IANA timezone sent from the widget
            if isinstance(parsed, dict):
                tz_hint = parsed.get("timezone")
                if tz_hint and isinstance(tz_hint, str) and len(tz_hint) < 64:
                    user_timezone = tz_hint

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
                            # Signal the frontend to show the file upload button if the next question asks for a file
                            if _bot_requests_file_upload(next_text):
                                await websocket.send_json({"type": "enable_file_upload"})
                    else:
                        # Workflow done – move state forward (ConversationState is imported at top of module)
                        flow_controller.collected_data["workflowAnswers"] = wm.get_workflow_answers()
                        flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                        await ensure_rag_ready()
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

            await ensure_rag_ready()

            # Bad or partial resume blobs may leave state on default GREETING while the client
            # already showed the greeting. The next user action is lead-type selection — not a
            # mid-flow switch. Stale collected_data.leadType from a merge bug would otherwise
            # satisfy "existing lead type" + state != LEAD_TYPE_SELECTION and fire the switch
            # handler on the *first* button tap (re-greeting / wrong branch).
            if flow_controller.state == ConversationState.GREETING:
                flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)
                _user_turns_so_far = sum(
                    1 for m in conversation_history if m.get("role") == "user"
                )
                if _user_turns_so_far == 0:
                    flow_controller.update_collected_data("leadType", None)
                    flow_controller.collected_data["title"] = None
                    flow_controller.collected_data["leadTypeSwitchHistory"] = []

            integration = context.get("integration") or {}
            app_id_for_calendar = app_id  # from query params when widget is opened with ?app_id=...

            # Global lead-type switch interrupt:
            # If user selects a different lead type mid-flow, terminate current branch
            # (including calendar/workflow) and restart from the new lead type gracefully.
            lead_types_list = context.get("lead_types", [])
            # Lead-type *switch* detection only when not in explicit lead-type selection
            # (that path uses resolve_lead_type(..., LEAD_SELECTION) inside ResponseGenerator).
            # MID_FLOW_SWITCH ignores plain digits and uses strict matching so workflow
            # numeric answers are not mistaken for lead types.
            switched_lead_type = None
            if flow_controller.state != ConversationState.LEAD_TYPE_SELECTION:
                switched_lead_type = resolve_lead_type(
                    str(user_text), lead_types_list, LeadTypeResolutionMode.MID_FLOW_SWITCH
                )

            # Only treat as a mid-flow "switch" if we already had a lead type and we are past
            # the greeting / initial lead menu (see _LEAD_TYPE_SWITCH_ALLOWED_STATES).
            _existing_lt = (flow_controller.collected_data.get("leadType") or "").strip()
            if (
                switched_lead_type
                and _existing_lt
                and flow_controller.state in _LEAD_TYPE_SWITCH_ALLOWED_STATES
                and _existing_lt.lower()
                != (switched_lead_type.get("value") or "").strip().lower()
            ):
                previous_lead_type = flow_controller.collected_data.get("leadType")
                # Break any ongoing calendar branch
                calendar_flow = None
                calendar_days = []
                calendar_slots = []
                calendar_free_slots = []
                calendar_selected_day = None
                calendar_pending_slot = {}

                # Reset service/workflow context and apply the new lead type
                if flow_controller.workflow_manager:
                    flow_controller.workflow_manager.reset()
                flow_controller.update_collected_data("leadType", switched_lead_type.get("value"))
                flow_controller.update_collected_data("serviceType", None)
                flow_controller.update_collected_data("workflowAnswers", {})
                flow_controller.update_collected_data("appointmentSlot", None)

                # Keep already provided personal/verification info on lead-type switch.
                # We only reset service/workflow/booking branch data.

                existing_switch_history = flow_controller.collected_data.get("leadTypeSwitchHistory") or []
                switch_history = list(existing_switch_history) if isinstance(existing_switch_history, list) else []
                switch_history.append({
                    "from": previous_lead_type,
                    "to": switched_lead_type.get("value"),
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                flow_controller.update_collected_data("leadTypeSwitchHistory", switch_history)

                # Keep already collected name; continue from the right next step
                if not flow_controller.collected_data.get("leadName"):
                    flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                elif not flow_controller.collected_data.get("leadEmail"):
                    flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                elif (
                    not flow_controller.skip_phone_collection
                    and not flow_controller.collected_data.get("leadPhoneNumber")
                ):
                    flow_controller.transition_to(ConversationState.PHONE_COLLECTION)
                else:
                    if flow_controller.is_booking_lead_type():
                        flow_controller.transition_to(ConversationState.SERVICE_SELECTION)
                    else:
                        flow_controller.transition_to(ConversationState.WORKFLOW_QUESTION)

                # Polished switch confirmation using empathy prefix
                try:
                    _lt_empathy = await response_generator._generate_empathy_prefix_for_lead_type(
                        switched_lead_type.get("value"), context.get("lead_types", [])
                    )
                except Exception:
                    _lt_empathy = ""
                _lt_label = switched_lead_type.get("text") or switched_lead_type.get("value") or "the new option"
                switch_msg = (
                    f"{_lt_empathy} I've updated your request to **{_lt_label}**."
                    if _lt_empathy
                    else f"Got it — switched to **{_lt_label}**."
                )
                next_prompt = await response_generator._generate_state_response(
                    flow_controller.state, "", conversation_history, context
                )
                reply = f"{switch_msg}\n\n{next_prompt}"
                conversation_history.append({"role": "assistant", "content": reply})
                await websocket.send_json({"type": "bot", "content": reply})
                continue

            # ── Side-question interjection ─────────────────────────────────────────────
            # When a required step is pending and the user asks an in-industry question,
            # answer via RAG/FAQ then re-prompt the exact pending step — without advancing
            # state or writing any collected-data fields on this turn.
            _INTERJECTION_STATES = {
                ConversationState.SERVICE_SELECTION,
                ConversationState.WORKFLOW_QUESTION,
                ConversationState.NAME_COLLECTION,
                ConversationState.EMAIL_COLLECTION,
                ConversationState.PHONE_COLLECTION,
                ConversationState.APPOINTMENT_OFFER,
                ConversationState.CALENDAR_BOOKING,
                ConversationState.APPOINTMENT_CONFIRMATION,
            }
            _Q_STARTERS = {
                "what", "how", "why", "when", "where", "is", "does", "can", "do",
                "are", "will", "would", "could", "should", "which", "who",
                "tell", "explain", "describe",
            }

            def _looks_like_question(txt: str) -> bool:
                t = (txt or "").strip()
                if not t:
                    return False
                if t.endswith("?"):
                    return True
                first = t.lower().split()[0] if t.lower().split() else ""
                return first in _Q_STARTERS

            if (
                flow_controller.state in _INTERJECTION_STATES
                and not calendar_flow  # calendar flow has its own dedicated handler
                and _looks_like_question(str(user_text))
            ):
                _rag_answer: Optional[str] = None
                try:
                    _rag_answer = await rag_service.answer_faq_question(
                        str(user_text),
                        profession=response_generator.profession,
                        context_data=context,
                    )
                except Exception as _rag_err:
                    logger.warning("interjection RAG answer failed: %s", _rag_err)
                # Only intercept when we got a clean answer (not a JSON lead payload)
                if _rag_answer and not _is_lead_json(_rag_answer):
                    _pending_step = await response_generator._generate_state_response(
                        flow_controller.state, "", conversation_history, context
                    )
                    _interject_reply = f"{_rag_answer}\n\n---\n\n{_pending_step}"
                    logger.info(
                        "interjection_detected state=%s interjection_answered_from_rag=True interjection_resumed_state=%s",
                        flow_controller.state.value, flow_controller.state.value
                    )
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": _interject_reply})
                    await websocket.send_json({"type": "bot", "content": _interject_reply})
                    continue
                # No usable RAG answer → fall through to normal state-machine processing

            # Strict personal-info-first guard:
            # Once lead type is selected, enforce name -> email -> phone before
            # service/workflow/booking states.
            if flow_controller.collected_data.get("leadType") and flow_controller.state in (
                ConversationState.SERVICE_SELECTION,
                ConversationState.WORKFLOW_QUESTION,
                ConversationState.APPOINTMENT_OFFER,
                ConversationState.CALENDAR_BOOKING,
                ConversationState.APPOINTMENT_CONFIRMATION,
            ):
                target_state = None
                if not flow_controller.collected_data.get("leadName"):
                    target_state = ConversationState.NAME_COLLECTION
                elif not flow_controller.collected_data.get("leadEmail"):
                    target_state = ConversationState.EMAIL_COLLECTION
                elif (
                    not flow_controller.skip_phone_collection
                    and not flow_controller.collected_data.get("leadPhoneNumber")
                ):
                    target_state = ConversationState.PHONE_COLLECTION

                if target_state is not None:
                    flow_controller.transition_to(target_state)
                    enforced_prompt = await response_generator._generate_state_response(
                        flow_controller.state, "", conversation_history, context
                    )
                    conversation_history.append({"role": "assistant", "content": enforced_prompt})
                    await websocket.send_json({"type": "bot", "content": enforced_prompt})
                    continue

            # ── Calendar flow: show days/slots from connected calendar, allow booking ──
            if calendar_flow and app_id_for_calendar and str(user_text).strip():
                raw_lower = str(user_text).strip().lower()
                if raw_lower in ("cancel", "back", "exit"):
                    calendar_flow = None
                    calendar_days = []
                    calendar_slots = []
                    calendar_free_slots = []
                    calendar_selected_day = None
                    calendar_pending_slot = {}
                    reply = "Cancelled. How can I help?"
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
                # ── Confirm step: user must explicitly confirm selected slot ──
                if calendar_flow == "confirm" and calendar_pending_slot:
                    confirm_text = str(user_text).strip().lower()
                    if confirm_text not in {"confirm", "yes", "y", "book", "book now"}:
                        reply = (
                            "Please confirm your booking to continue.\n"
                            "<button value=\"confirm\">Confirm booking</button> "
                            "<button value=\"cancel\">Cancel</button>"
                        )
                        conversation_history.append({"role": "user", "content": user_text})
                        conversation_history.append({"role": "assistant", "content": reply})
                        await websocket.send_json({"type": "bot", "content": reply})
                        continue

                    service_title = str(flow_controller.collected_data.get("serviceType") or "").strip() or "Appointment"
                    slot = calendar_pending_slot
                    start_iso = slot.get("start", "")
                    end_iso = slot.get("end", "")
                    slot_timezone = slot.get("timezone")
                    calendar_service = CalendarService(settings)
                    customer_name = str(flow_controller.collected_data.get("leadName") or "").strip() or None
                    customer_email = str(flow_controller.collected_data.get("leadEmail") or "").strip() or None
                    customer_phone = str(flow_controller.collected_data.get("leadPhoneNumber") or "").strip() or None
                    _service_plans_ctx = context.get("service_plans", [])
                    _post_booking_note_raw = _find_post_booking_note(_service_plans_ctx, service_title)
                    _post_booking_note_chat = _format_post_booking_note_for_chat(_post_booking_note_raw)
                    book_result = await calendar_service.book_appointment(
                        app_id_for_calendar, start_iso, end_iso, service_title,
                        attendee_email=customer_email,
                        time_zone=slot_timezone,
                        customer_name=customer_name,
                        customer_phone=customer_phone,
                        lead_id=lead_id,
                        post_booking_note=_post_booking_note_raw,
                    )
                    calendar_flow = None
                    calendar_days = []
                    calendar_slots = []
                    calendar_free_slots = []
                    calendar_selected_day = None
                    calendar_pending_slot = {}
                    if book_result.get("success"):
                        _display_tz = integration.get("calendarTimezone") or slot_timezone or "UTC"
                        flow_controller.update_collected_data("appointmentSlot", {
                            "start": start_iso,
                            "end": end_iso,
                            "timezone": _display_tz,
                            "serviceType": service_title,
                        })
                        _bk_start = _format_slot_time_local(start_iso, _display_tz)
                        _bk_end = _format_slot_time_local(end_iso, _display_tz)
                        try:
                            from zoneinfo import ZoneInfo
                            _bk_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(ZoneInfo(_display_tz))
                            _bk_date = _bk_dt.strftime("%a %d/%m/%Y")
                        except Exception:
                            _bk_date = start_iso[:10]
                        time_str = f"{_bk_date}, {_bk_start} – {_bk_end}"
                        reply = f"✅ Your *{service_title}* is booked for {time_str}. You'll receive a confirmation shortly."
                        if _post_booking_note_chat:
                            reply += f"\n\n📋 **Important Instructions**\n{_post_booking_note_chat}"
                        _review_url_inline = _integration_google_review_url(integration)
                        if _review_url_inline:
                            reply += "\n\n" + get_string("review_prompt", lang_code, _review_url_inline)
                        # Booking confirmation is the terminal step for this lead cycle.
                        try:
                            if lead_id:
                                full_history = conversation_history + [
                                    {"role": "user", "content": user_text},
                                    {"role": "assistant", "content": reply},
                                ]
                                customer_name = flow_controller.collected_data.get("leadName") or "Customer"
                                lead_type_val = flow_controller.collected_data.get("leadType") or "inquiry"
                                appointment_details = {
                                    "eventId": book_result.get("eventId"),
                                    "start": book_result.get("start") or start_iso,
                                    "end": book_result.get("end") or end_iso,
                                    "link": book_result.get("link"),
                                    "confirmed": True,
                                }
                                completion_payload = {
                                    "status": "confirmed",
                                    "title": f"{service_title} – {customer_name}",
                                    "serviceType": flow_controller.collected_data.get("serviceType"),
                                    "leadType": flow_controller.collected_data.get("leadType"),
                                    "leadName": flow_controller.collected_data.get("leadName"),
                                    "leadEmail": flow_controller.collected_data.get("leadEmail"),
                                    "leadPhoneNumber": flow_controller.collected_data.get("leadPhoneNumber"),
                                    "appointmentDetails": appointment_details,
                                    "summary": f"{customer_name} booked a {service_title} appointment for {time_str}.",
                                    "description": f"{customer_name} ({lead_type_val}) booked {service_title}. Appointment confirmed at {time_str}.",
                                    "history": full_history,
                                }
                                await lead_service.update_lead(user_id, lead_id, completion_payload)
                        except Exception as completion_exc:
                            logger.warning("Lead completion update after booking failed: %s", completion_exc)
                    else:
                        reply = book_result.get("error") or "Booking failed. Please try again or contact us."
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    if book_result.get("success"):
                        await websocket.send_json({"type": "session_complete"})
                        widget_session_complete = True
                        if resume_key:
                            WIDGET_CHAT_RESUME.pop(resume_key, None)
                    continue
                try:
                    num = _extract_index_choice(str(user_text))
                    if num is None:
                        raise ValueError("No numeric choice parsed")
                    if calendar_flow == "slots" and 1 <= num <= len(calendar_slots):
                        # Save selected slot and ask user to confirm booking
                        slot = calendar_slots[num - 1]
                        calendar_pending_slot = slot
                        calendar_flow = "confirm"
                        _confirm_tz = integration.get("calendarTimezone") or "UTC"
                        t_start = _format_slot_time_local(slot.get("start", ""), _confirm_tz)
                        t_end = _format_slot_time_local(slot.get("end", ""), _confirm_tz)
                        try:
                            from zoneinfo import ZoneInfo
                            _dt_local = datetime.fromisoformat(slot.get("start", "").replace("Z", "+00:00")).astimezone(ZoneInfo(_confirm_tz))
                            _date_label = _dt_local.strftime("%a %d/%m/%Y")
                        except Exception:
                            _date_label = slot.get("start", "")[:10]
                        selected_service = str(flow_controller.collected_data.get("serviceType") or "").strip() or "Appointment"
                        reply = (
                            f"📋 Booking Summary\n"
                            f"──────────────────\n"
                            f"Service:  {selected_service}\n"
                            f"Date:     {_date_label}\n"
                            f"Time:     🕒 {t_start} – {t_end}\n\n"
                            "Please confirm your booking:\n"
                            "<button value=\"confirm\">Confirm booking</button> "
                            "<button value=\"cancel\">Cancel</button>"
                        )
                        conversation_history.append({"role": "user", "content": user_text})
                        conversation_history.append({"role": "assistant", "content": reply})
                        await websocket.send_json({"type": "bot", "content": reply})
                        continue
                    if calendar_flow == "days" and 1 <= num <= len(calendar_days):
                        day_info = calendar_days[num - 1]
                        selected_date = day_info.get("date", "")
                        _cal_tz_filter = integration.get("calendarTimezone") or "UTC"
                        _now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
                        def _slot_local_date(s: Dict[str, Any], tz: str) -> str:
                            try:
                                from zoneinfo import ZoneInfo
                                dt_u = datetime.fromisoformat((s.get("start") or "").replace("Z", "+00:00"))
                                return dt_u.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d")
                            except Exception:
                                return (s.get("start") or "")[:10]
                        slots_for_day = [
                            s for s in calendar_free_slots
                            if _slot_local_date(s, _cal_tz_filter) == selected_date
                            and datetime.fromisoformat((s.get("start") or "1970-01-01T00:00:00Z").replace("Z", "+00:00")).replace(tzinfo=timezone.utc) > _now_utc
                        ]
                        calendar_flow = "slots"
                        calendar_slots = slots_for_day
                        calendar_selected_day = selected_date
                        slot_tz = integration.get("calendarTimezone") or (slots_for_day[0].get("timezone") if slots_for_day else None) or "UTC"
                        tz_label = _get_tz_label(slot_tz)
                        lines = [f"Times on {day_info.get('label', selected_date)}:"]
                        lines.append(f"🌐 All times shown in {tz_label}")
                        lines.append("Choose a time slot:")
                        for i, s in enumerate(slots_for_day[:15], 1):
                            start = s.get("start", "")
                            end = s.get("end", "")
                            if start and end:
                                t_start = _format_slot_time_local(start, slot_tz)
                                t_end = _format_slot_time_local(end, slot_tz)
                                lines.append(f"<button value=\"{i}\">🕒 {i}. {t_start}–{t_end}</button>")
                        reply = "\n".join(lines)
                        conversation_history.append({"role": "user", "content": user_text})
                        conversation_history.append({"role": "assistant", "content": reply})
                        await websocket.send_json({"type": "bot", "content": reply})
                        continue
                except ValueError:
                    pass

            # Calendar availability shortcut should only run in booking-related states.
            # This prevents premature date/time prompts before personal info/service/workflow steps.
            if (
                app_id_for_calendar
                and flow_controller.state in (ConversationState.APPOINTMENT_OFFER, ConversationState.CALENDAR_BOOKING)
                and _is_availability_intent(str(user_text))
            ):
                from_date, to_date = _availability_window_from_tomorrow_utc(integration, horizon_days=14)
                slot_minutes = integration.get("calendarSlotMinutes", 30)
                if slot_minutes not in (15, 30, 60):
                    slot_minutes = 30
                try:
                    calendar_service = CalendarService(settings)
                    result = await calendar_service.get_availability(
                        app_id_for_calendar, from_date, to_date, slot_minutes=slot_minutes
                    )
                    if result.get("error"):
                        reply = "I couldn't fetch availability right now. Please try again later."
                    else:
                        cal_tz_fc = _calendar_tz_from_integration(integration)
                        _now_utc_sc = datetime.utcnow().replace(tzinfo=timezone.utc)
                        free_slots = [
                            s for s in (result.get("freeSlots") or [])
                            if datetime.fromisoformat((s.get("start") or "1970-01-01T00:00:00Z").replace("Z", "+00:00")).replace(tzinfo=timezone.utc) > _now_utc_sc
                        ]
                        free_slots = _filter_slots_from_tomorrow_local(free_slots, cal_tz_fc)
                        if not free_slots:
                            if not result.get("calendarConnected"):
                                reply = "Calendar isn't connected for this app, so I can't show availability. Please contact the business."
                            else:
                                reply = "I don't have any free slots in the next 7 days. You can ask for a different period or contact us directly."
                        else:
                            cal_tz = integration.get("calendarTimezone") or "UTC"
                            by_date: Dict[str, List[Dict[str, Any]]] = {}
                            for slot in free_slots:
                                start = slot.get("start") or ""
                                if not start:
                                    continue
                                try:
                                    from zoneinfo import ZoneInfo
                                    dt_utc = datetime.fromisoformat(start.replace("Z", "+00:00"))
                                    dt_local = dt_utc.astimezone(ZoneInfo(cal_tz))
                                    date_key = dt_local.strftime("%Y-%m-%d")
                                except Exception:
                                    date_key = start[:10]
                                by_date.setdefault(date_key, []).append(slot)
                            days_order = sorted(by_date.keys())
                            calendar_days_list = []
                            for d in days_order:
                                try:
                                    from zoneinfo import ZoneInfo
                                    dt = datetime.fromisoformat(d + "T12:00:00").replace(tzinfo=ZoneInfo(cal_tz))
                                    calendar_days_list.append({"date": d, "label": dt.strftime("%a %d/%m/%Y")})
                                except Exception:
                                    calendar_days_list.append({"date": d, "label": d})
                            calendar_flow = "days"
                            calendar_days = calendar_days_list
                            calendar_free_slots = free_slots
                            calendar_slots = []
                            calendar_selected_day = None
                            max_days = 14
                            show_days = calendar_days_list[:max_days]
                            service_title = str(flow_controller.collected_data.get("serviceType") or "").strip()
                            if service_title:
                                intro = f"Here are the available dates for your **{service_title}** appointment. Please choose a day:"
                            else:
                                intro = "Here are the available dates for your appointment. Please choose a day:"
                            lines = [intro]
                            for i, day in enumerate(show_days, 1):
                                lines.append(f"<button value=\"{i}\">📅 {i}. {day['label']}</button>")
                            reply = "\n".join(lines)
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
                except Exception as cal_exc:
                    logger.exception("WebSocket: Calendar availability error: %s", cal_exc)
                    reply = "I couldn't fetch availability right now. Please try again later."
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
            # ─────────────────────────────────────────────────────────────────

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
                        if ok:
                            reply = get_string("otp_resend", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed(
                                flow_controller, response_generator, conversation_history, context, lang_code, "email"
                            )
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
                            reply = get_string("perfect_otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed(
                                flow_controller, response_generator, conversation_history, context, lang_code, "email"
                            )
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
                    reply = await _continue_after_otp_delivery_failed(
                        flow_controller, response_generator, conversation_history, context, lang_code, "email"
                    )
                
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
                    reply = await _continue_after_otp_delivery_failed(
                        flow_controller, response_generator, conversation_history, context, lang_code, "phone"
                    )
                
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
                    reply = await _continue_after_otp_delivery_failed(
                        flow_controller, response_generator, conversation_history, context, lang_code, "email"
                    )
                
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
                    reply = await _continue_after_otp_delivery_failed(
                        flow_controller, response_generator, conversation_history, context, lang_code, "phone"
                    )
                
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
                        if ok:
                            reply = get_string("otp_resend", lang_code, phone)
                        else:
                            reply = await _continue_after_otp_delivery_failed(
                                flow_controller, response_generator, conversation_history, context, lang_code, "phone"
                            )
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
                        if ok:
                            reply = get_string("otp_resend", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed(
                                flow_controller, response_generator, conversation_history, context, lang_code, "email"
                            )
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
                            reply = get_string("perfect_otp_sent_phone", lang_code, phone)
                        else:
                            reply = await _continue_after_otp_delivery_failed(
                                flow_controller, response_generator, conversation_history, context, lang_code, "phone"
                            )
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
                            reply = get_string("perfect_otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed(
                                flow_controller, response_generator, conversation_history, context, lang_code, "email"
                            )
                    else:
                        reply = get_string("no_problem_email", lang_code)
                        flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                    
                    conversation_history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "bot", "content": reply})
                    continue
            
            # Check if JSON was generated (all data collected)
            if str(reply).strip() == "BOOK_APPOINTMENT_REQUESTED":
                from_date, to_date = _availability_window_from_tomorrow_utc(integration, horizon_days=14)
                slot_minutes = integration.get("calendarSlotMinutes", 30)
                if slot_minutes not in (15, 30, 60):
                    slot_minutes = 30
                try:
                    calendar_service = CalendarService(settings)
                    result = await calendar_service.get_availability(
                        app_id_for_calendar, from_date, to_date, slot_minutes=slot_minutes
                    )
                    if result.get("error"):
                        reply = "I couldn't fetch availability right now. Please try again later."
                    else:
                        cal_tz_fc = _calendar_tz_from_integration(integration)
                        _now_utc_bk = datetime.utcnow().replace(tzinfo=timezone.utc)
                        free_slots = [
                            s for s in (result.get("freeSlots") or [])
                            if datetime.fromisoformat((s.get("start") or "1970-01-01T00:00:00Z").replace("Z", "+00:00")).replace(tzinfo=timezone.utc) > _now_utc_bk
                        ]
                        free_slots = _filter_slots_from_tomorrow_local(free_slots, cal_tz_fc)
                        if not free_slots:
                            reply = "I don't have any free slots in the next 7 days. You can ask for a different period or contact us directly."
                        else:
                            cal_tz = integration.get("calendarTimezone") or "UTC"
                            by_date: Dict[str, List[Dict[str, Any]]] = {}
                            for slot in free_slots:
                                start = slot.get("start") or ""
                                if not start:
                                    continue
                                try:
                                    from zoneinfo import ZoneInfo
                                    dt_utc = datetime.fromisoformat(start.replace("Z", "+00:00"))
                                    dt_local = dt_utc.astimezone(ZoneInfo(cal_tz))
                                    date_key = dt_local.strftime("%Y-%m-%d")
                                except Exception:
                                    date_key = start[:10]
                                by_date.setdefault(date_key, []).append(slot)
                            days_order = sorted(by_date.keys())
                            calendar_days_list = []
                            for d in days_order:
                                try:
                                    from zoneinfo import ZoneInfo
                                    dt = datetime.fromisoformat(d + "T12:00:00").replace(tzinfo=ZoneInfo(cal_tz))
                                    calendar_days_list.append({"date": d, "label": dt.strftime("%a %d/%m/%Y")})
                                except Exception:
                                    calendar_days_list.append({"date": d, "label": d})
                            calendar_flow = "days"
                            calendar_days = calendar_days_list
                            calendar_free_slots = free_slots
                            calendar_slots = []
                            calendar_selected_day = None
                            service_title = str(flow_controller.collected_data.get("serviceType") or "").strip()
                            if service_title:
                                intro = f"Great! Here are the available dates for your **{service_title}** appointment. Please choose a day:"
                            else:
                                intro = "Great! Here are the available dates for your appointment. Please choose a day:"
                            lines = [intro]
                            for i, day in enumerate(calendar_days_list[:14], 1):
                                lines.append(f"<button value=\"{i}\">📅 {i}. {day['label']}</button>")
                            reply = "\n".join(lines)
                except Exception as cal_exc:
                    logger.exception("WebSocket: Calendar availability error: %s", cal_exc)
                    reply = "I couldn't fetch availability right now. Please try again later."

            parsed_json = _maybe_parse_json(reply)
            if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
                # Add appId if available (for app-scoped leads)
                if app_id:
                    parsed_json["appId"] = app_id
                # Update existing interaction lead if available, otherwise create a new lead
                try:
                    if lead_id:
                        parsed_json["status"] = "complete"
                        ok, _ = await lead_service.update_lead(user_id, lead_id, parsed_json)
                    else:
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
                conversation_history.append({"role": "assistant", "content": final_msg})
                await websocket.send_json({"type": "session_complete"})
                widget_session_complete = True
                if resume_key:
                    WIDGET_CHAT_RESUME.pop(resume_key, None)
                continue

            # Hard safety guard: never expose internal JSON/payloads in chat bubbles.
            if _looks_like_internal_lead_payload(reply):
                logger.error("Blocked internal payload from being sent to user chat (user_id=%s)", user_id)
                graceful_msg = get_string("final_fallback", lang_code)
                conversation_history.append({"role": "assistant", "content": graceful_msg})
                await websocket.send_json({"type": "bot", "content": graceful_msg})
                await websocket.send_json({"type": "session_complete"})
                widget_session_complete = True
                if resume_key:
                    WIDGET_CHAT_RESUME.pop(resume_key, None)
                continue
            
            # Regular conversation response — send the bot reply before PATCHing the lead so
            # the client is not left idle during slow HTTP to the CRM API (avoids 1006 / user bounce).
            conversation_history.append({"role": "assistant", "content": reply})
            try:
                await websocket.send_json({"type": "bot", "content": reply})
                if _bot_requests_file_upload(reply):
                    await websocket.send_json({"type": "enable_file_upload"})
            except WebSocketDisconnect:
                raise
            except RuntimeError as send_exc:
                if "send" in str(send_exc).lower() and "close" in str(send_exc).lower():
                    logger.info(
                        "Client socket closed before bot reply could be sent (user_id=%s)",
                        user_id,
                    )
                    raise WebSocketDisconnect from send_exc
                raise
            # Progressive lead sync after the user-visible message
            if lead_id:
                snapshot = {
                    "leadType": flow_controller.collected_data.get("leadType"),
                    "serviceType": flow_controller.collected_data.get("serviceType"),
                    "leadName": flow_controller.collected_data.get("leadName"),
                    "leadEmail": flow_controller.collected_data.get("leadEmail"),
                    "leadPhoneNumber": flow_controller.collected_data.get("leadPhoneNumber"),
                    "leadTypeSwitchHistory": flow_controller.collected_data.get("leadTypeSwitchHistory"),
                    "status": "in_progress"
                }
                if snapshot != last_snapshot:
                    try:
                        await lead_service.update_lead(user_id, lead_id, snapshot)
                        last_snapshot = dict(snapshot)
                    except Exception as sync_exc:
                        logger.warning("Lead progressive update failed: %s", sync_exc)
            
            # State transitions are handled in response_generator
            # Only update if we're still in the same state (no transition happened)
            # The response generator handles state transitions internally
            
            # Keep history manageable
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for user_id=%s", user_id)
    except Exception as loop_exc:
        logger.exception("Unhandled error in WebSocket loop for user_id=%s: %s", user_id, loop_exc)
        try:
            await websocket.send_json({"type": "error", "content": "An unexpected error occurred. Please try again."})
        except Exception:
            pass
    finally:
        if (
            resume_key
            and app_id
            and fetch_by_app
            and not widget_session_complete
            and flow_controller.state != ConversationState.COMPLETE
        ):
            try:
                WIDGET_CHAT_RESUME[resume_key] = {
                    "saved_at": time.time(),
                    "app_id": app_id,
                    "terminal": False,
                    "conversation_history": conversation_history[-50:],
                    "calendar_flow": calendar_flow,
                    "calendar_days": calendar_days,
                    "calendar_slots": calendar_slots,
                    "calendar_free_slots": calendar_free_slots,
                    "calendar_selected_day": calendar_selected_day,
                    "calendar_pending_slot": calendar_pending_slot,
                    "user_timezone": user_timezone,
                    "flow": _snapshot_flow_controller_for_resume(flow_controller),
                    "lead_id": lead_id,
                }
                _prune_widget_resume_store()
            except Exception:
                pass


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
            flow_controller.update_collected_data("sourceChannel", "whatsapp")
            
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
            flow_controller.update_collected_data("sourceChannel", "whatsapp")
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
            
            # Determine selection context from stateful collected data (industry-agnostic).
            # This is more reliable than keyword checks in prior user messages.
            has_lead_type = bool(flow_controller.collected_data.get("leadType"))
            has_service = bool(flow_controller.collected_data.get("serviceType"))

            # Backward-safe fallback for older sessions where leadType marker was embedded in history.
            if not has_lead_type:
                has_lead_type = any(
                    "leadType:" in msg.get("content", "")
                    for msg in conversation_history[-3:]
                    if msg.get("role") == "user"
                )

            logger.info(
                f"WhatsApp: Selection context - state={flow_controller.state.value}, "
                f"has_lead_type={has_lead_type}, has_service={has_service}"
            )
            
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
                service_plans = context.get("service_plans", [])
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
        
        integration = context.get("integration") or {}
        calendar_flow = session.get("calendar_flow")
        calendar_slots = session.get("calendar_slots") or []
        calendar_days = session.get("calendar_days") or []

        # Calendar flow: user already in day/slot selection; handle numeric reply or cancel
        if calendar_flow and app_id and user_text.strip():
            raw_lower = user_text.strip().lower()
            if raw_lower in ("cancel", "back", "exit"):
                session["calendar_flow"] = None
                session["calendar_days"] = None
                session["calendar_slots"] = None
                session["calendar_free_slots"] = None
                session["calendar_selected_day"] = None
                session["calendar_pending_slot"] = None
                reply = "Cancelled. How can I help?"
                conversation_history.append({"role": "user", "content": user_text})
                conversation_history.append({"role": "assistant", "content": reply})
                whatsapp_sessions[session_id]["history"] = conversation_history
                await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            # ── Confirm step: user replied with service/reason — book with that title ──
            if calendar_flow == "confirm":
                pending_slot = session.get("calendar_pending_slot") or {}
                service_title = user_text.strip() or "Appointment"
                start_iso = pending_slot.get("start", "")
                end_iso = pending_slot.get("end", "")
                slot_timezone = pending_slot.get("timezone")
                calendar_service = CalendarService(settings)
                book_result = await calendar_service.book_appointment(app_id, start_iso, end_iso, service_title, time_zone=slot_timezone)
                session["calendar_flow"] = None
                session["calendar_days"] = None
                session["calendar_slots"] = None
                session["calendar_free_slots"] = None
                session["calendar_selected_day"] = None
                session["calendar_pending_slot"] = None
                if book_result.get("success"):
                    _wa_tz = slot_timezone or "UTC"
                    _wa_start = _format_slot_time_local(start_iso, _wa_tz)
                    _wa_end = _format_slot_time_local(end_iso, _wa_tz)
                    try:
                        from zoneinfo import ZoneInfo
                        _wa_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(ZoneInfo(_wa_tz))
                        _wa_date = _wa_dt.strftime("%a %d/%m/%Y")
                    except Exception:
                        _wa_date = start_iso[:10]
                    time_str = f"{_wa_date}, {_wa_start} – {_wa_end}"
                    reply = f"✅ Your *{service_title}* is booked for {time_str}. You'll receive a confirmation shortly."
                else:
                    reply = book_result.get("error") or "Booking failed. Please try again or contact us."
                conversation_history.append({"role": "user", "content": user_text})
                conversation_history.append({"role": "assistant", "content": reply})
                whatsapp_sessions[session_id]["history"] = conversation_history
                await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            try:
                num = _extract_index_choice(user_text)
                if num is None:
                    raise ValueError("No numeric choice parsed")
                if calendar_flow == "slots" and 1 <= num <= len(calendar_slots):
                    slot = calendar_slots[num - 1]
                    # Save slot and ask for service/reason before booking
                    session["calendar_pending_slot"] = slot
                    session["calendar_flow"] = "confirm"
                    _wa_tz = integration.get("calendarTimezone") or "UTC"
                    _wa_t_start = _format_slot_time_local(slot.get("start", ""), _wa_tz)
                    _wa_t_end = _format_slot_time_local(slot.get("end", ""), _wa_tz)
                    try:
                        from zoneinfo import ZoneInfo
                        _wa_dt = datetime.fromisoformat(slot.get("start", "").replace("Z", "+00:00")).astimezone(ZoneInfo(_wa_tz))
                        _wa_date = _wa_dt.strftime("%a %d/%m/%Y")
                    except Exception:
                        _wa_date = slot.get("start", "")[:10]
                    reply = f"What service or treatment is this appointment for?\n(e.g. Enzyme Facial, Consultation, Follow-up)\n\nSelected time: 🕒 {_wa_date}, {_wa_t_start} – {_wa_t_end}"
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    whatsapp_sessions[session_id]["history"] = conversation_history
                    await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                    return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
                if calendar_flow == "days" and 1 <= num <= len(calendar_days):
                    day_info = calendar_days[num - 1]
                    selected_date = day_info.get("date", "")
                    free_slots_all = session.get("calendar_free_slots") or []
                    slots_for_day = [s for s in free_slots_all if (s.get("start") or "")[:10] == selected_date]
                    session["calendar_flow"] = "slots"
                    session["calendar_slots"] = slots_for_day
                    session["calendar_selected_day"] = selected_date
                    slot_tz = (slots_for_day[0].get("timezone") if slots_for_day else None) or "UTC"
                    tz_label = _get_tz_label(slot_tz)
                    lines = [f"Times on {day_info.get('label', selected_date)}:"]
                    lines.append(f"🌐 All times in {tz_label}")
                    lines.append("Choose a time slot:")
                    for i, s in enumerate(slots_for_day[:15], 1):
                        start = s.get("start", "")
                        end = s.get("end", "")
                        if start and end:
                            t_start = _format_slot_time_local(start, slot_tz)
                            t_end = _format_slot_time_local(end, slot_tz)
                            lines.append(f"<button value=\"{i}\">🕒 {i}. {t_start}–{t_end}</button>")
                    reply = "\n".join(lines)
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    whatsapp_sessions[session_id]["history"] = conversation_history
                    await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                    return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            except ValueError:
                pass

        # Calendar availability: user asks "when are you free?" / "book appointment" -> fetch and show days
        if (
            app_id
            and flow_controller.state in (ConversationState.APPOINTMENT_OFFER, ConversationState.CALENDAR_BOOKING)
            and _is_availability_intent(enhanced_user_text)
        ):
            from_date, to_date = _availability_window_from_tomorrow_utc(integration, horizon_days=14)
            slot_minutes = integration.get("calendarSlotMinutes", 30)
            if slot_minutes not in (15, 30, 60):
                slot_minutes = 30
            try:
                calendar_service = CalendarService(settings)
                result = await calendar_service.get_availability(app_id, from_date, to_date, slot_minutes=slot_minutes)
                if result.get("error"):
                    reply = "I couldn't fetch availability right now. Please try again later."
                else:
                    cal_tz_wa = _calendar_tz_from_integration(integration)
                    _now_utc_wa = datetime.utcnow().replace(tzinfo=timezone.utc)
                    free_slots = [
                        s for s in (result.get("freeSlots") or [])
                        if datetime.fromisoformat((s.get("start") or "1970-01-01T00:00:00Z").replace("Z", "+00:00")).replace(tzinfo=timezone.utc) > _now_utc_wa
                    ]
                    free_slots = _filter_slots_from_tomorrow_local(free_slots, cal_tz_wa)
                    if not free_slots:
                        if not result.get("calendarConnected"):
                            reply = "Calendar isn't connected for this app, so I can't show availability. Please contact the business."
                        else:
                            reply = "I don't have any free slots in the next 7 days. You can ask for a different period or contact us directly."
                    else:
                        by_date: Dict[str, List[Dict[str, Any]]] = {}
                        for slot in free_slots:
                            start = slot.get("start") or ""
                            if not start:
                                continue
                            try:
                                from zoneinfo import ZoneInfo

                                dt_utc = datetime.fromisoformat(start.replace("Z", "+00:00"))
                                date_key = dt_utc.astimezone(ZoneInfo(cal_tz_wa)).strftime("%Y-%m-%d")
                            except Exception:
                                date_key = start[:10]
                            by_date.setdefault(date_key, []).append(slot)
                        days_order = sorted(by_date.keys())
                        calendar_days_list = []
                        for d in days_order:
                            try:
                                from zoneinfo import ZoneInfo

                                dt = datetime.fromisoformat(d + "T12:00:00").replace(tzinfo=ZoneInfo(cal_tz_wa))
                                calendar_days_list.append({"date": d, "label": dt.strftime("%a %d/%m/%Y")})
                            except Exception:
                                calendar_days_list.append({"date": d, "label": d})
                        session["calendar_flow"] = "days"
                        session["calendar_days"] = calendar_days_list
                        session["calendar_free_slots"] = free_slots
                        session["calendar_slots"] = None
                        session["calendar_selected_day"] = None
                        max_days = 14
                        show_days = calendar_days_list[:max_days]
                        service_title = str(flow_controller.collected_data.get("serviceType") or "").strip()
                        if service_title:
                            intro = f"Here are the available dates for your **{service_title}** appointment. Please choose a day:"
                        else:
                            intro = "Here are the available dates for your appointment. Please choose a day:"
                        lines = [intro]
                        for i, day in enumerate(show_days, 1):
                            lines.append(f"<button value=\"{i}\">📅 {i}. {day['label']}</button>")
                        reply = "\n".join(lines)
                conversation_history.append({"role": "user", "content": enhanced_user_text})
                conversation_history.append({"role": "assistant", "content": reply})
                whatsapp_sessions[session_id]["history"] = conversation_history
                await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
            except Exception as cal_exc:
                logger.exception("WhatsApp: Calendar availability error: %s", cal_exc)
                reply = "I couldn't fetch availability right now. Please try again later."
                conversation_history.append({"role": "user", "content": enhanced_user_text})
                conversation_history.append({"role": "assistant", "content": reply})
                whatsapp_sessions[session_id]["history"] = conversation_history
                await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
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
                    
                    if ok:
                        reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email)
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            session.get("response_language_code", "en"),
                            "email",
                            email_validation_state=email_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                    logger.info(f"WhatsApp: Email OTP sent, returning early to prevent JSON generation")
                    return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
                else:
                    logger.info(f"WhatsApp: No valid email found in input; checking for stored email to retry")
                    # No new email extracted – if a previous email was stored (OTP send
                    # failed earlier), retry with the stored email instead of letting
                    # the AI generate a misleading "order complete" message.
                    stored_email = email_validation_state.get("email")
                    if stored_email:
                        logger.info(f"WhatsApp: Retrying OTP send with stored email={stored_email}")
                        customer_name = email_validation_state.get(
                            "customer_name",
                            flow_controller.collected_data.get("leadName", "Customer"),
                        )
                        ok, _ = await email_validation_service.send_otp_email(user_id, stored_email, customer_name)
                        if ok:
                            email_validation_state["otp_sent"] = True
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("otp_sent_email", session.get("response_language_code", "en"), stored_email)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                session.get("response_language_code", "en"),
                                "email",
                                email_validation_state=email_validation_state,
                            )
                        conversation_history.append({"role": "assistant", "content": reply})
                        await whatsapp_service.send_message(user_phone, reply, from_phone=twilio_phone)
                        logger.info(f"WhatsApp: Stored email OTP retry handled, returning early")
                        return Response(content=whatsapp_service.create_twiml_response(""), media_type="text/xml")
        
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
                if ok:
                    reply = get_string("otp_resend", session.get("response_language_code", "en"), email_validation_state["email"])
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        session.get("response_language_code", "en"),
                        "email",
                        email_validation_state=email_validation_state,
                    )
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
                    if ok:
                        reply = get_string("perfect_otp_sent_email", session.get("response_language_code", "en"), email)
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            session.get("response_language_code", "en"),
                            "email",
                            email_validation_state=email_validation_state,
                        )
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
            
            if ok:
                reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email)
            else:
                reply = await _continue_after_otp_delivery_failed_with_session(
                    flow_controller,
                    response_generator,
                    conversation_history,
                    context,
                    session.get("response_language_code", "en"),
                    "email",
                    email_validation_state=email_validation_state,
                )
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
            
            if ok:
                reply = get_string("otp_sent_phone", session.get("response_language_code", "en"), phone)
            else:
                reply = await _continue_after_otp_delivery_failed_with_session(
                    flow_controller,
                    response_generator,
                    conversation_history,
                    context,
                    session.get("response_language_code", "en"),
                    "phone",
                    phone_validation_state=phone_validation_state,
                )
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
            
            if ok:
                reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email)
            else:
                reply = await _continue_after_otp_delivery_failed_with_session(
                    flow_controller,
                    response_generator,
                    conversation_history,
                    context,
                    session.get("response_language_code", "en"),
                    "email",
                    email_validation_state=email_validation_state,
                )
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
            
            if ok:
                reply = get_string("otp_sent_phone", session.get("response_language_code", "en"), phone)
            else:
                reply = await _continue_after_otp_delivery_failed_with_session(
                    flow_controller,
                    response_generator,
                    conversation_history,
                    context,
                    session.get("response_language_code", "en"),
                    "phone",
                    phone_validation_state=phone_validation_state,
                )
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
                if ok:
                    reply = get_string("otp_resend", session.get("response_language_code", "en"), phone_validation_state["phone"])
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        session.get("response_language_code", "en"),
                        "phone",
                        phone_validation_state=phone_validation_state,
                    )
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
                if ok:
                    reply = get_string("otp_resend", session.get("response_language_code", "en"), email_validation_state["email"])
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        session.get("response_language_code", "en"),
                        "email",
                        email_validation_state=email_validation_state,
                    )
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
                    if ok:
                        reply = get_string("perfect_otp_sent_phone", session.get("response_language_code", "en"), phone)
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            session.get("response_language_code", "en"),
                            "phone",
                            phone_validation_state=phone_validation_state,
                        )
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
                    if ok:
                        reply = get_string("perfect_otp_sent_email", session.get("response_language_code", "en"), email)
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            session.get("response_language_code", "en"),
                            "email",
                            email_validation_state=email_validation_state,
                        )
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
                if ok:
                    reply = get_string("otp_sent_email", session.get("response_language_code", "en"), email)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        session.get("response_language_code", "en"),
                        "email",
                        email_validation_state=email_validation_state,
                    )
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
                if ok:
                    reply = get_string("otp_sent_phone", session.get("response_language_code", "en"), phone)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        session.get("response_language_code", "en"),
                        "phone",
                        phone_validation_state=phone_validation_state,
                    )
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
        if buttons and len(buttons) <= 3:
            # Try native interactive buttons first; fall back to numbered list on failure
            wa_buttons = [{"id": f"btn_{i}", "title": btn["title"][:20]} for i, btn in enumerate(buttons, 1)]
            success, message = await whatsapp_service.send_interactive_buttons(user_phone, cleaned_reply, wa_buttons, from_phone=twilio_phone)
            if not success:
                logger.info("Interactive buttons failed (%s), falling back to numbered list", message)
                button_text = "\n\n" + "\n".join([f"{i}. {btn['title']}" for i, btn in enumerate(buttons, 1)])
                full_message = cleaned_reply + button_text + "\n\nPlease reply with the number of your choice."
                success, message = await whatsapp_service.send_message(user_phone, full_message, from_phone=twilio_phone)
                if not success:
                    logger.error("Failed to send WhatsApp message: %s", message)
        elif buttons:
            # 4+ options — use numbered list
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


# ─────────────────────────────────────────────────────────────────────────────
# Facebook Messenger – Meta Graph API (mirrors Instagram implementation)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/webhook/messenger")
async def messenger_webhook_verification(request: Request):
    """
    Meta webhook verification handshake (GET).
    Meta sends hub.mode=subscribe, hub.verify_token, hub.challenge.
    We must echo back hub.challenge as plain text with 200 OK.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    verify_token = settings.meta_verify_token or "assistly_verify_token"

    if mode == "subscribe" and token == verify_token:
        logger.info("Messenger webhook verified successfully")
        return Response(content=challenge, media_type="text/plain")

    logger.warning(
        "Messenger webhook verification failed: mode=%s, token_match=%s",
        mode,
        token == verify_token,
    )
    return Response(content="Forbidden", status_code=403)


@app.post("/webhook/messenger")
async def messenger_webhook(request: Request):
    """
    Handle incoming Facebook Messenger messages via Meta Graph API.
    Same conversation flow as WhatsApp and Instagram – no Twilio involved.
    """
    try:
        raw_body = await request.body()

        # ── Optional: verify Meta signature ──────────────────────────────────
        app_secret = getattr(settings, "meta_app_secret", None)
        if app_secret:
            sig_header = request.headers.get("x-hub-signature-256", "")
            if not MessengerGraphService.verify_signature(raw_body, sig_header, app_secret):
                logger.warning("Messenger webhook: signature verification failed")
                return Response(content="Forbidden", status_code=403)

        body = json.loads(raw_body)
        logger.info("Messenger webhook received: %s", json.dumps(body)[:200])

        # ── Parse event ────────────────────────────────────────────────────────
        parsed = MessengerGraphService.parse_webhook_event(body)
        if not parsed:
            # Non-message events (delivery, read, echo…) – always 200 OK
            return {"status": "ok"}

        sender_id = parsed["sender_id"]       # PSID  (the user)
        recipient_id = parsed["recipient_id"]  # Facebook Page ID
        message_text = parsed["message_text"]

        if not sender_id or not recipient_id or not message_text:
            logger.warning("Messenger webhook: missing required fields after parsing")
            return {"status": "ok"}

        # ── Services ────────────────────────────────────────────────────────────
        context_service = ContextService(settings)
        lead_service = LeadService(settings)
        rag_service = RAGService(settings)
        email_validation_service = EmailValidationService(settings)
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        phone_validation_service = PhoneValidationService(
            settings, openai_client=openai_client, gpt_model=settings.gpt_model
        )
        messenger_service = MessengerGraphService()

        # ── Session ──────────────────────────────────────────────────────────────
        # recipient_id = Page ID  →  used as the "account" key for context lookup
        # sender_id    = PSID     →  used as the "user" key within that page
        session_id, is_new = get_or_create_messenger_session(recipient_id, sender_id)

        # ── NEW SESSION ──────────────────────────────────────────────────────────
        if is_new:
            try:
                context = await context_service.fetch_context_by_social_sender(recipient_id)
            except Exception as exc:
                logger.exception(
                    "Messenger: failed to fetch context for page_id=%s: %s", recipient_id, exc
                )
                return {"status": "error", "message": "Failed to fetch context"}

            user_data = context.get("user", {})
            app_data = context.get("app", {})
            app_id = app_data.get("id") if app_data else None
            owner_id = str(user_data.get("id") or "")
            page_access_token = context.get("messengerAccessToken")

            if not owner_id:
                logger.error(
                    "Messenger: no owner_id in context for page_id=%s", recipient_id
                )
                return {"status": "error", "message": "No owner found"}

            if not page_access_token:
                logger.error(
                    "Messenger: no page access token in context for page_id=%s", recipient_id
                )
                return {"status": "error", "message": "No Messenger page access token configured"}

            current_time = time.time()
            messenger_sessions[session_id] = {
                "user_id": sender_id,          # PSID
                "recipient_id": recipient_id,  # Facebook Page ID
                "history": [],
                "email_state": {
                    "email": None,
                    "otp_sent": False,
                    "otp_verified": False,
                    "customer_name": None,
                },
                "phone_state": {"phone": None, "otp_sent": False, "otp_verified": True},
                "context": context,
                "user_id_owner": owner_id,
                "app_id": app_id,
                "created_at": current_time,
                "last_activity": current_time,
                "flow_controller": None,
                "response_generator": None,
                "page_access_token": page_access_token,
            }

            flow_controller = FlowController(context)
            flow_controller.set_whatsapp(True)
            flow_controller.update_collected_data("leadPhoneNumber", "")
            flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)

            response_generator = ResponseGenerator(settings, rag_service)
            response_generator.set_profession(str(context.get("profession") or "Business"))
            response_generator.set_channel("messenger")
            flow_controller.update_collected_data("sourceChannel", "facebook")
            rag_service.build_vector_store(context)

            messenger_sessions[session_id]["flow_controller"] = flow_controller
            messenger_sessions[session_id]["response_generator"] = response_generator

            first_message = message_text or "(started)"
            initial_reply = await response_generator.generate_greeting(
                context, channel="messenger", first_message=first_message or None
            )
            messenger_sessions[session_id]["history"] = [
                {"role": "user", "content": first_message},
                {"role": "assistant", "content": initial_reply},
            ]
            messenger_key_to_session[f"messenger_{recipient_id}_{sender_id}"] = session_id

            cleaned_reply, buttons = _extract_buttons_from_response(initial_reply)
            if buttons:
                btn_text = "\n\n" + "\n".join(
                    [f"{i}. {btn['title']}" for i, btn in enumerate(buttons, 1)]
                )
                await messenger_service.send_message(
                    sender_id,
                    cleaned_reply + btn_text + "\n\nPlease reply with the number of your choice.",
                    page_access_token,
                )
            else:
                await messenger_service.send_message(sender_id, initial_reply, page_access_token)

            return {"status": "ok"}

        # ── EXISTING SESSION ──────────────────────────────────────────────────────
        session = messenger_sessions.get(session_id)
        if not session:
            session_id_new, _ = get_or_create_messenger_session(recipient_id, sender_id)
            session = messenger_sessions.get(session_id_new)
            if not session:
                logger.error("Messenger: session expired for PSID=%s", sender_id)
                return {"status": "ok"}
            session_id = session_id_new

        now = time.time()
        last_activity = session.get("last_activity", 0)
        app_id = session.get("app_id")
        applied_version = session.get("conversation_style_applied_version")
        pending = conversation_style_change_requests.get(app_id) if app_id else None
        if pending and applied_version != pending.get("version"):
            requested_at = float(pending.get("requested_at") or 0)
            idle_seconds = int(pending.get("idle_seconds") or 120)
            idle_start = max(last_activity, requested_at)
            if now - idle_start >= idle_seconds:
                session_context = session.get("context") or {}
                integration = session_context.get("integration") or {}
                if isinstance(integration, dict):
                    integration["conversationStyle"] = bool(pending.get("conversation_style"))
                    session["conversation_style_applied_version"] = pending.get("version")

        session["last_activity"] = now
        conversation_history = session["history"]
        context = session["context"]
        flow_controller = session.get("flow_controller")
        response_generator = session.get("response_generator")
        owner_id = session["user_id_owner"]
        app_id = session.get("app_id")
        page_access_token = session.get("page_access_token")
        email_validation_state = session["email_state"]
        phone_validation_state = session["phone_state"]

        if not flow_controller or not response_generator:
            logger.error("Messenger: flow_controller or response_generator missing in session")
            return {"status": "error"}
        flow_controller.update_collected_data("sourceChannel", "facebook")

        integration = context.get("integration") or {}
        calendar_flow = session.get("calendar_flow")
        calendar_slots = session.get("calendar_slots") or []
        calendar_days = session.get("calendar_days") or []

        # Calendar flow: same as WhatsApp – day/slot selection or cancel
        if calendar_flow and app_id and message_text.strip():
            raw_lower = message_text.strip().lower()
            if raw_lower in ("cancel", "back", "exit"):
                session["calendar_flow"] = None
                session["calendar_days"] = None
                session["calendar_slots"] = None
                session["calendar_free_slots"] = None
                session["calendar_selected_day"] = None
                reply = "Cancelled. How can I help?"
                conversation_history.append({"role": "user", "content": message_text})
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, reply, page_access_token)
                return {"status": "ok"}
            try:
                num = _extract_index_choice(message_text)
                if num is None:
                    raise ValueError("No numeric choice parsed")
                if calendar_flow == "slots" and 1 <= num <= len(calendar_slots):
                    slot = calendar_slots[num - 1]
                    start_iso = slot.get("start", "")
                    end_iso = slot.get("end", "")
                    slot_timezone = slot.get("timezone")
                    title = "Appointment"
                    calendar_service = CalendarService(settings)
                    book_result = await calendar_service.book_appointment(app_id, start_iso, end_iso, title, time_zone=slot_timezone)
                    session["calendar_flow"] = None
                    session["calendar_days"] = None
                    session["calendar_slots"] = None
                    session["calendar_free_slots"] = None
                    session["calendar_selected_day"] = None
                    if book_result.get("success"):
                        try:
                            dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                            time_str = dt.strftime("%H:%M")
                            reply = f"Booked for {time_str}. You'll receive a confirmation. Reply with anything to continue."
                        except Exception:
                            reply = "Appointment booked. Reply with anything to continue."
                    else:
                        reply = book_result.get("error") or "Booking failed. Please try again or contact us."
                    conversation_history.append({"role": "user", "content": message_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}
                if calendar_flow == "days" and 1 <= num <= len(calendar_days):
                    day_info = calendar_days[num - 1]
                    selected_date = day_info.get("date", "")
                    free_slots_all = session.get("calendar_free_slots") or []
                    slots_for_day = [s for s in free_slots_all if (s.get("start") or "")[:10] == selected_date]
                    session["calendar_flow"] = "slots"
                    session["calendar_slots"] = slots_for_day
                    session["calendar_selected_day"] = selected_date
                    lines = [f"Times on {day_info.get('label', selected_date)}:"]
                    lines.append("Choose a time slot:")
                    for i, s in enumerate(slots_for_day[:15], 1):
                        start = s.get("start", "")
                        end = s.get("end", "")
                        if start and end:
                            try:
                                dt_s = datetime.fromisoformat(start.replace("Z", "+00:00"))
                                dt_e = datetime.fromisoformat(end.replace("Z", "+00:00"))
                                lines.append(f"<button value=\"{i}\">🕒 {i}. {dt_s.strftime('%H:%M')}–{dt_e.strftime('%H:%M')}</button>")
                            except Exception:
                                lines.append(f"<button value=\"{i}\">🕒 {i}. {start}–{end}</button>")
                    reply = "\n".join(lines)
                    conversation_history.append({"role": "user", "content": message_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}
            except ValueError:
                pass

        # Calendar availability: user asks "when are you free?" / "book appointment" -> fetch and show days
        if (
            app_id
            and flow_controller.state in (ConversationState.APPOINTMENT_OFFER, ConversationState.CALENDAR_BOOKING)
            and _is_availability_intent(message_text)
        ):
            from_date, to_date = _availability_window_from_tomorrow_utc(integration, horizon_days=14)
            slot_minutes = integration.get("calendarSlotMinutes", 30)
            if slot_minutes not in (15, 30, 60):
                slot_minutes = 30
            try:
                calendar_service = CalendarService(settings)
                result = await calendar_service.get_availability(app_id, from_date, to_date, slot_minutes=slot_minutes)
                if result.get("error"):
                    reply = "I couldn't fetch availability right now. Please try again later."
                else:
                    cal_tz_ms = _calendar_tz_from_integration(integration)
                    _now_utc_ms = datetime.utcnow().replace(tzinfo=timezone.utc)
                    free_slots = [
                        s for s in (result.get("freeSlots") or [])
                        if datetime.fromisoformat((s.get("start") or "1970-01-01T00:00:00Z").replace("Z", "+00:00")).replace(tzinfo=timezone.utc) > _now_utc_ms
                    ]
                    free_slots = _filter_slots_from_tomorrow_local(free_slots, cal_tz_ms)
                    if not free_slots:
                        if not result.get("calendarConnected"):
                            reply = "Calendar isn't connected for this app, so I can't show availability. Please contact the business."
                        else:
                            reply = "I don't have any free slots in the next 7 days. You can ask for a different period or contact us directly."
                    else:
                        by_date: Dict[str, List[Dict[str, Any]]] = {}
                        for slot in free_slots:
                            start = slot.get("start") or ""
                            if not start:
                                continue
                            try:
                                from zoneinfo import ZoneInfo

                                dt_utc = datetime.fromisoformat(start.replace("Z", "+00:00"))
                                date_key = dt_utc.astimezone(ZoneInfo(cal_tz_ms)).strftime("%Y-%m-%d")
                            except Exception:
                                date_key = start[:10]
                            by_date.setdefault(date_key, []).append(slot)
                        days_order = sorted(by_date.keys())
                        calendar_days_list = []
                        for d in days_order:
                            try:
                                from zoneinfo import ZoneInfo

                                dt = datetime.fromisoformat(d + "T12:00:00").replace(tzinfo=ZoneInfo(cal_tz_ms))
                                calendar_days_list.append({"date": d, "label": dt.strftime("%a %d/%m/%Y")})
                            except Exception:
                                calendar_days_list.append({"date": d, "label": d})
                        session["calendar_flow"] = "days"
                        session["calendar_days"] = calendar_days_list
                        session["calendar_free_slots"] = free_slots
                        session["calendar_slots"] = None
                        session["calendar_selected_day"] = None
                        max_days = 14
                        show_days = calendar_days_list[:max_days]
                        service_title = str(flow_controller.collected_data.get("serviceType") or "").strip()
                        if service_title:
                            intro = f"Here are the available dates for your **{service_title}** appointment. Please choose a day:"
                        else:
                            intro = "Here are the available dates for your appointment. Please choose a day:"
                        lines = [intro]
                        for i, day in enumerate(show_days, 1):
                            lines.append(f"<button value=\"{i}\">📅 {i}. {day['label']}</button>")
                        reply = "\n".join(lines)
                conversation_history.append({"role": "user", "content": message_text})
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, reply, page_access_token)
                return {"status": "ok"}
            except Exception as cal_exc:
                logger.exception("Messenger: Calendar availability error: %s", cal_exc)
                reply = "I couldn't fetch availability right now. Please try again later."
                conversation_history.append({"role": "user", "content": message_text})
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, reply, page_access_token)
                return {"status": "ok"}

        # Add user message to history
        conversation_history.append({"role": "user", "content": message_text})

        # Detect language and configure response generator
        lang_code = detect_language(message_text)
        session["response_language_code"] = lang_code
        response_generator.set_response_language(get_language_name_for_prompt(lang_code))

        try:
            # ── PRE-CHECK: detect email when last bot message asked for it ────────
            validate_email = context.get("integration", {}).get("validateEmail", True)
            logger.info(
                "Messenger: email validation check – validate_email=%s otp_sent=%s history_len=%d",
                validate_email, email_validation_state["otp_sent"], len(conversation_history),
            )

            if validate_email and not email_validation_state["otp_sent"] and len(conversation_history) > 1:
                last_bot = next(
                    (m for m in reversed(conversation_history) if m["role"] == "assistant"), None
                )
                if last_bot and "email" in last_bot["content"].lower():
                    logger.info("Messenger: detected email collection phase, input=%s", message_text)
                    extractor = DataExtractor()
                    email = extractor.extract_email(message_text)
                    if email and _is_valid_email(email):
                        customer_name = flow_controller.collected_data.get("leadName", "Customer")
                        if customer_name == "Customer":
                            for msg in reversed(conversation_history):
                                if msg.get("role") == "user":
                                    name = extractor.extract_name(
                                        msg.get("content", ""), context.get("lead_types", [])
                                    )
                                    if name:
                                        customer_name = name
                                        break
                        email_validation_state["email"] = email
                        email_validation_state["customer_name"] = customer_name
                        flow_controller.collected_data["leadEmail"] = email
                        ok, _ = await email_validation_service.send_otp_email(
                            owner_id, email, customer_name
                        )
                        if ok:
                            email_validation_state["otp_sent"] = True
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "email",
                                email_validation_state=email_validation_state,
                            )
                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await messenger_service.send_message(sender_id, reply, page_access_token)
                        return {"status": "ok"}
                    else:
                        # No new email extracted – if a previous email was stored (OTP send
                        # failed earlier), retry with the stored email instead of letting
                        # the AI generate a misleading "order complete" message.
                        stored_email = email_validation_state.get("email")
                        if stored_email:
                            logger.info(
                                "Messenger: no new email in input; retrying stored email=%s",
                                stored_email,
                            )
                            customer_name = email_validation_state.get(
                                "customer_name",
                                flow_controller.collected_data.get("leadName", "Customer"),
                            )
                            ok, _ = await email_validation_service.send_otp_email(
                                owner_id, stored_email, customer_name
                            )
                            if ok:
                                email_validation_state["otp_sent"] = True
                                flow_controller.otp_state["email_sent"] = True
                                flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                            if ok:
                                reply = get_string("otp_sent_email", lang_code, stored_email)
                            else:
                                reply = await _continue_after_otp_delivery_failed_with_session(
                                    flow_controller,
                                    response_generator,
                                    conversation_history,
                                    context,
                                    lang_code,
                                    "email",
                                    email_validation_state=email_validation_state,
                                )
                            conversation_history.append({"role": "assistant", "content": reply})
                            session["history"] = conversation_history
                            await messenger_service.send_message(sender_id, reply, page_access_token)
                            return {"status": "ok"}

            # ── OTP VERIFICATION LOOP ─────────────────────────────────────────────
            if email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                logger.info("Messenger: in OTP verification mode, input=%s", message_text)
                temp_reply = await response_generator.generate_response(
                    flow_controller, message_text, conversation_history, context
                )
                retry_type, extracted_value = _detect_retry_request(temp_reply)

                if retry_type == "resend_otp" and email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                    customer_name = email_validation_state.get(
                        "customer_name", flow_controller.collected_data.get("leadName", "Customer")
                    )
                    ok, _ = await email_validation_service.send_otp_email(
                        owner_id, email_validation_state["email"], customer_name
                    )
                    if ok:
                        reply = get_string("otp_resend", lang_code, email_validation_state["email"])
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "email",
                            email_validation_state=email_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}

                elif retry_type == "change_email":
                    email_validation_state.update({"otp_sent": False, "otp_verified": False, "email": None})
                    flow_controller.otp_state.update({"email_sent": False, "email_verified": False})
                    if extracted_value:
                        email = extracted_value
                        email_validation_state["email"] = email
                        email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                        flow_controller.collected_data["leadEmail"] = email
                        ok, _ = await email_validation_service.send_otp_email(
                            owner_id, email, email_validation_state["customer_name"]
                        )
                        if ok:
                            email_validation_state["otp_sent"] = True
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("perfect_otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "email",
                                email_validation_state=email_validation_state,
                            )
                    else:
                        reply = get_string("no_problem_email", lang_code)
                        flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}

                else:
                    otp_code = _extract_otp_from_text(message_text)
                    if otp_code:
                        ok, _ = await email_validation_service.verify_otp(
                            owner_id, email_validation_state["email"], otp_code
                        )
                        if ok:
                            email_validation_state["otp_verified"] = True
                            flow_controller.otp_state["email_verified"] = True
                            flow_controller.transition_to(flow_controller.get_next_state())
                            reply = await response_generator.generate_response(
                                flow_controller, "Email verified", conversation_history, context
                            )
                        else:
                            reply = temp_reply

                        # Check if the post-OTP reply is a lead JSON
                        parsed_json = _maybe_parse_json(reply)
                        if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
                            email_valid = _validate_email_verification(email_validation_state) if validate_email else True
                            if email_valid:
                                if app_id:
                                    parsed_json["appId"] = app_id
                                try:
                                    ok_lead, _ = await lead_service.create_public_lead(owner_id, parsed_json)
                                    final_msg = (
                                        get_string("final_success", lang_code)
                                        if ok_lead
                                        else get_string("final_fallback", lang_code)
                                    )
                                except Exception:
                                    final_msg = get_string("final_fallback", lang_code)
                                integration = context.get("integration") or {}
                                if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                                    await messenger_service.send_message(
                                        sender_id,
                                        get_string("review_prompt", lang_code, integration["googleReviewUrl"].strip()),
                                        page_access_token,
                                    )
                                await messenger_service.send_message(sender_id, final_msg, page_access_token)
                                key = f"messenger_{recipient_id}_{sender_id}"
                                messenger_key_to_session.pop(key, None)
                                messenger_sessions.pop(session_id, None)
                                return {"status": "ok"}

                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await messenger_service.send_message(sender_id, reply, page_access_token)
                        return {"status": "ok"}
                    else:
                        # Not an OTP – use the AI's natural response
                        reply = temp_reply
                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await messenger_service.send_message(sender_id, reply, page_access_token)
                        return {"status": "ok"}

            # ── Generate response via state machine ───────────────────────────────
            reply = await response_generator.generate_response(
                flow_controller, message_text, conversation_history, context
            )

            # ── SEND_EMAIL / SEND_PHONE signal interceptors ───────────────────────
            if "|||SEND_EMAIL:" in reply:
                parts = reply.split("|||SEND_EMAIL:", 1)
                answer, email = parts[0].strip(), parts[1].strip()
                conversation_history.append({"role": "assistant", "content": answer})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, answer, page_access_token)
                flow_controller.collected_data["leadEmail"] = email
                flow_controller.otp_state["email_sent"] = True
                flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
                customer_name = flow_controller.collected_data.get("leadName", "Customer")
                ok, _ = await email_validation_service.send_otp_email(owner_id, email, customer_name)
                if ok:
                    flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    email_validation_state.update({"otp_sent": True, "email": email, "customer_name": customer_name})
                if ok:
                    reply = get_string("otp_sent_email", lang_code, email)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "email",
                        email_validation_state=email_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, reply, page_access_token)
                return {"status": "ok"}

            elif reply.startswith("SEND_EMAIL:"):
                email = reply.split(":", 1)[1].strip()
                flow_controller.collected_data["leadEmail"] = email
                flow_controller.otp_state["email_sent"] = True
                flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
                customer_name = flow_controller.collected_data.get("leadName", "Customer")
                logger.info("Messenger: sending OTP email to %s", email)
                ok, _ = await email_validation_service.send_otp_email(owner_id, email, customer_name)
                if ok:
                    flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    email_validation_state.update({"otp_sent": True, "email": email, "customer_name": customer_name})
                if ok:
                    reply = get_string("otp_sent_email", lang_code, email)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "email",
                        email_validation_state=email_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, reply, page_access_token)
                return {"status": "ok"}

            elif "|||SEND_PHONE:" in reply:
                parts = reply.split("|||SEND_PHONE:", 1)
                answer, phone = parts[0].strip(), parts[1].strip()
                from app.utils.phone_utils import format_phone_number_with_gpt
                phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
                conversation_history.append({"role": "assistant", "content": answer})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, answer, page_access_token)
                flow_controller.collected_data["leadPhoneNumber"] = phone
                flow_controller.otp_state["phone_sent"] = True
                flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
                ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone)
                if ok:
                    flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                    phone_validation_state.update({"otp_sent": True, "phone": phone})
                if ok:
                    reply = get_string("otp_sent_phone", lang_code, phone)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "phone",
                        phone_validation_state=phone_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, reply, page_access_token)
                return {"status": "ok"}

            elif reply.startswith("SEND_PHONE:"):
                phone = reply.split(":", 1)[1].strip()
                from app.utils.phone_utils import format_phone_number_with_gpt
                phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
                flow_controller.collected_data["leadPhoneNumber"] = phone
                flow_controller.otp_state["phone_sent"] = True
                flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
                ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone)
                if ok:
                    flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                    phone_validation_state.update({"otp_sent": True, "phone": phone})
                if ok:
                    reply = get_string("otp_sent_phone", lang_code, phone)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "phone",
                        phone_validation_state=phone_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await messenger_service.send_message(sender_id, reply, page_access_token)
                return {"status": "ok"}

            # ── Lead JSON generation ───────────────────────────────────────────────
            parsed_json = _maybe_parse_json(reply)
            if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
                # Verify email OTP is done before finalising lead
                email_valid = _validate_email_verification(email_validation_state) if validate_email else True
                if email_valid:
                    if app_id:
                        parsed_json["appId"] = app_id
                    try:
                        ok, _ = await lead_service.create_public_lead(owner_id, parsed_json)
                        final_msg = (
                            get_string("final_success", lang_code)
                            if ok
                            else get_string("final_fallback", lang_code)
                        )
                    except Exception:
                        final_msg = get_string("final_fallback", lang_code)

                    integration = context.get("integration") or {}
                    if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                        review_url = integration["googleReviewUrl"].strip()
                        await messenger_service.send_message(
                            sender_id,
                            get_string("review_prompt", lang_code, review_url),
                            page_access_token,
                        )
                    await messenger_service.send_message(sender_id, final_msg, page_access_token)
                    key = f"messenger_{recipient_id}_{sender_id}"
                    messenger_key_to_session.pop(key, None)
                    messenger_sessions.pop(session_id, None)
                    return {"status": "ok"}

            # ── Retry detection ───────────────────────────────────────────────────
            retry_type, extracted_value = _detect_retry_request(reply)
            if retry_type:
                if retry_type == "resend_otp" and email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                    customer_name = email_validation_state.get(
                        "customer_name", flow_controller.collected_data.get("leadName", "Customer")
                    )
                    ok, _ = await email_validation_service.send_otp_email(
                        owner_id, email_validation_state["email"], customer_name
                    )
                    if ok:
                        reply = get_string("otp_resend", lang_code, email_validation_state["email"])
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "email",
                            email_validation_state=email_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}

                elif retry_type == "resend_otp" and phone_validation_state["otp_sent"] and not phone_validation_state["otp_verified"]:
                    ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone_validation_state["phone"])
                    if ok:
                        reply = get_string("otp_resend", lang_code, phone_validation_state["phone"])
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "phone",
                            phone_validation_state=phone_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}

                elif retry_type == "change_email":
                    email_validation_state.update({"otp_sent": False, "otp_verified": False, "email": None})
                    flow_controller.otp_state.update({"email_sent": False, "email_verified": False})
                    if extracted_value:
                        email = extracted_value
                        email_validation_state["email"] = email
                        email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                        flow_controller.collected_data["leadEmail"] = email
                        ok, _ = await email_validation_service.send_otp_email(
                            owner_id, email, email_validation_state["customer_name"]
                        )
                        if ok:
                            email_validation_state["otp_sent"] = True
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("perfect_otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "email",
                                email_validation_state=email_validation_state,
                            )
                    else:
                        reply = get_string("no_problem_email", lang_code)
                        flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}

                elif retry_type == "change_phone":
                    phone_validation_state.update({"otp_sent": False, "otp_verified": False, "phone": None})
                    flow_controller.otp_state.update({"phone_sent": False, "phone_verified": False})
                    if extracted_value:
                        from app.utils.phone_utils import format_phone_number_with_gpt
                        phone = await format_phone_number_with_gpt(extracted_value, openai_client, settings.gpt_model)
                        phone_validation_state["phone"] = phone
                        flow_controller.collected_data["leadPhoneNumber"] = phone
                        ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone)
                        if ok:
                            phone_validation_state["otp_sent"] = True
                            flow_controller.otp_state["phone_sent"] = True
                            flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("perfect_otp_sent_phone", lang_code, phone)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "phone",
                                phone_validation_state=phone_validation_state,
                            )
                    else:
                        reply = get_string("no_problem_phone", lang_code)
                        flow_controller.transition_to(ConversationState.PHONE_COLLECTION)
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}

                elif retry_type == "send_email" and extracted_value:
                    if not validate_email:
                        # Email validation disabled – store email and let AI continue naturally
                        email = extracted_value
                        email_validation_state["email"] = email
                        flow_controller.collected_data["leadEmail"] = email
                        conversation_history.append({"role": "user", "content": "Email provided"})
                        reply = await response_generator.generate_response(
                            flow_controller, "Email provided", conversation_history, context
                        )
                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await messenger_service.send_message(sender_id, reply, page_access_token)
                        return {"status": "ok"}
                    # Email validation enabled
                    email = extracted_value
                    email_validation_state["email"] = email
                    email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                    if email_validation_state["customer_name"] == "Customer":
                        extractor = DataExtractor()
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "user":
                                name = extractor.extract_name(
                                    msg.get("content", ""), context.get("lead_types", [])
                                )
                                if name:
                                    email_validation_state["customer_name"] = name
                                    flow_controller.collected_data["leadName"] = name
                                    break
                    flow_controller.collected_data["leadEmail"] = email
                    ok, _ = await email_validation_service.send_otp_email(
                        owner_id, email, email_validation_state["customer_name"]
                    )
                    if ok:
                        email_validation_state["otp_sent"] = True
                        flow_controller.otp_state["email_sent"] = True
                        flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    if ok:
                        reply = get_string("otp_sent_email", lang_code, email)
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "email",
                            email_validation_state=email_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await messenger_service.send_message(sender_id, reply, page_access_token)
                    return {"status": "ok"}

            # ── Regular reply ─────────────────────────────────────────────────────
            conversation_history.append({"role": "assistant", "content": reply})
            session["history"] = conversation_history

            cleaned_reply, buttons = _extract_buttons_from_response(reply)
            if buttons:
                qrs = [{"title": btn["title"], "payload": btn["title"]} for btn in buttons]
                await messenger_service.send_quick_replies(sender_id, cleaned_reply, qrs, page_access_token)
            else:
                await messenger_service.send_message(sender_id, reply, page_access_token)

        except Exception as exc:
            logger.exception("Messenger: error processing message: %s", exc)
            try:
                await messenger_service.send_message(
                    sender_id,
                    "Sorry, I encountered an error. Please try again.",
                    page_access_token,
                )
            except Exception:
                pass

        return {"status": "ok"}

    except Exception as exc:
        logger.exception("Messenger webhook error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.get("/webhook/instagram")
async def instagram_webhook_verification(request: Request):
    """
    Meta webhook verification handshake (GET) for Instagram.
    Same as Messenger: hub.mode=subscribe, hub.verify_token, hub.challenge.
    Use meta_verify_token (shared with Messenger) for consistency.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    verify_token = settings.meta_verify_token

    if mode == "subscribe" and token == verify_token:
        logger.info("Instagram webhook verified successfully")
        return Response(content=challenge, media_type="text/plain")

    logger.warning(
        "Instagram webhook verification failed: mode=%s, token_match=%s",
        mode,
        token == verify_token,
    )
    return Response(content="Forbidden", status_code=403)


@app.post("/webhook/instagram")
async def instagram_webhook(request: Request):
    """
    Handle incoming Instagram messages via Meta Graph API.
    Same conversation flow as Messenger – no Twilio involved.
    """
    try:
        raw_body = await request.body()

        # ── Optional: verify Meta signature (same as Messenger) ─────────────────
        app_secret = getattr(settings, "meta_app_secret", None)
        if app_secret:
            sig_header = request.headers.get("x-hub-signature-256", "")
            if not InstagramGraphService.verify_signature(raw_body, sig_header, app_secret):
                logger.warning("Instagram webhook: signature verification failed")
                return Response(content="Forbidden", status_code=403)

        body = json.loads(raw_body)
        logger.info("Instagram webhook received: %s", json.dumps(body)[:200])

        # ── Parse event ────────────────────────────────────────────────────────
        parsed = InstagramGraphService.parse_webhook_event(body)
        if not parsed:
            # Non-message events (read, delivery, echo…) – always 200 OK
            return {"status": "ok"}

        sender_id = parsed["sender_id"]       # IGSID (Instagram-scoped sender ID)
        recipient_id = parsed["recipient_id"]  # Instagram Business Account ID
        message_text = parsed["message_text"]

        if not sender_id or not recipient_id or not message_text:
            logger.warning("Instagram webhook: missing required fields after parsing")
            return {"status": "ok"}
        
        # Initialize services
        context_service = ContextService(settings)
        lead_service = LeadService(settings)
        rag_service = RAGService(settings)
        email_validation_service = EmailValidationService(settings)
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        phone_validation_service = PhoneValidationService(settings, openai_client=openai_client, gpt_model=settings.gpt_model)
        instagram_service = InstagramGraphService()
        
        # Get or create session for this Instagram user
        session_id, is_new = get_or_create_instagram_session(recipient_id, sender_id)
        
        if is_new:
            # Fetch app context by Instagram Business Account ID
            try:
                context = await context_service.fetch_context_by_social_sender(recipient_id)
            except Exception as exc:
                logger.exception(f"Instagram: failed to fetch context for business_account_id={recipient_id}: {exc}")
                # Get access token from context if available, otherwise can't send error message
                return {"status": "error", "message": "Failed to fetch context"}
            
            user_data = context.get("user", {})
            app_data = context.get("app", {})
            app_id = app_data.get("id") if app_data else None
            owner_id = str(user_data.get("id") or "")
            # Page access token works for both Messenger and Instagram when Page has Instagram linked
            instagram_access_token = context.get("instagramAccessToken") or context.get("messengerAccessToken")
            
            if not owner_id:
                logger.error(f"Instagram: no owner_id in context for business_account_id={recipient_id}")
                return {"status": "error", "message": "No owner found"}
            
            if not instagram_access_token:
                logger.error(f"Instagram: no access token in context for business_account_id={recipient_id}")
                return {"status": "error", "message": "No Instagram access token configured"}
            
            current_time = time.time()
            instagram_sessions[session_id] = {
                "user_id": sender_id,  # IGSID
                "recipient_id": recipient_id,  # IG Business Account ID
                "history": [],
                "email_state": {"email": None, "otp_sent": False, "otp_verified": False, "customer_name": None},
                "phone_state": {"phone": None, "otp_sent": False, "otp_verified": True},
                "context": context,
                "user_id_owner": owner_id,
                "app_id": app_id,
                "created_at": current_time,
                "last_activity": current_time,
                "flow_controller": None,
                "response_generator": None,
                "instagram_access_token": instagram_access_token,
            }
            
            # Initialize flow controller and response generator
            flow_controller = FlowController(context)
            flow_controller.set_whatsapp(True)
            flow_controller.update_collected_data("leadPhoneNumber", "")
            flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)
            
            response_generator = ResponseGenerator(settings, rag_service)
            response_generator.set_profession(str(context.get("profession") or "Business"))
            response_generator.set_channel("instagram")
            flow_controller.update_collected_data("sourceChannel", "instagram")
            rag_service.build_vector_store(context)
            
            instagram_sessions[session_id]["flow_controller"] = flow_controller
            instagram_sessions[session_id]["response_generator"] = response_generator
            
            first_message = message_text or "(started)"
            initial_reply = await response_generator.generate_greeting(context, channel="instagram", first_message=first_message or None)
            instagram_sessions[session_id]["history"] = [
                {"role": "user", "content": first_message},
                {"role": "assistant", "content": initial_reply}
            ]
            
            instagram_key_to_session[f"instagram_{recipient_id}_{sender_id}"] = session_id
            
            # Send reply via Meta Graph API
            cleaned_reply, buttons = _extract_buttons_from_response(initial_reply)
            if buttons:
                button_text = "\n\n" + "\n".join([f"{i}. {btn['title']}" for i, btn in enumerate(buttons, 1)])
                full_message = cleaned_reply + button_text + "\n\nPlease reply with the number of your choice."
                await instagram_service.send_message(sender_id, full_message, instagram_access_token)
            else:
                await instagram_service.send_message(sender_id, initial_reply, instagram_access_token)
            
            return {"status": "ok"}
        
        # ── EXISTING SESSION ──────────────────────────────────────────────────────
        session = instagram_sessions.get(session_id)
        if not session:
            session_id_new, _ = get_or_create_instagram_session(recipient_id, sender_id)
            session = instagram_sessions.get(session_id_new)
            if not session:
                logger.error("Instagram: session expired for IGSID=%s", sender_id)
                return {"status": "ok"}
            session_id = session_id_new

        now = time.time()
        last_activity = session.get("last_activity", 0)
        app_id = session.get("app_id")
        applied_version = session.get("conversation_style_applied_version")
        pending = conversation_style_change_requests.get(app_id) if app_id else None
        if pending and applied_version != pending.get("version"):
            requested_at = float(pending.get("requested_at") or 0)
            idle_seconds = int(pending.get("idle_seconds") or 120)
            idle_start = max(last_activity, requested_at)
            if now - idle_start >= idle_seconds:
                session_context = session.get("context") or {}
                integration = session_context.get("integration") or {}
                if isinstance(integration, dict):
                    integration["conversationStyle"] = bool(pending.get("conversation_style"))
                    session["conversation_style_applied_version"] = pending.get("version")

        session["last_activity"] = now
        conversation_history = session["history"]
        context = session["context"]
        email_validation_state = session["email_state"]
        phone_validation_state = session["phone_state"]
        flow_controller = session.get("flow_controller")
        response_generator = session.get("response_generator")
        owner_id = session["user_id_owner"]
        app_id = session.get("app_id")
        instagram_access_token = session.get("instagram_access_token")

        if not flow_controller or not response_generator:
            logger.error("Instagram: flow_controller or response_generator not initialized")
            return {"status": "error"}
        flow_controller.update_collected_data("sourceChannel", "instagram")

        integration = context.get("integration") or {}
        calendar_flow = session.get("calendar_flow")
        calendar_slots = session.get("calendar_slots") or []
        calendar_days = session.get("calendar_days") or []

        # Calendar flow: same as WhatsApp/Messenger – day/slot selection or cancel
        if calendar_flow and app_id and message_text.strip():
            raw_lower = message_text.strip().lower()
            if raw_lower in ("cancel", "back", "exit"):
                session["calendar_flow"] = None
                session["calendar_days"] = None
                session["calendar_slots"] = None
                session["calendar_free_slots"] = None
                session["calendar_selected_day"] = None
                reply = "Cancelled. How can I help?"
                conversation_history.append({"role": "user", "content": message_text})
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, reply, instagram_access_token)
                return {"status": "ok"}
            try:
                num = _extract_index_choice(message_text)
                if num is None:
                    raise ValueError("No numeric choice parsed")
                if calendar_flow == "slots" and 1 <= num <= len(calendar_slots):
                    slot = calendar_slots[num - 1]
                    start_iso = slot.get("start", "")
                    end_iso = slot.get("end", "")
                    slot_timezone = slot.get("timezone")
                    title = "Appointment"
                    calendar_service = CalendarService(settings)
                    book_result = await calendar_service.book_appointment(app_id, start_iso, end_iso, title, time_zone=slot_timezone)
                    session["calendar_flow"] = None
                    session["calendar_days"] = None
                    session["calendar_slots"] = None
                    session["calendar_free_slots"] = None
                    session["calendar_selected_day"] = None
                    if book_result.get("success"):
                        try:
                            dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                            time_str = dt.strftime("%H:%M")
                            reply = f"Booked for {time_str}. You'll receive a confirmation. Reply with anything to continue."
                        except Exception:
                            reply = "Appointment booked. Reply with anything to continue."
                    else:
                        reply = book_result.get("error") or "Booking failed. Please try again or contact us."
                    conversation_history.append({"role": "user", "content": message_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}
                if calendar_flow == "days" and 1 <= num <= len(calendar_days):
                    day_info = calendar_days[num - 1]
                    selected_date = day_info.get("date", "")
                    free_slots_all = session.get("calendar_free_slots") or []
                    slots_for_day = [s for s in free_slots_all if (s.get("start") or "")[:10] == selected_date]
                    session["calendar_flow"] = "slots"
                    session["calendar_slots"] = slots_for_day
                    session["calendar_selected_day"] = selected_date
                    lines = [f"Times on {day_info.get('label', selected_date)}:"]
                    lines.append("Choose a time slot:")
                    for i, s in enumerate(slots_for_day[:15], 1):
                        start = s.get("start", "")
                        end = s.get("end", "")
                        if start and end:
                            try:
                                dt_s = datetime.fromisoformat(start.replace("Z", "+00:00"))
                                dt_e = datetime.fromisoformat(end.replace("Z", "+00:00"))
                                lines.append(f"<button value=\"{i}\">🕒 {i}. {dt_s.strftime('%H:%M')}–{dt_e.strftime('%H:%M')}</button>")
                            except Exception:
                                lines.append(f"<button value=\"{i}\">🕒 {i}. {start}–{end}</button>")
                    reply = "\n".join(lines)
                    conversation_history.append({"role": "user", "content": message_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}
            except ValueError:
                pass

        # Calendar availability: user asks "when are you free?" / "book appointment" -> fetch and show days
        if (
            app_id
            and flow_controller.state in (ConversationState.APPOINTMENT_OFFER, ConversationState.CALENDAR_BOOKING)
            and _is_availability_intent(message_text)
        ):
            from_date, to_date = _availability_window_from_tomorrow_utc(integration, horizon_days=14)
            slot_minutes = integration.get("calendarSlotMinutes", 30)
            if slot_minutes not in (15, 30, 60):
                slot_minutes = 30
            try:
                calendar_service = CalendarService(settings)
                result = await calendar_service.get_availability(app_id, from_date, to_date, slot_minutes=slot_minutes)
                if result.get("error"):
                    reply = "I couldn't fetch availability right now. Please try again later."
                else:
                    cal_tz_ig = _calendar_tz_from_integration(integration)
                    _now_utc_ig = datetime.utcnow().replace(tzinfo=timezone.utc)
                    free_slots = [
                        s for s in (result.get("freeSlots") or [])
                        if datetime.fromisoformat((s.get("start") or "1970-01-01T00:00:00Z").replace("Z", "+00:00")).replace(tzinfo=timezone.utc) > _now_utc_ig
                    ]
                    free_slots = _filter_slots_from_tomorrow_local(free_slots, cal_tz_ig)
                    if not free_slots:
                        if not result.get("calendarConnected"):
                            reply = "Calendar isn't connected for this app, so I can't show availability. Please contact the business."
                        else:
                            reply = "I don't have any free slots in the next 7 days. You can ask for a different period or contact us directly."
                    else:
                        by_date: Dict[str, List[Dict[str, Any]]] = {}
                        for slot in free_slots:
                            start = slot.get("start") or ""
                            if not start:
                                continue
                            try:
                                from zoneinfo import ZoneInfo

                                dt_utc = datetime.fromisoformat(start.replace("Z", "+00:00"))
                                date_key = dt_utc.astimezone(ZoneInfo(cal_tz_ig)).strftime("%Y-%m-%d")
                            except Exception:
                                date_key = start[:10]
                            by_date.setdefault(date_key, []).append(slot)
                        days_order = sorted(by_date.keys())
                        calendar_days_list = []
                        for d in days_order:
                            try:
                                from zoneinfo import ZoneInfo

                                dt = datetime.fromisoformat(d + "T12:00:00").replace(tzinfo=ZoneInfo(cal_tz_ig))
                                calendar_days_list.append({"date": d, "label": dt.strftime("%a %d/%m/%Y")})
                            except Exception:
                                calendar_days_list.append({"date": d, "label": d})
                        session["calendar_flow"] = "days"
                        session["calendar_days"] = calendar_days_list
                        session["calendar_free_slots"] = free_slots
                        session["calendar_slots"] = None
                        session["calendar_selected_day"] = None
                        max_days = 14
                        show_days = calendar_days_list[:max_days]
                        service_title = str(flow_controller.collected_data.get("serviceType") or "").strip()
                        if service_title:
                            intro = f"Here are the available dates for your **{service_title}** appointment. Please choose a day:"
                        else:
                            intro = "Here are the available dates for your appointment. Please choose a day:"
                        lines = [intro]
                        for i, day in enumerate(show_days, 1):
                            lines.append(f"<button value=\"{i}\">📅 {i}. {day['label']}</button>")
                        reply = "\n".join(lines)
                conversation_history.append({"role": "user", "content": message_text})
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, reply, instagram_access_token)
                return {"status": "ok"}
            except Exception as cal_exc:
                logger.exception("Instagram: Calendar availability error: %s", cal_exc)
                reply = "I couldn't fetch availability right now. Please try again later."
                conversation_history.append({"role": "user", "content": message_text})
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, reply, instagram_access_token)
                return {"status": "ok"}
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": message_text})

        # Detect language and configure response generator
        lang_code = detect_language(message_text)
        session["response_language_code"] = lang_code
        response_generator.set_response_language(get_language_name_for_prompt(lang_code))

        try:
            # ── PRE-CHECK: detect email when last bot message asked for it ────────
            validate_email = context.get("integration", {}).get("validateEmail", True)
            logger.info(
                "Instagram: email validation check – validate_email=%s otp_sent=%s history_len=%d",
                validate_email, email_validation_state["otp_sent"], len(conversation_history),
            )

            if validate_email and not email_validation_state["otp_sent"] and len(conversation_history) > 1:
                last_bot = next(
                    (m for m in reversed(conversation_history) if m["role"] == "assistant"), None
                )
                if last_bot and "email" in last_bot["content"].lower():
                    logger.info("Instagram: detected email collection phase, input=%s", message_text)
                    extractor = DataExtractor()
                    email = extractor.extract_email(message_text)
                    if email and _is_valid_email(email):
                        customer_name = flow_controller.collected_data.get("leadName", "Customer")
                        if customer_name == "Customer":
                            for msg in reversed(conversation_history):
                                if msg.get("role") == "user":
                                    name = extractor.extract_name(
                                        msg.get("content", ""), context.get("lead_types", [])
                                    )
                                    if name:
                                        customer_name = name
                                        break
                        email_validation_state["email"] = email
                        email_validation_state["customer_name"] = customer_name
                        flow_controller.collected_data["leadEmail"] = email
                        ok, _ = await email_validation_service.send_otp_email(
                            owner_id, email, customer_name
                        )
                        if ok:
                            email_validation_state["otp_sent"] = True
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "email",
                                email_validation_state=email_validation_state,
                            )
                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await instagram_service.send_message(sender_id, reply, instagram_access_token)
                        return {"status": "ok"}
                    else:
                        stored_email = email_validation_state.get("email")
                        if stored_email:
                            logger.info("Instagram: no new email in input; retrying stored email=%s", stored_email)
                            customer_name = email_validation_state.get(
                                "customer_name",
                                flow_controller.collected_data.get("leadName", "Customer"),
                            )
                            ok, _ = await email_validation_service.send_otp_email(
                                owner_id, stored_email, customer_name
                            )
                            if ok:
                                email_validation_state["otp_sent"] = True
                                flow_controller.otp_state["email_sent"] = True
                                flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                            if ok:
                                reply = get_string("otp_sent_email", lang_code, stored_email)
                            else:
                                reply = await _continue_after_otp_delivery_failed_with_session(
                                    flow_controller,
                                    response_generator,
                                    conversation_history,
                                    context,
                                    lang_code,
                                    "email",
                                    email_validation_state=email_validation_state,
                                )
                            conversation_history.append({"role": "assistant", "content": reply})
                            session["history"] = conversation_history
                            await instagram_service.send_message(sender_id, reply, instagram_access_token)
                            return {"status": "ok"}

            # ── OTP VERIFICATION LOOP ─────────────────────────────────────────────
            if email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                logger.info("Instagram: in OTP verification mode, input=%s", message_text)
                temp_reply = await response_generator.generate_response(
                    flow_controller, message_text, conversation_history, context
                )
                retry_type, extracted_value = _detect_retry_request(temp_reply)

                if retry_type == "resend_otp" and email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                    customer_name = email_validation_state.get(
                        "customer_name", flow_controller.collected_data.get("leadName", "Customer")
                    )
                    ok, _ = await email_validation_service.send_otp_email(
                        owner_id, email_validation_state["email"], customer_name
                    )
                    if ok:
                        reply = get_string("otp_resend", lang_code, email_validation_state["email"])
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "email",
                            email_validation_state=email_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}

                elif retry_type == "change_email":
                    email_validation_state.update({"otp_sent": False, "otp_verified": False, "email": None})
                    flow_controller.otp_state.update({"email_sent": False, "email_verified": False})
                    if extracted_value:
                        email = extracted_value
                        email_validation_state["email"] = email
                        email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                        flow_controller.collected_data["leadEmail"] = email
                        ok, _ = await email_validation_service.send_otp_email(
                            owner_id, email, email_validation_state["customer_name"]
                        )
                        if ok:
                            email_validation_state["otp_sent"] = True
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("perfect_otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "email",
                                email_validation_state=email_validation_state,
                            )
                    else:
                        reply = get_string("no_problem_email", lang_code)
                        flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}

                else:
                    otp_code = _extract_otp_from_text(message_text)
                    if otp_code:
                        ok, _ = await email_validation_service.verify_otp(
                            owner_id, email_validation_state["email"], otp_code
                        )
                        if ok:
                            email_validation_state["otp_verified"] = True
                            flow_controller.otp_state["email_verified"] = True
                            flow_controller.transition_to(flow_controller.get_next_state())
                            reply = await response_generator.generate_response(
                                flow_controller, "Email verified", conversation_history, context
                            )
                        else:
                            reply = temp_reply

                        parsed_json = _maybe_parse_json(reply)
                        if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
                            email_valid = _validate_email_verification(email_validation_state) if validate_email else True
                            if email_valid:
                                if app_id:
                                    parsed_json["appId"] = app_id
                                try:
                                    ok_lead, _ = await lead_service.create_public_lead(owner_id, parsed_json)
                                    final_msg = (
                                        get_string("final_success", lang_code)
                                        if ok_lead
                                        else get_string("final_fallback", lang_code)
                                    )
                                except Exception:
                                    final_msg = get_string("final_fallback", lang_code)
                                integration = context.get("integration") or {}
                                if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                                    await instagram_service.send_message(
                                        sender_id,
                                        get_string("review_prompt", lang_code, integration["googleReviewUrl"].strip()),
                                        instagram_access_token,
                                    )
                                await instagram_service.send_message(sender_id, final_msg, instagram_access_token)
                                key = f"instagram_{recipient_id}_{sender_id}"
                                instagram_key_to_session.pop(key, None)
                                instagram_sessions.pop(session_id, None)
                                return {"status": "ok"}

                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await instagram_service.send_message(sender_id, reply, instagram_access_token)
                        return {"status": "ok"}
                    else:
                        reply = temp_reply
                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await instagram_service.send_message(sender_id, reply, instagram_access_token)
                        return {"status": "ok"}

            # ── Generate response via state machine ───────────────────────────────
            reply = await response_generator.generate_response(
                flow_controller, message_text, conversation_history, context
            )

            # ── SEND_EMAIL / SEND_PHONE signal interceptors ───────────────────────
            if "|||SEND_EMAIL:" in reply:
                parts = reply.split("|||SEND_EMAIL:", 1)
                answer, email = parts[0].strip(), parts[1].strip()
                conversation_history.append({"role": "assistant", "content": answer})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, answer, instagram_access_token)
                flow_controller.collected_data["leadEmail"] = email
                flow_controller.otp_state["email_sent"] = True
                flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
                customer_name = flow_controller.collected_data.get("leadName", "Customer")
                ok, _ = await email_validation_service.send_otp_email(owner_id, email, customer_name)
                if ok:
                    flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    email_validation_state.update({"otp_sent": True, "email": email, "customer_name": customer_name})
                if ok:
                    reply = get_string("otp_sent_email", lang_code, email)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "email",
                        email_validation_state=email_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, reply, instagram_access_token)
                return {"status": "ok"}

            elif reply.startswith("SEND_EMAIL:"):
                email = reply.split(":", 1)[1].strip()
                flow_controller.collected_data["leadEmail"] = email
                flow_controller.otp_state["email_sent"] = True
                flow_controller.transition_to(ConversationState.EMAIL_OTP_SENT)
                customer_name = flow_controller.collected_data.get("leadName", "Customer")
                ok, _ = await email_validation_service.send_otp_email(owner_id, email, customer_name)
                if ok:
                    flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    email_validation_state.update({"otp_sent": True, "email": email, "customer_name": customer_name})
                if ok:
                    reply = get_string("otp_sent_email", lang_code, email)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "email",
                        email_validation_state=email_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, reply, instagram_access_token)
                return {"status": "ok"}

            elif "|||SEND_PHONE:" in reply:
                parts = reply.split("|||SEND_PHONE:", 1)
                answer, phone = parts[0].strip(), parts[1].strip()
                from app.utils.phone_utils import format_phone_number_with_gpt
                phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
                conversation_history.append({"role": "assistant", "content": answer})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, answer, instagram_access_token)
                flow_controller.collected_data["leadPhoneNumber"] = phone
                flow_controller.otp_state["phone_sent"] = True
                flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
                ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone)
                if ok:
                    flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                    phone_validation_state.update({"otp_sent": True, "phone": phone})
                if ok:
                    reply = get_string("otp_sent_phone", lang_code, phone)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "phone",
                        phone_validation_state=phone_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, reply, instagram_access_token)
                return {"status": "ok"}

            elif reply.startswith("SEND_PHONE:"):
                phone = reply.split(":", 1)[1].strip()
                from app.utils.phone_utils import format_phone_number_with_gpt
                phone = await format_phone_number_with_gpt(phone, openai_client, settings.gpt_model)
                flow_controller.collected_data["leadPhoneNumber"] = phone
                flow_controller.otp_state["phone_sent"] = True
                flow_controller.transition_to(ConversationState.PHONE_OTP_SENT)
                ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone)
                if ok:
                    flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                    phone_validation_state.update({"otp_sent": True, "phone": phone})
                if ok:
                    reply = get_string("otp_sent_phone", lang_code, phone)
                else:
                    reply = await _continue_after_otp_delivery_failed_with_session(
                        flow_controller,
                        response_generator,
                        conversation_history,
                        context,
                        lang_code,
                        "phone",
                        phone_validation_state=phone_validation_state,
                    )
                conversation_history.append({"role": "assistant", "content": reply})
                session["history"] = conversation_history
                await instagram_service.send_message(sender_id, reply, instagram_access_token)
                return {"status": "ok"}

            # ── Lead JSON generation ───────────────────────────────────────────────
            parsed_json = _maybe_parse_json(reply)
            if parsed_json and isinstance(parsed_json, dict) and flow_controller.can_generate_json():
                email_valid = _validate_email_verification(email_validation_state) if validate_email else True
                if email_valid:
                    if app_id:
                        parsed_json["appId"] = app_id
                    try:
                        ok, _ = await lead_service.create_public_lead(owner_id, parsed_json)
                        final_msg = (
                            get_string("final_success", lang_code)
                            if ok
                            else get_string("final_fallback", lang_code)
                        )
                    except Exception:
                        final_msg = get_string("final_fallback", lang_code)

                    integration = context.get("integration") or {}
                    if integration.get("googleReviewEnabled") and integration.get("googleReviewUrl"):
                        review_url = integration["googleReviewUrl"].strip()
                        await instagram_service.send_message(
                            sender_id,
                            get_string("review_prompt", lang_code, review_url),
                            instagram_access_token,
                        )
                    await instagram_service.send_message(sender_id, final_msg, instagram_access_token)
                    key = f"instagram_{recipient_id}_{sender_id}"
                    instagram_key_to_session.pop(key, None)
                    instagram_sessions.pop(session_id, None)
                    return {"status": "ok"}

            # ── Retry detection ───────────────────────────────────────────────────
            retry_type, extracted_value = _detect_retry_request(reply)
            if retry_type:
                if retry_type == "resend_otp" and email_validation_state["otp_sent"] and not email_validation_state["otp_verified"]:
                    customer_name = email_validation_state.get(
                        "customer_name", flow_controller.collected_data.get("leadName", "Customer")
                    )
                    ok, _ = await email_validation_service.send_otp_email(
                        owner_id, email_validation_state["email"], customer_name
                    )
                    if ok:
                        reply = get_string("otp_resend", lang_code, email_validation_state["email"])
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "email",
                            email_validation_state=email_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}

                elif retry_type == "resend_otp" and phone_validation_state["otp_sent"] and not phone_validation_state["otp_verified"]:
                    ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone_validation_state["phone"])
                    if ok:
                        reply = get_string("otp_resend", lang_code, phone_validation_state["phone"])
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "phone",
                            phone_validation_state=phone_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}

                elif retry_type == "change_email":
                    email_validation_state.update({"otp_sent": False, "otp_verified": False, "email": None})
                    flow_controller.otp_state.update({"email_sent": False, "email_verified": False})
                    if extracted_value:
                        email = extracted_value
                        email_validation_state["email"] = email
                        email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                        flow_controller.collected_data["leadEmail"] = email
                        ok, _ = await email_validation_service.send_otp_email(
                            owner_id, email, email_validation_state["customer_name"]
                        )
                        if ok:
                            email_validation_state["otp_sent"] = True
                            flow_controller.otp_state["email_sent"] = True
                            flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("perfect_otp_sent_email", lang_code, email)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "email",
                                email_validation_state=email_validation_state,
                            )
                    else:
                        reply = get_string("no_problem_email", lang_code)
                        flow_controller.transition_to(ConversationState.EMAIL_COLLECTION)
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}

                elif retry_type == "change_phone":
                    phone_validation_state.update({"otp_sent": False, "otp_verified": False, "phone": None})
                    flow_controller.otp_state.update({"phone_sent": False, "phone_verified": False})
                    if extracted_value:
                        from app.utils.phone_utils import format_phone_number_with_gpt
                        phone = await format_phone_number_with_gpt(extracted_value, openai_client, settings.gpt_model)
                        phone_validation_state["phone"] = phone
                        flow_controller.collected_data["leadPhoneNumber"] = phone
                        ok, _ = await phone_validation_service.send_sms_otp(owner_id, phone)
                        if ok:
                            phone_validation_state["otp_sent"] = True
                            flow_controller.otp_state["phone_sent"] = True
                            flow_controller.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
                        if ok:
                            reply = get_string("perfect_otp_sent_phone", lang_code, phone)
                        else:
                            reply = await _continue_after_otp_delivery_failed_with_session(
                                flow_controller,
                                response_generator,
                                conversation_history,
                                context,
                                lang_code,
                                "phone",
                                phone_validation_state=phone_validation_state,
                            )
                    else:
                        reply = get_string("no_problem_phone", lang_code)
                        flow_controller.transition_to(ConversationState.PHONE_COLLECTION)
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}

                elif retry_type == "send_email" and extracted_value:
                    if not validate_email:
                        email = extracted_value
                        email_validation_state["email"] = email
                        flow_controller.collected_data["leadEmail"] = email
                        conversation_history.append({"role": "user", "content": "Email provided"})
                        reply = await response_generator.generate_response(
                            flow_controller, "Email provided", conversation_history, context
                        )
                        conversation_history.append({"role": "assistant", "content": reply})
                        session["history"] = conversation_history
                        await instagram_service.send_message(sender_id, reply, instagram_access_token)
                        return {"status": "ok"}
                    email = extracted_value
                    email_validation_state["email"] = email
                    email_validation_state["customer_name"] = flow_controller.collected_data.get("leadName", "Customer")
                    if email_validation_state["customer_name"] == "Customer":
                        extractor = DataExtractor()
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "user":
                                name = extractor.extract_name(
                                    msg.get("content", ""), context.get("lead_types", [])
                                )
                                if name:
                                    email_validation_state["customer_name"] = name
                                    flow_controller.collected_data["leadName"] = name
                                    break
                    flow_controller.collected_data["leadEmail"] = email
                    ok, _ = await email_validation_service.send_otp_email(
                        owner_id, email, email_validation_state["customer_name"]
                    )
                    if ok:
                        email_validation_state["otp_sent"] = True
                        flow_controller.otp_state["email_sent"] = True
                        flow_controller.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
                    if ok:
                        reply = get_string("otp_sent_email", lang_code, email)
                    else:
                        reply = await _continue_after_otp_delivery_failed_with_session(
                            flow_controller,
                            response_generator,
                            conversation_history,
                            context,
                            lang_code,
                            "email",
                            email_validation_state=email_validation_state,
                        )
                    conversation_history.append({"role": "assistant", "content": reply})
                    session["history"] = conversation_history
                    await instagram_service.send_message(sender_id, reply, instagram_access_token)
                    return {"status": "ok"}

            # ── Regular reply ─────────────────────────────────────────────────────
            conversation_history.append({"role": "assistant", "content": reply})
            session["history"] = conversation_history

            cleaned_reply, buttons = _extract_buttons_from_response(reply)
            if buttons:
                qrs = [{"title": btn["title"], "payload": btn["title"]} for btn in buttons]
                await instagram_service.send_quick_replies(sender_id, cleaned_reply, qrs, instagram_access_token)
            else:
                await instagram_service.send_message(sender_id, reply, instagram_access_token)

        except Exception as exc:
            logger.exception("Instagram: error processing message: %s", exc)
            try:
                await instagram_service.send_message(
                    sender_id,
                    "Sorry, I encountered an error. Please try again.",
                    instagram_access_token,
                )
            except Exception:
                pass

        return {"status": "ok"}

    except Exception as exc:
        logger.exception("Instagram webhook error: %s", exc)
        return {"status": "error", "message": str(exc)}


