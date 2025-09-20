import json
import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .services.context_service import ContextService
from .services.lead_service import LeadService
from .services.gpt_service import GptService


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
            
            # Get GPT response
            reply = await gpt_service.agent_reply(conversation_history, user_text, context, {})
            
            # Check if it's JSON (lead completion)
            parsed_json = _maybe_parse_json(reply)
            if parsed_json and isinstance(parsed_json, dict):
                if _validate_required_fields(parsed_json):
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
