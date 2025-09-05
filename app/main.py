import json
import logging
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .services.context_service import ContextService
from .services.lead_service import LeadService
from .services.gpt_service import GptService
from .conversation.manager import ConversationManager


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

    manager = ConversationManager(
        gpt=gpt_service,
        lead_service=lead_service,
        context=context,
        user_id=user_id,
    )

    initial = await manager.start()
    await websocket.send_json({"type": "bot", "content": initial, "step": manager.step})

    try:
        while True:
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                user_text = parsed.get("content") or parsed.get("text") or data
            except json.JSONDecodeError:
                user_text = data

            # Ignore empty/no-op client messages to avoid double prompting on init
            if not user_text or not str(user_text).strip():
                continue

            reply, done = await manager.handle_user_message(user_text)
            if reply and str(reply).strip():
                await websocket.send_json({"type": "bot", "content": reply, "step": manager.step})

            if done:
                await websocket.close(code=1000)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for user_id=%s", user_id)
