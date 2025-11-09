import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.agent.v1.socket_client import AsyncV1SocketClient
from deepgram.extensions.types.sockets import (
    AgentV1Agent,
    AgentV1AgentAudioDoneEvent,
    AgentV1AudioConfig,
    AgentV1AudioInput,
    AgentV1AudioOutput,
    AgentV1ConversationTextEvent,
    AgentV1ControlMessage,
    AgentV1DeepgramSpeakProvider,
    AgentV1InjectAgentMessageMessage,
    AgentV1InjectionRefusedEvent,
    AgentV1Listen,
    AgentV1ListenProvider,
    AgentV1MediaMessage,
    AgentV1OpenAiThinkProvider,
    AgentV1SettingsAppliedEvent,
    AgentV1SettingsMessage,
    AgentV1Think,
    AgentV1SpeakProviderConfig,
    AgentV1ErrorEvent,
    AgentV1WarningEvent,
)

from .conversation_state import ConversationState, FlowController
from .response_generator import ResponseGenerator
from .rag_service import RAGService
from .context_service import ContextService
from .lead_service import LeadService
from ..utils.phone_utils import format_phone_number_with_gpt
from ..config import settings as app_settings

logger = logging.getLogger("assistly.voice_agent")


def _maybe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON if the text looks like JSON."""
    if not text:
        return None
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        return json.loads(stripped)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to parse JSON from voice agent response")
        return None


@dataclass
class VoiceAgentSession:
    """Manages a Deepgram agent session bridged with a Twilio media stream."""

    call_sid: str
    caller_phone: str
    user_id: str
    context: Dict[str, Any]
    flow_controller: FlowController
    response_generator: ResponseGenerator
    lead_service: LeadService
    rag_service: RAGService
    deepgram_client: AsyncDeepgramClient
    settings_message: AgentV1SettingsMessage
    openai_client: Any
    gpt_model: str
    keepalive_interval: Optional[int] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    on_stop: Optional[Callable[[str], None]] = None

    twilio_websocket: Optional[WebSocket] = None
    twilio_stream_sid: Optional[str] = None
    pending_audio: List[str] = field(default_factory=list)
    pending_twilio_audio: List[bytes] = field(default_factory=list)
    pending_agent_messages: List[str] = field(default_factory=list)
    deepgram_ready: bool = False
    active: bool = True
    _last_user_text: Optional[str] = None
    _stopping: bool = False
    twilio_media_logged: int = 0
    _last_injected_message: Optional[str] = None
    _allow_agent_audio: bool = False
    socket_cm: Optional[Any] = None
    agent_socket: Optional[AsyncV1SocketClient] = None

    def __post_init__(self) -> None:
        self.listen_task: Optional[asyncio.Task] = None
        self.keepalive_task: Optional[asyncio.Task] = None
        self.deepgram_ready_event = asyncio.Event()

    async def start(self) -> None:
        """Open the Deepgram connection and send initial settings."""
        if self.agent_socket:
            return

        self.socket_cm = self.deepgram_client.agent.v1.connect()
        self.agent_socket = await self.socket_cm.__aenter__()

        self.agent_socket.on(EventType.MESSAGE, self._on_agent_message)
        self.agent_socket.on(EventType.ERROR, self._on_agent_error)
        self.agent_socket.on(EventType.CLOSE, self._on_agent_close)

        self.listen_task = asyncio.create_task(self.agent_socket.start_listening())
        await self.agent_socket.send_settings(self.settings_message)

        await self.deepgram_ready_event.wait()
        logger.info("Deepgram agent websocket started for call %s", self.call_sid)

    async def stop(self) -> None:
        """Stop session, close sockets, and trigger cleanup."""
        if self._stopping:
            return
        self._stopping = True
        if not self.active:
            self._stopping = False
            return
        self.active = False

        tasks = [self.listen_task, self.keepalive_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self.agent_socket:
            try:
                await self.agent_socket._websocket.close()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close Deepgram websocket for call %s", self.call_sid)
            self.agent_socket = None

        if self.socket_cm:
            try:
                await self.socket_cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to exit Deepgram websocket context for call %s", self.call_sid)
            self.socket_cm = None

        if self.on_stop:
            try:
                self.on_stop(self.call_sid)
            except Exception:  # noqa: BLE001
                logger.exception("Error running voice agent stop callback for call %s", self.call_sid)
            finally:
                self.on_stop = None
        self._stopping = False

    async def enqueue_agent_message(self, message: str, *, add_to_history: bool = False) -> None:
        """Queue or immediately send an agent message."""
        if not message:
            return
        if add_to_history:
            self.conversation_history.append({"role": "assistant", "content": message})
        if self.deepgram_ready and self.twilio_websocket and self.agent_socket:
            await self._send_agent_message(message)
        else:
            self.pending_agent_messages.append(message)

    async def handle_twilio_stream(
        self,
        websocket: WebSocket,
        initial_messages: Optional[List[str]] = None,
    ) -> None:
        """Attach Twilio media stream websocket."""
        self.twilio_websocket = websocket
        logger.info("Twilio websocket accepted for call %s", self.call_sid)

        await self._flush_pending_twilio_audio()
        await self._flush_pending_agent_messages()
        await self._flush_pending_outbound_audio()

        try:
            if initial_messages:
                for message in initial_messages:
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("Invalid initial JSON from Twilio stream: %s", message)
                        continue
                    await self._process_twilio_event(payload)

            while True:
                message = await websocket.receive_text()
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from Twilio stream: %s", message)
                    continue
                await self._process_twilio_event(payload)
        except WebSocketDisconnect:
            logger.info("Twilio stream disconnected for call %s", self.call_sid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing Twilio stream for call %s: %s", self.call_sid, exc)
        finally:
            self.twilio_websocket = None
            if self.active:
                await self.stop()

    async def _process_twilio_event(self, payload: Dict[str, Any]) -> None:
        event_type = payload.get("event")
        if event_type == "start":
            start_info = payload.get("start", {})
            self.twilio_stream_sid = start_info.get("streamSid")
            logger.info(
                "Twilio stream started for call %s (streamSid=%s)",
                self.call_sid,
                self.twilio_stream_sid,
            )
            await self._flush_pending_twilio_audio()
            await self._flush_pending_agent_messages()
            await self._flush_pending_outbound_audio()
        elif event_type == "media":
            media = payload.get("media", {})
            audio_payload = media.get("payload")
            if audio_payload:
                try:
                    audio_bytes = base64.b64decode(audio_payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to decode Twilio media for call %s: %s", self.call_sid, exc)
                    return
                if self.twilio_media_logged < 5:
                    logger.info(
                        "Twilio media chunk #%d for call %s: b64_len=%d decoded_len=%d preview=%s",
                        self.twilio_media_logged + 1,
                        self.call_sid,
                        len(audio_payload),
                        len(audio_bytes),
                        audio_bytes[:16],
                    )
                    self.twilio_media_logged += 1
                if self.deepgram_ready:
                    await self._forward_audio_to_deepgram(audio_bytes)
                else:
                    self.pending_twilio_audio.append(audio_bytes)
        elif event_type in {"stop", "closed"}:
            logger.info("Twilio stream stopped for call %s", self.call_sid)
            if self.agent_socket:
                try:
                    await self.agent_socket.send_control(AgentV1ControlMessage())
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to flush Deepgram input for call %s: %s", self.call_sid, exc)
            await self.stop()

    async def _forward_audio_to_deepgram(self, audio_chunk: bytes) -> None:
        """Forward raw μ-law audio from Twilio to Deepgram agent."""
        if not self.agent_socket:
            logger.debug("Deepgram websocket closed; dropping audio for call %s", self.call_sid)
            return
        try:
            await self.agent_socket.send_media(audio_chunk)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to forward audio to Deepgram for call %s: %s", self.call_sid, exc)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split()).lower()

    async def _send_audio_to_twilio(self, audio_bytes: bytes) -> None:
        """Send Deepgram audio back to Twilio stream."""
        if not audio_bytes:
            return

        base64_payload = base64.b64encode(audio_bytes).decode("ascii")

        if not self.twilio_websocket or not self.twilio_stream_sid:
            self.pending_audio.append(base64_payload)
            return

        if not self._allow_agent_audio:
            logger.debug(
                "Blocking Deepgram audio for call %s because it was not requested",
                self.call_sid,
            )
            return

        message = {
            "event": "media",
            "streamSid": self.twilio_stream_sid,
            "media": {
                "payload": base64_payload,
            },
        }

        try:
            await self.twilio_websocket.send_text(json.dumps(message))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to relay audio to Twilio for call %s: %s", self.call_sid, exc)

    async def _handle_user_input(self, user_text: str) -> None:
        """Handle user transcript by generating our application response."""
        cleaned_text = user_text.strip()
        if not cleaned_text:
            return

        if cleaned_text == self._last_user_text:
            logger.debug("Skipping duplicate transcript for call %s: %s", self.call_sid, cleaned_text)
            return
        self._last_user_text = cleaned_text

        logger.info("VoiceAgent (%s) user: %s", self.call_sid, cleaned_text)
        self.conversation_history.append({"role": "user", "content": cleaned_text})

        try:
            reply = await self.response_generator.generate_response(
                self.flow_controller,
                cleaned_text,
                self.conversation_history,
                self.context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("VoiceAgent (%s) failed to generate response: %s", self.call_sid, exc)
            fallback = (
                "I'm sorry, I didn't catch that. Could you please repeat what you need?"
            )
            await self.enqueue_agent_message(fallback, add_to_history=True)
            return

        parsed_json = _maybe_parse_json(reply)
        if parsed_json:
            if not parsed_json.get("leadPhoneNumber"):
                parsed_json["leadPhoneNumber"] = self.flow_controller.collected_data.get("leadPhoneNumber")
            parsed_json.setdefault("history", self.conversation_history)

            try:
                ok, _ = await self.lead_service.create_public_lead(self.user_id, parsed_json)
                final_msg = (
                    "Thanks! I have your details and someone will follow up with you very soon."
                    if ok
                    else "Thanks! I captured your details. The team will follow up with you shortly."
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to create lead for voice call %s: %s", self.call_sid, exc)
                final_msg = "Thanks! I captured your details. The team will follow up with you shortly."

            self.conversation_history.append({"role": "assistant", "content": final_msg})
            await self.enqueue_agent_message(final_msg)
            await self.stop()
            return

        logger.info("VoiceAgent (%s) reply: %s", self.call_sid, reply)
        self.conversation_history.append({"role": "assistant", "content": reply})
        await self.enqueue_agent_message(reply)

    async def _send_agent_message(self, message: str) -> None:
        if not self.agent_socket:
            self.pending_agent_messages.append(message)
            return
        payload = AgentV1InjectAgentMessageMessage(message=message)
        try:
            if not self.deepgram_ready:
                self.pending_agent_messages.append(message)
                return
            await self.agent_socket.send_inject_agent_message(payload)
            self._last_injected_message = self._normalize_text(message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to send message to Deepgram for call %s: %s", self.call_sid, exc)

    async def _flush_pending_twilio_audio(self) -> None:
        if not self.deepgram_ready or not self.pending_twilio_audio or not self.agent_socket:
            return
        for chunk in list(self.pending_twilio_audio):
            await self._forward_audio_to_deepgram(chunk)
        self.pending_twilio_audio.clear()

    async def _flush_pending_agent_messages(self) -> None:
        if not self.pending_agent_messages:
            return
        if not (self.deepgram_ready and self.twilio_websocket and self.agent_socket):
            return
        for message in list(self.pending_agent_messages):
            await self._send_agent_message(message)
        self.pending_agent_messages.clear()

    async def _flush_pending_outbound_audio(self) -> None:
        if not self.pending_audio:
            return
        if not (self.twilio_websocket and self.twilio_stream_sid):
            return
        for payload in list(self.pending_audio):
            message = {
                "event": "media",
                "streamSid": self.twilio_stream_sid,
                "media": {"payload": payload},
            }
            try:
                await self.twilio_websocket.send_text(json.dumps(message))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to flush pending audio to Twilio for call %s: %s", self.call_sid, exc)
                break
            self.pending_audio.remove(payload)

    async def _keepalive_loop(self) -> None:
        if not self.agent_socket or not self.keepalive_interval:
            return
        try:
            while True:
                await asyncio.sleep(self.keepalive_interval)
                try:
                    await self.agent_socket.send_control(AgentV1ControlMessage())
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to send keepalive for call %s", self.call_sid)
                    return
        except asyncio.CancelledError:
            return

    async def _on_agent_message(self, message: Any) -> None:
        if isinstance(message, AgentV1SettingsAppliedEvent):
            self.deepgram_ready = True
            self.deepgram_ready_event.set()
            await self._flush_pending_twilio_audio()
            await self._flush_pending_agent_messages()
            await self._flush_pending_outbound_audio()
            if not self.keepalive_task and self.keepalive_interval:
                self.keepalive_task = asyncio.create_task(self._keepalive_loop())
            return

        if isinstance(message, (bytes, bytearray)):
            await self._send_audio_to_twilio(message)
            return

        if isinstance(message, AgentV1ConversationTextEvent):
            role = (message.role or "").lower()
            if role == "user":
                await self._handle_user_input(message.content)
                return
            if role == "assistant":
                logger.info(
                    "Deepgram agent (%s) assistant output: %s",
                    self.call_sid,
                    message.content,
                )
                normalized = self._normalize_text(message.content)
                if self._last_injected_message and normalized == self._last_injected_message:
                    self._allow_agent_audio = True
                else:
                    self._allow_agent_audio = False
                    logger.info(
                        "Suppressing unsolicited Deepgram audio for call %s", self.call_sid
                    )
                return

        if isinstance(message, AgentV1AgentAudioDoneEvent):
            await self._flush_pending_agent_messages()
            self._allow_agent_audio = False
            return

        if isinstance(message, AgentV1InjectionRefusedEvent):
            logger.warning("Deepgram injection refused for call %s", self.call_sid)
            return

        if isinstance(message, AgentV1ErrorEvent):
            logger.error("Deepgram error for call %s: %s", self.call_sid, message)
            return

        if isinstance(message, AgentV1WarningEvent):
            logger.warning("Deepgram warning for call %s: %s", self.call_sid, message)
            return

        if isinstance(message, dict):
            event_type = (message.get("type") or message.get("event") or "").lower()
            if event_type == "conversationtext" and (message.get("role") or "").lower() == "user":
                await self._handle_user_input(message.get("content", ""))
                return
            if event_type == "agentaudiodone":
                await self._flush_pending_agent_messages()
                return
            if event_type == "error":
                logger.error("Deepgram error for call %s: %s", self.call_sid, message)
                return
            if event_type == "warning":
                logger.warning("Deepgram warning for call %s: %s", self.call_sid, message)
                return

        logger.debug("Unhandled Deepgram event for call %s: %s", self.call_sid, message)

    async def _on_agent_error(self, error: Any) -> None:
        logger.error("Deepgram websocket error for call %s: %s", self.call_sid, error)

    async def _on_agent_close(self, _event: Any) -> None:
        logger.info("Deepgram websocket closed for call %s", self.call_sid)
        self.deepgram_ready = False
        if not self.deepgram_ready_event.is_set():
            self.deepgram_ready_event.set()
        if self.active and not self._stopping:
            await self.stop()


class VoiceAgentService:
    """Service for managing Deepgram voice agent sessions."""

    def __init__(self, settings: Any = app_settings):
        self.settings = settings
        self.deepgram_api_key: Optional[str] = getattr(settings, "deepgram_api_key", None)
        self.context_service = ContextService(settings)
        self.lead_service = LeadService(settings)
        self.sessions: Dict[str, VoiceAgentSession] = {}

        from openai import AsyncOpenAI

        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.gpt_model = settings.gpt_model

        if self.deepgram_api_key:
            self.deepgram_client = AsyncDeepgramClient(api_key=self.deepgram_api_key)
        else:
            self.deepgram_client = None
            logger.warning("Deepgram API key is not configured")

    async def _build_agent_prompt(self, context: Dict[str, Any]) -> str:
        profession = context.get("profession", "clinic")
        integration = context.get("integration", {}) or {}
        assistant_name = integration.get("assistantName") or "Assistant"
        lead_types = context.get("lead_types", []) or []
        services = context.get("service_types", []) or []

        lead_type_names = ", ".join(
            [lt.get("text", "") for lt in lead_types if isinstance(lt, dict) and lt.get("text")]
        ) or "our available services"
        service_names = ", ".join([str(s) for s in services if s]) or "our services"

        prompt = (
            f"You are {assistant_name}, an AI assistant for a {profession}. "
            "You will receive the caller's speech as text and must not respond autonomously. "
            "Wait for injected agent messages from the application and speak them verbatim. "
            "Do not improvise or add additional commentary. "
            f"Lead type options: {lead_type_names}. "
            f"Service options: {service_names}. "
            "Speak clearly and naturally."
        )
        return prompt

    async def start_session(
        self,
        call_sid: str,
        caller_phone: str,
        twilio_phone: Optional[str],
    ) -> VoiceAgentSession:
        if not self.deepgram_client:
            raise ValueError("Deepgram client not configured")

        if call_sid in self.sessions and self.sessions[call_sid].active:
            logger.info("VoiceAgent session already active for %s", call_sid)
            return self.sessions[call_sid]

        logger.info(
            "Starting voice agent session for call %s from %s (twilio=%s)",
            call_sid,
            caller_phone,
            twilio_phone,
        )

        lookup_phone = twilio_phone or caller_phone
        context = await self.context_service.fetch_user_context_by_twilio(lookup_phone)
        user_data = context.get("user", {}) or {}
        user_id = user_data.get("id")
        if not user_id:
            raise ValueError("User id missing in context response for voice call")

        integration = context.setdefault("integration", {})
        integration["validateEmail"] = False
        integration["validatePhoneNumber"] = False
        integration["channel"] = "voice"

        rag_service = RAGService(self.settings)
        rag_service.build_vector_store(context)

        flow_controller = FlowController(context)
        flow_controller.set_voice_agent()

        if not self.openai_client:
            raise ValueError("OpenAI client not configured for phone formatting")
        formatted_phone = await format_phone_number_with_gpt(caller_phone, self.openai_client, self.gpt_model)
        flow_controller.update_collected_data("leadPhoneNumber", formatted_phone)
        flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)

        response_generator = ResponseGenerator(self.settings, rag_service)
        response_generator.set_profession(str(context.get("profession") or "Clinic"))
        response_generator.set_channel("voice")

        agent_prompt = await self._build_agent_prompt(context)
        fallback_greeting = ""
        dynamic_greeting = await response_generator.generate_greeting(context, channel="voice")

        audio_config = AgentV1AudioConfig(
            input=AgentV1AudioInput(encoding="mulaw", sample_rate=8000),
            output=AgentV1AudioOutput(encoding="mulaw", sample_rate=8000, container="none"),
        )

        listen_config = AgentV1Listen(provider=AgentV1ListenProvider(model="nova-3", smart_format=False))
        think_config = AgentV1Think(
            provider=AgentV1OpenAiThinkProvider(model=self.gpt_model or "gpt-4.1-nano", temperature=0.3),
            prompt=agent_prompt,
        )
        speak_config = AgentV1SpeakProviderConfig(
            provider=AgentV1DeepgramSpeakProvider(model="aura-2-thalia-en")
        )

        settings_message = AgentV1SettingsMessage(
            audio=audio_config,
            agent=AgentV1Agent(
                language="en",
                listen=listen_config,
                think=think_config,
                speak=speak_config,
                greeting=fallback_greeting,
            ),
        )

        session = VoiceAgentSession(
            call_sid=call_sid,
            caller_phone=formatted_phone,
            user_id=user_id,
            context=context,
            flow_controller=flow_controller,
            response_generator=response_generator,
            lead_service=self.lead_service,
            rag_service=rag_service,
            deepgram_client=self.deepgram_client,
            settings_message=settings_message,
            openai_client=self.openai_client,
            gpt_model=self.gpt_model,
            keepalive_interval=8,
        )

        self.sessions[call_sid] = session
        session.on_stop = lambda sid: self.sessions.pop(sid, None)

        await session.start()

        greeting_to_send = dynamic_greeting or fallback_greeting
        if greeting_to_send:
            await session.enqueue_agent_message(greeting_to_send, add_to_history=True)

        return session

    async def stop_session(self, call_sid: str) -> None:
        session = self.sessions.pop(call_sid, None)
        if not session:
            return
        await session.stop()

    def get_session(self, call_sid: str) -> Optional[VoiceAgentSession]:
        return self.sessions.get(call_sid)

    async def attach_twilio_stream(
        self,
        call_sid: str,
        websocket: WebSocket,
        initial_messages: Optional[List[str]] = None,
    ) -> None:
        session = self.sessions.get(call_sid)
        if not session:
            raise ValueError(f"No active voice agent session for {call_sid}")
        await session.handle_twilio_stream(websocket, initial_messages)

