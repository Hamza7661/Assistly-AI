import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from deepgram import AsyncDeepgramClient

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
    """Manages a voice agent session using Deepgram STT + TTS."""

    call_sid: str
    caller_phone: str
    user_id: str
    context: Dict[str, Any]
    flow_controller: FlowController
    response_generator: ResponseGenerator
    lead_service: LeadService
    rag_service: RAGService
    deepgram_client: AsyncDeepgramClient
    gpt_model: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    on_stop: Optional[Callable[[str], None]] = None

    twilio_websocket: Optional[WebSocket] = None
    twilio_stream_sid: Optional[str] = None
    pending_audio: List[str] = field(default_factory=list)
    pending_twilio_audio: List[bytes] = field(default_factory=list)
    active: bool = True
    _last_user_text: Optional[str] = None
    _stopping: bool = False

    # Deepgram STT connection
    deepgram_stt_connection: Optional[Any] = None
    deepgram_stt_connection_cm: Optional[Any] = None
    deepgram_stt_listen_task: Optional[asyncio.Task] = None
    deepgram_stt_ready: bool = False
    deepgram_stt_ready_event: Optional[asyncio.Event] = None
    
    # Pending greeting to send when Twilio stream is ready
    pending_greeting: Optional[str] = None

    def __post_init__(self) -> None:
        self.deepgram_stt_ready_event = asyncio.Event()

    async def start(self) -> None:
        """Initialize session. Deepgram STT will start when Twilio stream is ready."""
        # Don't start Deepgram STT yet - wait for Twilio stream to be ready
        # This prevents timeout errors from Deepgram waiting for audio
        logger.info("Voice agent session started for call %s", self.call_sid)

    async def _start_deepgram_stt(self) -> None:
        """Start Deepgram STT streaming connection."""
        try:
            # Use Deepgram's live transcription API (Nova-3 for fastest STT)
            # SDK 5.3 uses connect() as an async context manager
            connection_cm = self.deepgram_client.listen.v1.connect(
                model="nova-3",  # Using Nova-3 for fastest STT latency
                language="en",
                smart_format="false",
                encoding="mulaw",
                sample_rate="8000",
                interim_results="true",
            )
            
            # Store the context manager and enter it to get the connection
            self.deepgram_stt_connection_cm = connection_cm
            self.deepgram_stt_connection = await connection_cm.__aenter__()
            
            # Start a task to listen for messages
            self.deepgram_stt_listen_task = asyncio.create_task(self._listen_deepgram_stt())
            
            self.deepgram_stt_ready = True
            self.deepgram_stt_ready_event.set()
            
            logger.info("Deepgram STT connection started for call %s", self.call_sid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start Deepgram STT for call %s: %s", self.call_sid, exc)
            raise

    async def _listen_deepgram_stt(self) -> None:
        """Listen for messages from Deepgram STT connection."""
        try:
            if not self.deepgram_stt_connection:
                return
            
            # Iterate over messages from the connection
            async for message in self.deepgram_stt_connection:
                await self._on_deepgram_stt_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in Deepgram STT listener for call %s: %s", self.call_sid, exc)


    def _on_deepgram_stt_open(self, *args, **kwargs) -> None:
        """Handle Deepgram STT connection open."""
        logger.info("Deepgram STT connection opened for call %s", self.call_sid)
        self.deepgram_stt_ready = True
        self.deepgram_stt_ready_event.set()

    async def _on_deepgram_stt_message(self, message: Any) -> None:
        """Handle Deepgram STT messages (including transcripts)."""
        try:
            # Handle different message types
            transcript_text = None
            is_final = False
            
            if isinstance(message, dict):
                msg_type = message.get("type")
                if msg_type == "Results":
                    # This is a transcript result
                    channel = message.get("channel")
                    if channel:
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            transcript_text = alternatives[0].get("transcript", "")
                            is_final = message.get("is_final", False)
            elif hasattr(message, "type"):
                # Object format
                if message.type == "Results":
                    channel = getattr(message, "channel", None)
                    if channel:
                        alternatives = getattr(channel, "alternatives", [])
                        if alternatives and len(alternatives) > 0:
                            transcript_text = alternatives[0].transcript
                            is_final = getattr(message, "is_final", False)
            
            if transcript_text and is_final:
                # Process final transcript
                await self._handle_user_input(transcript_text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing Deepgram message for call %s: %s", self.call_sid, exc)

    def _on_deepgram_stt_error(self, error: Any) -> None:
        """Handle Deepgram STT error."""
        logger.error("Deepgram STT error for call %s: %s", self.call_sid, error)

    def _on_deepgram_stt_close(self) -> None:
        """Handle Deepgram STT connection close."""
        logger.info("Deepgram STT connection closed for call %s", self.call_sid)
        self.deepgram_stt_ready = False
        if self.active and not self._stopping:
            asyncio.create_task(self.stop())

    async def _text_to_speech_deepgram(self, text: str) -> bytes:
        """Convert text to speech using Deepgram TTS and return μ-law audio bytes for Twilio."""
        tts_start = time.perf_counter()
        try:
            # Use Deepgram TTS API - generate() returns an AsyncIterator[bytes]
            audio_chunks = []
            api_start = time.perf_counter()
            async for chunk in self.deepgram_client.speak.v1.audio.generate(
                text=text,
                model="aura-2-asteria-en",  # High-quality English voice
                encoding="mulaw",
                sample_rate=8000,
            ):
                audio_chunks.append(chunk)
            api_time = time.perf_counter() - api_start
            
            # Combine all chunks into a single bytes object
            combine_start = time.perf_counter()
            audio_data = b''.join(audio_chunks)
            combine_time = time.perf_counter() - combine_start
            
            total_time = time.perf_counter() - tts_start
            logger.info(
                "TTS timing for call %s: API=%.3fs, combine=%.3fs, total=%.3fs, text_len=%d, audio_len=%d",
                self.call_sid, api_time, combine_time, total_time, len(text), len(audio_data)
            )
            
            # Deepgram TTS already outputs in μ-law format at 8kHz, so we can use it directly
            return audio_data
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to convert text to speech using Deepgram TTS for call %s: %s", self.call_sid, exc)
            raise

    async def stop(self) -> None:
        """Stop session, close sockets, and trigger cleanup."""
        if self._stopping:
            return
        self._stopping = True
        if not self.active:
            self._stopping = False
            return
        self.active = False


        # Stop Deepgram STT
        if self.deepgram_stt_listen_task and not self.deepgram_stt_listen_task.done():
            self.deepgram_stt_listen_task.cancel()
            try:
                await self.deepgram_stt_listen_task
            except asyncio.CancelledError:
                pass
        
        if self.deepgram_stt_connection_cm:
            try:
                # Exit the context manager to close the connection
                await self.deepgram_stt_connection_cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close Deepgram STT connection for call %s", self.call_sid)
            self.deepgram_stt_connection_cm = None
            self.deepgram_stt_connection = None

        if self.on_stop:
            try:
                self.on_stop(self.call_sid)
            except Exception:  # noqa: BLE001
                logger.exception("Error running voice agent stop callback for call %s", self.call_sid)
            finally:
                self.on_stop = None
        self._stopping = False

    async def enqueue_agent_message(self, message: str, *, add_to_history: bool = False) -> None:
        """Convert text message to speech using Deepgram TTS and stream to Twilio."""
        total_start = time.perf_counter()
        if not message:
            logger.warning("Attempted to enqueue empty message for call %s", self.call_sid)
            return
        
        logger.info("Converting text to speech for call %s: %s", self.call_sid, message[:100])
        
        if add_to_history:
            self.conversation_history.append({"role": "assistant", "content": message})
        
        try:
            # Stream TTS audio directly to Twilio as chunks arrive (much faster!)
            first_chunk_time = None
            tts_start = time.perf_counter()
            total_audio_size = 0
            chunk_count = 0
            
            async for chunk in self.deepgram_client.speak.v1.audio.generate(
                text=message,
                model="aura-2-asteria-en",  # High-quality English voice
                    encoding="mulaw",
                sample_rate=8000,
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - tts_start
                    logger.info(
                        "TTS first chunk timing for call %s: time_to_first_chunk=%.3fs",
                        self.call_sid, first_chunk_time
                    )
                
                # Send chunk immediately to Twilio
                await self._send_audio_to_twilio(chunk)
                total_audio_size += len(chunk)
                chunk_count += 1
            
            tts_time = time.perf_counter() - tts_start
            total_time = time.perf_counter() - total_start
            logger.info(
                "enqueue_agent_message timing for call %s: first_chunk=%.3fs, total_tts=%.3fs, total=%.3fs, chunks=%d, audio_size=%d bytes",
                self.call_sid, first_chunk_time or 0, tts_time, total_time, chunk_count, total_audio_size
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to convert text to speech and send to Twilio for call %s: %s", self.call_sid, exc)

    async def handle_twilio_stream(
        self,
        websocket: WebSocket,
        initial_messages: Optional[List[str]] = None,
    ) -> None:
        """Attach Twilio media stream websocket."""
        self.twilio_websocket = websocket
        logger.info("Twilio websocket accepted for call %s", self.call_sid)

        # Start Deepgram STT now that we have Twilio stream (will receive audio soon)
        if not self.deepgram_stt_ready:
            try:
                await self._start_deepgram_stt()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to start Deepgram STT when Twilio stream connected for call %s: %s", self.call_sid, exc)

        await self._flush_pending_twilio_audio()
        
        # Don't send greeting here - wait for "start" event to set streamSid

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
            
            # Now that streamSid is set, send the pending greeting
            if hasattr(self, 'pending_greeting') and self.pending_greeting:
                logger.info("Sending pending greeting now that streamSid is set for call %s", self.call_sid)
                greeting = self.pending_greeting
                self.pending_greeting = None
                # Wait a moment for everything to be fully ready
                await asyncio.sleep(0.2)
                await self.enqueue_agent_message(greeting, add_to_history=True)
        elif event_type == "media":
            media = payload.get("media", {})
            audio_payload = media.get("payload")
            if audio_payload:
                try:
                    audio_bytes = base64.b64decode(audio_payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to decode Twilio media for call %s: %s", self.call_sid, exc)
                    return
                # Forward to Deepgram STT
                if self.deepgram_stt_ready and self.deepgram_stt_connection:
                    await self._forward_audio_to_deepgram_stt(audio_bytes)
                else:
                    if not hasattr(self, "pending_twilio_audio"):
                        self.pending_twilio_audio = []
                    self.pending_twilio_audio.append(audio_bytes)
        elif event_type in {"stop", "closed"}:
            logger.info("Twilio stream stopped for call %s", self.call_sid)
            await self.stop()

    async def _forward_audio_to_deepgram_stt(self, audio_chunk: bytes) -> None:
        """Forward raw μ-law audio from Twilio to Deepgram STT."""
        if not self.deepgram_stt_connection:
            logger.debug("Deepgram STT connection closed; dropping audio for call %s", self.call_sid)
            return
        try:
            # Deepgram SDK 5.3 - send audio data through the connection
            # The error message suggests using _send() method
            if hasattr(self.deepgram_stt_connection, '_send'):
                await self.deepgram_stt_connection._send(audio_chunk)
            elif hasattr(self.deepgram_stt_connection, 'send_audio'):
                await self.deepgram_stt_connection.send_audio(audio_chunk)
            elif hasattr(self.deepgram_stt_connection, 'write'):
                await self.deepgram_stt_connection.write(audio_chunk)
            elif hasattr(self.deepgram_stt_connection, '_websocket'):
                # Access underlying websocket if available
                await self.deepgram_stt_connection._websocket.send(audio_chunk)
            else:
                logger.error("Deepgram STT connection has no method to send audio for call %s", self.call_sid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to forward audio to Deepgram STT for call %s: %s", self.call_sid, exc)

    async def _send_audio_to_twilio(self, audio_bytes: bytes) -> None:
        """Send audio back to Twilio stream."""
        if not audio_bytes:
            return

        if not self.twilio_websocket or not self.twilio_stream_sid:
            logger.warning("Twilio websocket or streamSid not ready for call %s", self.call_sid)
            return

        base64_payload = base64.b64encode(audio_bytes).decode("ascii")

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

    async def _flush_pending_twilio_audio(self) -> None:
        """Flush pending Twilio audio to Deepgram STT."""
        if not self.deepgram_stt_ready or not hasattr(self, "pending_twilio_audio"):
            return
        if not self.pending_twilio_audio:
            return
        for chunk in list(self.pending_twilio_audio):
            await self._forward_audio_to_deepgram_stt(chunk)
        self.pending_twilio_audio.clear()

    async def _handle_user_input(self, user_text: str) -> None:
        """Handle user transcript by generating our application response."""
        input_start = time.perf_counter()
        cleaned_text = user_text.strip()
        if not cleaned_text:
            return

        if cleaned_text == self._last_user_text:
            logger.debug("Skipping duplicate transcript for call %s: %s", self.call_sid, cleaned_text)
            return
        self._last_user_text = cleaned_text

        logger.info("VoiceAgent (%s) user: %s", self.call_sid, cleaned_text)
        self.conversation_history.append({"role": "user", "content": cleaned_text})

        response_time = 0.0
        try:
            response_start = time.perf_counter()
            reply = await self.response_generator.generate_response(
                self.flow_controller,
                cleaned_text,
                self.conversation_history,
                self.context,
            )
            response_time = time.perf_counter() - response_start
            logger.info(
                "Response generation timing for call %s: generate_response=%.3fs",
                self.call_sid, response_time
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
        
        enqueue_start = time.perf_counter()
        await self.enqueue_agent_message(reply)
        enqueue_time = time.perf_counter() - enqueue_start
        
        total_time = time.perf_counter() - input_start
        logger.info(
            "Total _handle_user_input timing for call %s: response_gen=%.3fs, enqueue=%.3fs, total=%.3fs",
            self.call_sid, response_time, enqueue_time, total_time
        )


class VoiceAgentService:
    """Service for managing voice agent sessions using Deepgram STT + TTS."""

    def __init__(self, settings: Any = app_settings):
        self.settings = settings
        self.deepgram_api_key: Optional[str] = getattr(settings, "deepgram_api_key", None)
        self.context_service = ContextService(settings)
        self.lead_service = LeadService(settings)
        self.sessions: Dict[str, VoiceAgentSession] = {}

        from openai import AsyncOpenAI

        # OpenAI client still needed for phone formatting
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.gpt_model = settings.gpt_model

        if self.deepgram_api_key:
            self.deepgram_client = AsyncDeepgramClient(api_key=self.deepgram_api_key)
        else:
            self.deepgram_client = None
            logger.warning("Deepgram API key is not configured")

    async def start_session(
        self,
        call_sid: str,
        caller_phone: str,
        twilio_phone: Optional[str],
    ) -> VoiceAgentSession:
        if not self.deepgram_client:
            raise ValueError("Deepgram client not configured")
        if not self.openai_client:
            raise ValueError("OpenAI client not configured for phone formatting")

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

        dynamic_greeting = await response_generator.generate_greeting(context, channel="voice")

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
            gpt_model=self.gpt_model,
        )

        self.sessions[call_sid] = session
        session.on_stop = lambda sid: self.sessions.pop(sid, None)

        await session.start()

        # Don't send greeting yet - wait for Twilio stream to be ready
        # The greeting will be sent when the Twilio stream connects
        if dynamic_greeting:
            # Store greeting to send when Twilio stream is ready
            session.pending_greeting = dynamic_greeting

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
