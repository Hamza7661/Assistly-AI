import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from deepgram import AsyncDeepgramClient

# Try to import Settings model if available
try:
    from deepgram.agent.v1.models import Settings
    DEEPGRAM_SETTINGS_AVAILABLE = True
except ImportError:
    try:
        from deepgram.agent.v1 import Settings
        DEEPGRAM_SETTINGS_AVAILABLE = True
    except ImportError:
        DEEPGRAM_SETTINGS_AVAILABLE = False

from .context_service import ContextService
from .lead_service import LeadService
from ..utils.phone_utils import format_phone_number_with_gpt
from ..utils.greeting_utils import get_greeting_with_fallback
from ..config import settings as app_settings

logger = logging.getLogger("assistly.voice_agent")


def _maybe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON if the text looks like JSON. Extracts JSON from anywhere in the text."""
    if not text:
        return None
    stripped = text.strip()
    
    # First try: if entire text is JSON
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:  # noqa: BLE001
            pass
    
    # Second try: extract JSON object from anywhere in the text
    # Find the first { and try to match it with the last }
    start_idx = stripped.find("{")
    if start_idx == -1:
        return None
    
    # Find matching closing brace
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(stripped)):
        if stripped[i] == "{":
            brace_count += 1
        elif stripped[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if end_idx != -1:
        json_str = stripped[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to parse extracted JSON from voice agent response")
            return None
    
    return None


@dataclass
class VoiceAgentSession:
    """Manages a voice agent session using Deepgram Voice Agent API."""

    call_sid: str
    caller_phone: str
    user_id: str
    context: Dict[str, Any]
    lead_service: LeadService
    deepgram_client: AsyncDeepgramClient
    on_stop: Optional[Callable[[str], None]] = None
    app_id: Optional[str] = None  # For app-scoped leads

    twilio_websocket: Optional[WebSocket] = None
    twilio_stream_sid: Optional[str] = None
    pending_twilio_audio: List[bytes] = field(default_factory=list)
    active: bool = True
    _stopping: bool = False

    # Deepgram Voice Agent API connection
    deepgram_agent_connection: Optional[Any] = None
    deepgram_agent_connection_cm: Optional[Any] = None
    deepgram_agent_listen_task: Optional[asyncio.Task] = None
    deepgram_agent_ready: bool = False
    deepgram_agent_ready_event: Optional[asyncio.Event] = None
    cached_trigger_audio: Optional[bytes] = None
    _last_user_audio_time: Optional[float] = None  # Track when user last spoke
    _first_response_sent: bool = False  # Track if we've sent first response chunk
    _json_buffer: str = ""  # Buffer for incomplete JSON messages

    def __post_init__(self) -> None:
        self.deepgram_agent_ready_event = asyncio.Event()

    async def start(self) -> None:
        """Initialize session. Deepgram Voice Agent will start when Twilio stream is ready."""
        logger.info("Voice agent session started for call %s", self.call_sid)

    async def _build_agent_prompt(self) -> str:
        """Build system prompt for Deepgram Voice Agent to collect lead information."""
        profession = self.context.get("profession", "clinic")
        integration = self.context.get("integration", {}) or {}
        assistant_name = integration.get("assistantName") or "Assistant"
        lead_types = self.context.get("lead_types", []) or []
        treatment_plans = self.context.get("treatment_plans", []) or []
        faqs = self.context.get("faqs", []) or []
        workflows = self.context.get("workflows", []) or []

        # Build lead types list
        lead_type_list = []
        for lt in lead_types:
            if isinstance(lt, dict):
                text = lt.get("text", "")
                value = lt.get("value", "")
                if text:
                    lead_type_list.append(f"- {text} (value: {value})")
        lead_types_text = "\n".join(lead_type_list) if lead_type_list else "No specific lead types"

        # Build treatment plans with workflows
        treatment_plans_text = []
        for plan in treatment_plans:
            if isinstance(plan, dict):
                question = plan.get("question", "")
                answer = plan.get("answer", "")
                attached_workflows = plan.get("attachedWorkflows", [])
                
                plan_info = f"- {question}"
                if answer:
                    plan_info += f" (Answer: {answer})"
                
                # Check for workflows - match workflowId with full workflows array
                if attached_workflows:
                    workflow_questions = []
                    for attached_wf in attached_workflows:
                        workflow_id = attached_wf.get("workflowId") or attached_wf.get("workflow", {}).get("_id")
                        # Find the full workflow from workflows array
                        full_workflow = None
                        for wf in workflows:
                            if isinstance(wf, dict) and wf.get("_id") == workflow_id:
                                full_workflow = wf
                                break
                        
                        # Get questions from full workflow
                        if full_workflow:
                            wf_questions = full_workflow.get("questions", [])
                            # Sort by order if available
                            wf_questions = sorted(wf_questions, key=lambda x: x.get("order", 0))
                            for q in wf_questions:
                                q_text = q.get("question", "")
                                if q_text:
                                    workflow_questions.append(f"  * {q_text}")
                    
                    if workflow_questions:
                        plan_info += "\n  Workflow questions to ask:\n" + "\n".join(workflow_questions)
                
                treatment_plans_text.append(plan_info)
        
        treatment_plans_formatted = "\n".join(treatment_plans_text) if treatment_plans_text else "No specific treatment plans"

        # Build FAQs/knowledge base
        faqs_text = []
        for faq in faqs:
            if isinstance(faq, dict):
                question = faq.get("question", "")
                answer = faq.get("answer", "")
                if question and answer:
                    faqs_text.append(f"Q: {question}\nA: {answer}")
        
        faqs_formatted = "\n".join(faqs_text) if faqs_text else "No FAQs available"

        # Get greeting from integration settings with placeholder replacement
        greeting_message = get_greeting_with_fallback(self.context)
        
        prompt = f"""You are {assistant_name}, a warm and empathetic British English speaking AI assistant for a {profession}. Be conversational and make callers feel heard.

KNOWLEDGE BASE (answer questions from this):
{faqs_formatted}

LEAD TYPES:
{lead_types_text}

SERVICES:
{treatment_plans_formatted}

YOUR TASKS:
1. When you hear "initiate greeting", say warmly: "{greeting_message}"
2. Answer questions using the knowledge base above - be helpful and friendly.
3. Collect: service type, full name, email. (Phone: {self.caller_phone} is already known)

WORKFLOW QUESTIONS (CRITICAL):
- If a service has "Workflow questions to ask" listed and user select's that service, you MUST ask those EXACT questions in order BEFORE asking for name/email.
- Ask them word-for-word, do NOT rephrase or make up questions.
- Be empathetic when asking - acknowledge their responses naturally.

JSON OUTPUT (CRITICAL - ABSOLUTE RULE):
- When you have service type, name, and email, you MUST output ONLY the JSON below in your internal response text.
- DO NOT speak the JSON out loud - output it silently in your response text only. The JSON should NOT be converted to speech.
- DO NOT say "Thank you", "Perfect", "Great", "Let me save that", "I'll output the JSON", "Here's your information", or ANY words before the JSON.
- DO NOT say anything after the JSON.
- DO NOT summarize or confirm verbally.
- Just output the JSON object immediately in text only - no preamble, no summary, no confirmation, no announcements, no speaking it aloud.

Output this JSON format:
{{
  "title": "text from lead type matching leadType value",
  "leadName": "full name",
  "leadEmail": "email address",
  "leadPhoneNumber": "{self.caller_phone}",
  "leadType": "lead type value",
  "serviceType": "selected service name",
  "summary": "brief conversation summary",
  "description": "what customer wants"
}}

RULES:
- British English accent
- Be warm, empathetic, and conversational - not robotic
- Show understanding and acknowledge responses naturally
- Ask one question at a time
- If service has workflow questions, ask them first (exact wording, in order)
- When ready: output JSON ONLY - no words before, no words after, no exceptions"""
        return prompt

    async def _start_deepgram_agent(self) -> None:
        """Start Deepgram Voice Agent API connection."""
        try:
            # Build agent prompt
            agent_prompt = await self._build_agent_prompt()
            
            # Use Deepgram Voice Agent API - handles STT, LLM, and TTS
            # Connect to the agent
            connection_cm = self.deepgram_client.agent.v1.connect()
            
            # Store the context manager and enter it to get the connection
            self.deepgram_agent_connection_cm = connection_cm
            self.deepgram_agent_connection = await connection_cm.__aenter__()
            
            # Build greeting message using integration settings
            greeting_text = get_greeting_with_fallback(self.context)
            
            # Send configuration with prompt/instructions
            # Build config dict first
            config_dict = {
                "audio": {
                    "input": {
                        "encoding": "mulaw",
                        "sample_rate": 8000
                    },
                    "output": {
                        "encoding": "mulaw",
                        "sample_rate": 8000
                    }
                },
                "agent": {
                    "language": "en",
                    "listen": {
                        "provider": {
                            "type": "deepgram",
                            "model": "nova-2"
                        }
                    },
                    "speak": {
                        "provider": {
                            "type": "deepgram",
                            "model": "aura-athena-en"
                        }
                    },
                    "think": {
                        "provider": {
                            "type": "open_ai",
                            "model": "gpt-4o-mini",
                            "temperature": 0.7
                        },
                        "prompt": agent_prompt
                    }
                }
            }
            
            # Convert to Settings model if available, otherwise use dict
            if DEEPGRAM_SETTINGS_AVAILABLE:
                try:
                    config = Settings(**config_dict)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to create Settings model, using dict: %s", exc)
                    config = config_dict
            else:
                config = config_dict
            
            # Start a task to listen for messages FIRST (before sending config)
            self.deepgram_agent_listen_task = asyncio.create_task(self._listen_deepgram_agent())
            
            # Send settings - try to use Settings model if available, otherwise send as JSON
            settings_sent = False
            if hasattr(self.deepgram_agent_connection, 'send_settings') and DEEPGRAM_SETTINGS_AVAILABLE:
                try:
                    settings_model = Settings(**config_dict)
                    await self.deepgram_agent_connection.send_settings(settings_model)
                    settings_sent = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to use send_settings() with Settings model: %s", exc)
            
            # Fallback: send as JSON message
            if not settings_sent:
                settings_message = {"type": "Settings", **config_dict}
                await self._send_agent_message(settings_message)
            
            self.deepgram_agent_ready = True
            self.deepgram_agent_ready_event.set()
            
            logger.info("Deepgram Voice Agent ready for call %s", self.call_sid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start Deepgram Voice Agent for call %s: %s", self.call_sid, exc)
            raise

    async def _send_agent_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the Deepgram Voice Agent."""
        if not self.deepgram_agent_connection:
            logger.warning("Deepgram Voice Agent connection not ready for call %s", self.call_sid)
            return
        
        # Check if connection is closed
        if hasattr(self.deepgram_agent_connection, '_websocket'):
            websocket = self.deepgram_agent_connection._websocket
            if websocket and (websocket.closed or websocket.close_code is not None):
                return
        
        try:
            sent = False
            if hasattr(self.deepgram_agent_connection, 'configure'):
                await self.deepgram_agent_connection.configure(message)
                sent = True
            elif hasattr(self.deepgram_agent_connection, 'send_message'):
                await self.deepgram_agent_connection.send_message(message)
                sent = True
            elif hasattr(self.deepgram_agent_connection, 'send'):
                if isinstance(message, dict):
                    await self.deepgram_agent_connection.send(message)
                else:
                    await self.deepgram_agent_connection.send(json.dumps(message))
                sent = True
            elif hasattr(self.deepgram_agent_connection, '_send'):
                if isinstance(message, dict):
                    await self.deepgram_agent_connection._send(message)
                else:
                    await self.deepgram_agent_connection._send(json.dumps(message))
                sent = True
            elif hasattr(self.deepgram_agent_connection, 'write'):
                if isinstance(message, dict):
                    await self.deepgram_agent_connection.write(message)
                else:
                    await self.deepgram_agent_connection.write(json.dumps(message))
                sent = True
            
            if not sent:
                logger.error("Deepgram Voice Agent connection has no method to send messages for call %s", self.call_sid)
        except (ConnectionError, OSError) as exc:
            # Connection already closed - this is expected if connection was closed
            logger.debug("Deepgram Voice Agent connection closed for call %s: %s", self.call_sid, exc)
        except Exception as exc:  # noqa: BLE001
            # Only log unexpected errors, not connection closed errors
            if "closed" not in str(exc).lower() and "1005" not in str(exc):
                logger.exception("Failed to send message to Deepgram Voice Agent for call %s: %s", self.call_sid, exc)

    async def _listen_deepgram_agent(self) -> None:
        """Listen for messages from Deepgram Voice Agent API."""
        try:
            if not self.deepgram_agent_connection:
                logger.warning("Deepgram Agent connection is None, cannot listen for call %s", self.call_sid)
                return
            
            # Check if connection is iterable
            if not hasattr(self.deepgram_agent_connection, '__aiter__'):
                logger.error("Deepgram Agent connection is not async iterable for call %s", self.call_sid)
                return
            
            # Iterate over messages from the agent connection
            async for message in self.deepgram_agent_connection:
                # Handle different message types
                if hasattr(message, 'type'):
                    msg_type = message.type
                    # Log error messages with full details
                    if msg_type == "error" or hasattr(message, 'error'):
                        error_details = {}
                        if hasattr(message, 'error'):
                            error_details['error'] = message.error
                        if hasattr(message, 'message'):
                            error_details['message'] = message.message
                        if hasattr(message, 'code'):
                            error_details['code'] = message.code
                        logger.error("Deepgram Agent error for call %s: %s", self.call_sid, error_details)
                
                if isinstance(message, dict):
                    msg_type = message.get("type", "unknown")
                    # Log error messages specifically
                    if msg_type == "error" or "error" in message:
                        logger.error("Deepgram Agent error for call %s: %s", self.call_sid, message)
                
                await self._on_deepgram_agent_message(message)
        except asyncio.CancelledError:
            pass
        except (ConnectionError, OSError):
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in Deepgram Voice Agent listener for call %s: %s", self.call_sid, exc)

    def _extract_response_text(self, message: Any) -> Optional[str]:
        """Extract response text from message in any format."""
        # Try object format first
        if hasattr(message, 'response'):
            return getattr(message, 'response', None)
        if hasattr(message, 'text'):
            return getattr(message, 'text', None)
        if hasattr(message, 'content'):
            return getattr(message, 'content', None)
        
        # Try dict format
        if isinstance(message, dict):
            return message.get("response") or message.get("text") or message.get("content")
        
        return None

    async def _on_deepgram_agent_message(self, message: Any) -> None:
        """Handle messages from Deepgram Voice Agent API."""
        try:
            # Handle raw bytes (audio data) - Deepgram sends audio as raw bytes
            if isinstance(message, bytes):
                # Calculate time since user last spoke (only log first chunk of response)
                if self._last_user_audio_time and not self._first_response_sent:
                    elapsed = time.time() - self._last_user_audio_time
                    logger.info("Time to first audio for call %s: %.3fs (user spoke → first agent audio chunk)", self.call_sid, elapsed)
                    self._first_response_sent = True  # Mark that we've sent first response
                await self._send_audio_to_twilio(message)
                return
            
            # Handle message objects (AgentV1WelcomeMessage, AgentV1ErrorEvent, etc.)
            if hasattr(message, 'type'):
                msg_type = message.type
                
                # Handle error events
                if msg_type == "error" or hasattr(message, 'error') or "error" in str(type(message)).lower():
                    error_details = {}
                    for attr in ['error', 'message', 'description', 'code', 'details', 'type']:
                        if hasattr(message, attr):
                            error_details[attr] = getattr(message, attr)
                    if hasattr(message, '__dict__'):
                        error_details.update(message.__dict__)
                    logger.error("Deepgram Agent error for call %s: %s", self.call_sid, error_details)
                    return
                
                # Handle welcome message
                if msg_type == "welcome" or "welcome" in str(type(message)).lower():
                    return
                
                # Handle audio messages
                if hasattr(message, 'audio') or msg_type == "audio":
                    audio_data = getattr(message, 'audio', None) or getattr(message, 'data', None)
                    if audio_data:
                        # Calculate time since user last spoke (only log first chunk of response)
                        if self._last_user_audio_time and not self._first_response_sent:
                            elapsed = time.time() - self._last_user_audio_time
                            logger.info("Time to first audio for call %s: %.3fs (user spoke → first agent audio chunk)", self.call_sid, elapsed)
                            self._first_response_sent = True  # Mark that we've sent first response
                        if isinstance(audio_data, str):
                            audio_bytes = base64.b64decode(audio_data)
                        else:
                            audio_bytes = audio_data
                        await self._send_audio_to_twilio(audio_bytes)
                    return
                
                # Handle transcript messages
                if hasattr(message, 'transcript') or msg_type == "transcript":
                    transcript = getattr(message, 'transcript', None) or getattr(message, 'text', None)
                    if transcript:
                        logger.info("Deepgram Agent transcript for call %s: %s", self.call_sid, transcript)
                    return
            
            # Parse message if it's a string
            if isinstance(message, str):
                try:
                    message = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse agent message as JSON for call %s: %s", self.call_sid, message[:100])
                    return
            
            # Handle dict format messages
            if isinstance(message, dict):
                msg_type = message.get("type")
                
                # Audio output from agent
                if msg_type == "audio" or "audio" in message:
                    audio_data = message.get("audio") or message.get("data")
                    if audio_data:
                        # Decode if base64, or use directly
                        if isinstance(audio_data, str):
                            audio_bytes = base64.b64decode(audio_data)
                        else:
                            audio_bytes = audio_data
                        await self._send_audio_to_twilio(audio_bytes)
                    return
                
                # Text/transcript from agent (for logging)
                if msg_type == "transcript" or "transcript" in message:
                    transcript = message.get("transcript") or message.get("text")
                    if transcript:
                        logger.info("Deepgram Agent transcript for call %s: %s", self.call_sid, transcript)
                    return
            
            # Extract response text once (works for both object and dict formats)
            response_text = self._extract_response_text(message)
            if response_text:
                # Log full response text
                logger.info("Deepgram Agent full response text for call %s: %s", self.call_sid, response_text)
                
                # Check if response starts with { - if so, treat everything as JSON
                stripped = response_text.strip()
                starts_with_json = stripped.startswith("{")
                
                # If we're already buffering JSON, or this message starts with {, buffer it
                if self._json_buffer or starts_with_json:
                    # Add to JSON buffer
                    if self._json_buffer:
                        # If buffer doesn't end with } and new message doesn't start with {, add space
                        if not self._json_buffer.rstrip().endswith("}") and not stripped.startswith("{"):
                            self._json_buffer += " " + response_text
                        else:
                            self._json_buffer += response_text
                    else:
                        # Starting new JSON buffer
                        self._json_buffer = response_text
                    
                    logger.info("Buffering JSON for call %s (buffer length: %d)", self.call_sid, len(self._json_buffer))
                    
                    # Try to parse the buffered JSON
                    parsed_json = _maybe_parse_json(self._json_buffer)
                    if parsed_json:
                        logger.info("Deepgram Agent returned complete JSON for call %s: %s", self.call_sid, parsed_json)
                        self._json_buffer = ""  # Clear buffer
                        await self._handle_lead_json(parsed_json)
                    else:
                        # Check if JSON might be incomplete (doesn't end with } or has unmatched braces)
                        brace_count = self._json_buffer.count("{") - self._json_buffer.count("}")
                        if brace_count > 0:
                            logger.debug("JSON buffer incomplete for call %s (unmatched braces: %d), waiting for more", self.call_sid, brace_count)
                        elif len(self._json_buffer) > 20000:  # Prevent buffer from growing too large
                            logger.warning("JSON buffer too large for call %s, clearing", self.call_sid)
                            self._json_buffer = ""
                        else:
                            # Try to extract and fix incomplete JSON (e.g., incomplete history array)
                            logger.warning("JSON buffer not parseable for call %s, attempting to fix: %s", self.call_sid, self._json_buffer[:500])
                            # Try to close any open structures
                            fixed_json = self._json_buffer.rstrip()
                            # Count open vs closed braces
                            open_braces = fixed_json.count("{")
                            close_braces = fixed_json.count("}")
                            # Count open vs closed brackets
                            open_brackets = fixed_json.count("[")
                            close_brackets = fixed_json.count("]")
                            # Close any open structures
                            while open_braces > close_braces:
                                fixed_json += "}"
                                close_braces += 1
                            while open_brackets > close_brackets:
                                fixed_json += "]"
                                close_brackets += 1
                            # Try parsing the fixed JSON
                            try:
                                parsed_json = json.loads(fixed_json)
                                logger.info("Successfully parsed fixed JSON for call %s", self.call_sid)
                                self._json_buffer = ""  # Clear buffer
                                await self._handle_lead_json(parsed_json)
                            except Exception:  # noqa: BLE001
                                logger.debug("Could not fix JSON for call %s, waiting for more data", self.call_sid)
                elif "{" in response_text and "leadType" in response_text:
                    # JSON might be starting mid-message - extract from { onwards
                    json_start = response_text.find("{")
                    if json_start != -1:
                        self._json_buffer = response_text[json_start:]
                        logger.info("Starting JSON buffer from mid-message for call %s", self.call_sid)
                else:
                    # Not JSON, clear buffer if it exists
                    if self._json_buffer:
                        logger.warning("Clearing JSON buffer for call %s (non-JSON message received)", self.call_sid)
                        self._json_buffer = ""
                        
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing Deepgram Agent message for call %s: %s", self.call_sid, exc)

    async def _handle_lead_json(self, lead_data: Dict[str, Any]) -> None:
        """Handle JSON lead data from Deepgram Agent and create lead."""
        try:
            # Ensure phone number is set
            if not lead_data.get("leadPhoneNumber"):
                lead_data["leadPhoneNumber"] = self.caller_phone
            
            # Extract title from lead types if missing (to match web/WhatsApp format)
            if not lead_data.get("title") and lead_data.get("leadType"):
                lead_types = self.context.get("lead_types", [])
                for lt in lead_types:
                    if isinstance(lt, dict) and lt.get("value") == lead_data.get("leadType"):
                        lead_data["title"] = lt.get("text", lead_data.get("leadType", ""))
                        break
                # Fallback: use leadType as title if not found
                if not lead_data.get("title"):
                    lead_data["title"] = lead_data.get("leadType", "")
            
            # Add appId if available (for app-scoped voice leads)
            if self.app_id:
                lead_data["appId"] = self.app_id
            
            # Create lead
            ok, _ = await self.lead_service.create_public_lead(self.user_id, lead_data)
            
            if ok:
                logger.info("Successfully created lead from Deepgram Agent for call %s", self.call_sid)
                closing_message = "Thanks! I have your details and someone will get back to you soon. Bye!"
            else:
                logger.warning("Failed to create lead from Deepgram Agent for call %s", self.call_sid)
                closing_message = "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!"
            
            # Generate and send closing message audio
            audio_chunks = []
            try:
                # Check if we can still send audio
                if not self.twilio_websocket or not self.active:
                    logger.warning("Cannot send closing message - Twilio connection not active for call %s", self.call_sid)
                else:
                    async for chunk in self.deepgram_client.speak.v1.audio.generate(
                        text=closing_message,
                        model="aura-athena-en",
                        encoding="mulaw",
                        sample_rate=8000,
                    ):
                        if chunk and self.active and self.twilio_websocket:
                            audio_chunks.append(chunk)
                            await self._send_audio_to_twilio(chunk)
                
                # Wait for the message to play before ending the call
                if audio_chunks:
                    wait_time = 10.0  # Hardcoded 10 seconds to ensure message plays completely
                    logger.info("Waiting %.2fs for closing message to play for call %s", wait_time, self.call_sid)
                    await asyncio.sleep(wait_time)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to send closing message for call %s: %s", self.call_sid, exc)
            
            # Stop the session after creating lead and sending closing message
            await self.stop()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error handling lead JSON from Deepgram Agent for call %s: %s", self.call_sid, exc)


    async def stop(self) -> None:
        """Stop session, close sockets, and trigger cleanup."""
        if self._stopping:
            return
        self._stopping = True
        if not self.active:
            self._stopping = False
            return
        self.active = False

        # Close Twilio websocket and hang up the call
        if self.twilio_websocket:
            try:
                await self.twilio_websocket.close()
                logger.info("Closed Twilio websocket for call %s", self.call_sid)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close Twilio websocket for call %s", self.call_sid)
            self.twilio_websocket = None
        
        # Hang up the Twilio call using REST API if credentials are available
        try:
            from twilio.rest import Client
            from app.config import app_settings
            if app_settings.twilio_account_sid and app_settings.twilio_auth_token:
                twilio_client = Client(app_settings.twilio_account_sid, app_settings.twilio_auth_token)
                call = twilio_client.calls(self.call_sid).update(status="completed")
                logger.info("Hanged up Twilio call %s via REST API", self.call_sid)
        except ImportError:
            logger.debug("Twilio client not available for hanging up call %s", self.call_sid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to hang up Twilio call %s via REST API: %s", self.call_sid, exc)

        # Stop Deepgram Voice Agent
        if self.deepgram_agent_listen_task and not self.deepgram_agent_listen_task.done():
            self.deepgram_agent_listen_task.cancel()
            try:
                await self.deepgram_agent_listen_task
            except asyncio.CancelledError:
                pass
        
        if self.deepgram_agent_connection_cm:
            try:
                # Exit the context manager to close the connection
                await self.deepgram_agent_connection_cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close Deepgram Voice Agent connection for call %s", self.call_sid)
            self.deepgram_agent_connection_cm = None
            self.deepgram_agent_connection = None

        if self.on_stop:
            try:
                self.on_stop(self.call_sid)
            except Exception:  # noqa: BLE001
                logger.exception("Error running voice agent stop callback for call %s", self.call_sid)
            finally:
                self.on_stop = None
        self._stopping = False


    async def handle_twilio_stream(
        self,
        websocket: WebSocket,
        initial_messages: Optional[List[str]] = None,
    ) -> None:
        """Attach Twilio media stream websocket."""
        self.twilio_websocket = websocket
        logger.info("Twilio websocket accepted for call %s", self.call_sid)

        # Start Deepgram Voice Agent now that we have Twilio stream
        if not self.deepgram_agent_ready:
            try:
                await self._start_deepgram_agent()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to start Deepgram Voice Agent when Twilio stream connected for call %s: %s", self.call_sid, exc)

        await self._flush_pending_twilio_audio()

        try:
            if initial_messages:
                for message in initial_messages:
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("Invalid initial JSON from Twilio stream: %s", message)
                        continue
                    await self._process_twilio_event(payload)

            while self.active:
                try:
                    message = await websocket.receive_text()
                except Exception:  # noqa: BLE001
                    # WebSocket closed or error - break the loop
                    if not self.active:
                        logger.info("Twilio stream loop ending - session stopped for call %s", self.call_sid)
                    break
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
            
            # Send cached trigger audio to initiate greeting
            if self.deepgram_agent_ready and self.deepgram_agent_connection and self.cached_trigger_audio:
                try:
                    await self.deepgram_agent_connection.send_media(self.cached_trigger_audio)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to send trigger audio for greeting for call %s: %s", self.call_sid, exc)
        elif event_type == "media":
            media = payload.get("media", {})
            audio_payload = media.get("payload")
            if audio_payload:
                try:
                    audio_bytes = base64.b64decode(audio_payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to decode Twilio media for call %s: %s", self.call_sid, exc)
                    return
                # Forward to Deepgram Voice Agent
                if self.deepgram_agent_ready and self.deepgram_agent_connection:
                    # Track when user audio is received (only if we're waiting for a response)
                    if self._first_response_sent or self._last_user_audio_time is None:
                        # User started speaking (either first time or after previous response)
                        self._last_user_audio_time = time.time()
                        self._first_response_sent = False  # Reset flag, waiting for new response
                    await self._forward_audio_to_deepgram_agent(audio_bytes)
                else:
                    if not hasattr(self, "pending_twilio_audio"):
                        self.pending_twilio_audio = []
                    self.pending_twilio_audio.append(audio_bytes)
        elif event_type in {"stop", "closed"}:
            logger.info("Twilio stream stopped for call %s", self.call_sid)
            await self.stop()

    async def _forward_audio_to_deepgram_agent(self, audio_chunk: bytes) -> None:
        """Forward raw μ-law audio from Twilio to Deepgram Voice Agent."""
        if not self.deepgram_agent_connection:
            logger.debug("Deepgram Voice Agent connection closed; dropping audio for call %s", self.call_sid)
            return
        
        # Check if connection is closed before sending audio
        if hasattr(self.deepgram_agent_connection, '_websocket'):
            websocket = self.deepgram_agent_connection._websocket
            if websocket and (websocket.closed or websocket.close_code is not None):
                return
        
        try:
            if hasattr(self.deepgram_agent_connection, 'send_media'):
                await self.deepgram_agent_connection.send_media(audio_chunk)
        except (ConnectionError, OSError):
            pass
        except Exception as exc:  # noqa: BLE001
            if "closed" not in str(exc).lower() and "1005" not in str(exc):
                logger.exception("Failed to forward audio to Deepgram Voice Agent for call %s: %s", self.call_sid, exc)

    async def _send_audio_to_twilio(self, audio_bytes: bytes) -> None:
        """Send audio back to Twilio stream."""
        if not audio_bytes:
            return

        # If we're buffering JSON, don't send audio (prevent JSON from being spoken)
        if self._json_buffer:
            logger.debug("Skipping audio send for call %s - JSON is being buffered", self.call_sid)
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

    async def _generate_and_cache_trigger_audio(self) -> None:
        """Generate and cache trigger audio in parallel."""
        try:
            audio_chunks = []
            async for chunk in self.deepgram_client.speak.v1.audio.generate(
                text="initiate greeting",
                model="aura-athena-en",
                encoding="mulaw",
                sample_rate=8000,
            ):
                if chunk:
                    audio_chunks.append(chunk)
            
            if audio_chunks:
                self.cached_trigger_audio = b''.join(audio_chunks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to generate trigger audio for call %s: %s", self.call_sid, exc)

    async def _flush_pending_twilio_audio(self) -> None:
        """Flush pending Twilio audio to Deepgram Voice Agent."""
        if not self.deepgram_agent_ready or not hasattr(self, "pending_twilio_audio"):
            return
        if not self.pending_twilio_audio:
            return
        for chunk in list(self.pending_twilio_audio):
            await self._forward_audio_to_deepgram_agent(chunk)
        self.pending_twilio_audio.clear()


class VoiceAgentService:
    """Service for managing voice agent sessions using Deepgram Voice Agent API."""

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
        
        # Extract app_id from context for app-scoped leads
        app_data = context.get("app", {})
        app_id = app_data.get("id") if app_data else None

        integration = context.setdefault("integration", {})
        integration["validateEmail"] = False
        integration["validatePhoneNumber"] = False
        integration["channel"] = "voice"

        if not self.openai_client:
            raise ValueError("OpenAI client not configured for phone formatting")
        formatted_phone = await format_phone_number_with_gpt(caller_phone, self.openai_client, self.gpt_model)

        session = VoiceAgentSession(
            call_sid=call_sid,
            caller_phone=formatted_phone,
            user_id=user_id,
            context=context,
            lead_service=self.lead_service,
            deepgram_client=self.deepgram_client,
            app_id=app_id  # Pass app_id for app-scoped leads
        )

        self.sessions[call_sid] = session
        session.on_stop = lambda sid: self.sessions.pop(sid, None)

        # Generate trigger audio in parallel while starting session
        asyncio.create_task(session._generate_and_cache_trigger_audio())

        await session.start()

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
