"""Production-grade response generator using state machine and minimal prompts"""
from typing import Dict, Any, List, Optional
import logging
from openai import AsyncOpenAI
import json

from app.services.conversation_state import FlowController, ConversationState
from app.services.data_extractors import DataExtractor
from app.services.lead_type_resolver import LeadTypeResolutionMode, resolve_lead_type
from app.services.validators import Validator
from app.services.workflow_manager import WorkflowManager

logger = logging.getLogger("assistly.response_generator")


class ResponseGenerator:
    """Generate responses based on conversation state with minimal prompts"""
    
    def __init__(self, settings: Any, rag_service: Any):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.model = settings.gpt_model
        self.rag_service = rag_service
        self.profession = "Business"  # Default fallback - will be overridden by app's industry
        self.channel: str = "web"
        self.response_language: Optional[str] = None  # e.g. "Spanish" for prompts; None/English = no instruction
        # Backend API base for generating attachment download URLs
        self.api_base_url: str = getattr(settings, "api_base_url", "").rstrip("/") + "/api/v1"
        # Cache for lead-type empathy prefixes to avoid repeated LLM calls
        self._empathy_prefix_cache: Dict[str, str] = {}

    def set_response_language(self, language_name: Optional[str]) -> None:
        """Set language for all user-facing replies (e.g. 'Spanish'). None or 'English' = keep default."""
        self.response_language = language_name

    @staticmethod
    def _conversation_style_enabled(context: Dict[str, Any]) -> bool:
        integration = (context.get("integration") or {}) if isinstance(context, dict) else {}
        value = integration.get("conversationStyle")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _language_instruction(self) -> str:
        """Return system instruction to respond in response_language, or empty string if English/default."""
        if not self.response_language:
            return ""
        name = (self.response_language or "").strip()
        if name.lower() in ("en", "english", ""):
            return ""
        return f"Respond only in {name}. All your replies must be in this language."

    def set_profession(self, profession: str):
        """Set profession for responses"""
        self.profession = profession or self.profession

    def set_channel(self, channel: str):
        """Set current channel (web, whatsapp, messenger, voice)"""
        normalized = (channel or "web").lower()
        if normalized not in {"web", "whatsapp", "messenger", "instagram", "voice"}:
            normalized = "web"
        logger.info(f"ResponseGenerator channel set to {normalized}")
        self.channel = normalized

    @staticmethod
    def _format_voice_list(options: List[str]) -> str:
        """Format options for voice channel (natural language list)"""
        cleaned = [opt.strip() for opt in options if opt and opt.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} or {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", or {cleaned[-1]}"

    @staticmethod
    def _filter_services_by_lead_type(
        service_plans: List[Any],
        lead_types: List[Dict[str, Any]],
        collected_lead_type: Optional[str]
    ) -> Optional[List[str]]:
        """Filter service plans by lead type's relevantServicePlans. Returns None if no filtering needed."""
        if not collected_lead_type or not lead_types:
            return None
        # Find the lead type that was selected
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            if (lt.get("value") or "").strip().lower() != collected_lead_type.strip().lower():
                continue
            # Found matching lead type - check for relevantServicePlans
            relevant = lt.get("relevantServicePlans")
            if not relevant or not isinstance(relevant, list):
                return None  # No filtering configured
            requested = [str(s).strip() for s in relevant if s and str(s).strip()]
            if not requested:
                return None

            # Build a case-insensitive lookup from available service plans.
            available: Dict[str, str] = {}
            for plan in service_plans:
                if isinstance(plan, dict):
                    name = (plan.get("question") or plan.get("name") or plan.get("title") or "").strip()
                else:
                    name = str(plan).strip()
                if not name:
                    continue
                key = name.lower()
                if key not in available:
                    available[key] = name

            # Preserve the configured order of relevantServicePlans.
            filtered: List[str] = []
            for r in requested:
                key = r.lower()
                if key in available:
                    filtered.append(available[key])
            return filtered if filtered else None
        return None

    @staticmethod
    def _normalize_lead_option_for_voice(text: str) -> str:
        """Strip preference phrasing like 'I would like' for natural voice questions."""
        if not text:
            return ""

        cleaned = text.strip()
        lowered = cleaned.lower()
        prefixes = (
            "i would like",
            "i'd like",
            "i want",
        )

        for prefix in prefixes:
            if lowered.startswith(prefix):
                remainder = cleaned[len(prefix):].lstrip(" ,.")
                cleaned = remainder or cleaned
                break

        return cleaned.rstrip(" .?")

    async def _generate_empathy_prefix_for_lead_type(
        self,
        lead_type_value: Optional[str],
        lead_types: List[Dict[str, Any]],
    ) -> str:
        """Return a short empathy line tailored to selected lead type."""
        lead_value = (lead_type_value or "").strip().lower()
        lead_text = ""
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            if (lt.get("value") or "").strip().lower() == lead_value:
                base = (lt.get("text") or lt.get("value") or "").strip()
                emoji = str(lt.get("emoji") or "").strip()
                lead_text = f"{emoji} {base}".strip() if emoji else base
                break
        label = lead_text or (lead_type_value or "your request")
        fallback = f"Great choice - I am here to help with {label}."
        cache_key = f"{(lead_type_value or '').strip().lower()}|{(self.response_language or '').strip().lower()}|{self.channel}"
        if cache_key in self._empathy_prefix_cache:
            return self._empathy_prefix_cache[cache_key]

        if not self.client:
            self._empathy_prefix_cache[cache_key] = fallback
            return fallback

        system_prompt = (
            f"Write one short, warm empathy sentence for a customer who selected this lead type: '{label}'. "
            "Keep it under 14 words. No emojis. No question marks."
        )
        if self._language_instruction():
            system_prompt += f" {self._language_instruction()}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=40,
                temperature=0.4,
            )
            line = (response.choices[0].message.content or "").strip()
            result = line or fallback
            self._empathy_prefix_cache[cache_key] = result
            return result
        except Exception:
            self._empathy_prefix_cache[cache_key] = fallback
            return fallback

    def _display_label_for_lead_type_value(
        self, lead_type_value: Optional[str], lead_types: List[Dict[str, Any]]
    ) -> str:
        """Human-facing label (emoji + text) for a lead type value, for LLM grounding."""
        lead_value = (lead_type_value or "").strip().lower()
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            if (lt.get("value") or "").strip().lower() == lead_value:
                base = (lt.get("text") or lt.get("value") or "").strip()
                emoji = str(lt.get("emoji") or "").strip()
                return f"{emoji} {base}".strip() if emoji else base
        return (lead_type_value or "").strip() or "their current request"

    @staticmethod
    def _is_booking_lead_value(lead_type_value: Optional[str]) -> bool:
        lt = (lead_type_value or "").lower()
        if not lt:
            return False
        return any(k in lt for k in ("book", "appointment", "treatment"))

    def _resolve_session_lead_type_value(
        self,
        context: Dict[str, Any],
        flow_controller: Optional[FlowController],
    ) -> Optional[str]:
        if flow_controller is not None:
            v = flow_controller.collected_data.get("leadType")
            if (v or "").strip():
                return str(v).strip()
        v = context.get("_session_lead_type_value")
        return str(v).strip() if (v or "").strip() else None

    def _lead_path_alignment_for_llm(
        self,
        context: Dict[str, Any],
        flow_controller: Optional[FlowController],
        *,
        conversation_state: Optional[ConversationState] = None,
        lead_type_value_override: Optional[str] = None,
    ) -> str:
        """
        Ground the model on the CURRENT lead type so mid-flow switches do not leak prior-path
        wording (e.g. 'booking process' after switching to feedback). Used for any LLM turn
        after a lead type is known.
        """
        lt_val = (lead_type_value_override or "").strip() or self._resolve_session_lead_type_value(
            context, flow_controller
        )
        if not lt_val:
            return ""
        if conversation_state is not None and conversation_state in (
            ConversationState.GREETING,
            ConversationState.LEAD_TYPE_SELECTION,
        ):
            return ""
        lead_types = context.get("lead_types") or []
        label = self._display_label_for_lead_type_value(lt_val, lead_types)
        is_booking = self._is_booking_lead_value(lt_val)
        if conversation_state in (
            ConversationState.APPOINTMENT_OFFER,
            ConversationState.CALENDAR_BOOKING,
            ConversationState.APPOINTMENT_CONFIRMATION,
        ):
            is_booking = True
        lines = [
            "CRITICAL — CURRENT USER PATH (your entire reply must match this for this turn):",
            f"The user is on: {label} (internal: {lt_val}).",
            "Frame every sentence for THIS path only. Ignore earlier assistant/user lines that referred to a different path they already left.",
        ]
        if not is_booking:
            lines.append(
                "Do not mention booking, appointments, treatments, or a 'booking process' unless the CURRENT path above is clearly booking/scheduling."
            )
        else:
            lines.append("Scheduling or booking language is appropriate when it matches this path.")
        return "\n".join(lines)

    
    async def _classify_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Classify user intent using LLM with structured output.
        Returns: {
            'is_question': bool,
            'question_type': str,  # 'pricing', 'general_info', 'procedure_info', 'location_hours', 'other', 'not_question'
            'confidence': float  # 0.0 to 1.0
        }
        Raises exception if LLM is unavailable (no fallback - system requires LLM).
        """
        if not self.client:
            raise ValueError("LLM client not available - intent classification requires LLM")
        
        # Use JSON mode for structured output
        system_prompt = """You are an intent classifier. Analyze the user message and classify:
1. Whether it's a question (seeking information, clarification, or explanation)
2. The type of question: pricing, general_info, procedure_info, location_hours, other, or not_question
3. Your confidence (0.0 to 1.0)

Respond ONLY with valid JSON in this exact format (no other text):
{
  "is_question": true/false,
  "question_type": "pricing|general_info|procedure_info|location_hours|other|not_question",
  "confidence": 0.0-1.0
}"""

        # Try with JSON mode first (for newer models like gpt-4o, gpt-4-turbo)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this message: '{user_message}'"}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=100,
                response_format={"type": "json_object"}  # JSON mode for structured output
            )
        except Exception as json_mode_error:
            # Fallback: model doesn't support JSON mode, use regular prompt
            logger.debug(f"JSON mode not supported, using prompt-based classification: {json_mode_error}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this message: '{user_message}'"}
                ],
                temperature=0.1,
                max_tokens=100
            )
        
        # Parse JSON response
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM for intent classification")
        
        # Extract JSON from response (in case model adds extra text)
        content_clean = content.strip()
        json_start = content_clean.find('{')
        json_end = content_clean.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            content_clean = content_clean[json_start:json_end]
        
        try:
            result = json.loads(content_clean)
            # Validate and normalize result
            is_question = result.get("is_question", False)
            question_type = result.get("question_type", "not_question")
            confidence = float(result.get("confidence", 0.5))
            
            # Normalize: if question_type is 'not_question', ensure is_question is False
            if question_type == "not_question":
                is_question = False
            
            result = {
                "is_question": is_question,
                "question_type": question_type,
                "confidence": max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            }
            logger.debug(f"Intent classified: {result}")
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from LLM: {e}, content: {content}")
    
    async def _classify_otp_intent(
        self, 
        user_message: str, 
        conversation_history: List[Dict[str, str]],
        current_email: Optional[str] = None,
        current_phone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify OTP-related intent (change email/phone, resend OTP).
        Returns: {
            'otp_intent': str,  # 'change_email', 'change_phone', 'resend_otp', 'enter_otp', 'other'
            'extracted_email': Optional[str],  # New email if change_email intent
            'extracted_phone': Optional[str],  # New phone if change_phone intent
            'confidence': float  # 0.0 to 1.0
        }
        """
        if not self.client:
            raise ValueError("LLM client not available - OTP intent classification requires LLM")
        
        # Extract email and phone from user message for context
        extractor = DataExtractor()
        extracted_email = extractor.extract_email(user_message)
        extracted_phone = extractor.extract_phone(user_message)
        
        # Build context about current state
        context_info = []
        if current_email:
            context_info.append(f"OTP was sent to email: {current_email}")
        if current_phone:
            context_info.append(f"OTP was sent to phone: {current_phone}")
        if extracted_email:
            context_info.append(f"User mentioned email: {extracted_email}")
        if extracted_phone:
            context_info.append(f"User mentioned phone: {extracted_phone}")
        
        context_text = "\n".join(context_info) if context_info else "No current contact information available"
        
        # Get recent conversation history for context
        recent_history = ""
        if conversation_history:
            recent_msgs = conversation_history[-5:]  # Last 5 messages
            recent_history = "\n".join([
                f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
                for msg in recent_msgs
            ])
        
        system_prompt = """You are an OTP intent classifier. Analyze the user message to determine their intent regarding OTP verification.

Possible intents:
1. change_email - User wants to change the email address to receive OTP (mentions different email or says "send to [email]", "wrong email", etc.)
2. change_phone - User wants to change the phone number to receive OTP (mentions different phone or says "send to [phone]", "wrong number", etc.)
3. resend_otp - User wants to resend OTP to the same contact (says "resend", "send again", "didn't receive", "send code again", etc.)
4. enter_otp - User is providing the OTP code (6-digit number)
5. other - Any other intent

Rules:
- If user mentions a different email/phone than the current one, it's ALWAYS change_email/change_phone
- If user says "send it to [email]" or "send to [email]" and mentions an email, it's change_email
- If user says "send it to [phone]" or "send to [phone]" and mentions a phone, it's change_phone
- If user mentions resending but doesn't specify a different contact, it's resend_otp
- If message contains a 6-digit number and user is clearly entering a code, it's enter_otp
- Examples of change_email: "send it to new@email.com", "wrong email, send to new@email.com", "can u send it to new@email.com", "use new@email.com instead"
- Examples of resend_otp: "resend code", "send again", "didn't receive", "send the code again"

Respond ONLY with valid JSON in this exact format (no other text):
{
  "otp_intent": "change_email|change_phone|resend_otp|enter_otp|other",
  "extracted_email": "email@example.com" or null,
  "extracted_phone": "+1234567890" or null,
  "confidence": 0.0-1.0
}"""

        user_prompt = f"""Context:
{context_text}

Recent conversation:
{recent_history}

User message: "{user_message}"

Classify the intent:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
        except Exception as json_mode_error:
            # Fallback: model doesn't support JSON mode
            logger.debug(f"JSON mode not supported for OTP intent, using prompt-based: {json_mode_error}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
        
        # Parse JSON response
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM for OTP intent classification")
        
        # Extract JSON from response
        content_clean = content.strip()
        json_start = content_clean.find('{')
        json_end = content_clean.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            content_clean = content_clean[json_start:json_end]
        
        try:
            result = json.loads(content_clean)
            otp_intent = result.get("otp_intent", "other")
            extracted_email = result.get("extracted_email")
            extracted_phone = result.get("extracted_phone")
            confidence = float(result.get("confidence", 0.5))
            
            # Use extracted email/phone from message if LLM didn't extract them
            if otp_intent == "change_email" and not extracted_email:
                extracted_email = extractor.extract_email(user_message)
            if otp_intent == "change_phone" and not extracted_phone:
                extracted_phone = extractor.extract_phone(user_message)
            
            result = {
                "otp_intent": otp_intent,
                "extracted_email": extracted_email,
                "extracted_phone": extracted_phone,
                "confidence": max(0.0, min(1.0, confidence))
            }
            logger.debug(f"OTP intent classified: {result}")
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from LLM for OTP intent: {e}, content: {content}")
    
    def _deterministic_lead_type_reprompt(self, context: Dict[str, Any]) -> str:
        """
        Short line + same <button> labels as the greeting. Avoids the LLM re-printing
        the entire custom greeting when a click/tap failed fuzzy matching.
        """
        lead_types = context.get("lead_types") or []
        parts: List[str] = []
        for lt in lead_types:
            if isinstance(lt, dict):
                text = str(lt.get("text") or lt.get("value") or "").strip()
                emoji = str(lt.get("emoji") or "").strip()
                label = f"{emoji} {text}".strip() if emoji else text
                if label:
                    parts.append(f"<button>{label}</button>")
        joined = " ".join(parts)
        if not joined.strip():
            return "Please reply with the number of your choice (1, 2, 3…) or tap an option again."
        return (
            "Please choose one of these options so we can continue. "
            f"{joined}"
        )

    def _deterministic_lead_type_reprompt_conversation(self, context: Dict[str, Any]) -> str:
        """Conversational mode: no <button> tags; list labels in plain language."""
        lead_types = context.get("lead_types") or []
        labels: List[str] = []
        for lt in lead_types:
            if isinstance(lt, dict):
                text = str(lt.get("text") or lt.get("value") or "").strip()
                emoji = str(lt.get("emoji") or "").strip()
                label = f"{emoji} {text}".strip() if emoji else text
                if label:
                    labels.append(label)
        if not labels:
            return "Could you tell me briefly what you need—booking, feedback, or general information?"
        if len(labels) == 1:
            opts = labels[0]
        elif len(labels) == 2:
            opts = f"{labels[0]} or {labels[1]}"
        else:
            opts = ", ".join(labels[:-1]) + f", or {labels[-1]}"
        return f"Which do you mean—{opts}?"

    async def generate_response(
        self,
        flow_controller: FlowController,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> str:
        """Generate response based on current state"""
        state = flow_controller.state
        # Resume blobs may omit `state`; FlowController then stays on default GREETING even
        # though the user already saw the greeting and should be picking a lead type.
        if state == ConversationState.GREETING:
            flow_controller.transition_to(ConversationState.LEAD_TYPE_SELECTION)
            state = flow_controller.state
        # Ground LLM replies on the active lead type (mid-flow switches update this every turn).
        context["_session_lead_type_value"] = flow_controller.collected_data.get("leadType")
        conversation_style_enabled = self._conversation_style_enabled(context) and self.channel != "voice"
        extractor = DataExtractor()
        validator = Validator()
        
        # Initialize or reuse workflow manager
        if flow_controller.workflow_manager is None:
            wm_context = dict(context)
            wm_context["api_base_url"] = self.api_base_url
            flow_controller.workflow_manager = WorkflowManager(wm_context)
        workflow_manager = flow_controller.workflow_manager
        
        # Extract data from user message (only extract what we need based on state)
        email = extractor.extract_email(user_message)
        phone = extractor.extract_phone(user_message)
        otp_code = extractor.extract_otp_code(user_message)
        
        # Only extract name when we're in NAME_COLLECTION state (prevents lead type text from being extracted as name)
        name = None
        if state == ConversationState.NAME_COLLECTION:
            name = extractor.extract_name(user_message, context.get("lead_types", []))
        
        # Handle OTP verification states - FIRST check for change/resend requests using intent classification
        if state == ConversationState.EMAIL_OTP_VERIFICATION:
            current_email = flow_controller.collected_data.get("leadEmail")
            
            # First check if it's a valid OTP code
            if otp_code and validator.is_valid_otp(otp_code):
                return "OTP_VERIFY_EMAIL:" + otp_code
            
            # Not a valid OTP - use intent classification to detect change/resend requests
            try:
                otp_intent_result = await self._classify_otp_intent(
                    user_message, 
                    conversation_history,
                    current_email=current_email
                )
                otp_intent = otp_intent_result.get("otp_intent", "other")
                
                if otp_intent == "change_email":
                    new_email = otp_intent_result.get("extracted_email")
                    if new_email:
                        return f"CHANGE_EMAIL_REQUESTED: {new_email}"
                    else:
                        return "CHANGE_EMAIL_REQUESTED"
                elif otp_intent == "resend_otp":
                    return "RETRY_OTP_REQUESTED"
                # If intent is "enter_otp" but no valid OTP found, or "other", continue to normal flow
            except Exception as e:
                logger.error(f"Error in OTP intent classification: {e}")
                # Continue to normal flow if classification fails
        
        if state == ConversationState.PHONE_OTP_VERIFICATION:
            current_phone = flow_controller.collected_data.get("leadPhoneNumber")
            
            # First check if it's a valid OTP code
            if otp_code and validator.is_valid_otp(otp_code):
                return "OTP_VERIFY_PHONE:" + otp_code
            
            # Not a valid OTP - use intent classification to detect change/resend requests
            try:
                otp_intent_result = await self._classify_otp_intent(
                    user_message,
                    conversation_history,
                    current_phone=current_phone
                )
                otp_intent = otp_intent_result.get("otp_intent", "other")
                
                if otp_intent == "change_phone":
                    new_phone = otp_intent_result.get("extracted_phone")
                    if new_phone:
                        return f"CHANGE_PHONE_REQUESTED: {new_phone}"
                    else:
                        return "CHANGE_PHONE_REQUESTED"
                elif otp_intent == "resend_otp":
                    return "RETRY_OTP_REQUESTED"
                # If intent is "enter_otp" but no valid OTP found, or "other", continue to normal flow
            except Exception as e:
                logger.error(f"Error in OTP intent classification: {e}")
                # Continue to normal flow if classification fails
        
        # Lead-type menu (non-conversational): never treat input as an FAQ "question".
        # The LLM classifier often marks button labels as questions; RAG then repeats the custom greeting.
        lead_type_probe = None
        if state == ConversationState.LEAD_TYPE_SELECTION:
            _lts = context.get("lead_types") or []
            lead_type_probe = resolve_lead_type(
                user_message, _lts, LeadTypeResolutionMode.LEAD_SELECTION
            )

        # Classify user intent (question detection using LLM)
        if state == ConversationState.LEAD_TYPE_SELECTION and not conversation_style_enabled:
            has_question = False
            question_type = "not_question"
        elif lead_type_probe:
            has_question = False
            question_type = "not_question"
        else:
            try:
                intent = await self._classify_intent(user_message)
                has_question = intent.get("is_question", False)
                question_type = intent.get("question_type", "not_question")

                logger.debug(
                    f"Intent classification: is_question={has_question}, type={question_type}, "
                    f"confidence={intent.get('confidence', 0.0)}"
                )
            except Exception as e:
                # If intent classification fails, log error and assume it's not a question
                logger.error(f"Intent classification failed: {e}, treating as non-question")
                has_question = False
                question_type = "not_question"

        app_industry = str((context.get("app") or {}).get("industry") or "").strip()

        async def _answer_if_relevant_question(query: str) -> Optional[str]:
            """
            Conversational-mode guard:
            - answer briefly when question is relevant to app industry / services / FAQs
            - for industry-relevant questions not in custom data, answer from general knowledge
            - return None only when question is a non-question (e.g. pure data input)
            """
            if not (conversation_style_enabled and has_question and query.strip()):
                return None
            if question_type == "not_question":
                return None

            scoped_query = query.strip()
            if app_industry:
                scoped_query = f"{query.strip()} (Industry: {app_industry})"

            rag_context = await self._get_rag_context(scoped_query, context, is_question=True)

            context_l = (rag_context or "").lower()
            # Relevance markers from our domain context payloads (FAQ/services/lead options/workflows)
            relevance_markers = (
                "[source: faq",
                "[source: service",
                "[source: lead_type",
                "faq",
                "service",
                "workflow",
                "lead type",
            )
            has_domain_signal = any(marker in context_l for marker in relevance_markers)

            # Industry-relevant question types that should be answered even without custom data
            industry_question_types = {"pricing", "general_info", "procedure_info", "location_hours", "other"}

            if not has_domain_signal and question_type not in industry_question_types:
                # Truly off-topic: no custom data and not an industry-type question — let the LLM redirect gracefully
                answer = await self._generate_question_response(
                    query, rag_context or "", conversation_history, context, flow_controller=flow_controller
                )
                return answer if answer else None

            answer = await self._generate_question_response(
                query, rag_context or "", conversation_history, context, flow_controller=flow_controller
            )
            if not answer:
                return None

            return answer
        
        # Handle data collection states - extract and validate before AI generation
        if state == ConversationState.LEAD_TYPE_SELECTION:
            lead_types_list = context.get("lead_types", [])
            lead_type = resolve_lead_type(
                user_message, lead_types_list, LeadTypeResolutionMode.LEAD_SELECTION
            )
            # If no match (e.g. user wrote in Urdu/other language), translate to English and retry
            if not lead_type and self.client and self.model and user_message.strip():
                from ..utils.translation_utils import translate_to_english
                try:
                    translated = await translate_to_english(self.client, self.model, user_message)
                    if translated and translated.strip().lower() != user_message.strip().lower():
                        lead_type = resolve_lead_type(
                            translated.strip(),
                            lead_types_list,
                            LeadTypeResolutionMode.LEAD_SELECTION,
                        )
                except Exception as e:
                    logger.debug("Translate-to-English for lead type match failed: %s", e)
            if lead_type:
                logger.info(f"Matched lead type: {lead_type.get('text')} (value: {lead_type.get('value')})")
                flow_controller.update_collected_data("leadType", lead_type.get("value"))
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                # Personal info first: after lead type, immediately collect name/email/phone
                if flow_controller.state == ConversationState.NAME_COLLECTION:
                    name_prompt = await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
                    empathy_line = await self._generate_empathy_prefix_for_lead_type(
                        flow_controller.collected_data.get("leadType"),
                        context.get("lead_types", []),
                    )
                    return f"{empathy_line} {name_prompt}"
                if flow_controller.state in (
                    ConversationState.EMAIL_COLLECTION,
                    ConversationState.PHONE_COLLECTION,
                ):
                    return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
                
                # Check if there's only one service for this lead type - if so, auto-select it
                service_plans = context.get("service_plans", [])
                lead_types = context.get("lead_types", [])
                filtered_services = self._filter_services_by_lead_type(service_plans, lead_types, lead_type.get("value"))
                
                if filtered_services is not None:
                    all_services = filtered_services
                else:
                    all_services = []
                    for plan in service_plans:
                        if isinstance(plan, dict):
                            plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                            if plan_name:
                                all_services.append(plan_name)
                        else:
                            all_services.append(str(plan))

                # Conversational shortcut:
                # If lead type is matched and the same message already implies a service (e.g., "I want to place an order"),
                # immediately continue into service handling/workflow instead of asking service intent again.
                if conversation_style_enabled and all_services:
                    service_from_same_message = extractor.match_service(user_message, all_services)
                    if service_from_same_message:
                        logger.info(
                            "Lead+service inferred from same message in conversational mode. lead_type=%s, service=%s",
                            lead_type.get("value"), service_from_same_message
                        )
                        return await self.generate_response(flow_controller, user_message, conversation_history, context)
                
                # If only one service exists, auto-select it and skip service selection
                if len(all_services) == 1:
                    single_service = all_services[0]
                    logger.info(f"Only one service available for lead type '{lead_type.get('value')}': '{single_service}' - auto-selecting")
                    flow_controller.update_collected_data("serviceType", single_service)
                    
                    # Check for workflows
                    if flow_controller.workflow_manager is None:
                        wm_context2 = dict(context)
                        wm_context2["api_base_url"] = self.api_base_url
                        flow_controller.workflow_manager = WorkflowManager(wm_context2)
                    workflow_manager = flow_controller.workflow_manager
                    
                    workflow_started = False
                    if workflow_manager.start_workflow_for_service(single_service):
                        # Start workflow questions
                        flow_controller.transition_to(ConversationState.WORKFLOW_QUESTION)
                        workflow_started = True
                        logger.info(f"✓ Started workflow for auto-selected service '{single_service}' - transitioning to WORKFLOW_QUESTION state")
                        current_question = workflow_manager.get_current_question()
                        if current_question:
                            question_text = workflow_manager.format_question_with_options(current_question)
                            logger.info(f"✓ First workflow question: '{current_question.get('question', '')}'")
                            # If user asked a question, answer it briefly then ask workflow question
                            if has_question and not conversation_style_enabled:
                                try:
                                    rag_context = await self._get_rag_context(f"{single_service} {user_message}", context, is_question=True)
                                    if rag_context and self.client:
                                        brief_system = f"You are a {self.profession} assistant. Answer the question briefly in 1-2 sentences."
                                        if self._language_instruction():
                                            brief_system += "\n\n" + self._language_instruction()
                                        response = await self.client.chat.completions.create(
                                            model=self.model,
                                            messages=[
                                                {"role": "system", "content": brief_system},
                                                {"role": "user", "content": f"Context: {rag_context}\n\nQuestion: {user_message}"}
                                            ],
                                            max_tokens=100,
                                            temperature=0.3
                                        )
                                        brief_answer = (response.choices[0].message.content or "").strip()
                                        if brief_answer:
                                            logger.info(f"✓ Answering question then asking workflow question")
                                            return f"{brief_answer}\n\n{question_text}"
                                except Exception as e:
                                    logger.warning(f"Failed to generate brief answer for question: {e}")
                            logger.info(f"✓ Returning workflow question (no user question to answer first)")
                            return question_text
                        else:
                            logger.warning(f"Workflow started but no questions found - continuing to next state")
                            workflow_manager.reset()
                            flow_controller.transition_to(flow_controller.get_next_state())
                    else:
                        logger.info(f"No workflow found for auto-selected service '{single_service}' - continuing to next state")
                        workflow_manager.reset()
                    
                    # If workflow was NOT started, continue to next state
                    if not workflow_started:
                        logger.info(f"No workflow started - transitioning to next state")
                        workflow_manager.reset()
                        flow_controller.transition_to(flow_controller.get_next_state())
                    
                    logger.info(f"Transitioned to state: {flow_controller.state.value}")
                    
                    # Check if user asked a question (only if no workflow)
                    if has_question and flow_controller.state != ConversationState.WORKFLOW_QUESTION:
                        if conversation_style_enabled:
                            # In conversational mode, answer only if relevant to app industry/context.
                            answer = await _answer_if_relevant_question(user_message)
                            next_prompt = await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
                            return f"{answer}\n\n{next_prompt}" if answer else next_prompt
                        rag_context = await self._get_rag_context(f"{single_service} {user_message}", context, is_question=True)
                        return await self._generate_data_collected_with_question_response(
                            "service", single_service, rag_context,
                            "name", conversation_history, context,
                            flow_controller=flow_controller,
                        )
                    elif flow_controller.state != ConversationState.WORKFLOW_QUESTION:
                        # Just acknowledge and move to name collection
                        return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
                else:
                    # Multiple services - show selection as before
                    logger.info(f"Multiple services available ({len(all_services)}) - showing service selection")
                    # Check if user asked a question along with lead type selection
                    if has_question:
                        if conversation_style_enabled:
                            answer = await _answer_if_relevant_question(user_message)
                            next_prompt = await self._generate_service_selection_response(
                                conversation_history, context,
                                collected_lead_type=lead_type.get("value")
                            )
                            return f"{answer}\n\n{next_prompt}" if answer else next_prompt
                        rag_context = await self._get_rag_context(user_message, context, is_question=True)
                        return await self._generate_data_collected_with_question_response(
                            "lead type", lead_type.get("text"), rag_context, 
                            "service selection", conversation_history, context,
                            collected_lead_type=lead_type.get("value"),
                            flow_controller=flow_controller,
                        )
                    else:
                        # Explicitly include services (filtered by lead type when configured)
                        return await self._generate_service_selection_response(
                            conversation_history, context,
                            collected_lead_type=lead_type.get("value")
                        )
            else:
                logger.warning(f"No lead type matched for user input: '{user_message}'. Available lead types: {[lt.get('text') for lt in context.get('lead_types', [])]}")
                # No match: avoid LLM _generate_state_response here — it often repeats the full custom greeting.
                if conversation_style_enabled:
                    answer = await _answer_if_relevant_question(user_message)
                    next_prompt = self._deterministic_lead_type_reprompt_conversation(context)
                    return f"{answer}\n\n{next_prompt}" if answer else next_prompt
                if has_question and self.client:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    try:
                        qa = await self._generate_question_response(
                            user_message, rag_context or "", conversation_history, context, flow_controller=flow_controller
                        )
                        if qa and self.channel != "voice":
                            return f"{qa}\n\n{self._deterministic_lead_type_reprompt(context)}"
                        if qa:
                            return f"{qa}\n\nPlease say which option you want, or reply with a number."
                    except Exception as e:
                        logger.warning("Lead-type phase: question response failed: %s", e)
                if self.channel != "voice":
                    return self._deterministic_lead_type_reprompt(context)
                return await self._generate_state_response(state, "", conversation_history, context, flow_controller=flow_controller)
        
        if state == ConversationState.SERVICE_SELECTION:
            service_plans = context.get("service_plans", [])
            lead_types = context.get("lead_types", [])
            collected_lead_type = flow_controller.collected_data.get("leadType")
            
            # Filter services by lead type's relevantServicePlans when configured
            filtered_names = self._filter_services_by_lead_type(service_plans, lead_types, collected_lead_type)
            if filtered_names is not None:
                all_service_options = filtered_names
                logger.info(f"SERVICE_SELECTION: Filtered to {len(filtered_names)} services for lead type '{collected_lead_type}'")
            else:
                all_service_options = []
                for plan in service_plans:
                    if isinstance(plan, dict):
                        plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                        if plan_name:
                            all_service_options.append(plan_name)
                    else:
                        all_service_options.append(str(plan))
                logger.info(f"SERVICE_SELECTION: No filtering - showing all {len(all_service_options)} services")
            
            # Numeric selection: "1", "2" = first, second service in the list shown to user
            service = None
            if user_message.strip().isdigit():
                num = int(user_message.strip())
                if 1 <= num <= len(all_service_options):
                    service = all_service_options[num - 1]
                    logger.info(f"SERVICE_SELECTION: Matched service by number #{num}: '{service}'")
            if not service:
                service = extractor.match_service(user_message, all_service_options)
            
            # Find the exact service plan name that was matched (for workflow detection)
            matched_service_name = None
            if service:
                # Find the exact service plan name from the list (case-insensitive match)
                for plan_name in all_service_options:
                    if plan_name.lower() == service.lower():
                        matched_service_name = plan_name
                        break
                # If no exact match found, use the service as-is
                if not matched_service_name:
                    matched_service_name = service
            
            # If no match, accept user input as service type (user can choose any service)
            if not service:
                # Check if it's a question - if so, handle it but stay in service selection
                if has_question:
                    logger.info(f"User asked a question about services: '{user_message}'")
                    if conversation_style_enabled:
                        answer = await _answer_if_relevant_question(user_message)
                        next_prompt = await self._generate_service_selection_response(
                            conversation_history, context,
                            collected_lead_type=collected_lead_type
                        )
                        return f"{answer}\n\n{next_prompt}" if answer else next_prompt
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context, flow_controller=flow_controller)
                
                # Accept user input as service type even if not in configured service options
                service = user_message.strip()
                logger.info(f"Accepted user input as service type (not in configured service options): '{service}'")
            
            if service:
                logger.info(f"Matched/selected service: {service}")
                flow_controller.update_collected_data("serviceType", service)
                
                # ALWAYS check for workflows first (even if user asked a question)
                # Use the exact service plan name for workflow detection
                workflow_started = False
                if matched_service_name and matched_service_name in all_service_options:
                    logger.info(f"Checking for workflows for service: '{matched_service_name}' (matched from service: '{service}')")
                    if workflow_manager.start_workflow_for_service(matched_service_name):
                        # Start workflow questions
                        flow_controller.transition_to(ConversationState.WORKFLOW_QUESTION)
                        workflow_started = True
                        logger.info(f"✓ Started workflow for service '{matched_service_name}' - transitioning to WORKFLOW_QUESTION state")
                        current_question = workflow_manager.get_current_question()
                        if current_question:
                            question_text = workflow_manager.format_question_with_options(current_question) or ""
                            logger.info(f"✓ First workflow question: '{current_question.get('question', '')}'")
                            # If user asked a question, answer it briefly then ask workflow question
                            if has_question and not conversation_style_enabled:
                                try:
                                    rag_context = await self._get_rag_context(f"{service} {user_message}", context, is_question=True)
                                    if rag_context and self.client:
                                        brief_system = f"You are a {self.profession} assistant. Answer the question briefly in 1-2 sentences."
                                        if self._language_instruction():
                                            brief_system += "\n\n" + self._language_instruction()
                                        response = await self.client.chat.completions.create(
                                            model=self.model,
                                            messages=[
                                                {"role": "system", "content": brief_system},
                                                {"role": "user", "content": f"Context: {rag_context}\n\nQuestion: {user_message}"}
                                            ],
                                            max_tokens=100,
                                            temperature=0.3
                                        )
                                        brief_answer = (response.choices[0].message.content or "").strip()
                                        if brief_answer:
                                            logger.info(f"✓ Answering question then asking workflow question")
                                            return f"{brief_answer}\n\n{question_text}"
                                except Exception as e:
                                    logger.warning(f"Failed to generate brief answer for question: {e}")
                            logger.info(f"✓ Returning workflow question (no user question to answer first)")
                            return question_text
                        else:
                            # No questions found, continue to next state
                            logger.warning(f"Workflow started but no questions found - continuing to next state")
                            workflow_manager.reset()
                            flow_controller.transition_to(flow_controller.get_next_state())
                    else:
                        # No workflow found
                        logger.info(f"No workflow found for service '{matched_service_name}' - continuing to next state")
                        workflow_manager.reset()
                
                # If workflow was NOT started, continue to next state
                if not workflow_started:
                    logger.info(f"No workflow started - transitioning to next state")
                    workflow_manager.reset()
                    flow_controller.transition_to(flow_controller.get_next_state())
                
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                
                # Check if user asked a question along with service selection (only if no workflow)
                if has_question and flow_controller.state != ConversationState.WORKFLOW_QUESTION:
                    if conversation_style_enabled:
                        answer = await _answer_if_relevant_question(user_message)
                        if flow_controller.state == ConversationState.SERVICE_SELECTION:
                            next_prompt = await self._generate_service_selection_response(
                                conversation_history,
                                context,
                                collected_lead_type=flow_controller.collected_data.get("leadType"),
                            )
                        else:
                            next_prompt = await self._generate_state_response(
                                flow_controller.state, "", conversation_history, context, flow_controller=flow_controller
                            )
                        return f"{answer}\n\n{next_prompt}" if answer else next_prompt
                    # Get RAG context for the question (pricing, info about the service)
                    rag_context = await self._get_rag_context(f"{service} {user_message}", context, is_question=True)
                    # Generate response that answers question AND asks for next step
                    return await self._generate_data_collected_with_question_response(
                        "service", service, rag_context,
                        "workflow", conversation_history, context,
                        flow_controller=flow_controller,
                    )
                elif flow_controller.state != ConversationState.WORKFLOW_QUESTION:
                    # Just acknowledge and move to name collection - no RAG needed
                    return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
            else:
                # This shouldn't happen now, but keep as fallback
                logger.warning(f"Could not determine service from user input: '{user_message}'. Available service options: {all_service_options}")
                # No match - use AI to handle questions, but ensure it stays in service selection
                rag_context = await self._get_rag_context(user_message if has_question else "service selection", context, is_question=has_question)
                return await self._generate_state_response(state, rag_context, conversation_history, context, flow_controller=flow_controller)
        
        if state == ConversationState.WORKFLOW_QUESTION:
            # Non-booking paths can reach WORKFLOW_QUESTION after personal info collection
            # without passing through SERVICE_SELECTION. In that case, honor lead-type rules
            # by resolving services first, then start the attached workflow.
            if not flow_controller.collected_data.get("serviceType"):
                service_plans = context.get("service_plans", [])
                lead_types = context.get("lead_types", [])
                collected_lead_type = flow_controller.collected_data.get("leadType")
                filtered_names = self._filter_services_by_lead_type(
                    service_plans, lead_types, collected_lead_type
                )
                if filtered_names is not None:
                    all_service_options = filtered_names
                else:
                    all_service_options = []
                    for plan in service_plans:
                        if isinstance(plan, dict):
                            plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                            if plan_name:
                                all_service_options.append(plan_name)
                        else:
                            all_service_options.append(str(plan))

                if len(all_service_options) == 1:
                    single_service = all_service_options[0]
                    flow_controller.update_collected_data("serviceType", single_service)
                    if workflow_manager.start_workflow_for_service(single_service):
                        current_question = workflow_manager.get_current_question()
                        if current_question:
                            return workflow_manager.format_question_with_options(current_question) or ""
                        workflow_manager.reset()
                    logger.info(
                        "WORKFLOW_QUESTION reached without active workflow; auto-selected single service '%s' but no workflow started",
                        single_service,
                    )
                elif len(all_service_options) > 1:
                    flow_controller.transition_to(ConversationState.SERVICE_SELECTION)
                    return await self._generate_service_selection_response(
                        conversation_history,
                        context,
                        collected_lead_type=collected_lead_type,
                    )

            # Graceful switch: if user selects another service while in workflow, restart service flow.
            service_plans = context.get("service_plans", [])
            all_services = []
            for plan in service_plans:
                if isinstance(plan, dict):
                    nm = plan.get("question", plan.get("name", plan.get("title", "")))
                    if nm:
                        all_services.append(nm)
                else:
                    all_services.append(str(plan))
            switched_service = extractor.match_service(user_message, all_services)
            if switched_service and (flow_controller.collected_data.get("serviceType") or "").lower() != switched_service.lower():
                flow_controller.update_collected_data("serviceType", switched_service)
                flow_controller.reset_service_flow()
                return f"Switching to {switched_service}. " + await self._generate_service_selection_response(
                    conversation_history, context, collected_lead_type=flow_controller.collected_data.get("leadType")
                )
            # Handle workflow question
            current_question = workflow_manager.get_current_question()
            if not current_question:
                # Workflow complete, store answers and move to next state
                workflow_answers = workflow_manager.get_workflow_answers()
                flow_controller.update_collected_data("workflowAnswers", workflow_answers)
                workflow_manager.reset()
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Workflow complete. Transitioned to state: {flow_controller.state.value}")
                return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
            
            # Check if user is asking a question instead of answering the workflow question
            if has_question:
                current_question_formatted = workflow_manager.format_question_with_options(current_question) or ""
                if conversation_style_enabled:
                    answer = await _answer_if_relevant_question(user_message)
                    return f"{answer}\n\n{current_question_formatted}" if answer else current_question_formatted
                logger.info(f"User asked a question during workflow: '{user_message}'. Answering it first.")
                rag_context = await self._get_rag_context(user_message, context, is_question=True)
                if rag_context and self.client:
                    try:
                        answer = await self._generate_question_response(
                            user_message, rag_context, conversation_history, context, flow_controller=flow_controller
                        )
                        return f"{answer}\n\n{current_question_formatted}"
                    except Exception as e:
                        logger.warning(f"Failed to generate answer for question: {e}")
                        return current_question_formatted
                else:
                    return current_question_formatted
            
            # Not a question - treat as workflow answer
            has_more = workflow_manager.record_answer(user_message)
            
            if has_more:
                next_question = workflow_manager.get_current_question()
                if next_question:
                    return workflow_manager.format_question_with_options(next_question) or ""
                else:
                    workflow_answers = workflow_manager.get_workflow_answers()
                    flow_controller.update_collected_data("workflowAnswers", workflow_answers)
                    workflow_manager.reset()
                    flow_controller.transition_to(flow_controller.get_next_state())
                    logger.info(f"Workflow complete. Transitioned to state: {flow_controller.state.value}")
                    if flow_controller.state == ConversationState.CALENDAR_BOOKING:
                        return "BOOK_APPOINTMENT_REQUESTED"
                    return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
            else:
                workflow_answers = workflow_manager.get_workflow_answers()
                flow_controller.update_collected_data("workflowAnswers", workflow_answers)
                workflow_manager.reset()
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Workflow complete. Transitioned to state: {flow_controller.state.value}")
                if flow_controller.state == ConversationState.CALENDAR_BOOKING:
                    return "BOOK_APPOINTMENT_REQUESTED"
                return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
        
        if state == ConversationState.NAME_COLLECTION:
            if name and validator.is_valid_name(name):
                flow_controller.update_collected_data("leadName", name)
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                
                # Check if user asked a question along with name
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    next_step = "service selection" if flow_controller.state == ConversationState.SERVICE_SELECTION else (
                        "email" if flow_controller.state == ConversationState.EMAIL_COLLECTION else (
                            "phone" if flow_controller.state == ConversationState.PHONE_COLLECTION else "name"
                        )
                    )
                    return await self._generate_data_collected_with_question_response(
                        "name", name, rag_context,
                        next_step, conversation_history, context,
                        collected_lead_type=flow_controller.collected_data.get("leadType"),
                        flow_controller=flow_controller,
                    )
                else:
                    # No question - just move to email collection
                    if flow_controller.state == ConversationState.SERVICE_SELECTION:
                        return await self._generate_service_selection_response(
                            conversation_history,
                            context,
                            collected_lead_type=flow_controller.collected_data.get("leadType"),
                        )
                    return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
            else:
                # Name not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context, flow_controller=flow_controller)
                else:
                    # Just ask for name again - no RAG needed
                    return await self._generate_state_response(state, "", conversation_history, context, flow_controller=flow_controller)
        
        if state == ConversationState.EMAIL_COLLECTION:
            if email and validator.is_valid_email(email):
                flow_controller.update_collected_data("leadEmail", email)
                
                # Check if user asked a question along with email
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    # Answer question and acknowledge email, then proceed
                    if flow_controller.validate_email:
                        # Answer question, acknowledge email, then trigger OTP sending
                        answer = await self._generate_question_response(
                            user_message, rag_context, conversation_history, context, flow_controller=flow_controller
                        )
                        # Return in format that main.py can handle: answer + SEND_EMAIL marker
                        # main.py will send answer first, then handle SEND_EMAIL
                        return f"{answer}|||SEND_EMAIL:{email}"
                    else:
                        flow_controller.transition_to(flow_controller.get_next_state())
                        logger.info(f"Transitioned to state: {flow_controller.state.value}")
                        if flow_controller.can_generate_json():
                            # Answer question then generate JSON
                            answer = await self._generate_question_response(
                                user_message, rag_context, conversation_history, context, flow_controller=flow_controller
                            )
                            json_data = await self._generate_json(flow_controller, conversation_history)
                            return f"{answer}\n\n{json_data}"
                        else:
                            next_step = "service selection" if flow_controller.state == ConversationState.SERVICE_SELECTION else (
                                "phone" if flow_controller.state == ConversationState.PHONE_COLLECTION else "email"
                            )
                            return await self._generate_data_collected_with_question_response(
                                "email", email, rag_context,
                                next_step, conversation_history, context,
                                collected_lead_type=flow_controller.collected_data.get("leadType"),
                                flow_controller=flow_controller,
                            )
                else:
                    # No question, proceed normally
                    if flow_controller.validate_email:
                        return "SEND_EMAIL:" + email
                    else:
                        flow_controller.transition_to(flow_controller.get_next_state())
                        # Check if we can generate JSON (WhatsApp) or need phone
                        if flow_controller.can_generate_json():
                            return await self._generate_json(flow_controller, conversation_history)
                        else:
                            # Ensure service selection keeps button options in non-conversational channels.
                            if flow_controller.state == ConversationState.SERVICE_SELECTION:
                                return await self._generate_service_selection_response(
                                    conversation_history,
                                    context,
                                    collected_lead_type=flow_controller.collected_data.get("leadType"),
                                )
                            if flow_controller.state == ConversationState.WORKFLOW_QUESTION:
                                # Re-enter main flow so non-booking lead rules can auto-select
                                # service and start attached workflow after contact capture.
                                return await self.generate_response(
                                    flow_controller, "", conversation_history, context
                                )
                            return await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
            else:
                # Email not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context, flow_controller=flow_controller)
                else:
                    # No RAG needed for standard email collection prompt
                    return await self._generate_state_response(state, "", conversation_history, context, flow_controller=flow_controller)
        
        if state == ConversationState.PHONE_COLLECTION:
            # Graceful email correction during phone step:
            # If the user provides/updates email before phone, accept latest email.
            # - With email verification enabled: restart email OTP on the new email.
            # - With email verification disabled: keep moving on phone collection.
            if email and validator.is_valid_email(email) and not (phone and validator.is_valid_phone(phone)):
                flow_controller.update_collected_data("leadEmail", email)
                if flow_controller.validate_email:
                    # Force re-verification for the updated email (latest email wins)
                    flow_controller.otp_state["email_sent"] = False
                    flow_controller.otp_state["email_verified"] = False
                    if has_question:
                        rag_context = await self._get_rag_context(user_message, context, is_question=True)
                        answer = await self._generate_question_response(
                            user_message, rag_context, conversation_history, context, flow_controller=flow_controller
                        )
                        return f"{answer}|||SEND_EMAIL:{email}"
                    return "SEND_EMAIL:" + email

                # Email verification disabled: accept latest email and continue phone step
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    answer = await self._generate_question_response(
                        user_message, rag_context, conversation_history, context, flow_controller=flow_controller
                    )
                    phone_prompt = await self._generate_state_response(
                        ConversationState.PHONE_COLLECTION, "", conversation_history, context, flow_controller=flow_controller
                    )
                    return f"{answer}\n\nThanks — I have updated your email to {email}.\n\n{phone_prompt}"
                return f"Thanks — I have updated your email to {email}. Please share your phone number to continue."

            if phone and validator.is_valid_phone(phone):
                flow_controller.update_collected_data("leadPhoneNumber", phone)
                
                # Check if user asked a question along with phone
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    if flow_controller.validate_phone:
                        # Answer question, acknowledge phone, then trigger OTP sending
                        answer = await self._generate_question_response(
                            user_message, rag_context, conversation_history, context, flow_controller=flow_controller
                        )
                        # Return in format that main.py can handle: answer + SEND_PHONE marker
                        return f"{answer}|||SEND_PHONE:{phone}"
                    else:
                        flow_controller.transition_to(flow_controller.get_next_state())
                        logger.info(f"Transitioned to state: {flow_controller.state.value}")
                        # Answer question and continue flow; only finalize when complete.
                        answer = await self._generate_question_response(
                            user_message, rag_context, conversation_history, context, flow_controller=flow_controller
                        )
                        if flow_controller.can_generate_json():
                            json_data = await self._generate_json(flow_controller, conversation_history)
                            return f"{answer}\n\n{json_data}"
                        next_prompt = await self._generate_state_response(flow_controller.state, "", conversation_history, context, flow_controller=flow_controller)
                        return f"{answer}\n\n{next_prompt}" if answer else next_prompt
                else:
                    # No question, proceed normally
                    if flow_controller.validate_phone:
                        return "SEND_PHONE:" + phone
                    else:
                        flow_controller.transition_to(flow_controller.get_next_state())
                        # Only generate JSON when all required fields are complete.
                        # For booking lead types, next_state is SERVICE_SELECTION, so we must
                        # continue the guided flow (service -> calendar) instead of finalizing.
                        if flow_controller.can_generate_json():
                            return await self._generate_json(flow_controller, conversation_history)
                        if flow_controller.state == ConversationState.SERVICE_SELECTION:
                            return await self._generate_service_selection_response(
                                conversation_history,
                                context,
                                collected_lead_type=flow_controller.collected_data.get("leadType"),
                            )
                        if flow_controller.state == ConversationState.WORKFLOW_QUESTION:
                            # Re-enter main flow so non-booking lead rules can auto-select
                            # service and start attached workflow after contact capture.
                            return await self.generate_response(
                                flow_controller, "", conversation_history, context
                            )
                        return await self._generate_state_response(
                            flow_controller.state, "", conversation_history, context, flow_controller=flow_controller
                        )
            else:
                # Phone not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context, flow_controller=flow_controller)
                else:
                    # No RAG needed for standard phone collection prompt
                    return await self._generate_state_response(state, "", conversation_history, context, flow_controller=flow_controller)

        if state == ConversationState.APPOINTMENT_OFFER:
            text = user_message.strip().lower()
            if text in {"yes", "y", "book", "book now", "sure"}:
                flow_controller.transition_to(ConversationState.CALENDAR_BOOKING)
                return "BOOK_APPOINTMENT_REQUESTED"
            if text in {"no", "n", "no thanks", "later"}:
                flow_controller.transition_to(ConversationState.COMPLETE)
                return await self._generate_json(flow_controller, conversation_history)
            return 'Would you like to book an appointment now? <button value="yes">Yes, book now</button> <button value="no">No thanks</button>'

        if state == ConversationState.CALENDAR_BOOKING:
            return "BOOK_APPOINTMENT_REQUESTED"

        if state == ConversationState.APPOINTMENT_CONFIRMATION:
            if user_message.strip().lower() in {"confirm", "yes", "ok"}:
                flow_controller.transition_to(ConversationState.COMPLETE)
                return await self._generate_json(flow_controller, conversation_history)
            return "Please confirm the selected appointment slot to continue."
        
        # Check if all data is collected and we can generate JSON
        if flow_controller.can_generate_json():
            return await self._generate_json(flow_controller, conversation_history)
        
        # Generate natural response based on state - no RAG needed for standard prompts
        return await self._generate_state_response(state, "", conversation_history, context, flow_controller=flow_controller)
    
    async def _get_rag_context(self, query: str, context: Dict[str, Any], is_question: bool = False) -> str:
        """Get relevant context from RAG. Only retrieves FAQs when is_question=True."""
        if not self.rag_service:
            logger.debug(f"RAG service not available for query: {query}")
            return ""
        
        # Only retrieve RAG context (including FAQs) when user is asking a question
        # For generic queries like "name collection", "email collection", don't retrieve FAQs
        if not is_question:
            # Check if query looks like a generic state query (not an actual user question)
            generic_queries = ["name collection", "email collection", "phone collection", 
                             "lead type selection", "service selection", "service selection"]
            if any(gq in query.lower() for gq in generic_queries):
                logger.debug(f"Skipping RAG retrieval for generic query: '{query}' (not a question)")
                return ""
        
        try:
            # Use the RAG service's method which handles retrieval correctly
            rag_context = await self.rag_service.get_relevant_context(query)
            
            # Log the retrieved context for debugging
            logger.info(f"RAG Context retrieved for query '{query}' (is_question={is_question}):")
            if rag_context:
                # Extract document count and details from the context if available
                # The get_relevant_context method returns formatted string, so we log it directly
                logger.info(f"  - RAG context length: {len(rag_context)} characters")
                logger.info(f"  - Full RAG context:\n{rag_context}")
            else:
                logger.info(f"  - No context retrieved for query")
            
            return rag_context
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}", exc_info=True)
            return ""
    
    async def _generate_service_selection_response(
        self,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any],
        collected_lead_type: Optional[str] = None
    ) -> str:
        """Generate service selection response. Filters by relevantServicePlans when configured."""
        if not self.client:
            return "Which service are you interested in?"
        
        conversation_style = self._conversation_style_enabled(context)
        
        service_plans = context.get("service_plans", [])
        lead_types = context.get("lead_types", [])
        
        # Try to filter by lead type's relevantServicePlans
        filtered = self._filter_services_by_lead_type(service_plans, lead_types, collected_lead_type)
        if filtered is not None:
            all_services = filtered
            logger.info(f"Filtered {len(filtered)} services for lead type '{collected_lead_type}': {filtered}")
        else:
            all_services = []
            for plan in service_plans:
                if isinstance(plan, dict):
                    plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                    if plan_name:
                        all_services.append(plan_name)
                else:
                    all_services.append(str(plan))
            logger.info(f"No filtering - showing all {len(all_services)} services")
        
        if self.channel == "voice":
            services_text = self._format_voice_list(all_services)
            system_prompt = f"""You are a {self.profession} assistant.

CRITICAL RULES:
1. The user has already selected a lead type (callback, appointment, or information request)
2. Ask: "Which service are you interested in?" and LIST the service names verbally (no numbers, no buttons).
3. Mention ALL services exactly once in natural language: {services_text or "No services provided"}
4. DO NOT ask the user to pick a number or say 'option one' etc. Ask them to say the service name.
5. Keep the response concise (one or two sentences) and return plain text only.
6. DO NOT ask for date/time - that is NOT part of this flow.
7. Service selection is MANDATORY - every user must select a service."""
            if self._language_instruction():
                system_prompt += "\n\n" + self._language_instruction()
            
            messages = [{"role": "system", "content": system_prompt}]
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            messages.extend(recent_history)
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.3
                )
                answer = (response.choices[0].message.content or "").strip()
                # Ensure services are included even if AI doesn't add them
                if services_text and services_text.lower() not in answer.lower():
                    prefix = "Which service are you interested in?"
                    if answer:
                        return f"{answer} Available services include {services_text}."
                    return f"{prefix} Available services include {services_text}."
                return answer
            except Exception as e:
                logger.error(f"Error generating service selection response: {e}")
                if services_text:
                    return f"Which service are you interested in? Available services include {services_text}."
                return "Which service are you interested in?"
        else:
            # Format services as buttons for text channels (web and WhatsApp)
            if conversation_style:
                return "What kind of service are you looking for today?"

            display_services = list(all_services) if all_services else []
            if self.response_language and display_services and self.client and self.model:
                from ..utils.translation_utils import translate_batch
                # Extract app_id for caching translations per app
                app_data = context.get("app", {})
                app_id = str(app_data.get("id")) if app_data and app_data.get("id") else None
                try:
                    display_services = await translate_batch(
                        self.client, self.model, display_services, self.response_language, app_id
                    )
                except Exception as e:
                    logger.warning("Service names translation failed: %s", e)
            services_text = " ".join([f"<button>{s}</button>" for s in display_services if s])
            if services_text:
                return f"Which service are you interested in? {services_text}"
            return "Which service are you interested in?"
    
    async def _generate_data_collected_with_question_response(
        self,
        data_type: str,  # "lead type", "service", "name", "email", "phone"
        data_value: str,  # The actual value collected
        rag_context: str,
        next_step: str,  # "service selection", "name", "email", "phone"
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any],
        collected_lead_type: Optional[str] = None,
        *,
        flow_controller: Optional[FlowController] = None,
    ) -> str:
        """Generate response when data is collected AND user asked a question"""
        conversation_style = bool((context.get("integration") or {}).get("conversationStyle"))
        def _get_service_list(ctx: Dict[str, Any], lead_type_val: Optional[str] = None) -> List[str]:
            """Helper to get filtered or all services"""
            service_plans = ctx.get("service_plans", [])
            lead_types = ctx.get("lead_types", [])
            filtered = self._filter_services_by_lead_type(service_plans, lead_types, lead_type_val)
            if filtered is not None:
                return filtered
            all_services = []
            for plan in service_plans:
                if isinstance(plan, dict):
                    plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                    if plan_name:
                        all_services.append(plan_name)
                else:
                    all_services.append(str(plan))
            return all_services
        
        if not self.client:
            if next_step == "service selection":
                if self.channel != "voice":
                    if conversation_style:
                        return "What kind of service are you looking for today?"
                    all_services = _get_service_list(context, collected_lead_type)
                    services_text = " ".join([f"<button>{s}</button>" for s in all_services if s])
                    if services_text:
                        return f"Which service are you interested in? {services_text}"
                return "Which service are you interested in?"
            elif next_step == "name":
                return "What's your name?"
            elif next_step == "email":
                return "What's your email address?"
            elif next_step == "phone":
                return "What's your phone number?"
            return "How can I help you?"
        
        # Map next step to question
        next_questions = {
            "service selection": "What kind of service are you looking for today?" if conversation_style else "Which service are you interested in?",
            "name": "What's your name?",
            "email": "What's your email address?",
            "phone": "What's your phone number?"
        }
        next_question = next_questions.get(next_step, "How can I help you?")
        
        system_prompt = f"""You are a {self.profession} assistant. 

CRITICAL RULES:
1. The user has provided their {data_type}: {data_value}
2. The user also asked a question in their message
3. Answer their question briefly (1-2 sentences) using the context provided below
4. After answering, acknowledge the {data_type} and ask: "{next_question}"
5. DO NOT show previous options again - data is already collected
6. DO NOT ask for date/time - that is NOT part of this flow
7. Move forward to the next step: {next_step}

Format: [Answer to question]. Great! I've noted your {data_type}: {data_value}. {next_question}"""
        if conversation_style and self.channel != "voice":
            system_prompt += "\n\nDo NOT output <button> tags or numbered lists. Continue the conversation naturally."
        if self._language_instruction():
            system_prompt += "\n\n" + self._language_instruction()
        _align_lt = (collected_lead_type or "").strip() or self._resolve_session_lead_type_value(
            context, flow_controller
        )
        _path = self._lead_path_alignment_for_llm(
            context,
            flow_controller,
            conversation_state=None,
            lead_type_value_override=_align_lt or None,
        )
        if _path:
            system_prompt += "\n\n" + _path
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add RAG context (contains answer info)
        if rag_context:
            messages.append({"role": "system", "content": f"Context for answering the question:\n{rag_context}"})
        
        # Add recent conversation history
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        messages.extend(recent_history)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=250,
                temperature=0.3
            )
            answer = (response.choices[0].message.content or "").strip()
            # Ensure it asks for next step if AI doesn't
            if next_question.lower() not in answer.lower():
                answer += f" {next_question}"
            
            # For web and WhatsApp, if next step is service selection, add service buttons (filtered)
            if next_step == "service selection" and self.channel != "voice" and not conversation_style:
                all_services = _get_service_list(context, collected_lead_type)
                services_text = " ".join([f"<button>{s}</button>" for s in all_services if s])
                if services_text and services_text not in answer:
                    answer += f" {services_text}"
            
            return answer
        except Exception as e:
            logger.error(f"Error generating data+question response: {e}")
            fallback = f"Great! I've noted your {data_type}: {data_value}. {next_question}"
            
            # For web and WhatsApp, if next step is service selection, add service buttons (filtered)
            if next_step == "service selection" and self.channel != "voice" and not conversation_style:
                all_services = _get_service_list(context, collected_lead_type)
                services_text = " ".join([f"<button>{s}</button>" for s in all_services if s])
                if services_text:
                    fallback += f" {services_text}"
            
            return fallback
    
    async def _generate_question_response(
        self,
        user_message: str,
        rag_context: str,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any],
        *,
        flow_controller: Optional[FlowController] = None,
    ) -> str:
        """Generate response to answer a user's question using RAG context"""
        # Helper to get fallback message using greeting utility
        def get_fallback_message() -> str:
            from ..utils.greeting_utils import get_greeting_with_fallback
            return get_greeting_with_fallback(context)
        
        if not self.client:
            return get_fallback_message()
        
        system_prompt = f"""You are a knowledgeable, friendly {self.profession} assistant helping customers.

When answering the user's question, follow these guidelines:
- If the context below directly answers the question, use it.
- If the context doesn't cover it but the question is relevant to the {self.profession} industry or our services (e.g. asking about a treatment, procedure, product, or general industry topic), answer naturally and helpfully from your general knowledge — like a well-informed staff member would.
- If the question is completely unrelated to {self.profession} or our services (e.g. weather, politics, unrelated topics), respond with genuine warmth and empathy — acknowledge the question, then naturally redirect. Use varied, human-sounding phrases, for example:
  * "Oh, I wish I could help with that! I'm really only set up to assist with {self.profession} services — but I'd love to help you with a treatment or booking if you're interested?"
  * "Ha, that one's a little out of my world! I'm mostly here for {self.profession} stuff. Anything I can help you with on that front?"
  * "Good question, though I'm afraid that's a bit beyond what I'm here for! I specialise in {self.profession} — is there anything about our services I can help with?"
  Vary the phrasing naturally based on context. Never sound dismissive — always make the user feel welcome to ask about services.
- NEVER say "I don't have that information" or any robotic variation of it. Always sound warm, human, and helpful.
- Keep answers brief (1-2 sentences) then continue the conversation flow."""
        conversation_style = bool((context.get("integration") or {}).get("conversationStyle"))
        if conversation_style and self.channel != "voice":
            system_prompt += "\n\nDo NOT output <button> tags or numbered lists. Continue the conversation naturally."
        if self._language_instruction():
            system_prompt += "\n\n" + self._language_instruction()
        _path = self._lead_path_alignment_for_llm(context, flow_controller, conversation_state=None)
        if _path:
            system_prompt += "\n\n" + _path
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add RAG context
        if rag_context:
            messages.append({"role": "system", "content": f"Context:\n{rag_context}"})
        
        # Add recent conversation history
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.3
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error generating question response: {e}")
            return get_fallback_message()
    
    async def _generate_state_response(
        self,
        state: ConversationState,
        rag_context: str,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any],
        *,
        flow_controller: Optional[FlowController] = None,
    ) -> str:
        """Generate response using minimal prompt based on state"""
        if not self.client:
            return "I'm here to help you."

        # Minimal state-specific prompts (3-5 lines) - STRICT FLOW ENFORCEMENT
        # CRITICAL: If user asks a question, answer it briefly (1-2 sentences) then re-ask for required data
        state_prompts = {
            ConversationState.GREETING: f"You are a {self.profession} assistant. Greet the user and present lead type options from context.",
            ConversationState.LEAD_TYPE_SELECTION: f"""You are a {self.profession} assistant. 
- If user asks a question, answer it briefly (1-2 sentences) using context, then present lead type options.
- Present lead type options from context and wait for selection.
- DO NOT ask for date/time - that is NOT part of this flow.
- DO NOT ask for service selection yet - wait for lead type selection first.""",
            ConversationState.SERVICE_SELECTION: f"""You are a {self.profession} assistant. 
- The user has ALREADY selected a lead type - DO NOT show lead type options again.
- If user asks a question, answer it briefly (1-2 sentences) using context, then ask for service selection.
- CRITICAL: Ask 'Which service are you interested in?' and present ALL service options from context.
- DO NOT ask for date/time - that is NOT part of this flow.
- DO NOT show lead type options - move forward to service selection.
- Service selection is MANDATORY.""",
            ConversationState.NAME_COLLECTION: f"""You are a {self.profession} assistant. 
- If user asks a question, answer it briefly (1-2 sentences) using context, then ask for their name.
- Ask for the user's name naturally.
- DO NOT ask for date/time - that is NOT part of this flow.
- DO NOT show lead type or service options - continue with name collection.""",
            ConversationState.EMAIL_COLLECTION: f"""You are a {self.profession} assistant. 
- If user asks a question, answer it briefly (1-2 sentences) using context, then ask for their email.
- Ask for the user's email address naturally.
- DO NOT ask for date/time - that is NOT part of this flow.
- DO NOT show previous options - continue with email collection.""",
            ConversationState.PHONE_COLLECTION: f"""You are a {self.profession} assistant. 
- If user asks a question, answer it briefly (1-2 sentences) using context, then ask for their phone number.
- Ask for the user's phone number naturally.
- DO NOT ask for date/time - that is NOT part of this flow.
- DO NOT show previous options - continue with phone collection.""",
        }

        conversation_style = self._conversation_style_enabled(context)
        lead_types = context.get("lead_types", [])
        lead_type_examples: List[str] = []
        for lt in lead_types:
            if isinstance(lt, dict):
                txt = (lt.get("text") or "").strip()
            else:
                txt = str(lt).strip()
            if txt:
                lead_type_examples.append(self._normalize_lead_option_for_voice(txt))
        lead_type_examples = [x for x in lead_type_examples if x][:3]
        lead_type_examples_text = ", ".join(lead_type_examples)
        # Conversational mode: no option lists / buttons for non-voice channels.
        if conversation_style and self.channel != "voice":
            state_prompts[ConversationState.GREETING] = (
                f"You are a {self.profession} assistant. Greet the user and ask what they need "
                f"in a natural way. Do NOT present lead type options from context. "
                f"Do NOT output <button> tags or numbered lists."
            )
            state_prompts[ConversationState.LEAD_TYPE_SELECTION] = f"""You are a {self.profession} assistant.
- If user asks a question, answer it briefly (1-2 sentences) using context, then ask what they need next in free text.
- Ask the user what they'd like help with.{f" Examples from this business: {lead_type_examples_text}." if lead_type_examples_text else ""}
- DO NOT output <button> tags or numbered lists.
- DO NOT ask for date/time - that is NOT part of this flow.
- DO NOT ask for service selection yet - wait for the user to describe their need first."""
            state_prompts[ConversationState.SERVICE_SELECTION] = f"""You are a {self.profession} assistant.
- The user has ALREADY selected a lead type - do NOT show lead type options again.
- If user asks a question, answer it briefly (1-2 sentences) using context, then ask which service they want next in free text.
- Ask a natural question like 'Which service are you interested in?' (open-ended).
- DO NOT output <button> tags or numbered lists.
- DO NOT ask for date/time - that is NOT part of this flow.
- Service selection is MANDATORY."""

        # For LEAD_TYPE_SELECTION (text channels): inject exact options from context so we never show a different list
        if state == ConversationState.LEAD_TYPE_SELECTION and self.channel != "voice" and lead_types and not conversation_style:
            def format_lead_option(lt: dict, index: int) -> str:
                text = lt.get('text', '')
                emoji = lt.get('emoji', '').strip() if lt.get('emoji') else ''
                if emoji:
                    return f"{index}. {emoji} {text}"
                return f"{index}. {text}"
            numbered = "\n".join([format_lead_option(lt, i) for i, lt in enumerate(lead_types, 1) if isinstance(lt, dict)])
            state_prompts[ConversationState.LEAD_TYPE_SELECTION] = f"""You are a {self.profession} assistant.

CRITICAL - SERVICE OPTIONS (use ONLY this list from context; never invent or substitute another list):
{numbered}

RULES:
1. When the user asks for any option in their own language or words, they mean one of OUR options above. Always show the EXACT list above—do NOT replace it with a different list (e.g. do NOT show product categories, menu items, or other lists that are not these options).
2. You MUST respond with EXACTLY the options above in this order INCLUDING the emojis (you may translate the labels to the user's language but keep the emojis). Do NOT add, remove, or replace with other options.
3. If user asks a question, answer briefly (1-2 sentences) using context, then show the EXACT list above (with emojis) and ask them to choose by number.
4. DO NOT ask for date/time. Wait for lead type selection first."""

        if self.channel == "voice":
            lead_types = context.get("lead_types", [])
            lead_options = []
            for lt in lead_types:
                if isinstance(lt, dict):
                    lead_options.append(self._normalize_lead_option_for_voice(lt.get("text", "")))
                else:
                    lead_options.append(self._normalize_lead_option_for_voice(str(lt)))
            lead_voice_list = self._format_voice_list(lead_options)
            lead_voice_question = (
                f"Would you like {lead_voice_list}?" if lead_voice_list else "No lead types provided"
            )

            service_plans = context.get("service_plans", [])
            service_names = []
            for plan in service_plans:
                if isinstance(plan, dict):
                    plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                    if plan_name:
                        service_names.append(plan_name)
                else:
                    service_names.append(str(plan))
            service_voice_text = self._format_voice_list(service_names)

            state_prompts[ConversationState.GREETING] = f"""You are a {self.profession} assistant interacting over voice.
 - Offer a friendly greeting.
 - Mention the available lead types verbally: {lead_voice_question}.
 - Ask which lead type they would like without referencing option numbers.
 - Keep the response short (1-2 sentences)."""

            state_prompts[ConversationState.LEAD_TYPE_SELECTION] = f"""You are a {self.profession} assistant on a voice call.
 - If the user asks a question, answer briefly then guide them back to choosing a lead type.
 - List the lead type options naturally (no numbers, no buttons): {lead_voice_question}.
 - Ask them which lead type they prefer and remind them to say the option name.
 - Do NOT ask for date/time."""

            state_prompts[ConversationState.SERVICE_SELECTION] = f"""You are a {self.profession} assistant on a voice call.
 - The user has already selected a lead type.
 - If they ask a question, answer briefly then ask for a service selection.
 - Mention all service options verbally and naturally (no numbers): {service_voice_text or "No services provided"}.
 - Ask them to say the service name they are interested in.
 - Do NOT ask for date/time."""

            state_prompts[ConversationState.NAME_COLLECTION] = f"""You are a {self.profession} assistant on a voice call.
 - If the user asks a question, answer briefly and then ask for their name.
 - Ask for their full name naturally (no buttons).
 - Keep the response short and polite."""

            state_prompts[ConversationState.EMAIL_COLLECTION] = f"""You are a {self.profession} assistant on a voice call.
 - If the user asks a question, answer briefly then ask for their email address.
 - Ask for the email in a friendly, conversational way.
 - Do NOT mention numbers or buttons."""
        
        # For WhatsApp/Messenger/Instagram in LEAD_TYPE_SELECTION: format lead types directly with emojis
        # (like generate_greeting) instead of relying on LLM to preserve emojis
        if state == ConversationState.LEAD_TYPE_SELECTION and self.channel in ("whatsapp", "messenger", "instagram") and not conversation_style:
            lead_types = context.get("lead_types", [])
            if lead_types:
                from ..utils.language_utils import detect_language, get_language_name_for_prompt, get_language_name
                from ..utils.response_strings import get_string
                from ..utils.translation_utils import translate_batch
                
                # Detect language from conversation history
                lang_code = "en"
                if conversation_history:
                    last_user_msg = next((msg.get("content", "") for msg in reversed(conversation_history) if msg.get("role") == "user"), None)
                    if last_user_msg:
                        lang_code = detect_language(last_user_msg)
                
                need_translation = lang_code and lang_code != "en"
                translated_options: List[str] = []
                
                if need_translation and lead_types:
                    use_labels = all(
                        isinstance(lt, dict) and isinstance(lt.get("labels"), dict) and (lang_code or "") in (lt.get("labels") or {})
                        for lt in lead_types if isinstance(lt, dict)
                    )
                    if not use_labels:
                        to_translate = [str(lt.get("text", "") or lt.get("value", "")).strip() for lt in lead_types if isinstance(lt, dict)]
                        if to_translate and self.client and self.model:
                            translated_options = await translate_batch(
                                self.client, self.model, to_translate, get_language_name(lang_code)
                            )
                
                def option_text(lt: dict, index: int = 0) -> str:
                    text = lt.get("text", "") or lt.get("value", "")
                    emoji = lt.get("emoji", "").strip() if lt.get("emoji") else ""
                    
                    display_text = str(text)
                    if need_translation:
                        labels = lt.get("labels") or {}
                        if isinstance(labels, dict) and labels.get(lang_code):
                            display_text = labels[lang_code]
                        elif translated_options and 0 <= index < len(translated_options):
                            display_text = translated_options[index]
                    
                    if emoji:
                        return f"{emoji} {display_text}"
                    return display_text
                
                # Format options with emojis
                options = "\n".join([f"{i}. {option_text(lt, i - 1)}" for i, lt in enumerate(lead_types, 1) if isinstance(lt, dict)])
                reply_line = get_string("please_reply_number", lang_code) if lang_code else "Please reply with the number of your choice."
                
                # If there's RAG context (user asked a question), generate answer first, then show options
                if rag_context and self.client:
                    try:
                        answer_system = f"You are a {self.profession} assistant. Answer the question briefly in 1-2 sentences using the provided context."
                        if self._language_instruction():
                            answer_system += "\n\n" + self._language_instruction()
                        
                        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
                        answer_messages = [
                            {"role": "system", "content": answer_system},
                            {"role": "system", "content": f"Context: {rag_context}"}
                        ]
                        answer_messages.extend(recent_history)
                        
                        answer_response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=answer_messages,
                            max_tokens=100,
                            temperature=0.3
                        )
                        answer = (answer_response.choices[0].message.content or "").strip()
                        if answer:
                            return f"{answer}\n\n{options}\n\n{reply_line}"
                    except Exception as e:
                        logger.warning(f"Failed to generate answer for question in LEAD_TYPE_SELECTION: {e}")
                
                # No question or answer generation failed - just show options
                return f"{options}\n\n{reply_line}"
        
        system_prompt = state_prompts.get(state, f"You are a {self.profession} assistant. Continue the conversation naturally.")
        if self._language_instruction():
            system_prompt += "\n\n" + self._language_instruction()
        _path = self._lead_path_alignment_for_llm(
            context, flow_controller, conversation_state=state
        )
        if _path:
            system_prompt += "\n\n" + _path
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add RAG context if available
        if rag_context:
            messages.append({"role": "system", "content": f"Context: {rag_context}"})
        
        # Add recent conversation history (last 10 messages)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        messages.extend(recent_history)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm here to help you. How can I assist you today?"
    
    async def generate_lead_json(
        self,
        flow_controller: FlowController,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Public method to generate lead JSON with summary, description, and history"""
        return await self._generate_json(flow_controller, conversation_history)
    
    async def _generate_json(
        self, 
        flow_controller: FlowController, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate JSON from collected data with summary, description, and history"""
        import json
        
        # Get base JSON data with history
        data = flow_controller.get_json_data(conversation_history)
        
        # Generate summary and description from conversation history if available
        if conversation_history and self.client:
            try:
                # Create summary of the conversation
                summary = await self._generate_conversation_summary(conversation_history, data)
                data["summary"] = summary
                
                # Create description of the lead/interaction
                description = await self._generate_lead_description(data, conversation_history)
                data["description"] = description
            except Exception as e:
                logger.error(f"Error generating summary/description: {e}")
                # Fallback: create simple summary/description
                data["summary"] = self._create_fallback_summary(data, conversation_history)
                data["description"] = f"Lead inquiry for {data.get('serviceType', 'service')} - {data.get('leadType', 'request')}"
        else:
            # Fallback if no history or LLM not available
            data["summary"] = self._create_fallback_summary(data, conversation_history)
            data["description"] = f"Lead inquiry for {data.get('serviceType', 'service')} - {data.get('leadType', 'request')}"
        
        flow_controller.transition_to(ConversationState.COMPLETE)
        return json.dumps(data)
    
    async def _generate_conversation_summary(
        self, 
        conversation_history: List[Dict[str, str]], 
        collected_data: Dict[str, Any]
    ) -> str:
        """Generate a brief summary of the conversation"""
        if not self.client:
            return self._create_fallback_summary(collected_data, conversation_history)
        
        try:
            # Format conversation history for summary
            history_text = "\n".join([
                f"{msg.get('role', 'unknown').title()}: {msg.get('content', '')}"
                for msg in conversation_history[-20:]  # Last 20 messages
            ])
            
            system_prompt = f"""You are a {self.profession} assistant. Generate a brief 2-3 sentence summary of this conversation.
Focus on:
- What the customer requested (lead type: {collected_data.get('leadType', 'N/A')})
- Which service they're interested in ({collected_data.get('serviceType', 'N/A')})
- Any key questions or concerns they raised
- Keep it concise and professional"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation:\n{history_text}\n\nGenerate a brief summary:"}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = (response.choices[0].message.content or "").strip()
            return summary if summary else self._create_fallback_summary(collected_data, conversation_history)
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return self._create_fallback_summary(collected_data, conversation_history)
    
    async def _generate_lead_description(
        self,
        collected_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a description of the lead/interaction"""
        if not self.client:
            return f"Lead inquiry for {collected_data.get('serviceType', 'service')} - {collected_data.get('leadType', 'request')}"
        
        try:
            lead_type = collected_data.get('leadType', 'inquiry')
            service_type = collected_data.get('serviceType', 'service')
            customer_name = collected_data.get('leadName', 'Customer')
            
            system_prompt = f"""You are a {self.profession} assistant. Generate a brief 1-2 sentence description of this lead.
Customer: {customer_name}
Lead Type: {lead_type}
Service Interest: {service_type}
Make it professional and informative."""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate a brief lead description:"}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            description = (response.choices[0].message.content or "").strip()
            return description if description else f"{customer_name} - {lead_type} inquiry for {service_type}"
        except Exception as e:
            logger.error(f"Error generating lead description: {e}")
            return f"Lead inquiry for {collected_data.get('serviceType', 'service')} - {collected_data.get('leadType', 'request')}"
    
    def _create_fallback_summary(
        self, 
        collected_data: Dict[str, Any], 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Create a simple fallback summary when LLM is not available"""
        customer_name = collected_data.get('leadName', 'Customer')
        lead_type = collected_data.get('leadType', 'inquiry')
        service_type = collected_data.get('serviceType', 'service')
        
        return f"{customer_name} requested {lead_type} for {service_type}."
    
    async def generate_greeting(
        self,
        context: Dict[str, Any],
        channel: Optional[str] = None,
        first_message: Optional[str] = None,
    ) -> str:
        """Generate initial greeting. If first_message is provided, detect its language and greet in that language with options in sync."""
        from ..utils.greeting_utils import get_greeting_with_fallback
        from ..utils.language_utils import detect_language, get_language_name_for_prompt, get_language_name
        from ..utils.response_strings import get_string
        from ..utils.translation_utils import translate_batch

        current_channel = (channel or self.channel or "web").lower()
        conversation_style = bool((context.get("integration") or {}).get("conversationStyle"))
        lang_code = "en"
        if first_message and str(first_message).strip():
            lang_code = detect_language(str(first_message))
            self.set_response_language(get_language_name_for_prompt(lang_code))

        # Get greeting in detected language (or English)
        greeting = get_greeting_with_fallback(context, lang_code=lang_code if lang_code != "en" else None)
        if not greeting:
            greeting = get_greeting_with_fallback(context)

        lead_types = context.get("lead_types", [])
        need_translation = lang_code and lang_code != "en"
        translated_options: List[str] = []

        # Extract app_id for caching translations per app
        app_data = context.get("app", {})
        app_id = str(app_data.get("id")) if app_data and app_data.get("id") else None

        if need_translation and lead_types:
            # Prefer DB labels; for any without a label, collect text and translate in one batch
            use_labels = all(
                isinstance(lt, dict) and isinstance(lt.get("labels"), dict) and (lang_code or "") in (lt.get("labels") or {})
                for lt in lead_types if isinstance(lt, dict)
            )
            if not use_labels:
                to_translate = [str(lt.get("text", "") or lt.get("value", "")).strip() for lt in lead_types if isinstance(lt, dict)]
                if to_translate:
                    # Pass app_id for caching translations per app
                    translated_options = await translate_batch(
                        self.client, self.model, to_translate, get_language_name(lang_code), app_id
                    )

        def option_text(lt: dict, index: int = 0) -> str:
            text = lt.get("text", "") or lt.get("value", "")
            emoji = lt.get("emoji", "").strip() if lt.get("emoji") else ""
            
            # Get translated text if needed
            display_text = str(text)
            if need_translation:
                labels = lt.get("labels") or {}
                if isinstance(labels, dict) and labels.get(lang_code):
                    display_text = labels[lang_code]
                elif translated_options and 0 <= index < len(translated_options):
                    display_text = translated_options[index]
            
            # Prepend emoji to text if present
            if emoji:
                return f"{emoji} {display_text}"
            return display_text

        if current_channel in ("whatsapp", "messenger", "instagram"):
            # Log lead types for debugging
            logger.info(f"generate_greeting: Processing {len(lead_types)} lead types for channel {current_channel}")
            for idx, lt in enumerate(lead_types[:3]):  # Log first 3
                if isinstance(lt, dict):
                    logger.info(f"  Lead type {idx+1}: text='{lt.get('text')}', emoji='{lt.get('emoji', 'NONE')}'")

            if conversation_style:
                # Conversational mode: return configured greeting as-is (avoid repetitive appended prompt)
                return greeting
            
            options = "\n".join([f"{i}. {option_text(lt, i - 1)}" for i, lt in enumerate(lead_types, 1) if isinstance(lt, dict)])
            reply_line = get_string("please_reply_number", lang_code) if lang_code else "Please reply with the number of your choice."
            logger.info(f"Generated options string (first 100 chars): {options[:100]}")
            return f"{greeting}\n\n{options}\n\n{reply_line}"
        elif current_channel == "voice":
            option_names = []
            for idx, lt in enumerate(lead_types):
                if isinstance(lt, dict):
                    option_names.append(self._normalize_lead_option_for_voice(option_text(lt, idx)))
                else:
                    option_names.append(self._normalize_lead_option_for_voice(str(lt)))
            options_text = self._format_voice_list(option_names)
            if options_text:
                return f"{greeting}\n\nWould you like {options_text}?"
            return greeting
        else:
            if conversation_style:
                # Conversational mode: return configured greeting as-is (avoid repetitive appended prompt)
                return greeting

            buttons = " ".join([f"<button>{option_text(lt, idx)}</button>" for idx, lt in enumerate(lead_types) if isinstance(lt, dict)])
            return f"{greeting} {buttons}"

