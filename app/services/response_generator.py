"""Production-grade response generator using state machine and minimal prompts"""
from typing import Dict, Any, List, Optional
import logging
from openai import AsyncOpenAI
import json

from app.services.conversation_state import FlowController, ConversationState
from app.services.data_extractors import DataExtractor
from app.services.validators import Validator

logger = logging.getLogger("assistly.response_generator")


class ResponseGenerator:
    """Generate responses based on conversation state with minimal prompts"""
    
    def __init__(self, settings: Any, rag_service: Any):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.model = settings.gpt_model
        self.rag_service = rag_service
        self.profession = "Clinic"
        self.channel = "web"  # web, whatsapp, voice
    
    def set_profession(self, profession: str):
        """Set profession for responses"""
        self.profession = profession or self.profession

    def set_channel(self, channel: str):
        """Configure output channel (web, whatsapp, voice)."""
        if channel:
            self.channel = channel.lower()
    
    def _merge_treatment_plans_into_services(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge treatment plans into service_types array for unified service selection"""
        service_types = context.get("service_types", [])
        treatment_plans = context.get("treatment_plans", [])
        
        # Convert treatment plans to service format and merge with service types
        merged_services = list(service_types)  # Start with existing services (can be strings or dicts)
        for plan in treatment_plans:
            if isinstance(plan, dict) and "question" in plan:
                # Convert treatment plan to service format
                service_item = {
                    "name": plan["question"],
                    "title": plan["question"],
                    "description": plan.get("description", ""),
                    "is_treatment_plan": True
                }
                merged_services.append(service_item)
        
        # Update context with merged services
        context = context.copy()
        context["service_types"] = merged_services
        return context
    
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
    
    async def generate_response(
        self,
        flow_controller: FlowController,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> str:
        """Generate response based on current state"""
        # Merge treatment plans into service_types for unified handling
        context = self._merge_treatment_plans_into_services(context)
        
        state = flow_controller.state
        extractor = DataExtractor()
        validator = Validator()
        
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
        
        # Classify user intent (question detection using LLM)
        try:
            intent = await self._classify_intent(user_message)
            has_question = intent.get("is_question", False)
            question_type = intent.get("question_type", "not_question")
            
            logger.debug(f"Intent classification: is_question={has_question}, type={question_type}, confidence={intent.get('confidence', 0.0)}")
        except Exception as e:
            # If intent classification fails, log error and assume it's not a question
            # This allows the flow to continue even if classification fails
            logger.error(f"Intent classification failed: {e}, treating as non-question")
            has_question = False
            question_type = "not_question"
        
        # Handle data collection states - extract and validate before AI generation
        if state == ConversationState.LEAD_TYPE_SELECTION:
            lead_type = extractor.match_lead_type(user_message, context.get("lead_types", []))
            if lead_type:
                logger.info(f"Matched lead type: {lead_type.get('text')} (value: {lead_type.get('value')})")
                flow_controller.update_collected_data("leadType", lead_type.get("value"))
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                
                # Check if user asked a question along with lead type selection
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context)
                    return await self._generate_data_collected_with_question_response(
                        "lead type", lead_type.get("text"), rag_context, 
                        "service selection", conversation_history, context
                    )
                else:
                    # Explicitly include services in prompt for service selection
                    return await self._generate_service_selection_response(flow_controller, conversation_history, context)
            else:
                logger.warning(f"No lead type matched for user input: '{user_message}'. Available lead types: {[lt.get('text') for lt in context.get('lead_types', [])]}")
                # No match found - use AI to handle questions, but with strict prompt to only show lead types
                rag_context = await self._get_rag_context(user_message if has_question else "lead type selection", context)
                return await self._generate_state_response(state, rag_context, conversation_history, context)
        
        if state == ConversationState.SERVICE_SELECTION:
            # Services already include treatment plans (merged above)
            services = context.get("service_types", [])
            # Extract service names for matching
            all_services = []
            for s in services:
                if isinstance(s, dict):
                    all_services.append(s.get("name", s.get("title", "")))
                else:
                    all_services.append(str(s))
            service = extractor.match_service(user_message, all_services)
            if service:
                logger.info(f"Matched service: {service}")
                flow_controller.update_collected_data("serviceType", service)
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                
                # Check if user asked a question along with service selection
                if has_question:
                    # Get RAG context for the question (pricing, info about the service)
                    rag_context = await self._get_rag_context(f"{service} {user_message}", context)
                    # Generate response that answers question AND asks for name
                    return await self._generate_data_collected_with_question_response(
                        "service", service, rag_context,
                        "name", conversation_history, context
                    )
                else:
                    # Just acknowledge and move to name collection
                    rag_context = await self._get_rag_context("name collection", context)
                    return await self._generate_state_response(flow_controller.state, rag_context, conversation_history, context)
            else:
                logger.warning(f"No service matched for user input: '{user_message}'. Available services: {all_services}")
                # No match - use AI to handle questions, but ensure it stays in service selection
                rag_context = await self._get_rag_context(user_message if has_question else "service selection", context)
                return await self._generate_state_response(state, rag_context, conversation_history, context)
        
        if state == ConversationState.NAME_COLLECTION:
            if name and validator.is_valid_name(name):
                flow_controller.update_collected_data("leadName", name)
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                
                # Check if user asked a question along with name
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context)
                    return await self._generate_data_collected_with_question_response(
                        "name", name, rag_context,
                        "email", conversation_history, context
                    )
                else:
                    rag_context = await self._get_rag_context("email collection", context)
                    return await self._generate_state_response(flow_controller.state, rag_context, conversation_history, context)
            else:
                # Name not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
                else:
                    # Just ask for name again
                    rag_context = await self._get_rag_context("name collection", context)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
        
        if state == ConversationState.EMAIL_COLLECTION:
            if email and validator.is_valid_email(email):
                flow_controller.update_collected_data("leadEmail", email)
                
                # Check if user asked a question along with email
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context)
                    # Answer question and acknowledge email, then proceed
                    if flow_controller.validate_email:
                        # Answer question, acknowledge email, then trigger OTP sending
                        answer = await self._generate_question_response(user_message, rag_context, conversation_history, context)
                        # Return in format that main.py can handle: answer + SEND_EMAIL marker
                        # main.py will send answer first, then handle SEND_EMAIL
                        return f"{answer}|||SEND_EMAIL:{email}"
                    else:
                        flow_controller.transition_to(flow_controller.get_next_state())
                        logger.info(f"Transitioned to state: {flow_controller.state.value}")
                        if flow_controller.can_generate_json():
                            # Answer question then generate JSON
                            answer = await self._generate_question_response(user_message, rag_context, conversation_history, context)
                            json_data = await self._generate_json(flow_controller, conversation_history)
                            return f"{answer}\n\n{json_data}"
                        else:
                            return await self._generate_data_collected_with_question_response(
                                "email", email, rag_context,
                                "phone", conversation_history, context
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
                            rag_context = await self._get_rag_context("phone collection", context)
                            return await self._generate_state_response(flow_controller.state, rag_context, conversation_history, context)
            else:
                # Email not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
                else:
                    rag_context = await self._get_rag_context("email collection", context)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
        
        if state == ConversationState.PHONE_COLLECTION:
            if phone and validator.is_valid_phone(phone):
                flow_controller.update_collected_data("leadPhoneNumber", phone)
                
                # Check if user asked a question along with phone
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context)
                    if flow_controller.validate_phone:
                        # Answer question, acknowledge phone, then trigger OTP sending
                        answer = await self._generate_question_response(user_message, rag_context, conversation_history, context)
                        # Return in format that main.py can handle: answer + SEND_PHONE marker
                        return f"{answer}|||SEND_PHONE:{phone}"
                    else:
                        flow_controller.transition_to(flow_controller.get_next_state())
                        logger.info(f"Transitioned to state: {flow_controller.state.value}")
                        # Answer question then generate JSON
                        answer = await self._generate_question_response(user_message, rag_context, conversation_history, context)
                        json_data = await self._generate_json(flow_controller, conversation_history)
                        return f"{answer}\n\n{json_data}"
                else:
                    # No question, proceed normally
                    if flow_controller.validate_phone:
                        return "SEND_PHONE:" + phone
                    else:
                        flow_controller.transition_to(flow_controller.get_next_state())
                        # Generate JSON
                        return await self._generate_json(flow_controller, conversation_history)
            else:
                # Phone not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
                else:
                    rag_context = await self._get_rag_context("phone collection", context)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
        
        # Check if all data is collected and we can generate JSON
        if flow_controller.can_generate_json():
            return await self._generate_json(flow_controller, conversation_history)
        
        # Generate natural response based on state
        rag_context = await self._get_rag_context(flow_controller.get_state_prompt_context(), context)
        return await self._generate_state_response(state, rag_context, conversation_history, context)
    
    async def _get_rag_context(self, query: str, context: Dict[str, Any]) -> str:
        """Get relevant context from RAG"""
        if not self.rag_service:
            logger.debug(f"RAG service not available for query: {query}")
            return ""
        
        try:
            # Use the RAG service's method which handles retrieval correctly
            rag_context = await self.rag_service.get_relevant_context(query)
            
            # Log the retrieved context for debugging
            logger.info(f"RAG Context retrieved for query '{query}':")
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
        flow_controller: FlowController,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> str:
        """Generate service selection response with explicit service list"""
        if not self.client:
            return "Which service are you interested in?"
        
        # Get all services (already includes treatment plans merged above)
        services = context.get("service_types", [])
        all_services = []
        
        for s in services:
            if isinstance(s, dict):
                all_services.append(s.get("name", s.get("title", "")))
            else:
                all_services.append(str(s))
        channel = "voice" if getattr(flow_controller, "is_voice", False) or self.channel == "voice" else self.channel

        if channel == "whatsapp":
            services_display = "\n".join([f"{idx}. {svc}" for idx, svc in enumerate(all_services, 1) if svc])
            instruction = f"Show ALL services as a numbered list:\n{services_display}"
        elif channel == "voice":
            services_display = "\n".join([f"- {svc}" for svc in all_services if svc])
            instruction = (
                "Provide the list of services as natural language bullet points (no numbers)."
                f"\nHere are the services:\n{services_display}"
            )
        else:
            services_display = " ".join([f"<button>{s}</button>" for s in all_services if s])
            instruction = f"Show ALL services as buttons: {services_display}"

        system_prompt = f"""You are a {self.profession} assistant. 
 1. The user has ALREADY selected a lead type - DO NOT show lead type options again.
 2. You MUST ask: "Which service are you interested in?" 
3. {instruction}
4. If the user asked a question, answer briefly (1-2 sentences) using context, then ask for service selection.
5. Do NOT ask for date/time - that is NOT part of this flow.
6. Do NOT acknowledge the lead type selection in detail - just move to service selection
7. Service selection is MANDATORY - every user must select a service
8. Response must be concise."""
        
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
            if channel == "whatsapp":
                if services_display not in answer:
                    return f"{answer}\n\n{services_display}" if answer else f"Which service are you interested in?\n{services_display}"
            elif channel == "voice":
                if services_display not in answer:
                    return f"Which service are you interested in?\n{services_display}" if not answer else f"{answer}\n\n{services_display}"
            else:
                if services_display not in answer:
                    return f"{answer} {services_display}" if answer else f"Which service are you interested in? {services_display}"
            return answer
        except Exception as e:
            logger.error(f"Error generating service selection response: {e}")
            if channel == "whatsapp":
                return f"Which service are you interested in?\n{services_display}"
            if channel == "voice":
                return f"Which service are you interested in?\n{services_display}"
            return f"Which service are you interested in? {services_display}"
    
    async def _generate_data_collected_with_question_response(
        self,
        data_type: str,  # "lead type", "service", "name", "email", "phone"
        data_value: str,  # The actual value collected
        rag_context: str,
        next_step: str,  # "service selection", "name", "email", "phone"
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> str:
        """Generate response when data is collected AND user asked a question"""
        if not self.client:
            if next_step == "service selection":
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
            "service selection": "Which service are you interested in?",
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
            return answer
        except Exception as e:
            logger.error(f"Error generating data+question response: {e}")
            return f"Great! I've noted your {data_type}: {data_value}. {next_question}"
    
    async def _generate_question_response(
        self,
        user_message: str,
        rag_context: str,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> str:
        """Generate response to answer a user's question using RAG context"""
        # Helper to get fallback message based on assistant name
        def get_fallback_message() -> str:
            assistant_name = context.get("integration", {}).get("assistantName", "").strip()
            if assistant_name:
                return f"Thanks for reaching out. My name is {assistant_name}, your virtual Ai assistant how can i help you today"
            else:
                return "Thanks for reaching out. I am your virtual AI assistant how can i help you today"
        
        if not self.client:
            return get_fallback_message()
        
        system_prompt = f"""You are a {self.profession} assistant. 
Answer the user's question briefly (1-2 sentences) using ONLY the context provided below.
If the context doesn't contain the answer, say "I don't have that information, but let me help you with..." and continue the conversation."""
        
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
        context: Dict[str, Any]
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
        
        system_prompt = state_prompts.get(state, f"You are a {self.profession} assistant. Continue the conversation naturally.")

        channel = self.channel
        if channel == "voice":
            if state == ConversationState.LEAD_TYPE_SELECTION:
                system_prompt += "\n- Present the lead type options as short bullet points without numbering or buttons."
            elif state == ConversationState.SERVICE_SELECTION:
                system_prompt += "\n- Present service options as natural language bullet points (no numbering or buttons)."
            else:
                system_prompt += "\n- Keep responses concise and conversational for a spoken experience."
 
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
        is_whatsapp: bool = False,
        is_voice: bool = False,
    ) -> str:
        """Generate initial greeting tailored to the channel."""
        integration = context.get("integration", {})
        greeting = integration.get("greeting", "").strip()
        
        # Fallback to default greeting if empty or missing
        if not greeting:
            assistant_name = integration.get("assistantName", "").strip()
            if assistant_name:
                greeting = f"Thanks for reaching out. My name is {assistant_name}, your virtual Ai assistant how can i help you today"
            else:
                greeting = "Thanks for reaching out. I am your virtual AI assistant how can i help you today"
        
        lead_types = context.get("lead_types", [])
        channel_preference = (channel or ("whatsapp" if is_whatsapp or self.channel == "whatsapp" else "voice" if is_voice or self.channel == "voice" else self.channel))

        if channel_preference == "whatsapp":
            # Numbered list for WhatsApp text interface
            options = "\n".join([
                f"{i}. {lt.get('text', '')}" for i, lt in enumerate(lead_types, 1) if isinstance(lt, dict)
            ])
            return f"{greeting}\n\n{options}\n\nPlease reply with the number of your choice."

        if channel_preference == "voice":
            # Voice channel: list options naturally without numbers
            option_lines = [f"- {lt.get('text', '')}" for lt in lead_types if isinstance(lt, dict)]
            options_text = "\n".join(option_lines)
            if options_text:
                return f"{greeting}\n\nHere are your options:\n{options_text}"
            return greeting

        # Default web channel: return HTML buttons
        buttons = " ".join([f"<button>{lt.get('text', '')}</button>" for lt in lead_types if isinstance(lt, dict)])
        return f"{greeting} {buttons}"

