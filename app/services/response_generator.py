"""Production-grade response generator using state machine and minimal prompts"""
from typing import Dict, Any, List, Optional
import logging
from openai import AsyncOpenAI
import json

from app.services.conversation_state import FlowController, ConversationState
from app.services.data_extractors import DataExtractor
from app.services.validators import Validator
from app.services.workflow_manager import WorkflowManager

logger = logging.getLogger("assistly.response_generator")


class ResponseGenerator:
    """Generate responses based on conversation state with minimal prompts"""
    
    def __init__(self, settings: Any, rag_service: Any):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.model = settings.gpt_model
        self.rag_service = rag_service
        self.profession = "Clinic"
        self.channel: str = "web"
    
    def set_profession(self, profession: str):
        """Set profession for responses"""
        self.profession = profession or self.profession

    def set_channel(self, channel: str):
        """Set current channel (web, whatsapp, voice)"""
        normalized = (channel or "web").lower()
        if normalized not in {"web", "whatsapp", "voice"}:
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
        state = flow_controller.state
        extractor = DataExtractor()
        validator = Validator()
        
        # Initialize or reuse workflow manager
        if flow_controller.workflow_manager is None:
            flow_controller.workflow_manager = WorkflowManager(context)
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
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_data_collected_with_question_response(
                        "lead type", lead_type.get("text"), rag_context, 
                        "service selection", conversation_history, context
                    )
                else:
                    # Explicitly include services in prompt for service selection
                    return await self._generate_service_selection_response(conversation_history, context)
            else:
                logger.warning(f"No lead type matched for user input: '{user_message}'. Available lead types: {[lt.get('text') for lt in context.get('lead_types', [])]}")
                # No match found - use AI to handle questions, but with strict prompt to only show lead types
                rag_context = await self._get_rag_context(user_message if has_question else "lead type selection", context, is_question=has_question)
                return await self._generate_state_response(state, rag_context, conversation_history, context)
        
        if state == ConversationState.SERVICE_SELECTION:
            # Use treatment plans directly
            treatment_plans = context.get("treatment_plans", [])
            # Extract treatment plan names for matching
            all_treatment_plans = []
            for plan in treatment_plans:
                if isinstance(plan, dict):
                    # Use "question" field as the identifier for treatment plans
                    plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                    if plan_name:
                        all_treatment_plans.append(plan_name)
                else:
                    all_treatment_plans.append(str(plan))
            
            # Try to match against treatment plans first
            service = extractor.match_service(user_message, all_treatment_plans)
            
            # Find the exact treatment plan name that was matched (for workflow detection)
            matched_treatment_plan_name = None
            if service:
                # Find the exact treatment plan name from the list (case-insensitive match)
                for plan_name in all_treatment_plans:
                    if plan_name.lower() == service.lower():
                        matched_treatment_plan_name = plan_name
                        break
                # If no exact match found, use the service as-is
                if not matched_treatment_plan_name:
                    matched_treatment_plan_name = service
            
            # If no match, accept user input as service type (user can choose any service)
            if not service:
                # Check if it's a question - if so, handle it but stay in service selection
                if has_question:
                    logger.info(f"User asked a question about services: '{user_message}'")
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
                
                # Accept user input as service type even if not in treatment plans
                service = user_message.strip()
                logger.info(f"Accepted user input as service type (not in treatment plans): '{service}'")
            
            if service:
                logger.info(f"Matched/selected service: {service}")
                flow_controller.update_collected_data("serviceType", service)
                
                # ALWAYS check for workflows first (even if user asked a question)
                # Use the exact treatment plan name for workflow detection
                workflow_started = False
                if matched_treatment_plan_name and matched_treatment_plan_name in all_treatment_plans:
                    logger.info(f"Checking for workflows for treatment plan: '{matched_treatment_plan_name}' (matched from service: '{service}')")
                    if workflow_manager.start_workflow_for_treatment_plan(matched_treatment_plan_name):
                        # Start workflow questions
                        flow_controller.transition_to(ConversationState.WORKFLOW_QUESTION)
                        workflow_started = True
                        logger.info(f"✓ Started workflow for treatment plan '{matched_treatment_plan_name}' - transitioning to WORKFLOW_QUESTION state")
                        current_question = workflow_manager.get_current_question()
                        if current_question:
                            question_text = current_question.get("question", "")
                            logger.info(f"✓ First workflow question: '{question_text}'")
                            # If user asked a question, answer it briefly then ask workflow question
                            if has_question:
                                # Get a brief answer to the question using RAG
                                try:
                                    rag_context = await self._get_rag_context(f"{service} {user_message}", context, is_question=True)
                                    if rag_context and self.client:
                                        # Generate a brief answer (1-2 sentences)
                                        response = await self.client.chat.completions.create(
                                            model=self.model,
                                            messages=[
                                                {"role": "system", "content": f"You are a {self.profession} assistant. Answer the question briefly in 1-2 sentences."},
                                                {"role": "user", "content": f"Context: {rag_context}\n\nQuestion: {user_message}"}
                                            ],
                                            max_tokens=100,
                                            temperature=0.3
                                        )
                                        brief_answer = (response.choices[0].message.content or "").strip()
                                        if brief_answer:
                                            logger.info(f"✓ Answering question then asking workflow question")
                                            return f"{brief_answer} Now, {question_text}"
                                except Exception as e:
                                    logger.warning(f"Failed to generate brief answer for question: {e}")
                                    # If answer generation fails, just ask workflow question
                            logger.info(f"✓ Returning workflow question (no user question to answer first)")
                            return question_text
                        else:
                            # No questions found, continue to name collection
                            logger.warning(f"Workflow started but no questions found - continuing to name collection")
                            workflow_manager.reset()
                            flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                    else:
                        # No workflow found
                        logger.info(f"No workflow found for treatment plan '{matched_treatment_plan_name}' - continuing to name collection")
                        workflow_manager.reset()
                
                # If workflow was NOT started, continue to name collection
                if not workflow_started:
                    logger.info(f"No workflow started - transitioning to name collection")
                    workflow_manager.reset()
                    flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                
                # Check if user asked a question along with service selection (only if no workflow)
                if has_question and flow_controller.state != ConversationState.WORKFLOW_QUESTION:
                    # Get RAG context for the question (pricing, info about the service)
                    rag_context = await self._get_rag_context(f"{service} {user_message}", context, is_question=True)
                    # Generate response that answers question AND asks for name
                    return await self._generate_data_collected_with_question_response(
                        "service", service, rag_context,
                        "name", conversation_history, context
                    )
                elif flow_controller.state != ConversationState.WORKFLOW_QUESTION:
                    # Just acknowledge and move to name collection - no RAG needed
                    return await self._generate_state_response(flow_controller.state, "", conversation_history, context)
            else:
                # This shouldn't happen now, but keep as fallback
                logger.warning(f"Could not determine service from user input: '{user_message}'. Available treatment plans: {all_treatment_plans}")
                # No match - use AI to handle questions, but ensure it stays in service selection
                rag_context = await self._get_rag_context(user_message if has_question else "service selection", context, is_question=has_question)
                return await self._generate_state_response(state, rag_context, conversation_history, context)
        
        if state == ConversationState.WORKFLOW_QUESTION:
            # Handle workflow question
            current_question = workflow_manager.get_current_question()
            if not current_question:
                # Workflow complete, store answers and move to name collection
                workflow_answers = workflow_manager.get_workflow_answers()
                flow_controller.update_collected_data("workflowAnswers", workflow_answers)
                workflow_manager.reset()
                flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                logger.info(f"Workflow complete. Transitioned to state: {flow_controller.state.value}")
                # No RAG needed for standard name collection prompt
                return await self._generate_state_response(flow_controller.state, "", conversation_history, context)
            
            # Check if user is asking a question instead of answering the workflow question
            if has_question:
                # User asked a question - answer it first, then re-ask the workflow question
                logger.info(f"User asked a question during workflow: '{user_message}'. Answering it first.")
                rag_context = await self._get_rag_context(user_message, context, is_question=True)
                current_question_text = current_question.get("question", "")
                
                # Generate answer to the question
                if rag_context and self.client:
                    try:
                        answer = await self._generate_question_response(user_message, rag_context, conversation_history, context)
                        # Return answer + re-ask workflow question
                        return f"{answer} Now, {current_question_text}"
                    except Exception as e:
                        logger.warning(f"Failed to generate answer for question: {e}")
                        # Fallback: just re-ask workflow question
                        return current_question_text
                else:
                    # No RAG context available, just re-ask workflow question
                    return current_question_text
            
            # Not a question - treat as workflow answer
            # Record answer and move to next question
            has_more = workflow_manager.record_answer(user_message)
            
            if has_more:
                # More questions remain, ask next question
                next_question = workflow_manager.get_current_question()
                if next_question:
                    return next_question.get("question", "")
                else:
                    # No more questions, complete workflow
                    workflow_answers = workflow_manager.get_workflow_answers()
                    flow_controller.update_collected_data("workflowAnswers", workflow_answers)
                    workflow_manager.reset()
                    flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                    logger.info(f"Workflow complete. Transitioned to state: {flow_controller.state.value}")
                    # No RAG needed for standard name collection prompt
                    return await self._generate_state_response(flow_controller.state, "", conversation_history, context)
            else:
                # Last question answered, complete workflow
                workflow_answers = workflow_manager.get_workflow_answers()
                flow_controller.update_collected_data("workflowAnswers", workflow_answers)
                workflow_manager.reset()
                flow_controller.transition_to(ConversationState.NAME_COLLECTION)
                logger.info(f"Workflow complete. Transitioned to state: {flow_controller.state.value}")
                # No RAG needed for standard name collection prompt
                return await self._generate_state_response(flow_controller.state, "", conversation_history, context)
        
        if state == ConversationState.NAME_COLLECTION:
            if name and validator.is_valid_name(name):
                flow_controller.update_collected_data("leadName", name)
                flow_controller.transition_to(flow_controller.get_next_state())
                logger.info(f"Transitioned to state: {flow_controller.state.value}")
                
                # Check if user asked a question along with name
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_data_collected_with_question_response(
                        "name", name, rag_context,
                        "email", conversation_history, context
                    )
                else:
                    # No question - just move to email collection
                    return await self._generate_state_response(flow_controller.state, "", conversation_history, context)
            else:
                # Name not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
                else:
                    # Just ask for name again - no RAG needed
                    return await self._generate_state_response(state, "", conversation_history, context)
        
        if state == ConversationState.EMAIL_COLLECTION:
            if email and validator.is_valid_email(email):
                flow_controller.update_collected_data("leadEmail", email)
                
                # Check if user asked a question along with email
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
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
                            # No RAG needed for standard phone collection prompt
                            return await self._generate_state_response(flow_controller.state, "", conversation_history, context)
            else:
                # Email not extracted - check if user asked a question
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
                else:
                    # No RAG needed for standard email collection prompt
                    return await self._generate_state_response(state, "", conversation_history, context)
        
        if state == ConversationState.PHONE_COLLECTION:
            if phone and validator.is_valid_phone(phone):
                flow_controller.update_collected_data("leadPhoneNumber", phone)
                
                # Check if user asked a question along with phone
                if has_question:
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
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
                    rag_context = await self._get_rag_context(user_message, context, is_question=True)
                    return await self._generate_state_response(state, rag_context, conversation_history, context)
                else:
                    # No RAG needed for standard phone collection prompt
                    return await self._generate_state_response(state, "", conversation_history, context)
        
        # Check if all data is collected and we can generate JSON
        if flow_controller.can_generate_json():
            return await self._generate_json(flow_controller, conversation_history)
        
        # Generate natural response based on state - no RAG needed for standard prompts
        return await self._generate_state_response(state, "", conversation_history, context)
    
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
        context: Dict[str, Any]
    ) -> str:
        """Generate service selection response with explicit treatment plan list"""
        if not self.client:
            return "Which service are you interested in?"
        
        # Get all treatment plans
        treatment_plans = context.get("treatment_plans", [])
        all_services = []
        
        for plan in treatment_plans:
            if isinstance(plan, dict):
                # Use "question" field as the identifier for treatment plans
                plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                if plan_name:
                    all_services.append(plan_name)
            else:
                all_services.append(str(plan))
        
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
            services_text = " ".join([f"<button>{s}</button>" for s in all_services if s])
            
            # For web and WhatsApp, always return service buttons immediately
            # No need for LLM - just show the services directly
            if services_text:
                return f"Which service are you interested in? {services_text}"
            else:
                return "Which service are you interested in?"
    
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
                # For web and WhatsApp, include service buttons
                if self.channel != "voice":
                    treatment_plans = context.get("treatment_plans", [])
                    all_services = []
                    for plan in treatment_plans:
                        if isinstance(plan, dict):
                            plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                            if plan_name:
                                all_services.append(plan_name)
                        else:
                            all_services.append(str(plan))
                    
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
            
            # For web and WhatsApp, if next step is service selection, add service buttons
            if next_step == "service selection" and self.channel != "voice":
                treatment_plans = context.get("treatment_plans", [])
                all_services = []
                for plan in treatment_plans:
                    if isinstance(plan, dict):
                        plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                        if plan_name:
                            all_services.append(plan_name)
                    else:
                        all_services.append(str(plan))
                
                services_text = " ".join([f"<button>{s}</button>" for s in all_services if s])
                if services_text and services_text not in answer:
                    answer += f" {services_text}"
            
            return answer
        except Exception as e:
            logger.error(f"Error generating data+question response: {e}")
            fallback = f"Great! I've noted your {data_type}: {data_value}. {next_question}"
            
            # For web and WhatsApp, if next step is service selection, add service buttons
            if next_step == "service selection" and self.channel != "voice":
                treatment_plans = context.get("treatment_plans", [])
                all_services = []
                for plan in treatment_plans:
                    if isinstance(plan, dict):
                        plan_name = plan.get("question", plan.get("name", plan.get("title", "")))
                        if plan_name:
                            all_services.append(plan_name)
                    else:
                        all_services.append(str(plan))
                
                services_text = " ".join([f"<button>{s}</button>" for s in all_services if s])
                if services_text:
                    fallback += f" {services_text}"
            
            return fallback
    
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
                return f"Hi this is {assistant_name} your virtual ai assistant from Palm Dental Services. How can I help u today"
            else:
                return "Hi this is your virtual ai assistant from Palm Dental Services. How can I help u today"
        
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

            treatment_plans = context.get("treatment_plans", [])
            service_names = []
            for plan in treatment_plans:
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
        
        system_prompt = state_prompts.get(state, f"You are a {self.profession} assistant. Continue the conversation naturally.")
        
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
    
    async def generate_greeting(self, context: Dict[str, Any], channel: Optional[str] = None) -> str:
        """Generate initial greeting"""
        current_channel = (channel or self.channel or "web").lower()
        integration = context.get("integration", {})
        greeting = integration.get("greeting", "").strip()
        
        # Fallback to default greeting if empty or missing
        if not greeting:
            assistant_name = integration.get("assistantName", "").strip()
            if assistant_name:
                greeting = f"Hi this is {assistant_name} your virtual ai assistant from Palm Dental Services. How can I help u today"
            else:
                greeting = "Hi this is your virtual ai assistant from Palm Dental Services. How can I help u today"
        
        lead_types = context.get("lead_types", [])

        if current_channel == "whatsapp":
            # Numbered list for WhatsApp
            options = "\n".join([f"{i}. {lt.get('text', '')}" for i, lt in enumerate(lead_types, 1) if isinstance(lt, dict)])
            return f"{greeting}\n\n{options}\n\nPlease reply with the number of your choice."
        elif current_channel == "voice":
            option_names = []
            for lt in lead_types:
                if isinstance(lt, dict):
                    option_names.append(self._normalize_lead_option_for_voice(lt.get("text", "")))
                else:
                    option_names.append(self._normalize_lead_option_for_voice(str(lt)))
            options_text = self._format_voice_list(option_names)
            if options_text:
                return f"{greeting}\n\nWould you like {options_text}?"
            return greeting
        else:
            # Buttons for web
            buttons = " ".join([f"<button>{lt.get('text', '')}</button>" for lt in lead_types if isinstance(lt, dict)])
            return f"{greeting} {buttons}"

