"""Production-grade response generator using state machine and minimal prompts"""
from typing import Dict, Any, List, Optional
import logging
from openai import AsyncOpenAI

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
    
    def set_profession(self, profession: str):
        """Set profession for responses"""
        self.profession = profession or self.profession
    
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
        
        # Extract data from user message (only extract what we need based on state)
        email = extractor.extract_email(user_message)
        phone = extractor.extract_phone(user_message)
        otp_code = extractor.extract_otp_code(user_message)
        
        # Only extract name when we're in NAME_COLLECTION state (prevents lead type text from being extracted as name)
        name = None
        if state == ConversationState.NAME_COLLECTION:
            name = extractor.extract_name(user_message, context.get("lead_types", []))
        
        # Handle OTP verification states
        if state == ConversationState.EMAIL_OTP_VERIFICATION:
            if otp_code and validator.is_valid_otp(otp_code):
                return "OTP_VERIFY_EMAIL:" + otp_code
            return "That code doesn't look right. Please check and try entering the 6-digit code again."
        
        if state == ConversationState.PHONE_OTP_VERIFICATION:
            if otp_code and validator.is_valid_otp(otp_code):
                return "OTP_VERIFY_PHONE:" + otp_code
            return "That code doesn't look right. Please check and try entering the 6-digit code again."
        
        # Handle data collection states - extract and validate before AI generation
        if state == ConversationState.LEAD_TYPE_SELECTION:
            lead_type = extractor.match_lead_type(user_message, context.get("lead_types", []))
            if lead_type:
                flow_controller.update_collected_data("leadType", lead_type.get("value"))
                flow_controller.transition_to(flow_controller.get_next_state())
                # Explicitly include services in prompt for service selection
                return await self._generate_service_selection_response(conversation_history, context)
        
        if state == ConversationState.SERVICE_SELECTION:
            services = context.get("service_types", [])
            treatment_plans = context.get("treatment_plans", [])
            all_services = services + [tp.get("question", "") for tp in treatment_plans if isinstance(tp, dict)]
            service = extractor.match_service(user_message, all_services)
            if service:
                flow_controller.update_collected_data("serviceType", service)
                flow_controller.transition_to(flow_controller.get_next_state())
                rag_context = await self._get_rag_context("name collection", context)
                return await self._generate_state_response(flow_controller.state, rag_context, conversation_history, context)
        
        if state == ConversationState.NAME_COLLECTION:
            if name and validator.is_valid_name(name):
                flow_controller.update_collected_data("leadName", name)
                flow_controller.transition_to(flow_controller.get_next_state())
                rag_context = await self._get_rag_context("email collection", context)
                return await self._generate_state_response(flow_controller.state, rag_context, conversation_history, context)
        
        if state == ConversationState.EMAIL_COLLECTION:
            if email and validator.is_valid_email(email):
                flow_controller.update_collected_data("leadEmail", email)
                if flow_controller.validate_email:
                    return "SEND_EMAIL:" + email
                else:
                    flow_controller.transition_to(flow_controller.get_next_state())
                    # Check if we can generate JSON (WhatsApp) or need phone
                    if flow_controller.can_generate_json():
                        return self._generate_json(flow_controller)
                    else:
                        rag_context = await self._get_rag_context("phone collection", context)
                        return await self._generate_state_response(flow_controller.state, rag_context, conversation_history, context)
        
        if state == ConversationState.PHONE_COLLECTION:
            if phone and validator.is_valid_phone(phone):
                flow_controller.update_collected_data("leadPhoneNumber", phone)
                if flow_controller.validate_phone:
                    return "SEND_PHONE:" + phone
                else:
                    flow_controller.transition_to(flow_controller.get_next_state())
                    # Generate JSON
                    return self._generate_json(flow_controller)
        
        # Check if all data is collected and we can generate JSON
        if flow_controller.can_generate_json():
            return self._generate_json(flow_controller)
        
        # Generate natural response based on state
        rag_context = await self._get_rag_context(flow_controller.get_state_prompt_context(), context)
        return await self._generate_state_response(state, rag_context, conversation_history, context)
    
    async def _get_rag_context(self, query: str, context: Dict[str, Any]) -> str:
        """Get relevant context from RAG"""
        if not self.rag_service or not self.rag_service.retriever:
            return ""
        
        try:
            docs = self.rag_service.retriever.get_relevant_documents(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return ""
    
    async def _generate_service_selection_response(
        self,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> str:
        """Generate service selection response with explicit service list"""
        if not self.client:
            return "Which service are you interested in?"
        
        # Get all services
        services = context.get("service_types", [])
        treatment_plans = context.get("treatment_plans", [])
        all_services = []
        
        for s in services:
            if isinstance(s, dict):
                all_services.append(s.get("name", s.get("title", "")))
            else:
                all_services.append(str(s))
        
        for tp in treatment_plans:
            if isinstance(tp, dict):
                all_services.append(tp.get("question", ""))
        
        # Format services as buttons
        services_text = " ".join([f"<button>{s}</button>" for s in all_services if s])
        
        system_prompt = f"You are a {self.profession} assistant. CRITICAL: Ask 'Which service are you interested in?' and show these services: {services_text}. DO NOT ask for date/time or any other information. Service selection is MANDATORY and comes immediately after lead type selection."
        
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
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error generating service selection response: {e}")
            return f"Which service are you interested in? {services_text}"
    
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
        state_prompts = {
            ConversationState.GREETING: f"You are a {self.profession} assistant. Greet the user and present lead type options from context.",
            ConversationState.LEAD_TYPE_SELECTION: f"You are a {self.profession} assistant. Present lead type options from context and wait for selection.",
            ConversationState.SERVICE_SELECTION: f"You are a {self.profession} assistant. CRITICAL: Ask 'Which service are you interested in?' and present ALL service options from context. DO NOT ask for date/time or any other information. Service selection is MANDATORY.",
            ConversationState.NAME_COLLECTION: f"You are a {self.profession} assistant. Ask for the user's name naturally. DO NOT ask for date/time or other information.",
            ConversationState.EMAIL_COLLECTION: f"You are a {self.profession} assistant. Ask for the user's email address naturally. DO NOT ask for date/time or other information.",
            ConversationState.PHONE_COLLECTION: f"You are a {self.profession} assistant. Ask for the user's phone number naturally. DO NOT ask for date/time or other information.",
        }
        
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
    
    def _generate_json(self, flow_controller: FlowController) -> str:
        """Generate JSON from collected data"""
        import json
        data = flow_controller.get_json_data()
        flow_controller.transition_to(ConversationState.COMPLETE)
        return json.dumps(data)
    
    async def generate_greeting(self, context: Dict[str, Any], is_whatsapp: bool = False) -> str:
        """Generate initial greeting"""
        integration = context.get("integration", {})
        greeting = integration.get("greeting", "Hi! How can I help you today?")
        lead_types = context.get("lead_types", [])
        
        if is_whatsapp:
            # Numbered list for WhatsApp
            options = "\n".join([f"{i}. {lt.get('text', '')}" for i, lt in enumerate(lead_types, 1) if isinstance(lt, dict)])
            return f"{greeting}\n\n{options}\n\nPlease reply with the number of your choice."
        else:
            # Buttons for web
            buttons = " ".join([f"<button>{lt.get('text', '')}</button>" for lt in lead_types if isinstance(lt, dict)])
            return f"{greeting} {buttons}"

