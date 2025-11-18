from typing import Any, Dict, List, Optional, Tuple
import time
import re

from openai import AsyncOpenAI
import logging

logger = logging.getLogger("assistly.gpt")


class GptService:
    def __init__(self, settings: Any, rag_service: Optional[Any] = None) -> None:
        self.model: str = settings.gpt_model
        self.max_history: int = settings.max_history_messages
        self.client: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        )
        self.profession: str = "Clinic"
        self.rag_service: Optional[Any] = rag_service  # RAG service for retrieval
        self.current_context: Optional[Dict[str, Any]] = None  # Store current context for RAG

    def set_profession(self, profession: str) -> None:
        self.profession = profession or self.profession

    
    def _get_workflow_prompt(self, context: Dict[str, Any]) -> str:
        """Generate workflow instructions for GPT based on custom workflows"""
        workflows = context.get("workflows", [])
        if not workflows:
            return ""
        
        # Find root workflows
        root_workflows = [w for w in workflows if w.get("isRoot", False) and w.get("isActive", True)]
        
        if not root_workflows:
            return ""
        
        workflow_instructions = [
            "CUSTOM WORKFLOW CONVERSATION:",
            "You have custom workflow questions defined. Use them to guide the conversation:",
            ""
        ]
        
        # Build workflow tree information
        for workflow in root_workflows:
            workflow_instructions.append(f"ROOT QUESTION: {workflow.get('title', 'Untitled')}")
            workflow_instructions.append(f"Question: {workflow.get('question', '')}")
            
            # Get options
            options = workflow.get("options", [])
            if options:
                workflow_instructions.append("Options:")
                for opt in options:
                    opt_text = opt.get("text", "")
                    is_terminal = opt.get("isTerminal", False)
                    next_q_id = opt.get("nextQuestionId")
                    
                    if is_terminal:
                        workflow_instructions.append(f"  - {opt_text} (terminates conversation)")
                    elif next_q_id:
                        # Find the linked workflow
                        linked_wf = next((w for w in workflows if w.get("_id") == next_q_id), None)
                        if linked_wf:
                            workflow_instructions.append(f"  - {opt_text} → leads to: {linked_wf.get('question', '')}")
                        else:
                            workflow_instructions.append(f"  - {opt_text}")
                    else:
                        workflow_instructions.append(f"  - {opt_text}")
            
            workflow_instructions.append("")
        
        workflow_instructions.append("IMPORTANT: Follow the custom workflow structure. Present the options as buttons when appropriate.")
        workflow_instructions.append("When user selects an option, progress to the next question in the workflow.")
        workflow_instructions.append("")
        
        return "\n".join(workflow_instructions)

    async def short_reply(self, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any]) -> str:
        if not self.client:
            return "Here's a brief answer based on available info."

        import json

        system = (
            "You are a concise, friendly lead-generation assistant for a {profession}. "
            "Answer briefly (1-2 sentences). If unsure, say so. Then continue the flow."
        ).format(profession=self.profession)

        trimmed = history[-self.max_history :] if self.max_history > 0 else history
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(trimmed)
        
        # Pass raw JSON context to GPT
        context_json = json.dumps(context, indent=2)
        messages.append({
            "role": "system",
            "content": f"User context data:\n{context_json}",
        })
        messages.append({"role": "user", "content": user_message})

        start_time = time.time()
        logger.info("Sending GPT short_reply request at %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
            temperature=0.3,
            top_p=0.7,
            presence_penalty=0,
            frequency_penalty=0,
            stream=False,
        )

        end_time = time.time()
        duration = end_time - start_time
        logger.info("Received GPT short_reply response at %s (took %.3fs)", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration)
        return (resp.choices[0].message.content or "").strip() or "Okay."

    async def agent_greet(self, context: Dict[str, Any], state: Dict[str, Any], is_whatsapp: bool = False) -> str:
        if not self.client:
            # Minimal fallback using provided context
            lead_types = context.get("lead_types") or []
            # Use custom greeting from context if available
            integration = context.get("integration", {})
            greeting = integration.get("greeting", "How can i help u today")
            lines = [greeting]
            for lt in lead_types:
                txt = lt.get("text") if isinstance(lt, dict) else str(lt)
                lines.append(f"<button> {txt} </button>")
            return "\n".join(lines)
        
        # Initialize RAG (always enabled)
        if self.rag_service:
            self.current_context = context
            # Build vector store from context
            self.rag_service.build_vector_store(context)
            logger.info("RAG vector store built from context")
            
            # Try to get initial greeting from LangChain (uses greeting from context)
            try:
                rag_greeting = await self.rag_service.get_initial_greeting(
                    profession=self.profession,
                    is_whatsapp=is_whatsapp,
                    context_data=context
                )
                if rag_greeting:
                    logger.info("Using LangChain-generated greeting")
                    return rag_greeting
            except Exception as e:
                logger.warning(f"Could not get LangChain greeting, falling back to GPT: {e}")
        
        # Get workflow instructions if custom workflows exist
        workflow_prompt = self._get_workflow_prompt(context)
        if workflow_prompt:
            # Add workflow to state so it can be used in generation
            state["workflow_instructions"] = workflow_prompt
        
        return await self._agent_generate(history=[], user_message="__INIT__", context=context, state=state, is_init=True, is_whatsapp=is_whatsapp)

    async def agent_reply(self, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any], state: Dict[str, Any], is_whatsapp: bool = False) -> str:
        if not self.client:
            return "Got it."
        
        # Update RAG if context changed (RAG is always enabled)
        if self.rag_service and self.current_context != context:
            self.current_context = context
            self.rag_service.build_vector_store(context)
            logger.info("RAG vector store rebuilt from updated context")
        
        # Get workflow instructions if custom workflows exist
        workflow_prompt = self._get_workflow_prompt(context)
        if workflow_prompt:
            # Add workflow to state so it can be used in generation
            state["workflow_instructions"] = workflow_prompt
        
        return await self._agent_generate(history=history, user_message=user_message, context=context, state=state, is_init=False, is_whatsapp=is_whatsapp)

    async def _agent_generate(self, *, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any], state: Dict[str, Any], is_init: bool, is_whatsapp: bool = False) -> str:
        import json
        
        # Use custom greeting from context if available
        integration = context.get("integration", {})
        custom_greeting = integration.get("greeting", "Hi! How can i help u today?")
        
        # WhatsApp-specific flow (skip phone number collection)
        if is_whatsapp:
            flow_steps = (
                "CONVERSATION FLOW:\n"
                "1) Start with: '{greeting}' and along with that simply present lead type options as buttons\n"
                "2) MANDATORY: Ask about service type: 'Which service are you looking to avail?' and present ALL available service types and treatment plans as buttons (treat them as regular services) - THIS STEP IS REQUIRED FOR ALL LEAD TYPES INCLUDING CALLBACK\n"
                "3) Get their name: 'Great! What's your name?'\n"
                "4) Get their email: 'Thank you, [Name]! Could you please provide your email address?'\n"
                "5) IMMEDIATELY after collecting email, output ONLY the JSON lead (no other text)\n"
                "6) DO NOT ask for phone number - we already have it from WhatsApp\n"
                "7) DO NOT ask for anything else after email - just generate JSON\n\n"
            )
            json_fields = "4 fields (leadType, serviceType, leadName, leadEmail)"
            final_instruction = "CRITICAL: When you have ALL 4 fields (leadType, serviceType, leadName, leadEmail), output ONLY the JSON immediately"
        else:
            # Regular web chat flow (includes phone number)
            flow_steps = (
                "CONVERSATION FLOW:\n"
                "1) Start with: '{greeting}' and along with that simply present lead type options as buttons\n"
                "2) MANDATORY: Ask about service type: 'Which service are you looking to avail?' and present ALL available service types and treatment plans as buttons (treat them as regular services) - THIS STEP IS REQUIRED FOR ALL LEAD TYPES INCLUDING CALLBACK\n"
                "3) Get their name: 'Great! What's your name?'\n"
                "4) Get their email: 'Thank you, [Name]! Could you please provide your email address?'\n"
                "5) Get their phone: 'Perfect! And what's your phone number?'\n"
                "6) IMMEDIATELY after collecting phone number, output ONLY the JSON lead (no other text)\n"
                "7) DO NOT ask for anything else after phone number - just generate JSON\n\n"
            )
            json_fields = "5 fields (leadType, serviceType, leadName, leadEmail, leadPhoneNumber)"
            final_instruction = "CRITICAL: When you have ALL 5 fields (leadType, serviceType, leadName, leadEmail, leadPhoneNumber), output ONLY the JSON immediately"
        
        # Get validation flags to adjust flow
        validate_email = integration.get("validateEmail", True)
        validate_phone = integration.get("validatePhoneNumber", True)
        
        # Simplified prompt - LangChain handles everything
        # Note: For WhatsApp, phone is already verified, so skip phone OTP steps even if phone validation is enabled
        if is_whatsapp:
            if validate_email:
                flow = "lead type → treatment plan → name → email → send email OTP → verify email OTP → JSON (phone from WhatsApp, already verified)"
            else:
                flow = "lead type → treatment plan → name → email → JSON (phone from WhatsApp, already verified)"
            json_example = '{"leadType": "...", "serviceType": "...", "leadName": "...", "leadEmail": "..."}'
        else:
            # For web chat, include phone OTP steps if phone validation is enabled
            if validate_email and validate_phone:
                flow = "lead type → treatment plan → name → email → send email OTP → verify email OTP → phone → send phone OTP → verify phone OTP → JSON"
            elif validate_email:
                flow = "lead type → treatment plan → name → email → send email OTP → verify email OTP → phone → JSON"
            elif validate_phone:
                flow = "lead type → treatment plan → name → email → phone → send phone OTP → verify phone OTP → JSON"
            else:
                flow = "lead type → treatment plan → name → email → phone → JSON"
            json_example = '{"leadType": "...", "serviceType": "...", "leadName": "...", "leadEmail": "...", "leadPhoneNumber": "..."}'
        
        system = (
            f"You are a {self.profession} assistant. Follow the flow: {flow}.\n"
            f"Match user input to options from context. Use exact values. Be conversational.\n"
            f"Special responses: 'SEND_EMAIL: [email]', 'SEND_PHONE: [phone]', 'RETRY_OTP_REQUESTED', 'CHANGE_PHONE_REQUESTED: [phone]', 'CHANGE_EMAIL_REQUESTED: [email]'.\n"
            f"IMPORTANT: Complete ALL OTP verification steps before generating JSON. Do NOT skip OTP steps.\n"
            f"When all info collected AND OTP verification complete (if required), output ONLY JSON: {json_example}"
        )

        # Use LangChain for ALL responses - get accurate answer from RAG
        rag_answer = None
        if (self.rag_service and 
            not is_init and 
            user_message != "__INIT__"):
            try:
                # Get accurate answer using LangChain QA chain - handles everything
                # Pass FULL context and conversation history to RAG so it knows the complete flow state
                # Include current user message in history for context
                # Use FULL history (not trimmed) so LangChain can see the entire conversation from the start
                history_with_current = history + [{"role": "user", "content": user_message}]
                rag_answer = await self.rag_service.get_accurate_answer(
                    user_message, 
                    profession=self.profession,
                    is_whatsapp=is_whatsapp,
                    context_data=context,
                    conversation_history=history_with_current
                )
                
                if rag_answer:
                    logger.info(f"LangChain generated response: {rag_answer[:50]}...")
                    # Use LangChain's answer directly - it's already correct and contextual
                    return rag_answer
            except Exception as e:
                logger.error(f"Error getting LangChain answer: {e}")
        
        # Minimal messages - LangChain handles context via vector store
        trimmed = history[-self.max_history :] if self.max_history > 0 else history
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(trimmed)
        if is_init:
            messages.append({"role": "user", "content": "__INIT__"})
        else:
            messages.append({"role": "user", "content": user_message})

        start_time = time.time()
        request_type = "agent_greet" if is_init else "agent_reply"
        logger.info("Sending GPT %s request at %s", request_type, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
        
        if not is_init:
            logger.info("User message to GPT: '%s'", user_message)

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=200,
            temperature=0.3,
            top_p=0.7,
            presence_penalty=0,
            frequency_penalty=0,
            stream=False,
        )

        end_time = time.time()
        duration = end_time - start_time
        logger.info("Received GPT %s response at %s (took %.3fs)", request_type, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration)
        
        response_content = (resp.choices[0].message.content or "").strip()
        logger.info("GPT response content: '%s'", response_content)
        
        return response_content
    
    def extract_buttons_from_response(self, response: str) -> Tuple[str, List[Dict[str, str]]]:
        """Extract buttons from GPT response and return cleaned text + button data"""
        buttons = []
        
        # Single comprehensive pattern to match button tags with various formats
        # This pattern handles: <button>text</button>, < button >text</ button >, <button>text</button, etc.
        # But avoids matching malformed tags like <button>text</button> where > is part of content
        button_pattern = r'<\s*button[^>]*>\s*([^<]+?)\s*</\s*button[^>]*>'
        
        # Extract all buttons with the single pattern
        button_matches = re.findall(button_pattern, response, re.IGNORECASE | re.DOTALL)
        
        # Use a set to track unique button texts (case-insensitive)
        seen_buttons = set()
        for button_text in button_matches:
            clean_text = button_text.strip()
            # Remove any leading ">" characters that might have been captured
            clean_text = clean_text.lstrip('>').strip()
            
            if clean_text and clean_text.lower() not in seen_buttons:
                seen_buttons.add(clean_text.lower())
                buttons.append({
                    "id": f"button_{len(buttons) + 1}",
                    "title": clean_text
                })
        
        logger.info(f"Extracted {len(buttons)} unique buttons: {[btn['title'] for btn in buttons]}")
        
        # Remove all button tag variations from response text
        cleaned_response = re.sub(button_pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
        
        # Also remove any remaining malformed button tags
        cleaned_response = re.sub(r'<\s*button[^>]*>.*?</\s*button[^>]*>', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        cleaned_response = re.sub(r'<\s*button[^>]*>.*?$', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        
        cleaned_response = cleaned_response.strip()
        
        return cleaned_response, buttons
    
    def create_whatsapp_buttons_message(self, text: str, buttons: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create WhatsApp interactive button message format"""
        return {
            "type": "interactive_buttons",
            "text": text,
            "buttons": buttons
        }
    
    def create_whatsapp_list_message(self, text: str, button_text: str, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create WhatsApp interactive list message format"""
        return {
            "type": "interactive_list",
            "text": text,
            "button_text": button_text,
            "sections": sections
        }
