from typing import Any, Dict, List, Optional, Tuple
import time
import re

from openai import AsyncOpenAI
import logging

logger = logging.getLogger("assistly.gpt")


class GptService:
    def __init__(self, settings: Any) -> None:
        self.model: str = settings.gpt_model
        self.max_history: int = settings.max_history_messages
        self.client: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        )
        self.profession: str = "Clinic"

    def set_profession(self, profession: str) -> None:
        self.profession = profession or self.profession

    def _merge_treatment_plans_into_services(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge treatment plans into service_types array for unified service selection"""
        service_types = context.get("service_types", [])
        treatment_plans = context.get("treatment_plans", [])
        
        # Convert treatment plans to service format and merge with service types
        merged_services = list(service_types)  # Start with existing services
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
        
        # Merge treatment plans into service types for unified service selection
        context = self._merge_treatment_plans_into_services(context)
        
        # Get workflow instructions if custom workflows exist
        workflow_prompt = self._get_workflow_prompt(context)
        if workflow_prompt:
            # Add workflow to state so it can be used in generation
            state["workflow_instructions"] = workflow_prompt
        
        return await self._agent_generate(history=[], user_message="__INIT__", context=context, state=state, is_init=True, is_whatsapp=is_whatsapp)

    async def agent_reply(self, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any], state: Dict[str, Any], is_whatsapp: bool = False) -> str:
        if not self.client:
            return "Got it."
        
        # Merge treatment plans into service types for unified service selection
        context = self._merge_treatment_plans_into_services(context)
        
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
        
        # Get workflow instructions if they exist
        workflow_instructions = state.get("workflow_instructions", "")
        
        system = (
            "You are a friendly and professional lead-generation assistant for a {profession}.\n"
            "Your goal is to have a natural conversation and collect all required information before creating a lead.\n\n"
        )
        
        # Add workflow instructions if they exist
        if workflow_instructions:
            system += workflow_instructions
        
        system += (
            "CRITICAL SYSTEM INSTRUCTIONS:\n"
            "- When user indicates they want to resend OTP (lost, didn't receive, send again), respond with ONLY: 'RETRY_OTP_REQUESTED'\n"
            "- When user indicates they want to change their phone number, respond with ONLY: 'CHANGE_PHONE_REQUESTED'\n"
            "- When user indicates they want to change their email address, respond with ONLY: 'CHANGE_EMAIL_REQUESTED'\n"
            "- CRITICAL: Use semantic understanding to detect user intent: if user provides different contact information than previously collected, this indicates a change request\n"
            "- CRITICAL: Use semantic understanding to detect user intent: if user wants to resend to the same contact information, this indicates a retry request\n"
            "- These special phrases are REQUIRED - do NOT add any other text or explanations\n"
            "- Use semantic understanding to detect user intent, not exact word matching\n\n"
            + flow_steps +
            "VALIDATION RULES:\n"
            "- Lead Type: Must be exactly one of the provided button options (use the 'value' field)\n"
            "- Service Type: MANDATORY - Must be intelligently matched to one of the provided service or treatment plan options using semantic understanding and natural language processing. THIS FIELD IS REQUIRED FOR ALL LEAD TYPES INCLUDING CALLBACK\n"
            "- Name: Must not be John Doe or Jane Doe\n"
            "- Email: Must be valid email format (contains @ and domain)\n"
            + ("" if is_whatsapp else "- Phone: Must be a valid phone number (digits, reasonable length)\n") +
            "- If any input is invalid, politely ask for correction and show options again\n\n"
            "IMPORTANT RULES:\n"
            "- Be conversational and empathetic\n"
            "- Present options as: <button> Option Text </button>\n"
            "- Use the service_types array from context data to create service buttons (treatment plans are already merged into service_types)\n"
            "- Present all services and treatment plans together as one unified list - do NOT ask about treatment plans separately\n"
            "- INTELLIGENT SERVICE MATCHING: Use semantic understanding to match user's natural language to available services. Analyze the user's intent and match it to the closest service from the provided options.\n"
            "- When user mentions any service-related terms, intelligently determine which service they're referring to and proceed immediately\n"
            "- SERVICE QUESTIONS: If user asks questions about a service (pricing, discounts, details, rates, costs), ALWAYS answer them briefly and warmly FIRST, then continue with the next step (asking for name)\n"
            "- Use your knowledge and understanding to provide helpful, accurate responses about services, pricing, and policies\n"
            "- CRITICAL: Do NOT ask for confirmation or show options again if the match is clear\n"
            "- Only show options if the user's request is completely unclear or doesn't match any available services\n"
            "- If user asks a question, answer it briefly and warmly, then continue with the next required step\n"
            "- Always guide the conversation back to collecting the required information\n"
            "- For email: If user provides invalid email format, politely explain and ask again in the same message.\n"
            "- For lead type: If user doesn't select from provided options, politely say 'Please choose from the options above' and show the buttons again\n"
            "- For service type:Always show service type after lead type. Use semantic understanding to intelligently match user's natural language to available services. Analyze their intent and match to the closest service from the provided options.\n"
            "- If user asks questions about the service (pricing, discounts, details), answer them briefly and warmly, then continue to next step\n"
            "- Only ask for clarification if the user's request is completely unclear or doesn't match any available services\n"
            "- For name: If user provides fake name like John Doe, politely say 'Please provide your correct name' and ask again\n"
            "- For email: If user provides invalid email format, politely say 'Please provide a valid email address' and ask again\n"
            "- For phone: If user provides invalid phone number, politely say 'Please provide a valid phone number' and ask again\n"
            "- CRITICAL RETRY HANDLING: When user says they lost, didn't receive, or want to resend OTP, you MUST respond with ONLY: 'RETRY_OTP_REQUESTED'\n"
            "- CRITICAL RETRY HANDLING: When user says they want to change phone number, you MUST respond with ONLY: 'CHANGE_PHONE_REQUESTED'\n"
            "- CRITICAL RETRY HANDLING: When user says they want to change email, you MUST respond with ONLY: 'CHANGE_EMAIL_REQUESTED'\n"
            "- CRITICAL: These special phrases are REQUIRED for the system to work. Do NOT add any other text, explanations, or responses.\n"
            "- CRITICAL: If user asks for OTP resend, respond with ONLY 'RETRY_OTP_REQUESTED' - nothing else\n"
            "- Be natural and conversational - don't give cold error messages\n"
            "- If email validation is disabled in context, just collect email without OTP verification\n"
            "- For OTP: If user asks questions or provides non-OTP responses, answer naturally and guide them back to entering the code\n"
            "- For OTP: If user provides wrong OTP, be empathetic and ask them to try again in a friendly way\n"
            "- For OTP: If user indicates they didn't receive the code or it's not working, respond with: 'RETRY_OTP_REQUESTED' - DO NOT provide any other text\n"
            "- For OTP: If user indicates they provided wrong contact information, respond with: 'CHANGE_PHONE_REQUESTED' or 'CHANGE_EMAIL_REQUESTED' - DO NOT provide any other text\n"
            + ("" if is_whatsapp else "- For phone: If user provides invalid phone number, politely explain the issue in their response in a way a simple user can understand like you didnt provide bla bla i asked for bla bla and ask again in the same message\n") +
            ("" if is_whatsapp else "- For phone OTP: Handle phone verification naturally like email verification\n") +
            ("- CRITICAL: The flow should be ask lead type first, then service type, then name, then email (we have phone from WhatsApp)\n" if is_whatsapp else "- CRITICAL: The flow should be ask lead type first, then service type, then name, then email and then phone number\n") +
            "- CRITICAL: When user selects a service (either by number or natural language), immediately proceed to the next step (asking for name). Do NOT ask for confirmation or show options again.\n" +
            f"- {final_instruction}\n"
            "- JSON format: {{\"title\": \"...\", \"summary\": \"...\", \"description\": \"...\", \"leadName\": \"...\", \"leadPhoneNumber\": \"...\", \"leadEmail\": \"...\", \"leadType\": \"...\", \"serviceType\": \"...\"}}\n"
            "- IMPORTANT: Use the 'value' field from lead_types for leadType (e.g., 'callback', 'appointment arrangement', 'further information')\n"
            "- NEVER show JSON to user or ask for confirmation - just output the JSON when ready\n"
            "- Help the user, if the user asks you to show something again do that. Help in booking\n"
            "- Do NOT add any text before or after the JSON - just the JSON object\n"
            "- CRITICAL: After collecting phone number, immediately generate the JSON - do NOT ask for anything else\n"
            "- Do NOT repeat questions you've already asked - if you have all info, generate JSON\n"
            + (
                "- REMEMBER: Once you have leadType, serviceType, leadName, and leadEmail - output JSON immediately (phone from WhatsApp)\n"
                "- FINAL STEP: After email verification, generate JSON immediately - do NOT ask for more information\n"
                if is_whatsapp else
                "- REMEMBER: Once you have leadType, serviceType, leadName, leadEmail, and leadPhoneNumber - output JSON immediately\n"
                "- FINAL STEP: After phone number collection, generate JSON immediately - do NOT ask for more information\n"
                "- SPECIAL CASE: If the last message mentions 'phone number has been verified' and you have all 5 fields, generate JSON immediately\n"
            )
        ).format(profession=self.profession, greeting=custom_greeting)

        trimmed = history[-self.max_history :] if self.max_history > 0 else history
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        
        # Pass raw JSON context to GPT
        context_json = json.dumps(context, indent=2)
        messages.append({"role": "system", "content": f"User context data:\n{context_json}"})
        
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
