from typing import Any, Dict, List, Optional
import time

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

    async def agent_greet(self, context: Dict[str, Any], state: Dict[str, Any]) -> str:
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
        return await self._agent_generate(history=[], user_message="__INIT__", context=context, state=state, is_init=True)

    async def agent_reply(self, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any], state: Dict[str, Any]) -> str:
        if not self.client:
            return "Got it."
        return await self._agent_generate(history=history, user_message=user_message, context=context, state=state, is_init=False)

    async def _agent_generate(self, *, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any], state: Dict[str, Any], is_init: bool) -> str:
        import json
        
        # Use custom greeting from context if available
        integration = context.get("integration", {})
        custom_greeting = integration.get("greeting", "Hi! How can i help u today?")
        
        system = (
            "You are a friendly and professional lead-generation assistant for a {profession}.\n"
            "Your goal is to have a natural conversation and collect all required information before creating a lead.\n\n"
            "CONVERSATION FLOW:\n"
            "1) Start with: '{greeting}' and present lead type options as buttons\n"
            "2) Ask about service type: 'Which service are you looking to avail?' and present ALL available service types and treatment plans as buttons (treat them as regular services)\n"
            "3) Get their full name: 'Great! What's your full name?'\n"
            "4) Get their email: 'Thank you, [Name]! Could you please provide your email address?'\n"
            "5) Get their phone: 'Perfect! And what's your phone number?'\n"
            "6) When you have ALL information, output ONLY the JSON lead (no other text)\n\n"
            "IMPORTANT RULES:\n"
            "- Be conversational and empathetic\n"
            "- Present options as: <button> Option Text </button>\n"
            "- Use the service_types AND treatment_plans from the context data to create service buttons\n"
            "- For treatment_plans, use the 'question' field as the button text\n"
            "- Present all services and treatment plans together as one unified list - do NOT ask about treatment plans separately\n"
            "- If user asks a question, answer it briefly and warmly, then continue with the next required step\n"
            "- Always guide the conversation back to collecting the required information\n"
            "- For email: If user provides invalid email, politely explain and ask again in the same message\n"
            "- Be natural and conversational - don't give cold error messages\n"
            "- If email validation is disabled in context, just collect email without OTP verification\n"
            "- For OTP: If user asks questions or provides non-OTP responses, answer naturally and guide them back to entering the code\n"
            "- For OTP: If user provides wrong OTP, be empathetic and ask them to try again in a friendly way\n"
            "- When you have ALL 5 fields (leadType, serviceType, leadName, leadEmail, leadPhoneNumber), output ONLY the JSON\n"
            "- JSON format: {{\"title\": \"...\", \"summary\": \"...\", \"description\": \"...\", \"leadName\": \"...\", \"leadPhoneNumber\": \"...\", \"leadEmail\": \"...\", \"leadType\": \"...\", \"serviceType\": \"...\"}}\n"
            "- NEVER show JSON to user or ask for confirmation - just output the JSON when ready\n"
            "- Do NOT add any text before or after the JSON - just the JSON object\n"
            "- IMPORTANT: After collecting phone number, immediately generate the JSON - do NOT ask for anything else\n"
            "- Do NOT repeat questions you've already asked - if you have all info, generate JSON\n"
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
        return (resp.choices[0].message.content or "").strip()
