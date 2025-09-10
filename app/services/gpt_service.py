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

        faqs = context.get("faqs") or []
        faq_text = "\n".join(
            f"- Q: {f.get('q') or f.get('question')}: A: {f.get('a') or f.get('answer')}"
            if isinstance(f, dict) else f"- {str(f)}"
            for f in faqs[:6]
        )

        system = (
            "You are a concise, friendly lead-generation assistant for a {profession}. "
            "Answer briefly (1-2 sentences). If unsure, say so. Then continue the flow."
        ).format(profession=self.profession)

        trimmed = history[-self.max_history :] if self.max_history > 0 else history
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(trimmed)
        if faq_text:
            messages.append({
                "role": "system",
                "content": f"FAQ context (may help):\n{faq_text}",
            })
        messages.append({"role": "user", "content": user_message})

        start_time = time.time()
        logger.info("Sending GPT short_reply request at %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
            temperature=0.2,
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
                lines.append(f"#Button# {txt} #Button#")
            return "\n".join(lines)
        return await self._agent_generate(history=[], user_message="__INIT__", context=context, state=state, is_init=True)

    async def agent_reply(self, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any], state: Dict[str, Any]) -> str:
        if not self.client:
            return "Got it."
        return await self._agent_generate(history=history, user_message=user_message, context=context, state=state, is_init=False)

    async def _agent_generate(self, *, history: List[Dict[str, str]], user_message: str, context: Dict[str, Any], state: Dict[str, Any], is_init: bool) -> str:
        lead_types = context.get("lead_types") or []
        service_types = context.get("service_types") or []
        faqs = context.get("faqs") or []

        lead_types_text = "\n".join(
            f"- {lt.get('text')} | value={lt.get('value')}" if isinstance(lt, dict) else f"- {str(lt)}"
            for lt in lead_types
        )
        service_types_text = "\n".join(f"- {str(s)}" for s in service_types)
        faq_text = "\n".join(
            f"- Q: {f.get('question')}: A: {f.get('answer')}" if isinstance(f, dict) else f"- {str(f)}"
            for f in faqs[:10]
        )

        # Use custom greeting from context if available
        integration = context.get("integration", {})
        custom_greeting = integration.get("greeting", "Hi! How can i help u today?")
        
        system = (
            "You are an empathetic lead-generation agent for a {profession}.\n"
            "GOAL: collect leadType -> serviceType -> leadName -> leadEmail -> leadPhoneNumber.\n"
            "RULES:\n"
            "1) On init, greet with: '{greeting}' followed by selectable lead types.\n"
            "2) Next, ask for service type by saying 'Which service are you looking to avail?' (use provided options if any in selectable same format as lead types).\n"
            "3) Then ask for name, then email, then phone.\n"
            "4) If user asks a question, answer shortly and warmly with emotions, then continue.\n"
            "5) Wrap each selectable option line exactly as: #Button# <text> #Button# (no numbers, one per line).\n"
            "6) When all required fields are collected, output ONLY a JSON object with keys:\n"
            "   title, summary, description, leadName, leadPhoneNumber (string or null), leadEmail (string or null), leadType, serviceType.\n"
            "   Do not add any commentary around the JSON.\n"
        ).format(profession=self.profession, greeting=custom_greeting)

        trimmed = history[-self.max_history :] if self.max_history > 0 else history
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        # Provide context to the model
        context_block = (
            f"Context:\n"
            f"Lead types (text|value):\n{lead_types_text or '- (none provided)'}\n\n"
            f"Service types:\n{service_types_text or '- (none provided)'}\n\n"
            f"FAQs:\n{faq_text or '- (none)'}\n\n"
            f"Known state JSON: {state}"
        )
        messages.append({"role": "system", "content": context_block})
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
            max_tokens=100,
            temperature=0.2,
            top_p=0.7,
            presence_penalty=0,
            frequency_penalty=0,
            stream=False,
        )

        end_time = time.time()
        duration = end_time - start_time
        logger.info("Received GPT %s response at %s (took %.3fs)", request_type, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration)
        return (resp.choices[0].message.content or "").strip()
