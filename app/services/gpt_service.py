from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


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

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=120,
        )
        return (resp.choices[0].message.content or "").strip() or "Okay."
