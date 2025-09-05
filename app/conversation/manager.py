import re
from typing import Any, Dict, List, Optional, Tuple


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


class ConversationManager:
    def __init__(self, *, gpt: Any, lead_service: Any, context: Dict[str, Any], user_id: Optional[str] = None) -> None:
        self.gpt = gpt
        self.lead_service = lead_service
        self.context = context
        self.user_id = user_id

        self.step: str = "lead_type"
        self.state: Dict[str, Any] = {
            "lead_type": None,
            "service_type": None,
            "name": None,
            "phone": None,
            "history": [],
        }

        # Build lead type options from backend-provided objects
        self.lead_type_options: List[Dict[str, str]] = []
        for item in (context.get("lead_types") or []):
            if isinstance(item, dict):
                value = str(item.get("value") or "")
                text = str(item.get("text") or value)
                self.lead_type_options.append({"value": value, "text": text})
            else:
                sval = str(item)
                self.lead_type_options.append({"value": sval, "text": sval})
        self.service_types: List[str] = [str(x) for x in context.get("service_types") or []]

        self.gpt.set_profession(str(context.get("profession") or "Clinic"))

        # Track that the initial lead-type prompt has been sent
        self.has_shown_lead_prompt: bool = False

    async def start(self) -> str:
        self.has_shown_lead_prompt = True
        return self._prompt_lead_type(greeting=True)

    async def handle_user_message(self, message: str) -> Tuple[str, bool]:
        self._remember_user(message)

        if self.step == "lead_type":
            choice_value = self._parse_lead_choice(message, self.lead_type_options, synonyms={
                "callback": ["call", "phone", "call back", "callback", "ring"],
                "appointment arrangement": ["appointment", "arrange", "arrangement", "book", "visit", "schedule"],
                "further information": ["info", "information", "more info", "details"],
            })
            if choice_value is None:
                maybe_answer = await self._maybe_answer_question(message)
                if maybe_answer:
                    # Answer user question but avoid re-listing the same options again
                    return maybe_answer, False
                # Suppress duplicate prompt while waiting for selection
                if self.has_shown_lead_prompt:
                    return "", False
                return self._prompt_lead_type(), False
            self.state["lead_type"] = choice_value
            self.step = "service_type"
            return self._prompt_service_type(), False

        if self.step == "service_type":
            choice = self._parse_choice(message, self.service_types)
            if choice is None:
                maybe_answer = await self._maybe_answer_question(message)
                if maybe_answer:
                    return maybe_answer + "\n\n" + self._prompt_service_type(), False
                return self._prompt_service_type(), False
            self.state["service_type"] = choice
            self.step = "name"
            return "Great. May I have your full name?", False

        if self.step == "name":
            name = self._extract_name(message)
            if not name:
                return "Please share your full name (e.g., John Smith).", False
            self.state["name"] = name
            self.step = "phone"
            return "Thanks. What is the best phone number to reach you?", False

        if self.step == "phone":
            phone = self._extract_phone(message)
            if not phone:
                return "Please provide a valid phone number (digits, spaces or +).", False
            self.state["phone"] = phone

            created_text = await self._finalize_lead()
            self.step = "complete"
            return created_text, True

        return "Session complete.", True

    def _prompt_lead_type(self, greeting: bool = False) -> str:
        header = "How can i help u today" if greeting else "Please choose a lead type:"
        return self._render_lead_choices(header, self.lead_type_options)

    def _prompt_service_type(self) -> str:
        header = "Thanks! Which service are you interested in?"
        return self._render_choices(header, self.service_types)

    def _render_choices(self, header: str, options: List[str]) -> str:
        lines = [header]
        for idx, opt in enumerate(options, start=1):
            lines.append(f"#Button# {opt} #Button#")
        return "\n".join(lines)

    def _render_lead_choices(self, header: str, options: List[Dict[str, str]]) -> str:
        lines = [header]
        for idx, opt in enumerate(options, start=1):
            lines.append(f"#Button# {opt.get('text')} #Button#")
        return "\n".join(lines)

    def _parse_choice(self, user_text: str, options: List[str], synonyms: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
        if not options:
            return None
        text = _normalize_text(user_text)
        # number selection
        num = re.findall(r"\d+", text)
        if num:
            try:
                idx = int(num[0]) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except Exception:  # noqa: BLE001
                pass
        # direct match
        for opt in options:
            if _normalize_text(opt) in text or text in _normalize_text(opt):
                return opt
        # synonyms
        if synonyms:
            for key, syns in synonyms.items():
                if _normalize_text(key) in text:
                    # return canonical option if present
                    for opt in options:
                        if _normalize_text(opt) == _normalize_text(key):
                            return opt
                    return key
                for s in syns:
                    if _normalize_text(s) in text:
                        for opt in options:
                            if _normalize_text(opt) == _normalize_text(key):
                                return opt
                        return key
        return None

    def _parse_lead_choice(
        self,
        user_text: str,
        options: List[Dict[str, str]],
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[str]:
        if not options:
            return None
        text = _normalize_text(user_text)
        # number selection
        nums = re.findall(r"\d+", text)
        if nums:
            try:
                idx = int(nums[0]) - 1
                if 0 <= idx < len(options):
                    return options[idx].get("value")
            except Exception:  # noqa: BLE001
                pass
        # direct match on text or value
        for opt in options:
            val = _normalize_text(opt.get("value") or "")
            txt = _normalize_text(opt.get("text") or "")
            if val and (val in text or text in val):
                return opt.get("value")
            if txt and (txt in text or text in txt):
                return opt.get("value")
        # synonyms
        if synonyms:
            for key, syns in synonyms.items():
                if _normalize_text(key) in text:
                    for opt in options:
                        if _normalize_text(opt.get("value") or "") == _normalize_text(key):
                            return opt.get("value")
                    return key
                for s in syns:
                    if _normalize_text(s) in text:
                        for opt in options:
                            if _normalize_text(opt.get("value") or "") == _normalize_text(key):
                                return opt.get("value")
                        return key
        return None

    async def _maybe_answer_question(self, user_message: str) -> Optional[str]:
        text = user_message.strip()
        if not text:
            return None
        if any(q in text.lower() for q in ["?", "what", "how", "price", "cost", "when", "where", "do you", "can i", "is it"]):
            reply = await self.gpt.short_reply(self.state["history"], text, self.context)
            self._remember_bot(reply)
            return reply
        return None

    def _extract_name(self, text: str) -> Optional[str]:
        name = re.sub(r"[^A-Za-z\s.'-]", "", text).strip()
        if 2 <= len(name) <= 80 and " " in name:
            return name
        if 2 <= len(name) <= 80:
            return name
        return None

    def _extract_phone(self, text: str) -> Optional[str]:
        digits = re.sub(r"[^0-9+]+", "", text)
        digits = re.sub(r"(?<!^)\+", "", digits)  # keep leading + only
        numbers = re.sub(r"\D+", "", digits)
        if len(numbers) >= 8:
            return digits
        return None

    def _remember_user(self, content: str) -> None:
        self.state["history"].append({"role": "user", "content": content})
        self.state["history"] = self.state["history"][-10:]

    def _remember_bot(self, content: str) -> None:
        self.state["history"].append({"role": "assistant", "content": content})
        self.state["history"] = self.state["history"][-10:]

    async def _finalize_lead(self) -> str:
        lead_type = self.state["lead_type"]
        service_type = self.state["service_type"]
        name = self.state["name"]
        phone = self.state["phone"]

        title = f"Lead - {service_type}"
        summary = f"{lead_type} request from {name}"
        description = (
            f"Lead generated via chatbot.\n"
            f"Name: {name}\n"
            f"Phone: {phone}\n"
            f"Lead Type: {lead_type}\n"
            f"Service Type: {service_type}"
        )

        payload = {
            "title": title,
            "summary": summary,
            "description": description,
            "leadType": lead_type,
            "serviceType": service_type,
        }

        if self.user_id:
            ok, _ = await self.lead_service.create_public_lead(self.user_id, payload)
        else:
            ok = False

        if ok:
            self._remember_bot("Thanks! We've recorded your details.")
            return "Thanks! I have your details and someone will get back to you soon. Bye!"
        return (
            "Thanks! I captured your details. There was a small issue creating the lead right now,"
            " but the team will still follow up shortly. Bye!"
        )
