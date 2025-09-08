import json
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
            "leadType": None,
            "serviceType": None,
            "leadName": None,
            "leadEmail": None,
            "leadPhoneNumber": None,
            "history": [],
        }

        # Build lead type options from backend-provided objects (for fallback greet only)
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
        self._last_bot_message: Optional[str] = None

    async def start(self) -> str:
        self.has_shown_lead_prompt = True
        reply = await self.gpt.agent_greet(self.context, self.state)
        self._remember_bot(reply)
        self._last_bot_message = reply.strip()
        return reply

    async def handle_user_message(self, message: str) -> Tuple[str, bool]:
        self._remember_user(message)

        reply = await self.gpt.agent_reply(self.state["history"], message, self.context, self.state)

        # Drop identical repeated greetings
        if self._last_bot_message and reply.strip() == self._last_bot_message:
            return "", False

        parsed = self._maybe_parse_json(reply)
        if parsed is not None and isinstance(parsed, dict):
            if self.user_id:
                ok, _ = await self.lead_service.create_public_lead(self.user_id, parsed)
            else:
                ok = False
            self.step = "complete"
            if ok:
                final_msg = "Thanks! I have your details and someone will get back to you soon. Bye!"
            else:
                final_msg = (
                    "Thanks! I captured your details. There was a small issue creating the lead right now,"
                    " but the team will still follow up shortly. Bye!"
                )
            self._remember_bot(final_msg)
            self._last_bot_message = final_msg
            return final_msg, True

        self._remember_bot(reply)
        self._last_bot_message = reply.strip()
        return reply, False

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

    def _maybe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        content = text.strip()
        if not (content.startswith("{") and content.endswith("}")):
            return None
        try:
            return json.loads(content)
        except Exception:  # noqa: BLE001
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
        lead_type = self.state.get("leadType")
        service_type = self.state.get("serviceType")
        name = self.state.get("leadName")
        phone = self.state.get("leadPhoneNumber")

        title = f"Lead - {service_type}" if service_type else "Lead"
        summary = f"{lead_type} request from {name}" if lead_type and name else "Lead request"
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
            "leadName": name,
            "leadPhoneNumber": phone,
            "leadEmail": self.state.get("leadEmail"),
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
