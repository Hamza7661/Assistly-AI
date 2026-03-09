"""
Unified Twilio messaging for WhatsApp, Facebook Messenger, and Instagram (when available).
Same conversation flow runs on all channels; this module handles channel-specific address formats.
"""
from typing import Any, Dict, List, Optional, Tuple
import time
import logging
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

logger = logging.getLogger("assistly.twilio_messaging")

# Channel prefix for Twilio API: whatsapp:, messenger:, instagram:
CHANNEL_WHATSAPP = "whatsapp"
CHANNEL_MESSENGER = "messenger"
CHANNEL_INSTAGRAM = "instagram"


class TwilioMessagingService:
    """Send and receive messages via Twilio across WhatsApp, Messenger, and compatible channels."""

    def __init__(self, account_sid: str, auth_token: str) -> None:
        self.client: Optional[Client] = None
        if account_sid and auth_token:
            self.client = Client(account_sid, auth_token)

    @staticmethod
    def normalize_address(channel: str, address: str) -> str:
        """Ensure address has channel prefix (e.g. whatsapp:+1234, messenger:page_123)."""
        if not address:
            return address
        addr = str(address).strip()
        prefix = f"{channel}:"
        if addr.startswith(prefix):
            return addr
        return f"{prefix}{addr}"

    async def send_message(
        self,
        channel: str,
        to_address: str,
        from_address: str,
        message: str,
    ) -> Tuple[bool, str]:
        """
        Send a text message. Addresses must include channel prefix (whatsapp:+n, messenger:id).
        """
        if not self.client:
            logger.warning("Twilio client not initialized - message not sent")
            return False, "Twilio messaging not configured"
        to_addr = self.normalize_address(channel, to_address.replace(f"{channel}:", ""))
        from_addr = self.normalize_address(channel, from_address.replace(f"{channel}:", ""))
        try:
            message_obj = self.client.messages.create(
                body=message,
                from_=from_addr,
                to=to_addr,
            )
            logger.info(
                "Twilio message sent channel=%s to=%s sid=%s",
                channel,
                to_addr,
                message_obj.sid,
            )
            return True, f"Message sent - SID: {message_obj.sid}"
        except Exception as e:
            logger.error("Twilio send_message failed: %s", str(e))
            return False, str(e)

    def create_twiml_response(self, message: str) -> str:
        """TwiML response for webhooks (no reply body if we reply via API)."""
        response = MessagingResponse()
        if message:
            response.message(message)
        return str(response)

    @staticmethod
    def parse_webhook_data(form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse incoming webhook from Twilio. Detects channel from From/To (whatsapp:, messenger:).
        Returns: from, to, body, channel, button_id, list_id, etc.
        """
        raw_from = (form_data.get("From") or "").strip()
        raw_to = (form_data.get("To") or "").strip()
        body = (form_data.get("Body") or "").strip()

        if raw_from.startswith("whatsapp:") or raw_to.startswith("whatsapp:"):
            channel = CHANNEL_WHATSAPP
            from_id = raw_from.replace("whatsapp:", "").strip()
            to_id = raw_to.replace("whatsapp:", "").strip()
        elif raw_from.startswith("instagram:") or raw_to.startswith("instagram:"):
            channel = CHANNEL_INSTAGRAM
            from_id = raw_from.replace("instagram:", "").strip()
            to_id = raw_to.replace("instagram:", "").strip()
        elif raw_from.startswith("messenger:") or raw_to.startswith("messenger:"):
            channel = CHANNEL_MESSENGER
            from_id = raw_from.replace("messenger:", "").strip()
            to_id = raw_to.replace("messenger:", "").strip()
        else:
            channel = CHANNEL_WHATSAPP
            from_id = raw_from
            to_id = raw_to

        return {
            "from": from_id,
            "to": to_id,
            "body": body,
            "channel": channel,
            "from_raw": raw_from,
            "to_raw": raw_to,
            "message_sid": form_data.get("MessageSid", ""),
            "profile_name": form_data.get("ProfileName", ""),
            "button_id": form_data.get("ButtonId", ""),
            "button_title": form_data.get("ButtonTitle", ""),
            "list_id": form_data.get("ListId", ""),
            "list_title": form_data.get("ListTitle", ""),
        }
