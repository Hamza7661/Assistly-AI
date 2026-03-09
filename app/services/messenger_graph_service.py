"""
Facebook Messenger Graph API Messaging Service.
Handles sending messages and parsing webhook events via Meta's Messenger API.
Docs: https://developers.facebook.com/docs/messenger-platform/webhooks
"""

import hashlib
import hmac
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = os.getenv("META_GRAPH_API_VERSION", "v21.0")
GRAPH_API_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class MessengerGraphService:
    """Service for sending Facebook Messenger messages via Meta's Graph API."""

    def __init__(self) -> None:
        self.base_url = GRAPH_API_BASE_URL
        logger.info("MessengerGraphService initialized with API version %s", GRAPH_API_VERSION)

    async def send_message(
        self,
        recipient_psid: str,
        message_text: str,
        page_access_token: str,
    ) -> Dict[str, Any]:
        """
        Send a text message to a Messenger user.

        POST /me/messages with:
          { recipient: {id: PSID}, message: {text: "..."}, messaging_type: "RESPONSE" }

        Args:
            recipient_psid:    Page-Scoped User ID (PSID) of the recipient.
            message_text:      Text content (capped at 2 000 chars by Messenger).
            page_access_token: Facebook Page Access Token.

        Returns:
            Graph API response dict (contains recipient_id + message_id).
        """
        if not page_access_token:
            raise ValueError("Page access token is required to send Messenger messages")

        # Messenger caps plain-text messages at 2 000 chars
        if len(message_text) > 2000:
            message_text = message_text[:1997] + "..."

        url = f"{self.base_url}/me/messages"
        payload = {
            "recipient": {"id": recipient_psid},
            "message": {"text": message_text},
            "messaging_type": "RESPONSE",
        }
        params = {"access_token": page_access_token}

        logger.info(
            "Sending Messenger message to PSID=%s (%d chars)",
            recipient_psid,
            len(message_text),
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, json=payload, params=params)
                response.raise_for_status()
                result = response.json()
                logger.info(
                    "Messenger message sent successfully. Message ID: %s",
                    result.get("message_id", "unknown"),
                )
                return result
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Messenger Graph API error: %s – %s",
                    e.response.status_code,
                    e.response.text,
                )
                raise
            except Exception as e:
                logger.error("Failed to send Messenger message: %s", e)
                raise

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    @staticmethod
    def verify_signature(
        payload_body: bytes,
        signature_header: str,
        app_secret: str,
    ) -> bool:
        """
        Verify the X-Hub-Signature-256 header that Meta attaches to every
        webhook POST.  Returns True if the payload is genuine.

        Args:
            payload_body:     Raw request body (bytes).
            signature_header: Value of X-Hub-Signature-256 header,
                              e.g. "sha256=abcdef...".
            app_secret:       Meta App Secret (from your App Dashboard).
        """
        if not signature_header or not app_secret:
            logger.warning("Messenger: missing signature header or app secret for verification")
            return False
        try:
            algo, _, digest = signature_header.partition("=")
            if algo != "sha256" or not digest:
                logger.warning(
                    "Messenger: unexpected signature format: %s", signature_header
                )
                return False
            expected = hmac.new(
                app_secret.encode("utf-8"),
                payload_body,
                hashlib.sha256,
            ).hexdigest()
            return hmac.compare_digest(expected, digest.lower())
        except Exception as exc:
            logger.error("Messenger: signature verification error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Webhook parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_webhook_event(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a single Messenger webhook event payload.

        Meta wraps events as:
        {
          "object": "page",
          "entry": [{
            "id": "<PAGE_ID>",
            "messaging": [{
              "sender":    {"id": "<PSID>"},
              "recipient": {"id": "<PAGE_ID>"},
              "timestamp": 1234567890,
              "message":   {"mid": "...", "text": "Hello!"}
            }]
          }]
        }

        Returns a dict with sender_id, recipient_id, page_id, message_text,
        message_id, timestamp – or None if the event should not be processed.
        """
        try:
            if body.get("object") != "page":
                logger.warning(
                    "Messenger: received non-page webhook (object=%s)", body.get("object")
                )
                return None

            entries = body.get("entry", [])
            if not entries:
                logger.warning("Messenger: no entries in webhook payload")
                return None

            entry = entries[0]
            page_id = entry.get("id")
            messaging_events = entry.get("messaging", [])

            if not messaging_events:
                logger.debug("Messenger: no messaging events (may be changes/standby channel)")
                return None

            event = messaging_events[0]
            message = event.get("message", {})

            # Skip echo messages (sent by the Page itself)
            if message.get("is_echo"):
                logger.debug("Messenger: skipping echo message")
                return None

            # Skip delivery / read receipts
            if "delivery" in event or "read" in event:
                logger.debug("Messenger: skipping delivery/read receipt")
                return None

            sender_id = event.get("sender", {}).get("id")       # PSID
            recipient_id = event.get("recipient", {}).get("id")  # Page ID
            timestamp = event.get("timestamp")
            message_id = message.get("mid")

            # 1. Plain text message
            message_text = message.get("text", "").strip()

            # 2. Postback (Get Started, persistent menu, template buttons)
            if not message_text:
                postback = event.get("postback", {})
                message_text = (
                    postback.get("title") or postback.get("payload") or ""
                ).strip()

            # 3. Quick-reply payload
            if not message_text:
                qr = message.get("quick_reply", {})
                message_text = qr.get("payload", "").strip()

            if not sender_id:
                logger.debug("Messenger: missing sender PSID in event")
                return None

            if not message_text:
                logger.debug(
                    "Messenger: empty message text (attachment/sticker not supported)"
                )
                return None

            logger.info(
                "Messenger: message from PSID=%s → Page=%s: '%s'",
                sender_id,
                recipient_id,
                message_text[:60] + ("..." if len(message_text) > 60 else ""),
            )

            return {
                "sender_id": sender_id,          # PSID – conversation partner
                "recipient_id": recipient_id,    # Facebook Page ID
                "page_id": page_id or recipient_id,
                "message_text": message_text,
                "message_id": message_id,
                "timestamp": timestamp,
            }

        except Exception as exc:
            logger.error(
                "Messenger: error parsing webhook payload: %s", exc, exc_info=True
            )
            return None

    @staticmethod
    def parse_all_events(body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse EVERY message event in a (potentially batched) webhook payload.
        Returns a list; empty if nothing is processable.
        """
        results: List[Dict[str, Any]] = []
        if body.get("object") != "page":
            return results

        for entry in body.get("entry", []):
            page_id = entry.get("id")
            for event in entry.get("messaging", []):
                message = event.get("message", {})

                if message.get("is_echo"):
                    continue
                if "delivery" in event or "read" in event:
                    continue

                sender_id = event.get("sender", {}).get("id")
                recipient_id = event.get("recipient", {}).get("id")
                timestamp = event.get("timestamp")
                message_id = message.get("mid")

                text = message.get("text", "").strip()
                if not text:
                    postback = event.get("postback", {})
                    text = (postback.get("title") or postback.get("payload") or "").strip()
                if not text:
                    qr = message.get("quick_reply", {})
                    text = qr.get("payload", "").strip()

                if not sender_id or not text:
                    continue

                results.append(
                    {
                        "sender_id": sender_id,
                        "recipient_id": recipient_id,
                        "page_id": page_id or recipient_id,
                        "message_text": text,
                        "message_id": message_id,
                        "timestamp": timestamp,
                    }
                )

        return results
