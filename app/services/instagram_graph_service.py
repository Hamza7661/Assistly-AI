"""
Instagram Graph API Messaging Service
Handles sending messages and parsing webhook events via Meta's Instagram Messaging API (Graph API).
Docs: https://developers.facebook.com/docs/messenger-platform/instagram/features/webhook
"""

import hashlib
import hmac
import os
import logging
import httpx
from typing import Dict, Optional, TypedDict

logger = logging.getLogger(__name__)


class InstagramSendMessageResponse(TypedDict, total=False):
    """Response from Meta Graph API when sending a message."""
    recipient_id: str
    message_id: str


class _InstagramWebhookParsedEventBase(TypedDict):
    """Required fields for parsed Instagram webhook event."""
    sender_id: str
    recipient_id: str
    message_text: str


class InstagramWebhookParsedEvent(_InstagramWebhookParsedEventBase, total=False):
    """Parsed Instagram webhook event. message_id and timestamp are optional."""
    message_id: Optional[str]
    timestamp: Optional[int]

# Graph API configuration
GRAPH_API_VERSION = os.getenv("META_GRAPH_API_VERSION", "v21.0")
GRAPH_API_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class InstagramGraphService:
    """Service for sending Instagram DMs via Meta's Graph API."""
    
    def __init__(self):
        self.base_url = GRAPH_API_BASE_URL
        logger.info(f"InstagramGraphService initialized with API version {GRAPH_API_VERSION}")
    
    async def send_message(
        self,
        recipient_id: str,
        message_text: str,
        access_token: str
    ) -> InstagramSendMessageResponse:
        """
        Send a text message to an Instagram user via Graph API.
        
        Args:
            recipient_id: Instagram-scoped sender ID (IGSID) of the recipient
            message_text: Text content to send
            access_token: Page access token for the Instagram Business Account
        
        Returns:
            Response from Graph API containing message_id
        
        Raises:
            httpx.HTTPError: If the API request fails
        """
        if not access_token:
            logger.error("Instagram access token is required but not provided")
            raise ValueError("Instagram access token is required")
        
        url = f"{self.base_url}/me/messages"
        
        payload = {
            "recipient": {
                "id": recipient_id
            },
            "message": {
                "text": message_text
            }
        }
        
        params = {
            "access_token": access_token
        }
        
        logger.info(f"Sending Instagram message to {recipient_id} (length: {len(message_text)} chars)")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    params=params
                )
                response.raise_for_status()
                result = response.json()
                
                message_id = result.get("message_id", "unknown")
                logger.info(f"Instagram message sent successfully. Message ID: {message_id}")
                
                return result
                
            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                logger.error(
                    f"Instagram Graph API error: {e.response.status_code} - {error_body}"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to send Instagram message: {str(e)}")
                raise

    # ------------------------------------------------------------------
    # Security (Meta uses X-Hub-Signature-256)
    # ------------------------------------------------------------------

    @staticmethod
    def verify_signature(
        payload_body: bytes,
        signature_header: str,
        app_secret: str,
    ) -> bool:
        """
        Verify the X-Hub-Signature-256 header that Meta attaches to webhook POSTs.
        Returns True if the payload is genuine.
        """
        if not signature_header or not app_secret:
            logger.warning("Instagram: missing signature header or app secret for verification")
            return False
        try:
            algo, _, digest = signature_header.partition("=")
            if algo != "sha256" or not digest:
                logger.warning("Instagram: unexpected signature format: %s", signature_header)
                return False
            expected = hmac.new(
                app_secret.encode("utf-8"),
                payload_body,
                hashlib.sha256,
            ).hexdigest()
            return hmac.compare_digest(expected, digest.lower())
        except Exception as exc:
            logger.error("Instagram: signature verification error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Webhook parsing (text, postback, quick_reply, skip echo/read/delivery)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_webhook_event(body: Dict[str, object]) -> Optional[InstagramWebhookParsedEvent]:
        """
        Parse incoming Instagram webhook event from Meta.
        
        Expected structure:
        {
          "object": "instagram",
          "entry": [
            {
              "id": "<INSTAGRAM_BUSINESS_ACCOUNT_ID>",
              "time": 1234567890,
              "messaging": [
                {
                  "sender": {"id": "<IGSID>"},
                  "recipient": {"id": "<INSTAGRAM_BUSINESS_ACCOUNT_ID>"},
                  "timestamp": 1234567890,
                  "message": {
                    "mid": "message_id",
                    "text": "Hello!"
                  }
                }
              ]
            }
          ]
        }
        
        Args:
            body: Raw webhook payload from Meta
        
        Returns:
            Parsed event dict with sender_id, recipient_id (IG Business Account ID), 
            message_text, and message_id, or None if not a valid message event
        """
        try:
            if body.get("object") != "instagram":
                logger.warning("Instagram: received non-instagram webhook (object=%s)", body.get("object"))
                return None

            entries = body.get("entry", [])
            if not entries:
                logger.warning("Instagram: no entries in webhook payload")
                return None

            entry = entries[0]
            messaging_events = entry.get("messaging", [])
            if not messaging_events:
                logger.information("Instagram: no messaging events (may be read/delivery)")
                return None

            event = messaging_events[0]
            sender_id = event.get("sender", {}).get("id")
            recipient_id = event.get("recipient", {}).get("id")
            timestamp = event.get("timestamp")

            # Skip read receipts (messaging_seen)
            if "read" in event:
                logger.information("Instagram: skipping read receipt")
                return None

            # Handle postback (Icebreaker, Generic Template buttons)
            postback = event.get("postback", {})
            if postback:
                message_text = (
                    postback.get("title") or postback.get("payload") or ""
                ).strip()
                message_id = postback.get("mid")
                if sender_id and message_text:
                    logger.info(
                        "Instagram: postback from %s to %s: '%s'",
                        sender_id, recipient_id, message_text[:60] + ("..." if len(message_text) > 60 else ""),
                    )
                    return {
                        "sender_id": sender_id,
                        "recipient_id": recipient_id,
                        "message_text": message_text,
                        "message_id": message_id,
                        "timestamp": timestamp,
                    }
                return None

            # Handle message (text, quick_reply, etc.)
            message = event.get("message", {})
            if message.get("is_echo"):
                logger.information("Instagram: skipping echo message")
                return None
            if message.get("is_deleted") or message.get("is_unsupported"):
                logger.information("Instagram: skipping deleted/unsupported message")
                return None

            message_id = message.get("mid")

            # 1. Plain text
            message_text = message.get("text", "").strip()

            # 2. Quick reply payload
            if not message_text:
                qr = message.get("quick_reply", {})
                message_text = qr.get("payload", "").strip()

            if not sender_id:
                logger.information("Instagram: missing sender IGSID in event")
                return None
            if not message_text:
                logger.information("Instagram: empty message text (attachment/sticker not supported)")
                return None

            logger.info(
                "Instagram: message from IGSID=%s → IG=%s: '%s'",
                sender_id,
                recipient_id,
                message_text[:60] + ("..." if len(message_text) > 60 else ""),
            )

            return {
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message_text": message_text,
                "message_id": message_id,
                "timestamp": timestamp,
            }

        except Exception as exc:
            logger.error("Instagram: error parsing webhook payload: %s", exc, exc_info=True)
            return None
