"""
Instagram Graph API Messaging Service
Handles sending messages via Meta's Instagram Messaging API (Graph API).
Docs: https://developers.facebook.com/docs/messenger-platform/instagram
"""

import os
import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

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
    ) -> Dict[str, Any]:
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
    
    @staticmethod
    def parse_webhook_event(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                logger.warning(f"Received non-Instagram webhook: {body.get('object')}")
                return None
            
            entries = body.get("entry", [])
            if not entries:
                logger.warning("No entries in Instagram webhook")
                return None
            
            # Process first entry's first messaging event
            entry = entries[0]
            messaging_events = entry.get("messaging", [])
            
            if not messaging_events:
                logger.debug("No messaging events in entry")
                return None
            
            event = messaging_events[0]
            
            # Extract message details
            sender = event.get("sender", {})
            recipient = event.get("recipient", {})
            message = event.get("message", {})
            
            sender_id = sender.get("id")
            recipient_id = recipient.get("id")  # This is the Instagram Business Account ID
            message_text = message.get("text", "").strip()
            message_id = message.get("mid")
            timestamp = event.get("timestamp")
            
            if not sender_id or not message_text:
                logger.debug("Missing sender ID or message text")
                return None
            
            logger.info(
                f"Parsed Instagram message from {sender_id} to {recipient_id}: '{message_text[:50]}...'"
            )
            
            return {
                "sender_id": sender_id,  # IGSID (Instagram-scoped sender ID)
                "recipient_id": recipient_id,  # Instagram Business Account ID
                "message_text": message_text,
                "message_id": message_id,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Error parsing Instagram webhook: {str(e)}", exc_info=True)
            return None
