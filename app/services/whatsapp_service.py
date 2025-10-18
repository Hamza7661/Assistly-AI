from typing import Any, Dict, List, Optional, Tuple
import time
import logging
import json
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

logger = logging.getLogger("assistly.whatsapp")


class WhatsAppService:
    def __init__(self, settings: Any) -> None:
        self.account_sid = settings.twilio_account_sid
        self.auth_token = settings.twilio_auth_token
        self.whatsapp_from = settings.twilio_whatsapp_from
        
        # Initialize Twilio client if credentials are provided
        self.client: Optional[Client] = None
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
    
    def _format_phone_number(self, phone: str) -> str:
        """Format phone number for WhatsApp (remove + if present, ensure it starts with country code)"""
        # Remove any non-digit characters except +
        cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
        
        # If it doesn't start with +, assume it's a US number and add +1
        if not cleaned.startswith('+'):
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned
            elif len(cleaned) == 11 and cleaned.startswith('1'):
                cleaned = '+' + cleaned
            else:
                cleaned = '+' + cleaned
        
        return cleaned
    
    async def send_message(self, to_phone: str, message: str) -> Tuple[bool, str]:
        """Send a simple text message via WhatsApp"""
        if not self.client:
            logger.warning("Twilio client not initialized - WhatsApp message not sent")
            return False, "WhatsApp service not configured"
        
        try:
            formatted_phone = self._format_phone_number(to_phone)
            whatsapp_to = f"whatsapp:{formatted_phone}"
            # Ensure whatsapp_from has whatsapp: prefix
            whatsapp_from = self.whatsapp_from if self.whatsapp_from.startswith('whatsapp:') else f"whatsapp:{self.whatsapp_from}"
            
            start_time = time.time()
            logger.info("Sending WhatsApp message at %s to %s", 
                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), formatted_phone)
            
            message_obj = self.client.messages.create(
                body=message,
                from_=whatsapp_from,
                to=whatsapp_to
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info("WhatsApp message sent successfully at %s (took %.3fs) - SID: %s", 
                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, message_obj.sid)
            
            return True, f"Message sent successfully - SID: {message_obj.sid}"
            
        except Exception as e:
            logger.error("Failed to send WhatsApp message: %s", str(e))
            return False, f"Failed to send message: {str(e)}"
    
    async def send_interactive_buttons(self, to_phone: str, body_text: str, buttons: List[Dict[str, str]]) -> Tuple[bool, str]:
        """Send an interactive message with buttons via WhatsApp"""
        if not self.client:
            logger.warning("Twilio client not initialized - WhatsApp interactive message not sent")
            return False, "WhatsApp service not configured"
        
        try:
            formatted_phone = self._format_phone_number(to_phone)
            whatsapp_to = f"whatsapp:{formatted_phone}"
            whatsapp_from = f"whatsapp:{self.whatsapp_from}"
            
            # Limit to 3 buttons as per WhatsApp API
            buttons = buttons[:3]
            
            # Create interactive message payload
            interactive_content = {
                "type": "button",
                "body": {
                    "text": body_text
                },
                "action": {
                    "buttons": []
                }
            }
            
            for i, button in enumerate(buttons, 1):
                button_obj = {
                    "type": "reply",
                    "reply": {
                        "id": button.get("id", f"button_{i}"),
                        "title": button.get("title", f"Option {i}")
                    }
                }
                interactive_content["action"]["buttons"].append(button_obj)
            
            start_time = time.time()
            logger.info("Sending WhatsApp interactive message at %s to %s with %d buttons", 
                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), 
                       formatted_phone, len(buttons))
            
            message_obj = self.client.messages.create(
                from_=whatsapp_from,
                to=whatsapp_to,
                content_sid=None,  # We'll use the interactive content directly
                content_variables=json.dumps(interactive_content)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info("WhatsApp interactive message sent successfully at %s (took %.3fs) - SID: %s", 
                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, message_obj.sid)
            
            return True, f"Interactive message sent successfully - SID: {message_obj.sid}"
            
        except Exception as e:
            logger.error("Failed to send WhatsApp interactive message: %s", str(e))
            return False, f"Failed to send interactive message: {str(e)}"
    
    async def send_interactive_list(self, to_phone: str, body_text: str, button_text: str, sections: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Send an interactive list message via WhatsApp"""
        if not self.client:
            logger.warning("Twilio client not initialized - WhatsApp list message not sent")
            return False, "WhatsApp service not configured"
        
        try:
            formatted_phone = self._format_phone_number(to_phone)
            whatsapp_to = f"whatsapp:{formatted_phone}"
            whatsapp_from = f"whatsapp:{self.whatsapp_from}"
            
            # Create interactive list payload
            interactive_content = {
                "type": "list",
                "body": {
                    "text": body_text
                },
                "action": {
                    "button": button_text,
                    "sections": sections
                }
            }
            
            start_time = time.time()
            logger.info("Sending WhatsApp list message at %s to %s", 
                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), formatted_phone)
            
            message_obj = self.client.messages.create(
                from_=whatsapp_from,
                to=whatsapp_to,
                content_sid=None,
                content_variables=json.dumps(interactive_content)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info("WhatsApp list message sent successfully at %s (took %.3fs) - SID: %s", 
                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, message_obj.sid)
            
            return True, f"List message sent successfully - SID: {message_obj.sid}"
            
        except Exception as e:
            logger.error("Failed to send WhatsApp list message: %s", str(e))
            return False, f"Failed to send list message: {str(e)}"
    
    def create_twiml_response(self, message: str) -> str:
        """Create a TwiML response for webhook handling"""
        response = MessagingResponse()
        response.message(message)
        return str(response)
    
    def parse_webhook_data(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse incoming webhook data from Twilio"""
        try:
            # Extract message details from Twilio webhook
            message_data = {
                "from": form_data.get("From", "").replace("whatsapp:", ""),
                "to": form_data.get("To", "").replace("whatsapp:", ""),
                "body": form_data.get("Body", ""),
                "message_sid": form_data.get("MessageSid", ""),
                "profile_name": form_data.get("ProfileName", ""),
                "button_id": form_data.get("ButtonId", ""),  # For button responses
                "button_title": form_data.get("ButtonTitle", ""),  # For button responses
                "list_id": form_data.get("ListId", ""),  # For list responses
                "list_title": form_data.get("ListTitle", ""),  # For list responses
            }
            
            logger.info("Parsed WhatsApp webhook data: %s", json.dumps(message_data, indent=2))
            return message_data
            
        except Exception as e:
            logger.error("Failed to parse WhatsApp webhook data: %s", str(e))
            return {}
