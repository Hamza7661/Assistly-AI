"""Production-grade conversation state machine for lead generation flow"""
from enum import Enum
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger("assistly.conversation_state")


class ConversationState(Enum):
    """Defines all possible conversation states"""
    GREETING = "greeting"
    LEAD_TYPE_SELECTION = "lead_type_selection"
    SERVICE_SELECTION = "service_selection"
    NAME_COLLECTION = "name_collection"
    EMAIL_COLLECTION = "email_collection"
    EMAIL_OTP_SENT = "email_otp_sent"
    EMAIL_OTP_VERIFICATION = "email_otp_verification"
    PHONE_COLLECTION = "phone_collection"
    PHONE_OTP_SENT = "phone_otp_sent"
    PHONE_OTP_VERIFICATION = "phone_otp_verification"
    COMPLETE = "complete"


class FlowController:
    """Manages conversation flow using state machine"""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.state = ConversationState.GREETING
        self.collected_data: Dict[str, Any] = {
            "leadType": None,
            "serviceType": None,
            "leadName": None,
            "leadEmail": None,
            "leadPhoneNumber": None,
            "title": None
        }
        self.otp_state = {
            "email_sent": False,
            "email_verified": False,
            "phone_sent": False,
            "phone_verified": False
        }
        
        # Get validation flags
        integration = context.get("integration", {})
        self.validate_email = integration.get("validateEmail", True)
        self.validate_phone = integration.get("validatePhoneNumber", True)
        self.is_whatsapp = False  # Set externally
        self.channel: str = integration.get("channel", "web")
        self.skip_phone_collection: bool = False
    
    def set_whatsapp(self, is_whatsapp: bool):
        """Set WhatsApp mode (phone already verified)"""
        self.is_whatsapp = is_whatsapp
        if is_whatsapp:
            self.channel = "whatsapp"
            self.skip_phone_collection = True
            # For WhatsApp, phone is already verified
            self.otp_state["phone_verified"] = True
            # Extract phone from WhatsApp context if available
            # This would come from Twilio webhook
        else:
            self.channel = "web"
            self.skip_phone_collection = False

    def set_voice_agent(self):
        """Configure flow for voice agent channel (phone comes from Twilio caller)"""
        self.channel = "voice"
        self.skip_phone_collection = True
        # Phone verification is implicit (caller ID)
        self.otp_state["phone_verified"] = True
    
    def update_collected_data(self, field: str, value: Any):
        """Update collected data and handle title extraction"""
        self.collected_data[field] = value
        
        # If leadType is set, also set title from lead types
        if field == "leadType" and value:
            lead_types = self.context.get("lead_types", [])
            for lt in lead_types:
                if isinstance(lt, dict) and lt.get("value") == value:
                    self.collected_data["title"] = lt.get("text", value)
                    break
    
    def get_next_state(self) -> ConversationState:
        """Determine next state based on current state and collected data"""
        if self.state == ConversationState.GREETING:
            return ConversationState.LEAD_TYPE_SELECTION
        
        elif self.state == ConversationState.LEAD_TYPE_SELECTION:
            if self.collected_data["leadType"]:
                return ConversationState.SERVICE_SELECTION
            return ConversationState.LEAD_TYPE_SELECTION
        
        elif self.state == ConversationState.SERVICE_SELECTION:
            if self.collected_data["serviceType"]:
                return ConversationState.NAME_COLLECTION
            return ConversationState.SERVICE_SELECTION
        
        elif self.state == ConversationState.NAME_COLLECTION:
            if self.collected_data["leadName"]:
                return ConversationState.EMAIL_COLLECTION
            return ConversationState.NAME_COLLECTION
        
        elif self.state == ConversationState.EMAIL_COLLECTION:
            if self.collected_data["leadEmail"]:
                if self.validate_email:
                    return ConversationState.EMAIL_OTP_SENT
                else:
                    # Skip email OTP, go to phone or complete
                    if self.is_whatsapp or self.skip_phone_collection:
                        return ConversationState.COMPLETE
                    else:
                        return ConversationState.PHONE_COLLECTION
            return ConversationState.EMAIL_COLLECTION
        
        elif self.state == ConversationState.EMAIL_OTP_SENT:
            return ConversationState.EMAIL_OTP_VERIFICATION
        
        elif self.state == ConversationState.EMAIL_OTP_VERIFICATION:
            if self.otp_state["email_verified"]:
                if self.is_whatsapp or self.skip_phone_collection:
                    return ConversationState.COMPLETE
                else:
                    return ConversationState.PHONE_COLLECTION
            return ConversationState.EMAIL_OTP_VERIFICATION
        
        elif self.state == ConversationState.PHONE_COLLECTION:
            if self.skip_phone_collection:
                return ConversationState.COMPLETE
            if self.collected_data["leadPhoneNumber"]:
                if self.validate_phone:
                    return ConversationState.PHONE_OTP_SENT
                else:
                    return ConversationState.COMPLETE
            return ConversationState.PHONE_COLLECTION
        
        elif self.state == ConversationState.PHONE_OTP_SENT:
            return ConversationState.PHONE_OTP_VERIFICATION
        
        elif self.state == ConversationState.PHONE_OTP_VERIFICATION:
            if self.otp_state["phone_verified"]:
                return ConversationState.COMPLETE
            return ConversationState.PHONE_OTP_VERIFICATION
        
        elif self.state == ConversationState.COMPLETE:
            return ConversationState.COMPLETE
        
        return self.state
    
    def transition_to(self, new_state: ConversationState):
        """Transition to new state"""
        logger.info(f"State transition: {self.state.value} → {new_state.value}")
        self.state = new_state
    
    def can_generate_json(self) -> bool:
        """Check if all required data is collected for JSON generation"""
        required_fields = ["leadType", "serviceType", "leadName", "leadEmail"]
        
        # Check required fields
        for field in required_fields:
            if not self.collected_data.get(field):
                return False
        
        # Check phone (if not skipping phone collection)
        if not self.skip_phone_collection and not self.collected_data.get("leadPhoneNumber"):
            return False
        
        # Check OTP verification
        if self.validate_email and not self.otp_state["email_verified"]:
            return False
        
        if self.validate_phone and not self.skip_phone_collection and not self.otp_state["phone_verified"]:
            return False
        
        return True
    
    def get_json_data(self, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Get collected data as JSON-ready dict"""
        data = {
            "leadType": self.collected_data["leadType"],
            "serviceType": self.collected_data["serviceType"],
            "leadName": self.collected_data["leadName"],  # Customer name
            "leadEmail": self.collected_data["leadEmail"],
            "title": self.collected_data.get("title", "")
        }
        
        if self.collected_data.get("leadPhoneNumber"):
            data["leadPhoneNumber"] = self.collected_data.get("leadPhoneNumber", "")
        
        # Add conversation history if provided
        if conversation_history:
            data["history"] = conversation_history
        
        return data
    
    def get_state_prompt_context(self) -> str:
        """Get minimal prompt context for current state"""
        state_prompts = {
            ConversationState.GREETING: "Greet the user and present lead type options.",
            ConversationState.LEAD_TYPE_SELECTION: "Present lead type options and wait for selection.",
            ConversationState.SERVICE_SELECTION: "Present service options and wait for selection.",
            ConversationState.NAME_COLLECTION: "Ask for the user's name.",
            ConversationState.EMAIL_COLLECTION: "Ask for the user's email address.",
            ConversationState.EMAIL_OTP_SENT: "Acknowledge OTP sent and wait for verification code.",
            ConversationState.EMAIL_OTP_VERIFICATION: "Verify the OTP code provided.",
            ConversationState.PHONE_COLLECTION: "Ask for the user's phone number.",
            ConversationState.PHONE_OTP_SENT: "Acknowledge OTP sent and wait for verification code.",
            ConversationState.PHONE_OTP_VERIFICATION: "Verify the OTP code provided.",
            ConversationState.COMPLETE: "All information collected. Generate JSON."
        }
        return state_prompts.get(self.state, "Continue conversation naturally.")

