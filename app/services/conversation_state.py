"""Production-grade conversation state machine for lead generation flow"""
from enum import Enum
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger("assistly.conversation_state")


# Forward declaration to avoid circular import
class WorkflowManager:
    pass


class ConversationState(Enum):
    """Defines all possible conversation states"""
    GREETING = "greeting"
    LEAD_TYPE_SELECTION = "lead_type_selection"
    SERVICE_SELECTION = "service_selection"
    WORKFLOW_QUESTION = "workflow_question"
    NAME_COLLECTION = "name_collection"
    EMAIL_COLLECTION = "email_collection"
    EMAIL_OTP_SENT = "email_otp_sent"
    EMAIL_OTP_VERIFICATION = "email_otp_verification"
    PHONE_COLLECTION = "phone_collection"
    PHONE_OTP_SENT = "phone_otp_sent"
    PHONE_OTP_VERIFICATION = "phone_otp_verification"
    APPOINTMENT_OFFER = "appointment_offer"
    CALENDAR_BOOKING = "calendar_booking"
    APPOINTMENT_CONFIRMATION = "appointment_confirmation"
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
            "sourceChannel": None,
            "leadId": None,
            "appointmentSlot": None,
            "title": None,
            "workflowAnswers": {}
        }
        self.otp_state = {
            "email_sent": False,
            "email_verified": False,
            "phone_sent": False,
            "phone_verified": False
        }
        
        # Get validation flags
        integration = context.get("integration", {})
        self.capture_lead_name = integration.get("captureLeadName", True)
        self.capture_lead_email = integration.get("captureLeadEmail", True)
        self.capture_lead_phone = integration.get("captureLeadPhoneNumber", True)
        self.validate_email = bool(integration.get("validateEmail", True) and self.capture_lead_email)
        plan_addons = ((context or {}).get("appPlan") or {}).get("addons") or {}
        if "smsVerification" in plan_addons:
            self.validate_phone = bool(plan_addons.get("smsVerification") and self.capture_lead_phone)
        else:
            self.validate_phone = bool(integration.get("validatePhoneNumber", True) and self.capture_lead_phone)
        self.is_whatsapp = False  # Set externally
        self.channel: str = integration.get("channel", "web")
        self.skip_phone_collection: bool = False
        self.workflow_manager: Optional[Any] = None  # Will be set by response_generator
        # Mid-workflow booking block: { question_id, stage: yes_no|cancel_reason|calendar, config }
        self.booking_block_ctx: Optional[Dict[str, Any]] = None

    def is_booking_lead_type(self) -> bool:
        lead_type = (self.collected_data.get("leadType") or "").lower()
        if not lead_type:
            return False
        keywords = ("book", "appointment", "treatment")
        return any(k in lead_type for k in keywords)

    def reset_service_flow(self):
        self.collected_data["serviceType"] = None
        self.collected_data["workflowAnswers"] = {}
        self.collected_data["appointmentSlot"] = None
        if self.workflow_manager:
            self.workflow_manager.reset()
        self.transition_to(ConversationState.SERVICE_SELECTION)
    
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
                if self.capture_lead_name:
                    return ConversationState.NAME_COLLECTION
                if self.capture_lead_email:
                    return ConversationState.EMAIL_COLLECTION
                if self.capture_lead_phone and not (self.is_whatsapp or self.skip_phone_collection):
                    return ConversationState.PHONE_COLLECTION
                return ConversationState.SERVICE_SELECTION if self.is_booking_lead_type() else ConversationState.WORKFLOW_QUESTION
            return ConversationState.LEAD_TYPE_SELECTION
        
        elif self.state == ConversationState.SERVICE_SELECTION:
            if self.collected_data["serviceType"]:
                return ConversationState.WORKFLOW_QUESTION
            return ConversationState.SERVICE_SELECTION
        
        elif self.state == ConversationState.WORKFLOW_QUESTION:
            # Workflow questions are handled by workflow_manager
            # Check if workflow is complete
            if self.workflow_manager and self.workflow_manager.is_workflow_complete():
                if self.is_booking_lead_type():
                    return ConversationState.CALENDAR_BOOKING
                return ConversationState.APPOINTMENT_OFFER
            return ConversationState.WORKFLOW_QUESTION
        
        elif self.state == ConversationState.NAME_COLLECTION:
            if self.collected_data["leadName"]:
                if self.capture_lead_email:
                    return ConversationState.EMAIL_COLLECTION
                if self.capture_lead_phone and not (self.is_whatsapp or self.skip_phone_collection):
                    return ConversationState.PHONE_COLLECTION
                return ConversationState.SERVICE_SELECTION if self.is_booking_lead_type() else ConversationState.WORKFLOW_QUESTION
            return ConversationState.NAME_COLLECTION
        
        elif self.state == ConversationState.EMAIL_COLLECTION:
            if self.collected_data["leadEmail"]:
                if self.validate_email:
                    return ConversationState.EMAIL_OTP_SENT
                else:
                    # Skip email OTP, go to phone or continue flow
                    if (not self.capture_lead_phone) or self.is_whatsapp or self.skip_phone_collection:
                        return ConversationState.SERVICE_SELECTION if self.is_booking_lead_type() else ConversationState.WORKFLOW_QUESTION
                    else:
                        return ConversationState.PHONE_COLLECTION
            return ConversationState.EMAIL_COLLECTION
        
        elif self.state == ConversationState.EMAIL_OTP_SENT:
            return ConversationState.EMAIL_OTP_VERIFICATION
        
        elif self.state == ConversationState.EMAIL_OTP_VERIFICATION:
            if self.otp_state["email_verified"]:
                if (not self.capture_lead_phone) or self.is_whatsapp or self.skip_phone_collection:
                    return ConversationState.SERVICE_SELECTION if self.is_booking_lead_type() else ConversationState.WORKFLOW_QUESTION
                else:
                    return ConversationState.PHONE_COLLECTION
            return ConversationState.EMAIL_OTP_VERIFICATION
        
        elif self.state == ConversationState.PHONE_COLLECTION:
            if self.skip_phone_collection:
                return ConversationState.SERVICE_SELECTION if self.is_booking_lead_type() else ConversationState.WORKFLOW_QUESTION
            if self.collected_data["leadPhoneNumber"]:
                if self.validate_phone:
                    return ConversationState.PHONE_OTP_SENT
                else:
                    return ConversationState.SERVICE_SELECTION if self.is_booking_lead_type() else ConversationState.WORKFLOW_QUESTION
            return ConversationState.PHONE_COLLECTION
        
        elif self.state == ConversationState.PHONE_OTP_SENT:
            return ConversationState.PHONE_OTP_VERIFICATION
        
        elif self.state == ConversationState.PHONE_OTP_VERIFICATION:
            if self.otp_state["phone_verified"]:
                return ConversationState.SERVICE_SELECTION if self.is_booking_lead_type() else ConversationState.WORKFLOW_QUESTION
            return ConversationState.PHONE_OTP_VERIFICATION

        elif self.state == ConversationState.APPOINTMENT_OFFER:
            return ConversationState.APPOINTMENT_OFFER

        elif self.state == ConversationState.CALENDAR_BOOKING:
            return ConversationState.CALENDAR_BOOKING

        elif self.state == ConversationState.APPOINTMENT_CONFIRMATION:
            return ConversationState.APPOINTMENT_CONFIRMATION
        
        elif self.state == ConversationState.COMPLETE:
            return ConversationState.COMPLETE
        
        return self.state
    
    def transition_to(self, new_state: ConversationState):
        """Transition to new state"""
        logger.info(f"State transition: {self.state.value} → {new_state.value}")
        self.state = new_state

    def skip_email_verification_after_send_failure(self) -> None:
        """
        Advance when email OTP could not be delivered (provider/network).
        Does not apply when the user enters a wrong code — that path stays in verification.
        """
        self.update_collected_data("emailVerificationSkipped", True)
        self.otp_state["email_verified"] = True
        self.transition_to(ConversationState.EMAIL_OTP_VERIFICATION)
        next_state = self.get_next_state()
        self.transition_to(next_state)

    def skip_phone_verification_after_send_failure(self) -> None:
        """
        Advance when SMS OTP could not be delivered (provider/network).
        Does not apply when the user enters a wrong code — that path stays in verification.
        """
        self.update_collected_data("phoneVerificationSkipped", True)
        self.otp_state["phone_verified"] = True
        self.transition_to(ConversationState.PHONE_OTP_VERIFICATION)
        next_state = self.get_next_state()
        self.transition_to(next_state)
    
    def can_generate_json(self) -> bool:
        """Check if all required data is collected for JSON generation"""
        required_fields = ["leadType", "serviceType", "leadName", "leadEmail"]
        
        # Check required fields
        for field in required_fields:
            if field == "leadName" and not self.capture_lead_name:
                continue
            if field == "leadEmail" and not self.capture_lead_email:
                continue
            if not self.collected_data.get(field):
                return False
        
        # Check phone (if not skipping phone collection)
        if self.capture_lead_phone and (not self.skip_phone_collection) and not self.collected_data.get("leadPhoneNumber"):
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
            "title": self.collected_data.get("title", "")
        }
        if self.capture_lead_name and self.collected_data.get("leadName"):
            data["leadName"] = self.collected_data["leadName"]
        if self.capture_lead_email and self.collected_data.get("leadEmail"):
            data["leadEmail"] = self.collected_data["leadEmail"]
        
        if self.capture_lead_phone and self.collected_data.get("leadPhoneNumber"):
            data["leadPhoneNumber"] = self.collected_data.get("leadPhoneNumber", "")

        if self.collected_data.get("sourceChannel"):
            data["sourceChannel"] = self.collected_data.get("sourceChannel")
        if self.collected_data.get("leadId"):
            data["leadId"] = self.collected_data.get("leadId")
        if self.collected_data.get("appointmentSlot"):
            data["appointmentSlot"] = self.collected_data.get("appointmentSlot")
        if self.collected_data.get("emailVerificationSkipped"):
            data["emailVerificationSkipped"] = True
        if self.collected_data.get("phoneVerificationSkipped"):
            data["phoneVerificationSkipped"] = True
        
        # Add workflow answers if present
        workflow_answers = self.collected_data.get("workflowAnswers", {})
        if workflow_answers:
            data["workflowAnswers"] = workflow_answers
        
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
            ConversationState.WORKFLOW_QUESTION: "Ask workflow question and wait for answer.",
            ConversationState.NAME_COLLECTION: "Ask for the user's name.",
            ConversationState.EMAIL_COLLECTION: "Ask for the user's email address.",
            ConversationState.EMAIL_OTP_SENT: "Acknowledge OTP sent and wait for verification code.",
            ConversationState.EMAIL_OTP_VERIFICATION: "Verify the OTP code provided.",
            ConversationState.PHONE_COLLECTION: "Ask for the user's phone number.",
            ConversationState.PHONE_OTP_SENT: "Acknowledge OTP sent and wait for verification code.",
            ConversationState.PHONE_OTP_VERIFICATION: "Verify the OTP code provided.",
            ConversationState.APPOINTMENT_OFFER: "Ask if the user wants to book now.",
            ConversationState.CALENDAR_BOOKING: "Show and collect available date/time slots.",
            ConversationState.APPOINTMENT_CONFIRMATION: "Ask user to confirm selected appointment details.",
            ConversationState.COMPLETE: "All information collected. Generate JSON."
        }
        return state_prompts.get(self.state, "Continue conversation naturally.")

