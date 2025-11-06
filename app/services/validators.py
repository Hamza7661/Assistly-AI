"""Data validators for extracted information"""
import re
from typing import Optional
import logging

logger = logging.getLogger("assistly.validators")


class Validator:
    """Validate extracted data"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format"""
        if not email:
            return False
        
        pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
        is_valid = bool(re.match(pattern, email))
        if not is_valid:
            logger.warning(f"Invalid email format: {email}")
        return is_valid
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """Validate phone number format (basic check)"""
        if not phone:
            return False
        
        # Remove common separators
        cleaned = re.sub(r'[-.\s()]', '', phone)
        
        # Check if it's all digits (with optional + prefix)
        if cleaned.startswith('+'):
            cleaned = cleaned[1:]
        
        # Should be 10-15 digits
        is_valid = cleaned.isdigit() and 10 <= len(cleaned) <= 15
        if not is_valid:
            logger.warning(f"Invalid phone format: {phone}")
        return is_valid
    
    @staticmethod
    def is_valid_otp(otp: str) -> bool:
        """Validate OTP code format"""
        if not otp:
            return False
        
        # Should be exactly 6 digits
        is_valid = bool(re.match(r'^\d{6}$', otp))
        if not is_valid:
            logger.warning(f"Invalid OTP format: {otp}")
        return is_valid
    
    @staticmethod
    def is_valid_name(name: str) -> bool:
        """Validate name format"""
        if not name:
            return False
        
        # Should be 2-50 characters, mostly letters
        if not 2 <= len(name) <= 50:
            return False
        
        # Should contain at least one letter
        if not re.search(r'[A-Za-z]', name):
            return False
        
        # Should not contain numbers (unless it's a valid name with numbers)
        # Allow common name characters
        if not re.match(r'^[A-Za-z\s\-\'\.]+$', name):
            return False
        
        return True

