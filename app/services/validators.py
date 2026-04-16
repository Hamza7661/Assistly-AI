"""Data validators for extracted information"""
import re
from typing import Optional
import logging

logger = logging.getLogger("assistly.validators")

# Common regions tried when the user's number has no country prefix.
# Ordered roughly by global mobile subscriber share so the most likely
# match is found quickly.
_PHONE_FALLBACK_REGIONS = [
    "GB", "US", "PK", "IN", "AU", "CA", "AE", "NG", "ZA", "DE",
    "FR", "BD", "BR", "ID", "MX", "PH", "EG", "KE", "SA", "TR",
    "MY", "GH", "TZ", "SG", "IE", "NZ", "NL", "IT", "ES", "UA",
]


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
    def is_valid_phone(phone: str, country_hint: Optional[str] = None) -> bool:
        """Validate phone number using Google's libphonenumber.

        Accepts numbers in any of these forms:
        - E.164 international format  (+447700900123)
        - Local format with a country hint  (07700900123, hint="GB")
        - Local format with no hint — tried against _PHONE_FALLBACK_REGIONS

        Returns False for structurally bogus numbers (wrong length, invalid
        area code, reserved ranges, etc.) even if digit count looks right.
        Falls back to basic digit-length check when the library is unavailable.
        """
        if not phone:
            return False

        cleaned = re.sub(r'[\s\-\.\(\)]', '', phone).strip()
        if not cleaned:
            return False

        try:
            import phonenumbers  # type: ignore

            # ── E.164 path (has leading +) ──────────────────────────────────
            if cleaned.startswith('+'):
                try:
                    parsed = phonenumbers.parse(cleaned, None)
                    valid = phonenumbers.is_valid_number(parsed)
                    if not valid:
                        logger.warning(f"Invalid E.164 phone: {phone}")
                    return valid
                except phonenumbers.NumberParseException:
                    logger.warning(f"Cannot parse E.164 phone: {phone}")
                    return False

            # ── Local format path ────────────────────────────────────────────
            regions = []
            if country_hint and re.fullmatch(r'[A-Za-z]{2}', country_hint):
                regions.append(country_hint.upper())
            for r in _PHONE_FALLBACK_REGIONS:
                if r not in regions:
                    regions.append(r)

            for region in regions:
                try:
                    parsed = phonenumbers.parse(cleaned, region)
                    if phonenumbers.is_valid_number(parsed):
                        logger.info(f"Phone {phone} valid for region {region}")
                        return True
                except phonenumbers.NumberParseException:
                    continue

            logger.warning(f"Phone {phone} failed validation against all regions")
            return False

        except ImportError:
            # phonenumbers not installed — fall back to basic digit-length check
            digits = re.sub(r'\D', '', cleaned)
            is_valid = digits.isdigit() and 10 <= len(digits) <= 15
            if not is_valid:
                logger.warning(f"Invalid phone format (fallback): {phone}")
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

