from typing import Any, Dict, Optional, Tuple
import time

import httpx
import logging
import json
import phonenumbers
from phonenumbers import NumberParseException

from ..utils.signing import build_signature, generate_nonce, generate_ts_millis

logger = logging.getLogger("assistly.phone_validation")


class PhoneValidationService:
    def __init__(self, settings: Any) -> None:
        self.backend_url: str = settings.api_base_url.rstrip("/")
        self.secret: Optional[str] = settings.tp_sign_secret

    def _detect_country_from_prefix(self, phone: str) -> str:
        """Detect country code from phone number prefix if it starts with +"""
        if not phone.startswith('+'):
            return None
            
        # Remove + and get the first few digits
        digits = phone[1:]
        
        # Check for country codes in order of specificity (longer codes first)
        if digits.startswith('1'):
            return "US"  # +1 for US/Canada
        elif digits.startswith('92'):
            return "PK"  # +92 for Pakistan
        elif digits.startswith('44'):
            return "GB"  # +44 for UK
        else:
            return None
    
    def _get_country_with_priority(self, phone: str, user_country: str = None, api_country: str = "US") -> str:
        """Get country code with priority: user_country > phone_prefix > api_country"""
        # Priority 1: User country from chat response
        if user_country:
            logger.info(f"Using user country '{user_country}' (highest priority)")
            return user_country
        
        # Priority 2: Detect from phone number prefix
        detected_country = self._detect_country_from_prefix(phone)
        if detected_country:
            logger.info(f"Using detected country '{detected_country}' from phone prefix '{phone}' (second priority)")
            return detected_country
        
        # Priority 3: API country as fallback
        logger.info(f"Using API country '{api_country}' as fallback (lowest priority)")
        return api_country

    def _format_phone_with_country_code(self, phone: str, country_code: str = "US", user_country: str = None) -> str:
        """Format phone number with country code using phonenumbers library."""
        logger.info(f"Converting phone number: '{phone}' with country_code: '{country_code}', user_country: '{user_country}'")
        
        # Get country with priority: user_country > phone_prefix > api_country
        final_country = self._get_country_with_priority(phone, user_country, country_code)
        
        try:
            # Parse the phone number with the final country code
            parsed_number = phonenumbers.parse(phone, final_country)
            
            # Format as international number
            formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
            logger.info(f"Phone number conversion successful: '{phone}' -> '{formatted}' using country '{final_country}'")
            return formatted
        except NumberParseException as e:
            logger.warning(f"Failed to parse phone number '{phone}' with country '{final_country}': {e}")
            # Fallback: if it already starts with +, return as is, otherwise add +
            fallback = phone if phone.startswith('+') else f"+{phone}"
            logger.info(f"Using fallback phone format: '{phone}' -> '{fallback}'")
            return fallback

    async def send_sms_otp(self, user_id: str, phone: str, country_code: str = "US", user_country: str = None) -> Tuple[bool, str]:
        """Send SMS OTP to user's phone number."""
        # Format phone number with country code (with priority: user_country > phone_prefix > api_country)
        formatted_phone = self._format_phone_with_country_code(phone, country_code, user_country)
        
        path = f"/api/v1/otp/send-sms/{user_id}"
        url = f"{self.backend_url}{path}"
        
        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature(self.secret, ts, nonce, method="POST", path=path, user_id=user_id)
        
        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        
        payload = {
            "phoneNumber": formatted_phone
        }

        start_time = time.time()
        logger.info("Sending SMS OTP request at %s for user_id=%s, phone=%s", 
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), user_id, formatted_phone)

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 200 and resp.status_code < 300:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("Received SMS OTP response at %s (took %.3fs) for user_id=%s", 
                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                return True, "SMS OTP sent successfully"
            try:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("Received SMS OTP error response at %s (took %.3fs) for user_id=%s", 
                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                error_data = resp.json()
                return False, error_data.get("message", "Failed to send SMS OTP")
            except Exception:
                return False, f"Failed to send SMS OTP (status: {resp.status_code})"
                
    async def verify_sms_otp(self, user_id: str, phone: str, otp: str, country_code: str = "US", user_country: str = None) -> Tuple[bool, str]:
        """Verify the SMS OTP code entered by user."""
        # Format phone number with country code (with priority: user_country > phone_prefix > api_country)
        formatted_phone = self._format_phone_with_country_code(phone, country_code, user_country)
        
        path = f"/api/v1/otp/verify-sms/{user_id}"
        url = f"{self.backend_url}{path}"
        
        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature(self.secret, ts, nonce, method="POST", path=path, user_id=user_id)
        
        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        
        payload = {
            "phoneNumber": formatted_phone,
            "otp": otp
        }

        start_time = time.time()
        logger.info("Sending SMS OTP verification request at %s for user_id=%s, phone=%s", 
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), user_id, formatted_phone)

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 200 and resp.status_code < 300:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("Received SMS OTP verification response at %s (took %.3fs) for user_id=%s", 
                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                return True, "SMS OTP verified successfully"
            try:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("Received SMS OTP verification error response at %s (took %.3fs) for user_id=%s", 
                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                error_data = resp.json()
                return False, error_data.get("message", "Failed to verify SMS OTP")
            except Exception:
                return False, f"Failed to verify SMS OTP (status: {resp.status_code})"
