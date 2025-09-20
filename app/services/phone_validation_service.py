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

    def _format_phone_with_country_code(self, phone: str, country_code: str = "US") -> str:
        """Format phone number with country code using phonenumbers library."""
        logger.info(f"Converting phone number: '{phone}' with country code: '{country_code}'")
        
        try:
            # Parse the phone number with the given country code
            parsed_number = phonenumbers.parse(phone, country_code)
            
            # Format as international number
            formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
            logger.info(f"Phone number conversion successful: '{phone}' -> '{formatted}'")
            return formatted
        except NumberParseException as e:
            logger.warning(f"Failed to parse phone number '{phone}' with country '{country_code}': {e}")
            # Fallback: if it already starts with +, return as is, otherwise add +
            fallback = phone if phone.startswith('+') else f"+{phone}"
            logger.info(f"Using fallback phone format: '{phone}' -> '{fallback}'")
            return fallback

    async def send_sms_otp(self, user_id: str, phone: str, country_code: str = "US") -> Tuple[bool, str]:
        """Send SMS OTP to user's phone number."""
        # Format phone number with country code
        formatted_phone = self._format_phone_with_country_code(phone, country_code)
        
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
                
    async def verify_sms_otp(self, user_id: str, phone: str, otp: str, country_code: str = "US") -> Tuple[bool, str]:
        """Verify the SMS OTP code entered by user."""
        # Format phone number with country code
        formatted_phone = self._format_phone_with_country_code(phone, country_code)
        
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
