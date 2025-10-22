from typing import Any, Dict, Optional, Tuple
import time

import httpx
import logging
import json
import phonenumbers
from phonenumbers import NumberParseException

from ..utils.signing import build_signature, generate_nonce, generate_ts_millis
from ..utils.phone_utils import format_phone_number, is_valid_phone_number

logger = logging.getLogger("assistly.phone_validation")


class PhoneValidationService:
    def __init__(self, settings: Any) -> None:
        self.backend_url: str = settings.api_base_url.rstrip("/")
        self.secret: Optional[str] = settings.tp_sign_secret

    async def send_sms_otp(self, user_id: str, phone: str) -> Tuple[bool, str]:
        """Send SMS OTP to user's phone number using phone utils for formatting."""
        try:
            # Format phone number using phone utils (automatically detects country)
            formatted_phone = format_phone_number(phone)
            
            # Validate phone number
            if not is_valid_phone_number(formatted_phone):
                return False, f"Invalid phone number: {formatted_phone}"
            
            logger.info(f"Sending SMS OTP to formatted phone: {formatted_phone}")
            
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
                    
        except Exception as e:
            logger.error(f"Error sending SMS OTP: {e}")
            return False, f"Failed to send SMS OTP: {str(e)}"
                
    async def verify_sms_otp(self, user_id: str, phone: str, otp: str) -> Tuple[bool, str]:
        """Verify the SMS OTP code entered by user using phone utils for formatting."""
        try:
            # Format phone number using phone utils (automatically detects country)
            formatted_phone = format_phone_number(phone)
            
            # Validate phone number
            if not is_valid_phone_number(formatted_phone):
                return False, f"Invalid phone number: {formatted_phone}"
            
            logger.info(f"Verifying SMS OTP for formatted phone: {formatted_phone}")
            
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
                    error_message = error_data.get("message", "Failed to verify SMS OTP")
                    # Handle maximum attempts exceeded more gracefully
                    if "maximum attempts" in error_message.lower():
                        return False, "Please try again with a new verification code"
                    return False, error_message
                except Exception:
                    return False, f"Failed to verify SMS OTP (status: {resp.status_code})"
                    
        except Exception as e:
            logger.error(f"Error verifying SMS OTP: {e}")
            return False, f"Failed to verify SMS OTP: {str(e)}"
