from typing import Any, Dict, Optional, Tuple
import time
import httpx
import logging
import json

from ..utils.signing import build_signature, generate_nonce, generate_ts_millis

logger = logging.getLogger("assistly.email_validation")


class EmailValidationService:
    def __init__(self, settings: Any) -> None:
        self.backend_url: str = settings.api_base_url.rstrip("/")
        self.frontend_url: str = settings.frontend_base_url.rstrip("/")
        self.secret: Optional[str] = settings.tp_sign_secret

    async def get_otp_template(self, customer_name: str) -> str:
        """Get OTP verification template from the frontend API."""
        path = f"/templates/otp-verification?customerName={customer_name}"
        url = f"{self.frontend_url}{path}"
        
        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature(self.secret, ts, nonce, method="GET", path=path, user_id="")
        
        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "accept": "application/json",
        }
        
        start_time = time.time()
        logger.info("Sending OTP template request at %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            
            # Try to parse as JSON first, fallback to HTML content
            try:
                template_data = resp.json()
                html_template = template_data.get("htmlTemplate", "")
            except Exception:
                # If JSON parsing fails, treat as HTML content
                html_template = resp.text
            
        end_time = time.time()
        duration = end_time - start_time
        logger.info("Received OTP template response at %s (took %.3fs)", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration)
        
        return html_template

    async def send_otp_email(self, user_id: str, email: str, customer_name: str) -> Tuple[bool, str]:
        """Send OTP verification email to user."""
        # Get the OTP template
        html_template = await self.get_otp_template(customer_name)
        
        path = f"/api/v1/otp/send-email/{user_id}"
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
            "email": email,
            "htmlTemplate": html_template
        }
        
        start_time = time.time()
        logger.info("Sending OTP email request at %s for user_id=%s, email=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), user_id, email)
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            
            if resp.status_code >= 200 and resp.status_code < 300:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("OTP email sent successfully at %s (took %.3fs) for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                return True, "OTP sent successfully"
            else:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("OTP email failed at %s (took %.3fs) for user_id=%s, status=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id, resp.status_code)
                try:
                    error_data = resp.json()
                    return False, error_data.get("message", "Failed to send OTP email")
                except Exception:
                    return False, f"Failed to send OTP email (status: {resp.status_code})"
                    
    async def verify_otp(self, user_id: str, email: str, otp: str) -> Tuple[bool, str]:
        """Verify the OTP code entered by user."""
        path = f"/api/v1/otp/verify-email/{user_id}"
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
            "email": email,
            "otp": otp
        }
        
        start_time = time.time()
        logger.info("Sending OTP verification request at %s for user_id=%s, email=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), user_id, email)
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            
            if resp.status_code >= 200 and resp.status_code < 300:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("OTP verification successful at %s (took %.3fs) for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                return True, "Email verified successfully"
            else:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("OTP verification failed at %s (took %.3fs) for user_id=%s, status=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id, resp.status_code)
                try:
                    error_data = resp.json()
                    return False, error_data.get("message", "Invalid OTP code")
                except Exception:
                    return False, f"OTP verification failed (status: {resp.status_code})"
