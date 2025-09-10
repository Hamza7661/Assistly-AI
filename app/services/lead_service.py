from typing import Any, Dict, Optional, Tuple
import time

import httpx
import logging
from ..utils.signing import build_signature, generate_nonce, generate_ts_millis

logger = logging.getLogger("assistly.lead")


class LeadService:
    def __init__(self, settings: Any) -> None:
        self.base_url: str = settings.api_base_url.rstrip("/")
        self.secret: Optional[str] = getattr(settings, "tp_sign_secret", None)

    # Private lead creation (JWT) removed per public-only requirement

    async def create_public_lead(self, user_id: str, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any] | str]:
        path = f"/api/v1/leads/public/{user_id}"
        url = f"{self.base_url}{path}"
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

        # Mask sensitive fields in logs
        masked = dict(payload)
        if masked.get("leadName"):
            masked["leadName"] = "***"
        if masked.get("leadEmail"):
            masked["leadEmail"] = "***@***"
        if masked.get("leadPhoneNumber"):
            masked["leadPhoneNumber"] = "********"
        logger.info("Creating public lead for user %s: %s", user_id, masked)

        start_time = time.time()
        logger.info("Sending lead creation API request at %s for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), user_id)

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 200 and resp.status_code < 300:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("Received lead creation API response at %s (took %.3fs) for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                return True, resp.json()
            try:
                end_time = time.time()
                duration = end_time - start_time
                logger.info("Received lead creation API error response at %s (took %.3fs) for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                return False, resp.json()
            except Exception:  # noqa: BLE001
                end_time = time.time()
                duration = end_time - start_time
                logger.info("Received lead creation API error response at %s (took %.3fs) for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)
                return False, resp.text
