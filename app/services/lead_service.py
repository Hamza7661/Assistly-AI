from typing import Any, Dict, Optional, Tuple
import time

import httpx
import logging
from ..utils.signing import build_signature, build_signature_with_param, generate_nonce, generate_ts_millis

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

    async def create_interaction_lead(
        self,
        app_id: Optional[str],
        user_id: str,
        location: Optional[Dict[str, Any]] = None,
        initial_interaction: str = "widget_opened",
        source_channel: Optional[str] = None,
        dedupe_window_hours: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any] | str]:
        payload: Dict[str, Any] = {
            "status": "interacting",
            "initialInteraction": initial_interaction,
            "clickedItems": [],
        }
        if app_id:
            payload["appId"] = app_id
        if location:
            payload["location"] = location
        if source_channel:
            payload["sourceChannel"] = source_channel
        if isinstance(dedupe_window_hours, int) and dedupe_window_hours > 0:
            payload["dedupeWindowHours"] = dedupe_window_hours
        return await self.create_public_lead(user_id, payload)

    async def update_lead(self, user_id: str, lead_id: str, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any] | str]:
        path = f"/api/v1/leads/public/{user_id}/{lead_id}"
        url = f"{self.base_url}{path}"
        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature(self.secret, ts, nonce, method="PATCH", path=path, user_id=user_id)
        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.patch(url, headers=headers, json=payload)
            if 200 <= resp.status_code < 300:
                return True, resp.json()
            try:
                return False, resp.json()
            except Exception:
                return False, resp.text

    async def consume_channel_conversation(
        self,
        app_id: str,
        channel: str,
        idempotency_key: str,
    ) -> Tuple[bool, Dict[str, Any] | str]:
        path = f"/api/v1/subscription-state/public/apps/{app_id}/consume-conversation"
        url = f"{self.base_url}{path}"
        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature_with_param(
            self.secret,
            ts,
            nonce,
            method="POST",
            path=path,
            param_name="appId",
            param_value=app_id,
        )

        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        payload: Dict[str, Any] = {
            "channel": channel,
            "idempotencyKey": idempotency_key,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if 200 <= resp.status_code < 300:
                return True, resp.json()
            try:
                return False, resp.json()
            except Exception:
                return False, resp.text
