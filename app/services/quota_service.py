"""Signed calls to backend AppPlan quota API."""

from typing import Any, Dict, Optional, Tuple

import httpx
import logging

from ..utils.signing import build_signature_with_param, generate_nonce, generate_ts_millis

logger = logging.getLogger("assistly.quota")


class AppPlanChannelDenied(Exception):
    """Raised when a channel is disabled or quota is exceeded (user-facing message)."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


def user_message_for_app_plan_denial(reason: str) -> str:
    if reason == "channel_disabled":
        return (
            "This contact channel is not available for this business right now. "
            "Please try again later or reach them another way."
        )
    if reason == "quota_exceeded":
        return (
            "We're unable to start new conversations on this channel at the moment. "
            "Please try again later."
        )
    return "Service is temporarily unavailable. Please try again later."


def apply_app_plan_sms_addon(context: Dict[str, Any]) -> None:
    """If AppPlan includes SMS verification add-on, turn on phone OTP in integration."""
    plan = context.get("appPlan")
    if not isinstance(plan, dict):
        return
    addons = plan.get("addons")
    if not isinstance(addons, dict) or "smsVerification" not in addons:
        return
    integration = context.setdefault("integration", {})
    integration["validatePhoneNumber"] = bool(addons.get("smsVerification"))


async def enforce_app_plan_for_channel(
    settings: Any,
    context: Dict[str, Any],
    app_id: Optional[str],
    channel: str,
) -> Optional[str]:
    """
    Returns None if traffic is allowed, or 'channel_disabled' | 'quota_exceeded'.
    Missing appPlan = backward compatible (allow).
    """
    if not app_id:
        return None
    plan = context.get("appPlan")
    if not isinstance(plan, dict) or not plan:
        return None
    ch_cfg = (plan.get("channels") or {}).get(channel) or {}
    if ch_cfg.get("enabled") is False:
        return "channel_disabled"
    quota_svc = QuotaService(settings)
    _ok, data = await quota_svc.check_and_increment(str(app_id), channel)
    if isinstance(data, dict) and data.get("allowed") is False:
        return "quota_exceeded"
    return None


class QuotaService:
    def __init__(self, settings: Any) -> None:
        self.base_url: str = settings.api_base_url.rstrip("/")
        self.secret: Optional[str] = getattr(settings, "tp_sign_secret", None)

    async def check_and_increment(self, app_id: str, channel: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns (http_ok, data) where data includes allowed: bool when success.
        If no secret configured, allows traffic (fail-open for local dev).
        """
        if not self.secret:
            logger.warning("tp_sign_secret missing — quota check skipped (allowed)")
            return True, {"allowed": True, "remaining": None, "unlimited": True, "reason": "no_secret"}

        path = f"/api/v1/app-plan/apps/{app_id}/quota/{channel}/check-and-increment"
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
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0)) as client:
            resp = await client.post(url, headers=headers, json={})
            if resp.status_code < 200 or resp.status_code >= 300:
                logger.warning("Quota API error %s: %s", resp.status_code, resp.text[:200])
                return False, {"allowed": True, "reason": "api_error_fail_open"}
            try:
                body = resp.json()
            except Exception:  # noqa: BLE001
                return False, {"allowed": True, "reason": "parse_error_fail_open"}
            data = (body or {}).get("data") or {}
            return True, data
