from typing import Any, Dict, Optional, Tuple

import httpx
from ..utils.signing import build_signature, generate_nonce, generate_ts_millis


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
        # Optional: print or integrate with a logger
        # print("Creating public lead:", masked)

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 200 and resp.status_code < 300:
                return True, resp.json()
            try:
                return False, resp.json()
            except Exception:  # noqa: BLE001
                return False, resp.text
