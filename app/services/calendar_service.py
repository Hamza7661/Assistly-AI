"""Calendar service: call backend for availability (and later create event)."""
from typing import Any, Dict, List, Optional
import time
from urllib.parse import urlencode

import httpx
import logging

from ..utils.signing import build_signature_with_param, generate_nonce, generate_ts_millis

logger = logging.getLogger("assistly.calendar")


class CalendarService:
    """Calls backend calendar API (HMAC-signed) for availability."""

    def __init__(self, settings: Any) -> None:
        self.base_url: str = settings.api_base_url.rstrip("/")
        self.secret: Optional[str] = getattr(settings, "tp_sign_secret", None)

    async def get_availability(
        self,
        app_id: str,
        from_date: str,
        to_date: str,
        slot_minutes: int = 30,
    ) -> Dict[str, Any]:
        """
        GET /api/v1/calendar/apps/:appId/availability?from=...&to=...
        Returns backend response with freeSlots, busy, calendarConnected.
        """
        path = f"/api/v1/calendar/apps/{app_id}/availability"
        query = {"from": from_date, "to": to_date}
        if slot_minutes != 30:
            query["slotMinutes"] = str(slot_minutes)
        path_with_query = f"{path}?{urlencode(query)}"
        url = f"{self.base_url}{path_with_query}"

        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature_with_param(
            self.secret,
            ts,
            nonce,
            method="GET",
            path=path,
            param_name="appId",
            param_value=str(app_id),
        )

        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "accept": "application/json",
        }

        logger.info("Calendar get_availability request app_id=%s from=%s to=%s", app_id, from_date, to_date)

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status")
        payload = data.get("data", {})
        if status != "success":
            return {"calendarConnected": False, "freeSlots": [], "busy": [], "error": payload.get("message", "Unknown error")}

        return {
            "calendarConnected": payload.get("calendarConnected", False),
            "freeSlots": payload.get("freeSlots", []),
            "busy": payload.get("busy", []),
            "timeMin": payload.get("timeMin"),
            "timeMax": payload.get("timeMax"),
            "message": payload.get("message"),
        }

    async def book_appointment(
        self,
        app_id: str,
        start_iso: str,
        end_iso: str,
        title: str = "Appointment",
        attendee_email: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST /api/v1/calendar/apps/:appId/appointments
        Body: { start, end, title, attendeeEmail?, description? }
        """
        path = f"/api/v1/calendar/apps/{app_id}/appointments"
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
            param_value=str(app_id),
        )

        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        body: Dict[str, Any] = {
            "start": start_iso,
            "end": end_iso,
            "title": title,
        }
        if attendee_email:
            body["attendeeEmail"] = attendee_email
        if description:
            body["description"] = description

        logger.info("Calendar book_appointment app_id=%s start=%s", app_id, start_iso)

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status")
        payload = data.get("data", {})
        if status != "success" and status != "error":
            return {"success": False, "error": payload.get("message", "Unknown error")}
        return {
            "success": payload.get("success", False),
            "eventId": payload.get("eventId"),
            "link": payload.get("link"),
            "start": payload.get("start"),
            "end": payload.get("end"),
            "title": payload.get("title"),
            "error": payload.get("error"),
        }
