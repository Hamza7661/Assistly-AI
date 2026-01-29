from typing import Any, Dict, List, Optional
import time
from urllib.parse import quote

import httpx
import logging
import json

from ..utils.signing import build_signature, build_signature_with_param, generate_nonce, generate_ts_millis

logger = logging.getLogger("assistly.context")


class ContextService:
    def __init__(self, settings: Any) -> None:
        self.base_url: str = settings.api_base_url.rstrip("/")
        self.secret: Optional[str] = settings.tp_sign_secret

    async def fetch_user_context(self, user_id: str) -> Dict[str, Any]:
        path = f"/api/v1/users/public/{user_id}/context"
        url = f"{self.base_url}{path}"

        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature(self.secret, ts, nonce, method="GET", path=path, user_id=user_id)

        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "accept": "application/json",
        }

        start_time = time.time()
        logger.info("Sending context API request at %s for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), user_id)

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        end_time = time.time()
        duration = end_time - start_time
        logger.info("Received context API response at %s (took %.3fs) for user_id=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, user_id)

        try:
            logger.info("Context API response for user_id=%s: %s", user_id, json.dumps(data))
        except Exception:  # noqa: BLE001
            logger.info("Context API response for user_id=%s (non-serializable)", user_id)

        normalized = self._normalize_context(data)
        try:
            logger.info("Normalized context for user_id=%s: %s", user_id, json.dumps(normalized))
        except Exception:  # noqa: BLE001
            logger.info("Normalized context for user_id=%s (non-serializable)", user_id)

        return normalized
    
    async def fetch_user_context_by_twilio(self, twilio_phone: str) -> Dict[str, Any]:
        """Fetch user context using Twilio phone number"""
        # Remove 'whatsapp:' prefix if present
        clean_phone = twilio_phone.replace("whatsapp:", "")
        
        # URL encode the phone number for the actual request URL
        encoded_phone = quote(clean_phone, safe='')
        
        # Path for signature (use encoded version)
        path = f"/api/v1/users/by-twilio/{encoded_phone}/context"
        url = f"{self.base_url}{path}"

        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        # Use the parameterized signature builder with twilioPhoneNumber
        sign = build_signature_with_param(
            self.secret, 
            ts, 
            nonce, 
            method="GET", 
            path=path, 
            param_name="twilioPhoneNumber",
            param_value=clean_phone  # Use non-encoded phone in the signature
        )

        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "accept": "application/json",
        }

        start_time = time.time()
        logger.info("Sending context API request at %s for Twilio phone=%s", 
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), clean_phone)

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.get(url, headers=headers)
            
            # Log response details
            logger.info("Context API response status: %s for Twilio phone=%s", resp.status_code, clean_phone)
            try:
                response_text = resp.text
                logger.info("Context API response body: %s", response_text)
            except Exception:
                logger.info("Could not read response body")
            
            resp.raise_for_status()
            data = resp.json()

        end_time = time.time()
        duration = end_time - start_time
        logger.info("Received context API response at %s (took %.3fs) for Twilio phone=%s", 
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, clean_phone)

        try:
            logger.info("Context API response for Twilio phone=%s: %s", clean_phone, json.dumps(data))
        except Exception:  # noqa: BLE001
            logger.info("Context API response for Twilio phone=%s (non-serializable)", clean_phone)

        normalized = self._normalize_context(data)
        try:
            logger.info("Normalized context for Twilio phone=%s: %s", clean_phone, json.dumps(normalized))
        except Exception:  # noqa: BLE001
            logger.info("Normalized context for Twilio phone=%s (non-serializable)", clean_phone)

        return normalized

    async def fetch_context_by_app(self, app_id: str) -> Dict[str, Any]:
        """Fetch context by app ID (app-wise embedding; same user can have multiple apps)."""
        path = f"/api/v1/users/public/apps/{quote(app_id, safe='')}/context"
        url = f"{self.base_url}{path}"

        ts = str(generate_ts_millis())
        nonce = generate_nonce()
        sign = build_signature_with_param(
            self.secret,
            ts,
            nonce,
            method="GET",
            path=path,
            param_name="appId",
            param_value=app_id,
        )

        headers = {
            "x-tp-ts": ts,
            "x-tp-nonce": nonce,
            "x-tp-sign": sign,
            "accept": "application/json",
        }

        start_time = time.time()
        logger.info("Sending context API request at %s for app_id=%s",
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), app_id)

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        end_time = time.time()
        duration = end_time - start_time
        logger.info("Received context API response at %s (took %.3fs) for app_id=%s",
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), duration, app_id)

        normalized = self._normalize_context(data)
        return normalized

    def _normalize_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Many backends wrap payload inside a top-level 'data' key
        src = data.get("data", data) if isinstance(data, dict) else {}

        def first_present(keys: List[str], default: Any) -> Any:
            for key in keys:
                if key in src and src[key] not in (None, [], ""):
                    return src[key]
            return default

        lead_types_raw = first_present([
            "lead_types",
            "leadTypes",
            "leadtypes",
            "lead_types_options",
        ], [])

        lead_types: List[Dict[str, Any]] = []
        if isinstance(lead_types_raw, list) and lead_types_raw and isinstance(lead_types_raw[0], dict):
            # Already in desired shape
            for idx, item in enumerate(lead_types_raw, start=1):
                value = str(item.get("value") or item.get("id") or idx)
                text = str(item.get("text") or value)
                lead_types.append({"id": item.get("id") or idx, "value": value, "text": text})
        elif isinstance(lead_types_raw, list):
            # List of strings; map to generic objects
            for idx, val in enumerate(lead_types_raw, start=1):
                sval = str(val)
                lead_types.append({
                    "id": idx,
                    "value": sval,
                    "text": sval,
                })
        else:
            lead_types = []

        # FAQs may be under 'faq' on the root, or nested inside src
        faqs = first_present([
            "faqs",
            "FAQs",
            "faq",
            "questions",
        ], [])

        # Treatment plans
        treatment_plans = first_present([
            "treatmentPlans",
            "treatment_plans",
            "treatment_plans_options",
            "treatments",
        ], [])
        
        # Capitalize first letter of treatment plan questions
        if isinstance(treatment_plans, list):
            for plan in treatment_plans:
                if isinstance(plan, dict) and "question" in plan:
                    plan["question"] = plan["question"].capitalize()

        # Profession description may be under src['user']['professionDescription']
        profession = ""
        if isinstance(src, dict):
            user_obj = src.get("user") or {}
            profession = (
                user_obj.get("professionDescription")
                or user_obj.get("profession")
                or src.get("professionDescription")
                or src.get("profession")
                or ""
            )

        # Extract integration data (assistantName, greeting, etc.)
        integration = first_present([
            "integration",
            "Integration",
            "assistant",
            "Assistant",
        ], {})

        # Extract country code
        country = first_present([
            "country",
            "Country",
            "countryCode",
            "country_code",
        ], "US")

        # Extract user data
        user_data = first_present([
            "user",
            "User",
            "userData",
            "user_data",
        ], {})

        # Extract workflows (needed for workflow questions)
        workflows = first_present([
            "workflows",
            "Workflows",
            "workflow",
            "Workflow",
        ], [])

        return {
            "lead_types": lead_types,
            "treatment_plans": treatment_plans,
            "faqs": faqs,
            "profession": profession,
            "integration": integration,
            "country": country,
            "user": user_data,  # Preserve user data
            "workflows": workflows,  # Preserve workflows for workflow questions
        }
