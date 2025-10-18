import hashlib
import hmac
import secrets
import time
from typing import Optional


def generate_ts_millis() -> int:
    return int(time.time() * 1000)


def generate_nonce(num_bytes: int = 16) -> str:
    return secrets.token_hex(num_bytes)


def build_signature(
    secret: Optional[str],
    ts_millis: str,
    nonce: str,
    *,
    method: str,
    path: str,
    user_id: str,
) -> str:
    # Exact format required by backend:
    # METHOD\nPATH\nuserId=USER_ID\nTS_MILLIS\nNONCE
    to_sign = f"{method.upper()}\n{path}\nuserId={user_id}\n{ts_millis}\n{nonce}"
    if not secret:
        return hashlib.sha256(to_sign.encode("utf-8")).hexdigest()
    key = secret.encode("utf-8")
    return hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def build_signature_with_param(
    secret: Optional[str],
    ts_millis: str,
    nonce: str,
    *,
    method: str,
    path: str,
    param_name: str,
    param_value: str,
) -> str:
    # Format for parameterized routes:
    # METHOD\nPATH\nPARAM_NAME=PARAM_VALUE\nTS_MILLIS\nNONCE
    to_sign = f"{method.upper()}\n{path}\n{param_name}={param_value}\n{ts_millis}\n{nonce}"
    if not secret:
        return hashlib.sha256(to_sign.encode("utf-8")).hexdigest()
    key = secret.encode("utf-8")
    return hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
