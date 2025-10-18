import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    api_base_url: str = Field(default="http://localhost:5000", alias="API_BASE_URL")
    frontend_base_url: str = Field(default="http://localhost:3000", alias="FRONTEND_BASE_URL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    gpt_model: str = Field(default="gpt-5-nano", alias="GPT_MODEL")

    tp_sign_secret: Optional[str] = Field(default=None, alias="TP_SIGN_SECRET")

    max_history_messages: int = Field(default=10, alias="MAX_HISTORY_MESSAGES")
    
    # Twilio WhatsApp configuration
    twilio_account_sid: Optional[str] = Field(default=None, alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(default=None, alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_from: Optional[str] = Field(default=None, alias="TWILIO_WHATSAPP_FROM")
    whatsapp_webhook_token: Optional[str] = Field(default=None, alias="WHATSAPP_WEBHOOK_TOKEN")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(**os.environ)


settings = get_settings()
