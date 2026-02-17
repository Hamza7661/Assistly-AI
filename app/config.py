import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    api_base_url: str = Field(default="http://localhost:5000", alias="API_BASE_URL")
    frontend_base_url: str = Field(default="http://localhost:3000", alias="FRONTEND_BASE_URL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    gpt_model: str = Field(default="gpt-4.1-nano", alias="GPT_MODEL")

    tp_sign_secret: Optional[str] = Field(default=None, alias="TP_SIGN_SECRET")

    max_history_messages: int = Field(default=10, alias="MAX_HISTORY_MESSAGES")
    
    # Session configuration
    session_timeout_seconds: int = Field(default=300, alias="SESSION_TIMEOUT_SECONDS")  # 5 minutes default
    # Secret for backend to call invalidate-whatsapp-sessions (optional; if set, header X-Invalidate-Sessions-Secret must match).
    # Leave unset for now; add INVALIDATE_SESSIONS_SECRET to .env later (and same value as ASSISTLY_INVALIDATE_SESSIONS_SECRET in backend).
    invalidate_sessions_secret: Optional[str] = Field(default=None, alias="INVALIDATE_SESSIONS_SECRET")
    
    # Twilio WhatsApp configuration
    twilio_account_sid: Optional[str] = Field(default=None, alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(default=None, alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_from: Optional[str] = Field(default=None, alias="TWILIO_WHATSAPP_FROM")
    whatsapp_webhook_token: Optional[str] = Field(default=None, alias="WHATSAPP_WEBHOOK_TOKEN")
    
    # Meta (Facebook/Instagram) configuration
    meta_verify_token: Optional[str] = Field(default="assistly_instagram_verify_token", alias="META_VERIFY_TOKEN")
    meta_graph_api_version: str = Field(default="v21.0", alias="META_GRAPH_API_VERSION")

    # Deepgram voice agent
    deepgram_api_key: Optional[str] = Field(default=None, alias="DEEPGRAM_API_KEY")
    
    # RAG (Retrieval-Augmented Generation) configuration
    rag_k: int = Field(default=3, alias="RAG_K")  # Number of documents to retrieve
    rag_persist_directory: Optional[str] = Field(default=None, alias="RAG_PERSIST_DIRECTORY")  # Optional persistent storage


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(**os.environ)


settings = get_settings()
