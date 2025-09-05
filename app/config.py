import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    api_base_url: str = Field(default="http://localhost:5000", alias="API_BASE_URL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    gpt_model: str = Field(default="gpt-5-nano", alias="GPT_MODEL")

    tp_sign_secret: Optional[str] = Field(default=None, alias="TP_SIGN_SECRET")

    max_history_messages: int = Field(default=10, alias="MAX_HISTORY_MESSAGES")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(**os.environ)


settings = get_settings()
