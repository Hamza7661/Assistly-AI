"""Deepgram Voice Agent integration helpers."""
from __future__ import annotations

from typing import Any, Optional

from deepgram import AsyncDeepgramClient
from deepgram.extensions.types.sockets import (
    AgentV1Agent,
    AgentV1AudioConfig,
    AgentV1AudioInput,
    AgentV1DeepgramSpeakProvider,
    AgentV1Listen,
    AgentV1ListenProvider,
    AgentV1OpenAiThinkProvider,
    AgentV1SettingsMessage,
    AgentV1SpeakProviderConfig,
    AgentV1Think,
)

from app.config import settings


class VoiceAgentService:
    """Factory for creating Deepgram Voice Agent connections."""

    def __init__(self, dg_settings: Any = settings) -> None:
        if not dg_settings.deepgram_api_key:
            raise ValueError(
                "Deepgram API key not configured. Set DEEPGRAM_API_KEY in environment variables."
            )

        self._client = AsyncDeepgramClient(api_key=dg_settings.deepgram_api_key)
        self._listen_model = dg_settings.deepgram_listen_model
        self._think_model = dg_settings.deepgram_think_model
        self._speak_model = dg_settings.deepgram_speak_model
        self._sample_rate = dg_settings.deepgram_sample_rate

    def create_connection(self):
        """Return an async context manager for a Deepgram Voice Agent connection."""
        return self._client.agent.v1.connect()

    def create_settings_message(self, prompt: Optional[str] = None) -> AgentV1SettingsMessage:
        """Build the agent settings payload."""
        think_prompt = prompt or "You are a helpful voice assistant."

        return AgentV1SettingsMessage(
            audio=AgentV1AudioConfig(
                input=AgentV1AudioInput(
                    encoding="linear16",
                    sample_rate=self._sample_rate,
                )
            ),
            agent=AgentV1Agent(
                listen=AgentV1Listen(
                    provider=AgentV1ListenProvider(
                        type="deepgram",
                        model=self._listen_model,
                        smart_format=True,
                    )
                ),
                think=AgentV1Think(
                    provider=AgentV1OpenAiThinkProvider(
                        type="open_ai",
                        model=self._think_model,
                        temperature=0.3,
                    ),
                    prompt=think_prompt,
                ),
                speak=AgentV1SpeakProviderConfig(
                    provider=AgentV1DeepgramSpeakProvider(
                        type="deepgram",
                        model=self._speak_model,
                    )
                ),
            ),
        )
