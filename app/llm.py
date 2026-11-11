"""Chat model access, kept behind one small interface so the model is easy to swap.

The rest of the app only calls `get_chat_model().complete(messages)`. Which
model answers is set by environment variables:

    LLM_PROVIDER   a key of PROVIDERS (default: gemini)
    LLM_MODEL      model name (default: the provider's default_model)

The API key is read from the provider's environment variable (for Gemini,
GEMINI_API_KEY) and is never stored in code or files in the repo. A local
.env file (see .env.example) is loaded too; variables already set in the
shell take priority.

To add a provider with an OpenAI-compatible endpoint, add an entry to
PROVIDERS. For one with a different API, write a class with the same
`complete` method and register it in `get_chat_model`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[Message], temperature: float = 0.0,
                 max_tokens: int | None = None) -> str: ...


@dataclass(frozen=True)
class Provider:
    base_url: str | None
    api_key_env: str
    default_model: str


PROVIDERS = {
    "gemini": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-3.5-flash-lite",
    ),
}


class OpenAICompatibleChat:
    """Any chat model served through an OpenAI-compatible chat completions API."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.name = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[Message], temperature: float = 0.0,
                 max_tokens: int | None = None) -> str:
        kwargs = {"model": self.name, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        reply = self.client.chat.completions.create(**kwargs)
        return reply.choices[0].message.content or ""


def get_chat_model(provider: str | None = None, model: str | None = None) -> ChatModel:
    load_dotenv(ENV_FILE, override=False)
    provider = provider or os.environ.get("LLM_PROVIDER", "gemini")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; choose from {sorted(PROVIDERS)}")
    config = PROVIDERS[provider]
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"set {config.api_key_env} in .env (see .env.example) or in your shell to use {provider}")
    model = model or os.environ.get("LLM_MODEL") or config.default_model
    return OpenAICompatibleChat(model, api_key, config.base_url)
