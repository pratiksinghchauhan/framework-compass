"""Chat model access, kept behind one small interface.

The rest of the app only calls `get_chat_model().complete(messages)`, so the
model can be changed in one place. Answers come from Gemini, called through
its OpenAI-compatible endpoint.

    GEMINI_API_KEY   required, from Google AI Studio
    LLM_MODEL        optional Gemini model name (default: gemini-3.5-flash-lite)

The key is never stored in code or files in the repo. A local .env file
(see .env.example) is loaded too; variables already set in the shell take
priority.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[Message], temperature: float = 0.0,
                 max_tokens: int | None = None) -> str: ...


class GeminiChat:
    """A Gemini model served through Google's OpenAI-compatible chat completions API."""

    def __init__(self, model: str, api_key: str):
        from openai import OpenAI

        self.name = model
        self.client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    def complete(self, messages: list[Message], temperature: float = 0.0,
                 max_tokens: int | None = None) -> str:
        kwargs = {"model": self.name, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        reply = self.client.chat.completions.create(**kwargs)
        return reply.choices[0].message.content or ""


def get_chat_model(model: str | None = None) -> ChatModel:
    load_dotenv(ENV_FILE, override=False)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("set GEMINI_API_KEY in .env (see .env.example) or in your shell")
    return GeminiChat(model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL, api_key)
