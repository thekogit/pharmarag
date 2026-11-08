"""Thin OpenAI-compatible client for the generator model (Ollama Cloud beam:cloud by default)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from src.config import Settings


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLM:
    model: str
    client: Any
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_settings(cls, s: Settings) -> LLM:
        if not s.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and fill in the LLM settings.")
        return cls(s.llm_model, OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=180))

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False, max_tokens: int = 2048) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0,
                                  "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = self.client.chat.completions.create(**kwargs)
        self.usage.calls += 1
        if r.usage is not None:
            self.usage.prompt_tokens += r.usage.prompt_tokens or 0
            self.usage.completion_tokens += r.usage.completion_tokens or 0
        return str(r.choices[0].message.content or "")
