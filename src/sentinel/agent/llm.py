"""Thin LLM client seam. Only Anthropic is implemented, but call sites depend on this
protocol rather than the SDK directly, so a second backend is a one-file addition.
"""

from __future__ import annotations

from typing import Any, Protocol

from sentinel.config import settings


class LLMClient(Protocol):
    def create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Any: ...


class AnthropicClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)
        self._model = model or settings.anthropic_model

    def create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Any:
        return self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools,
        )
