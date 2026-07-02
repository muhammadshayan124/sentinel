"""Thin LLM client seam. Call sites depend on this protocol rather than an SDK directly,
so a new backend is a self-contained addition here rather than a change scattered across
the agent loop, CLI, and API.

AnthropicClient is the default (cloud, requires ANTHROPIC_API_KEY). OllamaClient runs
fully offline against a local model server (https://ollama.com) -- no API key, no network
egress. It translates to/from Ollama's OpenAI-compatible endpoint internally so the agent
loop, which is written against Anthropic's message/content-block shape, works unmodified
against either backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"


@dataclass
class _OllamaMessage:
    content: list[_TextBlock | _ToolUseBlock]
    stop_reason: str


def _field(block: Any, key: str) -> Any:
    """Blocks in the agent loop's message history are a mix of plain dicts (tool_result,
    appended by the loop itself) and attribute-style objects (assistant content blocks,
    returned by a client's create_message). Read either shape uniformly.
    """
    return block[key] if isinstance(block, dict) else getattr(block, key)


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    openai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            openai_messages.append({"role": message["role"], "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            block_type = _field(block, "type")
            if block_type == "text":
                text_parts.append(_field(block, "text"))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": _field(block, "id"),
                        "type": "function",
                        "function": {
                            "name": _field(block, "name"),
                            "arguments": json.dumps(_field(block, "input")),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(block)

        if tool_calls:
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": " ".join(text_parts) or None,
                    "tool_calls": tool_calls,
                }
            )
        for result in tool_results:
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _field(result, "tool_use_id"),
                    "content": _field(result, "content"),
                }
            )
        if text_parts and not tool_calls:
            openai_messages.append({"role": message["role"], "content": " ".join(text_parts)})

    return openai_messages


def _to_anthropic_response(data: dict[str, Any]) -> _OllamaMessage:
    message = data["choices"][0]["message"]
    blocks: list[_TextBlock | _ToolUseBlock] = []

    if message.get("content"):
        blocks.append(_TextBlock(text=message["content"]))

    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        blocks.append(
            _ToolUseBlock(
                id=call["id"],
                name=call["function"]["name"],
                input=json.loads(call["function"]["arguments"]),
            )
        )

    return _OllamaMessage(content=blocks, stop_reason="tool_use" if tool_calls else "end_turn")


class OllamaClient:
    """Runs the agent against a local Ollama server -- no API key, works fully offline.

    Requires a tool-calling-capable local model (e.g. `ollama pull llama3.1`).
    """

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model

    def create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> _OllamaMessage:
        import httpx

        payload = {
            "model": self._model,
            "messages": _to_openai_messages(system, messages),
            "stream": False,
        }
        if tools:
            payload["tools"] = _to_openai_tools(tools)

        response = httpx.post(f"{self._base_url}/v1/chat/completions", json=payload, timeout=120)
        response.raise_for_status()
        return _to_anthropic_response(response.json())


def get_llm_client() -> LLMClient:
    """Backend selection point: SENTINEL_LLM_BACKEND=ollama for a fully offline agent,
    anything else (default) uses the Anthropic API.
    """
    if settings.llm_backend == "ollama":
        return OllamaClient()
    return AnthropicClient()
