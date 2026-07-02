import json

from sentinel.agent.llm import (
    OllamaClient,
    _to_anthropic_response,
    _to_openai_messages,
    _to_openai_tools,
)


def test_to_openai_tools_converts_anthropic_shape():
    tools = [
        {
            "name": "calculator",
            "description": "does math",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    openai_tools = _to_openai_tools(tools)
    assert openai_tools == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "does math",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_to_openai_messages_plain_text_turn():
    messages = [{"role": "user", "content": "hello"}]
    result = _to_openai_messages("be helpful", messages)
    assert result == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]


def test_to_openai_messages_handles_tool_use_and_tool_result_roundtrip():
    from sentinel.agent.llm import _ToolUseBlock

    messages = [
        {"role": "user", "content": "what's 2+2?"},
        {
            "role": "assistant",
            "content": [_ToolUseBlock(id="call_1", name="calculator", input={"expression": "2+2"})],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "4"}],
        },
    ]
    result = _to_openai_messages("system prompt", messages)

    assert result[0] == {"role": "system", "content": "system prompt"}
    assert result[1] == {"role": "user", "content": "what's 2+2?"}
    assert result[2]["role"] == "assistant"
    assert result[2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(result[2]["tool_calls"][0]["function"]["arguments"]) == {"expression": "2+2"}
    assert result[3] == {"role": "tool", "tool_call_id": "call_1", "content": "4"}


def test_to_anthropic_response_text_only():
    data = {"choices": [{"message": {"content": "the answer is 4", "tool_calls": None}}]}
    result = _to_anthropic_response(data)
    assert result.stop_reason == "end_turn"
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "the answer is 4"


def test_to_anthropic_response_with_tool_call():
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"expression": "2+2"}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    result = _to_anthropic_response(data)
    assert result.stop_reason == "tool_use"
    assert result.content[0].type == "tool_use"
    assert result.content[0].name == "calculator"
    assert result.content[0].input == {"expression": "2+2"}
    assert result.content[0].id == "call_1"


def test_ollama_client_create_message_calls_local_server(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "hi", "tool_calls": None}}]}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = OllamaClient(base_url="http://localhost:11434", model="llama3.1")
    result = client.create_message(
        messages=[{"role": "user", "content": "hi"}], tools=[], system="be helpful"
    )

    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["model"] == "llama3.1"
    assert "tools" not in captured["json"]
    assert result.content[0].text == "hi"
