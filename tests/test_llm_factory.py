import dataclasses

import sentinel.agent.llm as llm_module
from sentinel.agent.llm import AnthropicClient, OllamaClient, get_llm_client


def test_get_llm_client_defaults_to_anthropic(monkeypatch):
    patched = dataclasses.replace(
        llm_module.settings, llm_backend="anthropic", anthropic_api_key="sk-test"
    )
    monkeypatch.setattr(llm_module, "settings", patched)
    assert isinstance(get_llm_client(), AnthropicClient)


def test_get_llm_client_selects_ollama(monkeypatch):
    patched = dataclasses.replace(llm_module.settings, llm_backend="ollama")
    monkeypatch.setattr(llm_module, "settings", patched)
    assert isinstance(get_llm_client(), OllamaClient)
