"""OllamaClient unit tests (httpx mocked, no real Ollama needed)."""

from __future__ import annotations

import logging

import httpx
import pytest

from src.ai.llm_client import OllamaClient
from src.config import LLMConfig


def _cfg(**overrides: object) -> LLMConfig:
    defaults = {"enabled": True, "base_url": "http://localhost:11434", "model": "llama3.2"}
    return LLMConfig(**{**defaults, **overrides})


def test_generate_returns_response_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient(_cfg())

    def _mock_post(self, url, **kw):
        resp = httpx.Response(200, json={"response": "LLM answer here"})
        resp._request = httpx.Request("POST", url)
        return resp

    monkeypatch.setattr(httpx.Client, "post", _mock_post)
    result = client.generate("What is SPY?")
    assert result == "LLM answer here"


def test_generate_returns_none_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient(_cfg())

    def _raise(self, url, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "post", _raise)
    assert client.generate("test") is None


def test_generate_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient(_cfg())

    def _raise(self, url, **kw):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx.Client, "post", _raise)
    assert client.generate("test") is None


def test_is_available_true_when_ollama_responds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient(_cfg())

    def _mock_get(self, url, **kw):
        resp = httpx.Response(200, json={"models": []})
        resp._request = httpx.Request("GET", url)
        return resp

    monkeypatch.setattr(httpx.Client, "get", _mock_get)
    assert client.is_available() is True


def test_is_available_false_when_connection_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient(_cfg())

    def _raise(self, url, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "get", _raise)
    assert client.is_available() is False


def test_generate_returns_none_when_disabled() -> None:
    client = OllamaClient(_cfg(enabled=False))
    assert client.generate("test prompt") is None


def test_is_available_false_when_disabled() -> None:
    client = OllamaClient(_cfg(enabled=False))
    assert client.is_available() is False


def test_generate_logs_warning_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = OllamaClient(_cfg())

    def _raise(self, url, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.Client, "post", _raise)
    caplog.set_level(logging.WARNING)
    client.generate("hello")
    assert any("Ollama generate failed" in r.message for r in caplog.records)


def test_embed_returns_vector_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient(_cfg())

    def _mock_post(self, url, **kw):
        resp = httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
        resp._request = httpx.Request("POST", url)
        return resp

    monkeypatch.setattr(httpx.Client, "post", _mock_post)
    vec = client.embed("hello")
    assert vec is not None
    assert len(vec) == 3
    assert vec[0] == pytest.approx(0.1)


def test_embed_returns_none_when_disabled() -> None:
    client = OllamaClient(_cfg(enabled=False))
    assert client.embed("x") is None
