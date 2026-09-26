from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.providers import FakeProvider, GroqProvider, ProviderError, ProviderRateLimitError, ProviderTimeoutError


class FakeClientProvider:
    def __init__(self, *, reply="hello world"):
        self.reply = reply

    async def chat(self, messages, *, stream=False):
        if stream:
            async def gen():
                for chunk in self.reply.split():
                    yield chunk + " "
            return gen()
        return self.reply


class TimeoutProvider(FakeClientProvider):
    async def chat(self, messages, *, stream=False):
        raise ProviderTimeoutError("Groq request timed out")


class FailProvider(FakeClientProvider):
    async def chat(self, messages, *, stream=False):
        raise ProviderError("Groq request failed")


class RateLimitProvider(FakeClientProvider):
    async def chat(self, messages, *, stream=False):
        raise ProviderRateLimitError("Groq rate limit exceeded")


@pytest.fixture
def client():
    app.state.provider = FakeClientProvider()
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_success(client):
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json()["message"] == "hello world"


def test_chat_stream_success(client):
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200
    assert response.text == "hello world "


def test_groq_chat_uses_async_sdk(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            if kwargs["stream"]:
                async def chunks():
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="stream hello"))]
                    )
                return chunks()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="async hello"))]
            )

    class FakeAsyncGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(AsyncGroq=FakeAsyncGroq))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    app.state.provider = GroqProvider(api_key="test-key", model="test-model")

    with TestClient(app) as test_client:
        response = test_client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        stream_response = test_client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert response.status_code == 200
    assert response.json() == {"message": "async hello"}
    assert stream_response.status_code == 200
    assert stream_response.text == "stream hello"


def test_groq_error_is_safe_and_logged_by_category(monkeypatch, caplog):
    class FakeCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("api_key=secret header=private body=private")

    class FakeAsyncGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(AsyncGroq=FakeAsyncGroq))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    app.state.provider = GroqProvider(api_key="test-key", model="test-model")

    with caplog.at_level("WARNING", logger="app.providers"), TestClient(app) as test_client:
        response = test_client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 503
    assert response.json() == {"detail": "Groq request failed"}
    assert "category=request" in caplog.text
    assert "secret" not in response.text + caplog.text
    assert "private" not in response.text + caplog.text


def test_missing_key_behavior(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    app.state.provider = FakeProvider(reply="Provider not configured. Set GROQ_API_KEY to enable online inference.")
    with TestClient(app) as client:
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200
        assert "Set GROQ_API_KEY" in response.json()["message"]


def test_provider_timeout_error():
    with TestClient(app) as client:
        app.state.provider = TimeoutProvider()
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_provider_error():
    with TestClient(app) as client:
        app.state.provider = FailProvider()
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 503
        assert "failed" in response.json()["detail"].lower()


def test_rate_limit_error():
    with TestClient(app) as client:
        app.state.provider = RateLimitProvider()
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()
