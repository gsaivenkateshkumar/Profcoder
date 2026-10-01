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


class ExplodingProvider:
    """Any provider call during an offline route is a bug; fail loudly instead."""

    async def chat(self, messages, *, stream=False):
        raise AssertionError("provider must not be called by an offline project route")

    async def chat_with_tools(self, messages, tools, *, max_completion_tokens, tool_choice="auto"):
        raise AssertionError("provider must not be called by an offline project route")


def test_project_definitions_returns_qualified_definition(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "class Box:\n    def volume(self, w, h, d):\n        return w * h * d\n",
        encoding="utf-8",
    )
    with TestClient(app) as client:
        app.state.provider = ExplodingProvider()
        response = client.get("/project/definitions", params={"name": "Box.volume"})
    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] is False
    assert body["unparsed_files"] == 0
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["path"] == "pkg/mod.py"
    assert result["kind"] == "method"
    assert result["qualified_name"] == "Box.volume"


def test_project_definitions_works_without_groq_key_and_never_calls_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "mod.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    with TestClient(app) as client:
        app.state.provider = ExplodingProvider()  # would raise if the route ever called it
        response = client.get("/project/definitions", params={"name": "greet"})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_project_definitions_requires_configured_project_root(monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    with TestClient(app) as client:
        response = client.get("/project/definitions", params={"name": "anything"})
    assert response.status_code == 503


def test_project_definitions_rejects_invalid_name(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/project/definitions", params={"name": "1not_an_identifier"})
    assert response.status_code == 400


def test_project_definitions_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/project/definitions", params={"name": "greet", "path": "../outside"})
    assert response.status_code == 400


def test_project_definitions_skips_excluded_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    excluded = tmp_path / "node_modules" / "pkg"
    excluded.mkdir(parents=True)
    (excluded / "evil.py").write_text("def hidden():\n    return 1\n", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/definitions", params={"name": "hidden"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_project_browser_serves_the_page(monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    with TestClient(app) as client:
        response = client.get("/project/browser")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<title>Project browser</title>" in body
    assert "/project/files" in body
    assert "/project/search" in body
    assert "/project/definitions" in body


def test_project_browser_works_without_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("REPO_ROOT", raising=False)
    with TestClient(app) as client:
        app.state.provider = ExplodingProvider()
        response = client.get("/project/browser")
    assert response.status_code == 200


def test_project_browser_page_never_uses_unsafe_dom_sinks():
    page = (Path(__file__).resolve().parents[1] / "app" / "static" / "project_browser.html").read_text(
        encoding="utf-8"
    )
    for unsafe in ("innerHTML", "insertAdjacentHTML", "document.write(", " eval("):
        assert unsafe not in page
    assert "textContent" in page


def test_search_hostile_filename_and_excerpt_stay_plain_text(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    hostile_name = "a&b'c.py"
    hostile_line = "# <script>alert(1)</script> & \"quoted\" 'text'"
    (tmp_path / hostile_name).write_text(hostile_line + "\n", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/search", params={"q": "alert"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["path"] == hostile_name
    # The API returns the raw text verbatim in a JSON string field; no HTML
    # escaping happens server-side. Safety relies on the page always writing
    # such fields with .textContent, which test_project_browser_page_never_
    # uses_unsafe_dom_sinks (above) guards, and which this exact value exercises.
    assert "<script>" in body["results"][0]["excerpt"]


def test_project_preview_returns_numbered_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "mod.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "mod.py"})
    assert response.status_code == 200
    body = response.json()
    assert body["path"] == "mod.py"
    assert body["start_line"] == 1
    assert body["end_line"] == 2
    assert body["total_lines"] == 2
    assert body["truncated"] is False
    assert body["lines"] == ["def greet():", "    return 'hi'"]


def test_project_preview_start_line_selects_a_window(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "mod.py").write_text("".join(f"line {i}\n" for i in range(1, 11)), encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "mod.py", "start": 5, "lines": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["start_line"] == 5
    assert body["end_line"] == 7
    assert body["total_lines"] == 10
    assert body["truncated"] is True
    assert body["lines"] == ["line 5", "line 6", "line 7"]


def test_project_preview_start_beyond_end_of_file_returns_no_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "mod.py").write_text("one\ntwo\n", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "mod.py", "start": 100})
    assert response.status_code == 200
    body = response.json()
    assert body["lines"] == []
    assert body["total_lines"] == 2


def test_project_preview_caps_line_count_and_reports_truncation(tmp_path, monkeypatch):
    from app.file_preview import MAX_PREVIEW_LINES

    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "big.py").write_text("".join(f"x = {i}\n" for i in range(MAX_PREVIEW_LINES + 100)), encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "big.py", "lines": MAX_PREVIEW_LINES})
    assert response.status_code == 200
    body = response.json()
    assert len(body["lines"]) == MAX_PREVIEW_LINES
    assert body["truncated"] is True
    # The route itself rejects a request for more than the cap outright.
    with TestClient(app) as client:
        over_cap = client.get("/project/preview", params={"path": "big.py", "lines": MAX_PREVIEW_LINES + 1})
    assert over_cap.status_code == 422


def test_project_preview_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "../outside.py"})
    assert response.status_code == 400


def test_project_preview_rejects_excluded_file(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "image.png"})
    assert response.status_code == 400


def test_project_preview_rejects_oversized_file(tmp_path, monkeypatch):
    from app.file_edits import MAX_FILE_BYTES

    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "huge.py").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "huge.py"})
    assert response.status_code == 400


def test_project_preview_rejects_hard_linked_file(tmp_path, monkeypatch):
    import os as _os

    monkeypatch.setenv("REPO_ROOT", str(tmp_path / "root"))
    root = tmp_path / "root"
    root.mkdir()
    target = root / "shared.py"
    target.write_text("shared = 1\n", encoding="utf-8")
    try:
        _os.link(target, tmp_path / "outside-link.py")
    except OSError:
        pytest.skip("hard links unavailable on this filesystem")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "shared.py"})
    assert response.status_code == 400


def test_project_definitions_excludes_private_model_lab_data(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "model_lab" / "runs").mkdir(parents=True)
    (tmp_path / "model_lab" / "runs" / "mod.py").write_text("def greet():\n    return 1\n", encoding="utf-8")
    with TestClient(app) as client:
        direct = client.get("/project/definitions", params={"name": "greet", "path": "model_lab/runs"})
        nested = client.get("/project/definitions", params={"name": "greet", "path": "model_lab"})
    assert direct.status_code == 400
    assert nested.status_code == 200
    assert nested.json()["results"] == []


def test_project_preview_excludes_private_model_lab_data(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    (tmp_path / "model_lab" / "data" / "local").mkdir(parents=True)
    (tmp_path / "model_lab" / "data" / "local" / "secret.txt").write_text("SENTINEL", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "model_lab/data/local/secret.txt"})
    assert response.status_code == 400


def test_project_preview_requires_configured_project_root(monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "mod.py"})
    assert response.status_code == 503


def test_project_preview_hostile_source_text_stays_plain_text(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    hostile = "# <script>alert(1)</script> & \"quoted\" 'text'\n"
    (tmp_path / "mod.py").write_text(hostile, encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/preview", params={"path": "mod.py"})
    assert response.status_code == 200
    body = response.json()
    # Returned verbatim as a plain JSON string; the page must render it with
    # .textContent (guarded by test_project_browser_page_never_uses_unsafe_dom_sinks).
    assert body["lines"][0] == hostile.rstrip("\n")


def test_project_definitions_bounded_results_are_truncated(tmp_path, monkeypatch):
    from app.project_files import MAX_SEARCH_RESULTS

    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    for index in range(MAX_SEARCH_RESULTS + 5):
        (tmp_path / f"mod_{index}.py").write_text("def shared():\n    return 1\n", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/project/definitions", params={"name": "shared"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == MAX_SEARCH_RESULTS
    assert body["truncated"] is True
