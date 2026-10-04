from __future__ import annotations

import asyncio
import os
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from app.agent_inspect import (
    MAX_AGENT_REQUEST_BYTES,
    MAX_INSPECT_SECONDS,
    AgentInspectError,
    AgentInspectRequest,
    inspect_project_question,
)
from app.providers import (
    ChatMessage,
    ChatProvider,
    FakeProvider,
    GroqProvider,
    MissingKeyError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.project_files import (
    MAX_PATH_CHARS,
    MAX_QUERY_CHARS,
    ProjectPathError,
    list_project_files,
    search_project_files,
)
from app.file_preview import MAX_PREVIEW_LINES, FileEditError, preview_project_file
from app.symbol_search import find_definitions
from app.syntax_check import check_python_syntax
from app.test_runner import GitChangeReader


load_dotenv(Path(__file__).resolve().parent.parent / '.env')

MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 20000
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(..., min_length=1)
    stream: bool = False


def build_provider() -> ChatProvider:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    if not api_key:
        return FakeProvider(reply="Provider not configured. Set GROQ_API_KEY to enable online inference.")
    return GroqProvider(api_key=api_key, model=model)


def build_git_reader() -> GitChangeReader | None:
    executable = os.getenv("PROFCODER_GIT_EXECUTABLE", "").strip()
    if not executable:
        return None
    path = Path(executable).expanduser()
    if not path.is_absolute():
        return None
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not path.is_file():
        return None
    try:
        return GitChangeReader(path, timeout_seconds=5.0, output_limit_bytes=4096)
    except ValueError:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = build_provider()
    app.state.git_reader = build_git_reader()
    project_root = os.getenv("REPO_ROOT", "").strip()
    try:
        app.state.project_root = Path(project_root).expanduser().resolve(strict=True) if project_root else None
    except (OSError, RuntimeError):
        app.state.project_root = None
    if app.state.project_root is not None and not app.state.project_root.is_dir():
        app.state.project_root = None
    yield


app = FastAPI(title="Profcoder API", lifespan=lifespan)
logger = logging.getLogger(__name__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/project/browser")
def project_browser() -> FileResponse:
    """Serve the read-only, offline project-navigation page. No key or provider call."""
    return FileResponse(STATIC_DIR / "project_browser.html", media_type="text/html")


def require_project_root() -> Path:
    project_root = getattr(app.state, "project_root", None)
    if project_root is None:
        raise HTTPException(status_code=503, detail="Project root is not configured")
    return project_root


@app.get("/project/files")
def project_files(path: str = Query(default=".", max_length=MAX_PATH_CHARS)):
    try:
        return list_project_files(require_project_root(), path)
    except ProjectPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/project/search")
def project_search(
    q: str = Query(min_length=1, max_length=MAX_QUERY_CHARS),
    path: str = Query(default=".", max_length=MAX_PATH_CHARS),
):
    try:
        return search_project_files(require_project_root(), q, path)
    except ProjectPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/project/definitions")
def project_definitions(
    name: str = Query(min_length=1, max_length=MAX_QUERY_CHARS),
    path: str = Query(default=".", max_length=MAX_PATH_CHARS),
):
    try:
        return find_definitions(require_project_root(), name, path)
    except ProjectPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/project/preview")
def project_preview(
    path: str = Query(min_length=1, max_length=MAX_PATH_CHARS),
    start: int = Query(default=1, ge=1),
    lines: int = Query(default=120, ge=1, le=MAX_PREVIEW_LINES),
):
    try:
        return preview_project_file(require_project_root(), path, start, lines)
    except (ProjectPathError, FileEditError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/project/syntax-check")
def project_syntax_check(path: str = Query(min_length=1, max_length=MAX_PATH_CHARS)):
    try:
        return check_python_syntax(require_project_root(), path)
    except (ProjectPathError, FileEditError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _read_bounded_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not content_length.isdecimal():
            raise HTTPException(status_code=400, detail="Invalid request size")
        if int(content_length) > MAX_AGENT_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Request is too large")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_AGENT_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Request is too large")
        body.extend(chunk)
    return bytes(body)


async def _run_inspect_request(request: Request, provider: object, project_root: Path) -> str:
    body = await _read_bounded_body(request)
    try:
        payload = json.loads(body)
        inspect_request = AgentInspectRequest.model_validate(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValidationError):
        raise HTTPException(status_code=422, detail="Invalid inspect request")

    return await inspect_project_question(
        provider,
        project_root,
        inspect_request.question,
        git_reader=getattr(app.state, "git_reader", None),
    )


@app.post("/agent/inspect")
async def agent_inspect(request: Request):
    project_root = getattr(app.state, "project_root", None)
    if project_root is None:
        raise HTTPException(status_code=503, detail="Project root is not configured")
    provider = getattr(app.state, "provider", None)
    if not callable(getattr(provider, "chat_with_tools", None)):
        raise HTTPException(status_code=503, detail="Groq tool calling is not configured")

    inspection_started = time.monotonic()
    try:
        answer = await asyncio.wait_for(
            _run_inspect_request(request, provider, project_root),
            timeout=MAX_INSPECT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        elapsed_ms = max(0, round((time.monotonic() - inspection_started) * 1000))
        logger.warning(
            "agent inspect phase=request category=timeout sdk_exception=TimeoutError status=504 elapsed_ms=%d",
            elapsed_ms,
        )
        raise HTTPException(status_code=504, detail="Inspection timed out") from exc
    except asyncio.CancelledError:
        elapsed_ms = max(0, round((time.monotonic() - inspection_started) * 1000))
        logger.warning(
            "agent inspect phase=request category=cancelled sdk_exception=CancelledError status=503 elapsed_ms=%d",
            elapsed_ms,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Inspection cancelled"},
        )
    except HTTPException:
        raise
    except ProviderTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Groq request timed out") from exc
    except ProviderRateLimitError as exc:
        raise HTTPException(status_code=429, detail="Groq rate limit exceeded") from exc
    except MissingKeyError as exc:
        raise HTTPException(status_code=503, detail="Groq key missing or invalid") from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail="Groq request failed") from exc
    except AgentInspectError as exc:
        raise HTTPException(status_code=502, detail="Agent inspection could not be completed") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="internal server error") from exc
    return JSONResponse(content={"answer": answer})


@app.post("/chat")
async def chat(req: ChatRequest):
    if len(req.messages) > MAX_MESSAGES:
        raise HTTPException(status_code=400, detail="Too many messages")

    try:
        normalized = []
        for item in req.messages:
            role = str(item["role"]).strip()
            content = str(item["content"])
            if len(content) > MAX_MESSAGE_CHARS:
                raise HTTPException(status_code=400, detail="Message too large")
            normalized.append(ChatMessage(role=role, content=content))

        provider = app.state.provider
        result = await provider.chat(normalized, stream=req.stream)
    except MissingKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ProviderRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="internal server error")

    if req.stream:
        async def stream_text() -> AsyncIterator[bytes]:
            if hasattr(result, "__aiter__"):
                async for token in result:
                    yield token.encode("utf-8")
            else:
                yield str(result).encode("utf-8")
        return StreamingResponse(stream_text(), media_type="text/plain")

    return {"message": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
