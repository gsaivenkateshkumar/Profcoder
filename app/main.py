from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = build_provider()
    project_root = os.getenv("REPO_ROOT", "").strip()
    try:
        app.state.project_root = Path(project_root).expanduser().resolve(strict=True) if project_root else None
    except (OSError, RuntimeError):
        app.state.project_root = None
    if app.state.project_root is not None and not app.state.project_root.is_dir():
        app.state.project_root = None
    yield


app = FastAPI(title="Profcoder API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
