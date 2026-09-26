from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Literal

logger = logging.getLogger(__name__)

_SAFE_GROQ_ERROR_TYPES = frozenset({
    "APIConnectionError",
    "APIResponseValidationError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "UnprocessableEntityError",
})


class ProviderError(RuntimeError):
    pass


class MissingKeyError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


def _raise_groq_provider_error(exc: Exception, sdk: object, *, phase: str, started: float) -> None:
    api_error_type = getattr(sdk, "APIError", ())
    api_status_error_type = getattr(sdk, "APIStatusError", ())
    api_timeout_error_type = getattr(sdk, "APITimeoutError", ())
    api_connection_error_type = getattr(sdk, "APIConnectionError", ())
    rate_limit_error_type = getattr(sdk, "RateLimitError", ())
    authentication_error_type = getattr(sdk, "AuthenticationError", ())

    is_api_error = isinstance(api_error_type, type) and isinstance(exc, api_error_type)
    is_status_error = isinstance(api_status_error_type, type) and isinstance(exc, api_status_error_type)
    status = getattr(exc, "status_code", None) if is_api_error else None
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        status = None

    if isinstance(api_timeout_error_type, type) and isinstance(exc, api_timeout_error_type):
        category = "timeout"
    elif (isinstance(rate_limit_error_type, type) and isinstance(exc, rate_limit_error_type)) or status == 429:
        category = "rate_limit"
    elif (isinstance(authentication_error_type, type) and isinstance(exc, authentication_error_type)) or status in {401, 403}:
        category = "authentication"
    elif is_status_error or status is not None:
        category = "http_error"
    elif isinstance(api_connection_error_type, type) and isinstance(exc, api_connection_error_type):
        category = "connection"
    else:
        category = "request"

    candidate_class = type(exc).__name__
    exception_class = candidate_class if is_api_error and candidate_class in _SAFE_GROQ_ERROR_TYPES else "UnexpectedError"
    status_label = str(status) if status is not None else "none"
    elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
    logger.warning(
        "Groq provider phase=%s category=%s sdk_exception=%s status=%s elapsed_ms=%d",
        phase,
        category,
        exception_class,
        status_label,
        elapsed_ms,
    )

    if category == "timeout":
        raise ProviderTimeoutError("Groq request timed out") from exc
    if category == "rate_limit":
        raise ProviderRateLimitError("Groq rate limit exceeded") from exc
    if category == "authentication":
        raise MissingKeyError("Groq key missing or invalid") from exc
    raise ProviderError("Groq request failed") from exc


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[ChatMessage], *, stream: bool = False) -> AsyncIterator[str] | str:
        raise NotImplementedError


class FakeProvider(ChatProvider):
    def __init__(self, *, reply: str = "stubbed reply"):
        self.reply = reply

    async def chat(self, messages: list[ChatMessage], *, stream: bool = False) -> AsyncIterator[str] | str:
        if stream:
            async def gen() -> AsyncIterator[str]:
                for chunk in self.reply.split():
                    yield chunk + " "
            return gen()
        return self.reply


class GroqProvider(ChatProvider):
    def __init__(self, api_key: str, model: str, *, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None
        self._tool_client = None

    async def chat(self, messages: list[ChatMessage], *, stream: bool = False) -> AsyncIterator[str] | str:
        started = time.monotonic()
        try:
            import groq
        except ImportError as exc:
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
            logger.warning(
                "Groq provider phase=chat_completion category=dependency sdk_exception=ImportError status=none elapsed_ms=%d",
                elapsed_ms,
            )
            raise ProviderError("Groq client dependency not installed") from exc

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0.2,
            "stream": stream,
        }

        try:
            if self._client is None:
                self._client = groq.AsyncGroq(api_key=self.api_key, timeout=self.timeout)

            if stream:
                response = await self._client.chat.completions.create(**payload)

                async def iterator() -> AsyncIterator[str]:
                    async for chunk in response:
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            yield content

                return iterator()

            response = await self._client.chat.completions.create(**payload)
            return response.choices[0].message.content
        except Exception as exc:
            _raise_groq_provider_error(exc, groq, phase="chat_completion", started=started)

    async def chat_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        max_completion_tokens: int,
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> object:
        started = time.monotonic()
        try:
            import groq
        except ImportError as exc:
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
            logger.warning(
                "Groq provider phase=tool_completion category=dependency sdk_exception=ImportError status=none elapsed_ms=%d",
                elapsed_ms,
            )
            raise ProviderError("Groq client dependency not installed") from exc

        try:
            if self._tool_client is None:
                self._tool_client = groq.AsyncGroq(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=0,
                )
            response = await self._tool_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                temperature=0.2,
                max_completion_tokens=max_completion_tokens,
            )
            return response.choices[0].message
        except Exception as exc:
            _raise_groq_provider_error(exc, groq, phase="tool_completion", started=started)
