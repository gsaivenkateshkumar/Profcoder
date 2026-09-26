from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


class MissingKeyError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


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

    async def chat(self, messages: list[ChatMessage], *, stream: bool = False) -> AsyncIterator[str] | str:
        try:
            import groq
        except ImportError as exc:
            logger.warning("Groq provider error category=dependency")
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
            message = str(exc)
            lower = message.lower()
            if "timeout" in lower:
                logger.warning("Groq provider error category=timeout")
                raise ProviderTimeoutError("Groq request timed out") from exc
            if "rate limit" in lower or "429" in lower:
                logger.warning("Groq provider error category=rate_limit")
                raise ProviderRateLimitError("Groq rate limit exceeded") from exc
            if "missing" in lower or "api key" in lower or "unauthorized" in lower:
                logger.warning("Groq provider error category=authentication")
                raise MissingKeyError("Groq key missing or invalid") from exc
            logger.warning("Groq provider error category=request")
            raise ProviderError("Groq request failed") from exc
