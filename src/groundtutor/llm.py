"""Chat providers. All three speak HTTP so tests can substitute an httpx client."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Protocol

from groundtutor.errors import ConfigurationError, ProviderError

USER_AGENT = "groundtutor/0.1.0"
Message = Mapping[str, str]


class LLM(Protocol):
    """Minimal chat interface used for answers, judging, and study items."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant text for ``messages``."""


class OpenAIProvider:
    """OpenAI Chat Completions. Reads ``OPENAI_API_KEY`` when ``api_key`` is omitted."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client=None,
        timeout: float = 60.0,
    ):
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.timeout = timeout

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = self._post(f"{self.base_url}/chat/completions", payload, self._headers())
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI returned an unexpected payload") from exc
        if not isinstance(content, str):
            raise ProviderError("OpenAI returned an empty message")
        return content

    def _headers(self) -> dict[str, str]:
        key = self._api_key if self._api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigurationError("Set OPENAI_API_KEY or pass api_key.")
        return {"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}

    def _post(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        return _post_json(self._http(), url, payload, headers, "OpenAI")

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client


class AnthropicProvider:
    """Anthropic Messages API. Reads ``ANTHROPIC_API_KEY`` when ``api_key`` is omitted."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        client=None,
        timeout: float = 60.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        del json_mode  # Anthropic follows the JSON instruction in the prompt.
        system, chat = _split_system(messages)
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": chat,
        }
        if system:
            payload["system"] = system
        body = _post_json(
            self._http(), f"{self.base_url}/v1/messages", payload, self._headers(), "Anthropic"
        )
        try:
            blocks = body["content"]
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise ProviderError("Anthropic returned an unexpected payload") from exc
        if not text:
            raise ProviderError("Anthropic returned an empty message")
        return text

    def _headers(self) -> dict[str, str]:
        key = self._api_key if self._api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError("Set ANTHROPIC_API_KEY or pass api_key.")
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "User-Agent": USER_AGENT,
        }

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client


class OllamaProvider:
    """Ollama chat API. Uses ``OLLAMA_HOST`` or ``http://127.0.0.1:11434``."""

    def __init__(
        self,
        model: str = "llama3.2",
        *,
        base_url: str | None = None,
        client=None,
        timeout: float = 120.0,
    ):
        host = base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self.model = model
        self.base_url = host.rstrip("/")
        self._client = client
        self.timeout = timeout

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        body = _post_json(
            self._http(), f"{self.base_url}/api/chat", payload, {"User-Agent": USER_AGENT}, "Ollama"
        )
        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError("Ollama returned an unexpected payload") from exc
        if not isinstance(content, str):
            raise ProviderError("Ollama returned an empty message")
        return content

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client


def resolve_llm(spec: str) -> LLM:
    """Build a provider from ``openai:<model>``, ``anthropic:<model>``, or ``ollama:<model>``."""
    name, _, model = spec.partition(":")
    key = name.strip().lower()
    model = model.strip()
    if key == "openai":
        return OpenAIProvider(model or "gpt-4o-mini")
    if key == "anthropic":
        return AnthropicProvider(model or "claude-3-5-haiku-latest")
    if key == "ollama":
        return OllamaProvider(model or "llama3.2")
    raise ConfigurationError(
        f"Unknown llm '{spec}'. Expected 'openai:<model>', 'anthropic:<model>', or 'ollama:<model>'."
    )


def _split_system(messages: Sequence[Message]) -> tuple[str, list[dict[str, str]]]:
    system: list[str] = []
    chat: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system.append(content)
        else:
            chat.append({"role": role, "content": content})
    if not chat:
        raise ProviderError("Anthropic requires at least one user message")
    return "\n\n".join(system), chat


def _post_json(client, url: str, payload: dict, headers: dict[str, str], name: str) -> dict:
    import httpx

    try:
        response = client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise ProviderError(f"{name} request failed: {exc}") from exc
    if response.status_code >= 400:
        raise ProviderError(f"{name} returned HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(f"{name} returned a body that is not JSON") from exc
