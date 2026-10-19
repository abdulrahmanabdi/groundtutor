import json

import httpx
import pytest

from groundtutor.embeddings import OpenAIEmbedder
from groundtutor.errors import ConfigurationError, ProviderError
from groundtutor.llm import AnthropicProvider, OllamaProvider, OpenAIProvider, resolve_llm


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_chat_and_embeddings_parse_responses():
    def chat(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["response_format"] == {"type": "json_object"}
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    provider = OpenAIProvider("gpt-4o-mini", api_key="test-key", client=_client(chat))
    assert provider.complete([{"role": "user", "content": "hi"}], json_mode=True) == '{"ok": true}'

    def embed(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == ["one", "two"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ]
            },
        )

    embedder = OpenAIEmbedder("custom-model", api_key="test-key", client=_client(embed))
    vectors = embedder.embed_documents(["one", "two"])
    assert vectors.shape == (2, 2)
    assert vectors[0, 0] == pytest.approx(1.0)
    assert embedder.dim == 2


def test_anthropic_splits_the_system_prompt():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["system"] == "Be brief"
        assert body["messages"] == [{"role": "user", "content": "Hi"}]
        assert request.headers["x-api-key"] == "anthropic-key"
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " there"}]
            },
        )

    provider = AnthropicProvider("claude-test", api_key="anthropic-key", client=_client(handler))
    text = provider.complete(
        [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}]
    )
    assert text == "Hello there"


def test_ollama_requests_json_format():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "llama3.2"
        assert body["format"] == "json"
        assert body["stream"] is False
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"a": 1}'}})

    provider = OllamaProvider("llama3.2", base_url="http://ollama.test", client=_client(handler))
    assert provider.complete([{"role": "user", "content": "Hi"}], json_mode=True) == '{"a": 1}'


def test_missing_keys_and_http_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        OpenAIProvider("gpt-4o-mini").complete([{"role": "user", "content": "hi"}])
    with pytest.raises(ConfigurationError):
        AnthropicProvider("claude").complete([{"role": "user", "content": "hi"}])

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="nope")

    provider = OpenAIProvider("gpt-4o-mini", api_key="x", client=_client(broken))
    with pytest.raises(ProviderError, match="401"):
        provider.complete([{"role": "user", "content": "hi"}])


def test_provider_specs():
    assert isinstance(resolve_llm("openai:gpt-4o-mini"), OpenAIProvider)
    assert isinstance(resolve_llm("anthropic"), AnthropicProvider)
    assert isinstance(resolve_llm("ollama:llama3.2"), OllamaProvider)
    with pytest.raises(ConfigurationError):
        resolve_llm("mystery:model")
