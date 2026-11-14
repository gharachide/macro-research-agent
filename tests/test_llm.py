import io
import json
import urllib.error

import pytest

from research_agent import llm
from research_agent.llm import (
    LLMAborted,
    LLMError,
    LLMTimeout,
    OpenAICompatClient,
    ToolsUnsupported,
    client_from_args,
    parse_completion,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, body, headers=None):
    return urllib.error.HTTPError("http://x", code, "err", headers or {}, io.BytesIO(body.encode()))


def payload(message, tokens=42):
    return {"choices": [{"message": message}], "usage": {"total_tokens": tokens}}


def test_parse_completion_with_tool_calls():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "get_value", "arguments": '{"series_id": "br.selic_target"}'},
            },
            {"function": {"name": "get_value", "arguments": "{not json"}},
        ],
    }
    c = parse_completion(payload(message))
    assert c.content == "" and c.tokens == 42 and c.message is message
    assert (c.tool_calls[0].id, c.tool_calls[0].name, c.tool_calls[0].arguments) == (
        "a",
        "get_value",
        {"series_id": "br.selic_target"},
    )
    assert c.tool_calls[1].id == "call_1" and c.tool_calls[1].arguments == "{not json"
    with pytest.raises(LLMError):
        parse_completion({"error": "nope"})


def test_request_shape(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(
            url=request.full_url,
            body=json.loads(request.data),
            auth=request.get_header("Authorization"),
            timeout=timeout,
        )
        return FakeResponse(json.dumps(payload({"role": "assistant", "content": "hi"})).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatClient("http://localhost:11434/v1/", "gemma", "secret", timeout=9)
    assert client.complete([{"role": "user", "content": "q"}], [{"type": "function"}]).content == "hi"
    assert (
        sent["url"] == "http://localhost:11434/v1/chat/completions"
        and sent["auth"] == "Bearer secret"
        and sent["timeout"] == 9
    )
    assert sent["body"] == {
        "model": "gemma",
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0,
        "tools": [{"type": "function"}],
    }
    client.complete([{"role": "user", "content": "q"}], None)
    assert "tools" not in sent["body"]


def test_rate_limit_is_retried(monkeypatch):
    waits = []
    replies = [http_error(429, "slow down", {"Retry-After": "2"}), http_error(429, "slow down"), None]

    def fake_urlopen(request, timeout):
        reply = replies.pop(0)
        if reply:
            raise reply
        return FakeResponse(json.dumps(payload({"content": "ok"})).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    assert OpenAICompatClient("http://x", "m").complete([], None).content == "ok"
    assert waits == [2.0, 10]


def test_errors(monkeypatch):
    def raising(error):
        def fake_urlopen(request, timeout):
            raise error

        return fake_urlopen

    client = OpenAICompatClient("http://x", "m", max_retries=0)
    monkeypatch.setattr(
        llm.urllib.request, "urlopen", raising(http_error(400, '{"error": "model does not support tools"}'))
    )
    with pytest.raises(ToolsUnsupported):
        client.complete([], [{"type": "function"}])
    with pytest.raises(
        LLMError
    ) as plain:  # the same error without tools in the request is not a tools problem
        client.complete([], None)
    assert not isinstance(plain.value, ToolsUnsupported)
    monkeypatch.setattr(llm.urllib.request, "urlopen", raising(http_error(429, "slow down")))
    with pytest.raises(LLMError, match="429"):
        client.complete([], None)
    monkeypatch.setattr(llm.urllib.request, "urlopen", raising(urllib.error.URLError("connection refused")))
    with pytest.raises(LLMError, match="connection refused") as refused:
        client.complete([], None)
    assert not isinstance(refused.value, LLMTimeout)
    for slow in (TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))):
        monkeypatch.setattr(llm.urllib.request, "urlopen", raising(slow))
        with pytest.raises(LLMTimeout):
            client.complete([], None)


def test_client_from_args(monkeypatch):
    monkeypatch.delenv("RA_TEST_KEY", raising=False)
    assert client_from_args("http://x", "m", None).api_key is None
    with pytest.raises(LLMError, match="RA_TEST_KEY"):
        client_from_args("http://x", "m", "RA_TEST_KEY")
    monkeypatch.setenv("RA_TEST_KEY", "k")
    assert client_from_args("http://x", "m", "RA_TEST_KEY").api_key == "k"


def test_embedder_batches_and_orders(monkeypatch):
    from research_agent.llm import OpenAICompatEmbedder

    sent = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        sent.append((request.full_url, body))
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return FakeResponse(json.dumps({"data": list(reversed(data))}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    vectors = OpenAICompatEmbedder("http://localhost:11434/v1", "embeddinggemma", batch=2).embed(
        ["a", "bb", "ccc"]
    )
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]  # back in input order
    assert [len(b["input"]) for _, b in sent] == [2, 1] and sent[0][0].endswith("/v1/embeddings")
    assert sent[0][1]["model"] == "embeddinggemma"

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda request, timeout: FakeResponse(b'{"data": []}'))
    with pytest.raises(LLMError, match="asked for"):
        OpenAICompatEmbedder("http://x", "m").embed(["a"])


def test_a_looping_generation_is_reported_as_aborted(monkeypatch):
    body = '{"error":{"message":"prediction aborted, token repeat limit reached","type":"api_error"}}'

    def fake_urlopen(request, timeout):
        raise http_error(500, body)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMAborted):
        OpenAICompatClient("http://x", "m", max_retries=0).complete([], None)
    monkeypatch.setattr(
        llm.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(http_error(500, "out of memory")),
    )
    with pytest.raises(LLMError) as other:
        OpenAICompatClient("http://x", "m", max_retries=0).complete([], None)
    assert not isinstance(other.value, LLMAborted)  # any other server error still stops the run


def test_max_tokens_is_sent_and_a_cut_reply_is_flagged(monkeypatch):
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        body = {"choices": [{"message": {"content": "and and"}, "finish_reason": "length"}], "usage": {}}
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    capped = OpenAICompatClient("http://x/v1", "m", max_tokens=1024).complete([], None)
    OpenAICompatClient("http://x/v1", "m").complete([], None)
    assert sent[0]["max_tokens"] == 1024 and "max_tokens" not in sent[1]
    assert capped.finish_reason == "length"
