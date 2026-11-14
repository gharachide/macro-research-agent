"""A minimal client for OpenAI-compatible chat completion endpoints.

Ollama (http://localhost:11434/v1), Groq and others expose this interface, so
one client covers a local model and the hosted ones. Standard library only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMError(RuntimeError):
    """The endpoint could not be reached or answered with an error."""


class LLMTimeout(LLMError):
    """The endpoint was reached but did not answer in time. The question failed; the endpoint is fine."""


class LLMAborted(LLMError):
    """The server stopped this generation (Ollama: "token repeat limit reached" when the model loops).
    The question failed; the endpoint is fine."""


ABORT_MARKERS = ("repeat limit", "prediction aborted")


class ToolsUnsupported(LLMError):
    """The model or server rejected the `tools` field; use the JSON protocol instead."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: object  # a dict when the model produced valid JSON


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: int = 0
    message: dict = field(default_factory=dict)  # the assistant message, to append to the conversation
    finish_reason: str = ""  # "length" when the reply was cut at the token cap


class ChatClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None) -> Completion: ...


def parse_completion(payload: dict) -> Completion:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"unexpected response: {json.dumps(payload)[:300]}") from None
    calls = []
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            args = raw  # left as a string: the tool layer reports it to the model
        calls.append(ToolCall(str(call.get("id") or f"call_{i}"), str(fn.get("name", "")), args))
    usage = payload.get("usage") or {}
    return Completion(
        message.get("content") or "",
        calls,
        int(usage.get("total_tokens") or 0),
        message,
        str(choice.get("finish_reason") or ""),
    )


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 300,
        max_retries: int = 5,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens  # cap on each reply; None leaves it to the server

    def complete(self, messages: list[dict], tools: list[dict] | None) -> Completion:
        body: dict = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, json.dumps(body).encode("utf-8"), headers, method="POST")
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return parse_completion(json.loads(response.read().decode("utf-8")))
            except urllib.error.HTTPError as err:
                detail = err.read().decode("utf-8", "replace")[:500]
                if err.code == 429 and attempt < self.max_retries:
                    # hosted free tiers: wait as asked, or back off 5 s, 10 s, 20 s ...
                    retry_after = err.headers.get("Retry-After", "") if err.headers else ""
                    time.sleep(
                        float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 5 * 2**attempt
                    )
                    continue
                if err.code == 500 and any(marker in detail.lower() for marker in ABORT_MARKERS):
                    raise LLMAborted(f"generation aborted by the server: {detail}") from None
                if tools and err.code in (400, 404, 422) and "tool" in detail.lower():
                    raise ToolsUnsupported(f"HTTP {err.code}: {detail}") from None
                raise LLMError(f"HTTP {err.code} from {self.url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                if isinstance(err, TimeoutError) or isinstance(getattr(err, "reason", None), TimeoutError):
                    raise LLMTimeout(f"no answer from {self.url} in {self.timeout:.0f} s") from None
                raise LLMError(f"could not get a completion from {self.url}: {err}") from None
        raise LLMError("rate limited: retries exhausted")


def client_from_args(
    base_url: str, model: str, api_key_env: str | None, timeout: float = 600, max_tokens: int | None = None
) -> OpenAICompatClient:
    key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not key:
        raise LLMError(f"environment variable {api_key_env} is not set")
    return OpenAICompatClient(base_url, model, key, timeout=timeout, max_tokens=max_tokens)


class OpenAICompatEmbedder:
    """Embeddings from an OpenAI-compatible /embeddings endpoint (Ollama serves one at /v1)."""

    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, *, timeout: float = 300, batch: int = 16
    ):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch = batch

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start : start + self.batch]
            body = json.dumps({"model": self.model, "input": chunk}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(self.url, body, headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                detail = err.read().decode("utf-8", "replace")[:300]
                raise LLMError(f"HTTP {err.code} from {self.url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                raise LLMError(f"could not get embeddings from {self.url}: {err}") from None
            try:
                rows = sorted(payload["data"], key=lambda row: row.get("index", 0))
                got = [[float(x) for x in row["embedding"]] for row in rows]
            except (KeyError, TypeError, ValueError):
                raise LLMError(f"unexpected embeddings response: {json.dumps(payload)[:300]}") from None
            if len(got) != len(chunk):
                raise LLMError(f"asked for {len(chunk)} embeddings, got {len(got)}")
            vectors.extend(got)
        return vectors
