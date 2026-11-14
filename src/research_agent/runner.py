"""Run the agent over a question set, one JSON line per question, resumable."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from research_agent.agent import answer_question
from research_agent.llm import ChatClient, LLMAborted, LLMTimeout, ToolsUnsupported
from research_agent.retrieval import Retriever
from research_agent.snapshot import Snapshot
from research_agent.tools import tool_specs


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["id"] for line in lines if line.strip()}


def _ask(
    client: ChatClient, snapshot: Snapshot, text: str, protocol: str, max_steps: int, docs: Retriever | None
) -> dict:
    try:
        return answer_question(client, snapshot, text, protocol=protocol, max_steps=max_steps, docs=docs)
    except (LLMTimeout, LLMAborted) as err:
        row = {"answer": None, "refused": False, "citations": [], "text": "", "tokens": 0, "steps": 0}
        return {**row, "error": "timeout" if isinstance(err, LLMTimeout) else "aborted", "trace": []}


def _wrote_calls_as_text(row: dict, docs: Retriever | None) -> bool:
    """The model never made a tool call, and its text names one of the tools."""
    if row.get("error") != "no_tool_call" or row.get("trace"):
        return False
    names = [t["function"]["name"] for t in tool_specs(docs is not None)]
    return any(name in str(row.get("text", "")) for name in names)


def run(
    client: ChatClient,
    snapshot: Snapshot,
    questions: list[dict],
    out: str | Path,
    *,
    lang: str = "en",
    protocol: str = "auto",
    max_steps: int = 8,
    limit: int | None = None,
    docs: Retriever | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Answer the questions not yet in `out` and append them. Returns how many were answered now.

    A failure of the endpoint (LLMError) stops the run and is raised; what was written stays,
    and running again continues from there. A timeout, or a generation the server aborted because the
    model looped, is different: that question failed, so it is recorded as a failed answer and the run
    goes on.

    With protocol "auto", the run starts with native tool calling and switches to the JSON
    protocol, once and for the rest of the run, when the endpoint rejects the tools field or when
    the model writes a tool call as plain text instead of making it. The question that showed
    it is asked again with the JSON protocol.
    """
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(path)
    todo = [q for q in questions if q["id"] not in skip][:limit]
    if skip:
        log(f"resuming: {len(skip)} already answered in {path}")
    current = "tools" if protocol == "auto" else protocol
    with path.open("a", encoding="utf-8") as f:
        for n, q in enumerate(todo, start=1):
            text = q.get(f"question_{lang}") or q["question_en"]
            started = time.perf_counter()
            try:
                row = _ask(client, snapshot, text, current, max_steps, docs)
                if protocol == "auto" and current == "tools" and _wrote_calls_as_text(row, docs):
                    log(f"{q['id']}: the model wrote its tool call as text: switching to the JSON protocol")
                    current = "json"
                    started = time.perf_counter()
                    row = _ask(client, snapshot, text, current, max_steps, docs)
            except ToolsUnsupported:
                if protocol != "auto":
                    raise
                log("the model has no native tool calling: switching to the JSON protocol")
                current = "json"
                started = time.perf_counter()
                row = _ask(client, snapshot, text, current, max_steps, docs)
            row = {
                "id": q["id"],
                **row,
                "model": getattr(client, "model", None),
                "protocol": current,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = row["error"] or ("refused" if row["refused"] else row["answer"])
            log(f"[{n}/{len(todo)}] {q['id']}: {status} ({row['steps']} steps, {row['latency_ms']} ms)")
    return len(todo)
