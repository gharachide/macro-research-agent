"""The agent: a tool-calling loop that ends with one structured answer.

Two wire protocols, same tools and same prompt rules:

- "tools": the endpoint's native tool calling;
- "json":  the model writes one JSON object per turn ({"tool": ..., "arguments": ...}).
           It works with models and servers that have no tool-calling support.

Whatever happens (no tool call, too many steps), the loop returns a response
row; a failure is recorded in "error" and scored as a wrong answer.
"""

from __future__ import annotations

import json
import re

from research_agent.llm import ChatClient, LLMAborted, LLMTimeout, ToolCall
from research_agent.retrieval import Retriever
from research_agent.snapshot import Snapshot
from research_agent.tools import SEARCH, SUBMIT, call_tool, catalog, tool_specs

RULES = """You answer questions about inflation and interest rates in Brazil, the United States and the euro area, \
using only the official series listed below. The data is a snapshot taken on {as_of}; treat that as today.

Rules:
- Every number you state must come from a tool result. Never use a number from memory.
- Dates given as a month refer to that month's observation. "Change" means the later value minus the earlier one.
- When a question needs two series, look up each one and compute the result yourself.
- You describe data. You do not give investment advice, say what to buy, sell or hold, or forecast. \
If the question asks for any of that, finish with refused=true, answer=null, no citations, and one sentence \
saying you only describe official data.
- Finish every question by calling {submit} once: the single number that answers it (full precision, in the unit \
asked), every observation it depends on as citations (series_id, date and value exactly as returned), and a short \
text in the language of the question.

Series:
{catalog}"""

JSON_PROTOCOL = """

You have no native tool calling. Reply with exactly one JSON object and nothing else:
{{"tool": "<name>", "arguments": {{...}}}}
You will then receive the tool result and can call another tool. Tools:
{tools}"""

MINUTES = """

You can also search the minutes of the Copom, the Central Bank of Brazil's monetary policy committee, with \
{search}. They cover the meetings that ended from {first} to {last}, are written in Portuguese, and are split \
into numbered paragraphs. Write queries in Portuguese, and when a question names a meeting, set from_date and \
to_date to the day it ended. Cite every paragraph you use by its passage_id. If neither the series nor the \
minutes contain the answer, finish with answer=null, refused=false, and one sentence saying what is missing."""

NUDGE = f"Reply with a tool call. When you have the answer, call {SUBMIT}."


def system_prompt(snapshot: Snapshot, protocol: str, docs: Retriever | None = None) -> str:
    text = RULES.format(as_of=snapshot.as_of, submit=SUBMIT, catalog=catalog(snapshot))
    if docs is not None:
        text += MINUTES.format(search=SEARCH, first=docs.first_date, last=docs.last_date)
    if protocol == "json":
        tools = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']} Arguments: {json.dumps(t['function']['parameters']['properties'])}"
            for t in tool_specs(docs is not None)
        )
        text += JSON_PROTOCOL.format(tools=tools)
    return text


MAX_BRACE_REPAIR = 2
RUNAWAY_TEXT_KEPT = 2000  # characters of a runaway reply kept in the row, enough to see what it was


def parse_json_call(content: str) -> ToolCall | None:
    """The first JSON object in the text that names a tool, tolerant of code fences and surrounding prose.

    Small models sometimes stop before the last closing brace or two. If no complete object names a
    tool, the text from the first brace is tried again with up to MAX_BRACE_REPAIR braces added; such
    a call gets the id "json-repaired", so every repair is counted in the results."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            obj, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            return ToolCall("json", obj["tool"], obj.get("arguments", {}))
    start = content.find("{")
    if start < 0:
        return None
    tail = re.sub(r"\s*```\s*$", "", content[start:].rstrip())
    for extra in range(1, MAX_BRACE_REPAIR + 1):
        try:
            obj = json.loads(tail + "}" * extra)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            return ToolCall("json-repaired", obj["tool"], obj.get("arguments", {}))
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):  # a number typed as a string is a formatting slip, not a wrong answer
        try:
            return float(value.strip().replace(",", "."))
        except ValueError:
            return None
    return None


def normalize_submission(arguments: object) -> dict:
    args = arguments if isinstance(arguments, dict) else {}
    citations = []
    raw_citations = args.get("citations")
    for c in raw_citations if isinstance(raw_citations, list) else []:
        if isinstance(c, dict) and c.get("passage_id") not in (None, ""):
            citations.append({"passage_id": str(c["passage_id"])})
        elif isinstance(c, dict):
            citations.append(
                {
                    "series_id": str(c.get("series_id", "")),
                    "date": str(c.get("date", "")),
                    "value": _number(c.get("value")),
                }
            )
    return {
        "answer": _number(args.get("answer")),
        "refused": args.get("refused") is True,
        "citations": citations,
        "text": str(args.get("text", "")),
    }


def answer_question(
    client: ChatClient,
    snapshot: Snapshot,
    question: str,
    *,
    protocol: str = "tools",
    max_steps: int = 8,
    docs: Retriever | None = None,
) -> dict:
    """Run one question to a response row. Raises only LLMError (the endpoint failed), never on model behaviour.

    With `docs`, the agent can also search the Copom minutes."""
    if protocol not in ("tools", "json"):
        raise ValueError("protocol must be 'tools' or 'json'")
    specs = tool_specs(docs is not None)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(snapshot, protocol, docs)},
        {"role": "user", "content": question},
    ]
    row: dict = {
        "answer": None,
        "refused": False,
        "citations": [],
        "text": "",
        "tokens": 0,
        "steps": 0,
        "error": None,
        "json_repairs": 0,
        "trace": [],
    }
    nudged = False
    for step in range(1, max_steps + 1):
        try:
            completion = client.complete(messages, specs if protocol == "tools" else None)
        except (LLMTimeout, LLMAborted) as err:
            # this question failed at this step; keep what happened before it, for the analysis
            row["steps"] = step
            row["error"] = "timeout" if isinstance(err, LLMTimeout) else "aborted"
            return row
        row["tokens"] += completion.tokens
        row["steps"] = step
        if protocol == "tools":
            calls = completion.tool_calls
            messages.append(completion.message or {"role": "assistant", "content": completion.content})
        else:
            call = parse_json_call(completion.content)
            calls = [call] if call else []
            if call and call.id == "json-repaired":
                row["json_repairs"] += 1
            messages.append({"role": "assistant", "content": completion.content})
        if not calls:
            if completion.finish_reason == "length":
                # cut at the token cap with no call in it: a runaway generation, not a reply to nudge
                row.update(error="length_limit", text=completion.content[:RUNAWAY_TEXT_KEPT])
                return row
            if nudged:
                row.update(error="no_tool_call", text=completion.content)
                return row
            nudged = True
            messages.append({"role": "user", "content": NUDGE})
            continue
        for call in calls:
            if call.name == SUBMIT:
                row.update(normalize_submission(call.arguments))
                return row
            result = call_tool(snapshot, call.name, call.arguments, docs)
            row["trace"].append({"tool": call.name, "arguments": call.arguments, "result": result})
            payload = json.dumps(result, ensure_ascii=False)
            if protocol == "tools":
                messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})
            else:
                messages.append({"role": "user", "content": f"TOOL RESULT: {payload}"})
    row["error"] = "max_steps"
    return row
