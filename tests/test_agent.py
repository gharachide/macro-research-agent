import json

import pytest

from research_agent.agent import answer_question, normalize_submission, parse_json_call, system_prompt
from research_agent.llm import Completion, LLMAborted, LLMError, LLMTimeout, ToolCall, ToolsUnsupported
from research_agent.runner import run


class Scripted:
    """A model that plays back fixed turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append((json.loads(json.dumps(messages)), tools))
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def native(name, arguments, call_id="c1"):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "function": {"name": name}}],
    }
    return Completion("", [ToolCall(call_id, name, arguments)], tokens=10, message=message)


def text(content):
    return Completion(content, [], tokens=7, message={"role": "assistant", "content": content})


def submit(snapshot, series_id="br.selic_target"):
    o = snapshot.get(series_id).latest()
    return {
        "answer": o.value,
        "refused": False,
        "citations": [{"series_id": series_id, "date": o.date, "value": o.value}],
        "text": "ok",
    }


def test_native_tool_loop(snapshot):
    client = Scripted(
        [
            native("get_value", {"series_id": "br.selic_target"}),
            native("submit_answer", submit(snapshot), "c2"),
        ]
    )
    row = answer_question(client, snapshot, "What is the Selic target?")
    latest = snapshot.get("br.selic_target").latest()
    assert (
        row["answer"] == latest.value and row["error"] is None and row["steps"] == 2 and row["tokens"] == 20
    )
    assert row["trace"] == [
        {
            "tool": "get_value",
            "arguments": {"series_id": "br.selic_target"},
            "result": {
                "series_id": "br.selic_target",
                "date": latest.date,
                "value": latest.value,
                "unit": "% a.a.",
            },
        }
    ]
    # the second request carries the assistant's tool call and the tool result, tied by id
    messages, tools = client.seen[1]
    assert tools is not None and [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[3]["tool_call_id"] == "c1" and json.loads(messages[3]["content"])["value"] == latest.value
    assert snapshot.as_of in messages[0]["content"] and "br.selic_target" in messages[0]["content"]


def test_json_protocol_loop(snapshot):
    client = Scripted(
        [
            text('```json\n{"tool": "get_value", "arguments": {"series_id": "br.selic_target"}}\n```'),
            text("Here it is: " + json.dumps({"tool": "submit_answer", "arguments": submit(snapshot)})),
        ]
    )
    row = answer_question(client, snapshot, "Qual é a Selic?", protocol="json")
    assert row["answer"] == snapshot.get("br.selic_target").latest().value and len(row["trace"]) == 1
    messages, tools = client.seen[1]
    assert tools is None and messages[-1]["content"].startswith("TOOL RESULT: ")
    assert "exactly one JSON object" in messages[0]["content"]


def test_tool_errors_go_back_to_the_model(snapshot):
    client = Scripted(
        [native("get_value", {"series_id": "br.selic"}), native("submit_answer", submit(snapshot), "c2")]
    )
    row = answer_question(client, snapshot, "q")
    assert "Did you mean" in row["trace"][0]["result"]["error"] and row["error"] is None


def test_refusal(snapshot):
    refusal = {"answer": None, "refused": True, "citations": [], "text": "I only describe official data."}
    row = answer_question(Scripted([native("submit_answer", refusal)]), snapshot, "Should I buy?")
    assert row["refused"] is True and row["answer"] is None and row["citations"] == []


def test_a_model_that_never_calls_a_tool_is_nudged_once_then_recorded(snapshot):
    client = Scripted([text("The Selic is 15%."), text("As I said, 15%.")])
    row = answer_question(client, snapshot, "q")
    assert row["error"] == "no_tool_call" and row["answer"] is None and row["text"] == "As I said, 15%."
    assert client.seen[1][0][-1]["role"] == "user"


def test_max_steps(snapshot):
    client = Scripted([native("get_value", {"series_id": "br.selic_target"}, f"c{i}") for i in range(3)])
    row = answer_question(client, snapshot, "q", max_steps=3)
    assert row["error"] == "max_steps" and row["steps"] == 3 and len(row["trace"]) == 3


def test_normalize_submission_is_strict_about_shape_and_lenient_about_number_format():
    out = normalize_submission(
        {
            "answer": "14,25",
            "refused": "yes",
            "citations": [{"series_id": "a", "date": "d", "value": "1.5"}, "x"],
            "text": 3,
        }
    )
    assert out == {
        "answer": 14.25,
        "refused": False,
        "citations": [{"series_id": "a", "date": "d", "value": 1.5}],
        "text": "3",
    }
    assert normalize_submission("nonsense") == {"answer": None, "refused": False, "citations": [], "text": ""}
    assert normalize_submission({"answer": True})["answer"] is None


def test_parse_json_call():
    assert (
        parse_json_call('note {"x": 1} then {"tool": "get_value", "arguments": {"series_id": "a"}}').name
        == "get_value"
    )
    assert parse_json_call("no json here") is None
    assert parse_json_call('{"tool": 3}') is None


def test_system_prompt_states_the_boundary(snapshot):
    prompt = system_prompt(snapshot, "tools")
    assert "investment advice" in prompt and "refused=true" in prompt and "JSON object" not in prompt
    with pytest.raises(ValueError):
        answer_question(Scripted([]), snapshot, "q", protocol="xml")


QUESTIONS = [
    {"id": "num-001", "kind": "latest", "question_en": "Selic?", "question_pt": "Selic (pt)?"},
    {"id": "num-002", "kind": "latest", "question_en": "Selic again?", "question_pt": "Selic de novo?"},
]


def test_runner_writes_rows_and_resumes(snapshot, tmp_path):
    out = tmp_path / "runs" / "r.jsonl"
    logs = []
    client = Scripted([native("submit_answer", submit(snapshot))])
    assert run(client, snapshot, QUESTIONS, out, lang="pt", limit=1, log=logs.append) == 1
    assert client.seen[0][0][1]["content"] == "Selic (pt)?"
    client = Scripted([native("submit_answer", submit(snapshot))])
    assert run(client, snapshot, QUESTIONS, out, log=logs.append) == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["num-001", "num-002"]
    assert rows[0]["protocol"] == "tools" and rows[0]["latency_ms"] >= 0 and "trace" in rows[0]
    assert rows[0]["model"] is None  # the scripted client has no model name; a real client records it
    assert any("resuming" in line for line in logs)
    assert run(Scripted([]), snapshot, QUESTIONS, out, log=logs.append) == 0


def test_runner_falls_back_to_json_and_keeps_it(snapshot, tmp_path):
    as_json = text(json.dumps({"tool": "submit_answer", "arguments": submit(snapshot)}))
    client = Scripted([ToolsUnsupported("no tools"), as_json, as_json])
    out = tmp_path / "r.jsonl"
    assert run(client, snapshot, QUESTIONS, out, log=lambda _: None) == 2
    assert [json.loads(line)["protocol"] for line in out.read_text(encoding="utf-8").splitlines()] == [
        "json",
        "json",
    ]
    assert [tools is None for _, tools in client.seen] == [False, True, True]


def test_runner_records_a_timeout_and_goes_on(snapshot, tmp_path):
    out = tmp_path / "r.jsonl"
    client = Scripted([LLMTimeout("no answer in 600 s"), native("submit_answer", submit(snapshot))])
    assert run(client, snapshot, QUESTIONS, out, log=lambda _: None) == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert (rows[0]["error"], rows[0]["answer"], rows[0]["trace"]) == ("timeout", None, [])
    assert rows[1]["error"] is None


def test_runner_stops_on_endpoint_failure_and_keeps_what_it_wrote(snapshot, tmp_path):
    out = tmp_path / "r.jsonl"
    client = Scripted([native("submit_answer", submit(snapshot)), LLMError("connection refused")])
    with pytest.raises(LLMError):
        run(client, snapshot, QUESTIONS, out, log=lambda _: None)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ToolsUnsupported):
        run(
            Scripted([ToolsUnsupported("no tools")]),
            snapshot,
            QUESTIONS,
            tmp_path / "t.jsonl",
            protocol="tools",
            log=lambda _: None,
        )


def test_runner_switches_when_the_model_writes_tool_calls_as_text(snapshot, tmp_path):
    as_text = text('get_value({series_id: "br.selic_target"})')  # what phi4-mini does through Ollama
    as_json = text(json.dumps({"tool": "submit_answer", "arguments": submit(snapshot)}))
    client = Scripted([as_text, as_text, as_json, as_json])
    logs = []
    out = tmp_path / "r.jsonl"
    assert run(client, snapshot, QUESTIONS, out, log=logs.append) == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["protocol"], r["error"]) for r in rows] == [("json", None), ("json", None)]
    assert any("as text" in line for line in logs)
    # with an explicit protocol there is no switch: the failure is recorded as it happened
    client = Scripted([as_text, as_text])
    run(client, snapshot, QUESTIONS[:1], tmp_path / "t.jsonl", protocol="tools", log=lambda _: None)
    assert json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8"))["error"] == "no_tool_call"


def test_an_empty_reply_is_not_a_tool_call_written_as_text(snapshot, tmp_path):
    client = Scripted([text(""), text(""), native("submit_answer", submit(snapshot))])
    out = tmp_path / "r.jsonl"
    run(client, snapshot, QUESTIONS, out, log=lambda _: None)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["protocol"], r["error"]) for r in rows] == [("tools", "no_tool_call"), ("tools", None)]


def test_a_json_call_missing_its_last_brace_is_repaired_and_counted(snapshot):
    args = json.dumps(submit(snapshot))
    cut = '{\n"tool": "submit_answer",\n"arguments": ' + args  # phi4-mini stops before the outer brace
    assert parse_json_call(cut).id == "json-repaired"
    assert parse_json_call("```json\n" + cut + "\n```").name == "submit_answer"
    assert (
        parse_json_call('{"tool": "submit_answer", "arguments": {"answer": [1, 2') is None
    )  # not just braces
    row = answer_question(Scripted([text(cut)]), snapshot, "q", protocol="json")
    assert row["answer"] == snapshot.get("br.selic_target").latest().value and row["json_repairs"] == 1


def test_runner_records_an_aborted_generation_and_goes_on(snapshot, tmp_path):
    out = tmp_path / "r.jsonl"
    client = Scripted([LLMAborted("token repeat limit reached"), native("submit_answer", submit(snapshot))])
    assert run(client, snapshot, QUESTIONS, out, log=lambda _: None) == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["error"] for r in rows] == ["aborted", None]


def test_a_failure_mid_question_keeps_the_steps_before_it(snapshot):
    client = Scripted(
        [native("get_value", {"series_id": "br.selic_target"}), LLMAborted("token repeat limit reached")]
    )
    row = answer_question(client, snapshot, "q")
    assert (row["error"], row["steps"], len(row["trace"])) == ("aborted", 2, 1)


def test_a_reply_cut_at_the_token_cap_with_no_call_fails_the_question(snapshot):
    looping = Completion("and and and " * 2000, [], tokens=1024, finish_reason="length")
    client = Scripted([native("get_value", {"series_id": "br.selic_target"}), looping])
    row = answer_question(client, snapshot, "q")
    assert row["error"] == "length_limit" and row["steps"] == 2 and row["answer"] is None
    assert len(row["trace"]) == 1 and len(row["text"]) == 2000 and not client.turns


def test_a_call_inside_a_reply_cut_at_the_token_cap_is_still_used(snapshot):
    call = json.dumps({"tool": "get_value", "arguments": {"series_id": "br.selic_target"}})
    cut = Completion(call + " and and and", [], tokens=1024, finish_reason="length")
    done = text(json.dumps({"tool": "submit_answer", "arguments": submit(snapshot)}))
    row = answer_question(Scripted([cut, done]), snapshot, "q", protocol="json")
    assert row["error"] is None and row["answer"] == snapshot.get("br.selic_target").latest().value
