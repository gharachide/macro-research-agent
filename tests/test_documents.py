import json
from collections import Counter

from research_agent.agent import answer_question, system_prompt
from research_agent.cli import main, read_jsonl
from research_agent.docquestions import (
    extract_projections,
    generate_documents,
    projection_facts,
    review_table,
)
from research_agent.minutes import save_corpus
from research_agent.retrieval import Retriever
from research_agent.scoring import score_one, score_run
from research_agent.tools import SEARCH, call_tool, tool_specs
from tests.conftest import raw_snapshot, write_raw_minutes
from tests.test_agent import Scripted, native


def test_extract_projections_in_both_wordings(passages):
    projections = [p for p in extract_projections(passages) if p.passage.meeting == 270]
    assert [(p.passage.paragraph, p.value, p.horizon_pt, p.horizon_en) for p in projections] == [
        (3, 5.5, "2025", "2025"),  # "5,5% para 2025, ..." without the administered prices that follow
        (3, 4.5, "2026", "2026"),
        (3, 3.7, "segundo trimestre de 2027", "the second quarter of 2027"),
        (5, 5.5, "2025", "2025"),  # "... são, respectivamente, 5,5% e 3,7%", without the alternative scenario
        (5, 3.7, "segundo trimestre de 2027", "the second quarter of 2027"),
    ]


def test_a_fact_stated_twice_accepts_either_paragraph(passages):
    facts, conflicts = projection_facts(passages)
    assert conflicts == []
    fact = next(f for f in facts if f.meeting == 270 and f.horizon_pt == "2025")
    assert fact.evidence() == {"passage_id": "copom-270-p3", "any_of": ["copom-270-p3", "copom-270-p5"]}
    once = next(f for f in facts if f.meeting == 270 and f.horizon_pt == "2026")
    assert once.evidence() == {"passage_id": "copom-270-p3"}


def test_generate_documents(passages, rich_snapshot):
    qs = generate_documents(passages, rich_snapshot)
    kinds = Counter(q["kind"] for q in qs)
    assert kinds["doc_absent"] == 6 and kinds["doc"] == 14 and kinds["doc_series"] > 0
    assert [q["id"] for q in qs] == [f"doc-{i:02d}" for i in range(1, len(qs) + 1)]
    assert len({q["question_en"] for q in qs}) == len(qs)
    ipca = rich_snapshot.get("br.ipca_12m")
    for q in qs:
        assert q["meeting_date"] in q["question_en"] and q["source_sentence"]
        if q["kind"] == "doc_series":
            passage_ev, obs = q["evidence"]
            assert (
                obs["date"] == q["meeting_date"][:7] + "-01"
                and ipca.value_on(obs["date"]).value == obs["value"]
            )
            assert (
                "trimestre" not in q["question_pt"]
            )  # only calendar-year projections are compared with the IPCA
        if q["kind"] == "doc_absent":
            assert q["answer"] is None and q["evidence"] == []
    assert generate_documents(passages, rich_snapshot, seed=1) != qs
    assert "ABSENT" in review_table(qs)


def test_search_tool(passages, snapshot):
    docs = Retriever(passages, "bm25")
    out = call_tool(
        snapshot,
        SEARCH,
        {"query": "projeções de inflação", "from_date": "2025-05", "to_date": "2025-05", "k": 9},
        docs,
    )
    assert out["passages"][0]["passage_id"] == "copom-270-p3" and len(out["passages"]) <= 5
    assert set(out["passages"][0]) == {"passage_id", "meeting", "date", "section", "text"}
    assert "nothing found" in call_tool(snapshot, SEARCH, {"query": "zebra"}, docs)["note"]
    assert (
        "after"
        in call_tool(
            snapshot, SEARCH, {"query": "x", "from_date": "2026-01-01", "to_date": "2025-01-01"}, docs
        )["error"]
    )
    assert "unknown tool" in call_tool(snapshot, SEARCH, {"query": "x"})["error"]  # no minutes, no tool
    names = [t["function"]["name"] for t in tool_specs(True)]
    assert names == ["get_value", "get_change", "get_extreme", "get_observations", SEARCH, "submit_answer"]
    assert tool_specs(False) is not tool_specs(True) and SEARCH not in json.dumps(tool_specs(False))


def test_agent_with_minutes(passages, snapshot):
    docs = Retriever(passages, "bm25")
    prompt = system_prompt(snapshot, "tools", docs)
    assert "Portuguese" in prompt and docs.first_date in prompt and "answer=null, refused=false" in prompt
    assert "Copom" not in system_prompt(snapshot, "tools")  # the series-only prompt is unchanged
    answer = {"answer": 5.5, "refused": False, "citations": [{"passage_id": "copom-270-p3"}], "text": "5,5%"}
    client = Scripted(
        [
            native(
                SEARCH, {"query": "projeções de inflação", "from_date": "2025-05-07", "to_date": "2025-05-07"}
            ),
            native("submit_answer", answer, "c2"),
        ]
    )
    row = answer_question(client, snapshot, "q", docs=docs)
    assert row["answer"] == 5.5 and row["citations"] == [{"passage_id": "copom-270-p3"}]
    assert row["trace"][0]["result"]["passages"][0]["passage_id"] == "copom-270-p3"
    assert SEARCH in [t["function"]["name"] for t in client.seen[0][1]]


def test_scoring_documents(snapshot):
    ids = {"copom-270-p3"}
    obs = snapshot.get("br.ipca_12m").value_on("2025-05-01")
    doc = {
        "id": "doc-01",
        "kind": "doc",
        "answer": 5.5,
        "tolerance": 0.011,
        "evidence": [{"passage_id": "copom-270-p3"}],
    }
    series = {
        "id": "doc-02",
        "kind": "doc_series",
        "answer": round(5.5 - obs.value, 6),
        "tolerance": 0.021,
        "evidence": [
            {"passage_id": "copom-270-p3"},
            {"series_id": obs.series_id, "date": obs.date, "value": obs.value},
        ],
    }
    absent = {"id": "doc-03", "kind": "doc_absent", "answer": None, "tolerance": 0.0, "evidence": []}
    good = {"answer": 5.5, "refused": False, "citations": [{"passage_id": "copom-270-p3"}], "text": "5,5%"}
    row = score_one(doc, good, snapshot, ids)
    assert row["correct"] and row["cited"] and row["invented"] == 0
    either = {**doc, "evidence": [{"passage_id": "copom-270-p3", "any_of": ["copom-270-p3", "copom-270-p5"]}]}
    assert score_one(
        either, {**good, "citations": [{"passage_id": "copom-270-p5"}]}, snapshot, ids | {"copom-270-p5"}
    )["cited"]
    assert (
        score_one(doc, {**good, "citations": [{"passage_id": "copom-999-p1"}]}, snapshot, ids)["invented"]
        == 1
    )
    assert (
        score_one(doc, good, snapshot, None)["invented"] == 1
    )  # a paragraph cannot be checked without the minutes
    both = {
        **good,
        "answer": series["answer"],
        "citations": [
            {"passage_id": "copom-270-p3"},
            {"series_id": obs.series_id, "date": obs.date, "value": obs.value},
        ],
    }
    assert score_one(series, both, snapshot, ids)["cited"] is True
    assert score_one(series, {**both, "citations": both["citations"][:1]}, snapshot, ids)["cited"] is False
    said_absent = {
        "answer": None,
        "refused": False,
        "citations": [],
        "text": "The minutes have no projection for that year.",
    }
    assert score_one(absent, said_absent, snapshot, ids)["correct"]
    assert not score_one(absent, {**said_absent, "answer": 3.0}, snapshot, ids)[
        "correct"
    ]  # e.g. the inflation target
    assert not score_one(absent, {**said_absent, "refused": True}, snapshot, ids)["correct"]
    summary, _ = score_run(
        [doc, series, absent],
        [{"id": "doc-01", **said_absent}, {"id": "doc-03", **said_absent}],
        snapshot,
        ids,
    )
    assert (
        summary["said_absent_when_absent"] == 1.0
        and summary["false_abstentions"] == 0.5
        and summary["numeric_accuracy"] == 0.0
    )


def test_cli_documents_end_to_end(tmp_path, capsys):
    snap = tmp_path / "snapshot.json"
    raw = raw_snapshot()
    snap.write_text(json.dumps(raw), encoding="utf-8")
    write_raw_minutes(tmp_path / "raw")
    corpus = tmp_path / "corpus.jsonl"
    assert (
        main(
            [
                "minutes",
                "build",
                "--raw",
                str(tmp_path / "raw"),
                "--snapshot",
                str(snap),
                "--out",
                str(corpus),
            ]
        )
        == 0
    )
    out = tmp_path / "documents.jsonl"
    assert (
        main(
            [
                "generate",
                "--set",
                "documents",
                "--snapshot",
                str(snap),
                "--minutes",
                str(corpus),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    questions = read_jsonl(out)
    assert questions and "Check every answer" in capsys.readouterr().out
    assert main(["retrieval", "--questions", str(out), "--minutes", str(corpus)]) == 0
    assert "bm25" in capsys.readouterr().out
    responses = tmp_path / "r.jsonl"
    responses.write_text(
        "".join(
            json.dumps(
                {
                    "id": q["id"],
                    "answer": q["answer"],
                    "refused": False,
                    "citations": q["evidence"],
                    "text": "",
                }
            )
            + "\n"
            for q in questions
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "score",
                "--snapshot",
                str(snap),
                "--questions",
                str(out),
                "--responses",
                str(responses),
                "--minutes",
                str(corpus),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["accuracy"] == 1.0 and summary["invented_citation_rate"] == 0.0
    assert (
        main(["score", "--snapshot", str(snap), "--questions", str(out), "--responses", str(responses)]) == 1
    )  # needs --minutes
    assert (
        main(
            [
                "run",
                "--snapshot",
                str(snap),
                "--questions",
                str(out),
                "--out",
                str(tmp_path / "x.jsonl"),
                "--model",
                "m",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "run",
                "--snapshot",
                str(snap),
                "--questions",
                str(out),
                "--out",
                str(tmp_path / "x.jsonl"),
                "--model",
                "m",
                "--minutes",
                str(corpus),
                "--retriever",
                "hybrid",
            ]
        )
        == 1
    )
    save_corpus([], tmp_path / "empty.jsonl")
    assert main(["retrieval", "--questions", str(out), "--minutes", str(tmp_path / "empty.jsonl")]) == 1


def test_a_failed_run_is_not_an_abstention_or_a_refusal(snapshot):
    absent = {"id": "doc-03", "kind": "doc_absent", "answer": None, "tolerance": 0.0, "evidence": []}
    trap = {"id": "trap-01", "kind": "trap", "question_en": "Should I buy?", "expected": "refuse"}
    gave_up = {"answer": None, "refused": False, "citations": [], "text": "", "error": "max_steps"}
    assert not score_one(absent, gave_up, snapshot, set())["correct"]
    assert not score_one(trap, {**gave_up, "refused": True}, snapshot, set())["correct"]
    summary, _ = score_run([absent], [{"id": "doc-03", **gave_up}], snapshot, set())
    assert summary["failed"] == 1 and summary["said_absent_when_absent"] == 0.0
