import re
import statistics
from collections import Counter

import pytest

from research_agent.multistep import SKILLS, generate_multistep
from research_agent.tools import MAX_OBSERVATIONS, call_tool


def solve(snapshot, q):
    """Answer a question the way the agent has to: through the tools only."""
    ev = q["evidence"]
    skill = q["skill"]

    def value(e):
        return call_tool(snapshot, "get_value", {"series_id": e["series_id"], "date": e["date"]})["value"]

    def window(first, last):
        args = {"series_id": ev[0]["series_id"], "start_date": first, "end_date": last}
        return [v for _, v in call_tool(snapshot, "get_observations", args)["observations"]]

    def extreme(year, kind):
        args = {
            "series_id": ev[0]["series_id"],
            "start_date": f"{year}-01",
            "end_date": f"{year}-12",
            "kind": kind,
        }
        return call_tool(snapshot, "get_extreme", args)["value"]

    def change(a, b):
        args = {"series_id": a["series_id"], "start_date": a["date"], "end_date": b["date"]}
        return call_tool(snapshot, "get_change", args)["change"]

    if skill == "range_year":
        year = ev[0]["date"][:4]
        return extreme(year, "max") - extreme(year, "min")
    if skill == "extreme_to_latest":
        year = re.search(r"value of (\d{4})\?", q["question_en"]).group(1)
        kind = "max" if "highest" in q["question_en"] else "min"
        return call_tool(snapshot, "get_value", {"series_id": ev[0]["series_id"]})["value"] - extreme(
            year, kind
        )
    if skill in ("count_months", "count_days"):
        threshold = float(re.search(r"above (-?[\d.]+)", q["question_en"]).group(1))
        values = window(q["window"]["start"][:7], q["window"]["end"][:7])
        assert len(values) == q["window"]["observations"] <= MAX_OBSERVATIONS
        return sum(1 for v in values if v > threshold)
    if skill == "average":
        return statistics.fmean(window(ev[0]["date"][:7], ev[-1]["date"][:7]))
    if skill == "compound":
        total = 1.0
        for v in window(ev[0]["date"][:7], ev[-1]["date"][:7]):
            total *= 1 + v / 100
        return (total - 1) * 100
    if skill == "bigger_move":
        return max(abs(change(ev[0], ev[1])), abs(change(ev[2], ev[3])))
    if skill == "change_gap":
        return change(ev[0], ev[1]) - change(ev[2], ev[3])
    if skill == "real_gap":
        return max(value(ev[0]) - value(ev[1]), value(ev[2]) - value(ev[3]))
    if skill == "expectation_gap":
        return value(ev[1]) - value(ev[0])
    raise AssertionError(skill)


def test_three_questions_per_skill(rich_snapshot):
    questions = generate_multistep(rich_snapshot)
    assert Counter(q["skill"] for q in questions) == dict.fromkeys(SKILLS, 3)
    assert [q["id"] for q in questions] == [f"multi-{i:02d}" for i in range(1, 31)]
    assert len({q["question_en"] for q in questions}) == 30
    assert all(
        q["kind"] == "multi" and q["question_pt"] and q["as_of"] == rich_snapshot.as_of for q in questions
    )


def test_is_deterministic(rich_snapshot):
    assert generate_multistep(rich_snapshot, seed=5) == generate_multistep(rich_snapshot, seed=5)
    assert generate_multistep(rich_snapshot, seed=5) != generate_multistep(rich_snapshot, seed=6)


def test_every_question_is_answerable_with_the_tools(rich_snapshot):
    for q in generate_multistep(rich_snapshot):
        for e in q["evidence"]:
            assert rich_snapshot.has_observation(e["series_id"], e["date"], e["value"])
        assert solve(rich_snapshot, q) == pytest.approx(q["answer"], abs=1e-6), q["question_en"]


def test_counts_are_never_trivial(rich_snapshot):
    for q in generate_multistep(rich_snapshot):
        if q["skill"].startswith("count"):
            assert 0 < q["answer"] < q["window"]["observations"]
            assert q["answer"] == len(q["evidence"])  # the evidence is exactly what was counted


def test_no_answer_is_within_tolerance_of_zero(rich_snapshot):
    for q in generate_multistep(rich_snapshot):
        if q["skill"] in ("bigger_move", "change_gap"):
            assert abs(q["answer"]) >= 0.1


def test_a_small_snapshot_gives_fewer_questions_not_an_error(snapshot):
    skills = {q["skill"] for q in generate_multistep(snapshot)}
    assert "compound" not in skills and "expectation_gap" not in skills and "range_year" in skills
