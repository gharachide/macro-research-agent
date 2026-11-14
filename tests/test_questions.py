from collections import Counter

from research_agent.questions import _when, generate


def test_generates_the_requested_mix(snapshot):
    questions = generate(snapshot, n=100, seed=7)
    kinds = Counter(q.kind for q in questions)
    # only 6 series in the test snapshot, so "latest" is capped at one per series
    assert kinds == {"latest": 6, "value_on": 30, "change": 30, "extreme": 15, "spread": 10}
    assert [q.id for q in questions] == [f"num-{i:03d}" for i in range(1, len(questions) + 1)]
    assert len({(q.question_en) for q in questions}) == len(questions)


def test_is_deterministic(snapshot):
    a = [q.to_dict() for q in generate(snapshot, n=40, seed=3)]
    b = [q.to_dict() for q in generate(snapshot, n=40, seed=3)]
    c = [q.to_dict() for q in generate(snapshot, n=40, seed=4)]
    assert a == b
    assert a != c


def test_every_answer_follows_from_its_evidence(snapshot):
    for q in generate(snapshot, n=100, seed=7):
        assert q.question_en and q.question_pt and q.as_of == snapshot.as_of
        for e in q.evidence:
            assert snapshot.has_observation(e["series_id"], e["date"], e["value"])
            assert e["source_code"] == snapshot.get(e["series_id"]).source.code
        values = [e["value"] for e in q.evidence]
        if q.kind in ("latest", "value_on", "extreme"):
            assert values == [q.answer]
        elif q.kind == "change":
            assert abs((values[1] - values[0]) - q.answer) < 1e-9
        else:
            assert abs((values[0] - values[1]) - q.answer) < 1e-9


def test_spread_names_each_series_with_its_own_date(snapshot):
    spreads = [q for q in generate(snapshot, n=100, seed=7) if q.kind == "spread"]
    assert spreads
    for q in spreads:
        assert "available on that date" not in q.question_en
        for e in q.evidence:
            series = snapshot.get(e["series_id"])
            en, pt = _when(series, e["date"])  # the month for monthly series, the ISO date otherwise
            assert f"{series.title} {en}" in q.question_en
            assert f"{series.title_pt} {pt}" in q.question_pt


def test_extreme_is_the_extreme_of_its_year(snapshot):
    for q in generate(snapshot, n=100, seed=7):
        if q.kind != "extreme":
            continue
        e = q.evidence[0]
        year = e["date"][:4]
        window = [o.value for o in snapshot.get(e["series_id"]).window(f"{year}-01-01", f"{year}-12-31")]
        assert q.answer in (max(window), min(window))
