from research_agent.tools import MAX_OBSERVATIONS, TOOL_SPECS, call_tool, catalog


def test_get_value_latest_on_a_date_and_on_a_month(snapshot):
    ipca = snapshot.get("br.ipca_12m")
    assert call_tool(snapshot, "get_value", {"series_id": "br.ipca_12m"})["date"] == ipca.latest().date
    assert (
        call_tool(snapshot, "get_value", {"series_id": "br.ipca_12m", "date": "2025-04-17"})["date"]
        == "2025-04-01"
    )
    april = call_tool(snapshot, "get_value", {"series_id": "br.ipca_12m", "date": "2025-04"})
    assert april == {
        "series_id": "br.ipca_12m",
        "date": "2025-04-01",
        "value": ipca.value_on("2025-04-01").value,
        "unit": "%",
    }


def test_get_change_and_extreme_match_the_series_functions(snapshot):
    selic = snapshot.get("br.selic_target")
    change = call_tool(
        snapshot,
        "get_change",
        {"series_id": "br.selic_target", "start_date": "2025-01-02", "end_date": "2026-01-02"},
    )
    assert change["change"] == selic.change("2025-01-02", "2026-01-02")[2]
    assert change["start"]["date"] == "2025-01-02" and change["end"]["date"] == "2026-01-02"
    top = call_tool(
        snapshot,
        "get_extreme",
        {"series_id": "br.selic_target", "start_date": "2025-01", "end_date": "2025-12", "kind": "max"},
    )
    assert top["value"] == selic.extreme("2025-01-01", "2025-12-31", "max").value


def test_get_observations_is_capped(snapshot):
    few = call_tool(
        snapshot,
        "get_observations",
        {"series_id": "br.ipca_12m", "start_date": "2025-01", "end_date": "2025-06"},
    )
    assert [d for d, _ in few["observations"]] == [f"2025-0{m}-01" for m in range(1, 7)]
    many = call_tool(
        snapshot,
        "get_observations",
        {"series_id": "us.fed_funds", "start_date": "2024-01", "end_date": "2025-12"},
    )
    assert str(MAX_OBSERVATIONS) in many["error"]


def test_mistakes_come_back_as_errors_the_model_can_fix(snapshot):
    assert "br.selic_target" in call_tool(snapshot, "get_value", {"series_id": "br.selic"})["error"]
    assert (
        "YYYY-MM-DD"
        in call_tool(snapshot, "get_value", {"series_id": "us.fed_funds", "date": "April 2025"})["error"]
    )
    assert (
        "starts on"
        in call_tool(snapshot, "get_value", {"series_id": "us.fed_funds", "date": "2020-01-01"})["error"]
    )
    assert (
        "missing"
        in call_tool(snapshot, "get_change", {"series_id": "us.fed_funds", "start_date": "2025-01-01"})[
            "error"
        ]
    )
    assert (
        "after"
        in call_tool(
            snapshot,
            "get_extreme",
            {"series_id": "us.fed_funds", "start_date": "2026-01", "end_date": "2025-01", "kind": "max"},
        )["error"]
    )
    assert (
        "kind"
        in call_tool(
            snapshot,
            "get_extreme",
            {"series_id": "us.fed_funds", "start_date": "2025-01", "end_date": "2025-02", "kind": "avg"},
        )["error"]
    )
    assert (
        "full date"
        in call_tool(snapshot, "get_value", {"series_id": "us.fed_funds", "date": "2025-04"})["error"]
    )
    month_on_daily = {"series_id": "us.fed_funds", "start_date": "2025-01", "end_date": "2025-06-02"}
    assert "full date" in call_tool(snapshot, "get_change", month_on_daily)["error"]
    assert "unknown tool" in call_tool(snapshot, "buy", {})["error"]
    assert "JSON object" in call_tool(snapshot, "get_value", "br.selic_target")["error"]


def test_specs_and_catalog(snapshot):
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert names == ["get_value", "get_change", "get_extreme", "get_observations", "submit_answer"]
    lines = catalog(snapshot).splitlines()
    assert len(lines) == len(snapshot.series) and lines[0].startswith("- br.ipca_12m: ")
