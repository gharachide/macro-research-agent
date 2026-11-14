import json

import pytest

from research_agent.snapshot import SnapshotError, load_snapshot, parse_snapshot


def test_loads_a_file(tmp_path, raw):
    path = tmp_path / "economy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    snap = load_snapshot(path)
    assert snap.as_of == "2026-10-30"
    assert snap.get("br.ipca_12m").source.code == "13522"


def test_value_on_uses_the_last_observation_in_force(snapshot):
    ipca = snapshot.get("br.ipca_12m")
    assert ipca.value_on("2026-03-20").date == "2026-03-01"
    assert ipca.value_on("2026-03-01").date == "2026-03-01"
    assert ipca.value_on("2022-12-31") is None
    assert ipca.latest().date == "2026-10-01"


def test_change_and_extreme(snapshot):
    selic = snapshot.get("br.selic_target")
    a, b, diff = selic.change("2025-01-02", "2026-01-02")
    assert diff == pytest.approx(b.value - a.value)
    top = selic.extreme("2025-01-01", "2025-12-31", "max")
    assert top.value == max(o.value for o in selic.window("2025-01-01", "2025-12-31"))
    assert selic.extreme("2030-01-01", "2030-12-31", "min") is None
    with pytest.raises(ValueError):
        selic.extreme("2025-01-01", "2025-12-31", "median")


def test_has_observation_catches_invented_numbers(snapshot):
    d, v = snapshot.get("us.fed_funds").points[10]
    assert snapshot.has_observation("us.fed_funds", d, v)
    assert not snapshot.has_observation("us.fed_funds", d, v + 0.5)
    assert not snapshot.has_observation("us.fed_funds", "2026-01-03", v)  # a Saturday
    assert not snapshot.has_observation("us.unknown", d, v)


@pytest.mark.parametrize(
    "breakage",
    [
        lambda r: r.update(schema=2),
        lambda r: r.update(series={}),
        lambda r: r["series"]["br.ipca_12m"].update(points=[]),
        lambda r: r["series"]["br.ipca_12m"]["points"].reverse(),
        lambda r: r["series"]["br.ipca_12m"]["points"].append(["2026-12-01", 5.0]),
    ],
)
def test_rejects_unusable_snapshots(raw, breakage):
    breakage(raw)
    with pytest.raises(SnapshotError):
        parse_snapshot(raw)
