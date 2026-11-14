"""The data the agent is allowed to know: one dated snapshot of official series.

The snapshot is exported by the Olbem platform (`npm run snapshot`). Every number
the agent states must come from here, and every question in the evaluation has
its answer computed from here, so a run is reproducible for a given file.

The functions below are deliberately small and pure. In week 2 they become the
agent's tools; the evaluation uses the same code to compute the expected answers.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1


class SnapshotError(ValueError):
    """The file is not a usable snapshot."""


@dataclass(frozen=True)
class Source:
    name: str
    code: str
    url: str


@dataclass(frozen=True)
class Observation:
    series_id: str
    date: str
    value: float


@dataclass(frozen=True)
class Series:
    id: str
    title: str
    title_pt: str
    unit: str
    frequency: str
    source: Source
    points: tuple[tuple[str, float], ...]
    note: str | None = None

    @property
    def dates(self) -> list[str]:
        return [d for d, _ in self.points]

    def _obs(self, i: int) -> Observation:
        d, v = self.points[i]
        return Observation(self.id, d, v)

    def latest(self) -> Observation:
        return self._obs(len(self.points) - 1)

    def value_on(self, date: str) -> Observation | None:
        """The last observation on or before `date` (ISO), or None if the series starts later."""
        i = bisect_right(self.dates, date)
        return self._obs(i - 1) if i else None

    def window(self, start: str, end: str) -> list[Observation]:
        """Observations with start <= date <= end."""
        return [Observation(self.id, d, v) for d, v in self.points if start <= d <= end]

    def change(self, start: str, end: str) -> tuple[Observation, Observation, float] | None:
        """Difference between the observations in force on `end` and on `start`, in the series' unit."""
        a, b = self.value_on(start), self.value_on(end)
        if a is None or b is None:
            return None
        return a, b, round(b.value - a.value, 6)

    def extreme(self, start: str, end: str, kind: str) -> Observation | None:
        """Highest (`max`) or lowest (`min`) observation in the window; the earliest one on a tie."""
        if kind not in ("max", "min"):
            raise ValueError("kind must be 'max' or 'min'")
        obs = self.window(start, end)
        if not obs:
            return None
        pick = max if kind == "max" else min
        best = pick(o.value for o in obs)
        return next(o for o in obs if o.value == best)


@dataclass(frozen=True)
class Snapshot:
    as_of: str
    generated_at: str
    series: dict[str, Series]
    missing: tuple[str, ...] = ()

    def get(self, series_id: str) -> Series:
        try:
            return self.series[series_id]
        except KeyError:
            raise SnapshotError(f"unknown series: {series_id}") from None

    def has_observation(self, series_id: str, date: str, value: float, tolerance: float = 0.005) -> bool:
        """True if (series, date, value) is literally in the snapshot. Used to catch invented numbers."""
        series = self.series.get(series_id)
        if series is None:
            return False
        return any(d == date and abs(v - value) <= tolerance for d, v in series.points)


def parse_snapshot(raw: dict) -> Snapshot:
    if raw.get("schema") != SCHEMA:
        raise SnapshotError(f"unsupported snapshot schema: {raw.get('schema')!r} (expected {SCHEMA})")
    as_of = raw.get("asOf")
    if not isinstance(as_of, str):
        raise SnapshotError("snapshot has no asOf date")
    series: dict[str, Series] = {}
    for sid, s in raw.get("series", {}).items():
        points = tuple((str(d), float(v)) for d, v in s["points"])
        if not points:
            raise SnapshotError(f"{sid}: empty series")
        dates = [d for d, _ in points]
        if dates != sorted(set(dates)):
            raise SnapshotError(f"{sid}: dates must be unique and ascending")
        if dates[-1] > as_of:
            raise SnapshotError(f"{sid}: observation after the snapshot date")
        src = s["source"]
        series[sid] = Series(
            id=sid,
            title=s["title"],
            title_pt=s.get("titlePt", s["title"]),
            unit=s["unit"],
            frequency=s["frequency"],
            source=Source(src["name"], src["code"], src["url"]),
            points=points,
            note=s.get("note"),
        )
    if not series:
        raise SnapshotError("snapshot has no series")
    return Snapshot(as_of, str(raw.get("generatedAt", "")), series, tuple(raw.get("missing", [])))


def load_snapshot(path: str | Path) -> Snapshot:
    return parse_snapshot(json.loads(Path(path).read_text(encoding="utf-8")))
