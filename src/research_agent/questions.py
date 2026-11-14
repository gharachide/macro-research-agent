"""Generate the numeric part of the evaluation from a snapshot.

Version 2 of the set: "spread" questions name each series with the date of its own
observation. Version 1 asked for "the latest observation available on" a date, which
is ambiguous for monthly series (reference month or publication date).

Each question is built by code and its expected answer is computed by the same
series functions the agent will use as tools, so there is no hand-labelled
number to get wrong. Generation is deterministic for a given snapshot and seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from research_agent.snapshot import Observation, Series, Snapshot

LOOKUP_TOLERANCE = 0.011  # an answer rounded to two decimals is still right
DERIVED_TOLERANCE = 0.021  # a difference of two rounded numbers

MONTHS_EN = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_PT = [
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
]

# (A, B): "A minus B" must mean something to a reader
SPREAD_PAIRS = [
    ("br.selic_target", "br.ipca_12m"),
    ("br.selic_target", "us.fed_funds"),
    ("br.ipca_12m", "us.cpi_yoy"),
    ("br.tesouro_pre_5y", "br.selic_target"),
    ("us.fed_funds", "us.cpi_yoy"),
    ("us.treasury_10y", "us.treasury_2y"),
    ("us.treasury_10y", "us.breakeven_10y"),
    ("us.core_cpi_yoy", "us.core_pce_yoy"),
]

# share of each kind in a generated set
MIX = (("latest", 0.15), ("value_on", 0.30), ("change", 0.30), ("extreme", 0.15), ("spread", 0.10))


@dataclass
class Question:
    id: str
    kind: str
    question_en: str
    question_pt: str
    answer: float
    unit: str
    tolerance: float
    evidence: list[dict] = field(default_factory=list)
    as_of: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _evidence(snapshot: Snapshot, *obs: Observation) -> list[dict]:
    return [
        {
            "series_id": o.series_id,
            "source_code": snapshot.get(o.series_id).source.code,
            "date": o.date,
            "value": o.value,
        }
        for o in obs
    ]


def _when(series: Series, date: str) -> tuple[str, str]:
    """How to name a date in a question: the month for monthly series, the ISO date otherwise."""
    if series.frequency == "monthly":
        year, month = date[:4], int(date[5:7])
        return f"in {MONTHS_EN[month - 1]} {year}", f"em {MONTHS_PT[month - 1]} de {year}"
    return f"on {date}", f"em {date}"


def _change_unit(unit: str) -> tuple[str, str]:
    if unit.startswith("%") or unit == "p.p.":
        return "percentage points", "pontos percentuais"
    return "index points", "pontos do índice"


def _days_apart(a: str, b: str) -> int:
    from datetime import date

    return abs((date.fromisoformat(b) - date.fromisoformat(a)).days)


def _latest(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    free = [s for s in snapshot.series.values() if ("latest", s.id) not in used]
    if not free:
        return None
    s = rng.choice(free)
    used.add(("latest", s.id))
    o = s.latest()
    return Question(
        "",
        "latest",
        f"As of {snapshot.as_of}, what is the most recent value of {s.title}? Give the value and its date.",
        f"Em {snapshot.as_of}, qual é o valor mais recente de {s.title_pt}? Informe o valor e a data.",
        o.value,
        s.unit,
        LOOKUP_TOLERANCE,
        _evidence(snapshot, o),
    )


def _value_on(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    s = rng.choice(list(snapshot.series.values()))
    d, v = rng.choice(s.points)
    if ("value_on", s.id, d) in used:
        return None
    used.add(("value_on", s.id, d))
    en, pt = _when(s, d)
    return Question(
        "",
        "value_on",
        f"What was {s.title} {en}?",
        f"Qual foi o valor de {s.title_pt} {pt}?",
        v,
        s.unit,
        LOOKUP_TOLERANCE,
        _evidence(snapshot, Observation(s.id, d, v)),
    )


def _change(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    s = rng.choice(list(snapshot.series.values()))
    if len(s.points) < 2:
        return None
    (d1, _), (d2, _) = sorted(rng.sample(s.points, 2))
    if _days_apart(d1, d2) < 90 or ("change", s.id, d1, d2) in used:
        return None
    used.add(("change", s.id, d1, d2))
    result = s.change(d1, d2)
    assert result is not None
    a, b, diff = result
    (en1, pt1), (en2, pt2) = _when(s, d1), _when(s, d2)
    unit_en, unit_pt = _change_unit(s.unit)
    return Question(
        "",
        "change",
        f"By how much did {s.title} change between its value {en1} and its value {en2}? Answer in {unit_en} (negative if it fell).",
        f"Quanto variou {s.title_pt} entre o valor {pt1} e o valor {pt2}? Responda em {unit_pt} (negativo se caiu).",
        diff,
        unit_en,
        DERIVED_TOLERANCE,
        _evidence(snapshot, a, b),
    )


def _extreme(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    s = rng.choice(list(snapshot.series.values()))
    years = sorted({d[:4] for d in s.dates})
    year, kind = rng.choice(years), rng.choice(["max", "min"])
    start, end = f"{year}-01-01", f"{year}-12-31"
    if len(s.window(start, end)) < 3 or ("extreme", s.id, year, kind) in used:
        return None
    used.add(("extreme", s.id, year, kind))
    o = s.extreme(start, end, kind)
    assert o is not None
    partial_en = f" (up to {snapshot.as_of})" if year == snapshot.as_of[:4] else ""
    partial_pt = f" (até {snapshot.as_of})" if year == snapshot.as_of[:4] else ""
    word_en, word_pt = ("highest", "maior") if kind == "max" else ("lowest", "menor")
    return Question(
        "",
        "extreme",
        f"What was the {word_en} value of {s.title} in {year}{partial_en}, and on which date was it first reached?",
        f"Qual foi o {word_pt} valor de {s.title_pt} em {year}{partial_pt}, e em que data ele foi atingido pela primeira vez?",
        o.value,
        s.unit,
        LOOKUP_TOLERANCE,
        _evidence(snapshot, o),
    )


def _spread(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    pairs = [(a, b) for a, b in SPREAD_PAIRS if a in snapshot.series and b in snapshot.series]
    if not pairs:
        return None
    a_id, b_id = rng.choice(pairs)
    a, b = snapshot.get(a_id), snapshot.get(b_id)
    # a date both series already cover, taken from the one with fewer observations
    sparse = a if len(a.points) <= len(b.points) else b
    first = max(a.dates[0], b.dates[0])
    candidates = [d for d in sparse.dates if d >= first]
    if not candidates:
        return None
    d = rng.choice(candidates)
    if ("spread", a_id, b_id, d) in used:
        return None
    used.add(("spread", a_id, b_id, d))
    oa, ob = a.value_on(d), b.value_on(d)
    assert oa is not None and ob is not None
    # each series is named with the date of its own observation: a monthly figure is the one
    # for that reference month, whenever it was published
    (a_en, a_pt), (b_en, b_pt) = _when(a, oa.date), _when(b, ob.date)
    return Question(
        "",
        "spread",
        f"What was {a.title} {a_en} minus {b.title} {b_en}, in percentage points?",
        f"Quanto era {a.title_pt} {a_pt} menos {b.title_pt} {b_pt}, em pontos percentuais?",
        round(oa.value - ob.value, 6),
        "percentage points",
        DERIVED_TOLERANCE,
        _evidence(snapshot, oa, ob),
    )


BUILDERS: dict[str, Callable[[Snapshot, random.Random, set], Question | None]] = {
    "latest": _latest,
    "value_on": _value_on,
    "change": _change,
    "extreme": _extreme,
    "spread": _spread,
}


def generate(snapshot: Snapshot, n: int = 100, seed: int = 7) -> list[Question]:
    """Up to `n` unique numeric questions. Fewer only if the snapshot is too small for the mix."""
    rng = random.Random(seed)
    used: set = set()
    out: list[Question] = []
    targets = {kind: round(n * share) for kind, share in MIX}
    targets["value_on"] += n - sum(targets.values())  # rounding leftovers
    for kind, target in targets.items():
        made, attempts = 0, 0
        while made < target and attempts < target * 200 + 200:
            attempts += 1
            q = BUILDERS[kind](snapshot, rng, used)
            if q is not None:
                out.append(q)
                made += 1
    for i, q in enumerate(out, start=1):
        q.id = f"num-{i:03d}"
        q.as_of = snapshot.as_of
    return out
