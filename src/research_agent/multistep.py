"""Generate the multi-step part of the evaluation.

The numeric set maps each question onto one or two tool calls, and a small local
model answers it almost perfectly. These questions need what that set does not:
several lookups combined, counting, averaging, compounding, comparing two results,
and working out a date. The wording is written by hand per skill; the series, the
dates and the expected answer come from the snapshot, so nothing is hand-labelled.

Every question is answerable with the agent's tools: windows stay within the limit
of `get_observations`, and no question needs more than six lookups.
"""

from __future__ import annotations

import calendar
import random
import statistics
from collections.abc import Callable

from research_agent.questions import (
    DERIVED_TOLERANCE,
    LOOKUP_TOLERANCE,
    Question,
    _change_unit,
    _days_apart,
    _evidence,
    _when,
)
from research_agent.snapshot import Observation, Series, Snapshot

COUNT_TOLERANCE = 0.001  # a count is exact
PER_SKILL = 3
MIN_ANSWER = 0.1  # a difference this close to zero would be scored right for an answer of "no change"

MONTHLY_INFLATION = ["br.ipca_12m", "us.cpi_yoy", "us.core_cpi_yoy", "us.core_pce_yoy", "ea.hicp_yoy"]
DAILY_LEVELS = [
    "us.treasury_10y",
    "us.spread_10y_2y",
    "br.tesouro_ipca_10y",
    "us.treasury_2y",
    "us.breakeven_10y",
]
# comparable pairs: the same kind of quantity, same frequency
MOVE_PAIRS = [
    ("br.ipca_12m", "us.cpi_yoy"),
    ("us.core_cpi_yoy", "us.core_pce_yoy"),
    ("us.treasury_10y", "us.treasury_2y"),
    ("us.breakeven_5y", "us.breakeven_10y"),
    ("br.tesouro_ipca_10y", "us.real_10y"),
]
# policy rate and the inflation it is compared with
REAL_PAIRS = [("br.selic_target", "br.ipca_12m"), ("us.fed_funds", "us.cpi_yoy")]


def _have(snapshot: Snapshot, ids: list[str]) -> list[Series]:
    return [snapshot.series[i] for i in ids if i in snapshot.series]


def _full_years(series: Series, minimum: int) -> list[str]:
    """Calendar years, before the snapshot's, in which the series has at least `minimum` observations."""
    last_year = series.dates[-1][:4]
    years = sorted({d[:4] for d in series.dates})
    return [y for y in years if y < last_year and len(series.window(f"{y}-01-01", f"{y}-12-31")) >= minimum]


def _month_end(year: int, month: int) -> str:
    return f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _q(
    skill: str,
    en: str,
    pt: str,
    answer: float,
    unit: str,
    tolerance: float,
    evidence: list[dict],
    window: tuple[str, str, int] | None = None,
) -> Question:
    q = Question("", "multi", en, pt, round(answer, 6), unit, tolerance, evidence)
    # for a count, the evidence is what was counted; the window says what was looked at
    q.window = window  # type: ignore[attr-defined]
    return q


def _range_year(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    s = rng.choice(list(snapshot.series.values()))
    years = _full_years(s, 6)
    if not years:
        return None
    year = rng.choice(years)
    if (s.id, year) in used:
        return None
    hi, lo = (
        s.extreme(f"{year}-01-01", f"{year}-12-31", "max"),
        s.extreme(f"{year}-01-01", f"{year}-12-31", "min"),
    )
    assert hi and lo
    if hi.value == lo.value:
        return None
    used.add((s.id, year))
    unit_en, unit_pt = _change_unit(s.unit)
    return _q(
        "range_year",
        f"In {year}, what was the gap between the highest and the lowest value of {s.title}, in {unit_en}?",
        f"Em {year}, qual foi a distância entre o maior e o menor valor de {s.title_pt}, em {unit_pt}?",
        hi.value - lo.value,
        unit_en,
        DERIVED_TOLERANCE,
        _evidence(snapshot, hi, lo),
    )


def _extreme_to_latest(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    s = rng.choice(list(snapshot.series.values()))
    years = _full_years(s, 6)
    if not years:
        return None
    year, kind = rng.choice(years), rng.choice(["max", "min"])
    if (s.id, year, kind) in used:
        return None
    used.add((s.id, year, kind))
    ext, latest = s.extreme(f"{year}-01-01", f"{year}-12-31", kind), s.latest()
    assert ext
    word_en, word_pt = ("highest", "maior") if kind == "max" else ("lowest", "menor")
    unit_en, unit_pt = _change_unit(s.unit)
    return _q(
        "extreme_to_latest",
        f"How far is the most recent value of {s.title} from its {word_en} value of {year}? Answer as the most recent value minus that {year} value, in {unit_en}.",
        f"Qual a distância entre o valor mais recente de {s.title_pt} e o seu {word_pt} valor de {year}? Responda como o valor mais recente menos esse valor de {year}, em {unit_pt}.",
        latest.value - ext.value,
        unit_en,
        DERIVED_TOLERANCE,
        _evidence(snapshot, latest, ext),
    )


def _threshold(values: list[float], step: float) -> float | None:
    """A round threshold near the median that no value equals and that splits the values."""
    base = round(statistics.median(values) / step) * step
    for candidate in (base, base + step, base - step):
        candidate = round(candidate, 2)
        above = sum(1 for v in values if v > candidate)
        if candidate not in values and 0 < above < len(values):
            return candidate
    return None


def _count_months(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    pool = _have(snapshot, MONTHLY_INFLATION)
    if not pool:
        return None
    s = rng.choice(pool)
    years = _full_years(s, 12)
    if not years:
        return None
    year = rng.choice(years)
    obs = s.window(f"{year}-01-01", f"{year}-12-31")
    thr = _threshold([o.value for o in obs], 0.5)
    if thr is None or (s.id, year) in used:
        return None
    used.add((s.id, year))
    counted = [o for o in obs if o.value > thr]
    return _q(
        "count_months",
        f"In how many months of {year} was {s.title} above {thr}%?",
        f"Em quantos meses de {year} {s.title_pt} ficou acima de {thr}%?",
        len(counted),
        "months",
        COUNT_TOLERANCE,
        _evidence(snapshot, *counted),
        window=(obs[0].date, obs[-1].date, len(obs)),
    )


def _count_days(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    pool = _have(snapshot, DAILY_LEVELS)
    if not pool:
        return None
    s = rng.choice(pool)
    month = rng.choice(sorted({d[:7] for d in s.dates})[1:-1] or [None])
    if month is None or (s.id, month) in used:
        return None
    obs = s.window(f"{month}-01", f"{month}-31")
    thr = _threshold([o.value for o in obs], 0.05) if len(obs) >= 15 else None
    if thr is None:
        return None
    used.add((s.id, month))
    en, pt = _when(Series(s.id, "", "", "", "monthly", s.source, ()), f"{month}-01")
    unit = "%" if s.unit.startswith("%") else f" {s.unit}"
    counted = [o for o in obs if o.value > thr]
    return _q(
        "count_days",
        f"On how many days with an observation {en} was {s.title} above {thr}{unit}?",
        f"Em quantos dias com observação {pt} {s.title_pt} ficou acima de {thr}{unit}?",
        len(counted),
        "days",
        COUNT_TOLERANCE,
        _evidence(snapshot, *counted),
        window=(obs[0].date, obs[-1].date, len(obs)),
    )


def _consecutive_months(s: Series, rng: random.Random, length: int) -> list[Observation] | None:
    if len(s.points) <= length:
        return None
    start = rng.randrange(0, len(s.points) - length)
    return [Observation(s.id, d, v) for d, v in s.points[start : start + length]]


def _average(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    pool = _have(snapshot, MONTHLY_INFLATION)
    if not pool:
        return None
    s = rng.choice(pool)
    obs = _consecutive_months(s, rng, rng.choice([3, 4, 6]))
    if obs is None or (s.id, obs[0].date, len(obs)) in used:
        return None
    used.add((s.id, obs[0].date, len(obs)))
    (en1, pt1), (en2, pt2) = _when(s, obs[0].date), _when(s, obs[-1].date)
    return _q(
        "average",
        f"What was the average of {s.title} over the {len(obs)} months from {en1[3:]} to {en2[3:]} (both included)?",
        f"Qual foi a média de {s.title_pt} nos {len(obs)} meses de {pt1[3:]} a {pt2[3:]} (inclusive)?",
        statistics.fmean(o.value for o in obs),
        s.unit,
        LOOKUP_TOLERANCE,
        _evidence(snapshot, *obs),
    )


def _compound(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    if "br.ipca_month" not in snapshot.series:
        return None
    s = snapshot.series["br.ipca_month"]
    obs = _consecutive_months(s, rng, rng.choice([3, 4]))
    if obs is None or (obs[0].date, len(obs)) in used:
        return None
    used.add((obs[0].date, len(obs)))
    total = 1.0
    for o in obs:
        total *= 1 + o.value / 100
    (en1, pt1), (en2, pt2) = _when(s, obs[0].date), _when(s, obs[-1].date)
    return _q(
        "compound",
        f"Compounding the monthly changes, what was Brazil's accumulated IPCA inflation over the {len(obs)} months from {en1[3:]} to {en2[3:]} (both included), in %?",
        f"Compondo as variações mensais, qual foi o IPCA acumulado nos {len(obs)} meses de {pt1[3:]} a {pt2[3:]} (inclusive), em %?",
        (total - 1) * 100,
        "%",
        LOOKUP_TOLERANCE,
        _evidence(snapshot, *obs),
    )


def _two_dates(a: Series, b: Series, rng: random.Random) -> tuple[str, str] | None:
    common = sorted(set(a.dates) & set(b.dates))
    if len(common) < 2:
        return None
    d1, d2 = sorted(rng.sample(common, 2))
    return (d1, d2) if _days_apart(d1, d2) >= 90 else None


def _moves(snapshot: Snapshot, rng: random.Random):
    pairs = [(a, b) for a, b in MOVE_PAIRS if a in snapshot.series and b in snapshot.series]
    if not pairs:
        return None
    a_id, b_id = rng.choice(pairs)
    a, b = snapshot.get(a_id), snapshot.get(b_id)
    dates = _two_dates(a, b, rng)
    if dates is None:
        return None
    ca, cb = a.change(*dates), b.change(*dates)
    assert ca and cb
    return a, b, dates, ca, cb


def _bigger_move(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    picked = _moves(snapshot, rng)
    if picked is None:
        return None
    a, b, (d1, d2), ca, cb = picked
    if (
        max(abs(ca[2]), abs(cb[2])) < MIN_ANSWER
        or abs(abs(ca[2]) - abs(cb[2])) < 0.05
        or ("bigger", a.id, d1, d2) in used
    ):  # too close to call
        return None
    used.add(("bigger", a.id, d1, d2))
    (en1, pt1), (en2, pt2) = _when(a, d1), _when(a, d2)
    return _q(
        "bigger_move",
        f"Between its value {en1} and its value {en2}, which moved more in absolute terms: {a.title} or {b.title}? Answer with the size of the larger move in percentage points, as a positive number.",
        f"Entre o valor {pt1} e o valor {pt2}, o que variou mais em termos absolutos: {a.title_pt} ou {b.title_pt}? Responda com o tamanho da maior variação em pontos percentuais, como número positivo.",
        max(abs(ca[2]), abs(cb[2])),
        "percentage points",
        DERIVED_TOLERANCE,
        _evidence(snapshot, ca[0], ca[1], cb[0], cb[1]),
    )


def _change_gap(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    picked = _moves(snapshot, rng)
    if picked is None:
        return None
    a, b, (d1, d2), ca, cb = picked
    if abs(ca[2] - cb[2]) < MIN_ANSWER or ("gap", a.id, d1, d2) in used or ("bigger", a.id, d1, d2) in used:
        return None
    used.add(("gap", a.id, d1, d2))
    (en1, pt1), (en2, pt2) = _when(a, d1), _when(a, d2)
    return _q(
        "change_gap",
        f"Between its value {en1} and its value {en2}, by how much did the change in {a.title} exceed the change in {b.title}? Answer as the first change minus the second, in percentage points (negative if it was smaller).",
        f"Entre o valor {pt1} e o valor {pt2}, quanto a variação de {a.title_pt} superou a variação de {b.title_pt}? Responda como a primeira variação menos a segunda, em pontos percentuais (negativo se foi menor).",
        ca[2] - cb[2],
        "percentage points",
        DERIVED_TOLERANCE,
        _evidence(snapshot, ca[0], ca[1], cb[0], cb[1]),
    )


def _real_gap(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    pairs = [(a, b) for a, b in REAL_PAIRS if a in snapshot.series and b in snapshot.series]
    if not pairs:
        return None
    rate_id, infl_id = rng.choice(pairs)
    rate, infl = snapshot.get(rate_id), snapshot.get(infl_id)
    months = [
        d for d in infl.dates if rate.dates[0] <= d and _month_end(int(d[:4]), int(d[5:7])) <= rate.dates[-1]
    ]
    if len(months) < 2:
        return None
    m1, m2 = sorted(rng.sample(months, 2))
    if (rate_id, m1, m2) in used:
        return None
    gaps = []
    for m in (m1, m2):
        end = _month_end(int(m[:4]), int(m[5:7]))
        r, i = rate.value_on(end), infl.value_on(m)
        assert r and i
        gaps.append((end, r, i, r.value - i.value))
    if abs(gaps[0][3] - gaps[1][3]) < 0.05:
        return None
    used.add((rate_id, m1, m2))
    (en1, pt1), (en2, pt2) = _when(infl, m1), _when(infl, m2)
    return _q(
        "real_gap",
        f"Define the gap as {rate.title} on the last day of a month minus {infl.title} for that month. Compare {en1[3:]} (rate on {gaps[0][0]}) with {en2[3:]} (rate on {gaps[1][0]}): what was the larger of the two gaps, in percentage points?",
        f"Defina a diferença como {rate.title_pt} no último dia de um mês menos {infl.title_pt} desse mês. Compare {pt1[3:]} (taxa em {gaps[0][0]}) com {pt2[3:]} (taxa em {gaps[1][0]}): qual foi a maior das duas diferenças, em pontos percentuais?",
        max(g[3] for g in gaps),
        "percentage points",
        DERIVED_TOLERANCE,
        _evidence(snapshot, gaps[0][1], gaps[0][2], gaps[1][1], gaps[1][2]),
    )


def _expectation_gap(snapshot: Snapshot, rng: random.Random, used: set) -> Question | None:
    if "br.ipca_expected_12m" not in snapshot.series or "br.ipca_12m" not in snapshot.series:
        return None
    expected, actual = snapshot.series["br.ipca_expected_12m"], snapshot.series["br.ipca_12m"]
    by_month: dict[str, str] = {}
    for d in expected.dates:  # the first observation of each month
        by_month.setdefault(d[:7], d)
    options = [(d, f"{int(m[:4]) + 1}{m[4:]}-01") for m, d in by_month.items()]
    options = [(d, later) for d, later in options if later in actual.dates]
    if not options:
        return None
    d, later = rng.choice(options)
    if d in used:
        return None
    used.add(d)
    e, a = expected.value_on(d), actual.value_on(later)
    assert e and a
    return _q(
        "expectation_gap",
        f"On {d}, the market expected a certain IPCA inflation for the following 12 months (Focus survey). Compare it with the 12-month IPCA actually recorded for the same calendar month one year later: what was actual inflation minus the expectation, in percentage points?",
        f"Em {d}, o mercado esperava uma certa inflação (IPCA) para os 12 meses seguintes (Focus). Compare com o IPCA em 12 meses efetivamente registrado no mesmo mês do ano seguinte: quanto foi a inflação efetiva menos a expectativa, em pontos percentuais?",
        a.value - e.value,
        "percentage points",
        DERIVED_TOLERANCE,
        _evidence(snapshot, e, a),
    )


SKILLS: dict[str, Callable[[Snapshot, random.Random, set], Question | None]] = {
    "range_year": _range_year,
    "extreme_to_latest": _extreme_to_latest,
    "count_months": _count_months,
    "count_days": _count_days,
    "average": _average,
    "compound": _compound,
    "bigger_move": _bigger_move,
    "change_gap": _change_gap,
    "real_gap": _real_gap,
    "expectation_gap": _expectation_gap,
}


def generate_multistep(snapshot: Snapshot, per_skill: int = PER_SKILL, seed: int = 11) -> list[dict]:
    """`per_skill` questions of each skill the snapshot supports, as dicts ready for JSONL."""
    rng = random.Random(seed)
    used: set = set()
    out: list[dict] = []
    for skill, build in SKILLS.items():
        made, attempts = 0, 0
        covered: set = set()  # series already asked about in this skill
        while made < per_skill and attempts < 300:
            attempts += 1
            trial = set(used)  # a discarded question must not use up its dates
            q = build(snapshot, rng, trial)
            if q is None:
                continue
            series = frozenset(e["series_id"] for e in q.evidence)
            if series in covered and attempts < 60:  # prefer variety; give up on it if the snapshot is small
                continue
            covered.add(series)
            used.update(trial)
            row = {**q.to_dict(), "skill": skill}
            if q.window:  # type: ignore[attr-defined]
                start, end, n = q.window  # type: ignore[attr-defined]
                row["window"] = {"start": start, "end": end, "observations": n}
            out.append(row)
            made += 1
    for i, q in enumerate(out, start=1):
        q["id"] = f"multi-{i:02d}"
        q["as_of"] = snapshot.as_of
    return out
