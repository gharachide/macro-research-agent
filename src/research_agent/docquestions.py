"""Questions about the Copom minutes, with answers taken from the text.

In every minutes, the Copom states its inflation projections in its reference
scenario. The wording changed over the period, and the extractor reads the three
forms found in the minutes from 2023 to 2026:

- "as projeções de inflação do Copom situam-se em 4,6% para 2023, 3,5% para 2024";
- "a projeção ... para o primeiro trimestre de 2026 é 3,5%";
- "as projeções ... para 2026, 2027, e para o segundo trimestre de 2028 ... são,
  respectivamente, 5,6%, 4,7%, e 3,5%".

Only paragraphs that mention the reference scenario are read, and within them only
sentences about projections, cut where they turn to the alternative scenario or to
administered prices. Market expectations (Focus) are stated in sentences that do
not mention projections, so they are not read. The same projection often appears in two paragraphs (a
summary near the top and the detail in section B); a question accepts either.

Because the extraction is a regular expression over prose, `research-agent
generate --set documents` also prints a review table: every expected answer and
the sentence it came from, to be read by a person before the set is used.

Three kinds:

- doc:        the projection for a horizon, from the minutes of a given meeting;
- doc_series: that projection minus the 12-month IPCA of the meeting's month
              (one paragraph and one series observation);
- doc_absent: a projection for a year the minutes do not cover. The right answer
              is to say it is not there.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

from research_agent.minutes import Passage
from research_agent.questions import DERIVED_TOLERANCE, LOOKUP_TOLERANCE, MONTHS_EN, MONTHS_PT
from research_agent.snapshot import Snapshot

COUNTS = {"doc": 14, "doc_series": 10, "doc_absent": 6}
QUARTERS_EN = {"primeiro": "first", "segundo": "second", "terceiro": "third", "quarto": "fourth"}

_HORIZON = r"(?:primeiro|segundo|terceiro|quarto)\s+trimestre\s+de\s+\d{4}|\d{4}"
_VALUE = r"-?\d{1,2},\d{1,2}"
_VALUE_FIRST = re.compile(rf"({_VALUE})\s*%\s*para\s+(?:o\s+)?({_HORIZON})", re.IGNORECASE)
_HORIZON_FIRST = re.compile(rf"para\s+(?:o\s+)?({_HORIZON})\b[^.%]{{0,160}}?({_VALUE})\s*%", re.IGNORECASE)
_HORIZONS = re.compile(_HORIZON, re.IGNORECASE)
_VALUES = re.compile(rf"({_VALUE})\s*%")


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


@dataclass(frozen=True)
class Projection:
    passage: Passage
    sentence: str
    value: float
    horizon_pt: str  # "2026" or "segundo trimestre de 2027"
    horizon_en: str  # "2026" or "the second quarter of 2027"
    year: int


def _sentences(text: str) -> list[str]:
    # decimals use a comma, so a period followed by a space and a capital ends a sentence
    return re.split(r"(?<=\.)\s+(?=[A-ZÀ-Ý])", text)


def _horizon(raw: str) -> tuple[str, str, int]:
    words = re.sub(r"\s+", " ", raw.strip()).split()
    if len(words) == 1:
        return words[0], words[0], int(words[0])
    pt = " ".join(w.lower() if i < len(words) - 1 else w for i, w in enumerate(words))
    return pt, f"the {QUARTERS_EN[words[0].lower()]} quarter of {words[-1]}", int(words[-1])


def _pairs(sentence: str) -> list[tuple[str, str]]:
    """(value, horizon) pairs in one sentence, whichever of the three forms it uses."""
    if "respectivamente" in sentence:
        before, after = re.split(r"respectivamente", sentence, maxsplit=1)
        horizons = _HORIZONS.findall(before)
        values = _VALUES.findall(after)
        return list(zip(values, horizons, strict=True)) if horizons and len(horizons) == len(values) else []
    if _VALUE_FIRST.search(sentence):
        return _VALUE_FIRST.findall(sentence)
    return [(v, h) for h, v in _HORIZON_FIRST.findall(sentence)]


def extract_projections(passages: list[Passage]) -> list[Projection]:
    """The reference-scenario inflation projections stated in each minutes."""
    found: list[Projection] = []
    for p in passages:
        if "cenario de referencia" not in _plain(p.text):
            continue
        for full in _sentences(p.text):
            # what follows the alternative scenario or administered prices is not the projection asked for
            sentence = re.split(
                r"cen[aá]rio alternativo|pre[cç]os administrados", full, maxsplit=1, flags=re.IGNORECASE
            )[0]
            plain = _plain(sentence)
            if "projec" not in plain:
                continue
            for value, raw in _pairs(sentence):
                pt, en, year = _horizon(raw)
                found.append(Projection(p, full, float(value.replace(",", ".")), pt, en, year))
    return found


@dataclass(frozen=True)
class Fact:
    """One projection of one meeting, with every paragraph that states it."""

    meeting: int
    horizon_pt: str
    horizon_en: str
    year: int
    value: float
    passages: tuple[Passage, ...]
    sentence: str

    @property
    def passage(self) -> Passage:
        return self.passages[0]

    def evidence(self) -> dict:
        ids = [p.id for p in self.passages]
        return {"passage_id": ids[0], "any_of": ids} if len(ids) > 1 else {"passage_id": ids[0]}


def projection_facts(passages: list[Passage]) -> tuple[list[Fact], list[str]]:
    """Projections grouped by meeting and horizon. A horizon stated with two different values in
    one minutes is left out and reported, rather than guessed."""
    groups: dict[tuple[int, str], list[Projection]] = {}
    for proj in extract_projections(passages):
        groups.setdefault((proj.passage.meeting, proj.horizon_pt), []).append(proj)
    facts, conflicts = [], []
    for (meeting, _), projs in sorted(groups.items()):
        values = {pr.value for pr in projs}
        if len(values) > 1:
            conflicts.append(
                f"meeting {meeting}, {projs[0].horizon_pt}: {sorted(values)} in {[pr.passage.id for pr in projs]}"
            )
            continue
        first = projs[0]
        unique = tuple(dict.fromkeys(pr.passage for pr in projs))
        facts.append(
            Fact(meeting, first.horizon_pt, first.horizon_en, first.year, first.value, unique, first.sentence)
        )
    return facts, conflicts


def _month(date: str) -> tuple[str, str]:
    y, m = date[:4], int(date[5:7])
    return f"{MONTHS_EN[m - 1]} {y}", f"{MONTHS_PT[m - 1]} de {y}"


def _q(
    kind: str,
    en: str,
    pt: str,
    answer: float | None,
    unit: str,
    tolerance: float,
    evidence: list[dict],
    p: Passage,
    source: str,
) -> dict:
    return {
        "id": "",
        "kind": kind,
        "question_en": en,
        "question_pt": pt,
        "answer": None if answer is None else round(answer, 6),
        "unit": unit,
        "tolerance": tolerance,
        "evidence": evidence,
        "meeting": p.meeting,
        "meeting_date": p.date,
        "source_sentence": source,
    }


def _doc(f: Fact) -> dict:
    p = f.passage
    article = "" if f.horizon_pt.isdigit() else "o "
    return _q(
        "doc",
        f"According to the minutes of the Copom meeting that ended on {p.date}, what was the Committee's inflation projection for {f.horizon_en} in its reference scenario, in %?",
        f"Segundo a ata da reunião do Copom encerrada em {p.date}, qual era a projeção de inflação do Comitê para {article}{f.horizon_pt} no cenário de referência, em %?",
        f.value,
        "%",
        LOOKUP_TOLERANCE,
        [f.evidence()],
        p,
        f.sentence,
    )


def _doc_series(f: Fact, snapshot: Snapshot) -> dict | None:
    p = f.passage
    ipca = snapshot.series.get("br.ipca_12m")
    month = f"{p.date[:7]}-01"
    if ipca is None or month not in ipca.dates:
        return None
    obs = ipca.value_on(month)
    assert obs is not None
    en_month, pt_month = _month(month)
    return _q(
        "doc_series",
        f"In the minutes of the Copom meeting that ended on {p.date}, the Committee gave an inflation projection for {f.horizon_en} in its reference scenario. How far was it from the 12-month IPCA inflation recorded for {en_month}, the month of that meeting? Answer as the projection minus the IPCA, in percentage points.",
        f"Na ata da reunião do Copom encerrada em {p.date}, o Comitê deu uma projeção de inflação para {f.horizon_pt} no cenário de referência. Qual a distância entre ela e o IPCA em 12 meses registrado em {pt_month}, mês dessa reunião? Responda como a projeção menos o IPCA, em pontos percentuais.",
        f.value - obs.value,
        "percentage points",
        DERIVED_TOLERANCE,
        [f.evidence(), {"series_id": obs.series_id, "date": obs.date, "value": obs.value}],
        p,
        f.sentence,
    )


def _doc_absent(f: Fact, covered: set[int]) -> dict:
    p = f.passage
    year = max(covered) + 6
    return _q(
        "doc_absent",
        f"According to the minutes of the Copom meeting that ended on {p.date}, what was the Committee's inflation projection for {year} in its reference scenario, in %?",
        f"Segundo a ata da reunião do Copom encerrada em {p.date}, qual era a projeção de inflação do Comitê para {year} no cenário de referência, em %?",
        None,
        "%",
        0.0,
        [],
        p,
        f.sentence,
    )


def generate_documents(passages: list[Passage], snapshot: Snapshot, seed: int = 13) -> list[dict]:
    """Questions spread over as many meetings as possible: each kind takes one candidate per meeting
    in a shuffled order, then a second one per meeting, and so on."""
    rng = random.Random(seed)
    facts, _ = projection_facts(passages)
    by_meeting: dict[int, list[Fact]] = {}
    for f in facts:
        by_meeting.setdefault(f.meeting, []).append(f)
    out: list[dict] = []
    for kind, count in COUNTS.items():
        order = sorted(by_meeting)
        rng.shuffle(order)
        pools: dict[int, list[dict]] = {}
        for meeting in order:
            group = by_meeting[meeting]
            if kind == "doc_absent":
                candidates = [_doc_absent(group[0], {f.year for f in group})]
            elif kind == "doc":
                candidates = [_doc(f) for f in group]
            else:
                candidates = [q for f in group if f.horizon_pt.isdigit() and (q := _doc_series(f, snapshot))]
            rng.shuffle(candidates)
            pools[meeting] = candidates
        depth = max((len(c) for c in pools.values()), default=0)
        picked = [pools[m][r] for r in range(depth) for m in order if r < len(pools[m])]
        out.extend(picked[:count])
    for i, q in enumerate(out, start=1):
        q["id"] = f"doc-{i:02d}"
        q["as_of"] = snapshot.as_of
    return out


def review_table(questions: list[dict]) -> str:
    """For a person to check every expected answer against the sentence it came from."""
    lines = []
    for q in questions:
        answer = "ABSENT" if q["answer"] is None else q["answer"]
        lines.append(f"{q['id']} {q['kind']:10} meeting {q['meeting']} ({q['meeting_date']}) -> {answer}")
        lines.append(f"    Q: {q['question_en']}")
        lines.append(f"    source: {q['source_sentence'][:400]}")
    return "\n".join(lines)
