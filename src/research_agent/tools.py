"""The agent's tools: read-only lookups over the snapshot and, optionally, search over the Copom minutes.

A tool never raises to the model. A bad series id, a bad date or an empty window
comes back as {"error": ...} with enough detail for the model to correct itself.
"""

from __future__ import annotations

import difflib
import json
import re

from research_agent.retrieval import Retriever
from research_agent.snapshot import Observation, Series, Snapshot

MAX_OBSERVATIONS = 40
SUBMIT = "submit_answer"
SEARCH = "search_minutes"
MAX_PASSAGES = 5
MAX_PASSAGE_CHARS = 1200

_SERIES = {"type": "string", "description": "A series id from the catalog, for example br.selic_target."}
_DATE = "ISO date YYYY-MM-DD. For monthly series, YYYY-MM is also accepted."
_BOUND = "ISO date YYYY-MM-DD, or YYYY-MM for a whole month."


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SPECS = [
    _fn(
        "get_value",
        "The observation of a series in force on a date: the last one on or before it. Omit the date for the most recent observation.",
        {"series_id": _SERIES, "date": {"type": "string", "description": _DATE}},
        ["series_id"],
    ),
    _fn(
        "get_change",
        "The change of a series between two dates: value in force on end_date minus value in force on start_date.",
        {
            "series_id": _SERIES,
            "start_date": {"type": "string", "description": _DATE},
            "end_date": {"type": "string", "description": _DATE},
        },
        ["series_id", "start_date", "end_date"],
    ),
    _fn(
        "get_extreme",
        "The highest or lowest observation of a series in a window, and the first date it was reached.",
        {
            "series_id": _SERIES,
            "start_date": {"type": "string", "description": _BOUND},
            "end_date": {"type": "string", "description": _BOUND},
            "kind": {"type": "string", "enum": ["max", "min"]},
        },
        ["series_id", "start_date", "end_date", "kind"],
    ),
    _fn(
        "get_observations",
        f"All observations of a series in a window, at most {MAX_OBSERVATIONS}. Use a narrower window or get_extreme for long periods.",
        {
            "series_id": _SERIES,
            "start_date": {"type": "string", "description": _BOUND},
            "end_date": {"type": "string", "description": _BOUND},
        },
        ["series_id", "start_date", "end_date"],
    ),
    _fn(
        SUBMIT,
        "Give the final answer. Call this exactly once, as the last step.",
        {
            "answer": {
                "type": ["number", "null"],
                "description": "The single number that answers the question, in the unit asked. null when refusing.",
            },
            "refused": {
                "type": "boolean",
                "description": "true when the question asks for investment advice or a forecast and you decline.",
            },
            "citations": {
                "type": "array",
                "description": "Every observation the answer depends on, exactly as a tool returned it.",
                "items": {
                    "type": "object",
                    "properties": {
                        "series_id": {"type": "string"},
                        "date": {"type": "string"},
                        "value": {"type": "number"},
                    },
                    "required": ["series_id", "date", "value"],
                },
            },
            "text": {"type": "string", "description": "A short answer in the language of the question."},
        },
        ["answer", "refused", "citations", "text"],
    ),
]


# With the minutes: one more tool, and the answer may cite paragraphs as well as observations.
SEARCH_SPEC = _fn(
    SEARCH,
    "Search the minutes of the Copom meetings (Central Bank of Brazil). Returns numbered paragraphs, "
    "each with a passage_id to cite. The minutes are in Portuguese: write the query in Portuguese. "
    "When the question names a meeting, set from_date and to_date to the day that meeting ended.",
    {
        "query": {"type": "string", "description": "What to look for, in Portuguese."},
        "from_date": {
            "type": "string",
            "description": "Only meetings that ended on or after this date. " + _BOUND,
        },
        "to_date": {
            "type": "string",
            "description": "Only meetings that ended on or before this date. " + _BOUND,
        },
        "k": {"type": "integer", "description": f"How many paragraphs, 1 to {MAX_PASSAGES} (default 3)."},
    },
    ["query"],
)

_DOC_SUBMIT = json.loads(json.dumps(TOOL_SPECS[-1]))
_DOC_SUBMIT["function"]["parameters"]["properties"]["answer"]["description"] = (
    "The single number that answers the question, in the unit asked. "
    "null when refusing, or when neither the series nor the minutes contain the answer."
)
_DOC_SUBMIT["function"]["parameters"]["properties"]["citations"] = {
    "type": "array",
    "description": "Everything the answer depends on: observations (series_id, date, value exactly as returned) "
    "and paragraphs of the minutes (passage_id exactly as returned).",
    "items": {
        "type": "object",
        "properties": {
            "series_id": {"type": "string"},
            "date": {"type": "string"},
            "value": {"type": "number"},
            "passage_id": {"type": "string"},
        },
    },
}


def tool_specs(with_minutes: bool) -> list[dict]:
    """The series tools alone are exactly what the published series-only runs used."""
    return TOOL_SPECS if not with_minutes else [*TOOL_SPECS[:-1], SEARCH_SPEC, _DOC_SUBMIT]


class ToolError(Exception):
    pass


def _date(value: object, month_as: str) -> str:
    """Normalize a date argument. A month (YYYY-MM) becomes its first or last day; '31' sorts after any real day."""
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return f"{text}-{'01' if month_as == 'start' else '31'}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    raise ToolError(f"bad date {text!r}: use YYYY-MM-DD or YYYY-MM")


def _series(snapshot: Snapshot, args: dict) -> Series:
    sid = str(args.get("series_id", "")).strip()
    if sid in snapshot.series:
        return snapshot.series[sid]
    close = difflib.get_close_matches(sid, list(snapshot.series), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ToolError(f"unknown series {sid!r}.{hint}")


def _need(args: dict, *names: str) -> None:
    missing = [n for n in names if args.get(n) in (None, "")]
    if missing:
        raise ToolError(f"missing argument(s): {', '.join(missing)}")


def _point_in_time(series: Series, value: object) -> str:
    """A date for a lookup. A bare month is only accepted for monthly series: on a daily one it
    would silently mean the end of the month, which may be after the date the question asked about."""
    text = str(value).strip()
    if series.frequency != "monthly" and re.fullmatch(r"\d{4}-\d{2}", text):
        raise ToolError(f"{series.id} is {series.frequency}: give a full date YYYY-MM-DD, not a month")
    return _date(text, "end")


def _obs(o: Observation) -> dict:
    return {"series_id": o.series_id, "date": o.date, "value": o.value}


def _window(series: Series, args: dict) -> tuple[str, str]:
    _need(args, "start_date", "end_date")
    start, end = _date(args["start_date"], "start"), _date(args["end_date"], "end")
    if start > end:
        raise ToolError("start_date is after end_date")
    return start, end


def _get_value(snapshot: Snapshot, args: dict) -> dict:
    series = _series(snapshot, args)
    if args.get("date") in (None, ""):
        return {**_obs(series.latest()), "unit": series.unit}
    o = series.value_on(_point_in_time(series, args["date"]))
    if o is None:
        raise ToolError(
            f"{series.id} has no observation on or before that date; it starts on {series.dates[0]}"
        )
    return {**_obs(o), "unit": series.unit}


def _get_change(snapshot: Snapshot, args: dict) -> dict:
    series = _series(snapshot, args)
    _need(args, "start_date", "end_date")
    result = series.change(
        _point_in_time(series, args["start_date"]), _point_in_time(series, args["end_date"])
    )
    if result is None:
        raise ToolError(
            f"{series.id} has no observation on or before one of the dates; it starts on {series.dates[0]}"
        )
    a, b, diff = result
    return {"start": _obs(a), "end": _obs(b), "change": diff, "unit": series.unit}


def _get_extreme(snapshot: Snapshot, args: dict) -> dict:
    series = _series(snapshot, args)
    start, end = _window(series, args)
    kind = str(args.get("kind", ""))
    if kind not in ("max", "min"):
        raise ToolError("kind must be 'max' or 'min'")
    o = series.extreme(start, end, kind)
    if o is None:
        raise ToolError(
            f"{series.id} has no observations in that window ({series.dates[0]} .. {series.dates[-1]})"
        )
    return {**_obs(o), "kind": kind, "unit": series.unit}


def _get_observations(snapshot: Snapshot, args: dict) -> dict:
    series = _series(snapshot, args)
    start, end = _window(series, args)
    obs = series.window(start, end)
    if len(obs) > MAX_OBSERVATIONS:
        raise ToolError(
            f"{len(obs)} observations in that window; the limit is {MAX_OBSERVATIONS}. Narrow it or use get_extreme"
        )
    return {"series_id": series.id, "unit": series.unit, "observations": [[o.date, o.value] for o in obs]}


def _search_minutes(docs: Retriever, args: dict) -> dict:
    _need(args, "query")
    from_date = _date(args["from_date"], "start") if args.get("from_date") not in (None, "") else None
    to_date = _date(args["to_date"], "end") if args.get("to_date") not in (None, "") else None
    if from_date and to_date and from_date > to_date:
        raise ToolError("from_date is after to_date")
    try:
        k = int(args.get("k") or 3)
    except (TypeError, ValueError):
        raise ToolError("k must be an integer") from None
    k = max(1, min(MAX_PASSAGES, k))
    found = docs.search(str(args["query"]), k, from_date, to_date)
    passages = [
        {
            "passage_id": p.id,
            "meeting": p.meeting,
            "date": p.date,
            "section": p.section,
            "text": p.text if len(p.text) <= MAX_PASSAGE_CHARS else p.text[:MAX_PASSAGE_CHARS] + " [...]",
        }
        for p in found
    ]
    if not passages:
        return {
            "passages": [],
            "note": f"nothing found; the minutes cover meetings that ended from {docs.first_date} to {docs.last_date}",
        }
    return {"passages": passages}


_HANDLERS = {
    "get_value": _get_value,
    "get_change": _get_change,
    "get_extreme": _get_extreme,
    "get_observations": _get_observations,
}


def call_tool(snapshot: Snapshot, name: str, arguments: object, docs: Retriever | None = None) -> dict:
    """Run one tool. `docs` enables search_minutes; without it the tool does not exist."""
    known = [*_HANDLERS, *([SEARCH] if docs is not None else []), SUBMIT]
    if name not in _HANDLERS and not (name == SEARCH and docs is not None):
        return {"error": f"unknown tool {name!r}. Tools: {', '.join(known)}"}
    if not isinstance(arguments, dict):
        return {"error": "arguments must be a JSON object"}
    try:
        if name == SEARCH:
            assert docs is not None
            return _search_minutes(docs, arguments)
        return _HANDLERS[name](snapshot, arguments)
    except ToolError as err:
        return {"error": str(err)}


def catalog(snapshot: Snapshot) -> str:
    """One line per series, for the system prompt."""
    return "\n".join(
        f"- {s.id}: {s.title} [{s.unit}, {s.frequency}, {s.dates[0]} to {s.dates[-1]}]"
        for s in snapshot.series.values()
    )
