"""Command line: the data (series, minutes), the question sets, runs of the agent and their scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent import minutes as mins
from research_agent.docquestions import generate_documents, review_table
from research_agent.llm import LLMError, OpenAICompatEmbedder, client_from_args
from research_agent.minutes import MinutesError
from research_agent.multistep import generate_multistep
from research_agent.questions import generate
from research_agent.retrieval import DenseIndex, RetrievalError, Retriever
from research_agent.retrieval_eval import evaluate, format_rows
from research_agent.runner import run
from research_agent.scoring import score_run
from research_agent.snapshot import SnapshotError, load_snapshot

DOC_KINDS = ("doc", "doc_series", "doc_absent")


def read_jsonl(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _series(args: argparse.Namespace) -> int:
    snap = load_snapshot(args.snapshot)
    print(f"snapshot {snap.as_of}: {len(snap.series)} series")
    for s in snap.series.values():
        print(
            f"  {s.id:26} {s.frequency:8} {len(s.points):5} obs  {s.dates[0]} .. {s.dates[-1]}  {s.source.code}"
        )
    if snap.missing:
        print(f"missing in this snapshot: {', '.join(snap.missing)}")
    return 0


def _retriever(args: argparse.Namespace, mode: str | None = None) -> Retriever | None:
    if not getattr(args, "minutes", None):
        return None
    passages = mins.load_corpus(args.minutes)
    mode = mode or args.retriever
    if mode == "bm25":
        return Retriever(passages, "bm25")
    if not args.index or not args.embed_model:
        raise RetrievalError(f"--retriever {mode} needs --index and --embed-model")
    embedder = OpenAICompatEmbedder(args.embed_base_url, args.embed_model)
    return Retriever(passages, mode, DenseIndex.load(args.index), embedder)


def _add_retrieval_args(p: argparse.ArgumentParser, needed: bool = False) -> None:
    p.add_argument(
        "--minutes", required=needed, help="corpus of the Copom minutes (data/minutes/corpus.jsonl)"
    )
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    p.add_argument("--index", help="embedding index built by `minutes index` (dense and hybrid)")
    p.add_argument("--embed-model", help="embedding model the index was built with")
    p.add_argument("--embed-base-url", default="http://localhost:11434/v1")


def _minutes_download(args: argparse.Namespace) -> int:
    saved = mins.download(args.out, args.since, delay=args.delay)
    print(f"{len(saved)} minutes in {args.out}")
    return 0


def _minutes_build(args: argparse.Namespace) -> int:
    snap = load_snapshot(args.snapshot)
    since = args.since or min(s.dates[0] for s in snap.series.values() if s.frequency == "daily")
    passages = mins.build_corpus(args.raw, since, snap.as_of)
    if not passages:
        raise MinutesError(f"no minutes in {args.raw} between {since} and {snap.as_of}")
    mins.save_corpus(passages, args.out)
    meetings = sorted({p.meeting for p in passages})
    print(
        f"{len(passages)} paragraphs from {len(meetings)} minutes (meetings {meetings[0]}..{meetings[-1]}) -> {args.out}"
    )
    return 0


def _minutes_index(args: argparse.Namespace) -> int:
    passages = mins.load_corpus(args.minutes)
    embedder = OpenAICompatEmbedder(args.base_url, args.model)
    print(f"embedding {len(passages)} paragraphs with {args.model} ...")
    index = DenseIndex.build(passages, embedder, args.query_prefix, args.doc_prefix)
    index.save(args.out)
    dims = len(next(iter(index.vectors.values())))
    print(f"{len(index.vectors)} vectors of {dims} dimensions -> {args.out}")
    return 0


def _retrieval(args: argparse.Namespace) -> int:
    questions = [q for path in args.questions for q in read_jsonl(path) if q.get("kind") in DOC_KINDS]
    modes = ["bm25"] + (["dense", "hybrid"] if args.index else [])
    rows = [row for mode in modes for row in evaluate(_retriever(args, mode), questions)]
    print(format_rows(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


def _generate(args: argparse.Namespace) -> int:
    snap = load_snapshot(args.snapshot)
    if args.set == "documents":
        if not args.minutes:
            raise MinutesError("--set documents needs --minutes")
        out, seed = args.out or "evals/documents.jsonl", 13 if args.seed is None else args.seed
        questions = generate_documents(mins.load_corpus(args.minutes), snap, seed)
        write_jsonl(out, questions)
        print(review_table(questions))
        print(
            f"\n{len(questions)} document questions (seed {seed}) -> {out}. Check every answer above against its source sentence."
        )
        return 0
    if args.set == "multistep":
        out, seed = args.out or "evals/multistep.jsonl", 11 if args.seed is None else args.seed
        questions, wanted = generate_multistep(snap, seed=seed), 30
    else:
        out, seed = args.out or "evals/numeric.jsonl", 7 if args.seed is None else args.seed
        questions, wanted = [q.to_dict() for q in generate(snap, args.n, seed)], args.n
    write_jsonl(out, questions)
    print(f"{len(questions)} {args.set} questions from snapshot {snap.as_of} (seed {seed}) -> {out}")
    if len(questions) < wanted:
        print(f"warning: expected {wanted}; the snapshot is too small for the full set", file=sys.stderr)
    return 0


def _score(args: argparse.Namespace) -> int:
    snap = load_snapshot(args.snapshot)
    questions = [q for path in args.questions for q in read_jsonl(path)]
    passages = None
    if args.minutes:
        passages = {p.id for p in mins.load_corpus(args.minutes)}
    elif any(q.get("kind") in DOC_KINDS for q in questions):
        raise MinutesError("questions about the minutes need --minutes to check cited paragraphs")
    summary, rows = score_run(questions, read_jsonl(args.responses), snap, passages)
    if args.rows:
        write_jsonl(args.rows, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _run(args: argparse.Namespace) -> int:
    snap = load_snapshot(args.snapshot)
    questions = [q for path in args.questions for q in read_jsonl(path)]
    client = client_from_args(args.base_url, args.model, args.api_key_env, args.timeout, args.max_tokens)
    docs = _retriever(args)
    if docs is None and any(q.get("kind") in DOC_KINDS for q in questions):
        raise MinutesError("questions about the minutes need --minutes")
    with_docs = f", minutes ({docs.mode})" if docs else ""
    print(
        f"{args.model} at {args.base_url}: {len(questions)} questions, snapshot {snap.as_of}{with_docs} -> {args.out}"
    )
    answered = run(
        client,
        snap,
        questions,
        args.out,
        lang=args.lang,
        protocol=args.protocol,
        max_steps=args.max_steps,
        limit=args.limit,
        docs=docs,
    )
    print(f"{answered} answered now. Score with: research-agent score --responses {args.out} ...")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("series", help="list the series in a snapshot")
    p.add_argument("--snapshot", required=True)
    p.set_defaults(run=_series)

    p = sub.add_parser("generate", help="generate a question set from a snapshot")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--set", choices=["numeric", "multistep", "documents"], default="numeric")
    p.add_argument("--out", help="default: evals/<set>.jsonl")
    p.add_argument("-n", type=int, default=100, help="size of the numeric set")
    p.add_argument("--seed", type=int, help="default: 7 (numeric), 11 (multistep), 13 (documents)")
    p.add_argument("--minutes", help="corpus of the Copom minutes (documents set)")
    p.set_defaults(run=_generate)

    p = sub.add_parser("score", help="score a run (JSONL of responses) against the questions")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--questions", nargs="+", required=True, help="one or more JSONL files")
    p.add_argument("--responses", required=True)
    p.add_argument("--rows", help="write the per-question results here (JSONL)")
    p.add_argument("--minutes", help="corpus of the Copom minutes, to check cited paragraphs")
    p.set_defaults(run=_score)

    p = sub.add_parser("minutes", help="download, parse and index the Copom minutes")
    msub = p.add_subparsers(dest="minutes_command", required=True)
    m = msub.add_parser("download", help="save the raw minutes from the Central Bank's API")
    m.add_argument("--out", default="data/minutes/raw")
    m.add_argument(
        "--since", default="2023-11-01", help="stop at the first meeting that ended before this date"
    )
    m.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    m.set_defaults(run=_minutes_download)
    m = msub.add_parser("build", help="split the raw minutes into numbered paragraphs")
    m.add_argument("--raw", default="data/minutes/raw")
    m.add_argument(
        "--snapshot", required=True, help="the corpus covers the snapshot's period, up to its date"
    )
    m.add_argument("--since", help="default: the first date of the snapshot's daily series")
    m.add_argument("--out", default="data/minutes/corpus.jsonl")
    m.set_defaults(run=_minutes_build)
    m = msub.add_parser("index", help="embed every paragraph with an embedding model")
    m.add_argument("--minutes", default="data/minutes/corpus.jsonl")
    m.add_argument("--model", required=True, help="embedding model, e.g. embeddinggemma")
    m.add_argument("--base-url", default="http://localhost:11434/v1")
    m.add_argument("--out", required=True)
    m.add_argument("--query-prefix", default="")
    m.add_argument("--doc-prefix", default="")
    m.set_defaults(run=_minutes_index)

    p = sub.add_parser("retrieval", help="measure the search alone: is the right paragraph in the top k?")
    p.add_argument("--questions", nargs="+", required=True)
    _add_retrieval_args(p, needed=True)
    p.add_argument("--out", help="write the table as JSON")
    p.set_defaults(run=_retrieval)

    p = sub.add_parser("run", help="run the agent over the questions with an OpenAI-compatible endpoint")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--questions", nargs="+", required=True, help="one or more JSONL files")
    p.add_argument("--out", required=True, help="responses JSONL; an existing file is resumed")
    p.add_argument("--model", required=True, help="model name as the endpoint knows it")
    p.add_argument("--base-url", default="http://localhost:11434/v1", help="default: a local Ollama")
    p.add_argument("--api-key-env", help="name of the environment variable holding the API key")
    p.add_argument("--lang", choices=["en", "pt"], default="en")
    p.add_argument("--protocol", choices=["auto", "tools", "json"], default="auto")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--limit", type=int, help="answer at most this many questions in this call")
    p.add_argument(
        "--timeout", type=float, default=600, help="seconds to wait for each model call (default 600)"
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        help="cap on each model reply; a reply cut there with no tool call fails the question (length_limit)",
    )
    _add_retrieval_args(p)
    p.set_defaults(run=_run)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (SnapshotError, FileNotFoundError, LLMError, MinutesError, RetrievalError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
