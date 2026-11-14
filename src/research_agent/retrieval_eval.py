"""Measure the search on its own, before the agent: is the right paragraph in the top k?

For every question that cites a paragraph, the question text itself is used as the
query, in English and in Portuguese, with and without restricting the search to the
meeting the question names. The agent writes its own queries, so this is a measure of
the search, not of the agent.
"""

from __future__ import annotations

from research_agent.retrieval import Retriever

KS = (1, 3, 5)
DEPTH = 10


def evaluate(retriever: Retriever, questions: list[dict]) -> list[dict]:
    """One row per (language, filter): recall@k and MRR@10 over the questions with a gold paragraph."""
    targets = [
        (q, {pid for e in q["evidence"] if "passage_id" in e for pid in e.get("any_of") or [e["passage_id"]]})
        for q in questions
    ]
    targets = [(q, gold) for q, gold in targets if gold]
    rows = []
    for lang in ("en", "pt"):
        for filtered in (False, True):
            ranks = []
            for q, gold in targets:
                date = q.get("meeting_date") if filtered else None
                found = retriever.search(q[f"question_{lang}"], DEPTH, date, date)
                rank = next((i for i, p in enumerate(found, start=1) if p.id in gold), None)
                ranks.append(rank)
            n = len(ranks) or 1
            row = {"mode": retriever.mode, "lang": lang, "meeting_filter": filtered, "questions": len(ranks)}
            for k in KS:
                row[f"recall@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
            row["mrr@10"] = round(sum(1 / r for r in ranks if r is not None) / n, 4)
            rows.append(row)
    return rows


def format_rows(rows: list[dict]) -> str:
    header = (
        f"{'mode':7} {'lang':4} {'filter':6} {'n':>3} "
        + " ".join(f"{'R@' + str(k):>6}" for k in KS)
        + f" {'MRR':>6}"
    )
    lines = [header]
    for r in rows:
        recalls = " ".join(f"{r[f'recall@{k}']:6.2f}" for k in KS)
        lines.append(
            f"{r['mode']:7} {r['lang']:4} {('yes' if r['meeting_filter'] else 'no'):6} {r['questions']:>3} {recalls} {r['mrr@10']:6.2f}"
        )
    return "\n".join(lines)
