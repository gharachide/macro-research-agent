"""Search over the Copom minutes: lexical (BM25), dense (embeddings) and hybrid.

Three modes so that retrieval can be measured on its own, before the agent:

- bm25:   exact words. Strong on numbers and named terms, blind across languages.
- dense:  embeddings from a multilingual model. Matches meaning, also across
          languages (the minutes are in Portuguese; questions may be in English).
- hybrid: reciprocal rank fusion of the two rankings.

Everything is plain Python. The corpus is a few hundred paragraphs, so a linear
scan is fast enough and there is no index server to run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from research_agent.minutes import Passage

MODES = ("bm25", "dense", "hybrid")
RRF_K = 60

STOPWORDS = set(
    """a ao aos as com como da das de do dos e em entre era essa esse esta este foi ha isso mais mas na nas no nos o os
    ou para pela pelas pelo pelos por que se sem ser seu sua sao tambem um uma
    the of and to in on for by with at from is was were be as that this it an or what which""".split()
)


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    """Lowercase, accents removed, words and numbers; stopwords dropped. "4,5%" gives "4" and "5"."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
    return [t for t in re.findall(r"[a-z0-9]+", plain) if t not in STOPWORDS]


def indexed_text(p: Passage) -> str:
    return f"{p.header()}. {p.text}"


def corpus_fingerprint(passages: Sequence[Passage]) -> str:
    digest = hashlib.sha256()
    for p in passages:
        digest.update(f"{p.id}\t{p.text}\n".encode())
    return digest.hexdigest()[:16]


class BM25:
    def __init__(self, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(indexed_text(p))) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                f = doc.get(t, 0)
                if f:
                    s += (
                        self.idf[t]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
                    )
            out.append(s)
        return out


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class DenseIndex:
    """Unit vectors of every passage, built once with an embedding model and saved to a file."""

    def __init__(
        self,
        model: str,
        fingerprint: str,
        vectors: dict[str, list[float]],
        query_prefix: str = "",
        doc_prefix: str = "",
    ):
        self.model, self.fingerprint, self.vectors = model, fingerprint, vectors
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix

    @classmethod
    def build(
        cls, passages: Sequence[Passage], embedder: Embedder, query_prefix: str = "", doc_prefix: str = ""
    ) -> DenseIndex:
        texts = [doc_prefix + indexed_text(p) for p in passages]
        vectors = embedder.embed(texts)
        return cls(
            embedder.model,
            corpus_fingerprint(passages),
            {p.id: _unit(v) for p, v in zip(passages, vectors, strict=True)},
            query_prefix,
            doc_prefix,
        )

    def save(self, path: str | Path) -> None:
        data = {
            "model": self.model,
            "fingerprint": self.fingerprint,
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
            "vectors": {k: [round(x, 6) for x in v] for k, v in self.vectors.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DenseIndex:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            d["model"], d["fingerprint"], d["vectors"], d.get("query_prefix", ""), d.get("doc_prefix", "")
        )


class RetrievalError(ValueError):
    pass


class Retriever:
    def __init__(
        self,
        passages: Sequence[Passage],
        mode: str = "bm25",
        dense: DenseIndex | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if mode not in MODES:
            raise RetrievalError(f"mode must be one of {', '.join(MODES)}")
        if mode != "bm25":
            if dense is None or embedder is None:
                raise RetrievalError(f"mode {mode} needs an embedding index and an embedding model")
            if dense.fingerprint != corpus_fingerprint(passages):
                raise RetrievalError("the embedding index was built from a different corpus; rebuild it")
            if dense.model != embedder.model:
                raise RetrievalError(f"the index was built with {dense.model}, not {embedder.model}")
        self.passages = list(passages)
        self.by_id = {p.id: p for p in self.passages}
        self.mode, self.dense, self.embedder = mode, dense, embedder
        self.bm25 = BM25(self.passages)

    @property
    def first_date(self) -> str:
        return min(p.date for p in self.passages)

    @property
    def last_date(self) -> str:
        return max(p.date for p in self.passages)

    def _dense_scores(self, query: str) -> list[float]:
        assert self.dense is not None and self.embedder is not None
        q = _unit(self.embedder.embed([self.dense.query_prefix + query])[0])
        return [sum(a * b for a, b in zip(q, self.dense.vectors[p.id], strict=True)) for p in self.passages]

    def search(
        self, query: str, k: int = 3, from_date: str | None = None, to_date: str | None = None
    ) -> list[Passage]:
        """The k best passages from meetings ending within [from_date, to_date] (both optional, ISO)."""
        allowed = [
            i
            for i, p in enumerate(self.passages)
            if (from_date is None or p.date >= from_date) and (to_date is None or p.date <= to_date)
        ]
        if not allowed or not query.strip():
            return []
        rankings: list[list[int]] = []
        if self.mode in ("bm25", "hybrid"):
            s = self.bm25.scores(query)
            rankings.append(sorted((i for i in allowed if s[i] > 0), key=lambda i: -s[i]))
        if self.mode in ("dense", "hybrid"):
            s = self._dense_scores(query)
            rankings.append(sorted(allowed, key=lambda i: -s[i]))
        if len(rankings) == 1:
            ranked = rankings[0]
        else:  # reciprocal rank fusion
            fused: dict[int, float] = {}
            for ranking in rankings:
                for rank, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
            ranked = sorted(fused, key=lambda i: -fused[i])
        return [self.passages[i] for i in ranked[:k]]
