import pytest

from research_agent.retrieval import DenseIndex, RetrievalError, Retriever, tokenize
from research_agent.retrieval_eval import evaluate, format_rows
from tests.conftest import FakeEmbedder


def test_tokenize_strips_accents_and_stopwords():
    assert tokenize("As projeções de inflação: 4,5% em 2026") == ["projecoes", "inflacao", "4", "5", "2026"]


def test_bm25_finds_the_projection_paragraph_and_filters_by_date(passages):
    r = Retriever(passages, "bm25")
    top = r.search("projeções de inflação cenário de referência", k=3)
    assert all(p.paragraph == 3 for p in top)
    only = r.search("projeções de inflação", k=5, from_date="2025-05-07", to_date="2025-05-07")
    assert {p.meeting for p in only} == {270}
    assert r.search("projeções", k=3, from_date="2030-01-01") == []
    assert r.search("   ", k=3) == []
    assert r.search("zebra", k=3) == []  # bm25 returns nothing rather than noise


def test_dense_and_hybrid(tmp_path, passages):
    embedder = FakeEmbedder()
    index = DenseIndex.build(passages, embedder)
    index.save(tmp_path / "idx.json")
    loaded = DenseIndex.load(tmp_path / "idx.json")
    assert loaded.model == "fake-embed" and set(loaded.vectors) == {p.id for p in passages}
    for mode in ("dense", "hybrid"):
        r = Retriever(passages, mode, loaded, embedder)
        target = next(p for p in passages if p.id == "copom-263-p3")
        top = r.search(
            target.header() + ". " + target.text,  # the fake embedder is only reliable on identical text
            k=2,
            from_date="2024-06-19",
            to_date="2024-06-19",
        )
        assert top[0].id == "copom-263-p3"


def test_dense_index_must_match_the_corpus_and_the_model(passages):
    index = DenseIndex.build(passages, FakeEmbedder())
    with pytest.raises(RetrievalError, match="different corpus"):
        Retriever(passages[:-1], "dense", index, FakeEmbedder())
    other = FakeEmbedder()
    other.model = "other"
    with pytest.raises(RetrievalError, match="built with"):
        Retriever(passages, "dense", index, other)
    with pytest.raises(RetrievalError, match="needs"):
        Retriever(passages, "hybrid")
    with pytest.raises(RetrievalError):
        Retriever(passages, "fuzzy")


def test_retrieval_eval(passages):
    questions = [
        {
            "kind": "doc",
            "question_en": "inflation projection reference scenario",
            "question_pt": "projeções de inflação cenário de referência",
            "meeting_date": "2025-05-07",
            "evidence": [{"passage_id": "copom-270-p3"}],
        },
        {
            "kind": "doc_absent",
            "question_en": "x",
            "question_pt": "x",
            "meeting_date": "2025-05-07",
            "evidence": [],
        },
    ]
    rows = evaluate(Retriever(passages, "bm25"), questions)
    by = {(r["lang"], r["meeting_filter"]): r for r in rows}
    assert by[("pt", True)]["recall@1"] == 1.0 and by[("pt", True)]["questions"] == 1
    assert by[("en", False)]["recall@5"] == 0.0  # English words do not match Portuguese text
    assert "R@1" in format_rows(rows)
