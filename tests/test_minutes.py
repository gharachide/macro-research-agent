import json

import pytest

from research_agent.minutes import (
    MinutesError,
    build_corpus,
    download,
    load_corpus,
    parse_minutes,
    save_corpus,
)
from tests.conftest import AS_OF, MEETINGS, raw_minutes


def test_parse_numbered_paragraphs_sections_and_footnotes():
    passages = parse_minutes(raw_minutes(*MEETINGS[0]))
    assert [p.id for p in passages] == [
        f"copom-260-p{i}" for i in range(1, 6)
    ]  # table captions are not paragraphs
    first = passages[0]
    assert (
        first.section == "A) Atualização da conjuntura econômica e do cenário do Copom"
    )  # no footnote marker
    assert first.text.startswith("O ambiente externo")  # the number is taken out of the text
    assert "4,85 e commodities" in first.text  # the footnote marker is dropped, not glued to the number
    assert (first.meeting, first.date, first.published) == (260, "2024-01-31", "2024-02-06")
    assert passages[2].section == "B) Cenários e análise de riscos"
    assert all(
        "Brasília" not in p.text and "Nota sobre" not in p.text for p in passages
    )  # tables and footnotes skipped


def test_record_may_come_unwrapped():
    item = raw_minutes(*MEETINGS[0])["conteudo"][0]
    assert len(parse_minutes(item)) == 5
    with pytest.raises(MinutesError):
        parse_minutes({"conteudo": []})


def test_build_keeps_the_period_and_round_trips(tmp_path, passages):
    meetings = sorted({p.meeting for p in passages})
    assert meetings == [m[0] for m in MEETINGS if m[2] <= AS_OF]  # 282 was published after the snapshot
    save_corpus(passages, tmp_path / "corpus.jsonl")
    assert load_corpus(tmp_path / "corpus.jsonl") == passages
    assert build_corpus(tmp_path / "raw", "2025-01-01", AS_OF)[0].meeting == 270


def test_load_rejects_duplicates(tmp_path, passages):
    path = tmp_path / "dup.jsonl"
    save_corpus([passages[0], passages[0]], path)
    with pytest.raises(MinutesError):
        load_corpus(path)


def test_download_walks_back_until_before_since_and_skips_saved_files(tmp_path):
    by_number = {m[0]: raw_minutes(*m) for m in MEETINGS}
    requested = []

    def get(url):
        requested.append(url)
        if url.endswith("atas?quantidade=1"):
            return {"conteudo": [{"nroReuniao": 282, "dataReferencia": "2026-10-28"}]}
        number = int(url.rsplit("=", 1)[1])
        if number not in by_number:
            raise MinutesError("404")
        return by_number[number]

    out = tmp_path / "raw"
    out.mkdir()
    (out / "copom-280.json").write_text(json.dumps(by_number[280]), encoding="utf-8")
    saved = download(out, "2026-01-01", get=get, sleep=lambda _: None, log=lambda _: None)
    assert saved == [282, 280, 277, 274]  # 274 ended before 2026-01-01: kept, then the walk stops
    assert not any(url.endswith("=280") for url in requested)
    with pytest.raises(MinutesError, match="three meetings"):
        download(tmp_path / "other", "2020-01-01", get=get, sleep=lambda _: None, log=lambda _: None)
