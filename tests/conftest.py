"""A synthetic snapshot for the tests. The numbers are made up; real snapshots live in data/snapshots/."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from research_agent.snapshot import parse_snapshot

AS_OF = "2026-10-30"


def _monthly(base: float, amp: float) -> list[list]:
    out, y, m = [], 2023, 1
    for i in range(46):  # 2023-01 .. 2026-10
        out.append([f"{y}-{m:02d}-01", round(base + amp * math.sin(i / 5), 2)])
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _daily(base: float, amp: float) -> list[list]:
    out, d, i = [], date(2023, 1, 2), 0
    while d.isoformat() <= AS_OF:
        if d.weekday() < 5:
            out.append([d.isoformat(), round(base + amp * math.cos(i / 40), 2)])
            i += 1
        d += timedelta(days=1)
    return out


def _series(title: str, unit: str, frequency: str, code: str, points: list[list]) -> dict:
    return {
        "title": title,
        "titlePt": title + " (pt)",
        "unit": unit,
        "frequency": frequency,
        "source": {"name": "Synthetic", "code": code, "url": "https://example.invalid/" + code},
        "points": points,
    }


def raw_snapshot() -> dict:
    return {
        "schema": 1,
        "asOf": AS_OF,
        "generatedAt": AS_OF + "T12:00:00.000Z",
        "generator": "tests",
        "missing": [],
        "errors": [],
        "series": {
            "br.ipca_12m": _series(
                "Brazil IPCA inflation, 12-month change", "%", "monthly", "13522", _monthly(4.5, 1.0)
            ),
            "br.selic_target": _series(
                "Brazil Selic policy rate target", "% a.a.", "daily", "432", _daily(12.0, 2.0)
            ),
            "us.cpi_yoy": _series(
                "US CPI inflation, 12-month change", "%", "monthly", "CPIAUCSL", _monthly(3.0, 0.8)
            ),
            "us.fed_funds": _series(
                "US effective federal funds rate", "% a.a.", "daily", "DFF", _daily(4.5, 0.7)
            ),
            "us.treasury_10y": _series(
                "US Treasury 10-year yield", "% a.a.", "daily", "DGS10", _daily(4.2, 0.5)
            ),
            "us.treasury_2y": _series(
                "US Treasury 2-year yield", "% a.a.", "daily", "DGS2", _daily(4.0, 0.9)
            ),
        },
        "focus": None,
    }


def rich_raw_snapshot() -> dict:
    """The base snapshot plus the series the multi-step questions need."""
    raw = raw_snapshot()
    raw["series"].update(
        {
            "br.ipca_month": _series(
                "Brazil IPCA inflation, monthly change", "%", "monthly", "433", _monthly(0.4, 0.3)
            ),
            "br.ipca_expected_12m": _series(
                "Brazil expected IPCA, next 12 months", "%", "daily", "Focus", _daily(4.0, 0.6)
            ),
            "us.core_cpi_yoy": _series(
                "US core CPI inflation, 12-month change", "%", "monthly", "CPILFESL", _monthly(3.4, 0.5)
            ),
            "us.core_pce_yoy": _series(
                "US core PCE inflation, 12-month change", "%", "monthly", "PCEPILFE", _monthly(2.9, 0.9)
            ),
            "us.spread_10y_2y": _series(
                "US Treasury 10-year minus 2-year spread", "p.p.", "daily", "T10Y2Y", _daily(0.2, 0.6)
            ),
        }
    )
    return raw


@pytest.fixture
def rich_snapshot():
    return parse_snapshot(rich_raw_snapshot())


@pytest.fixture
def raw() -> dict:
    return raw_snapshot()


@pytest.fixture
def snapshot():
    return parse_snapshot(raw_snapshot())


MEETINGS = [  # (number, last day, published, projection for the meeting's year, the next, and the relevant quarter)
    (260, "2024-01-31", "2024-02-06", 3.9, 3.2, 3.1),
    (263, "2024-06-19", "2024-06-25", 4.0, 3.4, 3.2),
    (266, "2024-11-06", "2024-11-12", 4.6, 3.9, 3.6),
    (270, "2025-05-07", "2025-05-13", 5.5, 4.5, 3.7),
    (274, "2025-12-10", "2025-12-16", 4.4, 3.6, 3.2),
    (277, "2026-03-18", "2026-03-24", 3.9, 3.6, 3.3),
    (280, "2026-08-05", "2026-08-11", 3.7, 3.4, 3.1),
    (282, "2026-10-28", "2026-11-03", 3.6, 3.3, 3.0),  # published after the test snapshot's date
]


def _br(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def raw_minutes(number: int, date: str, published: str, a: float, b: float, c: float) -> dict:
    year = int(date[:4])
    html = (
        '<div id="atacompleta"><div id="ataconteudo">'
        '<h3 class="secao">A) Atualização da conjuntura econômica e do cenário do Copom<sup>1</sup></h3>'
        '<p class="paragrafo">1. O ambiente externo se mantém adverso, com a inflação americana em 4,85<sup>2</sup> '
        "e <i>commodities</i> voláteis.</p>"
        '<p class="paragrafo">2. O mercado de trabalho segue dinâmico e a atividade econômica mostra moderação.</p>'
        '<h3 class="secao">B) Cenários e análise de riscos</h3>'
        f'<p class="paragrafo">3. As projeções de inflação do Copom em seu cenário de referência situam-se em {_br(a)}% '
        f"para {year}, {_br(b)}% para {year + 1} e {_br(c)}% para o segundo trimestre de {year + 2}, "
        f"e as projeções para os preços administrados situam-se em 9,0% para {year}. O Comitê avalia os riscos.</p>"
        '<h3 class="secao">C) Decisão de política monetária</h3>'
        f'<p class="paragrafo">4. O Copom decidiu manter a taxa Selic, na reunião {number}, em 15,00% a.a.</p>'
        f'<p class="paragrafo">5. No cenário de referência, as projeções para a inflação acumulada em quatro trimestres '
        f"para {year} e para o segundo trimestre de {year + 2}, atual horizonte relevante de política monetária, são, "
        f"respectivamente, {_br(a)}% e {_br(c)}% e, no cenário alternativo, a projeção é de 9,9%.</p>"
        '<p class="paragrafo"><b>Tabela 1</b></p>'
        '<p class="paragrafo"><b>Projeções de inflação no cenário de referência</b></p>'
        '</div><div id="ata_info"><table class="table"><tr><td>Local</td><td>Brasília</td></tr></table></div>'
        '<h4>Notas de rodapé </h4><hr><p class="notasrodape"><sup>1</sup> Nota sobre a conjuntura.</p></div>'
    )
    return {
        "conteudo": [
            {
                "nroReuniao": number,
                "dataReferencia": date,
                "dataPublicacao": published,
                "titulo": f"{number}ª Reunião - {date}",
                "urlPdfAta": f"https://example.invalid/{number}.pdf",
                "textoAta": html,
            }
        ]
    }


def write_raw_minutes(directory) -> None:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    for m in MEETINGS:
        (directory / f"copom-{m[0]}.json").write_text(json.dumps(raw_minutes(*m)), encoding="utf-8")


@pytest.fixture
def passages(tmp_path):
    from research_agent.minutes import build_corpus

    write_raw_minutes(tmp_path / "raw")
    return build_corpus(tmp_path / "raw", "2023-11-01", AS_OF)


class FakeEmbedder:
    """Deterministic vectors: hashed bag of words. Enough to test the dense path, not to judge it."""

    model = "fake-embed"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        from research_agent.retrieval import tokenize

        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * 32
            for tok in tokenize(t):
                v[sum(map(ord, tok)) % 32] += 1.0
            out.append(v)
        return out
