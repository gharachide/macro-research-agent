"""The Copom minutes: download from the Central Bank of Brazil and split into paragraphs.

The minutes come from the Central Bank's open-data API (dataset "Documentos do
Copom", ODbL). Each one is HTML with numbered paragraphs (<p class="paragrafo">),
under section headings (<h3 class="secao">). A passage is one numbered paragraph,
so a citation like copom-275-p5 points to something a reader can find in the
published minutes.

The minutes are in Portuguese; the API has no English version.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

API = "https://www.bcb.gov.br/api/servico/sitebcb/copom"
USER_AGENT = "research-agent/0.1 (open-source evaluation project; polite, one request per second)"


class MinutesError(RuntimeError):
    pass


@dataclass(frozen=True)
class Passage:
    id: str
    meeting: int
    date: str  # last day of the meeting (dataReferencia)
    published: str  # publication date of the minutes
    title: str
    section: str
    paragraph: int
    text: str

    def header(self) -> str:
        """What a reader needs to place the paragraph; also indexed with it."""
        return f"Ata da {self.title} do Copom, {self.section}, parágrafo {self.paragraph}"


# --- parsing ---------------------------------------------------------------------------


class _AtaParser(HTMLParser):
    """Numbered paragraphs and the section each one is in. Footnote markers (<sup>) are
    dropped: inside a number they would turn 4,85<sup>2</sup> into 4,852."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[tuple[str, str]] = []  # (section, text)
        self._section = ""
        self._buffer: list[str] | None = None
        self._target: str | None = None  # "p" or "h3"
        self._depth = 0  # nesting inside the current target
        self._sup = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "sup":
            self._sup += 1
        if self._target is None and (
            (tag == "p" and "paragrafo" in classes) or (tag == "h3" and "secao" in classes)
        ):
            self._target, self._buffer, self._depth = tag, [], 0
        elif self._target == tag:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "sup" and self._sup:
            self._sup -= 1
            return
        if self._target != tag:
            return
        if self._depth:
            self._depth -= 1
            return
        text = _clean("".join(self._buffer or []))
        if self._target == "h3":
            self._section = text
        elif text:
            self.paragraphs.append((self._section, text))
        self._target, self._buffer = None, None

    def handle_data(self, data: str) -> None:
        if self._buffer is not None and not self._sup:
            self._buffer.append(data)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_NUMBERED = re.compile(r"^(\d{1,3})\s*[.)]\s+")


def _record(raw: dict) -> dict:
    """The minutes record inside an API response, whichever way it is wrapped."""
    item = raw.get("conteudo", raw) if isinstance(raw, dict) else raw
    if isinstance(item, list):
        item = item[0] if item else {}
    if not isinstance(item, dict) or "nroReuniao" not in item:
        raise MinutesError("not a Copom minutes record")
    return item


def parse_minutes(raw: dict) -> list[Passage]:
    item = _record(raw)
    meeting = int(item["nroReuniao"])
    parser = _AtaParser()
    parser.feed(str(item.get("textoAta") or ""))
    parser.close()
    passages: list[Passage] = []
    seen: set[int] = set()
    for section, text in parser.paragraphs:
        match = _NUMBERED.match(text)
        if not match:  # captions of the table at the end ("Tabela 1"), not part of the numbered text
            continue
        number = int(match.group(1))
        text = text[match.end() :]
        if number in seen:  # a duplicated number would make two passages with one id
            number = max(seen) + 1
        seen.add(number)
        passages.append(
            Passage(
                id=f"copom-{meeting}-p{number}",
                meeting=meeting,
                date=str(item.get("dataReferencia", ""))[:10],
                published=str(item.get("dataPublicacao", ""))[:10],
                title=_clean(str(item.get("titulo", f"{meeting}ª Reunião"))),
                section=section,
                paragraph=number,
                text=text,
            )
        )
    return passages


# --- corpus files ----------------------------------------------------------------------


def save_corpus(passages: list[Passage], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(asdict(p), ensure_ascii=False) for p in passages)
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def load_corpus(path: str | Path) -> list[Passage]:
    passages = [
        Passage(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [p.id for p in passages]
    if len(set(ids)) != len(ids):
        raise MinutesError(f"{path}: duplicated passage ids")
    if not passages:
        raise MinutesError(f"{path}: empty corpus")
    return passages


def build_corpus(raw_dir: str | Path, since: str, until: str) -> list[Passage]:
    """Passages of the minutes of meetings ending on or after `since` and published on or before `until`."""
    passages: list[Passage] = []
    for path in sorted(Path(raw_dir).glob("copom-*.json")):
        minutes = parse_minutes(json.loads(path.read_text(encoding="utf-8")))
        if minutes and minutes[0].date >= since and minutes[0].published <= until:
            passages.extend(minutes)
    passages.sort(key=lambda p: (p.date, p.paragraph))
    return passages


# --- download --------------------------------------------------------------------------


def _get_json(url: str, timeout: float = 60) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        raise MinutesError(f"{url}: {err}") from None


def download(
    out_dir: str | Path,
    since: str,
    *,
    delay: float = 1.0,
    get: Callable[[str], dict] = _get_json,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> list[int]:
    """Save the raw API record of every minutes from the latest back to the first meeting
    ending before `since`. Files already present are not downloaded again."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latest = int(_record(get(f"{API}/atas?quantidade=1"))["nroReuniao"])
    saved, misses = [], 0
    for number in range(latest, 0, -1):
        path = out / f"copom-{number}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                raw = get(f"{API}/atas_detalhes?nro_reuniao={number}")
                _record(raw)
            except MinutesError as err:
                misses += 1
                log(f"meeting {number}: {err}")
                if misses >= 3:
                    raise MinutesError("three meetings in a row could not be read; stopping") from None
                continue
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            sleep(delay)
        misses = 0
        item = _record(raw)
        saved.append(number)
        log(f"meeting {number}: {item.get('dataReferencia', '')[:10]}")
        if str(item.get("dataReferencia", ""))[:10] < since:
            break
    return saved
