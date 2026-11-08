"""Fetch label PDFs and SPL metadata from DailyMed (NLM)."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

PDF_URL = "https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?setId={set_id}"
SPL_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
_EFFECTIVE = re.compile(r'<effectiveTime value="(\d{8})')
_VERSION = re.compile(r'<versionNumber value="(\d+)"')
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)


@dataclass(frozen=True)
class LabelRef:
    compound: str
    set_id: str
    publisher: str


@dataclass(frozen=True)
class LabelMeta:
    compound: str
    set_id: str
    publisher: str
    effective_date: str  # ISO yyyy-mm-dd
    spl_version: int
    title: str

    @property
    def pdf_name(self) -> str:
        return f"{self.set_id}.pdf"


def load_manifest(path: Path) -> list[LabelRef]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [LabelRef(**row) for row in data["label"]]


def parse_spl_header(ref: LabelRef, xml: str) -> LabelMeta:
    eff = _EFFECTIVE.search(xml)
    ver = _VERSION.search(xml)
    title = _TITLE.search(xml)
    if not eff:
        raise ValueError(f"No effectiveTime in SPL {ref.set_id}")
    d = eff.group(1)
    clean_title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", title.group(1))).strip() if title else ""
    return LabelMeta(
        compound=ref.compound,
        set_id=ref.set_id,
        publisher=ref.publisher,
        effective_date=f"{d[:4]}-{d[4:6]}-{d[6:]}",
        spl_version=int(ver.group(1)) if ver else 0,
        title=clean_title[:300],
    )


def fetch(refs: list[LabelRef], raw_dir: Path, *, force: bool = False) -> list[LabelMeta]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    metas: list[LabelMeta] = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for ref in refs:
            meta_path = raw_dir / f"{ref.set_id}.json"
            pdf_path = raw_dir / f"{ref.set_id}.pdf"
            if meta_path.exists() and pdf_path.exists() and not force:
                metas.append(LabelMeta(**json.loads(meta_path.read_text(encoding="utf-8"))))
                continue
            xml = client.get(SPL_URL.format(set_id=ref.set_id)).raise_for_status().text
            meta = parse_spl_header(ref, xml)
            pdf = client.get(PDF_URL.format(set_id=ref.set_id)).raise_for_status()
            if not pdf.content.startswith(b"%PDF"):
                raise ValueError(f"DailyMed did not return a PDF for {ref.set_id}")
            pdf_path.write_bytes(pdf.content)
            meta_path.write_text(json.dumps(asdict(meta), indent=1), encoding="utf-8")
            log.info("fetched %s %s (%s, %d KB)", ref.compound, ref.set_id, meta.effective_date,
                     len(pdf.content) // 1024)
            metas.append(meta)
    return metas
