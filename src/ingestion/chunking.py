"""Section-aware, table-aware chunking with a cross-chunk coherence gate.

A dosage or interaction table is only correct together with its title and footnotes: "10 mg QD *"
means nothing if the chunk that says "* Once daily, take with food" lands somewhere else. Docling
emits titles, header-row fragments and footnotes as separate items around the table, so this module
binds them back to the table and then checks the result (see `coherence_issues`).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from src.ingestion.sections import classify_heading

FOOTNOTE_MARKERS = "*†‡§¶#Þßàèðø"
_MARKER_START = re.compile(rf"^\s*[{re.escape(FOOTNOTE_MARKERS)}]")
_TABLE_TITLE = re.compile(r"^\s*Table\s+\d+[A-Za-z]?\s*[:.]", re.I)

TEXT_CHUNK_CHARS = 1200
TABLE_CHUNK_CHARS = 2400
TITLE_LOOKBACK = 3
TRAILING_WINDOW = 4
FOOTNOTE_SEARCH_WINDOW = 6


@dataclass
class Block:
    kind: Literal["header", "text", "table", "footnote"]
    text: str  # for tables: markdown


@dataclass
class Chunk:
    chunk_id: str
    set_id: str
    compound: str
    effective_date: str
    publisher: str
    section: str
    subsection: str
    part: Literal["highlights", "full"]
    kind: Literal["text", "table"]
    text: str
    coherence_issues: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def blocks_from_docling(doc: Any) -> list[Block]:
    """Flatten a DoclingDocument into blocks in reading order."""
    from docling_core.types.doc.document import TableItem  # type: ignore[attr-defined]

    blocks: list[Block] = []
    for item, _level in doc.iterate_items():
        label = str(item.label)
        if isinstance(item, TableItem):
            md = item.export_to_markdown(doc).strip()
            for ref in item.footnotes:
                fn = ref.resolve(doc)
                md += "\n" + getattr(fn, "text", "")
            if md:
                blocks.append(Block("table", md))
        elif label in {"section_header", "title"}:
            blocks.append(Block("header", item.text))
        elif label == "footnote":
            blocks.append(Block("footnote", item.text))
        elif label in {"text", "list_item", "paragraph", "caption"} and item.text.strip():
            blocks.append(Block("text", item.text))
    return blocks


def used_markers(table_md: str) -> set[str]:
    return {m for m in FOOTNOTE_MARKERS if m in table_md}


def coherence_issues(table_md: str, bound: list[str]) -> list[str]:
    """Markers used in table cells whose footnote text is not in the same chunk."""
    joined = "\n".join(bound)
    return [f"unbound_footnote:{m}" for m in sorted(used_markers(table_md)) if m not in joined]


def _split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(•])", text)
    out: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _split_table(md: str, limit: int) -> list[str]:
    """Split a long markdown table by rows, repeating the header row in every part."""
    if len(md) <= limit:
        return [md]
    lines = md.splitlines()
    head, rows = lines[:2], lines[2:]
    parts: list[str] = []
    cur: list[str] = []
    for row in rows:
        if cur and len("\n".join(head + cur + [row])) > limit:
            parts.append("\n".join(head + cur))
            cur = []
        cur.append(row)
    if cur:
        parts.append("\n".join(head + cur))
    return parts


@dataclass
class _State:
    zone: str = "body"
    section: str = "PREAMBLE"
    subsection: str = ""


def chunk_label(blocks: list[Block], meta: dict[str, str]) -> list[Chunk]:
    """Turn a label's blocks into chunks. `meta` needs set_id, compound, effective_date, publisher."""
    st = _State()
    chunks: list[Chunk] = []
    buf: list[str] = []
    consumed: set[int] = set()

    def emit(kind: Literal["text", "table"], text: str, issues: list[str] | None = None) -> None:
        if st.zone in {"toc", "tail"} or st.section == "PREAMBLE" or not text.strip():
            return
        n = len(chunks)
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta['set_id']}#{n}")),
            set_id=meta["set_id"], compound=meta["compound"], effective_date=meta["effective_date"],
            publisher=meta["publisher"], section=st.section, subsection=st.subsection,
            part="highlights" if st.zone == "highlights" else "full",
            kind=kind, text=text.strip(), coherence_issues=issues or [],
        ))

    def flush() -> None:
        if buf:
            for piece in _split_text(" ".join(buf), TEXT_CHUNK_CHARS):
                emit("text", piece)
            buf.clear()

    def build_text_chunks(i: int) -> None:
        text = blocks[i].text.strip()
        if buf and len(" ".join(buf)) + len(text) > TEXT_CHUNK_CHARS:
            flush()
        buf.append(text)

    for i, b in enumerate(blocks):
        if i in consumed:
            continue
        if b.kind != "table" and _TABLE_TITLE.match(b.text):
            t = _next_table(blocks, i)
            if t is not None:  # title + header-row fragments belong to the table
                consumed.update(range(i, t))
                continue
        if b.kind in {"header", "text"}:
            h = classify_heading(b.text, is_header=b.kind == "header")
            if h is not None:
                flush()
                if h.kind == "zone":
                    st.zone, st.section, st.subsection = h.value, "PREAMBLE", ""
                elif h.kind == "section":
                    st.section, st.subsection = h.value, ""
                    if st.zone == "body":
                        st.zone = "full"
                else:
                    if h.section_hint and st.zone != "highlights":
                        st.section = h.section_hint
                    st.subsection = h.value
                continue
        if st.zone == "tail":
            break
        if b.kind == "table":
            flush()
            _emit_table(blocks, i, consumed, emit)
            continue
        build_text_chunks(i)
    flush()
    return chunks


def _next_table(blocks: list[Block], i: int) -> int | None:
    for j in range(i + 1, min(i + 1 + TITLE_LOOKBACK, len(blocks))):
        if blocks[j].kind == "table":
            return j
    return None


def _emit_table(blocks: list[Block], i: int, consumed: set[int], emit: Any) -> None:
    # Title and stray header-row fragments between the title and the table.
    preamble: list[str] = []
    for j in range(i - 1, max(-1, i - 1 - TITLE_LOOKBACK), -1):
        if blocks[j].kind == "table":
            break
        if _TABLE_TITLE.match(blocks[j].text):
            preamble = [blocks[k].text.strip() for k in range(j, i)]
            break
    # Consecutive tables are one table that Docling split across a page or a header row.
    body = [blocks[i].text]
    k = i + 1
    while k < len(blocks) and blocks[k].kind == "table":
        body.append(blocks[k].text)
        consumed.add(k)
        k += 1
    md = "\n".join(body)
    # Trailing footnotes: marker-led items, Docling footnotes and short orphan fragments.
    notes: list[str] = []
    j = k
    while j < len(blocks) and j < k + TRAILING_WINDOW:
        nb = blocks[j]
        is_heading = classify_heading(nb.text, is_header=nb.kind == "header") is not None
        if nb.kind == "table" or (is_heading and not _MARKER_START.match(nb.text)):
            break
        if nb.kind == "footnote" or _MARKER_START.match(nb.text) or len(nb.text.strip()) <= 25:
            notes.append(nb.text.strip())
            consumed.add(j)
            j += 1
            continue
        break
    # Coherence gate: look further ahead for footnotes of markers still unexplained.
    missing = [m for m in used_markers(md) if m not in "\n".join(preamble + notes)]
    for jj in range(j, min(j + FOOTNOTE_SEARCH_WINDOW, len(blocks))):
        nb = blocks[jj]
        if nb.kind == "table" or (nb.kind == "header" and classify_heading(nb.text, is_header=True)):
            break
        hit = [m for m in missing if re.search(rf"(^|\s){re.escape(m)}\s*[A-Za-z]", nb.text)]
        if hit:
            notes.append(nb.text.strip())
            consumed.add(jj)
            missing = [m for m in missing if m not in hit]
        if not missing:
            break
    parts = _split_table(md, TABLE_CHUNK_CHARS)
    head = "\n".join(preamble)
    foot = "\n".join(notes)
    for part in parts:
        text = "\n".join(x for x in (head, part, foot) if x)
        emit("table", text, coherence_issues(part, preamble + notes))
