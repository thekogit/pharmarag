# ADR 0001: Docling table-aware ingestion with a footnote coherence gate

Status: accepted, 2026-11-08

## Context

The previous ingester read DailyMed text with PyMuPDF and split it with a character splitter. A
dosing table and its footnotes ("* Once daily", "† In patients with eGFR < 30") could land in
different chunks, so a retrieved row such as `10 mg QD *` lost the condition that qualifies it.

## Decision

- Parse label PDFs with Docling (layout model plus TableFormer in ACCURATE mode, OCR off because
  DailyMed PDFs are born-digital). Parses are cached in `data/parsed/` (440 s on CPU in total for the
  9 labels).
- Recognise label sections by canonical name (`src/ingestion/sections.py`), not by layout. Docling
  emits every heading at the same level and some numbered headings as plain text.
- Emit each table as one chunk with its title, the header-row fragments that Docling separates from
  it, and the footnotes after it. Consecutive Docling tables are merged, because Docling splits
  tables at page breaks. Long tables are split by rows, repeating the header row.
- Coherence gate: every footnote marker used in a table cell (`* † ‡ § ¶ # …`) must have its footnote
  text in the same chunk. If it doesn't, the gate searches up to 6 following items. Whatever stays
  unresolved is recorded as `coherence_issues` on the chunk, not silently dropped.

## Measured (`python -m scripts.build_index`, 2026-11-08, `evals/results/corpus_report.json`)

| | |
|---|---|
| Chunks (9 labels) | 1,410 (833 from current labels) |
| Table chunks | 120 |
| Table chunks using footnote markers | 60 |
| Still unresolved after the gate | 32 (most often `*` in a header cell, like `HR (95% CI) *`) |

Most unresolved cases are footnote definitions that Docling placed far from the table, or merged with
marker lists (`† ‡ § ¶ # Þ ß` followed by unrelated text). The gate makes these visible. It doesn't
fix them.

## Consequences

Docling pulls in torch and about 1 GB of models. The parse cache keeps that a one-time cost.
