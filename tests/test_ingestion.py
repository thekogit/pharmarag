from __future__ import annotations

from src.ingestion.chunking import Block, chunk_label, coherence_issues
from src.ingestion.dailymed import LabelRef, parse_spl_header
from src.ingestion.sections import canonical_section, classify_heading

META = {"set_id": "s1", "compound": "atorvastatin", "effective_date": "2024-04-15", "publisher": "V"}


def test_canonical_sections_numbered_and_unnumbered() -> None:
    assert canonical_section("INDICATIONS AND USAGE") == "INDICATIONS"
    assert canonical_section("Dosage and Administration") == "DOSAGE"
    assert canonical_section("DOSAGE FORMS AND STRENGTHS") == "DOSAGE_FORMS"
    assert canonical_section("PRECAUTIONS") == "WARNINGS"
    assert canonical_section("WARNING: RISK OF THYROID C-CELL TUMORS") == "BOXED_WARNING"
    assert canonical_section("Clinical trial results") is None


def test_classify_heading_numbered_text_and_subsections() -> None:
    h = classify_heading("4 CONTRAINDICATIONS", is_header=False)
    assert h is not None and (h.kind, h.value) == ("section", "CONTRAINDICATIONS")
    sub = classify_heading("5.3 Hepatic Dysfunction", is_header=True)
    assert sub is not None and sub.kind == "subsection" and sub.section_hint == "WARNINGS"
    assert classify_heading("Take with food.", is_header=False) is None
    zone = classify_heading("FULL PRESCRIBING INFORMATION: CONTENTS* 1 INDICATIONS", is_header=True)
    assert zone is not None and zone.value == "toc"


def _label(*body: Block) -> list[Block]:
    return [Block("header", "FULL PRESCRIBING INFORMATION"), *body]


def test_table_binds_title_fragments_and_footnotes() -> None:
    blocks = _label(
        Block("header", "2 DOSAGE AND ADMINISTRATION"),
        Block("text", "Dose adjustments are listed below."),
        Block("header", "Table 2: Dosage in Renal Impairment"),
        Block("text", "eGFR (mL/min) Dose"),
        Block("table", "| eGFR | Dose |\n|---|---|\n| <30 * | 2.5 mg † |"),
        Block("text", "* Not on dialysis."),
        Block("text", "† Twice daily."),
        Block("text", "Monitor renal function periodically."),
    )
    chunks = chunk_label(blocks, META)
    table = [c for c in chunks if c.kind == "table"]
    assert len(table) == 1
    t = table[0]
    assert t.section == "DOSAGE"
    assert t.text.startswith("Table 2: Dosage in Renal Impairment")
    assert "* Not on dialysis." in t.text and "† Twice daily." in t.text
    assert t.coherence_issues == []
    texts = " ".join(c.text for c in chunks if c.kind == "text")
    assert "Monitor renal function" in texts and "Not on dialysis" not in texts


def test_coherence_gate_searches_ahead_and_flags_what_it_cannot_bind() -> None:
    blocks = _label(
        Block("header", "12 CLINICAL PHARMACOLOGY"),
        Block("table", "| Drug | AUC ratio ‡ | Cmax § |\n|---|---|---|\n| A | 1.2 | 1.1 |"),
        Block("text", "The interaction was studied in healthy volunteers over seven days of dosing."),
        Block("text", "‡ Ratio of geometric means."),
    )
    t = next(c for c in chunk_label(blocks, META) if c.kind == "table")
    assert "‡ Ratio of geometric means." in t.text
    assert t.coherence_issues == ["unbound_footnote:§"]


def test_consecutive_tables_merge_and_long_tables_repeat_header() -> None:
    rows = "\n".join(f"| row{i} | {i} mg |" for i in range(200))
    blocks = _label(
        Block("header", "6 ADVERSE REACTIONS"),
        Block("table", "| Reaction | Rate |\n|---|---|\n" + rows),
        Block("table", "| tail | 1 |\n|---|---|\n| last | 2 |"),
    )
    tables = [c for c in chunk_label(blocks, META) if c.kind == "table"]
    assert len(tables) > 1
    assert all(t.text.splitlines()[0] == "| Reaction | Rate |" for t in tables)
    assert "last" in tables[-1].text


def test_toc_skipped_highlights_tagged_and_product_tail_dropped() -> None:
    blocks = [
        Block("header", "HIGHLIGHTS OF PRESCRIBING INFORMATION"),
        Block("header", "CONTRAINDICATIONS"),
        Block("text", "Acute liver failure or decompensated cirrhosis (4)."),
        Block("header", "FULL PRESCRIBING INFORMATION: CONTENTS* 1 INDICATIONS AND USAGE"),
        Block("header", "10 OVERDOSAGE 11 DESCRIPTION"),
        Block("header", "FULL PRESCRIBING INFORMATION"),
        Block("text", "4 CONTRAINDICATIONS"),
        Block("text", "Acute liver failure or decompensated cirrhosis."),
        Block("header", "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL"),
        Block("text", "NDC 0000-0000 90 tablets"),
    ]
    chunks = chunk_label(blocks, META)
    assert [(c.part, c.section) for c in chunks] == [("highlights", "CONTRAINDICATIONS"),
                                                     ("full", "CONTRAINDICATIONS")]
    assert not any("NDC" in c.text or "OVERDOSAGE" in c.text for c in chunks)


def test_coherence_issues_direct() -> None:
    assert coherence_issues("| a * |", ["* note"]) == []
    assert coherence_issues("| a * | b ¶ |", ["* note"]) == ["unbound_footnote:¶"]


def test_parse_spl_header() -> None:
    xml = ('<document><title>LIPITOR <b>tablets</b></title><effectiveTime value="20090806"/>'
           '<versionNumber value="5"/></document>')
    m = parse_spl_header(LabelRef("atorvastatin", "sid", "P"), xml)
    assert (m.effective_date, m.spl_version, m.title) == ("2009-08-06", 5, "LIPITOR tablets")
