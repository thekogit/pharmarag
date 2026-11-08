"""Canonical label sections and the heading classifier.

Docling flattens every heading to the same level and sometimes emits numbered headings
("4 CONTRAINDICATIONS") as plain text, so headings are recognised by name, not by layout.
Both PLR-format labels (numbered, with Highlights) and older unnumbered labels are handled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Ordered: the first matching pattern wins.
CANONICAL: list[tuple[str, str]] = [
    (r"WARNING:.*|BOXED WARNING.*", "BOXED_WARNING"),
    (r"RECENT MAJOR CHANGES", "RECENT_CHANGES"),
    (r"INDICATIONS (AND|&) USAGE|INDICATIONS", "INDICATIONS"),
    (r"DOSAGE FORMS (AND|&) STRENGTHS", "DOSAGE_FORMS"),
    (r"DOSAGE (AND|&) ADMINISTRATION", "DOSAGE"),
    (r"CONTRAINDICATIONS", "CONTRAINDICATIONS"),
    (r"WARNINGS (AND|&) PRECAUTIONS|WARNINGS|PRECAUTIONS", "WARNINGS"),
    (r"ADVERSE REACTIONS", "ADVERSE_REACTIONS"),
    (r"DRUG INTERACTIONS", "DRUG_INTERACTIONS"),
    (r"USE IN SPECIFIC POPULATIONS", "SPECIFIC_POPULATIONS"),
    (r"DRUG ABUSE (AND|&) DEPENDENCE", "ABUSE"),
    (r"OVERDOSAGE", "OVERDOSAGE"),
    (r"DESCRIPTION", "DESCRIPTION"),
    (r"CLINICAL PHARMACOLOGY", "CLINICAL_PHARMACOLOGY"),
    (r"NONCLINICAL TOXICOLOGY", "NONCLINICAL_TOXICOLOGY"),
    (r"CLINICAL STUDIES", "CLINICAL_STUDIES"),
    (r"REFERENCES", "REFERENCES"),
    (r"HOW SUPPLIED.*", "HOW_SUPPLIED"),
    (r"PATIENT COUNSELING INFORMATION|INFORMATION FOR PATIENTS", "PATIENT_COUNSELING"),
    (r"MEDICATION GUIDE|PATIENT INFORMATION|INSTRUCTIONS FOR USE", "PATIENT_INFO"),
]
_COMPILED = [(re.compile(rf"(?:{p})\s*", re.I), key) for p, key in CANONICAL]

# PLR section numbers, used to recover the section from a subsection number like "5.3".
PLR_NUMBERS: dict[int, str] = {
    1: "INDICATIONS", 2: "DOSAGE", 3: "DOSAGE_FORMS", 4: "CONTRAINDICATIONS", 5: "WARNINGS",
    6: "ADVERSE_REACTIONS", 7: "DRUG_INTERACTIONS", 8: "SPECIFIC_POPULATIONS", 9: "ABUSE",
    10: "OVERDOSAGE", 11: "DESCRIPTION", 12: "CLINICAL_PHARMACOLOGY", 13: "NONCLINICAL_TOXICOLOGY",
    14: "CLINICAL_STUDIES", 15: "REFERENCES", 16: "HOW_SUPPLIED", 17: "PATIENT_COUNSELING",
}

_NUM = re.compile(r"^(\d{1,2})(?:\.(\d{1,2}))?\s+(.*)$")
_PRODUCT_TAIL = re.compile(
    r"^(PACKAGE LABEL|PRINCIPAL DISPLAY PANEL|INGREDIENTS AND APPEARANCE|PRODUCT INFORMATION"
    r"|ACTIVE INGREDIENT/ACTIVE MOIETY)", re.I)

Zone = Literal["body", "highlights", "toc", "full", "tail"]


@dataclass(frozen=True)
class Heading:
    kind: Literal["zone", "section", "subsection"]
    value: str  # zone name, canonical section key, or subsection title
    section_hint: str | None = None  # for numbered subsections, the PLR section they belong to


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonical_section(name: str) -> str | None:
    n = _norm(name).rstrip(":").upper()
    for rx, key in _COMPILED:
        if rx.fullmatch(n):
            return key
    return None


def classify_heading(text: str, *, is_header: bool) -> Heading | None:
    """Classify a block. Plain-text blocks only count when they are an exact canonical name."""
    t = _norm(text)
    up = t.upper()
    if up.startswith("HIGHLIGHTS OF PRESCRIBING INFORMATION"):
        return Heading("zone", "highlights")
    if up.startswith(("FULL PRESCRIBING INFORMATION: CONTENTS", "FULL PRESCRIBING INFORMATION CONTENTS")):
        return Heading("zone", "toc")
    if up.rstrip(":") == "FULL PRESCRIBING INFORMATION":
        return Heading("zone", "full")
    if _PRODUCT_TAIL.match(t):
        return Heading("zone", "tail")
    if len(t) > 120:
        return None
    m = _NUM.match(t)
    if m and m.group(2) is not None:  # "5.3 Hepatic Dysfunction"
        hint = PLR_NUMBERS.get(int(m.group(1)))
        return Heading("subsection", t, section_hint=hint)
    name = m.group(3) if m else t
    key = canonical_section(name)
    if key is not None:
        return Heading("section", key)
    if is_header and not t.lower().startswith("table "):
        return Heading("subsection", t)
    return None
