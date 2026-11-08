"""The grounded answer contract. Validation needs the retrieved passages, passed as context:

    GroundedAnswer.model_validate_json(raw, context={"passages": [...], "conflicts": bool})

Every claim must cite at least one passage with a quote that is an exact substring of that passage.
"Exact" means character-for-character after collapsing runs of whitespace; nothing else is normalised.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

MIN_QUOTE_CHARS = 12


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def quote_in_passage(quote: str, passage: str) -> bool:
    q = squash(quote).strip("\"'“”‘’ ")
    return len(q) >= MIN_QUOTE_CHARS and q in squash(passage)


class Citation(BaseModel):
    passage: int = Field(description="1-based passage number from the context")
    quote: str = Field(description="verbatim span copied from that passage")


class Claim(BaseModel):
    text: str
    citations: list[Citation]


class GroundedAnswer(BaseModel):
    status: Literal["answered", "partial", "refused"]
    claims: list[Claim] = Field(default_factory=list)
    not_covered: list[str] = Field(default_factory=list)
    conflict_note: str | None = None

    @model_validator(mode="after")
    def verify_grounding(self, info: ValidationInfo) -> GroundedAnswer:
        ctx = info.context or {}
        passages: list[str] = ctx.get("passages", [])
        errors: list[str] = []
        if self.status == "refused":
            if self.claims:
                errors.append("status is 'refused' but claims were given; give no claims when refusing")
        elif not self.claims:
            errors.append(f"status is '{self.status}' but there are no claims")
        for ci, claim in enumerate(self.claims, 1):
            if not claim.citations:
                errors.append(f"claim {ci} has no citation")
            for cit in claim.citations:
                if not 1 <= cit.passage <= len(passages):
                    errors.append(f"claim {ci} cites passage {cit.passage}, which does not exist")
                elif not quote_in_passage(cit.quote, passages[cit.passage - 1]):
                    errors.append(
                        f"claim {ci}: quote {cit.quote[:80]!r} is not an exact substring of passage "
                        f"{cit.passage} (or is shorter than {MIN_QUOTE_CHARS} characters); copy it verbatim")
        if ctx.get("conflicts") and self.status != "refused" and not self.conflict_note:
            errors.append("the context contains conflicting label versions; conflict_note must describe them")
        if errors:
            raise ValueError("; ".join(errors))
        return self
