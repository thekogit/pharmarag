"""Answer generation: the grounded generator (validated, with bounded retries) and the baseline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.generation.llm import LLM
from src.generation.schema import Claim, GroundedAnswer, quote_in_passage
from src.retrieval.gates import Conflict
from src.retrieval.index import Hit

log = logging.getLogger(__name__)

MAX_RETRIES = 2
REFUSAL = "The provided FDA labels do not contain sufficient information to answer this question."

DECOMPOSE_PROMPT = """Split the user's question about drug labels into the minimal list of independent \
sub-questions that must each be answered. A single-aspect question stays as one item. At most 4 items. \
Keep drug names in every sub-question. Return JSON: {"sub_questions": ["..."]}"""

GROUNDED_PROMPT = """You answer questions about FDA drug labels using ONLY the numbered passages below.

Rules:
- Every claim needs at least one citation: the passage number and a quote copied VERBATIM from that passage \
(an exact substring, at least 12 characters, no paraphrase, no ellipsis).
- If the passages do not answer the question, return status "refused" with no claims.
- If only part is answered, return status "partial" and list the unanswered parts in not_covered.
{coverage_note}{conflict_note}
Return JSON only:
{{"status": "answered" | "partial" | "refused",
  "claims": [{{"text": "...", "citations": [{{"passage": 1, "quote": "..."}}]}}],
  "not_covered": ["..."],
  "conflict_note": null or "..."}}

PASSAGES:
{passages}"""

BASELINE_PROMPT = """You are a regulatory affairs specialist. Answer the question using only the numbered \
passages below. Cite passages inline as [n]. If the passages do not contain the answer, reply exactly: \
"{refusal}"

PASSAGES:
{passages}"""


def format_passages(hits: list[Hit]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        p = h.payload
        sub = f" / {p['subsection']}" if p.get("subsection") else ""
        out.append(f"[{i}] {p['compound']} label ({p['publisher']}, effective {p['effective_date']}) "
                   f"- {p['section']}{sub}\n{h.text}")
    return "\n\n".join(out)


def decompose(llm: LLM, question: str) -> list[str]:
    try:
        raw = llm.chat([{"role": "system", "content": DECOMPOSE_PROMPT},
                        {"role": "user", "content": question}], json_mode=True, max_tokens=400)
        subs = [s.strip() for s in json.loads(raw).get("sub_questions", []) if str(s).strip()]
        return subs[:4] or [question]
    except (json.JSONDecodeError, AttributeError) as e:
        log.warning("decomposition failed (%s); using the question as is", e)
        return [question]


@dataclass
class GroundedResult:
    answer: GroundedAnswer
    attempts: int
    first_attempt_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    salvaged: bool = False


def _salvage(raw: str, passages: list[str]) -> GroundedAnswer | None:
    """After the last retry, keep only the claims whose citations all verify."""
    try:
        data = json.loads(raw)
        claims = [Claim.model_validate(c) for c in data.get("claims", [])]
    except (json.JSONDecodeError, ValidationError, AttributeError):
        return None
    good = [c for c in claims if c.citations and all(
        1 <= x.passage <= len(passages) and quote_in_passage(x.quote, passages[x.passage - 1])
        for x in c.citations)]
    if not good:
        return None
    return GroundedAnswer.model_construct(status="partial", claims=good,
                                          not_covered=list(data.get("not_covered") or []),
                                          conflict_note=data.get("conflict_note"))


def generate_grounded(llm: LLM, question: str, hits: list[Hit], *, missing: list[str],
                      conflicts: list[Conflict]) -> GroundedResult:
    passages = [h.text for h in hits]
    coverage_note = ("- The retrieval gate found NO evidence for these parts; list them in not_covered and do "
                     f"not answer them: {missing}\n") if missing else ""
    conflict_note = ""
    if conflicts:
        lines = [f"  * {c.compound}: label effective {c.newer['effective_date']} says \"{c.newer['sentence']}\" "
                 f"but label effective {c.older['effective_date']} says \"{c.older['sentence']}\""
                 for c in conflicts[:5]]
        conflict_note = ("- CONFLICT DETECTED between label versions. Prefer the newer label, and explain the "
                         "difference in conflict_note:\n" + "\n".join(lines) + "\n")
    messages = [
        {"role": "system", "content": GROUNDED_PROMPT.format(coverage_note=coverage_note,
                                                             conflict_note=conflict_note,
                                                             passages=format_passages(hits))},
        {"role": "user", "content": question},
    ]
    ctx = {"passages": passages, "conflicts": bool(conflicts)}
    errors: list[str] = []
    raw = ""
    for attempt in range(1, MAX_RETRIES + 2):
        raw = llm.chat(messages, json_mode=True)
        try:
            ans = GroundedAnswer.model_validate_json(raw, context=ctx)
            return GroundedResult(ans, attempt, attempt == 1, errors)
        except ValidationError as e:
            msg = "; ".join(str(err["msg"]) for err in e.errors())
            errors.append(msg)
            log.info("grounding validation failed (attempt %d): %s", attempt, msg[:300])
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Your JSON failed validation: {msg}\nFix every error "
                                                     "and return the corrected JSON only."}]
    salvaged = _salvage(raw, passages)
    if salvaged is not None:
        return GroundedResult(salvaged, MAX_RETRIES + 1, False, errors, salvaged=True)
    return GroundedResult(GroundedAnswer.model_construct(status="refused", claims=[], not_covered=[],
                                                         conflict_note=None),
                          MAX_RETRIES + 1, False, errors)


def render(ans: GroundedAnswer, hits: list[Hit]) -> str:
    if ans.status == "refused":
        return REFUSAL
    lines = []
    for c in ans.claims:
        refs = "".join(f"[{x.passage}]" for x in c.citations)
        lines.append(f"- {c.text} {refs}")
    if ans.not_covered:
        lines.append("\nNot covered by the labels: " + "; ".join(ans.not_covered))
    if ans.conflict_note:
        lines.append(f"\nLabel version conflict: {ans.conflict_note}")
    lines.append("\nSources:")
    used = sorted({x.passage for c in ans.claims for x in c.citations})
    for n in used:
        p = hits[n - 1].payload
        lines.append(f"[{n}] {p['compound']} - {p['publisher']}, effective {p['effective_date']}, {p['section']}")
    return "\n".join(lines)


def generate_baseline(llm: LLM, question: str, hits: list[Hit]) -> str:
    if not hits:
        return REFUSAL
    return llm.chat([
        {"role": "system", "content": BASELINE_PROMPT.format(refusal=REFUSAL, passages=format_passages(hits))},
        {"role": "user", "content": question}])


def cited_passages(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
