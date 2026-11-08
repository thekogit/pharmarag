from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.generation.generator import REFUSAL, decompose, generate_grounded, render
from src.generation.schema import GroundedAnswer, quote_in_passage
from src.retrieval.gates import Conflict
from src.retrieval.index import Hit
from tests.conftest import ScriptedLLM

PASSAGES = ["The recommended dose is 5 mg taken orally twice daily.",
            "Reduce the dose to 2.5 mg twice daily in patients with any 2 of the following characteristics."]
HITS = [Hit(str(i), 0.9, {"text": t, "compound": "apixaban", "publisher": "P", "effective_date": "2025-04-17",
                          "section": "DOSAGE", "subsection": ""}) for i, t in enumerate(PASSAGES)]


def answer(quote: str, passage: int = 1, status: str = "answered", **kw: Any) -> dict[str, Any]:
    return {"status": status, "claims": [{"text": "claim", "citations": [{"passage": passage, "quote": quote}]}], **kw}


def validate(data: dict[str, Any], conflicts: bool = False) -> GroundedAnswer:
    import json
    return GroundedAnswer.model_validate_json(json.dumps(data), context={"passages": PASSAGES,
                                                                        "conflicts": conflicts})


def test_verbatim_quote_passes_and_whitespace_is_the_only_normalisation() -> None:
    validate(answer("5 mg taken orally   twice daily"))
    assert quote_in_passage('"5 mg taken orally twice daily"', PASSAGES[0])
    assert not quote_in_passage("5 mg taken by mouth twice daily", PASSAGES[0])
    assert not quote_in_passage("5 mg", PASSAGES[0])  # too short to be evidence


@pytest.mark.parametrize("data, needle", [
    (answer("5 mg by mouth twice daily"), "not an exact substring"),
    (answer("5 mg taken orally twice daily", passage=3), "does not exist"),
    ({"status": "answered", "claims": []}, "no claims"),
    ({"status": "answered", "claims": [{"text": "c", "citations": []}]}, "no citation"),
    (answer("5 mg taken orally twice daily", status="refused"), "refusing"),
])
def test_invalid_answers_are_rejected(data: dict[str, Any], needle: str) -> None:
    with pytest.raises(ValidationError, match=needle):
        validate(data)


def test_conflicting_context_requires_a_conflict_note() -> None:
    with pytest.raises(ValidationError, match="conflict_note"):
        validate(answer("5 mg taken orally twice daily"), conflicts=True)
    validate(answer("5 mg taken orally twice daily", conflict_note="older label differs"), conflicts=True)
    validate({"status": "refused", "claims": []}, conflicts=True)


def test_retry_loop_feeds_errors_back_until_valid() -> None:
    llm = ScriptedLLM([answer("5 mg by mouth twice daily"), answer("5 mg taken orally twice daily")])
    res = generate_grounded(llm, "dose?", HITS, missing=[], conflicts=[])  # type: ignore[arg-type]
    assert (res.attempts, res.first_attempt_valid, res.salvaged) == (2, False, False)
    assert "not an exact substring" in llm.calls[1][-1]["content"]
    assert res.answer.status == "answered"


def test_exhausted_retries_salvage_verified_claims_or_refuse() -> None:
    mixed = {"status": "answered", "claims": [
        {"text": "good", "citations": [{"passage": 1, "quote": "5 mg taken orally twice daily"}]},
        {"text": "bad", "citations": [{"passage": 2, "quote": "reduce to 2.5 mg always"}]}]}
    res = generate_grounded(ScriptedLLM([mixed] * 3), "q", HITS, missing=[], conflicts=[])  # type: ignore[arg-type]
    assert res.salvaged and [c.text for c in res.answer.claims] == ["good"]
    bad = answer("never in the passage at all")
    res = generate_grounded(ScriptedLLM([bad] * 3), "q", HITS, missing=[], conflicts=[])  # type: ignore[arg-type]
    assert res.answer.status == "refused" and res.attempts == 3


def test_prompt_carries_gate_findings() -> None:
    llm = ScriptedLLM([answer("5 mg taken orally twice daily", conflict_note="n")])
    conflict = Conflict("apixaban", {"chunk_id": "a", "set_id": "n", "effective_date": "2025", "sentence": "S1"},
                        {"chunk_id": "b", "set_id": "o", "effective_date": "2023", "sentence": "S2"}, 0.9)
    generate_grounded(llm, "q", HITS, missing=["price in Poland"], conflicts=[conflict])  # type: ignore[arg-type]
    system = llm.calls[0][0]["content"]
    assert "price in Poland" in system and "CONFLICT DETECTED" in system and '"S2"' in system


def test_render_lists_sources_and_refusal() -> None:
    ans = validate(answer("5 mg taken orally twice daily"))
    text = render(ans, HITS)
    assert "[1]" in text and "effective 2025-04-17" in text
    assert render(GroundedAnswer.model_construct(status="refused", claims=[], not_covered=[],
                                                 conflict_note=None), HITS) == REFUSAL


def test_decompose_parses_json_and_falls_back() -> None:
    assert decompose(ScriptedLLM([{"sub_questions": ["a", "b"]}]), "a and b") == ["a", "b"]  # type: ignore[arg-type]
    assert decompose(ScriptedLLM(["not json"]), "q") == ["q"]  # type: ignore[arg-type]
