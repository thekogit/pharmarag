# ADR 0005: Verbatim grounding with a Pydantic v2 validator and bounded retries

Status: accepted, 2026-11-08

## Decision

The generator returns JSON matching `GroundedAnswer` (`src/generation/schema.py`). An
`@model_validator(mode="after")` receives the retrieved passages through Pydantic's validation context
and rejects the answer unless:

- every claim cites at least one passage that exists,
- every citation's quote is an exact substring of that passage (only runs of whitespace are
  collapsed) and is at least 12 characters long,
- a refusal carries no claims, and
- a `conflict_note` is present when the contradiction gate flagged the context.

On failure, the validation errors go back to the model, with at most 2 retries. After the last retry,
claims whose citations all verify are kept (reported as "salvaged"). If none verify, the answer is a
refusal. Nothing unverified is ever shown.

## Measured (2026-11-08, `evals/results/level1.json`)

| | diagnostic | diagnostic, freshness off |
|---|---|---|
| Final quotes that verify | 177/177 | 222/222 |
| Answers valid on the first attempt | 81/85 | 82/88 |
| Mean attempts | 1.06 | 1.08 |
| Salvaged after retries | 1 | 0 |

100% of final quotes verify by construction: the validator does not let an unverified quote through.
The informative numbers are the first-attempt rate and the cost. Two answerable Zoloft questions were
refused after 3 attempts, because the model kept reformatting rows of a markdown dosing table instead
of copying them. Quoting tables verbatim is the main weakness of this design.
