# ADR 0006: Options rejected without an experiment

Status: recorded, 2026-11-08

These were rejected by reasoning alone. **No experiment was run**, so no numbers are claimed.

| Option | Why it was not pursued |
|---|---|
| Autonomous multi-agent loops (planner, researcher, critic agents) | Every extra LLM turn is another chance to paraphrase away the verbatim evidence. The pipeline is a fixed graph whose only loop is the bounded grounding retry (at most 2), which keeps traces reproducible and cost predictable. |
| A local generator model | The owner's decision for this project: answers come from Ollama Cloud `beam:cloud`. Only the judge runs locally. So the project does **not** claim zero cloud egress. |
| FP16 KV cache for the local judge | Not measured. Ollama's defaults are used for the Q5_K_M judge, which fits in 12 GB VRAM next to bge-m3. |
| Zero-shot 7B judge as a sole metric | Its agreement with humans is unmeasured until the label sheet is filled (`evals/calibrate.py`). One problem is already measured: on 2026-11-08 the Prometheus-2-7B judge failed all 30 correct out-of-scope refusals on "abstention" (13/30 after a post-hoc rubric fix), and its pairwise verdicts were position-consistent in only 61/104 pairs (59/104 after the fix). Judge numbers are reported next to the deterministic Level 1 checks, never instead of them. |
