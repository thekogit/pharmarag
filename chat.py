"""Interactive CLI. Answers go to stdout, logs to stderr."""

from __future__ import annotations

from src import log
from src.pipeline import build_pipeline


def main() -> None:
    log.setup()
    pipe = build_pipeline()
    print(f"PharmaRAG-Diagnostic ({pipe.cfg.name}: {pipe.cfg.mode}, rerank={pipe.cfg.rerank}, "
          f"freshness={pipe.cfg.freshness}, gates={pipe.cfg.gates}, grounding={pipe.cfg.grounding}). "
          "Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            out = pipe.run(q)
            print(f"\n{out['answer']}\n\n[outcome: {out['outcome']}]")


if __name__ == "__main__":
    main()
