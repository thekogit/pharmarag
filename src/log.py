"""Logging goes to stderr only, so stdout stays clean for CLI output and stdio protocols."""

from __future__ import annotations

import logging
import sys


def setup(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(getattr(h, "_pharmarag", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    handler._pharmarag = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "docling", "RapidOCR"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
