"""Download the label PDFs and SPL metadata listed in corpus/manifest.toml into data/raw/."""

from __future__ import annotations

import argparse

from src import log
from src.config import ROOT, Settings
from src.ingestion.dailymed import fetch, load_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()
    log.setup()
    settings = Settings.load()
    metas = fetch(load_manifest(ROOT / "corpus" / "manifest.toml"), settings.raw_dir, force=args.force)
    for m in sorted(metas, key=lambda m: (m.compound, m.effective_date)):
        print(f"{m.compound:13} {m.effective_date}  v{m.spl_version:<3} {m.set_id}  {m.publisher}")


if __name__ == "__main__":
    main()
