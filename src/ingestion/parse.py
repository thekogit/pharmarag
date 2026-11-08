"""PDF -> DoclingDocument, cached as JSON so the slow layout + TableFormer pass runs once per label."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _converter() -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # DailyMed PDFs are born-digital: OCR adds noise and time, TableFormer ACCURATE fixes merged cells.
    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    opts.table_structure_options.mode = TableFormerMode.ACCURATE  # type: ignore[attr-defined]
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def parse_pdf(pdf: Path, parsed_dir: Path, *, force: bool = False) -> Any:
    """Return the DoclingDocument for `pdf`, converting it only if no cached JSON exists."""
    from docling_core.types.doc.document import DoclingDocument

    parsed_dir.mkdir(parents=True, exist_ok=True)
    out = parsed_dir / f"{pdf.stem}.docling.json"
    if out.exists() and not force:
        return DoclingDocument.load_from_json(out)
    t0 = time.perf_counter()
    doc = _converter().convert(str(pdf)).document
    doc.save_as_json(out)
    log.info("parsed %s: %d pages in %.0fs", pdf.name, len(doc.pages), time.perf_counter() - t0)
    return doc
