"""Renders the Markdown combined report (report/markdown_report.py) to PDF.
Pure-Python pipeline (markdown -> HTML -> xhtml2pdf/reportlab) deliberately
instead of weasyprint, which needs system GTK/Pango -- consistent with this
project's preference for pure-Python/ONNX tooling over native-compiler deps
(see docs/ERRORS.md's webrtcvad entry for why that's a real constraint on
this machine)."""
from __future__ import annotations

import io

import markdown as _markdown
from xhtml2pdf import pisa


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    html = _markdown.markdown(markdown_text, extensions=["tables"])
    buf = io.BytesIO()
    result = pisa.CreatePDF(io.StringIO(html), dest=buf)
    if result.err:
        raise RuntimeError(f"PDF rendering failed with {result.err} error(s)")
    return buf.getvalue()
