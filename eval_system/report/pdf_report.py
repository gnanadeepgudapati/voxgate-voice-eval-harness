"""Renders a report to PDF. Pure-Python pipeline (HTML -> xhtml2pdf/reportlab)
deliberately instead of weasyprint, which needs system GTK/Pango -- consistent
with this project's preference for pure-Python/ONNX tooling over
native-compiler deps (see docs/ERRORS.md's webrtcvad entry for why that's a
real constraint on this machine).

`html_to_pdf_bytes()` is the primary path -- report.pdf is rendered directly
from report_html.render_html_report()'s output, which has the real layout
control (fixed column widths, badges, zebra striping) report.pdf needs;
`markdown_to_pdf_bytes()` (naive markdown -> HTML -> PDF, no custom CSS) is
kept for callers that only have the plain Markdown report and don't need
that layout control."""
from __future__ import annotations

import io

import markdown as _markdown
from xhtml2pdf import pisa


def html_to_pdf_bytes(html: str) -> bytes:
    buf = io.BytesIO()
    result = pisa.CreatePDF(io.StringIO(html), dest=buf)
    if result.err:
        raise RuntimeError(f"PDF rendering failed with {result.err} error(s)")
    return buf.getvalue()


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    html = _markdown.markdown(markdown_text, extensions=["tables"])
    return html_to_pdf_bytes(html)
