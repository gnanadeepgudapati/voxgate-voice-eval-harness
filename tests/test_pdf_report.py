from eval_system.report.pdf_report import html_to_pdf_bytes, markdown_to_pdf_bytes


def test_html_to_pdf_bytes_renders_valid_pdf():
    pdf_bytes = html_to_pdf_bytes("<html><body><h1>Hello</h1><p>Some body text.</p></body></html>")

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 100


def test_renders_valid_pdf_bytes():
    pdf_bytes = markdown_to_pdf_bytes("# Hello\n\nSome body text.")

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 100


def test_renders_tables():
    pdf_bytes = markdown_to_pdf_bytes("| A | B |\n|---|---|\n| 1 | 2 |\n")

    assert pdf_bytes.startswith(b"%PDF")
