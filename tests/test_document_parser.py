import io

import pytest
from fpdf import FPDF
from pypdf import PdfWriter

from app.document_parser import DocumentParser, PdfParser, get_parser


def _make_pdf(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.cell(text=text)
    return bytes(pdf.output())


# PdfParser.parse
class TestPdfParser:
    def test_extracts_text_from_single_page(self):
        parser = PdfParser()
        pdf_bytes = _make_pdf(["Hello world"])
        result = parser.parse(pdf_bytes, "test.pdf")
        assert "Hello world" in result

    def test_extracts_text_from_multiple_pages(self):
        parser = PdfParser()
        pdf_bytes = _make_pdf(["Page one", "Page two"])
        result = parser.parse(pdf_bytes, "doc.pdf")
        assert "Page one" in result
        assert "Page two" in result

    def test_handles_empty_page(self):
        parser = PdfParser()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        result = parser.parse(buf.getvalue(), "blank.pdf")
        assert result.strip() == ""

    def test_satisfies_protocol(self):
        parser = PdfParser()
        assert isinstance(parser, DocumentParser)


# get_parser
class TestGetParser:
    def test_returns_pdf_parser_for_pdf(self):
        parser = get_parser("report.pdf")
        assert isinstance(parser, PdfParser)

    def test_case_insensitive_extension(self):
        parser = get_parser("REPORT.PDF")
        assert isinstance(parser, PdfParser)

    def test_raises_for_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_parser("data.csv")

    def test_raises_for_no_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_parser("noextension")
