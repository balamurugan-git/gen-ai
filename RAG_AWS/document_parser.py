"""
document_parser.py
------------------
Extracts plain text from PDF, DOCX, HTML, and PPTX files.
Returns a list of text chunks ready for embedding.

Supported formats:
  - .pdf   → PyPDF2
  - .docx  → python-docx
  - .pptx  → python-pptx
  - .html  → BeautifulSoup + html2text
"""

import io
import logging
from pathlib import Path

import PyPDF2
from docx import Document as DocxDocument
from pptx import Presentation
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parses binary content of a document into a list of text chunks.

    Usage:
        parser = DocumentParser()
        chunks = parser.parse(file_bytes=b"...", file_extension=".pdf")
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],  # Paragraph → sentence → word
        )

    # ─── Public Entry Point ───────────────────────────────────────────────────

    def parse(self, file_bytes: bytes, file_extension: str) -> list[str]:
        """
        Extract and chunk text from a document.

        Args:
            file_bytes:      Raw bytes of the file (downloaded from S3).
            file_extension:  Lowercase extension e.g. ".pdf", ".docx", ".html", ".pptx"

        Returns:
            List of text chunk strings.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = file_extension.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Parsing document — type={ext}, size={len(file_bytes)} bytes")

        raw_text = self._extract_text(file_bytes, ext)

        if not raw_text or not raw_text.strip():
            logger.warning("No text extracted from document.")
            return []

        chunks = self.splitter.split_text(raw_text)
        logger.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        return chunks

    # ─── Private Extraction Methods ───────────────────────────────────────────

    def _extract_text(self, file_bytes: bytes, ext: str) -> str:
        if ext == ".pdf":
            return self._extract_pdf(file_bytes)
        elif ext == ".docx":
            return self._extract_docx(file_bytes)
        elif ext in (".html", ".htm"):
            return self._extract_html(file_bytes)
        elif ext in (".pptx", ".ppt"):
            return self._extract_pptx(file_bytes)
        else:
            raise ValueError(f"No extractor for extension: {ext}")

    def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF — page by page."""
        text_parts = []
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
            logger.info(f"PDF has {total_pages} pages")

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Clean up excessive whitespace from PDF extraction
                    page_text = " ".join(page_text.split())
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

        return "\n\n".join(text_parts)

    def _extract_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX — paragraph by paragraph, preserving tables."""
        text_parts = []
        try:
            doc = DocxDocument(io.BytesIO(file_bytes))

            # Extract paragraphs (includes headings, body text)
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    text_parts.append(text)

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        text_parts.append(row_text)

        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise

        return "\n\n".join(text_parts)

    def _extract_html(self, file_bytes: bytes) -> str:
        """Extract text from HTML — strips tags, keeps readable structure."""
        try:
            # Detect encoding
            html_str = file_bytes.decode("utf-8", errors="replace")

            soup = BeautifulSoup(html_str, "lxml")

            # Remove script, style, nav, footer noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Get clean text preserving some structure
            text = soup.get_text(separator="\n", strip=True)

            # Collapse multiple blank lines
            import re
            text = re.sub(r"\n{3,}", "\n\n", text)

        except Exception as e:
            logger.error(f"HTML extraction failed: {e}")
            raise

        return text

    def _extract_pptx(self, file_bytes: bytes) -> str:
        """Extract text from PPTX — slide by slide, all text shapes."""
        text_parts = []
        try:
            prs = Presentation(io.BytesIO(file_bytes))
            total_slides = len(prs.slides)
            logger.info(f"PPTX has {total_slides} slides")

            for slide_num, slide in enumerate(prs.slides, start=1):
                slide_texts = []
                for shape in slide.shapes:
                    if not shape.has_text_frame:
                        continue
                    for para in shape.text_frame.paragraphs:
                        text = "".join(run.text for run in para.runs).strip()
                        if text:
                            slide_texts.append(text)

                if slide_texts:
                    slide_content = "\n".join(slide_texts)
                    text_parts.append(f"[Slide {slide_num}]\n{slide_content}")

        except Exception as e:
            logger.error(f"PPTX extraction failed: {e}")
            raise

        return "\n\n".join(text_parts)
