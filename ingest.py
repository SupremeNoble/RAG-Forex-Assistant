"""Document ingestion utilities for the RAG Forex Assistant.

This module supports loading PDF and EPUB files, cleaning and splitting
their text, creating embeddings via HuggingFace, and persisting a Chroma
vector store. The functions are small and focused so they can be tested
independently.
"""

from typing import List
import logging
import os
import re

from ebooklib import epub
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document


logger = logging.getLogger(__name__)


def clean_text_advanced(text: str) -> str:
    """Return cleaned text suitable for chunking and embedding.

    - Removes short/noisy lines (fewer than 4 words).
    - Strips table-of-contents, headers, figure/page marks, and references.

    The function aims for generally useful heuristics rather than perfect
    document understanding.

    Args:
        text: Raw extracted text from a document.

    Returns:
        A single cleaned string.
    """
    lines = text.splitlines()
    out_lines: List[str] = []
    skip_references = False

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Skip very short lines (likely headings, page numbers, or noise)
        if len(s.split()) < 4:
            continue

        lower = s.lower()
        if any(k in lower for k in ("references", "bibliography")):
            skip_references = True
        if skip_references:
            continue

        if re.search(r"(page \d+|chapter \d+|table of contents|figure \d+|copyright)", s, re.IGNORECASE):
            continue

        out_lines.append(s)

    joined = " ".join(out_lines)
    # Normalize whitespace
    joined = re.sub(r"\s+", " ", joined)
    return joined.strip()


def load_epub_text(epub_path: str) -> str:
    """Extract text from all HTML/XHTML items in an EPUB file.

    Args:
        epub_path: Path to the .epub file.

    Returns:
        Concatenated text extracted from the EPUB's content files.
    """
    book = epub.read_epub(epub_path)
    parts: List[str] = []
    for item in book.get_items():
        if item.media_type in ("application/xhtml+xml", "text/html"):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            parts.append(soup.get_text(separator="\n"))
    return "\n".join(parts)


def load_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyPDFLoader.

    Args:
        pdf_path: Path to the .pdf file.

    Returns:
        Concatenated page text.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join(page.page_content for page in pages)


def ingest_docs(docs_path: str = "docs/", persist_directory: str = "vector_store/") -> None:
    """Load documents, clean and chunk text, embed, and persist to Chroma.

    Args:
        docs_path: Directory containing documents to ingest.
        persist_directory: Directory where Chroma will persist the vector store.
    """
    documents: List[Document] = []

    for filename in sorted(os.listdir(docs_path)):
        filepath = os.path.join(docs_path, filename)

        try:
            if filename.lower().endswith(".pdf"):
                logger.info("Loading PDF: %s", filename)
                raw_text = load_pdf_text(filepath)
            elif filename.lower().endswith(".epub"):
                logger.info("Loading EPUB: %s", filename)
                raw_text = load_epub_text(filepath)
            else:
                logger.info("Skipping unsupported file: %s", filename)
                continue
        except Exception:
            logger.exception("Failed to load %s; skipping", filename)
            continue

        cleaned = clean_text_advanced(raw_text)
        if len(cleaned.split()) > 50:
            documents.append(Document(page_content=cleaned, metadata={"source": filename}))

    logger.info("Loaded %d documents. Splitting into chunks...", len(documents))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    logger.info("Created %d chunks. Generating embeddings...", len(chunks))

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    logger.info("Vector store saved to %s", persist_directory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ingest_docs()