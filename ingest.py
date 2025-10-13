"""
ingest.py — Document ingestion and embedding for RAG Forex Assistant
--------------------------------------------------------------------
Features:
- Supports PDF and EPUB ingestion
- Cleans text (removes titles, TOC, references, headers, etc.)
- Splits into chunks for embedding
- Saves embeddings into ChromaDB
"""

import os
import re
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------------------
# 1. Advanced Cleaning Function
# ------------------------------
def clean_text_advanced(text: str) -> str:
    """Cleans raw text: removes junk lines, TOC, references, etc."""
    lines = text.splitlines()
    clean_lines = []
    skip_references = False

    for line in lines:
        line = line.strip()

        # Skip empty or very short lines
        if len(line.split()) < 4:
            continue

        # Detect start of references or bibliography
        if any(k in line.lower() for k in ["references", "bibliography"]):
            skip_references = True
        if skip_references:
            continue

        # Skip page numbers, headers, etc.
        if re.search(r'(page \d+|chapter \d+|table of contents|figure \d+|copyright)', line, re.IGNORECASE):
            continue

        clean_lines.append(line)

    clean_text = " ".join(clean_lines)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()


# ------------------------------
# 2. EPUB Loader
# ------------------------------
def load_epub_text(epub_path: str) -> str:
    """Extracts text from all XHTML/HTML items in the EPUB."""
    book = epub.read_epub(epub_path)
    all_text = []
    for item in book.get_items():
        if item.media_type in ["application/xhtml+xml", "text/html"]:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text(separator="\n")
            all_text.append(text)
    return "\n".join(all_text)


# ------------------------------
# 3. PDF Loader
# ------------------------------
def load_pdf_text(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join(page.page_content for page in pages)
    return text


# ------------------------------
# 4. Main Ingestion Pipeline
# ------------------------------
def ingest_docs(docs_path="docs/", persist_directory="vector_store/"):
    """Loads, cleans, embeds, and saves documents into ChromaDB."""

    all_texts = []

    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)
        if filename.lower().endswith(".pdf"):
            print(f"📘 Loading PDF: {filename}")
            raw_text = load_pdf_text(filepath)
        elif filename.lower().endswith(".epub"):
            print(f"📗 Loading EPUB: {filename}")
            raw_text = load_epub_text(filepath)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue

        cleaned_text = clean_text_advanced(raw_text)
        if len(cleaned_text.split()) > 50:
            all_texts.append(cleaned_text)

    print(f"\n✅ Loaded {len(all_texts)} documents. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents(all_texts)

    print(f"🧩 Created {len(chunks)} text chunks. Generating embeddings...")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print(f"💾 Vector store saved to {persist_directory}")
    print("✅ Ingestion complete!")


# ------------------------------
# 5. Run directly
# ------------------------------
if __name__ == "__main__":
    ingest_docs()