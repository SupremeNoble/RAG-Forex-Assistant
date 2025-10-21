"""Gradio UI entrypoint for the RAG Forex Assistant.

This module will ingest documents into the vector store if needed,
instantiate the QA chain (via `create_qa_chain`) and expose a simple
question -> (answer, sources) UI. The UI layout is unchanged; comments
and formatting were cleaned for readability and best practice.
"""

import logging
import os
import re
from typing import Tuple

import gradio as gr

from ingest import ingest_docs
from llm import create_qa_chain


logger = logging.getLogger(__name__)


VECTOR_STORE_DIR = "vector_store/"


def ensure_vector_store(docs_path: str = "docs/", persist_directory: str = VECTOR_STORE_DIR) -> None:
    """Ensure the Chroma vector store exists by ingesting documents when necessary."""
    if not os.path.exists(persist_directory) or len(os.listdir(persist_directory)) == 0:
        logger.info("Vector store not found; ingesting documents...")
        ingest_docs(docs_path=docs_path, persist_directory=persist_directory)
    else:
        logger.info("Vector store exists; skipping ingestion.")


# Ensure vector store exists before creating the QA chain
ensure_vector_store()

# Create the QA chain once (may be slow) — uses create_qa_chain from llm.py
try:
    qa_chain = create_qa_chain()
except Exception:
    logger.exception("Failed to create QA chain. Make sure model files are available.")
    raise


def answer_question(question: str) -> Tuple[str, str]:
    """Pass the user question to the QA chain and return (answer, sources_text)."""
    if not question or not question.strip():
        return "❌ Please enter a valid question.", ""

    prompt = f"""
    Answer the following question concisely in a single paragraph based only on the retrieved documents.
    If the answer is not known, say "I don't know."

    Question: {question}
    """

    result = qa_chain({"query": question})

    # Safely normalize and extract the model's 'helpful answer' portion.
    raw = result.get("result", "No answer found.")
    answer_text = str(raw).strip()

    # If the chain injects a label like 'Helpful Answer:', use text after it.
    parts = re.split(r"(?i)helpful answer:", answer_text, maxsplit=1)
    if len(parts) > 1:
        answer_text = parts[1].strip()

    # Remove trailing sections that look like a new label (Question:, Sources:, etc.).
    answer_text = re.split(r"(?i)\b(question:|helpful answer:|sources?:)\b", answer_text, maxsplit=1)[0].strip()

    # Prefer the first paragraph; if none, fall back to a short sentence-limited excerpt.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer_text) if p.strip()]
    if paragraphs:
        answer = paragraphs[0]
    else:
        sentences = re.split(r"(?<=[.!?])\s+", answer_text)
        max_sentences = 3
        if len(sentences) > max_sentences:
            answer = " ".join(sentences[:max_sentences]).strip() + "..."
        else:
            answer = answer_text
    # Collect sources while preserving order and deduplicating
    seen = set()
    sources = []
    for doc in result.get("source_documents", []):
        src = getattr(doc, "metadata", {}).get("source", "unknown")
        src = src.strip() if isinstance(src, str) else str(src)
        if src not in seen:
            seen.add(src)
            sources.append(src)

    sources_text = "\n".join(sources) if sources else "No sources available."
    return answer, sources_text


# Build the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 💹 AI Research Assistant for Forex Insights")
    gr.Markdown("Ask questions about Forex using uploaded PDFs, EPUBs, and market reports.")

    with gr.Row():
        question_input = gr.Textbox(
            label="Enter your question",
            placeholder="e.g., What is the RSI indicator?",
            lines=1,
        )
        answer_output = gr.Textbox(label="Answer", lines=5)
        sources_output = gr.Textbox(label="Sources", lines=5)

    question_input.submit(answer_question, inputs=question_input, outputs=[answer_output, sources_output])


# Launch the app
demo.launch(server_name="0.0.0.0", server_port=7860)