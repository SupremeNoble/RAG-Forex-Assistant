"""
gradio_app.py — Gradio interface for AI Research Assistant for Forex Insights
-----------------------------------------------------------------------------

Features:
- Automatically ingests PDFs/EPUBs from docs/ if vector store doesn't exist
- Initializes ChromaDB retriever and LLM (HuggingFacePipeline)
- Gradio UI for questions + answers + source documents
"""

import os
import gradio as gr
from ingest import ingest_docs
from llm import qa_chain  # make sure retriever.py uses the same vector_store/

# ------------------------------
# 1. Ingest docs if needed
# ------------------------------
VECTOR_STORE_DIR = "vector_store/"

if not os.path.exists(VECTOR_STORE_DIR) or len(os.listdir(VECTOR_STORE_DIR)) == 0:
    print("📂 Vector store not found, ingesting documents...")
    ingest_docs(docs_path="docs/", persist_directory=VECTOR_STORE_DIR)
else:
    print("✅ Vector store exists. Skipping ingestion.")

# ------------------------------
# 2. Gradio QA function
# ------------------------------
def answer_question(question: str):
    """
    Passes the user question to the RAG QA chain and returns answer + sources
    """
    if not question.strip():
        return "❌ Please enter a valid question.", ""
    
    result = qa_chain({"query": question})
    answer = result.get("result", "No answer found.")
    sources = [doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])]
    sources_text = "\n".join(sources) if sources else "No sources available."
    
    return answer, sources_text

# ------------------------------
# 3. Build Gradio Interface
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 💹 AI Research Assistant for Forex Insights")
    gr.Markdown("Ask questions about Forex using uploaded PDFs, EPUBs, and market reports.")

    with gr.Row():
        question_input = gr.Textbox(
            label="Enter your question",
            placeholder="e.g., What is the RSI indicator?",
            lines=1
        )
        answer_output = gr.Textbox(label="Answer", lines=5)
        sources_output = gr.Textbox(label="Sources", lines=5)

    question_input.submit(
        answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

# ------------------------------
# 4. Launch App
# ------------------------------
demo.launch()