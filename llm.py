"""LLM and RetrievalQA wiring for the RAG Forex Assistant.

This module loads a locally-persisted, BitsAndBytes-quantized causal LLM
and constructs a LangChain RetrievalQA chain using the project's
retriever. Heavy work (model loading) is performed inside
``create_qa_chain`` and guarded by ``if __name__ == '__main__'`` so
importing the module in tests or other tools doesn't automatically trigger
large downloads or device allocations.
"""

from typing import Optional
import logging
import os

import torch
from langchain.chains import RetrievalQA
from langchain_huggingface.llms import HuggingFacePipeline
from retriever import retriever
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)


logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Return the preferred device string for model loading."""
    if torch.cuda.is_available():
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
        return "cuda"
    logger.warning("GPU not detected — falling back to CPU (slower)")
    return "cpu"


def create_qa_chain(model_path: str = os.path.join("model", "Qwen15-4b")) -> RetrievalQA:
    """Load the local model, create HuggingFacePipeline LLM, and return a RetrievalQA chain.

    Args:
        model_path: Path to the local model directory.

    Returns:
        A configured RetrievalQA chain ready for inference.
    """
    device = _detect_device()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model not found at {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading model from %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    logger.info("QA chain ready")
    return qa


# Keep a module-level qa_chain when the module is run directly.
# Importers that only need the `create_qa_chain` function can avoid
# triggering the heavy model loading.
qa_chain: Optional[RetrievalQA] = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    qa_chain = create_qa_chain()
    # Optionally print to console for direct runs
    print("✅ Local model + RetrievalQA ready")