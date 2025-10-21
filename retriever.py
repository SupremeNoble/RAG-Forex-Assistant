"""Utilities for loading the persisted Chroma vector store and creating a retriever.

This module exposes a `get_retriever` helper and a module-level `retriever`
for backward compatibility with existing code that imports `retriever`.
"""

import logging
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def get_retriever(persist_directory: str = "vector_store/", k: int = 5) -> Any:
    """Load the Chroma vector store and return a Retriever with the given k.

    Args:
        persist_directory: Directory where Chroma persisted the vectors.
        k: Number of documents to return for similarity search.

    Returns:
        A LangChain retriever instance.
    """
    # Embedding must match the model used during ingestion
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

    logger.info("Loaded Chroma vector store from %s", persist_directory)
    return vectordb.as_retriever(search_kwargs={"k": k})


# Module-level retriever for backward compatibility
retriever = get_retriever()