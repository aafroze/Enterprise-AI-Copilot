"""
rag/ingest.py — PDF ingestion, chunking, embedding, and ChromaDB persistence.

Phase 4: Implements the RAG knowledge base pipeline.

Usage:
    python -m rag.ingest          # ingest all PDFs in data/
    python -m rag.ingest --reset  # wipe collection and re-ingest
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from loguru import logger

# Ensure project root is on sys.path when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import config  # noqa: E402


def _get_embeddings() -> OpenAIEmbeddings:
    kwargs = {
        "model": config.OPENAI_EMBEDDING_MODEL,
        "openai_api_key": config.OPENAI_API_KEY,
    }
    if config.OPENAI_API_BASE:
        kwargs["base_url"] = config.OPENAI_API_BASE
    return OpenAIEmbeddings(**kwargs)


def _get_vectorstore(embeddings: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def load_documents(data_dir: str) -> list:
    """Load all PDF and .txt files from the data directory."""
    docs = []
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error("Data directory not found: %s", data_dir)
        return docs

    for file_path in sorted(data_path.iterdir()):
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["source_file"] = file_path.name
                docs.extend(file_docs)
                logger.info("Loaded PDF: %s (%d pages)", file_path.name, len(file_docs))
            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["source_file"] = file_path.name
                docs.extend(file_docs)
                logger.info("Loaded TXT: %s", file_path.name)
        except Exception as exc:
            logger.warning("Could not load %s: %s", file_path.name, exc)

    logger.info("Total documents loaded: %d", len(docs))
    return docs


def chunk_documents(documents: list) -> list:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
    return chunks


def ingest(data_dir: str, reset: bool = False) -> Chroma:
    """
    Full ingestion pipeline: load → chunk → embed → persist.

    Args:
        data_dir: Directory containing source PDFs/TXTs.
        reset: If True, clear the existing collection before ingesting.

    Returns:
        The populated Chroma vectorstore.
    """
    config.validate()

    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)

    if reset:
        try:
            vectorstore.delete_collection()
            logger.info("ChromaDB collection reset.")
        except Exception:
            pass
        vectorstore = _get_vectorstore(embeddings)

    documents = load_documents(data_dir)
    if not documents:
        logger.warning("No documents found in %s. Vectorstore may be empty.", data_dir)
        return vectorstore

    chunks = chunk_documents(documents)
    vectorstore.add_documents(chunks)
    logger.success(
        "Ingestion complete. %d chunks stored in ChromaDB at '%s'.",
        len(chunks),
        config.CHROMA_PERSIST_DIR,
    )
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help="Path to the data directory containing PDFs.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing ChromaDB collection before ingesting.",
    )
    args = parser.parse_args()

    logger.add("logs/ingest.log", rotation="10 MB", level="INFO")
    ingest(data_dir=args.data_dir, reset=args.reset)
