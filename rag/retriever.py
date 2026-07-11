"""
rag/retriever.py — ChromaDB retriever for the RAG pipeline.

Phase 4: Provides semantic document search used both as a direct tool
and internally by the LangChain agent chain.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import Tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import config  # noqa: E402


class DocumentRetriever:
    """
    Thin wrapper around ChromaDB for semantic similarity search.

    Provides both a raw retrieval method and a LangChain Tool-compatible
    callable for agent integration.
    """

    def __init__(self) -> None:
        self._vectorstore: Optional[Chroma] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None

    def _ensure_loaded(self) -> None:
        if self._vectorstore is not None:
            return
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "openai_api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_API_BASE:
            kwargs["base_url"] = config.OPENAI_API_BASE
        self._embeddings = OpenAIEmbeddings(**kwargs)
        self._vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        logger.debug("DocumentRetriever connected to ChromaDB at '%s'.", config.CHROMA_PERSIST_DIR)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Run a similarity search and return the top-k documents.

        Args:
            query: Natural language search query.
            k: Number of results to return (defaults to config.RETRIEVER_K).

        Returns:
            List of LangChain Document objects with content and metadata.
        """
        self._ensure_loaded()
        k = k or config.RETRIEVER_K
        try:
            results = self._vectorstore.similarity_search(query, k=k)  # type: ignore[union-attr]
            logger.info("[RAG] Query: '%s' → %d results", query[:80], len(results))
            return results
        except Exception as exc:
            logger.error("[RAG] Retrieval failed: %s", exc)
            return []

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Return (Document, score) pairs for evaluation purposes."""
        self._ensure_loaded()
        k = k or config.RETRIEVER_K
        try:
            return self._vectorstore.similarity_search_with_score(query, k=k)  # type: ignore
        except Exception as exc:
            logger.error("[RAG] Scored retrieval failed: %s", exc)
            return []

    def format_results(self, docs: List[Document]) -> str:
        """Format retrieved documents into a readable string for the agent."""
        if not docs:
            return (
                "No relevant information found in the knowledge base. "
                "I recommend escalating this question to a human analyst."
            )
        parts: List[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f", page {page + 1}" if page != "" else ""
            parts.append(
                f"**[Source {i}: {source}{page_info}]**\n{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def search_and_format(self, query: str) -> str:
        """Search and return formatted results — used as the Tool function."""
        logger.info("[TOOL] DocumentSearch called: '%s'", query[:80])
        docs = self.retrieve(query)
        return self.format_results(docs)

    def as_langchain_tool(self) -> Tool:
        """Return a LangChain Tool wrapping this retriever."""
        return Tool(
            name="search_documents",
            func=self.search_and_format,
            description=(
                "Search the company knowledge base for information about policies, "
                "procedures, reports, handbooks, FAQs, or any business documentation. "
                "Use this tool when you need to find specific company information. "
                "Input should be a natural language search query."
            ),
        )

    def get_langchain_retriever(self):
        """Return the raw LangChain-compatible retriever object."""
        self._ensure_loaded()
        return self._vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})  # type: ignore


# Singleton instance
retriever = DocumentRetriever()
