"""
tools/search_docs.py — Document search tool exposed to the LangChain agent.

This is a thin re-export of the retriever as a standalone Tool so that
Phase 5 (tool calling) can demonstrate correct vs incorrect tool selection.
"""

from rag.retriever import retriever

search_docs_tool = retriever.as_langchain_tool()
