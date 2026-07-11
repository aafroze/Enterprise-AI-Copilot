"""
config.py — Centralised configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── OpenAI ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = os.getenv("APP_NAME", "Enterprise AI Operations Copilot")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "enterprise_docs"
    )

    # ── RAG ──────────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))

    # ── Agent ─────────────────────────────────────────────────────────────────
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "6"))
    AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.1"))

    # ── Safety ────────────────────────────────────────────────────────────────
    ENABLE_SAFETY_GUARDRAILS: bool = (
        os.getenv("ENABLE_SAFETY_GUARDRAILS", "true").lower() == "true"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")
    LOG_DIR: str = os.path.join(os.path.dirname(__file__), "..", "logs")

    @classmethod
    def validate(cls) -> None:
        """Raise if critical configuration is missing."""
        if not cls.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


config = Config()
