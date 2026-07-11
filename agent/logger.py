"""
agent/logger.py — Structured logging configuration for Phase 8.

Uses loguru for rich logging with:
  - Console output (INFO+)
  - Rotating file log (DEBUG+)
  - PII-safe output filter
  - Latency tracking
  - JSON-structured entries for tracing
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from agent.config import config
from agent.safety import is_pii_safe


LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_FILE = os.path.join(LOG_DIR, "agent.log")
_INTERACTION_FILE = os.path.join(LOG_DIR, "interactions.jsonl")

# Remove default loguru handler
logger.remove()

# Console handler — colourised, INFO and above
logger.add(
    sys.stdout,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan> | "
        "{message}"
    ),
    colorize=True,
    filter=lambda record: is_pii_safe(record["message"]),
)

# File handler — rotating, DEBUG and above
logger.add(
    _LOG_FILE,
    level="DEBUG",
    rotation="10 MB",
    retention="14 days",
    compression="gz",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
    filter=lambda record: is_pii_safe(record["message"]),
    encoding="utf-8",
)


def log_interaction(
    session_id: str,
    user_input: str,
    answer: str,
    tools_used: list,
    latency_ms: int,
    safety_triggered: bool,
    escalated: bool,
    sources: list,
    prompt_version: str = "V3",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a structured interaction record to interactions.jsonl.

    Automatically strips any PII before writing.
    Phase 8 requirement: interaction logging with latency capture.
    """
    # PII safety: do not log the raw input if it contains PII markers
    safe_input = user_input if is_pii_safe(user_input) else "[PII REDACTED]"
    safe_answer = answer if is_pii_safe(answer) else "[PII REDACTED]"

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "prompt_version": prompt_version,
        "user_input": safe_input[:500],
        "answer_snippet": safe_answer[:500],
        "tools_used": tools_used,
        "sources": sources,
        "latency_ms": latency_ms,
        "safety_triggered": safety_triggered,
        "escalated": escalated,
        **(extra or {}),
    }

    with open(_INTERACTION_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.debug(
        "Interaction logged | latency=%dms safety=%s escalated=%s",
        latency_ms,
        safety_triggered,
        escalated,
    )


def get_recent_interactions(n: int = 20) -> list:
    """Read the last n interaction records for display in the UI."""
    if not os.path.exists(_INTERACTION_FILE):
        return []
    records = []
    with open(_INTERACTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return records[-n:]
