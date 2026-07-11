"""
memory/conversation_memory.py — Conversation memory and user-preference store.

Handles short-term (session) memory and persistent user-feedback preferences
for Phase 6 (memory) and Phase 7 (adaptive behaviour).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_classic.memory import ConversationBufferWindowMemory
from loguru import logger


# ── Session memory ────────────────────────────────────────────────────────────

class SessionMemory:
    """
    Wraps LangChain ConversationBufferWindowMemory with convenience helpers.

    Keeps the last `window_size` exchanges in the buffer, providing
    multi-turn conversation quality for Phase 6.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._memory = ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="output",
        )
        self._user_context: Dict[str, str] = {}
        self.message_count: int = 0
        logger.debug("SessionMemory initialised (window=%d)", window_size)

    # ── Context ───────────────────────────────────────────────────────────────

    def set_user_context(self, key: str, value: str) -> None:
        """Store a piece of derived user context (e.g. department='Finance')."""
        self._user_context[key] = value
        logger.debug("User context updated: %s=%s", key, value)

    def get_user_context(self) -> Dict[str, str]:
        return dict(self._user_context)

    def get_context_summary(self) -> str:
        if not self._user_context:
            return ""
        parts = [f"{k}: {v}" for k, v in self._user_context.items()]
        return "Known about this user — " + "; ".join(parts)

    # ── LangChain memory delegation ───────────────────────────────────────────

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._memory.load_memory_variables(inputs)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        self._memory.save_context(inputs, outputs)
        self.message_count += 1

    def clear(self) -> None:
        self._memory.clear()
        self._user_context.clear()
        self.message_count = 0
        logger.info("SessionMemory cleared.")

    def get_history_messages(self) -> List[Any]:
        """Return raw message objects for display in the UI."""
        return self._memory.chat_memory.messages

    @property
    def langchain_memory(self) -> ConversationBufferWindowMemory:
        return self._memory


# ── User preference store ─────────────────────────────────────────────────────

_PREFS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "logs", "user_preferences.json"
)


class UserPreferenceStore:
    """
    Persists user feedback preferences between sessions (Phase 7).

    Stored as a simple JSON file so that subsequent runs adapt automatically.
    """

    _DEFAULTS: Dict[str, Any] = {
        "response_style": "standard",   # concise | detailed | standard
        "response_format": "prose",     # bullet | table | prose | standard
        "feedback_history": [],
        "last_updated": None,
    }

    def __init__(self, path: str = _PREFS_PATH) -> None:
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._prefs = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return dict(self._DEFAULTS)

    def _save(self) -> None:
        self._prefs["last_updated"] = datetime.utcnow().isoformat()
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._prefs, f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def record_feedback(self, feedback_text: str, rating: int) -> None:
        """
        Parse feedback and update stored preferences.

        Args:
            feedback_text: Raw feedback string from the user.
            rating: 1–5 star rating.
        """
        text = feedback_text.lower()

        # Style adaptation
        if any(w in text for w in ["too long", "shorter", "brief", "concise", "tldr"]):
            self._prefs["response_style"] = "concise"
        elif any(w in text for w in ["too short", "more detail", "elaborate", "explain more"]):
            self._prefs["response_style"] = "detailed"

        # Format adaptation
        if any(w in text for w in ["bullet", "list", "points", "bullets"]):
            self._prefs["response_format"] = "bullet"
        elif any(w in text for w in ["table", "tabular", "columns"]):
            self._prefs["response_format"] = "table"
        elif any(w in text for w in ["paragraph", "prose", "narrative"]):
            self._prefs["response_format"] = "prose"

        # Record history
        self._prefs["feedback_history"].append({
            "text": feedback_text,
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat(),
            "style_set": self._prefs["response_style"],
            "format_set": self._prefs["response_format"],
        })

        self._save()
        logger.info(
            "Feedback recorded → style=%s format=%s rating=%d",
            self._prefs["response_style"],
            self._prefs["response_format"],
            rating,
        )

    def get_style(self) -> str:
        return self._prefs.get("response_style", "standard")

    def get_format(self) -> str:
        return self._prefs.get("response_format", "prose")

    def reset(self) -> None:
        self._prefs = dict(self._DEFAULTS)
        self._save()
        logger.info("User preferences reset to defaults.")

    def summary(self) -> str:
        return (
            f"Response style: **{self.get_style()}** | "
            f"Format: **{self.get_format()}** | "
            f"Feedback items: {len(self._prefs.get('feedback_history', []))}"
        )
