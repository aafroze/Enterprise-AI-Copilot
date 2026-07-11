"""
safety.py — Safety guardrails for the Enterprise AI Operations Copilot.

Implements keyword-based and semantic checks to refuse requests that
violate the agent's operational boundaries (Phase 5 / 9 requirement).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


# ── Forbidden action patterns ────────────────────────────────────────────────

_FORBIDDEN_PATTERNS: List[str] = [
    # Data modification
    r"\bdelete\b.*\b(record|data|entry|row|customer|user|file|database)\b",
    r"\b(drop|truncate|alter)\b.*\b(table|database|schema)\b",
    r"\bmodify\b.*\b(database|record|data|entry)\b",
    r"\bupdate\b.*\b(database|record|table|entry)\b",
    r"\binsert\b.*\b(database|table|record)\b",
    # SQL execution
    r"\bexecute\b.*\bsql\b",
    r"\brun\b.*\b(query|sql|script)\b",
    r"\bsql\b.*\b(injection|execute|run)\b",
    # Financial transactions
    r"\b(transfer|send|move)\b.*\b(money|funds|payment|cash)\b",
    r"\b(approve|process)\b.*\b(payment|transaction|withdrawal)\b",
    # Approvals / authorisations
    r"\b(approve|authorise|authorize)\b.*\b(request|leave|purchase|order)\b",
    # Access credential requests
    r"\b(give|share|provide|show)\b.*\b(password|credentials|secret|token|api.?key)\b",
    # PII exposure
    r"\b(show|list|dump|export)\b.*\b(all\s+customers|all\s+users|pii|personal.?data)\b",
    # System operations
    r"\b(shutdown|restart|reboot)\b.*\b(server|system|service)\b",
    r"\bsend\b.*\b(email|sms|notification)\b.*\b(all|bulk|everyone)\b",
]

_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _FORBIDDEN_PATTERNS
]

# ── Escalation keywords ──────────────────────────────────────────────────────

_ESCALATION_KEYWORDS: List[str] = [
    "legal action",
    "sue",
    "lawsuit",
    "compensation",
    "complaint",
    "harassment",
    "discrimination",
    "fraud",
    "whistleblower",
    "regulatory",
    "compliance breach",
]


@dataclass
class SafetyCheckResult:
    is_safe: bool
    reason: Optional[str] = None
    requires_escalation: bool = False
    matched_pattern: Optional[str] = None


# ── Public API ───────────────────────────────────────────────────────────────

def check_safety(user_input: str) -> SafetyCheckResult:
    """
    Evaluate a user message against safety policies.

    Returns a SafetyCheckResult indicating whether the request is safe,
    requires escalation, or must be refused outright.
    """
    text = user_input.strip()

    # 1. Check for outright-forbidden patterns
    for pattern in _COMPILED_PATTERNS:
        m = pattern.search(text)
        if m:
            matched = m.group(0)
            logger.warning(f"[SAFETY] Blocked request. Matched: '{matched}' | Input: {text[:120]}")
            return SafetyCheckResult(
                is_safe=False,
                reason=(
                    "I'm sorry, but that request falls outside my operational boundaries. "
                    "I am a decision-support assistant and cannot modify data, execute "
                    "database operations, approve requests, or perform financial transactions.\n\n"
                    "If this is urgent, please contact your system administrator or human analyst."
                ),
                matched_pattern=matched,
            )

    # 2. Check for escalation scenarios
    lower = text.lower()
    for kw in _ESCALATION_KEYWORDS:
        if kw in lower:
            logger.info(f"[SAFETY] Escalation triggered by keyword: '{kw}'")
            return SafetyCheckResult(
                is_safe=True,
                requires_escalation=True,
                reason=(
                    "This query involves a sensitive matter that requires human review. "
                    "I am escalating this to a qualified human analyst. "
                    "Please contact HR / Legal / Compliance directly for assistance."
                ),
            )

    return SafetyCheckResult(is_safe=True)


def get_refusal_message(result: SafetyCheckResult) -> str:
    """Return the user-facing refusal or escalation message."""
    return result.reason or "I cannot process that request."


def is_pii_safe(text: str) -> bool:
    """
    Basic check that log output does not contain obvious PII patterns.
    Used before writing to log files.
    """
    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",          # SSN
        r"\b\d{16}\b",                       # Credit card (simplified)
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\+?[1-9]\d{9,14}\b",            # Phone number
    ]
    for p in pii_patterns:
        if re.search(p, text):
            return False
    return True
