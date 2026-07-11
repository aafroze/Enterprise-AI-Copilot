"""
tools/date_tool.py — Current date/time tool for the LangChain agent.

Phase 5: Provides temporal context so the agent can answer
"what quarter are we in?" or "is this policy currently active?"
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import Tool
from loguru import logger


def get_current_date(_: str = "") -> str:
    """Return current date, time, day-of-week, week number, and fiscal quarter."""
    logger.info("[TOOL] DateTool called.")
    now = datetime.now(tz=timezone.utc)

    month = now.month
    if month <= 3:
        quarter = "Q1"
        fiscal_year_note = "January – March"
    elif month <= 6:
        quarter = "Q2"
        fiscal_year_note = "April – June"
    elif month <= 9:
        quarter = "Q3"
        fiscal_year_note = "July – September"
    else:
        quarter = "Q4"
        fiscal_year_note = "October – December"

    week_num = now.isocalendar()[1]

    return (
        f"**Current Date & Time (UTC)**\n"
        f"- Date: {now.strftime('%A, %B %d, %Y')}\n"
        f"- Time: {now.strftime('%H:%M:%S')} UTC\n"
        f"- Week: {week_num} of {now.year}\n"
        f"- Fiscal Quarter: **{quarter}** ({fiscal_year_note} {now.year})\n"
        f"- Calendar Year: {now.year}"
    )


date_tool = Tool(
    name="get_current_date",
    func=get_current_date,
    description=(
        "Use this tool to get the current date, time, day of the week, "
        "week number, and fiscal quarter. "
        "Use it when the user asks about the current date, time, or quarter, "
        "or when you need to determine whether a policy or report is current."
    ),
)
