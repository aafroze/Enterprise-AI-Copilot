"""
tools/calculator.py — Business calculator tool for LangChain agent.

Supports arithmetic, percentages, growth rates, EMI, margins, and tax.
Phase 5 requirement: tool calling with correct routing.
"""

from __future__ import annotations

import math
import re
from typing import Union

from langchain_core.tools import Tool
from loguru import logger


def _safe_eval(expression: str) -> Union[float, str]:
    """
    Safely evaluate a mathematical expression string.

    Only allows numbers, operators, parentheses, and basic math functions.
    Never uses eval() on arbitrary code.
    """
    # Strip whitespace
    expr = expression.strip()

    # Replace common word operators
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")

    # Allow only safe characters
    allowed = re.compile(r"^[\d\s\+\-\*\/\(\)\.\%\,]+$")
    if not allowed.match(expr):
        # Try to handle percentage expressions like "12% of 350000"
        pct_match = re.match(
            r"(\d+\.?\d*)\s*%\s*of\s*([\d,]+\.?\d*)", expr, re.IGNORECASE
        )
        if pct_match:
            rate = float(pct_match.group(1))
            base = float(pct_match.group(2).replace(",", ""))
            return round(rate / 100 * base, 2)

        # Growth: "growth of X to Y" or "X grew to Y"
        growth_match = re.match(
            r"growth\s+(?:from\s+)?([\d,]+\.?\d*)\s+to\s+([\d,]+\.?\d*)",
            expr,
            re.IGNORECASE,
        )
        if growth_match:
            old = float(growth_match.group(1).replace(",", ""))
            new = float(growth_match.group(2).replace(",", ""))
            if old == 0:
                return "Error: Cannot calculate growth from zero."
            return round((new - old) / old * 100, 2)

        return f"Error: Unsupported expression: '{expression}'"

    # Remove commas from numbers
    expr = expr.replace(",", "")
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        return round(float(result), 4)
    except Exception as exc:
        return f"Error: {exc}"


def calculate(query: str) -> str:
    """
    Business calculator tool entry point.

    Handles:
    - Arithmetic:     "1500 * 12"
    - Percentages:    "12% of 350000"
    - Growth rates:   "growth from 250000 to 350000"
    - Revenue margin: "margin: revenue=500000 cost=380000"
    - EMI:            "emi: principal=100000 rate=8.5 months=36"
    """
    logger.info("[TOOL] Calculator called with: %s", query)
    q = query.strip()

    # Margin calculation
    margin_match = re.search(
        r"(?:margin|profit)[:\s]+revenue[=:\s]*([\d,]+\.?\d*)[,\s]+cost[=:\s]*([\d,]+\.?\d*)",
        q,
        re.IGNORECASE,
    )
    if margin_match:
        revenue = float(margin_match.group(1).replace(",", ""))
        cost = float(margin_match.group(2).replace(",", ""))
        if revenue == 0:
            return "Error: Revenue cannot be zero."
        margin = round((revenue - cost) / revenue * 100, 2)
        profit = round(revenue - cost, 2)
        return (
            f"**Profit Margin Calculation**\n"
            f"- Revenue: ${revenue:,.2f}\n"
            f"- Cost: ${cost:,.2f}\n"
            f"- Gross Profit: ${profit:,.2f}\n"
            f"- Margin: **{margin}%**"
        )

    # EMI calculation
    emi_match = re.search(
        r"emi[:\s]+principal[=:\s]*([\d,]+\.?\d*)[,\s]+rate[=:\s]*([\d\.]+)[,\s]+months[=:\s]*(\d+)",
        q,
        re.IGNORECASE,
    )
    if emi_match:
        P = float(emi_match.group(1).replace(",", ""))
        annual_rate = float(emi_match.group(2))
        n = int(emi_match.group(3))
        r = annual_rate / 12 / 100
        if r == 0:
            emi = P / n
        else:
            emi = P * r * (1 + r) ** n / ((1 + r) ** n - 1)
        total = round(emi * n, 2)
        interest = round(total - P, 2)
        return (
            f"**EMI Calculation**\n"
            f"- Principal: ${P:,.2f}\n"
            f"- Annual Rate: {annual_rate}%\n"
            f"- Tenure: {n} months\n"
            f"- Monthly EMI: **${emi:,.2f}**\n"
            f"- Total Payment: ${total:,.2f}\n"
            f"- Total Interest: ${interest:,.2f}"
        )

    # Generic calculation
    result = _safe_eval(q)
    if isinstance(result, str) and result.startswith("Error"):
        return result

    return f"**Result:** {result:,}" if isinstance(result, float) else f"**Result:** {result}"


calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description=(
        "Use this tool to perform business calculations such as arithmetic, percentages, "
        "growth rates, profit margins, and EMI calculations. "
        "Input should be a mathematical expression or a structured query like "
        "'12% of 350000', 'growth from 250000 to 350000', "
        "'margin: revenue=500000 cost=380000', or 'emi: principal=100000 rate=8.5 months=36'."
    ),
)
