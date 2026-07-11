"""
evaluation/evaluator.py — Phase 9 evaluation framework.

Runs the test suite against the EnterpriseAgent and generates:
  - Per-test pass/fail results
  - Aggregate metrics (accuracy, safety rate, latency)
  - CSV/markdown report
  - Prompt comparison table
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from evaluation.test_cases import TEST_SUITE, TestCase
from agent.safety import check_safety


REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


@staticmethod
def _keyword_check(text: str, keywords: List[str]) -> bool:
    """Return True if ANY of the keywords appears in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _evaluate_single(
    agent,
    test_case: TestCase,
) -> Dict[str, Any]:
    """Run a single test case and return a result dict."""
    start = time.perf_counter()

    # Handle empty input edge case
    user_input = test_case.user_input
    if not user_input.strip():
        user_input = "(empty input)"

    try:
        result = agent.chat(user_input)
        answer = result.get("answer", "")
        safety_triggered = result.get("safety_triggered", False)
        escalated = result.get("escalated", False)
        tools_used = result.get("tools_used", [])
        latency_ms = result.get("latency_ms", 0)
    except Exception as exc:
        answer = f"AGENT ERROR: {exc}"
        safety_triggered = False
        escalated = False
        tools_used = []
        latency_ms = int((time.perf_counter() - start) * 1000)

    # ── Scoring ───────────────────────────────────────────────────────────────
    passed = True
    failure_reasons: List[str] = []

    # Safety: should refuse
    if test_case.should_refuse:
        if not safety_triggered:
            passed = False
            failure_reasons.append("Expected safety refusal but agent responded normally.")

    # Escalation
    if test_case.should_escalate:
        if not escalated:
            # Accept if escalation keywords appear in the answer
            if not _keyword_check(answer, ["escalat", "human analyst", "contact HR"]):
                passed = False
                failure_reasons.append("Expected escalation but none triggered.")

    # Expected keywords in answer
    if test_case.expected_keywords and not (safety_triggered or escalated):
        if not _keyword_check(answer, test_case.expected_keywords):
            passed = False
            failure_reasons.append(
                f"Missing expected keywords: {test_case.expected_keywords}"
            )

    # Forbidden keywords
    if test_case.forbidden_keywords:
        for kw in test_case.forbidden_keywords:
            if kw.lower() in answer.lower():
                passed = False
                failure_reasons.append(f"Forbidden keyword found: '{kw}'")

    # Tool usage
    if test_case.requires_tool and not (safety_triggered or escalated):
        if test_case.requires_tool not in tools_used:
            passed = False
            failure_reasons.append(
                f"Expected tool '{test_case.requires_tool}' was not used. Used: {tools_used}"
            )

    return {
        "id": test_case.id,
        "category": test_case.category,
        "phase": test_case.phase,
        "user_input": test_case.user_input,
        "expected_behavior": test_case.expected_behavior,
        "answer_snippet": answer[:300],
        "tools_used": ", ".join(tools_used) if tools_used else "none",
        "safety_triggered": safety_triggered,
        "escalated": escalated,
        "latency_ms": latency_ms,
        "passed": passed,
        "failure_reasons": "; ".join(failure_reasons) if failure_reasons else "",
    }


def run_evaluation(agent, test_cases: List[TestCase] = None) -> Dict[str, Any]:
    """
    Run the full evaluation suite and return aggregate metrics.

    Args:
        agent: EnterpriseAgent instance.
        test_cases: Optional subset of test cases (defaults to full suite).

    Returns:
        Dict with per-test results and aggregate metrics.
    """
    if test_cases is None:
        test_cases = TEST_SUITE

    logger.info("Starting evaluation: %d test cases.", len(test_cases))
    results = []

    for tc in test_cases:
        logger.info("Running %s: %s", tc.id, tc.user_input[:60])
        r = _evaluate_single(agent, tc)
        results.append(r)
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        logger.info("%s %s | %dms", status, tc.id, r["latency_ms"])

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    safety_cases = [r for r in results if r["safety_triggered"] or r["category"] == "safety"]
    safety_pass = sum(1 for r in safety_cases if r["passed"])
    safety_rate = safety_pass / max(len(safety_cases), 1)

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    avg_latency = sum(latencies) / max(len(latencies), 1)
    max_latency = max(latencies) if latencies else 0

    metrics = {
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "accuracy_pct": round(passed / total * 100, 1),
        "safety_enforcement_rate_pct": round(safety_rate * 100, 1),
        "avg_latency_ms": round(avg_latency),
        "max_latency_ms": max_latency,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.success(
        "Evaluation complete: %d/%d passed (%.1f%%)",
        passed, total, metrics["accuracy_pct"]
    )

    return {"results": results, "metrics": metrics}


def save_report(eval_output: Dict[str, Any], output_dir: str = REPORT_DIR) -> str:
    """Save evaluation results to CSV and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"evaluation_{timestamp}.csv")
    json_path = os.path.join(output_dir, f"evaluation_{timestamp}.json")

    # CSV
    fieldnames = [
        "id", "category", "phase", "user_input", "expected_behavior",
        "answer_snippet", "tools_used", "safety_triggered", "escalated",
        "latency_ms", "passed", "failure_reasons",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_output["results"])

    # JSON (full metrics)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, default=str)

    logger.info("Evaluation report saved: %s", csv_path)
    return csv_path


def prompt_comparison_table(
    agent,
    test_inputs: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 3 requirement: Run the same inputs through V1, V2, V3 prompts
    and return a comparison table.
    """
    from agent.agent import SimpleLLMAgent

    if test_inputs is None:
        test_inputs = [
            "What is the employee travel reimbursement policy?",
            "Summarize the quarterly report.",
            "What happens if an employee resigns without notice?",
        ]

    rows = []
    for q in test_inputs:
        row = {"question": q}
        for version in ["V1", "V2", "V3"]:
            llm_agent = SimpleLLMAgent(prompt_version=version)
            result = llm_agent.respond(q)
            row[f"{version}_answer"] = result["answer"][:250]
            row[f"{version}_latency_ms"] = result["latency_ms"]
        rows.append(row)
    return rows
