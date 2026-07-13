"""
agent/agent.py — Core LangChain agent orchestrating RAG, tools, memory, and safety.

Implements:
  Phase 2  – Baseline agent (rules-only, no LLM)
  Phase 3  – LLM agent with versioned prompts
  Phase 4  – RAG-enabled agent
  Phase 5  – Tool-calling agent
  Phase 6  – Memory + multi-turn planning
  Phase 7  – Adaptive feedback agent
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from loguru import logger

from agent.config import config
from agent.prompts import PROMPT_V1, PROMPT_V2, PROMPT_V3, build_adaptive_prompt
from agent.safety import SafetyCheckResult, check_safety, get_refusal_message
from memory.conversation_memory import SessionMemory, UserPreferenceStore
from rag.retriever import retriever
from tools.calculator import calculator_tool
from tools.date_tool import date_tool


# ── Baseline Agent (Phase 2) ─────────────────────────────────────────────────

class BaselineAgent:
    """
    Phase 2: Simple rule-based agent with no LLM.

    Demonstrates the limitations of a rules-only approach.
    """

    _RULES: Dict[str, str] = {
        "travel": "Travel reimbursement is subject to company policy. Contact HR for details.",
        "leave": "Leave requests must be submitted via the HR portal at least 3 business days in advance.",
        "refund": "Refunds are processed within 5–7 business days. Contact the finance team.",
        "password": "Password resets must go through the IT Helpdesk portal.",
        "salary": "Salary queries should be directed to your HR Business Partner.",
        "purchase": "Purchase orders above $500 require manager approval.",
    }

    def respond(self, user_input: str) -> Dict[str, Any]:
        text = user_input.lower()
        for keyword, response in self._RULES.items():
            if keyword in text:
                return {
                    "answer": response,
                    "source": "Rule-based system",
                    "limitation": True,
                }
        return {
            "answer": (
                "I'm sorry, I don't have a pre-defined answer for that. "
                "Please contact the appropriate department."
            ),
            "source": "Default rule",
            "limitation": True,
        }


# ── LLM Agent (Phases 3–7) ───────────────────────────────────────────────────

# ReAct prompt template for the tool-calling agent
_REACT_TEMPLATE = """{system_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class EnterpriseAgent:
    """
    Full LangChain agent with RAG, tools, memory, and safety guardrails.
    """

    def __init__(
        self,
        prompt_version: str = "V3",
        preferred_style: str = "standard",
        preferred_format: str = "prose",
    ) -> None:
        config.validate()

        self.prompt_version = prompt_version
        self.memory = SessionMemory()
        self.preference_store = UserPreferenceStore()

        kwargs = {
            "model": config.OPENAI_MODEL,
            "temperature": config.AGENT_TEMPERATURE,
            "openai_api_key": config.OPENAI_API_KEY,
            "max_retries": 3,
        }
        if config.OPENAI_API_BASE:
            kwargs["base_url"] = config.OPENAI_API_BASE
            
        self._llm = ChatOpenAI(**kwargs)

        self._tools = [
            calculator_tool,
            date_tool,
            retriever.as_langchain_tool(),
        ]

        self._agent_executor = self._build_agent(prompt_version, preferred_style, preferred_format)
        logger.info(
            "EnterpriseAgent initialised (model=%s, prompt=%s).",
            config.OPENAI_MODEL,
            prompt_version,
        )

    # ── Builder ───────────────────────────────────────────────────────────────

    def _get_system_prompt(
        self,
        version: str,
        preferred_style: str,
        preferred_format: str,
    ) -> str:
        prompt_map = {"V1": PROMPT_V1, "V2": PROMPT_V2, "V3": PROMPT_V3}
        base = prompt_map.get(version, PROMPT_V3)
        
        # Clean up any literal placeholders used for RAG chains that conflict with ReAct format
        for placeholder in [
            "Context from knowledge base:",
            "Conversation history:",
            "User question:",
            "{context}",
            "{chat_history}",
            "{question}",
        ]:
            base = base.replace(placeholder, "")
        
        # Clean up trailing instructions if they conflict
        base = base.replace("Provide a helpful, safe, grounded response:", "")
        base = base.strip()
        
        return build_adaptive_prompt(base, preferred_style, preferred_format)

    def _build_agent(
        self,
        version: str,
        preferred_style: str,
        preferred_format: str,
    ) -> AgentExecutor:
        system_prompt = self._get_system_prompt(version, preferred_style, preferred_format)

        prompt = PromptTemplate.from_template(
            _REACT_TEMPLATE,
        ).partial(system_prompt=system_prompt)

        agent = create_react_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            max_iterations=config.MAX_ITERATIONS,
            handle_parsing_errors=True,
            early_stopping_method="force",
            verbose=True,
            return_intermediate_steps=True,
        )

    def _rebuild_with_preferences(self) -> None:
        """Rebuild the agent executor with current user preferences."""
        self._agent_executor = self._build_agent(
            self.prompt_version,
            self.preference_store.get_style(),
            self.preference_store.get_format(),
        )

    # ── Context extraction ─────────────────────────────────────────────────────

    def _extract_context_hints(self, text: str) -> None:
        """Detect and store user-context hints for memory."""
        import re

        dept_match = re.search(
            r"(?:i\s+(?:work|am)\s+(?:in|on)|my\s+(?:team|department)\s+is)\s+"
            r"([A-Za-z\s]{2,30})",
            text,
            re.IGNORECASE,
        )
        if dept_match:
            dept = dept_match.group(1).strip().title()
            self.memory.set_user_context("department", dept)

        role_match = re.search(
            r"(?:i\s+am\s+(?:a|an)|my\s+role\s+is)\s+([A-Za-z\s]{2,40})",
            text,
            re.IGNORECASE,
        )
        if role_match:
            role = role_match.group(1).strip().title()
            self.memory.set_user_context("role", role)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message through the full agent pipeline.

        Returns a dict with:
          - answer: str
          - sources: list[str]
          - tools_used: list[str]
          - latency_ms: int
          - safety_triggered: bool
          - escalated: bool
          - intermediate_steps: list
        """
        start = time.perf_counter()

        # ── Safety check (Phase 5 / 9) ────────────────────────────────────────
        safety: SafetyCheckResult = check_safety(user_input)
        if not safety.is_safe:
            return {
                "answer": get_refusal_message(safety),
                "sources": [],
                "tools_used": [],
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "safety_triggered": True,
                "escalated": False,
                "intermediate_steps": [],
            }

        if safety.requires_escalation:
            return {
                "answer": get_refusal_message(safety),
                "sources": [],
                "tools_used": [],
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "safety_triggered": False,
                "escalated": True,
                "intermediate_steps": [],
            }

        # ── Context extraction (Phase 6) ──────────────────────────────────────
        self._extract_context_hints(user_input)
        context_prefix = self.memory.get_context_summary()
        enriched_input = (
            f"{context_prefix}\n\n{user_input}" if context_prefix else user_input
        )

        # ── Build history string ──────────────────────────────────────────────
        history = self.memory.langchain_memory.load_memory_variables({})
        chat_history_str = ""
        if history.get("chat_history"):
            msgs = history["chat_history"]
            lines = []
            for msg in msgs[-6:]:  # last 3 exchanges
                role = "User" if msg.type == "human" else "Assistant"
                lines.append(f"{role}: {msg.content}")
            chat_history_str = "\n".join(lines)

        # ── Agent invocation ──────────────────────────────────────────────────
        try:
            result = self._agent_executor.invoke({
                "input": enriched_input,
                "chat_history": chat_history_str,
            })
        except Exception:
            logger.exception("Agent invocation error")
            return {
                "answer": (
                    "I'm sorry, I encountered an unexpected error processing your request. "
                    "Please try again or contact your system administrator."
                ),
                "sources": [],
                "tools_used": [],
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "safety_triggered": False,
                "escalated": False,
                "intermediate_steps": [],
            }

        answer = result.get("output", "I could not generate a response.")
        intermediate_steps = result.get("intermediate_steps", [])

        # ── Extract tool usage ────────────────────────────────────────────────
        tools_used = []
        sources = []
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, obs = step
                tool_name = getattr(action, "tool", "unknown")
                if tool_name not in tools_used:
                    tools_used.append(tool_name)
                if tool_name == "search_documents" and isinstance(obs, str):
                    import re
                    found = re.findall(r"\[Source \d+: ([^\]]+)\]", obs)
                    sources.extend(found)

        sources = list(dict.fromkeys(sources))  # deduplicate, preserve order

        # ── Save to memory (Phase 6) ──────────────────────────────────────────
        self.memory.save_context(
            {"question": user_input},
            {"output": answer},
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"[AGENT] Responded in {latency_ms}ms | tools={tools_used} | safety={safety.is_safe}"
        )

        return {
            "answer": answer,
            "sources": sources,
            "tools_used": tools_used,
            "latency_ms": latency_ms,
            "safety_triggered": False,
            "escalated": False,
            "intermediate_steps": intermediate_steps,
        }

    def submit_feedback(self, feedback_text: str, rating: int) -> str:
        """
        Record user feedback and rebuild the agent with adapted preferences.

        Phase 7: Adaptive behaviour.
        """
        self.preference_store.record_feedback(feedback_text, rating)
        self._rebuild_with_preferences()
        style = self.preference_store.get_style()
        fmt = self.preference_store.get_format()
        return (
            f"Thank you for your feedback! I've updated my response preferences: "
            f"**Style → {style}**, **Format → {fmt}**. "
            f"Future responses will reflect these preferences."
        )

    def reset_conversation(self) -> None:
        """Clear session memory (Phase 6 — memory reset behaviour)."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def get_preference_summary(self) -> str:
        return self.preference_store.summary()

    def switch_prompt(self, version: str) -> None:
        """Switch the active prompt version (Phase 3 — comparison)."""
        self.prompt_version = version
        self._agent_executor = self._build_agent(
            version,
            self.preference_store.get_style(),
            self.preference_store.get_format(),
        )
        logger.info(f"Prompt switched to {version}")


# ── Phase 2: Simple LLM without RAG ──────────────────────────────────────────

class SimpleLLMAgent:
    """
    Phase 3 baseline: LLM only, no tools, no RAG.

    Used to demonstrate the improvement when RAG and tools are added.
    """

    def __init__(self, prompt_version: str = "V1") -> None:
        config.validate()
        kwargs = {
            "model": config.OPENAI_MODEL,
            "temperature": 0.2,
            "openai_api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_API_BASE:
            kwargs["base_url"] = config.OPENAI_API_BASE
            
        self._llm = ChatOpenAI(**kwargs)
        self._prompt_version = prompt_version
        prompt_map = {"V1": PROMPT_V1, "V2": PROMPT_V2, "V3": PROMPT_V3}
        self._system_prompt = prompt_map.get(prompt_version, PROMPT_V1)

    def respond(self, user_input: str) -> Dict[str, Any]:
        start = time.perf_counter()
        messages = [
            SystemMessage(content=self._system_prompt),
        ]
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=user_input))

        try:
            response = self._llm.invoke(messages)
            answer = response.content
        except Exception as exc:
            answer = f"Error: {exc}"

        return {
            "answer": answer,
            "prompt_version": self._prompt_version,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "rag_used": False,
        }
