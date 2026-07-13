"""
cli.py - Simple terminal interface for the Enterprise AI Copilot.

This provides a chat-style loop that mirrors the Streamlit app:
1. Ask a question in the terminal
2. Invoke the agent
3. Print the reply and metadata
4. Continue until the user exits
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


def _print_banner() -> None:
    print("=" * 72)
    print("Enterprise AI Operations Copilot - CLI")
    print("Type your question and press Enter.")
    print("Commands: /reset, /prompt V1|V2|V3, /exit")
    print("=" * 72)


def _format_metadata(result: dict) -> str:
    tools = ", ".join(result.get("tools_used", [])) or "none"
    sources = ", ".join(result.get("sources", [])) or "none"
    latency = result.get("latency_ms", 0)
    safety = "yes" if result.get("safety_triggered") else "no"
    escalated = "yes" if result.get("escalated") else "no"
    return (
        f"tools={tools} | sources={sources} | latency={latency}ms | "
        f"safety={safety} | escalated={escalated}"
    )


def run_cli(prompt_version: str = "V3") -> int:
    from agent.agent import EnterpriseAgent
    from rag.ingest import ingest
    from pathlib import Path

    root = Path(__file__).parent
    data_dir = root / "data"

    agent = EnterpriseAgent(prompt_version=prompt_version)

    if data_dir.exists():
        print("Building knowledge base from local documents...")
        ingest(data_dir=str(data_dir), reset=False)

    _print_banner()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return 0

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "exit", "quit", "q"}:
            print("Exiting.")
            return 0

        if user_input.lower() == "/reset":
            agent.reset_conversation()
            print("Conversation memory cleared.")
            continue

        if user_input.lower().startswith("/prompt "):
            _, _, version = user_input.partition(" ")
            version = version.strip().upper()
            if version in {"V1", "V2", "V3"}:
                agent.switch_prompt(version)
                print(f"Switched to prompt version {version}.")
            else:
                print("Usage: /prompt V1|V2|V3")
            continue

        print("Agent: thinking...")
        result = agent.chat(user_input)
        print(f"\nAgent: {result['answer']}")
        print(f"Meta: {_format_metadata(result)}")


if __name__ == "__main__":
    raise SystemExit(run_cli())
