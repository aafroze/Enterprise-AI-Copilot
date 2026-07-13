import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scratch" / "Enterprise-AI-Copilot"))

from agent.agent import EnterpriseAgent
from langchain_core.globals import set_debug
import traceback

set_debug(True)

agent = EnterpriseAgent()
try:
    print("Sending RAG query...")
    res = agent._agent_executor.invoke({
        "input": "summarize the Q2 business report",
        "chat_history": ""
    })
    print("SUCCESS! Output:")
    print(res)
except Exception as e:
    print("ERROR:")
    traceback.print_exc()
