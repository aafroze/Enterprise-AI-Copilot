from dotenv import load_dotenv
import os
from pathlib import Path
import traceback

load_dotenv(dotenv_path=Path('.env'))
print('OPENAI_API_KEY', bool(os.getenv('OPENAI_API_KEY')))

from agent.agent import EnterpriseAgent

try:
    agent = EnterpriseAgent(prompt_version='V3')
    print('Agent loaded')
    result = agent._agent_executor.invoke({'input': 'What is the travel reimbursement policy?', 'chat_history': ''})
    print('Result:', result)
except Exception as exc:
    traceback.print_exc()
    print('ERROR TYPE:', type(exc).__name__)
    print('ERROR MSG:', exc)
