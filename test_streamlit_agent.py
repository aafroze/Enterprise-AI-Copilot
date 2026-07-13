"""
Minimal test to check if the agent can be imported and initialized.
"""
import os
import sys
from pathlib import Path

# Check env vars
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("OPENAI_API_BASE set:", bool(os.getenv("OPENAI_API_BASE")))
print("cwd:", os.getcwd())

# Try loading config
try:
    from agent.config import config
    print("✓ Config loaded")
    print(f"  API Key: {config.OPENAI_API_KEY[:20] if config.OPENAI_API_KEY else 'MISSING'}...")
    print(f"  API Base: {config.OPENAI_API_BASE}")
    print(f"  Model: {config.OPENAI_MODEL}")
except Exception as e:
    print(f"✗ Config load failed: {e}")
    sys.exit(1)

# Try loading agent
try:
    from agent.agent import EnterpriseAgent
    print("✓ Agent class imported")
    agent = EnterpriseAgent(prompt_version='V3')
    print("✓ Agent initialized")
except Exception as e:
    import traceback
    print(f"✗ Agent init failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try a simple chat
try:
    result = agent.chat("What is 2+2?")
    print("✓ Agent chat worked")
    print(f"  Answer: {result['answer'][:100]}")
except Exception as e:
    import traceback
    print(f"✗ Agent chat failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
