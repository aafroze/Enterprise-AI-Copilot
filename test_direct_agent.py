"""
Direct test of agent.chat() to capture the exact exception.
"""
import os
import sys
from pathlib import Path

# Set env vars as Streamlit would see them
os.environ['OPENAI_API_KEY'] = 'voc-111511149117535047104646a4a769b49c185.19508895'
os.environ['OPENAI_API_BASE'] = 'https://openai.vocareum.com/v1'

sys.path.insert(0, str(Path(__file__).parent))

import traceback
from agent.agent import EnterpriseAgent

try:
    agent = EnterpriseAgent(prompt_version='V3')
    print('✓ Agent loaded successfully')
    
    result = agent.chat('What is the travel reimbursement policy?')
    print('✓ Chat result:', result)
except Exception as e:
    print(f'✗ Exception caught: {type(e).__name__}: {e}')
    traceback.print_exc()
