import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scratch" / "Enterprise-AI-Copilot"))

print("Importing retriever...")
from rag.retriever import retriever

print("Calling retrieve...")
try:
    docs = retriever.retrieve("Q2")
    print("SUCCESS!")
    print("Docs:", docs)
except Exception as e:
    import traceback
    print("ERROR:")
    traceback.print_exc()
