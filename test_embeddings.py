import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scratch" / "Enterprise-AI-Copilot"))

from agent.config import config
from langchain_openai import OpenAIEmbeddings

print("API Key loaded (length):", len(config.OPENAI_API_KEY))
print("Base URL:", config.OPENAI_API_BASE)

models = ["text-embedding-3-small", "text-embedding-ada-002"]

for model in models:
    print(f"\n--- Testing model: {model} ---")
    kwargs = {
        "model": model,
        "openai_api_key": config.OPENAI_API_KEY,
    }
    if config.OPENAI_API_BASE:
        kwargs["base_url"] = config.OPENAI_API_BASE
    
    embeddings = OpenAIEmbeddings(**kwargs)
    try:
        res = embeddings.embed_query("hello world")
        print("SUCCESS! Embedding length:", len(res))
    except Exception as e:
        print("FAILED:", str(e))
