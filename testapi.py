import os
import time
from openai import OpenAI

# Explicitly test using your proxy configuration
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1"
)

print("🚀 Step 1: Testing Chat Completion via Vocareum...")
try:
    start_time = time.time()
    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",  # or whichever model your app config uses
        messages=[{"role": "user", "content": "Respond with the word 'Success'."}],
        timeout=10.0
    )
    latency = time.time() - start_time
    print(f"✅ Chat Success! Response: {chat_response.choices[0].message.content.strip()}")
    print(f"⏱️ Latency: {latency:.2f} seconds\n")
except Exception as e:
    print(f"❌ Chat Failed: {e}\n")

print("🚀 Step 2: Testing Embedding Generation via Vocareum...")
try:
    start_time = time.time()
    embed_response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Calculate 12% growth on $350,000"
    )
    latency = time.time() - start_time
    print(f"✅ Embedding Success! Generated vector size: {len(embed_response.data[0].embedding)}")
    print(f"⏱️ Latency: {latency:.2f} seconds\n")
except Exception as e:
    print(f"❌ Embedding Failed: {e}\n")