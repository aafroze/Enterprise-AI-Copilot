# Enterprise AI Operations Copilot — Implementation Plan

## Goal
Build a production-quality, LangChain-based Enterprise AI Operations Copilot for the capstone project covering all 9 phases with full evidence artifacts.

## Tech Stack
- **Framework**: LangChain
- **LLM**: OpenAI GPT-4o-mini
- **Vector Store**: ChromaDB
- **UI**: Streamlit
- **Embeddings**: OpenAI Embeddings
- **Memory**: ConversationBufferMemory
- **Deployment**: Local (Streamlit) + Docker

## Project Structure
```
Enterprise-AI-Copilot/
├── app.py                          # Main Streamlit UI
├── requirements.txt
├── README.md
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── data/                           # Sample PDFs
│   ├── employee_handbook.pdf
│   ├── travel_policy.pdf
│   ├── hr_policy.pdf
│   ├── company_faq.pdf
│   └── quarterly_report.pdf
├── agent/
│   ├── agent.py                    # Main LangChain agent
│   ├── prompts.py                  # Prompt versions
│   └── safety.py                   # Safety guardrails
├── rag/
│   ├── ingest.py                   # PDF ingestion + embedding
│   └── retriever.py                # RAG retrieval
├── tools/
│   ├── calculator.py
│   ├── date_tool.py
│   └── search_docs.py
├── memory/
│   └── conversation_memory.py
├── evaluation/
│   ├── test_cases.py
│   └── evaluator.py
├── logs/
│   └── agent.log
└── report/
    └── capstone_report.md
```

## Build Phases
1. Project scaffold + config
2. Sample business documents (PDFs)
3. RAG pipeline (ingest + retrieval)
4. Tools (calculator, date, search)
5. Agent core (LangChain + prompts)
6. Safety guardrails
7. Memory + feedback adaptation
8. Streamlit UI
9. Evaluation framework + logging
10. Docker + deployment
