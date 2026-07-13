# Enterprise AI Operations Copilot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?logo=langchain)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

**An intelligent decision-support assistant for enterprise business operations.**

</div>

---

## Overview

The **Enterprise AI Operations Copilot** is a production-quality AI agent built for the Industry Capstone project. It assists business analysts by answering questions from company documentation using **Retrieval-Augmented Generation (RAG)**, **tool calling**, **conversation memory**, and **adaptive feedback** — while strictly enforcing safety guardrails.

**Track:** LangChain | **Scenario:** Business Operations (Scenario 1)

---

## Architecture

```
User
  │
  ▼
Streamlit UI (app.py)
  │
  ├── Safety Guardrails ──→ Refuse / Escalate
  │
  ▼
LangChain ReAct Agent (agent/agent.py)
  │
  ├── Tool: search_documents ──→ ChromaDB ──→ Business Documents
  ├── Tool: calculator        ──→ Arithmetic / Margins / EMI
  └── Tool: get_current_date  ──→ Date / Quarter
  │
  ▼
ConversationBufferWindowMemory (memory/)
  │
  ▼
OpenAI GPT-4o-mini
  │
  ▼
Safe, Grounded Response
```

---

## Features by Phase

| Phase | Feature | Status |
|-------|---------|--------|
| Phase 2 | Baseline rule-based agent | ✅ |
| Phase 3 | LLM integration + 3 prompt versions | ✅ |
| Phase 4 | RAG with ChromaDB + 5 business docs | ✅ |
| Phase 5 | Tool calling (calculator, date, search) | ✅ |
| Phase 6 | Conversation memory + planning | ✅ |
| Phase 7 | Adaptive feedback (style + format) | ✅ |
| Phase 8 | Docker deployment + structured logging | ✅ |
| Phase 9 | Evaluation framework + 20 test cases | ✅ |

---

## Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/aafroze/My_repo.git
cd Enterprise-AI-Copilot
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Documents
```bash
python -m rag.ingest
```

### 4. Run the App
```bash
streamlit run app.py
```

### 5. Run in the Terminal
```bash
python cli.py
```

Terminal commands inside the CLI:
```text
/reset           Clear conversation memory
/prompt V1       Switch to prompt version V1
/prompt V2       Switch to prompt version V2
/prompt V3       Switch to prompt version V3
/exit            Quit the CLI
```

### 6. Docker (Alternative)
```bash
docker-compose up --build
```
Open **http://localhost:8501**

---

## Safety Guardrails

The agent **refuses** requests to:
- ❌ Delete or modify database records
- ❌ Execute SQL queries
- ❌ Approve or authorise requests
- ❌ Transfer money or process payments
- ❌ Share passwords or credentials
- ❌ Export PII

The agent **escalates** to a human analyst for:
- ⚠️ Legal threats or harassment claims
- ⚠️ Fraud allegations
- ⚠️ Compliance breaches

---

## Project Structure

```
Enterprise-AI-Copilot/
├── app.py                    # Streamlit UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── data/                     # Business documents (TXT/PDF)
│   ├── employee_handbook.txt
│   ├── travel_policy.txt
│   ├── hr_policy.txt
│   ├── quarterly_report.txt
│   └── company_faq.txt
├── agent/
│   ├── agent.py              # LangChain ReAct agent
│   ├── prompts.py            # V1/V2/V3 prompts
│   ├── safety.py             # Safety guardrails
│   ├── logger.py             # Structured logging
│   └── config.py             # Configuration
├── rag/
│   ├── ingest.py             # PDF/TXT → ChromaDB
│   └── retriever.py          # Semantic search
├── tools/
│   ├── calculator.py         # Business calculator
│   ├── date_tool.py          # Date/quarter tool
│   └── search_docs.py        # Document search
├── memory/
│   └── conversation_memory.py
├── evaluation/
│   ├── test_cases.py         # 20 test cases
│   └── evaluator.py          # Evaluation framework
└── logs/
    ├── agent.log
    └── interactions.jsonl
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain 0.2 |
| LLM | OpenAI GPT-4o-mini |
| Vector Store | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| UI | Streamlit |
| Memory | ConversationBufferWindowMemory |
| Logging | loguru |
| Deployment | Docker / Streamlit |

---

## Author

**Afroze Ahmed**
Industry Capstone — Enterprise AI Operations Copilot
