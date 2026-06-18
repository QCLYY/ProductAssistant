# ProductAssistant

ProductAssistant is a local AI knowledge-base assistant for product documents. Upload Markdown or PDF files, build a searchable local knowledge base, and ask questions through a browser chat UI.

## What It Does

- Imports Markdown and PDF documents
- Converts PDF files with MinerU
- Splits documents into searchable chunks
- Generates BGE-M3 embeddings
- Stores vectors in Milvus
- Extracts graph knowledge into Neo4j
- Stores chat history in MongoDB
- Stores document images in MinIO
- Answers questions from local knowledge
- Optionally enriches answers with Tavily web search

## Stack

- Python 3.12
- FastAPI + Uvicorn
- LangGraph
- Milvus + Attu
- Neo4j
- MongoDB
- MinIO
- BGE-M3 / FlagEmbedding
- OpenAI-compatible LLM APIs
- Tavily Search API
- Docker Compose

## Project Structure

```text
ProductAssistant/
  docker-compose.yml
  knowledge/
    api/                       # FastAPI routes
    front/                     # browser pages
    processor/
      import_process/          # document import pipeline
      query_process/           # question answering pipeline
    services/
    tools/
    utils/
    requirements.txt
    .env.example
```

## Quick Start

### 1. Start Local Services

```powershell
docker compose up -d
docker compose ps
```

Expected services:

- Milvus: `127.0.0.1:19530`
- Attu: `http://127.0.0.1:7000`
- MinIO: `http://127.0.0.1:9001`
- Neo4j: `http://127.0.0.1:7474`
- MongoDB: `127.0.0.1:27017`

### 2. Create Environment File

Create `knowledge/.env` from `knowledge/.env.example`.

Required groups:

- LLM: `OPENAI_API_BASE`, `OPENAI_API_KEY`, `MODEL`, `ITEM_MODEL`
- Embedding: `BGE_M3_PATH`, `BGE_DEVICE`, `BGE_FP16`
- Databases: Milvus, Neo4j, MongoDB, MinIO settings
- Optional web search: `ENABLE_WEB_SEARCH`, `WEB_SEARCH_PROVIDER`, `TAVILY_API_KEY`

Do not commit `knowledge/.env`.

### 3. Install Python Dependencies

```powershell
pip install -r knowledge/requirements.txt
```

### 4. Run API

```powershell
D:\App\Anaconda\envs\PA\python.exe -m uvicorn knowledge.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- Import page: `http://127.0.0.1:8000/`
- Chat page: `http://127.0.0.1:8000/front/chat.html`
- API docs: `http://127.0.0.1:8000/docs`

## Current Local Validation

The project has been validated locally with:

- Docker middleware running and healthy
- Markdown import completing all import stages
- local knowledge-base question answering
- Tavily web search answering non-local questions

## Useful Settings

```env
IMPORT_SMOKE_TEST=false
ENABLE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=tavily
BGE_DEVICE=cpu
BGE_FP16=False
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
```

Set `IMPORT_SMOKE_TEST=true` only when you want to test the infrastructure without real LLM/BGE processing.

## Notes

- `knowledge/.env` is ignored by Git because it contains API keys and passwords.
- The reranker model is optional. If `BGE_RERANKER_LARGE` is empty, query reranking gracefully falls back to the original order.
- Vision model support depends on whether the configured OpenAI-compatible provider supports image inputs.
