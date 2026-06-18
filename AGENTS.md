# AGENTS.md

This file gives coding agents the local operating context for ProductAssistant.

## Project Overview

ProductAssistant is a local AI knowledge-base assistant. It ingests product documents, converts and enriches them, stores searchable knowledge in local services, and answers questions through a web chat UI.

The current pipeline supports:

- Markdown import
- PDF import through MinerU
- optional Markdown image description and MinIO upload
- vector storage and retrieval through Milvus
- graph extraction and lookup through Neo4j
- chat history through MongoDB
- optional web search through Tavily
- OpenAI-compatible LLM/VLM providers

## Main Local Commands

Run these from the repository root.

```powershell
# Start infrastructure
docker compose up -d

# Check infrastructure
docker compose ps

# Install Python dependencies
pip install -r knowledge/requirements.txt

# Run the API
D:\App\Anaconda\envs\PA\python.exe -m uvicorn knowledge.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Useful local pages:

- API docs: `http://127.0.0.1:8000/docs`
- Import page: `http://127.0.0.1:8000/`
- Chat page: `http://127.0.0.1:8000/front/chat.html`
- Milvus Attu: `http://127.0.0.1:7000`
- MinIO console: `http://127.0.0.1:9001`
- Neo4j browser: `http://127.0.0.1:7474`

## Configuration

Runtime configuration is loaded from `knowledge/.env`. That file contains real credentials and must not be committed.

Use `knowledge/.env.example` as the safe template.

Important settings:

- `OPENAI_API_BASE`, `OPENAI_API_KEY`, `MODEL`, `ITEM_MODEL`: text LLM provider
- `VL_MODEL`: vision model for image descriptions
- `BGE_M3_PATH`, `BGE_DEVICE`, `BGE_FP16`: embedding model
- `MILVUS_URL`, `CHUNKS_COLLECTION`, `ITEM_NAME_COLLECTION`, `ENTITY_NAME_COLLECTION`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `MONGO_URL`, `MONGO_DB_NAME`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`
- `ENABLE_WEB_SEARCH`, `WEB_SEARCH_PROVIDER`, `TAVILY_API_KEY`: optional web search
- `MINERU_CMD`, `MINERU_BACKEND`, `MINERU_METHOD`, `MINERU_LANG`: PDF parsing
- `IMPORT_SMOKE_TEST`: set `true` only for infrastructure smoke tests

## Architecture

Import flow:

```text
entry_node
  -> pdf_to_md_node       # PDF only
  -> md_img_node
  -> document_split
  -> item_name_recognition
  -> bge_embedding
  -> import_milvus
  -> knowledge_graph
```

Query flow:

```text
item_name_confirm
  -> search_embedding
  -> search_embedding_hyde
  -> query_kg
  -> web_search_mcp       # optional Tavily/DashScope web search
  -> rrf
  -> rerank
  -> answer_output
```

The `web_search_mcp` node name is historical. It now supports Tavily as the default provider when configured.

## Development Notes

- Keep `.env`, runtime uploads, generated import artifacts, caches, and IDE files out of Git.
- Prefer small, focused edits. This project has several long-running model and database steps, so preserve working behavior unless a change is clearly needed.
- After changing import or query nodes, run at least:

```powershell
D:\App\Anaconda\envs\PA\python.exe -m py_compile knowledge\processor\import_process\nodes\*.py
D:\App\Anaconda\envs\PA\python.exe -m py_compile knowledge\processor\query_process\nodes\*.py
```

- Before pushing, run `git status --short` and verify no secrets are staged.
