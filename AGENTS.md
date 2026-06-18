# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

shopkeeper_brain is an AI-powered knowledge base pipeline that ingests product documents (PDF/Markdown), processes them through a LangGraph-based state machine, and stores results in Milvus (vectors), Neo4j (graph), MongoDB (documents), and MinIO (images).

## Architecture

The import pipeline is a **LangGraph StateGraph** defined in `knowledge/processor/import_process/main_graph.py`:

```
__start__ → entry_node → (PDF) pdf_to_md_node → md_img_node → __end__
                       → (MD)  md_img_node → __end__
```

- **`BaseNode`** (`base.py`): abstract node class. Subclasses implement `process(self, state) -> state`. The base `__call__` wraps `process` with logging and catches exceptions as `ImportProcessError`.
- **`ImportGraphState`** (`state.py`): TypedDict shared across all nodes — file paths, control flags, content, chunks.
- **`ImportConfig`** (`config.py`): dataclass singleton loaded from env vars via `python-dotenv` (`knowledge/.env`).
- **Nodes**: `EntryNode` (file type detection), `PdfToMdNode` (MinerU PDF→Markdown), `MarkDownImageNode` (VLM image description + MinIO upload), `Document_Spliter_Node` (chunking, WIP).
- **Utilities**: `VLMClient` (OpenAI-compatible vision model for image alt-text), `MinioClient` (S3-compatible object storage with public-read bucket policy).

Placeholder modules: `knowledge/api/`, `knowledge/schema/`, `knowledge/services/`, `knowledge/processor/query_process/`.

## Common Commands

```bash
# Activate virtual environment (Windows)
source knowledge/.venv/Scripts/activate

# Install dependencies
pip install -r knowledge/requirements.txt

# Run a single node test (each node has `if __name__ == "__main__"`)
python -m knowledge.processor.import_process.nodes.entry_node
python -m knowledge.processor.import_process.nodes.pdf_to_md_node
python -m knowledge.processor.import_process.nodes.md_img_node

# Run the full import pipeline
python knowledge/processor/import_process/main_graph.py

# Add a dependency
pip install <package> && pip freeze > knowledge/requirements.txt
```

## Key Dependencies

- **langgraph**: workflow orchestration (StateGraph with conditional routing)
- **mineru**: PDF-to-Markdown conversion (called via subprocess, must be CLI-available)
- **openai**: used for both LLM and VLM calls (compatible with any OpenAI-style API, e.g. DashScope)
- **minio**: S3-compatible object storage client
- **pymilvus, neo4j, pymongo**: vector, graph, and document databases (not yet wired into nodes)
- **sentence-transformers**: embedding generation

## Configuration

All config lives in `knowledge/.env`. The `.env` file contains **real credentials** and must not be committed (currently tracked — ensure `.gitignore` covers it). Key env vars:

- `OPENAI_API_BASE`, `OPENAI_API_KEY`: LLM API endpoint (DashScope by default)
- `VL_MODEL`: vision model name (e.g., `qwen3-vl-flash`)
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`
- `MILVUS_URL`, `NEO4J_URI`/`USERNAME`/`PASSWORD`, `MONGO_URL`
- `EMBEDDING_DIM`, `EMBEDDING_MODEL`

## Infrastructure (Docker Compose)

所有中间件部署在 **192.168.10.130** 虚拟机上，通过 Docker Compose 管理。配置文件位于资料包 `docker-compose脚本/docker-compose.yml`。

### 服务架构

```
etcd:2379 ← Milvus 元数据存储
minio:9000 (API) / :9001 (Console) ← Milvus 对象存储
milvus-standalone:19530 (gRPC) / :9091 (metrics) ← 向量数据库核心
attu:7000 ← Milvus 图形化管理界面 (zilliz/attu:v2.5.10)
neo4j:7474 (Browser) / :7687 (Bolt) ← 图数据库 (neo4j:2026.01.3)
```

网络统一为 `milvus` 桥接网络，容器间通过服务名互相访问。

### 服务详情

| 服务 | 容器名 | 镜像 | 端口 |
|------|--------|------|------|
| etcd | milvus-etcd | quay.io/coreos/etcd:v3.5.25 | 2379 |
| MinIO | milvus-minio | minio/minio:RELEASE.2024-12-18T13-15-44Z | 9000, 9001 |
| Milvus | milvus-standalone | milvusdb/milvus:v2.5.5 | 19530, 9091 |
| Attu | attu | zilliz/attu:v2.5.10 | 7000 |
| Neo4j | - | neo4j:2026.01.3 | 7474, 7687 |

### 关键凭证

- **MinIO**: `minioadmin` / `minioadmin`
- **Neo4j**: `neo4j` / `hzk123456`
- **etcd**: 无认证，直接连接 `etcd:2379`

### Milvus 依赖链

```
etcd ← ─ ─ ─ ┐
             ├── standalone (Milvus) ←── attu
minio ← ─ ─ ─ ┘
```

Milvus 使用 RocksMQ 作为消息队列（单机模式），无需额外部署 Pulsar/Kafka。向量指令集优化需要 `seccomp:unconfined`。

### .env 中对应连接配置

基于服务器 IP `192.168.10.130`，关键连接串应为：

- `MILVUS_URL=http://192.168.10.130:19530`
- `NEO4J_URI=bolt://192.168.10.130:7687`
- `MINIO_ENDPOINT=192.168.10.130:9000`
- `MINIO_ACCESS_KEY=minioadmin`
- `MINIO_SECRET_KEY=minioadmin`
- `NEO4J_USERNAME=neo4j`
- `NEO4J_PASSWORD=hzk123456`

## Code Conventions

- Python 3.12+, all code under `knowledge/` package
- Chinese log messages and comments
- Nodes are self-testable: each `.py` has a `if __name__ == "__main__"` block with sample state
- Commit messages in Chinese, concise and descriptive

## 行为准则

减少 LLM 常见编码错误。**权衡：** 以下准则偏向谨慎而非速度，简单任务可凭判断灵活处理。

### 1. 先想再写

**不要假设。不要隐藏困惑。主动暴露权衡。**

实施前：

- 明确陈述你的假设。不确定就问。
- 如果存在多种解释，列出它们——不要沉默地选一个。
- 如果有更简单的方案，说出来。必要时 push back。
- 如果不清楚，停下来。指出困惑点。问。

### 2. 简单优先

**用最少的代码解决问题。不写推测性代码。**

- 不添加需求之外的功能。
- 不对只使用一次的代码做抽象。
- 不为不可能发生的场景写错误处理。
- 不添加未被要求的"灵活性"或"可配置性"。
- 如果写了 200 行但其实 50 行就够了，重写。

自问："高级工程师会说这东西过度复杂吗？" 如果是，简化。

### 3. 手术式修改

**只碰必须碰的。只清理自己弄脏的。**

编辑已有代码时：

- 不要"改进"无关的代码、注释或格式。
- 不要重构没坏的东西。
- 匹配已有风格，即使你更喜欢另一种写法。
- 如果发现无关的死代码，提出来——不要直接删。

修改产生了孤儿代码时：

- 删除 YOUR 修改导致不再使用的 imports/variables/functions。
- 不要删除已有的死代码，除非被明确要求。

测试标准：每一行改动都能直接追溯到用户的需求。

### 4. 目标驱动执行

**定义成功标准。循环直到验证通过。**

把任务转化为可验证的目标：

- "加验证" → "先写无效输入测试，再让它们通过"
- "修 bug" → "先写复现测试，再修到通过"
- "重构 X" → "确保前后测试都通过"

多步骤任务，给出简短计划：

```
1. [步骤] → 验证: [方式]
2. [步骤] → 验证: [方式]
3. [步骤] → 验证: [方式]
```

强的成功标准让你能独立循环推进。弱的标准（"搞出来就行"）需要反复澄清。

---

**这些准则有效的标志：** diff 里不必要的改动更少，因过度复杂导致的返工更少，澄清问题出现在实现之前而非犯错之后。
