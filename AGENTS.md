# AGENTS.md

这个文件给后续参与本项目的编码助手提供本地上下文。当前项目仓库名是 `ProductAssistant`，前端产品名使用“品辅”。

## 项目概览

品辅是一个本地优先的产品资料问答助手。它可以导入产品资料，把资料转换、切分、向量化并写入本地索引，然后通过浏览器聊天页面回答问题。

当前主要能力：

- 导入 Markdown 资料
- 导入 PDF 资料，并通过 MinerU 转换为 Markdown
- 导入 ZIP，其中 ZIP 会解压后寻找第一个 Markdown 文件作为导入入口
- 可选处理 Markdown 图片，并上传到本地对象存储
- 写入本地向量索引，用于资料检索
- 抽取知识关联，用于辅助问答
- 使用 MongoDB 保存聊天历史
- 可选联网搜索，用于本地资料查不到时补充回答
- 支持 OpenAI 兼容格式的大模型、视觉模型接口

当前产品逻辑：

- 用户可以在上传页上传 PDF、Markdown 或 ZIP。
- 聊天逻辑是本地优先：优先基于已上传资料回答；本地资料查不到时，才根据开关决定是否联网搜索。
- 普通用户界面尽量不暴露底层服务名称，统一使用“本地资料”“本地搜索”“联网搜索”“知识关联”“本地索引”等产品化说法。

## 常用本地命令

以下命令都在项目根目录执行：

```powershell
# 启动 Docker 基础服务
docker compose up -d

# 查看 Docker 服务状态
docker compose ps

# 安装 Python 依赖
pip install -r knowledge/requirements.txt

# 启动 API
D:\App\Anaconda\envs\PA\python.exe -m uvicorn knowledge.api.app:app --host 127.0.0.1 --port 8000 --reload
```

常用页面：

- API 文档：`http://127.0.0.1:8000/docs`
- 上传页 / 本地资料管理页：`http://127.0.0.1:8000/`
- 聊天页：`http://127.0.0.1:8000/front/chat.html`
- 本地资料接口：`http://127.0.0.1:8000/materials`
- 配置检查接口：`http://127.0.0.1:8000/config/check`
- 本地索引管理页面：`http://127.0.0.1:7000`
- 对象存储控制台：`http://127.0.0.1:9001`
- 知识关联数据库页面：`http://127.0.0.1:7474`

## 配置说明

运行配置从 `knowledge/.env` 读取。这个文件包含真实密钥，不能提交到 Git。

重要配置项：

- `OPENAI_API_BASE`、`OPENAI_API_KEY`、`MODEL`、`ITEM_MODEL`：文本模型配置
- `VL_MODEL`：视觉模型配置
- `BGE_M3_PATH`、`BGE_DEVICE`、`BGE_FP16`：本地向量模型配置
- `MILVUS_URL`、`CHUNKS_COLLECTION`、`ITEM_NAME_COLLECTION`、`ENTITY_NAME_COLLECTION`：本地索引配置
- `NEO4J_URI`、`NEO4J_USERNAME`、`NEO4J_PASSWORD`、`NEO4J_DATABASE`：知识关联配置
- `MONGO_URL`、`MONGO_DB_NAME`：聊天历史配置
- `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET_NAME`：对象存储配置
- `ENABLE_WEB_SEARCH`、`WEB_SEARCH_PROVIDER`、`TAVILY_API_KEY`：联网搜索配置
- `MINERU_CMD`、`MINERU_BACKEND`、`MINERU_METHOD`、`MINERU_LANG`：PDF 解析配置
- `IMPORT_SMOKE_TEST`：只在基础服务联通性测试时临时设为 `true`

## 嵌入模型说明

`knowledge/tools/embedding_utils.py` 当前通过 `FlagEmbedding.BGEM3FlagModel` 做 BGE-M3 适配，并对外保持 `encode_documents` / `encode_queries` 的兼容接口。

注意：不要在模块导入阶段直接使用 `pymilvus.model.hybrid.BGEM3EmbeddingFunction`。当前 Windows + Conda 环境下，它可能在 API 启动时触发 `onnxruntime` access violation，导致进程直接崩溃。

## 架构流程

导入流程：

```text
entry_node
  -> pdf_to_md_node       # PDF 文件才会走这一步
  -> md_img_node
  -> document_split
  -> item_name_recognition
  -> bge_embedding
  -> import_milvus
  -> knowledge_graph
```

查询流程：

```text
item_name_confirm
  -> search_embedding
  -> search_embedding_hyde
  -> query_kg
  -> web_search_mcp       # 可选联网搜索，节点名是历史遗留
  -> rrf
  -> rerank
  -> answer_output
```

说明：

- `web_search_mcp` 是历史遗留节点名，现在默认适配 Tavily。
- `item_name` 是历史字段名，现在实际含义更接近“资料主体名称”或“产品/资料名称”，暂时不要贸然重命名数据库字段。

## 管理接口

```text
GET    /materials       # 查看已经写入本地索引的资料
DELETE /materials       # 按 file_title / material_name 删除某份资料的本地索引
GET    /config/check    # 检查配置和本地服务连通性，不返回密钥原文
```

## 开发注意事项

- 不要提交 `knowledge/.env`、运行时上传文件、生成的导入中间文件、缓存、IDE 私有文件。
- 优先做小而明确的改动，不要为了改名大范围重构数据库字段。
- 用户界面优先使用产品化说法，避免直接展示底层组件名。
- 修改导入或查询节点后，至少运行：

```powershell
D:\App\Anaconda\envs\PA\python.exe -m py_compile knowledge\processor\import_process\nodes\*.py
D:\App\Anaconda\envs\PA\python.exe -m py_compile knowledge\processor\query_process\nodes\*.py
```

- 修改 API 路由或公共服务后，至少运行：

```powershell
D:\App\Anaconda\envs\PA\python.exe -m py_compile knowledge\api\app.py knowledge\api\query_router.py knowledge\api\material_router.py knowledge\services\material_service.py
D:\App\Anaconda\envs\PA\python.exe -c "from knowledge.api.app import app; print(len(app.routes))"
```

- 推送前运行 `git status --short`，确认没有密钥或无关文件被加入 Git。
