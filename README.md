# ProductAssistant（品辅）

ProductAssistant，中文名为“品辅”，是一个结合本地与联网的 AI 产品资料问答助手。

它的核心功能是：上传任意产品资料或参考资料，例如 PDF、Markdown、ZIP 文档包，围绕这些资料进行问答。
项目默认采用“本地资料优先”的检索逻辑：优先检索已经上传的本地资料；如果本地资料没有查到，并且用户开启了联网搜索，再通过联网搜索补充答案。

项目具备完整的资料导入、资料管理、本地问答、联网补充、流式输出和浏览器页面交互能力。

## 核心能力

- 支持上传 Markdown、PDF、ZIP。
- PDF 可通过 MinerU 转换为 Markdown。
- Markdown 图片可进行处理，并上传到本地对象存储。
- 文档会被切分为片段，并写入本地向量索引。
- 支持使用 BGE-M3 生成文本向量。
- 支持基于大模型抽取资料中的实体关系，并写入本地知识关联服务。
- 支持本地资料清单查看和资料删除。
- 支持配置检查，方便确认本地服务和关键配置是否可用。
- 支持聊天页本地资料问答。
- 支持开启或关闭本地搜索。
- 支持开启或关闭联网搜索。
- 支持 Tavily 作为联网搜索源。
- 支持流式输出。
- 支持展示回答来源，便于判断答案来自本地资料还是联网补充。

## 问答逻辑

当前问答逻辑以“本地资料优先”为主：

1. 用户提问。
2. 如果开启了本地搜索，系统会先检索已经上传的本地资料。
3. 如果本地资料查到了相关内容，优先基于本地资料回答。
4. 如果本地资料没有查到，并且开启了联网搜索，系统再调用联网搜索补充。
5. 如果本地和联网都没有可用结果，系统会明确说明没有找到相关资料。
6. 本地结果和联网结果会尽量去重，避免重复内容堆叠。

## 技术栈

- Python 3.12
- FastAPI + Uvicorn
- LangGraph
- Docker Compose
- Milvus / Attu
- Neo4j
- MongoDB
- MinIO
- BGE-M3 / FlagEmbedding
- MinerU
- OpenAI 兼容格式大模型接口
- Tavily Search API

## 项目结构

```text
ProductAssistant/
  README.md
  AGENTS.md
  docker-compose.yml
  knowledge/
    api/                          # FastAPI 接口
    front/                        # 浏览器页面
    services/                     # 资料管理、导入服务等
    processor/
      import_process/             # 资料导入流程
      query_process/              # 问答查询流程
    tools/                        # 向量、图数据库、LLM、重排序等工具
    utils/                        # MinIO、SSE、任务状态等工具
    requirements.txt              # Python 依赖
    .env.example                  # 环境变量模板
```

## 运行前准备

### 1. 准备 Python 环境

建议使用 Conda 或 PyCharm 绑定的虚拟环境。

当前本地验证环境：

```text
Python 3.12
Conda 环境名：PA
```

安装依赖：

```powershell
pip install -r knowledge/requirements.txt
```

### 2. 准备环境变量

复制环境变量模板：

```powershell
copy knowledge\.env.example knowledge\.env
```

然后编辑：

```text
knowledge/.env
```

`knowledge/.env` 用于保存真实 API Key、模型配置和本地服务连接信息。这个文件已经被 `.gitignore` 忽略，不应该提交到 GitHub。

### 3. 启动本地 Docker 服务

项目依赖本地中间件服务，使用 Docker Compose 启动：

```powershell
docker compose up -d
```

查看服务状态：

```powershell
docker compose ps
```

常用管理页面：

- Attu：`http://127.0.0.1:7000`
- MinIO：`http://127.0.0.1:9001`
- Neo4j Browser：`http://127.0.0.1:7474`

## 关键配置说明

### 大模型配置

```env
OPENAI_API_BASE=
OPENAI_API_KEY=
MODEL=
ITEM_MODEL=
VL_MODEL=
```

说明：

- `OPENAI_API_BASE`：OpenAI 兼容接口地址。
- `OPENAI_API_KEY`：大模型 API Key。
- `MODEL`：主要问答和知识抽取模型。
- `ITEM_MODEL`：资料主体识别模型，可以和 `MODEL` 使用同一个。
- `VL_MODEL`：视觉模型，用于处理图片内容；如果暂时不处理图片，可以先使用支持图片输入的模型或后续再完善。

### 向量模型配置

```env
BGE_M3_PATH=BAAI/bge-m3
BGE_DEVICE=cpu
BGE_FP16=False
```

说明：

- `BGE_M3_PATH`：BGE-M3 模型名称或本地模型路径。
- `BGE_DEVICE`：CPU 环境使用 `cpu`，有合适显卡时可改为 `cuda`。
- `BGE_FP16`：CPU 环境建议使用 `False`。

### 联网搜索配置

```env
ENABLE_WEB_SEARCH=false
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=
TAVILY_API_URL=https://api.tavily.com/search
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH=basic
```

说明：

- `ENABLE_WEB_SEARCH`：默认是否启用联网搜索。
- `WEB_SEARCH_PROVIDER`：当前适配 Tavily。
- `TAVILY_API_KEY`：Tavily 的 API Key。
- `TAVILY_MAX_RESULTS`：每次搜索返回结果数量。
- `TAVILY_SEARCH_DEPTH`：搜索深度，通常 `basic` 即可。

### 本地服务配置

默认连接 Docker Compose 启动的本地服务：

```env
MILVUS_URL=http://127.0.0.1:19530
NEO4J_URI=bolt://127.0.0.1:7687
MONGO_URL=mongodb://<username>:<password>@127.0.0.1:27017
MINIO_ENDPOINT=127.0.0.1:9000
```

这些默认账号密码只适合本地开发。正式部署时应改为更安全的配置。

### PDF 解析配置

```env
MINERU_CMD=
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_LANG=ch
```

如果命令行找不到 `mineru`，可以把 `MINERU_CMD` 指向当前 Python 环境中的 `mineru.exe`。

## 启动项目

在项目根目录执行：

```powershell
D:\App\Anaconda\envs\PA\python.exe -m uvicorn knowledge.api.app:app --host 127.0.0.1 --port 8000 --reload
```

如果已经在 PyCharm 中配置了运行项，也可以直接运行 `ProductAssistant API`。

启动成功后打开：

- 上传和资料管理页面：`http://127.0.0.1:8000/`
- 聊天页面：`http://127.0.0.1:8000/front/chat.html`
- API 文档：`http://127.0.0.1:8000/docs`

## 推荐验证流程

### 1. 验证服务启动

打开以下页面：

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/front/chat.html
http://127.0.0.1:8000/docs
```

如果三个页面都能正常打开，说明 API 和静态页面已经启动成功。

### 2. 验证配置检查

在上传页面查看“配置检查”区域，确认本地索引、对象存储、知识关联、历史记录等服务是否正常。

### 3. 验证资料导入

上传一个 Markdown、PDF 或 ZIP 文件。

导入成功时，页面会显示完整的阶段进度，例如：

- Markdown 图片处理
- 文档切分
- 资料主体识别
- 向量生成
- 写入本地资料索引
- 生成知识关联

### 4. 验证本地资料清单

导入完成后，在上传页面查看“本地资料管理”区域。

这里应该能看到已经写入本地索引的资料名称、识别出的主体名称和片段数量。

### 5. 验证本地问答

进入聊天页面，开启“本地搜索”，关闭“联网搜索”，然后提问：

```text
现在有哪些本地资料？
```

或者围绕刚上传的资料提问。

如果本地资料中有答案，系统应优先基于本地资料回答；如果没有查到，应说明本地资料中没有找到相关内容。

### 6. 验证联网补充

进入聊天页面，同时开启“本地搜索”和“联网搜索”。

当本地资料没有查到时，系统会尝试通过 Tavily 联网搜索补充答案。

## 隐私和安全注意事项

- 不要提交 `knowledge/.env`。
- 不要把真实 API Key、数据库密码、服务 Token 写进 README 或代码。
- 可以提交 `knowledge/.env.example`，但只能放占位值或本地开发默认值。
- 上传到 GitHub 前建议执行：

```powershell
git status --ignored --short knowledge\.env .env
git ls-files .env knowledge/.env
```

正常情况下，真实 `.env` 文件应该显示为被忽略，且不应该出现在 `git ls-files` 结果里。

## 当前已验证状态

本项目目前已经在本地验证过：

- Docker Compose 服务可启动。
- API 服务可启动。
- 上传页面可打开。
- 聊天页面可打开。
- API 文档可打开。
- Markdown 资料导入可跑通。
- PDF 资料导入可跑通。
- 本地资料清单可查看。
- 本地资料可删除。
- 本地资料优先问答可用。
- Tavily 联网搜索已适配。
- `.env` 未被 Git 跟踪。

## 后续可优化方向

- 进一步优化聊天页视觉效果和移动端适配。
- 增强 PDF 解析失败时的错误提示。
- 增加资料重命名、资料标签、资料分组。
- 增加更清晰的引用来源展示。
- 增加会话记忆和历史会话管理。
- 增加批量删除和重新索引能力。
- 增加 Docker 一键启动 API 的配置。
- 清理历史遗留命名和旧代码路径。
