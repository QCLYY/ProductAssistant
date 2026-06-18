# ProductAssistant

ProductAssistant 是一个本地运行的 AI 产品知识库助手。它可以导入 Markdown / PDF 产品资料，构建本地可检索知识库，并通过浏览器聊天页面进行问答；同时也支持 Tavily 联网搜索，用于补充本地知识库之外的信息。

这个项目当前更偏向“个人产品知识助手 / 产品资料问答系统”的原型：上传什么资料，就可以围绕对应内容进行咨询，不局限于维修、技术顾问或某一种固定场景。

## 核心能力

- 导入 Markdown 文档
- 通过 MinerU 将 PDF 转换为 Markdown
- 处理 Markdown 中的图片，并可上传到 MinIO
- 对文档进行切分
- 使用 BGE-M3 生成向量
- 将文本向量写入 Milvus
- 使用大模型抽取知识图谱，并写入 Neo4j
- 使用 MongoDB 保存聊天历史
- 通过浏览器聊天页进行本地知识库问答
- 通过 Tavily 进行联网搜索补充
- 支持 OpenAI 兼容格式的大模型接口

## 技术栈

- Python 3.12
- FastAPI + Uvicorn
- LangGraph
- Milvus + Attu
- Neo4j
- MongoDB
- MinIO
- BGE-M3 / FlagEmbedding
- MinerU
- OpenAI-compatible LLM API
- Tavily Search API
- Docker Compose

## 项目结构

```text
ProductAssistant/
  docker-compose.yml              # 本地中间件服务
  README.md
  AGENTS.md
  knowledge/
    api/                          # FastAPI 接口
    front/                        # 前端页面
    processor/
      import_process/             # 文档导入流程
      query_process/              # 问答查询流程
    services/                     # 应用服务
    tools/                        # Milvus / Neo4j / LLM 等工具
    utils/                        # MinIO / VLM 等辅助工具
    requirements.txt
    .env.example                  # 环境变量模板
```

## 运行前准备

### 1. Python 环境

建议使用 Conda 或虚拟环境。当前本地验证环境为：

```text
Python 3.12
Conda 环境名：PA
```

安装依赖：

```powershell
pip install -r knowledge/requirements.txt
```

### 2. Docker 服务

项目依赖 Milvus、MinIO、Neo4j、MongoDB 等本地服务。

启动：

```powershell
docker compose up -d
```

检查：

```powershell
docker compose ps
```

常用管理页面：

- Attu / Milvus 管理：`http://127.0.0.1:7000`
- MinIO 控制台：`http://127.0.0.1:9001`
- Neo4j Browser：`http://127.0.0.1:7474`

## 配置环境变量

复制模板：

```powershell
copy knowledge\.env.example knowledge\.env
```

然后编辑：

```text
knowledge/.env
```

重要配置分组如下。

### 大模型配置

```env
OPENAI_API_BASE=
OPENAI_API_KEY=
MODEL=
ITEM_MODEL=
VL_MODEL=
```

含义：

- `MODEL`：主要问答、知识抽取模型
- `ITEM_MODEL`：用于识别文档或问题中的产品/主体名称
- `VL_MODEL`：用于图片理解，只有处理图片时才真正需要

### 向量模型配置

```env
BGE_M3_PATH=BAAI/bge-m3
BGE_DEVICE=cpu
BGE_FP16=False
```

如果本机已经下载了 BGE-M3，也可以把 `BGE_M3_PATH` 改成本地模型路径。

### Tavily 联网搜索

```env
ENABLE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=
TAVILY_API_URL=https://api.tavily.com/search
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH=basic
```

如果只想查询本地知识库，可以关闭联网搜索：

```env
ENABLE_WEB_SEARCH=false
```

### 数据库和对象存储

默认使用本地 Docker 服务：

```env
MILVUS_URL=http://127.0.0.1:19530
NEO4J_URI=bolt://127.0.0.1:7687
MONGO_URL=mongodb://admin:123456@127.0.0.1:27017
MINIO_ENDPOINT=127.0.0.1:9000
```

### PDF 解析

```env
MINERU_CMD=
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_LANG=ch
```

如果命令行找不到 `mineru`，可以把 `MINERU_CMD` 指向当前 Python 环境中的 `mineru.exe`。

## 启动 API

在项目根目录执行：

```powershell
D:\App\Anaconda\envs\PA\python.exe -m uvicorn knowledge.api.app:app --host 127.0.0.1 --port 8000 --reload
```

打开页面：

- 导入页面：`http://127.0.0.1:8000/`
- 聊天页面：`http://127.0.0.1:8000/front/chat.html`
- API 文档：`http://127.0.0.1:8000/docs`

## 推荐验证流程

### 1. 验证 API 页面

打开：

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/
http://127.0.0.1:8000/front/chat.html
```

都能正常访问，说明 API 和静态页面启动成功。

### 2. 验证 Markdown 导入

上传一个简单 Markdown，例如：

```markdown
# 测试产品A

测试产品A是一款用于验证知识库导入流程的设备。

## 使用方法

按下电源键 3 秒即可启动设备。

请勿在潮湿环境中使用。
```

导入页面显示 `已完成 7/7`，说明导入流程跑通。

### 3. 验证本地知识库问答

在聊天页提问：

```text
测试产品A是什么？
```

预期：能够基于刚导入的文档回答。

### 4. 验证 Tavily 联网搜索

在聊天页提问：

```text
OpenAI 官方网站是什么？
```

预期：能够通过 Tavily 搜索并回答。

## 当前已验证状态

本项目当前已经在本地验证过：

- Docker 中间件服务全部启动
- Markdown 导入完整跑通
- Milvus 向量写入可用
- Neo4j 图数据库连接可用
- 本地知识库问答可用
- Tavily 联网搜索可用
- `knowledge/.env` 未被 Git 跟踪

## 注意事项

- `knowledge/.env` 包含真实 API key 和密码，不能提交到 GitHub。
- `knowledge/.env.example` 只保存占位配置，可以提交。
- `IMPORT_SMOKE_TEST=true` 只用于基础流程测试；真实导入时应使用：

```env
IMPORT_SMOKE_TEST=false
```

- `BGE_RERANKER_LARGE` 是可选配置。为空时，重排序会自动降级，不影响基础问答。
- 视觉模型是否可用取决于具体模型平台是否支持 OpenAI 兼容的图片输入。

## 后续计划

- 优化聊天页 UI，使其更接近正式产品风格
- 统一上传页和聊天页的视觉设计
- 增强 PDF / 图片资料的处理效果
- 增加更多搜索源或搜索源切换能力
- 完善知识图谱展示和调试页面
