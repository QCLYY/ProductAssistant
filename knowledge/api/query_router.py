"""知识库查询 API 路由

POST /query         — 提交查询（流式/非流式）
GET  /stream/{sid}  — SSE 实时推送
GET  /history/{sid} — 查询会话历史
DELETE /history/{sid} — 清除会话历史
"""

import uuid
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, Request

logger = logging.getLogger("query.api")
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from knowledge.utils.task_utils import (
    update_task_status,
    add_done_task,
    set_task_result,
    get_task_result,
    clear_task_progress,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
)
from knowledge.utils.sse_util import (
    create_sse_queue,
    push_sse_event,
    sse_generator,
    SSEEvent,
)
from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.mongo_history_utils import (
    get_recent_messages,
    clear_history,
)
from knowledge.processor.query_process.main_graph import query_app


# ================================================================== #
#                    Pydantic 请求/响应模型                             #
# ================================================================== #

class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，留空自动生成")
    is_stream: bool = Field(False, description="是否流式返回")
    use_local_search: bool = Field(True, description="是否启用本地资料检索")
    use_web_search: bool = Field(True, description="是否启用联网搜索")


class HistoryItem(BaseModel):
    role: str
    text: str
    rewritten_query: str = ""
    item_names: list = []


# ================================================================== #
#                    FastAPI 应用                                       #
# ================================================================== #

def create_query_app() -> FastAPI:
    app = FastAPI(title="Query Service", description="知识库查询服务")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


def _register_routes(app: FastAPI):

    @app.post("/query")
    async def query(request: QueryRequest, background_tasks: BackgroundTasks):
        """提交查询请求。流式模式走后台+SSE，非流式同步返回。"""
        user_query = request.query
        session_id = request.session_id or str(uuid.uuid4())
        is_stream = request.is_stream
        use_local_search = request.use_local_search
        use_web_search = request.use_web_search

        update_task_status(session_id, TASK_STATUS_PROCESSING)

        if is_stream:
            create_sse_queue(session_id)
            background_tasks.add_task(
                _run_query_graph,
                session_id,
                user_query,
                is_stream,
                use_local_search,
                use_web_search,
            )
            await asyncio.sleep(0.1)
            return {"message": "Query submitted", "session_id": session_id, "task_id": session_id}
        else:
            _run_query_graph(
                session_id,
                user_query,
                is_stream,
                use_local_search,
                use_web_search,
            )
            answer = get_task_result(session_id, "answer", "")
            return {
                "message": "处理完成",
                "session_id": session_id,
                "answer": answer,
            }

    @app.get("/stream/{session_id}")
    async def stream(session_id: str, request: Request):
        """SSE 实时返回查询结果。"""
        return StreamingResponse(
            sse_generator(session_id, request),
            media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def history(session_id: str, limit: int = 50):
        """查询当前会话历史记录。"""
        records = get_recent_messages(session_id, limit=limit)
        items = [
            {
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "ts": r.get("timestamp", ""),
            }
            for r in records
        ]
        return {"session_id": session_id, "items": items}

    @app.delete("/history/{session_id}")
    async def clear_chat_history(session_id: str):
        """清除会话历史记录。"""
        count = clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


# ================================================================== #
#                    节点日志提取                                       #
# ================================================================== #

def _print_state(node_name: str, state: dict):
    """打印节点完成后的 state 摘要到控制台。"""
    lines = [f"\n{'='*50}", f"  [{node_name}]"]
    fields = [
        ("item_names", state.get("item_names")),
        ("rewritten_query", state.get("rewritten_query", "")),
        ("embedding_chunks", len(state.get("embedding_chunks", []))),
        ("hyde_embedding_chunks", len(state.get("hyde_embedding_chunks", []))),
        ("kg_chunks", len(state.get("kg_chunks", []))),
        ("kg_entities", state.get("kg_entities")),
        ("kg_aligned_entities", state.get("kg_aligned_entities")),
        ("kg_triples", len(state.get("kg_triples", []))),
        ("web_search_docs", len(state.get("web_search_docs", []))),
        ("rrf_chunks", len(state.get("rrf_chunks", []))),
        ("reranked_docs", len(state.get("reranked_docs", []))),
        ("answer", (state.get("answer", "") or "")[:100]),
    ]
    for label, val in fields:
        if val or val == 0:
            lines.append(f"  {label}: {val}")
    lines.append("=" * 50)
    print("\n".join(lines), flush=True)

def _build_node_log(node_name: str, result: dict) -> str:
    """从节点返回结果中提取关键信息，生成一行摘要日志。"""
    extractors = {
        "item_name_confirm": lambda r: (
            f"商品名: {r.get('item_names', [])} | 改写: {r.get('rewritten_query', '')[:50]}"
        ),
        "search_embedding": lambda r: (
            f"向量检索: {len(r.get('embedding_chunks', []))} 条"
        ),
        "search_embedding_hyde": lambda r: (
            f"HyDE检索: {len(r.get('hyde_embedding_chunks', []))} 条"
        ),
        "query_kg": lambda r: (
            f"KG: 实体{r.get('kg_entities', [])} → 对齐{r.get('kg_aligned_entities', [])} | "
            f"切片{len(r.get('kg_chunks', []))}条 三元组{len(r.get('kg_triples', []))}条"
        ),
        "web_search_mcp": lambda r: (
            f"网页搜索: {len(r.get('web_search_docs', []))} 条"
        ),
        "rrf": lambda r: (
            f"RRF融合: {len(r.get('rrf_chunks', []))} 条"
        ),
        "rerank": lambda r: (
            f"重排序: {len(r.get('reranked_docs', []))} 条"
        ),
        "answer_output": lambda r: (
            f"答案生成完成 ({len(r.get('answer', ''))} 字)"
        ),
    }
    fn = extractors.get(node_name)
    return fn(result) if fn else ""


# ================================================================== #
#                    后台查询执行                                       #
# ================================================================== #

def _run_query_graph(
        session_id: str,
        user_query: str,
        is_stream: bool,
        use_local_search: bool = True,
        use_web_search: bool = True,
):
    """后台任务：执行 LangGraph 查询流程图。"""
    clear_task_progress(session_id)  # 清除上次的进度，保留 status
    try:
        default_state = {
            "original_query": user_query,
            "session_id": session_id,
            "task_id": session_id,
            "is_stream": is_stream,
            "use_local_search": use_local_search,
            "use_web_search": use_web_search,
            "web_search_attempted": False,
        }

        final_state = None
        for event in query_app.stream(default_state):
            for node_name, node_result in event.items():
                add_done_task(session_id, node_name)
                final_state = node_result
                if is_stream and isinstance(node_result, dict):
                    log_msg = _build_node_log(node_name, node_result)
                    if log_msg:
                        push_sse_event(session_id, "node_log", {"msg": log_msg})

                # 打印每个节点完成后的 state 到控制台
                if isinstance(node_result, dict) and node_name not in ("multi_search", "join"):
                    _print_state(node_name, node_result)

        if final_state:
            answer = final_state.get("answer", "")
            set_task_result(session_id, "answer", answer)
            if is_stream:
                update_task_status(session_id, TASK_STATUS_COMPLETED)
                add_done_task(session_id, "answer_output")
                push_sse_event(
                    session_id, SSEEvent.FINAL,
                    {"answer": answer, "status": "completed"},
                )
            else:
                update_task_status(session_id, TASK_STATUS_COMPLETED)

    except Exception as e:
        if is_stream:
            push_sse_event(
                session_id, SSEEvent.FINAL,
                {"answer": f"查询出错: {e}", "status": "failed"},
            )
        update_task_status(session_id, "failed")


# ================================================================== #
#                    启动入口                                           #
# ================================================================== #

app = create_query_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
