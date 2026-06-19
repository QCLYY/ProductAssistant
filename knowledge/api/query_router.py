"""Query API routes for 品辅."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from knowledge.processor.query_process.main_graph import query_app
from knowledge.tools.mongo_history_utils import clear_history, get_recent_messages
from knowledge.utils.sse_util import (
    SSEEvent,
    create_sse_queue,
    push_sse_event,
    sse_generator,
)
from knowledge.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    clear_task_progress,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    set_task_result,
    update_task_status,
)

logger = logging.getLogger("query.api")


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话 ID，留空自动生成")
    is_stream: bool = Field(False, description="是否流式返回")
    use_local_search: bool = Field(True, description="是否启用本地资料搜索")
    use_web_search: bool = Field(True, description="是否启用联网搜索")


class HistoryItem(BaseModel):
    role: str
    text: str
    rewritten_query: str = ""
    item_names: list = []


def create_query_app() -> FastAPI:
    app = FastAPI(title="品辅查询服务", description="本地优先的资料问答服务")
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
        user_query = request.query
        session_id = request.session_id or str(uuid.uuid4())

        update_task_status(session_id, TASK_STATUS_PROCESSING)

        if request.is_stream:
            create_sse_queue(session_id)
            background_tasks.add_task(
                _run_query_graph,
                session_id,
                user_query,
                request.is_stream,
                request.use_local_search,
                request.use_web_search,
            )
            await asyncio.sleep(0.1)
            return {
                "message": "Query submitted",
                "session_id": session_id,
                "task_id": session_id,
            }

        _run_query_graph(
            session_id,
            user_query,
            request.is_stream,
            request.use_local_search,
            request.use_web_search,
        )
        return {
            "message": "处理完成",
            "session_id": session_id,
            "answer": get_task_result(session_id, "answer", ""),
            "sources": get_task_result(session_id, "sources", []),
            "done_list": get_done_task_list(session_id),
            "running_list": get_running_task_list(session_id),
        }

    @app.get("/stream/{session_id}")
    async def stream(session_id: str, request: Request):
        return StreamingResponse(
            sse_generator(session_id, request),
            media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def history(session_id: str, limit: int = 50):
        records = get_recent_messages(session_id, limit=limit)
        items = [
            {
                "role": record.get("role", ""),
                "text": record.get("text", ""),
                "ts": record.get("timestamp", ""),
            }
            for record in records
        ]
        return {"session_id": session_id, "items": items}

    @app.delete("/history/{session_id}")
    async def clear_chat_history(session_id: str):
        count = clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def _print_state(node_name: str, state: dict):
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
    lines = [f"\n{'=' * 50}", f"  [{node_name}]"]
    for label, value in fields:
        if value or value == 0:
            lines.append(f"  {label}: {value}")
    lines.append("=" * 50)
    print("\n".join(lines), flush=True)


def _build_node_log(node_name: str, result: dict) -> str:
    extractors = {
        "item_name_confirm": lambda r: (
            f"问题主体: {r.get('item_names', [])} | 改写: {r.get('rewritten_query', '')[:50]}"
        ),
        "search_embedding": lambda r: (
            f"本地资料检索: {len(r.get('embedding_chunks', []))} 条"
        ),
        "search_embedding_hyde": lambda r: (
            f"增强检索: {len(r.get('hyde_embedding_chunks', []))} 条"
        ),
        "query_kg": lambda r: (
            f"知识关联: 实体 {r.get('kg_entities', [])} | "
            f"片段 {len(r.get('kg_chunks', []))} 条，关系 {len(r.get('kg_triples', []))} 条"
        ),
        "web_search_mcp": lambda r: (
            f"联网搜索: {len(r.get('web_search_docs', []))} 条"
        ),
        "rrf": lambda r: (
            f"结果融合: {len(r.get('rrf_chunks', []))} 条"
        ),
        "rerank": lambda r: (
            f"相关性筛选: {len(r.get('reranked_docs', []))} 条"
        ),
        "answer_output": lambda r: (
            f"答案生成完成（{len(r.get('answer', ''))} 字）"
        ),
    }
    fn = extractors.get(node_name)
    return fn(result) if fn else ""


def _extract_sources(state: dict, max_sources: int = 6) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for doc in state.get("reranked_docs") or []:
        if not isinstance(doc, dict):
            continue
        source_type = doc.get("source") or "local"
        title = (
            str(doc.get("title") or "").strip()
            or str(doc.get("file_title") or "").strip()
            or ("联网结果" if source_type == "web" else "本地资料")
        )
        url = str(doc.get("url") or "").strip()
        chunk_id = str(doc.get("chunk_id") or "").strip()
        key = url or chunk_id or title
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "type": source_type,
                "title": title,
                "url": url,
                "chunk_id": chunk_id,
                "score": doc.get("score"),
            }
        )
        if len(sources) >= max_sources:
            break

    return sources


def _push_progress(session_id: str, status: str):
    push_sse_event(
        session_id,
        SSEEvent.PROGRESS,
        {
            "done_list": get_done_task_list(session_id),
            "running_list": get_running_task_list(session_id),
            "status": status,
        },
    )


def _run_query_graph(
    session_id: str,
    user_query: str,
    is_stream: bool,
    use_local_search: bool = True,
    use_web_search: bool = True,
):
    clear_task_progress(session_id)
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
                    _push_progress(session_id, TASK_STATUS_PROCESSING)
                    log_msg = _build_node_log(node_name, node_result)
                    if log_msg:
                        push_sse_event(session_id, "node_log", {"msg": log_msg})

                if isinstance(node_result, dict) and node_name not in ("multi_search", "join"):
                    _print_state(node_name, node_result)

        if not final_state:
            final_state = {"answer": "没有查询到相关内容。", "reranked_docs": []}

        answer = final_state.get("answer", "")
        sources = _extract_sources(final_state)
        set_task_result(session_id, "answer", answer)
        set_task_result(session_id, "sources", sources)

        update_task_status(session_id, TASK_STATUS_COMPLETED)
        if is_stream:
            add_done_task(session_id, "answer_output")
            push_sse_event(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": answer,
                    "status": "completed",
                    "sources": sources,
                    "done_list": get_done_task_list(session_id),
                    "running_list": get_running_task_list(session_id),
                },
            )

    except Exception as exc:
        logger.exception("Query failed")
        update_task_status(session_id, TASK_STATUS_FAILED)
        if is_stream:
            push_sse_event(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": f"查询出错: {exc}",
                    "status": "failed",
                    "sources": [],
                    "done_list": get_done_task_list(session_id),
                    "running_list": get_running_task_list(session_id),
                },
            )
        else:
            set_task_result(session_id, "answer", f"查询出错: {exc}")
            set_task_result(session_id, "sources", [])


app = create_query_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
