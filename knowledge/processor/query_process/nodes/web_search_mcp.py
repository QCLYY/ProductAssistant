"""Optional web search node for the query pipeline.

The graph still uses the historical node name ``web_search_mcp``. Internally,
the node can use Tavily directly or the older DashScope MCP path.
"""

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    """Fetch web search snippets when web search is enabled."""

    name = "web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query") or state.get("original_query", "")
        docs: list[dict] = []

        if not state.get("use_web_search", True):
            self.logger.info("Web search is disabled for this request; skipping search.")
            return {"web_search_docs": [], "web_search_attempted": True}

        if not getattr(self.config, "enable_web_search", False):
            self.logger.info("Web search is disabled; skipping search.")
            return {"web_search_docs": [], "web_search_attempted": True}

        if not query:
            self.logger.warning("Query is empty; skipping web search.")
            return {"web_search_docs": [], "web_search_attempted": True}

        provider = (getattr(self.config, "web_search_provider", "") or "tavily").lower()
        try:
            if provider == "tavily":
                docs = self._tavily_search(query)
            elif provider in {"dashscope", "dashscope_mcp", "mcp"}:
                docs = self._dashscope_mcp_search(query)
            else:
                self.logger.warning(f"Unsupported web search provider: {provider}")
        except Exception as e:
            self.logger.warning(f"Web search failed; continuing without web docs: {e}")

        return {"web_search_docs": self._dedupe_docs(docs), "web_search_attempted": True}

    def _tavily_search(self, query: str) -> list[dict]:
        api_key = getattr(self.config, "tavily_api_key", "")
        if not api_key:
            self.logger.warning("TAVILY_API_KEY is empty; skipping Tavily search.")
            return []

        payload = {
            "query": query,
            "search_depth": getattr(self.config, "tavily_search_depth", "basic"),
            "max_results": getattr(self.config, "tavily_max_results", 5),
            "include_answer": False,
            "include_raw_content": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=getattr(self.config, "tavily_api_url", "https://api.tavily.com/search"),
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Tavily HTTP {e.code}: {error_body}") from e

        docs = []
        for item in result.get("results") or []:
            snippet = (item.get("content") or item.get("snippet") or "").strip()
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            if snippet:
                docs.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "tavily",
                })

        self.log_step("done", f"Tavily returned {len(docs)} results.")
        return self._dedupe_docs(docs)

    def _dashscope_mcp_search(self, query: str) -> list[dict]:
        if not getattr(self.config, "mcp_dashscope_base_url", ""):
            self.logger.warning("MCP_DASHSCOPE_BASE_URL is empty; skipping MCP search.")
            return []

        result = asyncio.run(self._mcp_call(query))
        docs = []
        for item in (result or {}).get("pages") or []:
            snippet = (item.get("snippet") or "").strip()
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            if snippet:
                docs.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "dashscope_mcp",
                })
        self.log_step("done", f"DashScope MCP returned {len(docs)} results.")
        return self._dedupe_docs(docs)

    @staticmethod
    def _dedupe_docs(docs: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for doc in docs or []:
            if not isinstance(doc, dict):
                continue
            key = (
                (doc.get("url") or "").strip().lower()
                or f"{(doc.get('title') or '').strip()}::{(doc.get('snippet') or doc.get('content') or '').strip()[:160]}"
            )
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(doc)
        return unique

    async def _mcp_call(self, query: str) -> Any:
        from agents.mcp import MCPServerStreamableHttp

        mcp_client = MCPServerStreamableHttp(
            name="web_search",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {
                    "Authorization": (
                        f"Bearer {self.config.mcp_dashscope_api_key or self.config.openai_api_key}"
                    )
                },
                "timeout": 300,
                "sse_read_timeout": 300,
            },
            cache_tools_list=True,
            client_session_timeout_seconds=30,
        )
        try:
            await mcp_client.connect()
            execute_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
            if not execute_result or not execute_result.content:
                return None
            raw_text = execute_result.content[0].text
            return json.loads(raw_text)
        finally:
            await mcp_client.cleanup()


_node_instance = WebSearchMcpNode()


def node_web_search_mcp(state: QueryGraphState) -> QueryGraphState:
    return _node_instance(state)


if __name__ == "__main__":
    import uuid

    setup_logging()
    test_state = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        "rewritten_query": "test query",
    }
    print(node_web_search_mcp(test_state))
