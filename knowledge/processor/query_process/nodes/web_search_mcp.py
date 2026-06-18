"""MCP 网络搜索节点

通过 MCP 协议调用网络搜索服务获取外部信息。
使用 OpenAI Agents SDK 的 MCPServerSse 连接百炼 DashScope 搜索服务。
"""

import json
import asyncio
from typing import Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    """MCP 网络搜索节点。

    通过 MCP 协议（SSE 模式）连接到 DashScope 网络搜索服务，
    根据用户查询获取相关的网络搜索结果。
    """

    name = "web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.log_step("step_1", "获取查询内容")
        query = state.get("rewritten_query", "")
        docs: list = []

        if not query:
            self.logger.warning("查询内容为空，跳过网络搜索")
            return {}

        self.log_step("step_2", f"执行 MCP 搜索: {query}")
        try:
            result = asyncio.run(self._mcp_call(query))
            if result:
                pages = result.get("pages") or []
                for item in pages:
                    snippet = (item.get("snippet") or "").strip()
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    if not snippet:
                        continue
                    docs.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    })

                self.log_step("step_3", f"搜索完成，返回 {len(docs)} 条结果")
        except Exception as e:
            self.logger.error(f"MCP 搜索失败: {e}")

        if docs:
            return {"web_search_docs": docs}
        return {}

    async def _mcp_call(self, query: str) -> Any:
        """调用 MCP 搜索服务。

        使用 OpenAI Agents SDK 的 MCPServerSse 客户端，
        以 SSE 模式连接 DashScope 百炼平台的 Web 搜索 MCP 服务。
        """
        from agents.mcp import MCPServerStreamableHttp

        mcp_client = MCPServerStreamableHttp(
            name="通用搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
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


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = WebSearchMcpNode()


def node_web_search_mcp(state: QueryGraphState) -> QueryGraphState:
    return _node_instance(state)


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == "__main__":
    import uuid
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("MCP 网络搜索节点测试")
    print("=" * 60)

    test_state = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        "rewritten_query": "华为MateBook B5-440电脑如何打开护眼模式？",
        "item_names": ["华为MateBook B5-440笔记本电脑"],
    }

    print(f"\n【输入状态】")
    print(f"  rewritten_query: {test_state['rewritten_query']}")
    print("-" * 60)

    try:
        result = node_web_search_mcp(test_state)
        docs = result.get("web_search_docs", [])

        if not docs:
            print("\n搜索执行完成，但未返回任何结果。")
        else:
            print(f"\n搜索到 {len(docs)} 条结果:")
            for i, doc in enumerate(docs, 1):
                print(f"  [{i}] 标题: {doc.get('title', '无标题')}")
                print(f"      链接: {doc.get('url', '无链接')}")
                snippet = doc.get("snippet", "")
                print(f"      摘要: {snippet}")
                print()

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()
