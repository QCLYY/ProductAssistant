"""HyDE 向量搜索节点

使用 Hypothetical Document Embedding 技术：
先让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升召回质量。
"""

import os
from typing import List, Optional

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.prompt import HYDE_PROMPT_TEMPLATE


class SearchEmbeddingHydeNode(BaseNode):
    """HyDE 向量搜索节点。

    流程: LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
    """

    name = "search_embedding_hyde"

    SEARCH_TOP_K = 10
    RERANK_TOP_K = 10
    RANKER_WEIGHTS = (0.5, 0.5)
    OUTPUT_FIELDS = ["chunk_id", "content", "item_name"]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query") or state.get("original_query", "")
        if not query:
            self.logger.error("未找到用户查询")
            return {}

        item_names = state.get("item_names")
        collection_name = self.config.chunks_collection or os.getenv(
            "CHUNKS_COLLECTION", "kb_chunks_v2"
        )

        try:
            self.log_step("step_1", "生成假设性文档")
            hyde_doc = self._generate_hyde_doc(query)

            self.log_step("step_2", "执行混合搜索")
            chunks = self._search(query, hyde_doc, item_names, collection_name)

            self.log_step("step_3", f"搜索完成，返回 {len(chunks)} 条结果")
            return {"hyde_embedding_chunks": chunks, "hyde_doc": hyde_doc}

        except Exception as e:
            self.logger.error(f"HyDE 搜索失败: {e}")
            return {}

    def _generate_hyde_doc(self, query: str) -> str:
        """使用 LLM 根据用户查询生成假设性文档。"""
        from knowledge.tools.llm_utils import get_llm_client

        llm = get_llm_client()
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        return llm.invoke(prompt).content

    def _search(
            self, query: str, hyde_doc: str,
            item_names: Optional[List[str]] = None,
            collection_name: str = "kb_chunks_v2",
    ) -> List:
        """将查询与假设文档拼接后执行混合检索。"""
        from knowledge.tools.embedding_utils import generate_hybrid_embeddings
        from knowledge.tools.milvus_utils import (
            get_milvus_client,
            build_hybrid_search_requests,
            execute_hybrid_search,
        )

        combined_text = f"{query} {hyde_doc}"
        embeddings = generate_hybrid_embeddings([combined_text])

        reqs = build_hybrid_search_requests(
            dense_vector=embeddings["dense"][0],
            sparse_vector=embeddings["sparse"][0],
            filter_expr=self._build_filter_expr(item_names),
            top_k=self.SEARCH_TOP_K,
        )

        res = execute_hybrid_search(
            client=get_milvus_client(),
            collection_name=collection_name,
            search_requests=reqs,
            ranker_weights=self.RANKER_WEIGHTS,
            normalize_score=True,
            top_k=self.RERANK_TOP_K,
            output_fields=self.OUTPUT_FIELDS,
        )

        return res[0] if res else []

    @staticmethod
    def _build_filter_expr(item_names: Optional[List[str]]) -> Optional[str]:
        if not item_names:
            return None
        quoted = ", ".join(f'"{v}"' for v in item_names)
        return f"item_name in [{quoted}]"


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = SearchEmbeddingHydeNode()


def node_search_embedding_hyde(state: QueryGraphState) -> QueryGraphState:
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
    print("HyDE 向量搜索节点测试")
    print("=" * 60)

    test_state = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        "rewritten_query": "华为MateBook B5-440电脑如何打开护眼模式？",
        "item_names": ["华为MateBook B5-440笔记本电脑"],
    }

    print(f"\n【输入状态】")
    print(f"  rewritten_query: {test_state['rewritten_query']}")
    print(f"  item_names: {test_state['item_names']}")
    print("-" * 60)

    try:
        result = node_search_embedding_hyde(test_state)

        hyde_doc = result.get("hyde_doc", "")
        print(f"\n【LLM 生成的假设性文档】")
        print(f"  {hyde_doc}")
        print("-" * 60)

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"\n【检索结果】共 {len(chunks)} 条")
        for i, chunk in enumerate(chunks, 1):
            entity = chunk.get("entity", chunk) if isinstance(chunk, dict) else {}
            print(f"  [{i}] {entity.get('item_name','?')}")
            print(f"      chunk_id={entity.get('chunk_id','?')}")
            print(f"      score={chunk.get('distance', 0):.4f}")
            content = entity.get("content", "")
            print(f"      content={content}")
            print()

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()
