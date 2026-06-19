"""向量检索节点

对用户查询进行向量化，在 Milvus 中执行混合搜索（稠密 + 稀疏），返回相关切片。
"""

import os
from typing import List, Optional

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class SearchEmbeddingNode(BaseNode):
    """向量检索节点。

    流程: 查询向量化 → 构建过滤表达式 → 构建混合搜索请求 → 执行检索 → 返回结果
    """

    name = "search_embedding"

    # 检索参数
    SEARCH_TOP_K = 10              # 稠密/稀疏各路分别召回数
    RERANK_TOP_K = 10              # WeightedRanker 融合后返回数
    RANKER_WEIGHTS = (0.5, 0.5)    # 稠密:稀疏 权重
    OUTPUT_FIELDS = ["chunk_id", "content", "title", "file_title", "item_name"]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        if not state.get("use_local_search", True):
            self.log_step("skip", "本地资料检索未启用，跳过向量检索")
            return {"embedding_chunks": []}

        from knowledge.tools.embedding_utils import generate_hybrid_embeddings
        from knowledge.tools.milvus_utils import (
            get_milvus_client,
            build_hybrid_search_requests,
            execute_hybrid_search,
        )

        # Step 1: 获取查询参数
        query = state.get("rewritten_query", "")
        item_names = state.get("item_names")
        collection_name = self.config.chunks_collection or os.getenv(
            "CHUNKS_COLLECTION", "kb_chunks_v2"
        )

        self.log_step("step_1", f"查询向量化: {query}")

        # Step 2: 查询向量化（稠密 + 稀疏）
        embeddings = generate_hybrid_embeddings([query])

        # Step 3: 构建过滤表达式
        filter_expr = self._build_filter_expr(item_names)
        self.logger.debug(f"过滤表达式: {filter_expr}")

        # Step 4: 构建混合搜索请求
        reqs = build_hybrid_search_requests(
            dense_vector=embeddings["dense"][0],
            sparse_vector=embeddings["sparse"][0],
            dense_search_params={"metric_type": "IP"},
            sparse_search_params={"metric_type": "IP"},
            filter_expr=filter_expr,
            top_k=self.SEARCH_TOP_K,
        )

        # Step 5: 执行混合检索
        self.log_step("step_2", "执行混合搜索")
        res = execute_hybrid_search(
            client=get_milvus_client(),
            collection_name=collection_name,
            search_requests=reqs,
            ranker_weights=self.RANKER_WEIGHTS,
            normalize_score=True,
            top_k=self.RERANK_TOP_K,
            output_fields=self.OUTPUT_FIELDS,
        )

        # Step 6: 提取结果，写入 state
        chunks = res[0] if res else []
        self.log_step("step_3", f"搜索完成，返回 {len(chunks)} 条结果")

        return {"embedding_chunks": chunks}

    @staticmethod
    def _build_filter_expr(item_names: Optional[List[str]]) -> Optional[str]:
        """将商品名称列表转换为 Milvus 过滤表达式。

        Args:
            item_names: 商品名称列表，如 ["RS-12数字万用表", "示波器DS-100"]。

        Returns:
            'item_name in ["RS-12数字万用表", "示波器DS-100"]' 格式的字符串；
            列表为空则返回 None（不过滤）。
        """
        if not item_names:
            return None
        quoted = ", ".join(f'"{v}"' for v in item_names)
        return f"item_name in [{quoted}]"


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = SearchEmbeddingNode()


def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
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
    print("向量检索节点测试")
    print("=" * 60)

    # ---- 场景 1: 有商品名过滤 ----
    test_state = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        "rewritten_query": "华为MateBook B5-440电脑如何打开护眼模式？",
        "item_names": ["华为MateBook B5-440笔记本电脑"],
        "embedding_chunks": [],
    }

    print(f"\n【场景 1】有商品名过滤")
    print(f"  rewritten_query: {test_state['rewritten_query']}")
    print(f"  item_names: {test_state['item_names']}")
    print("-" * 60)

    try:
        result = node_search_embedding(test_state)
        chunks = result.get("embedding_chunks", [])

        print(f"\n检索到 {len(chunks)} 条结果:")
        for i, chunk in enumerate(chunks, 1):
            entity = chunk.get("entity", chunk) if isinstance(chunk, dict) else {}
            print(f"  [{i}] item_name={entity.get('item_name','?')}")
            print(f"      chunk_id={entity.get('chunk_id','?')}")
            print(f"      score={chunk.get('distance', 0):.4f}")
            content = entity.get("content", "")
            print(f"      content={content}")
            print()

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()

    # ---- 场景 2: 无商品名过滤 ----
    print("=" * 60)
    print("【场景 2】无商品名过滤")
    print("=" * 60)

    test_state_no_filter = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        "rewritten_query": "华为MateBook B5-440电脑如何打开护眼模式？",
        "item_names": [],
        "embedding_chunks": [],
    }

    print(f"\n  rewritten_query: {test_state_no_filter['rewritten_query']}")
    print(f"  item_names: [] (无过滤)")
    print("-" * 60)

    try:
        result2 = node_search_embedding(test_state_no_filter)
        chunks2 = result2.get("embedding_chunks", [])
        print(f"\n检索到 {len(chunks2)} 条结果（无过滤）:")
        for i, chunk in enumerate(chunks2[:3], 1):
            entity = chunk.get("entity", chunk) if isinstance(chunk, dict) else {}
            name = entity.get("item_name", "?")
            score = chunk.get("distance", 0)
            print(f"  [{i}] {name} | score={score:.4f}| entity={entity}")

    except Exception as e:
        print(f"\n执行失败: {e}")
