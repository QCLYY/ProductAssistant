"""RRF 融合排序节点

使用 Reciprocal Rank Fusion 算法融合多路检索结果。
"""

import hashlib
from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    """RRF 融合排序节点。

    流程: 收集四路检索结果 → RRF 加权融合 → 按得分降序返回
    """

    name = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # Step 1: 收集四路检索结果
        sources = {
            "embedding": (
                self._extract_entities(state.get("embedding_chunks")),
                1.0,
            ),
            "hyde": (
                self._extract_entities(state.get("hyde_embedding_chunks")),
                1.0,
            ),
            "kg": (
                self._extract_entities(state.get("kg_chunks")),
                self.config.rrf_kg_weight,
            ),
            "web": (
                self._normalize_web_docs(state.get("web_search_docs")),
                self.config.rrf_kg_weight,
            ),
        }

        self.logger.info(
            f"RRF 输入: {', '.join(f'{k}={len(v[0])}' for k, v in sources.items())}"
        )

        # Step 2-5: 执行 RRF 融合
        source_weights = list(sources.values())
        rrf_results = self._reciprocal_rank_fusion(
            source_weights,
            k=self.config.rrf_k,
            max_results=self.config.rrf_max_results,
        )

        # Step 6: 输出结果
        rrf_chunks = [doc for doc, _ in rrf_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        if rrf_results:
            scores = [s for _, s in rrf_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        return {"rrf_chunks": rrf_chunks}

    # ================================================================== #
    #                      RRF 算法实现                                   #
    # ================================================================== #

    @staticmethod
    def _reciprocal_rank_fusion(
        source_weights: List[Tuple[List[Dict], float]],
        k: int = 60,
        max_results: int = None,
    ) -> List[Tuple[Dict, float]]:
        """带权重的 RRF 融合。

        公式: score(d) = Σ weight_i / (k + rank_i(d))

        Args:
            source_weights: [(文档列表, 权重), ...]
            k: RRF 常数。默认 60。
            max_results: 返回前 N 个，None 则全部返回。

        Returns:
            [(文档, 得分), ...] 按得分降序。
        """
        score_map: Dict[str, float] = {}
        chunk_map: Dict[str, Dict] = {}

        for rank_list, weight in source_weights:
            for pos, item in enumerate(rank_list, start=1):
                chunk_id = item.get("chunk_id")
                if not chunk_id:
                    continue
                score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight / (k + pos)
                chunk_map.setdefault(chunk_id, item)

        merged = sorted(
            [(chunk_map[cid], score) for cid, score in score_map.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return merged[:max_results] if max_results else merged

    # ================================================================== #
    #                      工具方法                                        #
    # ================================================================== #

    @staticmethod
    def _extract_entities(state_list) -> List[Dict[str, Any]]:
        """统一规整为带 chunk_id 的字典列表。

        兼容格式：
        - {"entity": {"chunk_id": ..., ...}, "distance": ...} → 取 entity
        - {"chunk_id": ..., "content": ...} → 直接使用
        """
        out: List[Dict[str, Any]] = []
        for doc in (state_list or []):
            if not doc or not hasattr(doc, "get"):
                continue
            out.append(doc.get("entity") or doc)
        return out

    @staticmethod
    def _normalize_web_docs(docs) -> List[Dict[str, Any]]:
        """给网页搜索结果生成 chunk_id（用 URL 哈希），使其参与 RRF 排名。"""
        out: List[Dict[str, Any]] = []
        for doc in (docs or []):
            if not doc or not hasattr(doc, "get"):
                continue
            url = doc.get("url", "")
            cid = hashlib.md5(url.encode()).hexdigest()[:12]
            out.append({"chunk_id": cid, **doc})
        return out


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = RrfNode()


def node_rrf(state: QueryGraphState) -> QueryGraphState:
    return _node_instance(state)


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == "__main__":
    setup_logging()

    print("=" * 60)
    print("RRF 融合节点测试")
    print("=" * 60)

    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "#1-向量"}},
            {"entity": {"chunk_id": "chunk_2", "content": "#2-向量"}},
            {"entity": {"chunk_id": "chunk_3", "content": "#3-向量"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "#1-HyDE"}},
            {"entity": {"chunk_id": "chunk_1", "content": "#2-HyDE"}},
            {"entity": {"chunk_id": "chunk_4", "content": "#3-HyDE"}},
        ],
        "kg_chunks": [
            {"chunk_id": "chunk_5", "content": "#1-KG"},
            {"chunk_id": "chunk_1", "content": "#2-KG"},
        ],
        "web_search_docs": [
            {"title": "网页1", "url": "https://a.com/1", "snippet": "s1"},
            {"title": "网页2", "url": "https://b.com/2", "snippet": "s2"},
        ],
    }

    print(f"\n【输入状态】")
    print(f"  embedding: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde:      {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg:        {len(mock_state['kg_chunks'])} 条")
    print(f"  web:       {len(mock_state['web_search_docs'])} 条")
    print("-" * 60)

    result = node_rrf(mock_state)

    print(f"\n【融合结果】共 {len(result['rrf_chunks'])} 条")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        cid = chunk.get("chunk_id", "?")
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        label = title or content
        print(f"  [{i}] {cid}: {label[:80]}")
