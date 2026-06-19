"""重排序节点

使用 Reranker 模型对 RRF 融合结果和网络搜索结果进行精排，
并通过断崖检测实现动态 TopK 截断。
"""

from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class RerankNode(BaseNode):
    """重排序节点。

    流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
    """

    name = "rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        question = state.get("rewritten_query") or state.get("original_query", "")

        # Step 2-3: 合并本地 RRF 结果 + 网页搜索结果
        doc_items = self._dedupe_doc_items(self._merge_docs(state))

        # Step 4-5: Reranker 精排
        self.log_step("step_1", f"重排序 {len(doc_items)} 篇文档")
        scored_docs = self._rerank(question, doc_items)

        # Step 6: 断崖检测动态截断
        topk_docs = self._cliff_cutoff(scored_docs)

        self.logger.info(f"重排序完成: {len(doc_items)} → {len(topk_docs)}")
        return {"reranked_docs": topk_docs}

    # ================================================================== #
    #                      文档合并                                        #
    # ================================================================== #

    def _merge_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式。"""
        doc_items = []

        # 本地 RRF 结果
        for doc in (state.get("rrf_chunks") or []):
            if not isinstance(doc, dict) or not doc.get("content"):
                continue
            title = doc.get("title") or doc.get("file_title") or ""
            if doc.get("file_title") and doc.get("title") and doc.get("file_title") != doc.get("title"):
                title = f"{doc.get('file_title')} / {doc.get('title')}"
            doc_items.append(self._make_doc_item(
                text=doc["content"],
                chunk_id=doc.get("chunk_id") or doc.get("id"),
                title=title,
                source="local",
            ))

        # 网络搜索结果
        for doc in (state.get("web_search_docs") or []):
            text = (doc.get("snippet") or doc.get("content") or "").strip()
            if not text:
                continue
            doc_items.append(self._make_doc_item(
                text=text,
                title=doc.get("title", "").strip(),
                url=doc.get("url", "").strip(),
                source="web",
            ))

        self.logger.info(f"合并文档: {len(doc_items)} 篇")
        return doc_items

    @staticmethod
    def _dedupe_doc_items(doc_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 chunk_id、url 或正文片段去重，保留最先出现的结果。"""
        seen: set[str] = set()
        unique: List[Dict[str, Any]] = []
        for item in doc_items or []:
            chunk_id = str(item.get("chunk_id") or "").strip()
            url = str(item.get("url") or "").strip().lower()
            title = str(item.get("title") or "").strip()
            text = str(item.get("text") or "").strip()
            key = chunk_id or url or f"{title}::{text[:180]}"
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _make_doc_item(
        text: str, source: str = "",
        chunk_id=None, title: str = "", url: str = "",
    ) -> Dict[str, Any]:
        return {
            "text": text, "source": source,
            "chunk_id": chunk_id, "doc_id": chunk_id,
            "title": title, "url": url,
        }

    # ================================================================== #
    #                      Reranker 排序                                  #
    # ================================================================== #

    def _rerank(
        self, question: str, doc_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """计算相关性得分并排序，失败时降级返回原序。"""
        if not doc_items or not question:
            return []

        try:
            from knowledge.tools.reranker_utils import get_reranker_model

            reranker = get_reranker_model()
            if reranker is None:
                self.logger.warning("Reranker 未加载，返回原序")
                return [{**item, "score": None} for item in doc_items]

            # 构建 Query-Document 对
            pairs = [[question, item["text"]] for item in doc_items]

            # 批量计算得分
            scores = reranker.compute_score(pairs)

            # 附加得分并排序
            scored = [
                {**item, "score": float(s)}
                for item, s in zip(doc_items, scores)
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored

        except Exception as e:
            self.logger.error(f"重排序失败，降级为原序: {e}")
            return [{**item, "score": None} for item in doc_items]

    # ================================================================== #
    #                    断崖检测动态截断                                   #
    # ================================================================== #

    def _cliff_cutoff(
        self, scored_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not scored_docs:
            return []

        max_topk = min(self.config.rerank_max_top_k, len(scored_docs))
        min_topk = self.config.rerank_min_top_k
        gap_abs = self.config.rerank_gap_abs
        gap_ratio = self.config.rerank_gap_ratio

        topk = max_topk
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")
            s2 = scored_docs[i + 1].get("score")
            if s1 is None or s2 is None:
                continue

            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)

            if gap >= gap_abs or rel >= gap_ratio:
                topk = i + 1
                self.logger.debug(
                    f"断崖检测: 位置 {i+1}, gap={gap:.4f}, rel={rel:.4f}"
                )
                break

        return scored_docs[:topk]


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = RerankNode()


def node_rerank(state: QueryGraphState) -> QueryGraphState:
    return _node_instance(state)


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("重排序节点测试")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1", "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。",
            },
            {
                "chunk_id": "local_2", "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。",
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair", "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。",
            },
            {
                "url": "https://example.com/news", "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。",
            },
        ],
    }

    print(f"\n【输入状态】")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    result = node_rerank(mock_state)

    print(f"\n【重排序结果】共 {len(result.get('reranked_docs', []))} 篇")
    for i, doc in enumerate(result.get("reranked_docs", []), 1):
        score = doc.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  [{i}] score={score_str} | {doc['source']:5} | {doc['text'][:60]}...")
