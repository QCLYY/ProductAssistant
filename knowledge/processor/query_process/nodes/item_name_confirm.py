"""商品名称确认节点

从用户查询query中提取商品名称，通过向量相似度匹配与数据库中已有商品对齐确认。
"""

import os
import json
import re
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.prompt import ITEM_NAME_EXTRACT_TEMPLATE


class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点。

    流程: 获取历史 → LLM提取商品名 → 向量匹配 → 评分对齐 → 更新状态 → 写入历史
    """

    name = "item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id", "")
        query = state.get("original_query", "")
        self._dump_state(state, "输入")

        # 1. 获取历史记录
        history = self._get_history(session_id)

        # 1.1 保存用户问题（获取 message_id）
        message_id = self._save_message(
            session_id, "user", query,
            item_names=state.get("item_names", [])
        )

        # 2. LLM 提取商品名称
        extract_res = self._extract_item_names(query, history)
        item_names = extract_res.get("item_names", [])
        rewritten_query = extract_res.get("rewritten_query", query)
        self.logger.info(f"LLM提取结果: item_names={item_names}, rewritten={rewritten_query}")

        # 3. 向量匹配 + 评分对齐
        align_result = self._match_and_align(item_names) if item_names else {}
        self.logger.info(f"对齐结果: {align_result}")

        # 4. 更新状态
        state = self._update_state(state, align_result, rewritten_query, history)

        # 5. 写入历史
        self._write_history(state, session_id, rewritten_query, message_id)
        state["history"] = history

        self._dump_state(state, "输出")
        return state

    # ================================================================== #
    #                      历史记录操作                                    #
    # ================================================================== #

    def _get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """获取历史会话记录。"""
        from knowledge.tools.mongo_history_utils import get_recent_messages
        try:
            return get_recent_messages(session_id, limit=limit)
        except Exception as e:
            self.logger.warning(f"获取历史记录失败: {e}")
            return []

    def _save_message(
            self, session_id: str, role: str, text: str,
            rewritten_query: str = "", item_names: List[str] = None,
            message_id: str = "",
    ) -> str:
        """保存单条消息到历史记录。"""
        from knowledge.tools.mongo_history_utils import save_chat_message
        try:
            return save_chat_message(
                session_id=session_id,
                role=role,
                text=text,
                rewritten_query=rewritten_query,
                item_names=item_names or [],
                message_id=message_id,
            )
        except Exception as e:
            self.logger.warning(f"保存消息失败: {e}")
            return ""

    # ================================================================== #
    #                      LLM 提取商品名称                               #
    # ================================================================== #

    def _extract_item_names(self, query: str, history: List[Dict]) -> Dict[str, Any]:
        """使用 LLM 从查询和历史中提取商品名称。

        Returns:
            {"item_names": [...], "rewritten_query": "..."}
        """
        from knowledge.tools.llm_utils import get_llm_client

        model = self.config.item_model or os.getenv("ITEM_MODEL", "")
        client = get_llm_client(model, json_mode=True)

        history_text = "".join(
            f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"
            for msg in history
        )

        prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text, query=query
        )

        try:
            response = client.invoke([
                SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
                HumanMessage(content=prompt),
            ])

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].removeprefix("json").strip()

            result = json.loads(content)
            result.setdefault("item_names", [])
            result.setdefault("rewritten_query", query)
            result["item_names"] = [n.strip() for n in result["item_names"]]
            if not result["item_names"]:
                result["item_names"] = self._fallback_item_name_candidates(query, result)
            return result

        except Exception as e:
            self.logger.error(f"LLM 提取商品名称失败: {e}")
            return {"item_names": [], "rewritten_query": query}

    @staticmethod
    def _fallback_item_name_candidates(query: str, parsed: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []

        cleaned_query = re.sub(r"[？?。！!，,：:；;]", " ", query or "").strip()
        cleaned_query = re.sub(
            r"(是什么|是啥|如何|怎么|怎样|使用|操作|维修|咨询|介绍|说明|告诉我|请问)$",
            "",
            cleaned_query,
        ).strip()
        if cleaned_query:
            candidates.append(cleaned_query)

        for value in parsed.values():
            if isinstance(value, str):
                candidates.append(value.strip())
            elif isinstance(value, list):
                candidates.extend(str(item).strip() for item in value)

        deduped: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped and len(candidate) <= 80:
                deduped.append(candidate)
        return deduped[:3]

    # ================================================================== #
    #                  向量匹配 + 评分对齐                                 #
    # ================================================================== #

    def _match_and_align(self, item_names: List[str]) -> Dict[str, Any]:
        """向量检索 + 评分对齐。

        Returns:
            {"confirmed_item_names": [...], "options": [...]}
        """
        query_results = self._vector_search(item_names)
        return self._align_by_score(query_results)

    def _vector_search(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """批量向量检索商品名称。"""
        from knowledge.tools.embedding_utils import generate_hybrid_embeddings
        from knowledge.tools.milvus_utils import (
            get_milvus_client,
            build_hybrid_search_requests,
            execute_hybrid_search,
        )

        client = get_milvus_client()
        if not client:
            self.logger.error("无法连接到 Milvus")
            return []

        collection_name = self.config.item_name_collection or os.getenv(
            "ITEM_NAME_COLLECTION", "item_name_collection"
        )

        embeddings = generate_hybrid_embeddings(item_names)
        results = []

        max_options = self.config.item_name_max_options
        dense_w = self.config.item_name_dense_weight
        sparse_w = self.config.item_name_sparse_weight

        for i, name in enumerate(item_names):
            try:
                reqs = build_hybrid_search_requests(
                    dense_vector=embeddings["dense"][i],
                    sparse_vector=embeddings["sparse"][i],
                    top_k=max_options,
                )

                search_res = execute_hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    search_requests=reqs,
                    ranker_weights=(dense_w, sparse_w),
                    top_k=max_options,
                    normalize_score=True,
                    output_fields=["item_name"],
                )

                matches = [
                    {"item_name": hit["entity"]["item_name"], "score": hit["distance"]}
                    for hit in (search_res[0] if search_res else [])
                ]

                results.append({"extracted_name": name, "matches": matches})

            except Exception as e:
                self.logger.error(f"查询商品名称 {name} 失败: {e}")

        return results

    def _align_by_score(self, query_results: List[Dict]) -> Dict[str, Any]:
        """根据评分对齐商品名称。

        规则:
            - score > high_threshold 且唯一 → 直接确认
            - score > high_threshold 且多条 → 优先取与提取名完全匹配的，否则取最高分
            - mid_threshold ≤ score < high_threshold → 作为候选选项
            - score < mid_threshold → 忽略
        """
        high_threshold = self.config.item_name_high_confidence
        mid_threshold = self.config.item_name_mid_confidence
        max_options = self.config.item_name_max_options

        confirmed: List[str] = []
        options: List[str] = []

        for res in query_results:
            extracted = (res.get("extracted_name") or "").strip()

            matches = sorted(
                res.get("matches") or [],
                key=lambda m: m.get("score", 0),
                reverse=True,
            )

            if not matches:
                continue

            high = [m for m in matches if m["score"] > high_threshold]
            mid = [m for m in matches if m["score"] >= mid_threshold]

            if high:
                exact = next(
                    (m for m in high if m["item_name"].strip() == extracted),
                    None
                )
                confirmed.append((exact or high[0])["item_name"])
            elif mid:
                options.extend(m["item_name"] for m in mid[:max_options])

        return {
            "confirmed_item_names": confirmed,
            "options": options[:max_options],
        }

    # ================================================================== #
    #                      状态更新 & 历史写入                             #
    # ================================================================== #

    def _update_state(
            self, state: QueryGraphState, align_result: Dict,
            rewritten_query: str, history: List[Dict],
    ) -> QueryGraphState:
        """根据对齐结果更新 state。"""
        confirmed = align_result.get("confirmed_item_names", [])
        options = align_result.get("options", [])

        if confirmed:
            confirmed = list(dict.fromkeys(confirmed))
            self._backfill_history_item_names(history, confirmed)
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query

        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )

        else:
            if getattr(self.config, "enable_web_search", False):
                state["item_names"] = []
                state["rewritten_query"] = rewritten_query
                return state

            state["answer"] = (
                "抱歉，我无法识别您询问的具体产品名称，"
                "请提供更准确的产品名称或型号。"
            )

        return state

    def _backfill_history_item_names(
            self, history: List[Dict], item_names: List[str]
    ):
        """将确认的商品名称回填到没有商品名的历史记录。"""
        from knowledge.tools.mongo_history_utils import update_message_item_names

        ids_to_update = [
            msg["_id"] for msg in history if not msg.get("item_names")
        ]
        if not ids_to_update:
            return

        for msg in history:
            if not msg.get("item_names"):
                msg["item_names"] = item_names

        try:
            update_message_item_names(ids_to_update, item_names)
        except Exception as e:
            self.logger.warning(f"回填历史商品名称失败: {e}")

    def _write_history(
            self, state: QueryGraphState, session_id: str,
            rewritten_query: str, message_id: str,
    ):
        """将本轮对话写入历史（用户问题 + 助手回复）。"""
        query = (state.get("original_query") or "").strip()
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []

        if query:
            self._save_message(
                session_id,
                "user",
                query,
                rewritten_query=rewritten_query,
                item_names=item_names,
                message_id=message_id,
            )

        if answer:
            self._save_message(
                session_id,
                "assistant",
                answer,
                item_names=item_names,
            )

    # ================================================================== #
    #                        调试工具                                       #
    # ================================================================== #

    @staticmethod
    def _dump_state(state: dict, title: str = ""):
        """打印 state 关键字段，方便调试。"""
        label = f" [{title}]" if title else ""
        print(f"\n{'='*50}")
        print(f"State{label}:")
        print(f"  session_id:    {state.get('session_id', '')}")
        print(f"  original_query:{state.get('original_query', '')}")
        print(f"  item_names:    {state.get('item_names', [])}")
        print(f"  rewritten_query:{state.get('rewritten_query', '')}")
        print(f"  answer:        {(state.get('answer', '') or '')[:80]}")
        print(f"  history 条数:   {len(state.get('history', []))}")
        print(f"{'='*50}\n")


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = ItemNameConfirmNode()


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
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
    print("商品名称确认节点测试")
    print("=" * 60)

    test_state = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        #"original_query": "你们店里那款苏伯尔RS-12数字万用表怎么测电压？",
        "original_query": "你们的华为电脑MateBook B5-440怎么打开护眼模式？",
        "item_names": [],
        "rewritten_query": "",
        "answer": "",
        "history": [],
        "is_stream": False,
    }

    print(f"\n输入状态:")
    print(f"  session_id: {test_state['session_id']}")
    print(f"  original_query: {test_state['original_query']}")
    print("-" * 60)

    try:
        result = node_item_name_confirm(test_state)

        print("\n输出状态:")
        print(f"  item_names: {result.get('item_names')}")
        print(f"  rewritten_query: {result.get('rewritten_query')}")

        if result.get("answer"):
            print(f"\n拦截回复（流程中断）:")
            print(f"  {result.get('answer')}")
        else:
            print(f"\n确认成功，继续检索流程")

        print(f"\n历史记录条数: {len(result.get('history', []))}")

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试多轮对话
    print("\n" + "=" * 60)
    print("测试多轮对话（代词指代）")
    print("=" * 60)

    test_state_round2 = {
        "session_id": test_state["session_id"],
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        "original_query": "那它怎么一键恢复出厂？",
        "item_names": [],
        "rewritten_query": "",
        "answer": "",
        "history": [],
        "is_stream": False,
    }

    print(f"\n第二轮输入:")
    print(f"  original_query: {test_state_round2['original_query']}")
    print("-" * 60)

    try:
        result2 = node_item_name_confirm(test_state_round2)

        print("\n第二轮输出:")
        print(f"  item_names: {result2.get('item_names')}")
        print(f"  rewritten_query: {result2.get('rewritten_query')}")

        if result2.get("answer"):
            print(f"\n拦截回复: {result2.get('answer')}")
        else:
            print(f"\n代词已解析，确认成功")

    except Exception as e:
        print(f"\n执行失败: {e}")
