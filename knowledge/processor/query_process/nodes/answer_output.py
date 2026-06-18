"""答案输出节点

组装提示词、调用 LLM 生成答案，写入 MongoDB 历史记录。
"""

from typing import List, Dict

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.prompt import ANSWER_PROMPT


class AnswerOutputNode(BaseNode):
    """答案输出节点。

    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史
    """

    name = "answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        if state.get("answer"):
            return state

        # 构建提示词
        prompt = self._build_prompt(state)
        state["prompt"] = prompt

        # 调用 LLM 生成答案
        self.log_step("generate", "生成答案")
        is_stream = state.get("is_stream", False)
        task_id = state.get("task_id", "")
        if is_stream and task_id:
            state["answer"] = self._stream_generate(prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(prompt)

        # 写入 MongoDB 历史
        self._write_history(state)

        return state

    # ================================================================== #
    #                    提示词构建                                         #
    # ================================================================== #

    def _build_prompt(self, state: QueryGraphState) -> str:
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names") or []
        budget = self.config.max_context_chars

        context_str, budget = self._format_docs(
            state.get("reranked_docs") or [], budget)
        history_str, budget = self._format_history(
            state.get("history") or [], budget)
        graph_str, _ = self._format_triples(
            state.get("kg_triples") or [], budget)

        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str or "无历史对话",
            item_names=", ".join(item_names) if item_names else "无指定商品",
            graph_relation_description=graph_str or "无图谱关系",
            question=question,
        )

    def _format_docs(self, docs: List[Dict], budget: int) -> tuple:
        lines = []
        used = 0
        for i, doc in enumerate(docs, 1):
            text = (doc.get("text") or "").strip()
            if not text:
                continue

            meta = [f"[{i}]"]
            for key, fmt in [
                ("source", "[{}]"), ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"), ("title", "[title={}]"),
            ]:
                val = str(doc.get(key) or "").strip()
                if val:
                    meta.append(fmt.format(val))

            score = doc.get("score")
            if score is not None:
                meta.append(f"[score={float(score):.4f}]")

            doc_str = " ".join(meta) + "\n" + text
            if used + len(doc_str) > budget:
                break
            lines.append(doc_str)
            used += len(doc_str) + 2

        return "\n\n".join(lines), budget - used

    @staticmethod
    def _format_history(history: list, budget: int) -> tuple:
        lines = []
        used = 0
        for msg in (history or []):
            for role, key in [("用户", "user_question"), ("助手", "assistant_answer")]:
                text = msg.get(key) or msg.get("text")
                if not text:
                    continue
                line = f"{role}: {text}"
                used += len(line) + 1
                if used > budget:
                    return "\n".join(lines), budget - used
                lines.append(line)
        return "\n".join(lines), budget - used

    @staticmethod
    def _format_triples(triples: list, budget: int) -> tuple:
        lines = []
        used = 0
        for tr in (triples or []):
            line = (str(tr) if tr is not None else "").strip()
            if not line or used + len(line) > budget:
                if used + len(line) > budget:
                    break
                continue
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines), budget - used

    # ================================================================== #
    #                    LLM 生成                                          #
    # ================================================================== #

    def _invoke_generate(self, prompt: str) -> str:
        from knowledge.tools.llm_utils import get_llm_client

        llm = get_llm_client()
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误，请稍后重试。"

    def _stream_generate(self, prompt: str, task_id: str) -> str:
        from knowledge.tools.llm_utils import get_llm_client
        from knowledge.utils.sse_util import push_sse_event, SSEEvent

        llm = get_llm_client()
        result = ""
        try:
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    result += delta
                    push_sse_event(task_id, SSEEvent.DELTA, {"delta": delta})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return result

    # ================================================================== #
    #                    历史记录                                           #
    # ================================================================== #

    def _write_history(self, state: QueryGraphState):
        answer = (state.get("answer") or "").strip()
        if not answer:
            return

        from knowledge.tools.mongo_history_utils import save_chat_message
        try:
            save_chat_message(
                session_id=state.get("session_id", "default"),
                role="assistant",
                text=answer,
                rewritten_query="",
                item_names=state.get("item_names") or [],
            )
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = AnswerOutputNode()


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    return _node_instance(state)


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("答案生成节点测试")
    print("=" * 60)

    mock_state = {
        "session_id": "test_answer_001",
        "rewritten_query": "华为MateBook B5-440电脑如何打开护眼模式？",
        "item_names": ["华为MateBook B5-440笔记本电脑"],
        "reranked_docs": [
            {
                "text": "长期阅读时，建议您开启计算机护眼模式。右键点击桌面空白处，点击 显示更多选项 > 显示管理，点击开启护眼模式开关。开启护眼模式后，屏幕显示偏黄为正常现象。",
                "source": "local", "chunk_id": "chunk_001",
                "title": "F10 一键恢复出厂", "score": 0.9234,
            },
            {
                "text": "前往华为电脑管家的设置中心，在系统设置中，可以开启或关闭护眼模式。也可以使用快捷键 F10 快速打开华为电脑管家。",
                "source": "local", "chunk_id": "chunk_002",
                "title": "快捷键功能介绍", "score": 0.8756,
            },
        ],
        "kg_triples": [
            "[华为MateBook B5-440笔记本电脑] 切换功能键模式 -(HAS_STEP)-> 步骤1-按下Fn键",
            "[华为MateBook B5-440笔记本电脑] 切换功能键模式 -(HAS_STEP)-> 步骤2-设置功能键优先",
        ],
        "history": [],
    }

    print(f"\n【输入状态】")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  docs: {len(mock_state['reranked_docs'])} 篇")
    print(f"  kg_triples: {len(mock_state['kg_triples'])} 条")
    print("-" * 60)

    result = node_answer_output(mock_state)

    print(f"\n【提示词（前500字）】")
    prompt = result.get("prompt", "")
    print(prompt[:500])
    print("...")
    print("-" * 60)

    print(f"\n【最终答案】")
    print(result.get("answer", "无答案"))
