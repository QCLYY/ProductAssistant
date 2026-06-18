"""
知识图谱构建节点

从切片中提取实体和关系，写入 Neo4j（图）和 Milvus（实体向量）。

架构：
  KnowledgeGraphNode（流程编排，~360 行）
      ├── MilvusEntityWriter（Milvus 实体向量写入）→ writers/milvus_entity_writer.py
      └── Neo4jGraphWriter（Neo4j 图结构写入）     → writers/neo4j_graph_writer.py

处理流程：
1. 校验输入，清理旧数据（幂等）
2. 遍历 chunks，跳过无效切片
3. LLM 提取：每个切片通过大模型提取实体+关系 JSON（指数退避重试）
4. JSON 清洗：去围栏 → 解析 → 实体去重/截断 → 关系白名单/悬空过滤
5. Milvus 写入：委托 MilvusEntityWriter
6. Neo4j 写入：委托 Neo4jGraphWriter（事务内原子写入）
"""

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Set

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState

from knowledge.processor.import_process.nodes.writers import (
    MilvusEntityWriter,
    Neo4jGraphWriter,
)

from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.milvus_utils import get_milvus_client

from langchain_core.messages import SystemMessage, HumanMessage


class KnowledgeGraphNode(BaseNode):
    """知识图谱构建节点。

    对每个文本切片执行：LLM 实体/关系提取 → JSON 清洗 → Milvus 写入 → Neo4j 写入。
    单切片异常不阻断其他切片，外部存储写入失败会降级为警告日志。
    依赖 MilvusEntityWriter / Neo4jGraphWriter 处理存储细节。
    """

    name = "knowledge_graph"

    MAX_ENTITY_NAME_LENGTH = 20

    # ================================================================== #
    #                       1. LLM 提取提示词                              #
    # ================================================================== #

    # SYSTEM_PROMPT = """你是知识图谱信息抽取器。给你一段设备操作手册的文本切片，你必须抽取实体与关系，并只输出一个 JSON 对象（不要输出解释、不要 Markdown）。
    #
    # ## 允许的实体类型（label）
    # - Device：设备整体（如"万用表""仪表"）
    # - Part：部件或零件（如"电池后盖""螺母""表笔"）
    # - Operation：操作/功能名称（如"电池安装""电阻测量"），通常对应章节标题
    # - Step：操作步骤，name 用"步骤N-动作短语"格式（如"步骤1-断开表笔"），description 存原文
    # - Warning：警告/注意事项，name 用"警告-核心要点"格式（如"警告-操作前断开电源"），description 存原文
    # - Condition：前置条件或约束（如"电阻小于30Ω"）
    # - Tool：工具（如"螺丝刀"）
    #
    # ## 实体命名规则（非常重要）
    # - name 必须简短，不超过15个字。这是硬性要求。
    # - 禁止将整句原文作为 name。
    # - Step 格式：name="步骤N-动作短语"，description="原文完整步骤"
    # - Warning 格式：name="警告-核心要点"，description="原文完整警告"
    # - 同名同类型的实体只保留一个，不要重复。
    #
    # ## 允许的关系类型（type）
    # - HAS_OPERATION：Device → Operation
    # - HAS_PART：Device → Part
    # - HAS_STEP：Operation → Step
    # - USES_TOOL：Step → Tool
    # - HAS_WARNING：Operation/Step → Warning
    # - NEXT_STEP：Step → Step（按步骤顺序串联）
    # - AFFECTS：Step → Part（该步骤操作了哪个部件）
    # - REQUIRES：Step/Operation → Condition
    #
    # ## 抽取原则
    # - 只抽取文本中明确出现或可直接对应的实体与关系，禁止臆造。
    # - 步骤编号(1/2/3)时：每条作为 Step，并按顺序生成 NEXT_STEP 关系链。
    # - 关系的 head 和 tail 必须使用实体的 name 值（简短名），不要用 description。
    # - 如果无法判断某个关系，不要输出该关系。
    #
    # ## 输出 JSON Schema
    # {
    #   "entities": [
    #     {"name": "简短名称", "label": "类型", "description": "可选，原文内容或补充说明"}
    #   ],
    #   "relations": [
    #     {"head": "头实体name", "tail": "尾实体name", "type": "关系类型"}
    #   ]
    # }"""

    SYSTEM_PROMPT = """你是知识图谱信息抽取器。给你一段设备操作手册的文本切片，你必须抽取实体与关系，并只输出一个 JSON 对象（不要输出解释、不要 Markdown）。

## 允许的实体类型（label）
- Device：设备整体（如"万用表""仪表""计算机"）
- Part：部件或零件（如"电池后盖""螺母""表笔""HDMI接口""触摸板"）
- Operation：操作/功能/设置项（如"电池安装""电阻测量""护眼模式""夜间模式""恢复出厂"）。包括：章节标题中的操作名、文中提到的软件功能/系统设置项、硬件操作动作
- Step：操作步骤，name 用"步骤N-动作短语"格式（如"步骤1-断开表笔"），description 存原文
- Warning：警告/注意事项，name 用"警告-核心要点"格式（如"警告-操作前断开电源"），description 存原文
- Condition：前置条件或约束（如"电阻小于30Ω"）
- Tool：工具/软件（如"螺丝刀""华为电脑管家""扩展坞"）

## 实体命名规则（非常重要）
- name 必须简短，不超过15个字。这是硬性要求。
- 禁止将整句原文作为 name。
- Step 格式：name="步骤N-动作短语"，description="原文完整步骤"
- Warning 格式：name="警告-核心要点"，description="原文完整警告"
- Operation 不能只抽标题——文中明确提到的功能名、设置项（如"护眼模式"、"蓝牙"、"WiFi"）也必须作为 Operation 抽取
- 同名同类型的实体只保留一个，不要重复。

## 允许的关系类型（type）
- HAS_OPERATION：Device → Operation
- HAS_PART：Device → Part
- HAS_STEP：Operation → Step
- USES_TOOL：Step → Tool
- HAS_WARNING：Operation/Step → Warning
- NEXT_STEP：Step → Step（按步骤顺序串联）
- AFFECTS：Step → Part（该步骤操作了哪个部件）
- REQUIRES：Step/Operation → Condition

## 抽取原则
- 只抽取文本中明确出现或可直接对应的实体与关系，禁止臆造。
- 步骤编号(1/2/3)时：每条作为 Step，并按顺序生成 NEXT_STEP 关系链。
- 关系的 head 和 tail 必须使用实体的 name 值（简短名），不要用 description。
- 如果无法判断某个关系，不要输出该关系。
- **重要：软件功能/系统设置项（如"护眼模式""夜间模式""飞行模式""省电模式"等）应该作为 Operation 抽取，不要遗漏。**

## 输出 JSON Schema
{
  "entities": [
    {"name": "简短名称", "label": "类型", "description": "可选，原文内容或补充说明"}
  ],
  "relations": [
    {"head": "头实体name", "tail": "尾实体name", "type": "关系类型"}
  ]
}"""

    ALLOWED_RELATION_TYPES: Set[str] = {
        "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
        "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
        "MENTIONED_IN", "RELATED_TO",
    }

    ALLOWED_ENTITY_LABELS: Set[str] = {
        "Device", "Part", "Operation", "Step", "Warning", "Condition", "Tool",
    }

    def __init__(self, config=None):
        super().__init__(config)
        entity_collection = self.config.entity_name_collection
        self._milvus_writer = MilvusEntityWriter(entity_collection) if entity_collection else None
        self._neo4j_writer = Neo4jGraphWriter(database=self.config.neo4j_database)
        self._llm_semaphore = threading.Semaphore(2)   # LLM API 最多 2 并发
        self._embedding_lock = threading.Lock()         # PyTorch 向量化互斥

    # ================================================================== #
    #                        2. 主流程                                     #
    # ================================================================== #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()
        chunks = state.get("chunks", [])

        if not chunks:
            self.logger.info("chunks 为空，跳过知识图谱构建")
            return state

        if config.import_smoke_test:
            self.logger.warning("IMPORT_SMOKE_TEST=true，跳过知识图谱 LLM 抽取")
            return state

        self.log_step("start", f"开始处理 {len(chunks)} 个切片")

        # 预初始化外部资源
        milvus_client = None
        neo4j_driver = None
        item_name = state.get("item_name", "")

        # -- Milvus 初始化 + 幂等清理 --
        try:
            if self._milvus_writer:
                milvus_client = get_milvus_client()
                if item_name:
                    self._milvus_writer.clear(milvus_client, item_name)
        except Exception as e:
            self.logger.error(f"Milvus 初始化/清理失败: {e}")

        # -- Neo4j 初始化 + 幂等清理 --
        try:
            from knowledge.tools.neo4j_utils import get_neo4j_driver
            neo4j_driver = get_neo4j_driver()
            if item_name:
                self._neo4j_writer.clear(neo4j_driver, item_name)
        except Exception as e:
            self.logger.error(f"Neo4j 初始化/清理失败: {e}")

        llm = get_llm_client(json_mode=True)
        success = 0

        # 收集有效切片
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            content = chunk.get("content", "")
            chunk_id = self._resolve_chunk_id(chunk, content, i)
            chunk_item = chunk.get("item_name") or state.get("item_name", "")
            if not content or not chunk_item:
                self.logger.debug(f"跳过切片 {i}: content 或 item_name 为空")
                continue
            valid_chunks.append((content, chunk_id, chunk_item))

        if not valid_chunks:
            self.log_step("end", "无有效切片")
            return state

        total = len(valid_chunks)

        # 线程池并发处理（LLM 信号量限制 2 并发，向量化互斥锁串行）
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    self._process_single_chunk,
                    content, chunk_id, chunk_item, llm,
                    milvus_client, neo4j_driver, idx, total,
                ): chunk_id
                for idx, (content, chunk_id, chunk_item) in enumerate(valid_chunks, 1)
            }

            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    future.result()
                    success += 1
                except Exception as e:
                    self.logger.warning(f"切片 {chunk_id} 处理失败: {e}")

        self.log_step("end", f"知识图谱构建完成，{success}/{total} 个切片成功")
        self.log_step("end", f"知识图谱构建完成，{success}/{total} 个切片成功")
        return state

    def _process_single_chunk(
        self,
        content: str,
        chunk_id: str,
        item_name: str,
        llm,
        milvus_client,
        neo4j_driver,
        idx: int = 0,
        total: int = 0,
    ):
        """处理单个切片：LLM 提取 → 清洗 → Milvus 写入 → Neo4j 写入。"""
        progress = f"({idx}/{total})" if idx and total else ""

        # 1. LLM 提取（带指数退避重试）
        raw_response = self._extract_graph_with_retry(content, llm)
        if not raw_response:
            return

        # 2. 解析 JSON 并清洗
        graph_data = self._parse_and_clean(raw_response)
        entities = graph_data.get("entities", [])
        relations = graph_data.get("relations", [])

        if not entities:
            return

        self.logger.info(
            f"{progress} 切片 {chunk_id}: "
            f"提取到 {len(entities)} 个实体, {len(relations)} 条关系"
        )

        # 3. 实体写入 Milvus（委托 Writer，加锁保护 PyTorch 模型）
        if self._milvus_writer and milvus_client:
            try:
                self._milvus_writer.insert(
                    milvus_client, entities, chunk_id, content, item_name,
                    embedding_lock=self._embedding_lock,
                )
                self.logger.info(
                    f"{progress} Milvus 写入 {len(entities)} 条实体向量"
                )
            except Exception as e:
                self.logger.warning(f"{progress} Milvus 写入失败 (chunk={chunk_id}): {e}")

        # 4. 图数据写入 Neo4j（委托 Writer）
        if neo4j_driver:
            try:
                self._neo4j_writer.insert(neo4j_driver, entities, relations, chunk_id, item_name)
                self.logger.info(
                    f"{progress} Neo4j 写入: {len(entities)} 实体, {len(relations)} 关系"
                )
            except Exception as e:
                self.logger.warning(f"{progress} Neo4j 写入失败 (chunk={chunk_id}): {e}")

    # ================================================================== #
    #                      3. LLM 提取（带重试）                           #
    # ================================================================== #

    def _extract_graph_with_retry(self, content: str, llm) -> str:
        """LLM 提取实体与关系，指数退避重试，最多 3 次。受 _llm_semaphore 限流。"""
        last_error = None
        with self._llm_semaphore:
            for attempt in range(1, 4):
                try:
                    response = llm.invoke([
                        SystemMessage(content=self.SYSTEM_PROMPT),
                        HumanMessage(content=f"文本切片\n\n{content}"),
                    ])
                    result = (response.content or "").strip()
                    if result:
                        return result
                except Exception as e:
                    last_error = e
                    if attempt < 3:
                        delay = 0.5 * (2 ** (attempt - 1))
                        self.logger.warning(
                            f"LLM 调用失败（第 {attempt} 次），{delay:.1f}s 后重试: {e}"
                        )
                        time.sleep(delay)

        self.logger.error(f"LLM 提取最终失败（3 次）: {last_error}")
        return ""

    # ================================================================== #
    #                   4. JSON 解析与清洗                                 #
    # ================================================================== #

    def _parse_and_clean(self, raw_text: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 并执行多层清洗。"""
        if not raw_text:
            return {"entities": [], "relations": []}

        cleaned_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 解析失败: {e}, 原文前200字: {raw_text[:200]}")
            return {"entities": [], "relations": []}

        cleaned_entities = self._clean_entities(data.get("entities", []))
        valid_names = {e["name"] for e in cleaned_entities}
        cleaned_relations = self._clean_relations(
            data.get("relations", []), valid_names
        )

        return {"entities": cleaned_entities, "relations": cleaned_relations}

    # ---------- 实体清洗 ----------

    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """清洗实体：过滤无效项、截断过长名称、白名单校验、去重。"""
        seen: Set[tuple] = set()
        cleaned: List[Dict] = []

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            name = str(entity.get("name", "")).strip()
            label = str(entity.get("label", "")).strip()
            description = str(entity.get("description", "")).strip()

            if not name or not label:
                continue

            if label not in self.ALLOWED_ENTITY_LABELS:
                self.logger.debug(f"非法实体类型已跳过: {label}")
                continue

            if len(name) > self.MAX_ENTITY_NAME_LENGTH:
                name = name[:self.MAX_ENTITY_NAME_LENGTH]

            dedup_key = (name, label)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            result = {"name": name, "label": label}
            if description:
                result["description"] = description
            cleaned.append(result)

        return cleaned

    # ---------- 关系清洗 ----------

    def _clean_relations(
        self,
        relations: List[Dict],
        valid_entity_names: Set[str],
    ) -> List[Dict]:
        """清洗关系：修正字段、白名单校验关系类型、过滤悬空引用。"""
        cleaned: List[Dict] = []

        for rel in relations:
            if not isinstance(rel, dict):
                continue

            head = str(rel.get("head", "")).strip()
            tail = str(rel.get("tail", "")).strip()
            if not head or not tail:
                continue

            if len(head) > self.MAX_ENTITY_NAME_LENGTH:
                head = head[:self.MAX_ENTITY_NAME_LENGTH]
            if len(tail) > self.MAX_ENTITY_NAME_LENGTH:
                tail = tail[:self.MAX_ENTITY_NAME_LENGTH]

            rel_type = str(
                rel.get("type") or rel.get("label") or "RELATED_TO"
            ).strip()
            if rel_type not in self.ALLOWED_RELATION_TYPES:
                self.logger.debug(f"非法关系类型降级: {rel_type} → RELATED_TO")
                rel_type = "RELATED_TO"

            if head not in valid_entity_names:
                self.logger.debug(f"悬空关系 head 跳过: {head}")
                continue
            if tail not in valid_entity_names:
                self.logger.debug(f"悬空关系 tail 跳过: {tail}")
                continue

            cleaned.append({"head": head, "tail": tail, "type": rel_type})

        return cleaned

    # ================================================================== #
    #                         辅助方法                                    #
    # ================================================================== #

    @staticmethod
    def _resolve_chunk_id(chunk: Dict, content: str, index: int) -> str:
        """解析或生成稳定的 chunk_id。"""
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

node_knowledge_graph = KnowledgeGraphNode()


def test_kg_extraction():
    """测试：模拟单个切片，跑通 LLM 提取 → 解析清洗全流程。"""
    print("=== 开始测试知识图谱构建流程 ===\n")

    mock_state = {
        "item_name": "万用表",
        "chunks": [
            {
                "content": """# 电池安装
警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。
1. 把表笔与仪表断开。
2. 用螺丝刀拧开电池后盖上的螺母。
3. 正确安装电池，正负极应一致。
4. 盖上电池后盖并拧紧螺丝钉。
警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。
注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。""",
                "chunk_id": "chunk_test_001",
                "item_name": "万用表",
            }
        ]
    }

    node_knowledge_graph.process(mock_state)
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    setup_logging()
    test_kg_extraction()
