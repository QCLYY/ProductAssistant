"""
商品名称识别节点

从文档切片中调用 LLM 识别商品/产品名称，
使用 BGE-M3 生成混合嵌入（稠密 + 稀疏向量），
持久化到 Milvus 向量数据库，回填 item_name 到 state 和 chunks。
"""

import json
import os
import pathlib
from typing import List, Tuple, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError

from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.embedding_utils import get_bge_m3_model
from knowledge.tools.milvus_utils import get_milvus_client
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector

# BGE-M3 输出的稠密向量维度
BGE_M3_DIM = 1024


class ItemNameRecognitionNode(BaseNode):
    """
    商品名称识别节点

    处理流程（6 步）：
    1. 验证输入（file_title + chunks 非空）
    2. 从前 K 个切片构造识别上下文
    3. 调用 LLM 识别商品名称（品牌 + 型号 + 名称）
    4. 回填 item_name 到 state 和每个 chunk
    5. 使用 BGE-M3 生成混合嵌入向量（稠密 + 稀疏）
    6. 保存到 Milvus 向量数据库
    """

    name = "item_name_recognition"

    # ------------------------------------------------------------------ #
    #                           主流程                                     #
    # ------------------------------------------------------------------ #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # Step 1: 验证输入
        file_title, chunks = self._validate_inputs(state)

        # Step 2: 构造识别上下文
        context = self._build_context(chunks, config.item_name_chunk_k)

        # Step 3: 调用 LLM 识别
        item_name = self._recognize_item_name(file_title, context, config)

        # Step 4: 回填到 state 和 chunks
        self._backfill_item_name(state, chunks, item_name)

        # Step 5: 生成向量
        dense_vector, sparse_vector = self._generate_vectors(item_name, config)

        # Step 6: 保存到 Milvus
        self._save_to_milvus(
            state, file_title, item_name,
            dense_vector, sparse_vector, config,
        )

        return state

    # ------------------------------------------------------------------ #
    #                       Step 1: 验证输入                               #
    # ------------------------------------------------------------------ #

    def _validate_inputs(
        self, state: ImportGraphState
    ) -> Tuple[str, List[dict]]:
        self.log_step("step_1", "验证输入")

        file_title = state.get("file_title", "")
        chunks = state.get("chunks", [])

        if not file_title:
            raise ValidationError("file_title 为空", node_name=self.name)

        if not isinstance(chunks, list) or not chunks:
            raise ValidationError(
                "chunks 为空或无效，请确认 document_split 节点已执行",
                node_name=self.name,
            )

        self.logger.info(f"文件标题: {file_title}, 切片数: {len(chunks)}")
        return file_title, chunks

    # ------------------------------------------------------------------ #
    #                    Step 2: 构造识别上下文                              #
    # ------------------------------------------------------------------ #

    def _build_context(
        self, chunks: List[dict], k: int, max_chars: int = 2500
    ) -> str:
        self.log_step("step_2", "构造识别上下文")

        parts = []
        total = 0

        for i, chunk in enumerate(chunks[:k]):
            if not isinstance(chunk, dict):
                continue

            title = (chunk.get("title") or "").strip()
            content = (chunk.get("content") or "").strip()

            if not (title or content):
                continue

            # 截断过长内容，保留首部关键信息
            if len(content) > 800:
                content = content[:800] + "..."

            piece = f"【切片{i + 1}】\n标题：{title}\n内容：{content}"
            parts.append(piece)
            total += len(piece)

            if total >= max_chars:
                break

        context = "\n\n".join(parts)[:max_chars]
        self.logger.debug(f"构造上下文长度: {len(context)} 字符")
        return context

    # ------------------------------------------------------------------ #
    #                   Step 3: 调用 LLM 识别                              #
    # ------------------------------------------------------------------ #

    def _recognize_item_name(
        self, file_title: str, context: str, config
    ) -> str:
        self.log_step("step_3", "调用 LLM 识别")

        if not (config.openai_api_base and config.openai_api_key and (config.item_model or config.default_model)):
            self.logger.warning("LLM 配置不完整，回退使用文件标题作为商品名称")
            return file_title

        prompt = f"""请从以下信息中识别出商品名称与型号：
文件名：{file_title}

正文切片（用于辅助识别）：
{context}

要求：
1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
3. 如果无法识别商品名称，请返回空字符串。"""

        try:
            llm = get_llm_client(model=config.item_model, json_mode=False)
            resp = llm.invoke([
                SystemMessage(content="你是商品识别专家，只输出字符串。"),
                HumanMessage(content=prompt),
            ])

            item_name = getattr(resp, "content", "").strip().strip('"').strip("'").strip()

            if not item_name:
                self.logger.warning("LLM 未能识别商品名称，回退使用文件标题")
                item_name = file_title

            self.logger.info(f"识别结果: {item_name}")
            return item_name

        except Exception as e:
            self.logger.warning(
                f"LLM 调用失败: {e}，回退使用文件标题作为商品名称"
            )
            return file_title

    # ------------------------------------------------------------------ #
    #                    Step 4: 回填 item_name                            #
    # ------------------------------------------------------------------ #

    def _backfill_item_name(
        self, state: ImportGraphState, chunks: List[dict], item_name: str
    ):
        self.log_step("step_4", "回填 item_name")

        state["item_name"] = item_name

        for chunk in chunks:
            chunk["item_name"] = item_name

        state["chunks"] = chunks
        self.logger.debug(f"已将 item_name 回填到 {len(chunks)} 个切片")

    # ------------------------------------------------------------------ #
    #                    Step 5: 生成混合嵌入向量                            #
    # ------------------------------------------------------------------ #

    def _generate_vectors(
        self, item_name: str, config
    ) -> Tuple[Optional[List[float]], Optional[dict]]:
        self.log_step("step_5", "生成向量")

        if config.import_smoke_test:
            self.logger.warning("IMPORT_SMOKE_TEST=true，使用占位向量跳过 BGE-M3")
            return [0.0] * BGE_M3_DIM, {0: 1.0}

        try:
            bge_m3_ef = get_bge_m3_model()
            vectors = bge_m3_ef.encode_documents([item_name])

            if not vectors:
                self.logger.warning("BGE-M3 编码返回空结果")
                return None, None

            # 稠密向量
            dense_vector = vectors["dense"][0].tolist()

            # 稀疏向量：从 CSR 矩阵提取 {token_id: weight}
            sparse_matrix = vectors["sparse"]
            start = sparse_matrix.indptr[0]
            end = sparse_matrix.indptr[1]
            token_ids = sparse_matrix.indices[start:end].tolist()
            weights = sparse_matrix.data[start:end].tolist()
            sparse_vector = dict(zip(token_ids, weights))

            self.logger.info(
                f"向量生成成功: dense[{len(dense_vector)}], "
                f"sparse[{len(sparse_vector)} tokens]"
            )
            return dense_vector, sparse_vector

        except Exception as e:
            self.logger.warning(f"向量生成失败: {e}")
            return None, None

    # ------------------------------------------------------------------ #
    #                    Step 6: 保存到 Milvus                             #
    # ------------------------------------------------------------------ #

    def _save_to_milvus(
        self,
        state: ImportGraphState,
        file_title: str,
        item_name: str,
        dense_vector: Optional[List[float]],
        sparse_vector: Optional[dict],
        config,
    ):
        self.log_step("step_6", "保存到 Milvus")

        if not config.milvus_url or not config.item_name_collection:
            self.logger.warning("Milvus 配置不完整，跳过保存")
            return

        if dense_vector is None and sparse_vector is None:
            self.logger.warning("向量均为空，跳过 Milvus 保存")
            return

        try:
            client = get_milvus_client()
            collection_name = config.item_name_collection

            # 检查/创建集合
            if not client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(client, collection_name)

            # 准备数据
            data = {
                "file_title": file_title,
                "item_name": item_name,
            }
            if dense_vector is not None:
                data["dense_vector"] = dense_vector
            if sparse_vector is not None:
                data["sparse_vector"] = normalize_sparse_vector(sparse_vector)

            # 插入
            result = client.insert(collection_name=collection_name, data=[data])
            insert_ids = result.get("ids", [])
            self.logger.info(f"已保存到 Milvus 集合 {collection_name}，ID: {insert_ids}")

        except Exception as e:
            self.logger.warning(f"Milvus 保存失败: {e}")

    # ------------------------------------------------------------------ #
    #                   辅助: 创建 Milvus 集合                              #
    # ------------------------------------------------------------------ #

    def _create_item_name_collection(self, client, collection_name: str):
        """创建 item_name 集合的 Schema 和索引"""
        from pymilvus import DataType

        self.logger.info(f"创建 Milvus 集合: {collection_name}")

        schema = client.create_schema(enable_dynamic_fields=True)
        schema.add_field(
            field_name="pk", datatype=DataType.VARCHAR,
            is_primary=True, auto_id=True, max_length=100,
        )
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
            dim=BGE_M3_DIM,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 索引配置
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.logger.info(f"Milvus 集合 {collection_name} 创建完成")


# ================================================================== #
#                              测试                                   #
# ================================================================== #

if __name__ == "__main__":
    setup_logging()
    node = ItemNameRecognitionNode()

    print("=" * 60)
    print("ItemNameRecognitionNode 节点测试")
    print("=" * 60)

    # ——— 测试用例 1: 模拟数据 ———
    print("\n--- 测试用例 1: 模拟数据 ---")
    path=pathlib.Path(r"D:\path\to\ProductAssistant\knowledge\processor\import_process\import_temp_dir\chunks.json")
    with open(path, "rt", encoding="utf-8") as f:
        mock_chunks = json.load(f)
    try:
        result = node.process({
            "file_title": "hak180产品安全手册",
            "chunks": mock_chunks,
        })
        print(f"item_name: {result.get('item_name', 'N/A')}")
        print(f"chunks[0].item_name: {result['chunks'][0].get('item_name', 'N/A')}")
    except Exception as e:
        print(f"测试失败: {e}")

    # ——— 测试用例 2: 空 chunks 触发异常 ———
    print("\n--- 测试用例 2: 空 chunks (预期异常) ---")
    try:
        node.process({"file_title": "test", "chunks": []})
    except ValidationError as e:
        print(f"捕获到预期异常: {e}")

    # ——— 测试用例 3: 空 file_title 触发异常 ———
    print("\n--- 测试用例 3: 空 file_title (预期异常) ---")
    try:
        node.process({"file_title": "", "chunks": mock_chunks})
    except ValidationError as e:
        print(f"捕获到预期异常: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
