"""
MilvusEntityWriter — 实体向量化写入 Milvus

负责将清洗后的实体名称通过 BGE-M3 向量化（稠密 + 稀疏双向量），
写入 Milvus 实体集合，为图增强检索提供向量入口。

对外暴露两个方法：
- clear(): 导入前清理该 item_name 下的旧数据（幂等）
- insert(): 实体去重 → 向量化 → 建集合 → 批量插入
"""

import logging
from typing import Dict, List, Any

from knowledge.processor.import_process.exceptions import MilvusError
from knowledge.tools.embedding_utils import get_bge_m3_model
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector
from pymilvus import DataType


class MilvusEntityWriter:
    """负责将实体向量化并写入 Milvus。"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    # ---------- 公开方法 ----------

    def clear(self, milvus_client, item_name: str) -> None:
        """导入前清理该 item_name 下的所有旧 Milvus 数据。"""
        if not milvus_client:
            raise MilvusError("Milvus 客户端获取失败")

        try:
            if milvus_client.has_collection(self.collection_name):
                # 确保 collection 已加载（filter 操作依赖 loaded 状态）
                milvus_client.load_collection(collection_name=self.collection_name)

                filter_expr = f'item_name == "{item_name}"'

                # 先 query 验证 filter 能命中数据
                check = milvus_client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=["entity_name"],
                    limit=1,
                )
                self.logger.info(f"clear 预查询: filter={filter_expr}, 命中={len(check)} 条")

                # 执行删除
                result = milvus_client.delete(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                )
                delete_count = result.get("delete_count", 0) if result else 0
                self.logger.info(
                    f"Milvus 旧数据已清理: item_name={item_name}, filter={filter_expr}, "
                    f"删除 {delete_count} 条"
                )
        except Exception as e:
            raise MilvusError(f"Milvus 清理失败: {e}")

    def insert(
        self,
        milvus_client,
        entities: List[Dict],
        chunk_id: str,
        content: str,
        item_name: str,
        embedding_lock=None,
    ) -> None:
        """对外唯一入口：实体向量化 + 写入 Milvus。

        embedding_lock: 可选 threading.Lock，用于保护 PyTorch 模型并发推理。
        """
        if not entities:
            raise ValueError("参数校验失败，实体不存在")

        entities_names = list({e["name"] for e in entities})
        if not entities_names:
            raise ValueError("参数校验失败，无有效实体名")

        bge_ef_model = get_bge_m3_model()
        if bge_ef_model is None:
            raise MilvusError("嵌入模型获取失败")

        try:
            self._ensure_collection(milvus_client)
        except Exception as e:
            raise MilvusError(f"Milvus 创建集合失败: {e}")

        # 向量化（加锁保护 PyTorch 模型，多线程下必须串行）
        try:
            if embedding_lock:
                with embedding_lock:
                    embedded_result = bge_ef_model.encode_documents(entities_names)
            else:
                embedded_result = bge_ef_model.encode_documents(entities_names)
        except Exception as e:
            raise MilvusError(f"实体嵌入失败: {e}")

        records = self._build_records(entities_names, embedded_result, chunk_id, content, item_name)
        if not records:
            raise MilvusError("构建 Milvus 记录为空")

        try:
            milvus_client.insert(collection_name=self.collection_name, data=records)
            milvus_client.load_collection(collection_name=self.collection_name)
            self.logger.info(f"Milvus 写入 {len(records)} 条实体向量")
        except Exception as e:
            raise MilvusError(f"Milvus 插入数据失败: {e}")

    # ---------- 私有方法 ----------

    def _ensure_collection(self, client) -> None:
        """幂等创建实体集合（含 Schema 和双向量索引）。"""
        if client.has_collection(self.collection_name):
            return

        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field(field_name="pk",            datatype=DataType.INT64,
                         is_primary=True, auto_id=True)
        schema.add_field(field_name="entity_name",   datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="dense_vector",  datatype=DataType.FLOAT_VECTOR,
                         dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="source_chunk_id", datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="context",       datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="item_name",     datatype=DataType.VARCHAR,
                         max_length=65535)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _build_records(
        entities_names: List[str],
        embedded_result: Dict[str, Any],
        chunk_id: str,
        content: str,
        item_name: str,
    ) -> List[Dict[str, Any]]:
        """组装 Milvus 插入记录，从 CSR 矩阵提取稀疏向量并 L2 归一化。"""
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        dense_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")
        if not dense_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        context = content[:200]
        records: List[Dict] = []

        for idx, name in enumerate(entities_names):
            if idx >= len(dense_list):
                break

            dense = dense_list[idx]
            if hasattr(dense, "tolist"):
                dense = dense.tolist()

            record: Dict[str, Any] = {
                "entity_name": name,
                "dense_vector": dense,
                "source_chunk_id": chunk_id,
                "context": context,
                "item_name": item_name,
            }

            s = sparse_matrix.indptr[idx]
            e = sparse_matrix.indptr[idx + 1]
            indices = sparse_matrix.indices[s:e].tolist()
            data = sparse_matrix.data[s:e].tolist()
            sparse_dict = normalize_sparse_vector(dict(zip(indices, data)))
            record["sparse_vector"] = sparse_dict

            records.append(record)

        return records
