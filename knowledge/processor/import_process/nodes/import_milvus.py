"""
Milvus 向量导入节点

将向量化后的切片数据批量导入 Milvus，自动创建集合和索引，
并将 Milvus 生成的 chunk_id 回填到业务数据中。
"""

import json
import os
from typing import List, Dict

from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import MilvusError
from knowledge.tools.milvus_utils import get_milvus_client


class ImportMilvusNode(BaseNode):
    """
    Milvus 向量导入节点

    处理流程：
    1. 校验 chunks（非空 + 含向量）
    2. 连接 Milvus
    3. 集合不存在则自动创建（Schema + 索引）
    4. 批量插入数据
    5. 回填 chunk_id 到每个 chunk
    6. 写回 state
    """

    name = "import_milvus"

    # ================================================================== #
    #                           主流程                                     #
    # ================================================================== #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()
        chunks = state.get("chunks", [])

        # Step 1: 校验
        if not chunks:
            self.logger.warning("chunks 为空，跳过导入")
            return state

        vector_dim = self._get_vector_dim(chunks)
        total = len(chunks)
        self.log_step("step_1", f"准备导入 {total} 条数据，向量维度: {vector_dim}")

        # Step 2-5: 连接 → 建集合 → 插入 → 回填
        try:
            client = get_milvus_client()
            collection_name = config.chunks_collection

            if not client.has_collection(collection_name=collection_name):
                self.log_step("step_2", f"创建集合: {collection_name}")
                self._create_collection(client, collection_name, vector_dim)

            self.log_step("step_3", "执行插入")
            self._insert_and_backfill_ids(client, collection_name, chunks)

        except MilvusError:
            raise
        except Exception as e:
            raise MilvusError(
                f"Milvus 操作失败: {e}", node_name=self.name, cause=e
            )

        # Step 6: 写回 state
        state["chunks"] = chunks
        self.logger.info(f"导入完成，{total} 条数据已写入集合 {collection_name}")
        return state

    # ================================================================== #
    #                     2. 集合创建                                      #
    # ================================================================== #

    def _create_collection(
        self, client, collection_name: str, vector_dim: int
    ):
        """创建 Milvus 集合（Schema + 双向量索引）。"""
        schema = self._build_schema(client, vector_dim)
        index_params = self._build_index_params(client)

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.logger.info(f"集合 {collection_name} 创建完成")

    def _build_schema(self, client, vector_dim: int):
        """构建集合 Schema（1 主键 + 6 标量 + 2 向量）。"""
        schema = client.create_schema(enable_dynamic_fields=True)

        # 主键（INT64 自增）
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        # 标量字段
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        # 向量字段
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

        return schema

    def _build_index_params(self, client):
        """构建双向量索引（稠密 AUTOINDEX + 稀疏倒排索引）。"""
        index_params = client.prepare_index_params()

        # 稠密向量：自动选择最优算法
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        # 稀疏向量：倒排索引 + DAAT_MAXSCORE 动态剪枝
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        return index_params

    # ================================================================== #
    #                  3. 数据插入与 chunk_id 回填                         #
    # ================================================================== #

    @staticmethod
    def _get_vector_dim(chunks: List[Dict]) -> int:
        dim = len(chunks[0].get("dense_vector", []))
        if dim == 0:
            raise MilvusError(
                "切片数据不包含 dense_vector", node_name="import_milvus"
            )
        return dim

    def _insert_and_backfill_ids(
        self, client, collection_name: str, chunks: List[Dict]
    ):
        """批量插入并回填 Milvus 自动生成的 chunk_id。"""
        result = client.insert(collection_name=collection_name, data=chunks)
        insert_count = result.get("insert_count", 0)
        self.logger.info(f"成功插入 {insert_count} 条数据")

        inserted_ids = result.get("ids", [])
        if inserted_ids and len(inserted_ids) == len(chunks):
            for chunk, chunk_id in zip(chunks, inserted_ids):
                chunk["chunk_id"] = str(chunk_id)
            self.logger.info(f"已回填 {len(inserted_ids)} 个 chunk_id")
        else:
            self.logger.warning(
                f"chunk_id 回填异常: 返回 {len(inserted_ids)} 个 ID，"
                f"期望 {len(chunks)} 个"
            )


# ================================================================== #
#                              测试                                   #
# ================================================================== #

if __name__ == "__main__":
    setup_logging()
    node = ImportMilvusNode()

    # 模拟向量化后的 chunks
    mock_chunks = [
        {
            "content": "测试内容",
            "title": "测试标题",
            "parent_title": "root",
            "part": 0,
            "file_title": "test.pdf",
            "item_name": "测试商品",
            "dense_vector": [0.1] * 1024,
            "sparse_vector": {1: 0.5, 2: 0.3},
        }
    ]

    try:
        result = node.process({"chunks": mock_chunks})
        chunks_out = result.get("chunks", [])
        print(f"\n导入完成，{len(chunks_out)} 条数据")
        for c in chunks_out:
            print(f"  chunk_id: {c.get('chunk_id', '未回填')}")
            print(f"  title: {c.get('title', '')}")
    except Exception as e:
        print(f"测试失败: {e}")
