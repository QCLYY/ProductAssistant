"""
BGE-M3 切片向量化节点

为每个文档切片生成稠密向量 + 稀疏向量，支持批量处理与错误容忍。
"""

import json
import os
from typing import List, Optional

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import EmbeddingError
from knowledge.tools.embedding_utils import get_bge_m3_model
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector


class BgeEmbeddingNode(BaseNode):
    """
    BGE-M3 向量化节点

    处理流程：
    1. 获取 chunks，验证非空
    2. 初始化 BGE-M3 模型（全局单例，首次加载到 GPU）
    3. 按 batch 分批生成稠密 + 稀疏向量
    4. 向量回填到每个 chunk，写回 state
    """

    name = "bge_embedding"

    # ------------------------------------------------------------------ #
    #                           主流程                                     #
    # ------------------------------------------------------------------ #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # Step 1: 获取并验证切片
        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise EmbeddingError("chunks 为空或无效", node_name=self.name)

        self.log_step("step_1", f"开始为 {len(chunks)} 个切片生成向量")

        if config.import_smoke_test:
            self.logger.warning("IMPORT_SMOKE_TEST=true，使用占位向量跳过 BGE-M3")
            state["chunks"] = [
                {
                    "content": chunk.get("content"),
                    "title": chunk.get("title"),
                    "parent_title": chunk.get("parent_title", ""),
                    "part": chunk.get("part", 0),
                    "file_title": chunk.get("file_title"),
                    "item_name": chunk.get("item_name"),
                    "dense_vector": [0.0] * config.embedding_dim,
                    "sparse_vector": {0: 1.0},
                }
                for chunk in chunks
            ]
            return state

        # Step 2: 初始化 BGE-M3
        try:
            bge_m3_ef = get_bge_m3_model()
        except Exception as e:
            raise EmbeddingError(
                f"初始化 BGE-M3 失败: {e}", node_name=self.name, cause=e
            )

        # Step 3: 分批处理
        output_data = []
        batch_size = config.embedding_batch_size
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            batch_output = self._process_batch(bge_m3_ef, batch, i, total)
            output_data.extend(batch_output)

        # Step 4: 写回 state
        self.log_step("step_2", f"向量化完成，共 {len(output_data)} 个切片")
        state["chunks"] = output_data
        return state

    # ------------------------------------------------------------------ #
    #                        批次处理                                      #
    # ------------------------------------------------------------------ #

    def _process_batch(
        self,
        bge_m3_ef,
        batch: List[dict],
        start_idx: int,
        total: int,
    ) -> List[dict]:
        """处理一个批次的切片，失败时返回原始数据（不含向量）。"""
        try:
            # 构造输入文本：item_name + content，将商品名编码进向量
            texts = [
                (doc.get("item_name", "") or "")
                + "\n"
                + (doc.get("content", "") or "")
                for doc in batch
            ]

            embeddings = bge_m3_ef.encode_documents(texts)

            if not embeddings:
                self.logger.warning(
                    f"批次 {start_idx + 1}-{start_idx + len(batch)} 未能生成向量"
                )
                return batch

            output = []
            for j, doc in enumerate(batch):
                # 稠密向量
                dense_vector = embeddings["dense"][j].tolist()

                # 稀疏向量：从 CSR 矩阵提取并 L2 归一化
                sparse_matrix = embeddings["sparse"]
                s, e = sparse_matrix.indptr[j], sparse_matrix.indptr[j + 1]
                token_ids = sparse_matrix.indices[s:e].tolist()
                weights = sparse_matrix.data[s:e].tolist()
                sparse_vector = normalize_sparse_vector(dict(zip(token_ids, weights)))

                output.append({
                    "content": doc.get("content"),
                    "title": doc.get("title"),
                    "parent_title": doc.get("parent_title", ""),
                    "part": doc.get("part", 0),
                    "file_title": doc.get("file_title"),
                    "item_name": doc.get("item_name"),
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                })

            end_idx = min(start_idx + len(batch), total)
            self.logger.info(f"成功处理批次 {start_idx + 1}-{end_idx}/{total}")
            return output

        except Exception as e:
            self.logger.error(f"批次 {start_idx + 1} 处理失败: {e}")
            # 返回原始数据，不含向量——单批失败不阻断全局
            return batch


# ================================================================== #
#                              测试                                   #
# ================================================================== #

if __name__ == "__main__":
    setup_logging()
    node = BgeEmbeddingNode()

    # 构造模拟数据（含 item_name，模拟 item_name_recognition 节点的输出）
    mock_chunks = [
        {
            "content": "# HAK 180 烫金机\n\n感谢您购买 HAK 180 烫金机。",
            "title": "# HAK 180 烫金机",
            "parent_title": "hak180产品安全手册",
            "file_title": "hak180产品安全手册",
            "item_name": "HAK 180 烫金机",
        },
        {
            "content": "# 设备\n\n请先阅读这本手册，再尝试操作本设备。",
            "title": "# 设备",
            "parent_title": "hak180产品安全手册",
            "file_title": "hak180产品安全手册",
            "item_name": "HAK 180 烫金机",
        },
    ]

    try:
        result = node.process({"chunks": mock_chunks})
        chunks_out = result.get("chunks", [])
        print(f"\n处理完成，共 {len(chunks_out)} 个切片")
        for c in chunks_out:
            dv = c.get("dense_vector", [])
            sv = c.get("sparse_vector", {})
            print(f"  title: {c['title'][:40]}")
            print(f"  dense dim: {len(dv)}, sparse tokens: {len(sv)}")
    except Exception as e:
        print(f"测试失败: {e}")
