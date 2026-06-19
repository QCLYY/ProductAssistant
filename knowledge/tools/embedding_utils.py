"""BGE-M3 hybrid embedding helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

from scipy.sparse import csr_matrix

from knowledge.processor.import_process.config import get_config
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector

logger = logging.getLogger("import.embedding")

_bge_m3: Optional[Any] = None


class BGEM3FlagAdapter:
    """Compatibility wrapper around FlagEmbedding's BGEM3FlagModel.

    The older code expects ``encode_documents`` / ``encode_queries`` to return:
    ``{"dense": ndarray, "sparse": scipy.csr_matrix}``.
    """

    def __init__(self, model_path: str, device: str, use_fp16: bool):
        from FlagEmbedding import BGEM3FlagModel

        if device.lower().startswith("cpu"):
            use_fp16 = False

        self.model = BGEM3FlagModel(
            model_path,
            devices=device,
            use_fp16=use_fp16,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def encode_documents(self, texts: list[str]) -> dict[str, Any]:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> dict[str, Any]:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        result = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = result.get("dense_vecs")
        sparse = self._lexical_weights_to_csr(result.get("lexical_weights") or [])
        return {"dense": dense, "sparse": sparse}

    @staticmethod
    def _lexical_weights_to_csr(weights_list: list[dict[str, float]]) -> csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        max_col = 0

        for row_index, weights in enumerate(weights_list):
            for raw_token_id, raw_weight in (weights or {}).items():
                try:
                    token_id = int(raw_token_id)
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if weight == 0:
                    continue
                rows.append(row_index)
                cols.append(token_id)
                data.append(weight)
                max_col = max(max_col, token_id)

        shape = (len(weights_list), max_col + 1 if max_col else 1)
        return csr_matrix((data, (rows, cols)), shape=shape, dtype="float32")


def get_bge_m3_model(device: Optional[str] = None, use_fp16: Optional[bool] = None) -> Any:
    """Return the cached BGE-M3 embedding model."""
    global _bge_m3
    if _bge_m3 is not None:
        return _bge_m3

    config = get_config()
    device = device or getattr(config, "bge_device", "cpu")
    use_fp16 = use_fp16 if use_fp16 is not None else getattr(config, "bge_fp16", True)
    model_path = getattr(config, "bge_m3_path", "BAAI/bge-m3")

    logger.info("加载 BGE-M3 模型: %s, device=%s, fp16=%s", model_path, device, use_fp16)
    _bge_m3 = BGEM3FlagAdapter(model_path=model_path, device=device, use_fp16=use_fp16)
    logger.info("BGE-M3 模型加载完成")
    return _bge_m3


def generate_hybrid_embeddings(texts: list[str]) -> dict[str, list[Any]]:
    """Generate dense vectors and normalized sparse vectors for query texts."""
    model = get_bge_m3_model()
    embeddings = model.encode_queries(texts)

    dense_list = embeddings["dense"]
    if hasattr(dense_list, "tolist"):
        dense_list = [d.astype("float32").tolist() for d in dense_list]
    elif dense_list and hasattr(dense_list[0], "dtype"):
        dense_list = [d.astype("float32").tolist() for d in dense_list]

    sparse_matrix = embeddings["sparse"]
    sparse_list = []
    for index in range(sparse_matrix.shape[0]):
        start = sparse_matrix.indptr[index]
        end = sparse_matrix.indptr[index + 1]
        token_ids = sparse_matrix.indices[start:end].tolist()
        weights = sparse_matrix.data[start:end].tolist()
        sparse_list.append(normalize_sparse_vector(dict(zip(token_ids, weights))))

    return {"dense": dense_list, "sparse": sparse_list}
