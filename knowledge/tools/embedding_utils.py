"""BGE-M3 混合嵌入工具"""

import logging
from typing import Optional, List, Dict, Any
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from knowledge.processor.import_process.config import get_config
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector

logger = logging.getLogger("import.embedding")

_bge_m3: Optional[BGEM3EmbeddingFunction] = None


def get_bge_m3_model(device: Optional[str] = None, use_fp16: Optional[bool] = None) -> BGEM3EmbeddingFunction:
    """
    获取 BGE-M3 嵌入模型单例（首次调用加载到 GPU）。

    Args:
        device: 设备，默认使用 config.BGE_DEVICE 或 "cuda:0"
        use_fp16: 是否使用半精度，默认使用 config.BGE_FP16 或 True

    Returns:
        BGEM3EmbeddingFunction 实例
    """
    global _bge_m3
    if _bge_m3 is not None:
        return _bge_m3

    config = get_config()
    device = device or getattr(config, "bge_device", "cuda:0")
    use_fp16 = use_fp16 if use_fp16 is not None else getattr(config, "bge_fp16", True)
    model_path = getattr(config, "bge_m3_path", "BAAI/bge-m3")

    logger.info(f"加载 BGE-M3 模型: {model_path}, device={device}, fp16={use_fp16}")
    _bge_m3 = BGEM3EmbeddingFunction(
        model_name=model_path,
        device=device,
        use_fp16=use_fp16,
    )
    logger.info("BGE-M3 模型加载完成")
    return _bge_m3


def generate_hybrid_embeddings(texts: List[str]) -> Dict[str, List[Any]]:
    """为文本列表生成稠密和稀疏嵌入向量。

    Args:
        texts: 待编码的文本列表。

    Returns:
        {"dense": [...], "sparse": [...]}
        dense 是 list[list[float]]，sparse 是 list[dict] ({int_index: float_value})。
    """
    model = get_bge_m3_model()
    embeddings = model.encode_queries(texts)

    # 稠密向量：encode_queries 返回 list[np.ndarray]；fp16 时需转为 float32
    dense_list = embeddings["dense"]
    if hasattr(dense_list, "tolist"):
        dense_list = [d.astype("float32").tolist() for d in dense_list]
    elif dense_list and hasattr(dense_list[0], "dtype"):
        dense_list = [d.astype("float32").tolist() for d in dense_list]

    # 稀疏向量：用 CSR indptr/indices/data 逐行提取（与 bge_embedding 节点一致）
    sparse_matrix = embeddings["sparse"]
    sparse_list = []
    for j in range(sparse_matrix.shape[0]):
        s = sparse_matrix.indptr[j]
        e = sparse_matrix.indptr[j + 1]
        token_ids = sparse_matrix.indices[s:e].tolist()
        weights = sparse_matrix.data[s:e].tolist()
        sparse_dict = normalize_sparse_vector(dict(zip(token_ids, weights)))
        sparse_list.append(sparse_dict)

    return {"dense": dense_list, "sparse": sparse_list}
