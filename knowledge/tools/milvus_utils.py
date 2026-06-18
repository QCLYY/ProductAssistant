"""Milvus 客户端工具"""

import logging
from typing import Optional, List, Any, Dict
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from knowledge.processor.import_process.config import get_config

logger = logging.getLogger("import.milvus")

_client: Optional[MilvusClient] = None


def get_milvus_client(url: Optional[str] = None) -> Optional[MilvusClient]:
    """
    获取 Milvus 客户端单例。

    Args:
        url: Milvus 服务地址，默认使用 config.milvus_url

    Returns:
        MilvusClient 实例，连接失败返回 None。
    """
    global _client
    if _client is not None:
        return _client

    config = get_config()
    url = url or config.milvus_url or "http://192.168.10.130:19530"

    try:
        logger.info(f"连接 Milvus: {url}")
        _client = MilvusClient(uri=url)
        return _client
    except Exception as e:
        logger.error(f"连接 Milvus 失败: {e}")
        return None


def build_hybrid_search_requests(
    dense_vector: List[float],
    sparse_vector: Any,
    *,
    dense_search_params: Optional[Dict] = None,
    sparse_search_params: Optional[Dict] = None,
    filter_expr: Optional[str] = None,
    top_k: int = 5,
) -> List[AnnSearchRequest]:
    """构建混合检索请求（稠密 + 稀疏）。

    Args:
        dense_vector: 稠密嵌入向量（list[float]）。
        sparse_vector: 稀疏嵌入向量（{token_id: weight} 字典格式）。
        dense_search_params: 稠密路检索参数，默认 {"metric_type": "IP"}。
        sparse_search_params: 稀疏路检索参数，默认 {"metric_type": "IP"}。
        filter_expr: Milvus 标量过滤表达式，如 'item_name in ["xxx"]'。None 则不过滤。
        top_k: 每路召回数量。

    Returns:
        [dense_req, sparse_req] 混合检索请求列表。
    """
    if dense_search_params is None:
        dense_search_params = {"metric_type": "IP"}
    if sparse_search_params is None:
        sparse_search_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_search_params,
        expr=filter_expr,
        limit=top_k,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_search_params,
        expr=filter_expr,
        limit=top_k,
    )
    return [dense_req, sparse_req]


def execute_hybrid_search(
    client: MilvusClient,
    collection_name: str,
    search_requests: List[AnnSearchRequest],
    ranker_weights: tuple = (0.5, 0.5),
    top_k: int = 5,
    normalize_score: bool = True,
    output_fields: List[str] = None,
) -> List[Any]:
    """执行混合检索。

    Args:
        client: Milvus 客户端。
        collection_name: 集合名称。
        search_requests: AnnSearchRequest 列表。
        ranker_weights: 权重元组，如 (0.5, 0.5)。
        top_k: 最终返回数量。
        normalize_score: 是否归一化分数。
        output_fields: 要返回的标量字段。

    Returns:
        搜索结果列表，每个元素是一组 hits。
    """
    ranker = WeightedRanker(*ranker_weights)
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=search_requests,
        ranker=ranker,
        limit=top_k,
        output_fields=output_fields or [],
    )
    return results


def fetch_chunks_by_ids(
    client: MilvusClient,
    collection_name: str,
    chunk_ids: List[int],
    output_fields: List[str] = None,
) -> List[Dict]:
    """根据切片 ID 批量查询切片内容。

    Args:
        client: Milvus 客户端。
        collection_name: 集合名称。
        chunk_ids: 切片 ID 整数列表。
        output_fields: 要返回的字段列表。

    Returns:
        切片数据列表，每条包含请求的字段。
    """
    if not chunk_ids:
        return []
    if output_fields is None:
        output_fields = ["chunk_id", "content", "item_name"]

    filter_expr = f"chunk_id in {chunk_ids}"
    results = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=output_fields,
        limit=len(chunk_ids),
    )
    logger.info(f"按ID查询完成: collection={collection_name}, 命中={len(results)}")
    return results
