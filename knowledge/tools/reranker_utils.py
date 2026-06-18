"""BGE-Reranker 模型工具

交叉编码器精排模型，单例模式加载。
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger("import.reranker")

_reranker: Optional[FlagReranker] = None


def get_reranker_model() -> Optional[FlagReranker]:
    """获取 BGE-Reranker 模型单例。

    Returns:
        FlagReranker 实例，加载失败返回 None。
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    model_path = os.getenv("BGE_RERANKER_LARGE", "")
    device = os.getenv("BGE_RERANKER_DEVICE", "cuda:0")
    use_fp16 = os.getenv("BGE_RERANKER_FP16", "1") == "1"

    if not model_path:
        logger.warning("BGE_RERANKER_LARGE 未配置，重排序将降级")
        return None

    try:
        logger.info(f"加载 Reranker 模型: {model_path}, device={device}, fp16={use_fp16}")
        _reranker = FlagReranker(
            model_name_or_path=model_path,
            device=device,
            use_fp16=use_fp16,
        )
        logger.info("Reranker 模型加载完成")
        return _reranker
    except Exception as e:
        logger.error(f"Reranker 模型加载失败: {e}")
        return None
