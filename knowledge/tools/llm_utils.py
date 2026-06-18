"""LLM 客户端工具 — LangChain ChatOpenAI 封装"""

from typing import Optional
from langchain_openai import ChatOpenAI
from knowledge.processor.import_process.config import get_config

_llm_clients: dict = {}


def get_llm_client(
    model: Optional[str] = None,
    temperature: float = 0.1,
    json_mode: bool = False,
) -> ChatOpenAI:
    """
    获取 LLM 客户端单例（按 model 缓存，避免重复创建）。

    Args:
        model: 模型名称，默认使用 config.default_model
        temperature: 温度参数
        json_mode: 是否使用 JSON 模式

    Returns:
        ChatOpenAI 实例
    """
    config = get_config()
    key = f"{model}_{temperature}_{json_mode}"

    if key not in _llm_clients:
        kwargs = dict(
            model=model or config.default_model or "qwen-flash",
            temperature=temperature,
            api_key=config.openai_api_key or "not-needed",
            base_url=config.openai_api_base,
        )
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        _llm_clients[key] = ChatOpenAI(**kwargs)

    return _llm_clients[key]
