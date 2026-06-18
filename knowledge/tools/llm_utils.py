"""LLM client helpers."""

import os
from typing import Optional

from langchain_openai import ChatOpenAI

from knowledge.processor.import_process.config import get_config

_llm_clients: dict = {}


def get_llm_client(
    model: Optional[str] = None,
    temperature: float = 0.1,
    json_mode: bool = False,
) -> ChatOpenAI:
    """Return a cached ChatOpenAI-compatible client."""
    config = get_config()
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    key = f"{model}_{temperature}_{json_mode}_{timeout}_{max_retries}"

    if key not in _llm_clients:
        kwargs = dict(
            model=model or config.default_model or "qwen-flash",
            temperature=temperature,
            api_key=config.openai_api_key or "not-needed",
            base_url=config.openai_api_base,
            timeout=timeout,
            max_retries=max_retries,
        )
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        _llm_clients[key] = ChatOpenAI(**kwargs)

    return _llm_clients[key]
