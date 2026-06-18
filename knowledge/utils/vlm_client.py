"""VLM 视觉语言模型客户端"""

import base64
import logging
import time
from pathlib import Path

from openai import OpenAI

from knowledge.processor.import_process.config import get_config

logger = logging.getLogger("import.vlm")


class VLMClient:
    """视觉语言模型客户端，用于描述图片内容"""

    SYSTEM_PROMPT = (
        "你是一个专业的图片描述助手。请用简洁的中文描述图片内容，"
        "包括图片中的物体、文字、布局等信息。描述作为Markdown图片的alt文本，"
        "控制在50字以内。"
    )

    def __init__(self):
        config = get_config()
        self.client = OpenAI(
            base_url=config.openai_api_base,
            api_key=config.openai_api_key or "not-needed",
        )
        self.model = config.vl_model or "Qwen2.5-VL-7B-Instruct"
        self.rpm = config.requests_per_minute
        self._last_call_time = 0.0

    def describe_image(self, image_path: str) -> str:
        """
        使用 VLM 描述图片内容

        Args:
            image_path: 图片文件路径

        Returns:
            图片的中文描述
        """
        self._rate_limit()

        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"图片不存在: {image_path}")
            return ""

        suffix = image_path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        #data:[MIME类型];base64,[Base64数据]
        data_url = f"data:{media_type};base64,{image_data}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": "请描述这张图片的内容。",
                            },
                        ],
                    },
                ],
                max_tokens=200,
            )
            description = response.choices[0].message.content.strip()
            logger.info(f"VLM 描述: {image_path.name} -> {description}")
            return description

        except Exception as e:
            logger.error(f"VLM 描述失败: {image_path.name}, 错误: {e}")
            return ""

    def _rate_limit(self):
        """简单的速率限制"""
        now = time.time()
        min_interval = 60.0 / self.rpm
        elapsed = now - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()
