"""MarkDown 图片处理节点 — VLM 描述 + MinIO 上传"""

import re
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_client import MinioClient
from knowledge.utils.vlm_client import VLMClient

# Markdown 格式: ![]](images/xxx.jpg)
IMG_PATTERN_MD = re.compile(r"!\[([^\]]*)\]\((images/[^)]+)\)")
# HTML 格式: <img src="images/xxx.png" style="zoom:50%;" />
IMG_PATTERN_HTML = re.compile(r'(<img\s+[^>]*src="(images/[^"]+)"[^>]*/?>)', re.IGNORECASE)


def _find_all_image_refs(md_content: str):
    """查找所有图片引用，返回 [(raw_match, img_rel_path), ...]。
    raw_match: 原始匹配的完整字符串（用于后续替换）
    img_rel_path: images/xxx.jpg 相对路径
    """
    refs = []
    for m in IMG_PATTERN_MD.finditer(md_content):
        refs.append((m.group(0), m.group(2)))
    for m in IMG_PATTERN_HTML.finditer(md_content):
        refs.append((m.group(1), m.group(2)))
    return refs


class MarkDownImageNode(BaseNode):
    """
    MarkDown 图片处理节点

    1. 读取 md_path 对应的 markdown 文件
    2. 找到其中所有图片引用（支持 MD ![]() 和 HTML <img> 两种格式）
    3. 使用 VLM 解析图片并填入 alt 文本
    4. 将本地图片上传到 MinIO，替换链接为 MinIO URL
    5. 将更新后的 md_content 存储到 state 中
    """

    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "[校验 md 文件路径]")
        md_path = state.get("md_path", "")
        if not md_path:
            self.logger.warning("md_path 为空，跳过图片处理")
            return state

        md_file = Path(md_path)
        if not md_file.exists():
            raise FileProcessingError(f"MD 文件不存在: {md_path}", self.name)

        md_dir = md_file.parent
        md_content = md_file.read_text(encoding="utf-8")
        self.logger.info(f"读取 MD 文件: {md_path}, 长度: {len(md_content)} 字符")

        self.log_step("step2", "[查找图片引用]")
        refs = _find_all_image_refs(md_content)
        if not refs:
            self.logger.info("未找到图片引用（MD/HTML 格式均无），跳过处理")
            state["md_content"] = md_content
            return state

        self.logger.info(f"找到 {len(refs)} 个图片引用")

        self.log_step("step3", "[初始化 VLM 和 MinIO 客户端]")
        vlm = VLMClient()
        minio = MinioClient()

        for idx, (raw_match, img_rel_path) in enumerate(refs, 1):
            img_local_path = md_dir / img_rel_path
            self.log_step("step4", f"[{idx}/{len(refs)}] 处理图片: {img_rel_path}")

            # 图片文件不存在 → 跳过（保留原始引用，不 crash）
            if not img_local_path.exists():
                self.logger.warning(f"图片文件不存在，跳过: {img_local_path}")
                continue

            # 4a. VLM 解析图片
            description = ""
            try:
                description = vlm.describe_image(str(img_local_path))
            except Exception as e:
                self.logger.warning(f"VLM 描述失败: {img_rel_path}, 错误: {e}")

            # 4b. 上传到 MinIO
            try:
                minio_url = minio.upload_file(str(img_local_path))
            except Exception as e:
                self.logger.warning(f"MinIO 上传失败: {img_rel_path}, 错误: {e}")
                continue

            # 4c. 替换 md 中的图片引用（统一转为 MD 格式 ![description](url)）
            new_ref = f"![{description}]({minio_url})" if description else f"![]({minio_url})"
            md_content = md_content.replace(raw_match, new_ref, 1)
            self.logger.info(f"替换: {img_rel_path} -> {minio_url}")

        self.log_step("step5", "[保存验证副本]")
        file_title = state.get("file_title", md_file.stem)
        copy_path = md_dir / f"{file_title}(1).md"
        copy_path.write_text(md_content, encoding="utf-8")
        self.logger.info(f"验证副本已保存到: {copy_path}")

        self.log_step("step6", "[更新 state.md_content]")
        state["md_content"] = md_content
        return state
if __name__ == "__main__":
    setup_logging()
    # 简单测试
    node = MarkDownImageNode()
    test_state = {

        #"md_path": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md",
        "md_path": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.md",
        "md_content": "",
        "file_dir": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir",
        "import_file_path": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf",
        "file_title":"万用表RS-12的使用"
    }
    try:
        result_state = node.process(test_state)
        print("处理完成，更新后的 md_content:")
        print(result_state.get("md_content", ""))
        print("处理后的状态:", result_state)
    except Exception as e:
        print(f"处理失败: {e}")
