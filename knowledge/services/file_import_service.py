"""文件导入服务 — 保存文件 + 后台任务执行 LangGraph 流水线"""

import os
import shutil
import uuid
import zipfile
from pathlib import Path

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.main_graph import graph as import_graph
from knowledge.processor.import_process.state import create_default_state


class FileImportService:
    """管理文件上传缓存和 LangGraph 后台导入任务。"""

    def __init__(self, base_dir: str = "", task_service=None):
        self._base_dir = base_dir or get_local_base_dir()
        self._task_service = task_service

    def process_file_upload(self, file) -> tuple:
        """同步处理：保存上传文件 → 返回 task_id / file_dir / import_file_path。

        支持三种格式：
        - .pdf / .md: 直接保存
        - .zip: 解压到 task 目录，找到其中的 .md 文件作为入口
        """
        task_id = str(uuid.uuid4())
        file_dir = os.path.join(self._base_dir, task_id)
        os.makedirs(file_dir, exist_ok=True)

        original_name = file.filename or "untitled"
        import_file_path = os.path.join(file_dir, original_name)

        # 保存文件
        with open(import_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # zip 解压处理
        if original_name.lower().endswith(".zip"):
            import_file_path = self._extract_zip_and_find_md(import_file_path, file_dir)

        return task_id, file_dir, import_file_path

    @staticmethod
    def _extract_zip_and_find_md(zip_path: str, file_dir: str) -> str:
        """解压 zip 到 file_dir，递归搜索返回第一个 .md 文件的路径。"""
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                extract_path = os.path.normpath(os.path.join(file_dir, member.filename))
                if not extract_path.startswith(os.path.normpath(file_dir) + os.sep) and extract_path != os.path.normpath(file_dir):
                    raise ValueError(f"非法路径: {member.filename}")
            zf.extractall(file_dir)

        # 递归搜索 .md 文件
        md_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(file_dir)
            for f in files
            if f.lower().endswith(".md")
        )
        if not md_files:
            raise ValueError("zip 中未找到 .md 文件")

        return md_files[0]

    def run_upload_file_task(self, task_id: str, file_dir: str, import_file_path: str):
        """后台任务：流式执行 LangGraph 导入流水线。"""
        from knowledge.processor.import_process.base import setup_logging
        setup_logging()
        try:
            if self._task_service:
                self._task_service.update_task_status(task_id, "processing")

            # 构建初始状态
            initial_state = create_default_state(
                task_id=task_id,
                file_dir=file_dir,
                import_file_path=import_file_path,
            )

            # 流式执行
            for event in import_graph.stream(initial_state):
                for node_name, node_state in event.items():
                    print(f"[{task_id}] 完成节点: {node_name}")

            if self._task_service:
                self._task_service.update_task_status(task_id, "completed")

        except Exception as e:
            if self._task_service:
                self._task_service.update_task_status(task_id, "failed")
            print(f"[{task_id}] 导入失败: {e}")
