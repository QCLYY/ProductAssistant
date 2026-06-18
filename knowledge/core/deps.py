"""FastAPI 依赖注入"""

from functools import lru_cache

from knowledge.services.task_service import TaskService
from knowledge.services.file_import_service import FileImportService
from knowledge.core.paths import get_local_base_dir


@lru_cache
def get_task_service() -> TaskService:
    return TaskService()


@lru_cache
def get_file_import_service() -> FileImportService:
    base_dir = get_local_base_dir()
    return FileImportService(base_dir=base_dir, task_service=get_task_service())
