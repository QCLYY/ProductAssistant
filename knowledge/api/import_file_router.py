"""导入文件路由 — /upload 和 /status/{task_id}"""
import uvicorn
from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from knowledge.schema.upload_schema import UploadResponse
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.services.file_import_service import FileImportService
from knowledge.services.task_service import TaskService
from knowledge.core.deps import get_file_import_service, get_task_service
from knowledge.core.paths import get_front_page_dir

import os

router = APIRouter()

# 前端模板
_templates = Jinja2Templates(directory=get_front_page_dir()) if os.path.isdir(get_front_page_dir()) else None


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """前端导入页面"""
    if _templates:
        return _templates.TemplateResponse("import.html", {"request": request})
    # 降级：直接返回 HTML 文件内容
    html_path = os.path.join(get_front_page_dir(), "import.html")
    if os.path.isfile(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>import.html not found</h1>", status_code=404)


@router.post("/upload", response_model=UploadResponse)
async def upload_file_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: FileImportService = Depends(get_file_import_service),
) -> UploadResponse:
    # 1. 同步处理：保存文件
    task_id, file_dir, import_file_path = service.process_file_upload(file)

    # 2. 异步处理：后台线程执行 LangGraph 流水线
    background_tasks.add_task(
        service.run_upload_file_task, task_id, file_dir, import_file_path,
    )

    return UploadResponse(message="File uploaded successfully", task_id=task_id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def status_endpoint(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
) -> TaskStatusResponse:
    task_info = task_service.get_task_info(task_id)
    return TaskStatusResponse(**task_info)

