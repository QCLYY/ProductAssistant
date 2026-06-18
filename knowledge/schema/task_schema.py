"""任务状态响应模型"""

from typing import List
from pydantic import BaseModel


class TaskStatusResponse(BaseModel):
    status: str
    done_list: List[str] = []
    running_list: List[str] = []
