"""任务服务"""

from knowledge.utils.task_utils import (
    add_running_task,
    add_done_task,
    update_task_status,
    get_task_status,
    get_running_task_list,
    get_done_task_list,
)


class TaskService:
    """任务状态管理服务"""

    def mark_node_running(self, task_id: str, node_name: str) -> None:
        add_running_task(task_id, node_name)

    def mark_node_done(self, task_id: str, node_name: str) -> None:
        add_done_task(task_id, node_name)

    def update_task_status(self, task_id: str, status: str) -> None:
        update_task_status(task_id, status)

    def get_task_info(self, task_id: str) -> dict:
        return {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        }
