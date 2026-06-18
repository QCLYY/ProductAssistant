"""任务追踪工具 — 内存字典管理 running / done / status"""

from typing import Dict, List
from collections import defaultdict

# 正在运行的节点列表（key: task_id, value: [node_name, ...]）
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
# 已完成的节点列表
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
# 任务结果预留
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)
# 任务总体状态
_tasks_status: Dict[str, str] = {}

TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名称 → 中文展示名映射
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "上传文件",
    "entry": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "md_img_node": "Markdown图片处理",
    "document_split": "文档切分",
    "item_name_recognition": "主体名称识别",
    "bge_embedding": "向量生成",
    "import_milvus": "导入向量数据库",
    "knowledge_graph": "导入知识图谱",
    "__end__": "处理完成",
    # 查询流程节点
    "item_name_confirm": "确认问题产品",
    "multi_search": "多路搜索分发",
    "search_embedding": "向量检索",
    "search_embedding_hyde": "HyDE增强检索",
    "query_kg": "知识图谱查询",
    "web_search_mcp": "网络搜索",
    "join": "结果汇合",
    "rrf": "RRF融合排序",
    "rerank": "重排序精排",
    "answer_output": "生成答案",
}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)


def get_running_task_list(task_id: str) -> List[str]:
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def get_done_task_list(task_id: str) -> List[str]:
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str) -> None:
    _tasks_status[task_id] = status_name


def set_task_result(task_id: str, key: str, value: str) -> None:
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    return _tasks_result.get(task_id, {}).get(key, default)


def clear_task_progress(task_id: str) -> None:
    """只清除进度（running/done/result），不清除 status。"""
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_result.pop(task_id, None)


def clear_task(task_id: str) -> None:
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
