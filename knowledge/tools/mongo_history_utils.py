"""MongoDB 历史对话管理工具"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

from pymongo import MongoClient
from bson import ObjectId

from knowledge.processor.query_process.config import get_config

logger = logging.getLogger("query.mongo_history")

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    """获取 MongoDB 客户端单例。"""
    global _client
    if _client is None:
        config = get_config()
        url = config.mongo_url
        _client = MongoClient(url)
        logger.info(f"MongoDB 已连接: {url}")
    return _client


def _get_collection():
    """获取 chat_message 集合。"""
    config = get_config()
    return _get_client()[config.mongo_db_name]["chat_message"]


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict]:
    """获取指定会话的最近对话记录。

    Args:
        session_id: 会话 ID。
        limit: 返回的最大消息数。

    Returns:
        消息列表，按时间戳升序排列，每条消息的 _id 转为字符串。
    """
    collection = _get_collection()
    cursor = (
        collection.find({"session_id": session_id})
        .sort("timestamp", 1)
        .limit(limit)
    )
    messages = []
    for msg in cursor:
        msg["_id"] = str(msg["_id"])
        messages.append(msg)
    return messages


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: List[str] = None,
    message_id: str = "",
) -> str:
    """保存或更新一条聊天消息。

    Args:
        session_id: 会话 ID。
        role: 消息角色（"user" 或 "assistant"）。
        text: 消息文本。
        rewritten_query: 改写后的查询。
        item_names: 关联的商品名称列表。
        message_id: 如果提供，则更新已有消息；否则插入新消息。

    Returns:
        消息的 _id 字符串。
    """
    collection = _get_collection()
    doc = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names or [],
        "timestamp": datetime.utcnow(),
    }
    if message_id:
        collection.update_one({"_id": ObjectId(message_id)}, {"$set": doc})
        return message_id
    else:
        result = collection.insert_one(doc)
        return str(result.inserted_id)


def update_message_item_names(message_ids: List[str], item_names: List[str]):
    """批量更新消息的商品名称。

    Args:
        message_ids: 要更新的消息 _id 列表。
        item_names: 要设置的商品名称列表。
    """
    if not message_ids:
        return
    collection = _get_collection()
    object_ids = [ObjectId(mid) for mid in message_ids]
    collection.update_many(
        {"_id": {"$in": object_ids}},
        {"$set": {"item_names": item_names}},
    )
    logger.info(f"已回填 {len(message_ids)} 条消息的商品名称")


def clear_history(session_id: str) -> int:
    """清除指定会话的所有历史消息。

    Args:
        session_id: 会话 ID。

    Returns:
        删除的消息数量。
    """
    collection = _get_collection()
    result = collection.delete_many({"session_id": session_id})
    return result.deleted_count
