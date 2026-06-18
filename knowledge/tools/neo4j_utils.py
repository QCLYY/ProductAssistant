"""Neo4j 驱动单例工具"""

import logging
from typing import Optional
from neo4j import GraphDatabase, Driver
from knowledge.processor.import_process.config import get_config

logger = logging.getLogger("import.neo4j")

_driver: Optional[Driver] = None


def get_neo4j_driver(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Driver:
    """获取 Neo4j 驱动单例，首次调用时创建连接池。

    Args:
        uri: Bolt 连接地址，默认取 config.neo4j_uri
        username: 用户名，默认取 config.neo4j_username
        password: 密码，默认取 config.neo4j_password

    Returns:
        Neo4j Driver 实例
    """
    global _driver
    if _driver is not None:
        return _driver

    config = get_config()
    uri = uri or config.neo4j_uri
    username = username or config.neo4j_username
    password = password or config.neo4j_password

    if not uri:
        raise ValueError("neo4j_uri 未配置，请在 .env 中设置 NEO4J_URI")

    logger.info(f"连接 Neo4j: {uri}")
    _driver = GraphDatabase.driver(uri, auth=(username, password))
    _driver.verify_connectivity()
    logger.info("Neo4j 连接成功")
    return _driver
