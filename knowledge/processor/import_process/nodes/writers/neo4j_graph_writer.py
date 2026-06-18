"""
Neo4jGraphWriter — 图数据写入 Neo4j（双标签方案）

负责将清洗后的实体与关系写入 Neo4j 图数据库。

双标签方案：
- Entity 作为所有实体的收敛锚点（保证 MERGE 寻址一致）
- Device/Step/Warning 等作为原生语义标签（自带索引，查询快）

对外暴露两个方法：
- clear(): 导入前清理该 item_name 下的旧节点和关系（幂等）
- insert(): 单事务批量写入 Chunk + Entity + Relation（原子性保障）
"""

import logging
from typing import Dict, List

from knowledge.processor.import_process.exceptions import Neo4jError


class Neo4jGraphWriter:
    """负责将实体与关系写入 Neo4j。"""

    def __init__(self, database: str = ""):
        self.database = database
        self.logger = logging.getLogger(self.__class__.__name__)

    # ================================================================== #
    #                      Cypher 模板（双标签）                            #
    # ================================================================== #

    CYPHER_CLEAR_ITEM = (
        "MATCH (n {item_name: $item_name}) DETACH DELETE n"
    )

    CYPHER_MERGE_CHUNK = (
        "MERGE (c:Chunk {id: $chunk_id, item_name: $item_name})"
    )

    CYPHER_MERGE_ENTITY_TEMPLATE = (
        "MERGE (n:Entity {{name: $name, item_name: $item_name}}) "
        "ON CREATE SET "
        "    n.source_chunk_id = $chunk_id, "
        "    n.description     = $description "
        "ON MATCH SET "
        "    n.description = CASE "
        "        WHEN $description <> \"\" THEN $description "
        "        ELSE coalesce(n.description, \"\") "
        "    END "
        "SET n:`{label}`"
    )

    CYPHER_LINK_ENTITY_TO_CHUNK = (
        "MATCH (n:Entity {name: $name, item_name: $item_name}) "
        "MATCH (c:Chunk  {id: $chunk_id, item_name: $item_name}) "
        "MERGE (n)-[:MENTIONED_IN]->(c)"
    )

    CYPHER_MERGE_RELATION_TEMPLATE = (
        "MATCH (h:Entity {{name: $head, item_name: $item_name}}) "
        "MATCH (t:Entity {{name: $tail, item_name: $item_name}}) "
        "MERGE (h)-[:{rel_type}]->(t)"
    )

    # ================================================================== #
    #                          公开方法                                    #
    # ================================================================== #

    def clear(self, driver, item_name: str) -> None:
        """导入前清理该 item_name 下的所有旧节点和关系（幂等）。"""
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    lambda tx, name: tx.run(self.CYPHER_CLEAR_ITEM, item_name=name),
                    item_name,
                )
            self.logger.info(f"Neo4j 旧数据已清理: {item_name}")
        except Exception as e:
            raise Neo4jError(f"Neo4j 清理失败: {e}")

    def insert(
        self,
        driver,
        entities: List[Dict],
        relations: List[Dict],
        chunk_id: str,
        item_name: str,
    ) -> None:
        """单事务批量写入：Chunk + Entity + Relation，原子性保障。"""
        if not entities:
            raise ValueError("参数校验失败，实体列表为空")
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    self._write_graph_tx, entities, relations, chunk_id, item_name,
                )
            self.logger.info(
                f"Neo4j 写入: {len(entities)} 实体, {len(relations)} 关系"
            )
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}")

    # ================================================================== #
    #                          私有方法                                    #
    # ================================================================== #

    def _session(self, driver):
        """获取 Neo4j 会话上下文管理器。"""
        return driver.session(database=self.database) if self.database else driver.session()

    def _write_graph_tx(
        self,
        tx,
        entities: List[Dict],
        relations: List[Dict],
        chunk_id: str,
        item_name: str,
    ):
        """事务内写入逻辑（由 execute_write 调用）。"""
        # 1. Chunk 节点
        tx.run(self.CYPHER_MERGE_CHUNK, chunk_id=chunk_id, item_name=item_name)

        # 2. Entity 节点（双标签）+ 关联到 Chunk
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            raw_label = str(entity.get("label", "")).strip()
            description = str(entity.get("description", "")).strip()

            cypher_entity = self.CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)
            tx.run(cypher_entity, name=name, description=description,
                   chunk_id=chunk_id, item_name=item_name)

            tx.run(self.CYPHER_LINK_ENTITY_TO_CHUNK,
                   name=name, chunk_id=chunk_id, item_name=item_name)

        # 3. 实体间关系
        for rel in relations:
            head = str(rel.get("head", "")).strip()
            tail = str(rel.get("tail", "")).strip()
            if not head or not tail:
                continue
            rel_type = str(rel.get("type", "RELATED_TO")).strip() or "RELATED_TO"

            cypher_rel = self.CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher_rel, head=head, tail=tail, item_name=item_name)
