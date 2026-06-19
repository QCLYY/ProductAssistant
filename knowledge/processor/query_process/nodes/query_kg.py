"""知识图谱查询节点

从用户查询中抽取实体，经 Milvus 对齐后在 Neo4j 中检索相关子图，
最终回填切片文本内容。
"""

import os
import json
from typing import List, Dict, Any, Set, Tuple, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.prompt import ENTITY_EXTRACT_SYSTEM_PROMPT

NodeRef = Dict[str, str]
Triple = Dict[str, str]


class QueryKgNode(BaseNode):
    """知识图谱查询节点。

    流程: 预处理 → LLM抽取实体 → Milvus对齐 → Neo4j种子查找
         → 一跳扩展 → 切片关联 → Milvus文本回填 → 三元组转文本
    """

    name = "query_kg"

    # 权重
    W_SEED = 2.0
    W_NEIGHBOR = 1.0

    def process(self, state: QueryGraphState) -> QueryGraphState:
        if not state.get("use_local_search", True):
            self.log_step("skip", "本地资料检索未启用，跳过知识关联查询")
            return {
                "kg_chunks": [],
                "kg_triples": [],
                "kg_seed_nodes": [],
                "kg_entities": [],
                "kg_aligned_entities": [],
                "kg_alignments": [],
            }

        from knowledge.tools.llm_utils import get_llm_client

        # 1. 预处理
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = self._clean_item_names(state.get("item_names"))
        for name in item_names:
            question = question.replace(name, "")

        if not question.strip():
            self.logger.warning("问题为空，跳过KG查询")
            return {}

        # 2. LLM 实体抽取
        self.log_step("step_1", "LLM 抽取实体")
        entities = self._extract_entities(question, get_llm_client(json_mode=True))
        self.logger.info(f"抽取到 {len(entities)} 个实体: {entities}")

        # 3. Milvus 实体对齐
        self.log_step("step_2", "Milvus 对齐实体")
        align_result = self._align_entities(entities, item_names) if entities else {}
        aligned = align_result.get("aligned_entities", entities)
        self.logger.info(f"对齐后实体: {aligned}")

        # 4. Neo4j 种子节点查找
        self.log_step("step_3", "Neo4j 种子节点 + 扩展一跳")
        seed_nodes = self._find_seed_nodes(aligned, item_names)
        self.logger.info(f"种子节点: {len(seed_nodes)}")

        # 5. 一跳扩展
        triples = self._expand_one_hop(seed_nodes)
        self.logger.info(f"三元组: {len(triples)}")

        # 6. 获取关联切片
        kg_chunk_hits = self._get_chunk_refs(seed_nodes, triples)

        # 7. Milvus 文本回填
        self.log_step("step_4", "整理输出")
        kg_chunks = self._fetch_chunk_texts(kg_chunk_hits)

        return {
            "kg_chunks": kg_chunks,
            "kg_triples": self._triples_to_docs(triples),
            "kg_seed_nodes": seed_nodes,
            "kg_entities": entities,
            "kg_aligned_entities": aligned,
            "kg_alignments": align_result.get("alignments", []),
        }

    # ================================================================== #
    #                     1. 预处理                                        #
    # ================================================================== #

    @staticmethod
    def _clean_item_names(item_names: Any) -> List[str]:
        """清理商品名称：兼容 None/str/list，返回去重列表。"""
        if not item_names:
            return []
        if isinstance(item_names, str):
            return [item_names.strip()] if item_names.strip() else []

        seen: Set[str] = set()
        return [
            s for x in item_names
            if (s := str(x).strip()) and s not in seen and not seen.add(s)
        ]

    # ================================================================== #
    #                     2. LLM 实体抽取                                  #
    # ================================================================== #

    def _extract_entities(self, question: str, llm_client) -> List[str]:
        """LLM 抽取实体并解析 JSON。"""
        try:
            resp = llm_client.invoke(
                [SystemMessage(content=ENTITY_EXTRACT_SYSTEM_PROMPT),
                 HumanMessage(content=f"用户问题：{question}")],
            )

            data = json.loads((resp.content or "").strip())

            entities = list({
                e.strip() for e in data.get("entities", []) if e.strip()
            })
            return entities

        except Exception as e:
            self.logger.error(f"实体抽取失败: {e}")
            return []

    # ================================================================== #
    #                     3. Milvus 实体对齐                              #
    # ================================================================== #

    def _align_entities(
        self, entities: List[str], item_names: List[str], top_k: int = 5,
    ) -> Dict[str, Any]:
        """向量检索对齐实体名称。"""
        from knowledge.tools.embedding_utils import generate_hybrid_embeddings
        from knowledge.tools.milvus_utils import (
            get_milvus_client, build_hybrid_search_requests, execute_hybrid_search,
        )

        collection_name = self.config.entity_name_collection or os.getenv(
            "ENTITY_NAME_COLLECTION", "kb_graph_entity_names_v2"
        )
        client = get_milvus_client()
        if not client:
            return {"aligned_entities": entities, "alignments": []}

        min_score = (
            self.config.kg_entity_align_min_score
            or float(os.getenv("KG_ENTITY_ALIGN_MIN_SCORE", "0.6"))
        )
        expr = self._build_filter_expr(item_names)

        try:
            emb = generate_hybrid_embeddings(entities)
        except Exception as e:
            self.logger.error(f"Embedding 生成失败: {e}")
            return {"aligned_entities": entities, "alignments": []}

        alignments: List[Dict] = []
        aligned: List[str] = []
        seen: Set[str] = set()

        for idx, entity in enumerate(entities):
            dense = emb["dense"][idx]
            sparse = emb["sparse"][idx]

            try:
                reqs = build_hybrid_search_requests(
                    dense_vector=dense,
                    sparse_vector=sparse,
                    dense_search_params={"metric_type": "COSINE"},
                    sparse_search_params={"metric_type": "IP"},
                    filter_expr=expr,
                    top_k=top_k,
                )

                res = execute_hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    search_requests=reqs,
                    ranker_weights=(0.5, 0.5),
                    output_fields=["entity_name", "item_name"],
                )

                best = self._pick_best_hit(res[0] if res else [], min_score)

                if best:
                    name = best["entity"]["entity_name"]
                    if name not in seen:
                        seen.add(name)
                        aligned.append(name)
                    alignments.append({
                        "original": entity, "aligned": name, "score": best["distance"]
                    })
                else:
                    alignments.append({
                        "original": entity, "aligned": None, "reason": "no_hit"
                    })

            except Exception as e:
                alignments.append({
                    "original": entity, "aligned": None, "reason": f"error:{e}"
                })

        return {"aligned_entities": aligned, "alignments": alignments}

    @staticmethod
    def _pick_best_hit(hits: List[Dict], min_score: float) -> Optional[Dict]:
        """从命中列表中选分数最高且超过阈值的。"""
        if not hits:
            return None
        best = max(hits, key=lambda h: h.get("distance", 0))
        if best.get("distance", 0) >= min_score:
            return best
        return None

    # ================================================================== #
    #                     Neo4j 连接                                       #
    # ================================================================== #

    def _neo4j_session(self):
        """获取 Neo4j session（上下文管理器）。"""
        from contextlib import contextmanager
        from knowledge.tools.neo4j_utils import get_neo4j_driver

        @contextmanager
        def _session():
            driver = get_neo4j_driver()
            with driver.session(database=self.config.neo4j_database) as s:
                yield s

        return _session()

    # ================================================================== #
    #                     4. Neo4j 种子节点查找                            #
    # ================================================================== #

    def _find_seed_nodes(
        self, entities: List[str], item_names: List[str],
        per_entity: Optional[int] = None, max_total: Optional[int] = None,
    ) -> List[NodeRef]:
        """在 Neo4j 中查找种子节点（精确 → 模糊）。"""
        if not entities or not item_names:
            return []

        per_entity = per_entity or self.config.kg_max_seed_candidates
        max_total = max_total or self.config.kg_max_total_seeds

        try:
            with self._neo4j_session() as session:
                seeds: List[NodeRef] = []
                seen: Set[Tuple[str, str]] = set()

                for name in entities:
                    rows = session.execute_read(
                        self._tx_find_seeds, name, item_names, per_entity
                    )

                    for s in rows:
                        key = (s["item_name"], s["name"])
                        if key not in seen:
                            seen.add(key)
                            seeds.append(s)

                        if len(seeds) >= max_total:
                            return seeds

                    if len(seeds) >= max_total:
                        return seeds

                return seeds

        except Exception as e:
            self.logger.error(f"Neo4j 种子查询异常: {e}")
            return []

    @staticmethod
    def _tx_find_seeds(tx, name: str, item_names: List[str], limit: int):
        """事务: 精确匹配 → 模糊匹配。"""
        seeds = tx.run("""
            MATCH (n:Entity)
            WHERE n.name = $name AND n.item_name IN $item_names
            RETURN n.name AS name, n.item_name AS item_name
            LIMIT $limit
        """, name=name, item_names=item_names, limit=limit).data()

        if seeds:
            return seeds

        return tx.run("""
            MATCH (n:Entity)
            WHERE n.name IS NOT NULL
              AND toLower(n.name) CONTAINS toLower($name)
              AND n.item_name IN $item_names
            RETURN n.name AS name, n.item_name AS item_name
            LIMIT $limit
        """, name=name, item_names=item_names, limit=limit).data()

    # ================================================================== #
    #                     5. 一跳扩展                                      #
    # ================================================================== #

    def _expand_one_hop(
        self, seed_nodes: List[NodeRef],
        per_seed: Optional[int] = None, max_total: Optional[int] = None,
    ) -> List[Triple]:
        """扩展种子节点的一跳关系。"""
        if not seed_nodes:
            return []

        per_seed = per_seed or self.config.kg_max_triples_per_seed
        max_total = max_total or self.config.kg_max_total_triples

        try:
            with self._neo4j_session() as session:
                triples: List[Triple] = []
                seen: Set[Tuple[str, ...]] = set()

                for s in seed_nodes:
                    rows = session.execute_read(
                        self._tx_expand_triples,
                        s["name"], s["item_name"], per_seed
                    )

                    for tr in rows:
                        key = (tr["item_name"], tr["head"], tr["rel"], tr["tail"])
                        if key not in seen:
                            seen.add(key)
                            triples.append(tr)

                        if len(triples) >= max_total:
                            return triples

                    if len(triples) >= max_total:
                        return triples

                return triples

        except Exception as e:
            self.logger.error(f"Neo4j 扩展异常: {e}")
            return []

    @staticmethod
    def _tx_expand_triples(tx, seed_name: str, item_name: str, limit: int):
        """事务: 双向一跳扩展。"""
        rows = tx.run("""
            MATCH (seed:Entity {name: $seed, item_name: $item_name})
            CALL (seed) {
              MATCH (seed)-[r]->(nbr:Entity)
              WHERE type(r) <> 'MENTIONED_IN'
                AND nbr.item_name = $item_name
              RETURN seed.name AS head, type(r) AS rel, nbr.name AS tail

              UNION

              MATCH (nbr:Entity)-[r]->(seed)
              WHERE type(r) <> 'MENTIONED_IN'
                AND nbr.item_name = $item_name
              RETURN nbr.name AS head, type(r) AS rel, seed.name AS tail
            }
            RETURN head, rel, tail LIMIT $limit
        """, seed=seed_name, item_name=item_name, limit=limit).data()

        return [
            {"head": r["head"], "rel": r["rel"],
             "tail": r["tail"], "item_name": item_name}
            for r in rows
        ]

    # ================================================================== #
    #                     6. 获取关联切片                                  #
    # ================================================================== #

    def _get_chunk_refs(
        self, seed_nodes: List[NodeRef], triples: List[Triple],
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """通过 MENTIONED_IN 关系获取关联切片 ID（带权重排序）。"""
        nodes = self._build_weighted_nodes(seed_nodes, triples)
        if not nodes:
            return []

        max_chunks = max_chunks or self.config.kg_max_total_chunks

        try:
            with self._neo4j_session() as session:
                rows = session.run("""
                    UNWIND $nodes AS n
                    MATCH (e:Entity {name: n.name, item_name: n.item_name})
                          -[:MENTIONED_IN]->(c:Chunk {item_name: n.item_name})
                    WITH c, sum(n.w) AS score, count(DISTINCT e) AS cnt
                    RETURN c.id AS chunk_id, c.item_name AS item_name,
                           score, cnt
                    ORDER BY score DESC, cnt DESC, chunk_id ASC
                    LIMIT $limit
                """, nodes=nodes, limit=max_chunks).data()

            return [
                {
                    "id": None,
                    "distance": float(r.get("score", 0)),
                    "entity": {
                        "chunk_id": str(r["chunk_id"]),
                        "item_name": str(r["item_name"])
                    }
                }
                for r in rows
            ]

        except Exception as e:
            self.logger.error(f"切片引用查询异常: {e}")
            return []

    @staticmethod
    def _build_weighted_nodes(
        seed_nodes: List[NodeRef], triples: List[Triple],
        w_seed: float = 2.0, w_neighbor: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """构建带权重的节点列表（种子权重 > 邻居权重）。"""
        weights: Dict[Tuple[str, str], float] = {}

        for s in seed_nodes or []:
            key = (s["item_name"], s["name"])
            weights[key] = max(weights.get(key, 0), w_seed)

        for tr in triples or []:
            it = tr["item_name"]
            for n in (tr["head"], tr["tail"]):
                key = (it, n)
                weights[key] = max(weights.get(key, 0), w_neighbor)

        return [
            {"item_name": it, "name": n, "w": w}
            for (it, n), w in weights.items()
        ]

    # ================================================================== #
    #                     7. Milvus 文本回填                              #
    # ================================================================== #

    def _fetch_chunk_texts(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从 Milvus 回填切片文本。"""
        from knowledge.tools.milvus_utils import fetch_chunks_by_ids, get_milvus_client

        if not hits:
            return []

        collection_name = self.config.chunks_collection or os.getenv(
            "CHUNKS_COLLECTION", "kb_chunks_v2"
        )

        chunk_ids = list({
            int(str(h["entity"]["chunk_id"]))
            for h in hits
            if h.get("entity", {}).get("chunk_id") is not None
        })

        if not chunk_ids:
            return []

        try:
            rows = fetch_chunks_by_ids(
                client=get_milvus_client(),
                collection_name=collection_name,
                chunk_ids=chunk_ids,
                output_fields=["chunk_id", "content", "title", "file_title", "item_name"],
            )
        except Exception as e:
            self.logger.error(f"Milvus 回填异常: {e}")
            rows = []

        row_map = {
            str(r["chunk_id"]): r
            for r in (rows or [])
            if r.get("chunk_id") is not None
        }

        result = []
        for h in hits:
            ent = h.get("entity", {})
            row = row_map.get(str(ent.get("chunk_id")))
            if row:
                merged = dict(row)
                if ent.get("item_name") and not merged.get("item_name"):
                    merged["item_name"] = ent["item_name"]
                result.append(merged)

        self.logger.info(f"回填完成: {len(result)} 条切片")
        return result

    # ================================================================== #
    #                     8. 三元组转文本                                  #
    # ================================================================== #

    @staticmethod
    def _triples_to_docs(triples: List[Triple]) -> List[str]:
        """三元组 → 去重文本描述。"""
        seen: Set[str] = set()
        docs: List[str] = []

        for tr in triples:
            h, r, t = tr.get("head", ""), tr.get("rel", ""), tr.get("tail", "")
            if not all([h, r, t]):
                continue

            it = tr.get("item_name", "")
            doc = f"[{it}] {h} -({r})-> {t}" if it else f"{h} -({r})-> {t}"

            if doc not in seen:
                seen.add(doc)
                docs.append(doc)

        return docs

    # ================================================================== #
    #                    过滤表达式构建                                    #
    # ================================================================== #

    @staticmethod
    def _build_filter_expr(item_names: Optional[List[str]]) -> Optional[str]:
        if not item_names:
            return None
        quoted = ", ".join(f'"{v}"' for v in item_names)
        return f"item_name in [{quoted}]"


# ================================================================== #
#                        兼容入口                                      #
# ================================================================== #

_node_instance = QueryKgNode()


def node_query_kg(state: QueryGraphState) -> QueryGraphState:
    return _node_instance(state)


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == "__main__":
    import uuid
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("知识图谱查询节点测试")
    print("=" * 60)

    test_state = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        #"rewritten_query": "华为MateBook B5-440电脑如何打开护眼模式？",
        "rewritten_query": "华为MateBook B5-440电脑怎么连接到电视、显示器或投影仪?",
        #"original_query": "你们的华为电脑MateBook B5-440怎么打开护眼模式？",
        "original_query": "华为MateBook B5-440电脑怎么连接到电视、显示器或投影仪?",
        "item_names": ["华为MateBook B5-440笔记本电脑"],
    }

    print("\n【输入状态】:")
    print(f"  rewritten_query: {test_state['rewritten_query']}")
    print(f"  item_names: {test_state['item_names']}")
    print("-" * 60)

    try:
        result = node_query_kg(test_state)

        print("\n[第1步] LLM 原始抽取实体 (kg_entities):")
        print(f"   {result.get('kg_entities', [])}")

        print("\n[第2步] Milvus 对齐后实体 (kg_aligned_entities):")
        print(f"   {result.get('kg_aligned_entities', [])}")

        print("\n[第3步] Neo4j 命中的种子节点 (kg_seed_nodes):")
        for seed in result.get("kg_seed_nodes", []):
            print(f"   - {seed}")

        triples = result.get("kg_triples", [])
        print(f"\n[第4步] 扩展的一跳知识三元组 (共 {len(triples)} 条):")
        for t in triples[:5]:
            print(f"   - {t}")
        if len(triples) > 5:
            print(f"   - ... (省略其余 {len(triples) - 5} 条)")

        chunks = result.get("kg_chunks", [])
        print(f"\n[第5步] 最终召回的切片 (共 {len(chunks)} 条):")
        for i, chunk in enumerate(chunks[:10], 1):
            print(f"   [{i}] ID: {chunk.get('chunk_id')}")
            print(f"       商品: {chunk.get('item_name')}")
            content = chunk.get("content", "")
            print(f"       内容: {content}")
            print()

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()
