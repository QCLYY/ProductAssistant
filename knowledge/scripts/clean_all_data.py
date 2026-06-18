"""一键清空所有测试数据（Milvus + Neo4j + MongoDB）

运行前请确认：
python knowledge/scripts/clean_all_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

print("=" * 50)
print("开始清空所有数据...")
print("=" * 50)

# ---- 1. Milvus ----
print("\n[1/3] 清空 Milvus...")
try:
    from pymilvus import MilvusClient
    c = MilvusClient(uri=os.getenv("MILVUS_URL", "http://192.168.10.130:19530"))
    for name in c.list_collections():
        c.drop_collection(name)
        print(f"  ✅ 已删除集合: {name}")
except Exception as e:
    print(f"  ⚠️ Milvus 清理异常: {e}")

# ---- 2. Neo4j ----
print("\n[2/3] 清空 Neo4j...")
try:
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://192.168.10.130:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
    )
    with d.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run("MATCH ()-[r]->() DELETE r")
    d.close()
    print("  ✅ Neo4j 已清空")
except Exception as e:
    print(f"  ⚠️ Neo4j 清理异常: {e}")

# ---- 3. MongoDB ----
print("\n[3/3] 清空 MongoDB 历史消息...")
try:
    from pymongo import MongoClient
    m = MongoClient(os.getenv("MONGO_URL"))
    db = m[os.getenv("MONGO_DB_NAME", "kb001")]
    db["chat_message"].drop()
    print("  ✅ MongoDB chat_message 已删除")
except Exception as e:
    print(f"  ⚠️ MongoDB 清理异常: {e}")

print("\n" + "=" * 50)
print("全部清空完成！")
print("现在可以运行导入流程了：")
print("  python knowledge/processor/import_process/main_graph.py")
print("=" * 50)
