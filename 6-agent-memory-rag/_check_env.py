# -*- coding: utf-8 -*-
"""环境自检：验证 .env 中的 Qdrant / Neo4j / Embedding 是否可用"""
import os, sys, time
from dotenv import load_dotenv

load_dotenv()

COLL = os.getenv("QDRANT_COLLECTION", "hello_agents_vectors")

print("=" * 60)
print("1. Qdrant")
print("=" * 60)
from qdrant_client import QdrantClient
q = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY") or None,
                 timeout=int(os.getenv("QDRANT_TIMEOUT", "30")))
print(f"   URL        : {os.getenv('QDRANT_URL')}")
print(f"   服务端版本   : {q.info().version}")
names = [c.name for c in q.get_collections().collections]
print(f"   现有集合     : {names or '(空，首次运行示例时自动创建)'}")
stored_size = None
if COLL in names:
    info = q.get_collection(COLL)
    stored_size = info.config.params.vectors.size
    print(f"   集合 {COLL}: size={stored_size} points={info.points_count}")

print()
print("=" * 60)
print("2. Neo4j")
print("=" * 60)
from neo4j import GraphDatabase
uri = os.getenv("NEO4J_URI")
drv = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
                           connection_acquisition_timeout=10)
t0 = time.time()
with drv.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
    s.run("RETURN 1 AS ok").single()
    ver = s.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
print(f"   URI        : {uri}")
print(f"   用户        : {os.getenv('NEO4J_USERNAME')}")
print(f"   连接结果     : OK  ({time.time()-t0:.2f}s)")
print(f"   服务端版本   : {ver}")
drv.close()

print()
print("=" * 60)
print("3. Embedding")
print("=" * 60)
print(f"   EMBED_MODEL_TYPE : {os.getenv('EMBED_MODEL_TYPE')}")
print(f"   HF_ENDPOINT      : {os.getenv('HF_ENDPOINT') or '(未设置)'}")
from hello_agents.memory.embedding import get_text_embedder
t0 = time.time()
emb = get_text_embedder()
print(f"   实际实现          : {type(emb).__name__}")
print(f"   实际维度          : {emb.dimension}")
v = emb.encode("道岔健康度分析")
print(f"   编码测试          : OK, len={len(v)}  ({time.time()-t0:.2f}s)")

print()
print("=" * 60)
print("4. 维度一致性")
print("=" * 60)
if stored_size is None:
    print(f"   ✅ Qdrant 中无同名集合，首次运行时将按 {emb.dimension} 维自动创建")
elif stored_size != emb.dimension:
    print(f"   ❌ 不匹配：已有集合 {stored_size} 维，当前 embedding {emb.dimension} 维")
    print(f"      需删除后重建： curl -X DELETE {os.getenv('QDRANT_URL')}/collections/{COLL}")
else:
    print(f"   ✅ 匹配：{stored_size} 维")
