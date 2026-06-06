# -*- coding: utf-8 -*-
"""CPU 检索延迟测试"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "99"  # 不存在的 GPU ID，强制 fallback 到 CPU

import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
_spec = importlib.util.spec_from_file_location('conftest', 'test/conftest.py')
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_conftest.setup_backend()

import torch
print(f'CUDA available: {torch.cuda.is_available()}')

from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.bm25_service import get_bm25_service
from backend.app.services.chroma_service import get_chroma_service
from backend.app.services.rrf_service import get_rrf_service
from backend.app.services.doc_index_service import get_doc_index_service
from backend.app.core.database import engine, Base, SessionLocal

Base.metadata.create_all(bind=engine)
hybrid = get_hybrid_retrieval_service()
emb_svc = get_embedding_service()
bm25 = get_bm25_service()
chroma = get_chroma_service()
rrf = get_rrf_service()
doc_index = get_doc_index_service()
doc_index.sync_docs()

print(f'Embedding device: {emb_svc.device}')
print(f'Reranker device: {hybrid.reranker_service.device}')

queries = [
    'BM25分词和倒排索引的原理',
    'ChromaDB向量数据库配置和持久化',
    'ReAct框架的思考行动观察循环',
    '如何让大模型减少胡说八道的问题',
    '怎样把很长的文章切分成小段来搜索',
    '多个搜索结果怎么合并排成一个列表',
    'RAG系统的完整开发流程从数据准备到检索生成',
    'Agent开发中遇到的坑和实际经验教训',
    'BGE-M3嵌入模型和BGE-reranker重排序',
    'Plan-and-Execute架构模式的拓扑排序并行执行',
    '语义分块是怎么判断在哪里切开的',
    '对话记忆和会话管理的上下文窗口',
]

# warm-up
print('warm-up...')
for q in queries[:3]:
    qe = emb_svc.encode(q)
    bm25.search(q, top_k=20)
    chroma.search(qe, top_k=20)

# 纯 Embedding
print('\n=== 纯 Embedding ===')
emb_times = []
db = SessionLocal()
for q in queries:
    t0 = time.time()
    hybrid.search_pure_embedding(q, top_k=5, db=db)
    emb_times.append((time.time()-t0)*1000)
db.close()
print(f'  avg: {sum(emb_times)/len(emb_times):.0f}ms')

# Hybrid RRF
print('\n=== Hybrid RRF ===')
rrf_times = []
db = SessionLocal()
for q in queries:
    t0 = time.time()
    qe = emb_svc.encode(q)
    bm25_res = bm25.search(q, top_k=20)
    emb_res = chroma.search(qe, top_k=20)
    rrf.fuse(bm25_res, emb_res, top_k=5)
    rrf_times.append((time.time()-t0)*1000)
db.close()
print(f'  avg: {sum(rrf_times)/len(rrf_times):.0f}ms')

# Hybrid+Reranker
print('\n=== Hybrid+Reranker ===')
rerank_times = []
db = SessionLocal()
for q in queries:
    t0 = time.time()
    hybrid.search_hybrid(q, top_k=5, db=db)
    rerank_times.append((time.time()-t0)*1000)
db.close()
print(f'  avg: {sum(rerank_times)/len(rerank_times):.0f}ms')

print(f'\n=== CPU 汇总 ===')
print(f'纯 Embedding:      {sum(emb_times)/len(emb_times):.0f}ms')
print(f'Hybrid RRF:        {sum(rrf_times)/len(rrf_times):.0f}ms')
print(f'Hybrid+Reranker:   {sum(rerank_times)/len(rerank_times):.0f}ms')
