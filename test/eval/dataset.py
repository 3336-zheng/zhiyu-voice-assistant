"""
评估数据集
包含课堂场景的 golden QA 对
"""
from typing import List, Dict, Set


def get_golden_qa() -> List[Dict]:
    """
    获取 golden QA 数据集

    Returns:
        List[Dict]: 每个元素包含:
            - query: 查询文本
            - relevant_doc_ids: 相关文档 ID 集合
            - reference_answer: 参考答案（可选）
    """
    return [
        {
            "query": "什么是 RAG",
            "relevant_doc_ids": {"rag_intro", "rag_architecture"},
            "reference_answer": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的方法，通过从知识库中检索相关信息来增强大模型的生成质量。"
        },
        {
            "query": "向量数据库有哪些",
            "relevant_doc_ids": {"vector_db_comparison", "chroma_intro"},
            "reference_answer": "常见的向量数据库包括 Chroma、Pinecone、Weaviate、Milvus、Qdrant 等。"
        },
        {
            "query": "BM25 算法原理",
            "relevant_doc_ids": {"bm25_algorithm", "bm25_implement"},
            "reference_answer": "BM25 是基于词频和文档频率的检索算法，通过 TF-IDF 的变体来计算文档相关性分数。"
        },
        {
            "query": "Embedding 模型怎么选",
            "relevant_doc_ids": {"embedding_selection", "bge_model"},
            "reference_answer": "选择 Embedding 模型需要考虑维度、性能、语言支持等因素，BGE 系列是中文场景的常用选择。"
        },
        {
            "query": "RRF 融合排序",
            "relevant_doc_ids": {"rrf_algorithm", "hybrid_retrieval"},
            "reference_answer": "RRF (Reciprocal Rank Fusion) 通过倒数排名融合多个检索结果，公式为 score = Σ 1/(k+rank)。"
        },
        {
            "query": "Reranker 怎么用",
            "relevant_doc_ids": {"reranker_usage", "bge_reranker"},
            "reference_answer": "Reranker 用于对检索结果进行精排，BGE-reranker-v2-m3 是常用的中文 reranker 模型。"
        },
        {
            "query": "Agent 是什么",
            "relevant_doc_ids": {"agent_intro", "plan_execute_agent"},
            "reference_answer": "Agent 是能够自主规划和执行任务的 AI 系统，通过工具调用和决策循环来完成复杂任务。"
        },
        {
            "query": "LangGraph 怎么用",
            "relevant_doc_ids": {"langgraph_tutorial", "graph_state_machine"},
            "reference_answer": "LangGraph 是 LangChain 的图状态机扩展，通过定义节点和边来构建有状态的 Agent 工作流。"
        },
        {
            "query": "CRAG 是什么",
            "relevant_doc_ids": {"crag_intro", "crag_implement"},
            "reference_answer": "CRAG (Corrective RAG) 是一种检索后纠错机制，通过评估文档相关性来决定是否需要改写重检。"
        },
        {
            "query": "Query 改写方法",
            "relevant_doc_ids": {"query_rewrite", "hyde_rag_fusion"},
            "reference_answer": "Query 改写包括 HyDE（生成假设答案）和 RAG-Fusion（多视角查询）等方法，用于优化检索召回。"
        },
        {
            "query": "课堂笔记怎么写",
            "relevant_doc_ids": {"note_taking", "lecture_notes"},
            "reference_answer": "课堂笔记应包含知识点提纲、重点概念、课后疑问和复习卡片四个部分。"
        },
        {
            "query": "语音识别准确率",
            "relevant_doc_ids": {"asr_accuracy", "whisper_improve"},
            "reference_answer": "语音识别准确率受噪音、口音、语速等因素影响，可通过微调模型和后处理来提升。"
        },
        {
            "query": "FastAPI 部署",
            "relevant_doc_ids": {"fastapi_deploy", "docker_deploy"},
            "reference_answer": "FastAPI 可以通过 Uvicorn 部署，支持 Docker 容器化，适合构建高性能 API 服务。"
        },
        {
            "query": "SQLite 优缺点",
            "relevant_doc_ids": {"sqlite_comparison", "sqlite_usage"},
            "reference_answer": "SQLite 轻量级、无需服务器，适合嵌入式场景，但并发性能有限，不适合高并发写入。"
        },
        {
            "query": "ChromaDB 怎么用",
            "relevant_doc_ids": {"chroma_tutorial", "chroma_api"},
            "reference_answer": "ChromaDB 是轻量级向量数据库，支持 Python API，可以方便地存储和检索向量数据。"
        },
    ]


def get_evaluation_queries() -> List[Dict]:
    """
    获取评估查询（简短版，用于快速测试）

    Returns:
        List[Dict]: 查询列表
    """
    return get_golden_qa()[:5]


if __name__ == "__main__":
    # 测试数据集
    qa = get_golden_qa()
    print(f"Golden QA 数据集: {len(qa)} 条")
    for item in qa[:3]:
        print(f"  查询: {item['query']}")
        print(f"  相关文档: {item['relevant_doc_ids']}")
        print()
