"""
RAG 评估脚本
评估检索和生成质量
"""
import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from test.eval.retrieval_metrics import evaluate_retrieval, print_metrics
from test.eval.dataset import get_golden_qa, get_evaluation_queries

logger = logging.getLogger(__name__)


def mock_retrieval_fn(query: str) -> List[str]:
    """
    Mock 检索函数（用于测试）
    实际使用时替换为真实的检索函数

    Args:
        query: 查询文本

    Returns:
        List[str]: 检索到的文档 ID 列表
    """
    # 简单的关键词匹配 mock
    keyword_map = {
        "RAG": ["rag_intro", "rag_architecture"],
        "向量": ["vector_db_comparison", "chroma_intro"],
        "BM25": ["bm25_algorithm", "bm25_implement"],
        "Embedding": ["embedding_selection", "bge_model"],
        "RRF": ["rrf_algorithm", "hybrid_retrieval"],
        "Reranker": ["reranker_usage", "bge_reranker"],
        "Agent": ["agent_intro", "plan_execute_agent"],
        "LangGraph": ["langgraph_tutorial", "graph_state_machine"],
        "CRAG": ["crag_intro", "crag_implement"],
        "Query": ["query_rewrite", "hyde_rag_fusion"],
        "笔记": ["note_taking", "lecture_notes"],
        "语音": ["asr_accuracy", "whisper_improve"],
        "FastAPI": ["fastapi_deploy", "docker_deploy"],
        "SQLite": ["sqlite_comparison", "sqlite_usage"],
        "Chroma": ["chroma_tutorial", "chroma_api"],
    }

    results = []
    for keyword, doc_ids in keyword_map.items():
        if keyword.lower() in query.lower():
            results.extend(doc_ids)

    # 去重
    return list(dict.fromkeys(results))


def run_evaluation(use_mock: bool = True):
    """
    运行评估

    Args:
        use_mock: 是否使用 mock 检索函数
    """
    print("=" * 60)
    print("智语 RAG 评估系统")
    print("=" * 60)

    # 获取评估数据
    qa_data = get_golden_qa()
    print(f"评估数据集: {len(qa_data)} 条查询")

    # 选择检索函数
    if use_mock:
        retrieval_fn = mock_retrieval_fn
        print("使用 Mock 检索函数")
    else:
        # TODO: 接入真实检索函数
        print("警告: 真实检索函数未实现，使用 Mock")
        retrieval_fn = mock_retrieval_fn

    # 运行评估
    print("\n开始评估...")
    results = evaluate_retrieval(
        queries=qa_data,
        retrieval_fn=retrieval_fn,
        k_values=[1, 3, 5, 10]
    )

    # 打印结果
    print_metrics(results)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"test/eval/eval_results_{timestamp}.json"

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "dataset_size": len(qa_data),
            "use_mock": use_mock,
            "metrics": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存到: {result_file}")

    return results


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)

    # 运行评估
    run_evaluation(use_mock=True)
