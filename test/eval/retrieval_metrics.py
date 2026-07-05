"""
检索评估指标
实现 Hit@K / MRR / NDCG 三个指标
"""
from typing import List, Dict, Set
import math


def hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Hit@K: 在前 K 个结果中是否命中相关文档

    Args:
        retrieved_ids: 检索到的文档 ID 列表（按相关性排序）
        relevant_ids: 相关文档 ID 集合
        k: 截断位置

    Returns:
        float: 1.0（命中）或 0.0（未命中）
    """
    top_k = retrieved_ids[:k]
    for doc_id in top_k:
        if doc_id in relevant_ids:
            return 1.0
    return 0.0


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名

    Args:
        retrieved_ids: 检索到的文档 ID 列表（按相关性排序）
        relevant_ids: 相关文档 ID 集合

    Returns:
        float: 1/rank（rank 是第一个相关文档的位置）
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain)

    Args:
        retrieved_ids: 检索到的文档 ID 列表（按相关性排序）
        relevant_ids: 相关文档 ID 集合
        k: 截断位置

    Returns:
        float: NDCG 分数
    """
    # 计算 DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 log2(1) = 0

    # 计算 IDCG（理想情况下的 DCG）
    idcg = 0.0
    num_relevant = min(len(relevant_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval(
    queries: List[Dict],
    retrieval_fn,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    评估检索系统

    Args:
        queries: 查询列表，每个查询包含 {"query": str, "relevant_doc_ids": Set[str]}
        retrieval_fn: 检索函数，接收 query 返回 doc_id 列表
        k_values: K 值列表

    Returns:
        Dict: 评估结果
    """
    results = {f"hit@{k}": [] for k in k_values}
    results["mrr"] = []
    for k in k_values:
        results[f"ndcg@{k}"] = []

    for q in queries:
        query = q["query"]
        relevant_ids = q["relevant_doc_ids"]

        # 执行检索
        retrieved_ids = retrieval_fn(query)

        # 计算指标
        results["mrr"].append(mrr(retrieved_ids, relevant_ids))

        for k in k_values:
            results[f"hit@{k}"].append(hit_at_k(retrieved_ids, relevant_ids, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(retrieved_ids, relevant_ids, k))

    # 计算平均值
    avg_results = {}
    for metric, values in results.items():
        avg_results[metric] = sum(values) / len(values) if values else 0.0

    return avg_results


def print_metrics(results: Dict):
    """打印评估结果"""
    print("\n" + "=" * 50)
    print("检索评估结果")
    print("=" * 50)

    for metric, value in results.items():
        print(f"{metric:15s}: {value:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    # 示例用法
    print("检索评估指标模块")
    print("支持指标: Hit@K, MRR, NDCG@K")
