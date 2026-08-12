"""通用 RAG 检索评测命令行入口。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from test.eval.dataset import (
    DatasetValidationError,
    EvaluationDataset,
    load_evaluation_dataset,
)
from test.eval.real_retriever import EvaluationRetriever
from test.eval.retrieval_metrics import (
    aggregate_by_category,
    aggregate_metrics,
    evaluate_query,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_METHODS = EvaluationRetriever.SUPPORTED_METHODS
DEFAULT_K_VALUES = (1, 3, 5)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """使用最近秩计算延迟百分位，避免额外统计依赖。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def _retrieval_snapshot(top_k: int, k_values: Sequence[int]) -> dict[str, object]:
    return {
        "bm25_top_k": settings.bm25_top_k,
        "embedding_top_k": settings.embedding_top_k,
        "rrf_k": settings.rrf_k,
        "rrf_top_k": settings.rrf_top_k,
        "evaluation_top_k": top_k,
        "metric_k_values": list(k_values),
        "rag_final_top_k": settings.rag_final_top_k,
        "context_token_budget": settings.rag_context_token_budget,
        "parent_chunk_chars": settings.rag_parent_chunk_chars,
        "parent_chunk_overlap_chars": settings.rag_parent_chunk_overlap_chars,
        "child_chunk_chars": settings.rag_child_chunk_chars,
        "child_chunk_overlap_chars": settings.rag_child_chunk_overlap_chars,
    }


def _evidence_metrics(
    dataset: EvaluationDataset,
    retrieved_ids: Sequence[str],
    evidence: Sequence[dict[str, Any]],
    k_values: Sequence[int],
) -> dict[str, float]:
    """按原文字符区间计算证据命中与覆盖，独立于 relevance 标签。"""
    documents = {document.doc_id: document for document in dataset.documents}
    valid_evidence = [
        item
        for item in evidence
        if isinstance(item.get("source_id"), str)
        and isinstance(item.get("start_char"), int)
        and isinstance(item.get("end_char"), int)
        and item["start_char"] < item["end_char"]
    ]
    metrics = {}
    for k in sorted(set(k_values)):
        retrieved = [documents[item] for item in retrieved_ids[:k] if item in documents]
        recalled = 0
        covered_chars = 0
        total_chars = 0
        for item in valid_evidence:
            start, end = item["start_char"], item["end_char"]
            total_chars += end - start
            intervals = []
            for document in retrieved:
                metadata = document.metadata
                if metadata.get("source_path") != item["source_id"]:
                    continue
                chunk_start = metadata.get("source_start_char")
                chunk_end = metadata.get("source_end_char")
                if not isinstance(chunk_start, int) or not isinstance(chunk_end, int):
                    continue
                left, right = max(start, chunk_start), min(end, chunk_end)
                if left < right:
                    intervals.append((left, right))
            if intervals:
                recalled += 1
                merged = []
                for left, right in sorted(intervals):
                    if not merged or left > merged[-1][1]:
                        merged.append([left, right])
                    else:
                        merged[-1][1] = max(merged[-1][1], right)
                covered_chars += sum(right - left for left, right in merged)
        metrics[f"evidence_recall@{k}"] = (
            recalled / len(valid_evidence) if valid_evidence else 0.0
        )
        metrics[f"evidence_coverage@{k}"] = (
            covered_chars / total_chars if total_chars else 0.0
        )
    return metrics


def evaluate_method(
    retriever: EvaluationRetriever,
    dataset: EvaluationDataset,
    method: str,
    *,
    top_k: int,
    k_values: Sequence[int],
    unavailable_error: str | None = None,
) -> dict[str, Any]:
    """运行一种检索方法并保留逐查询证据。"""
    query_results = []
    latencies = []
    failures = 0

    answerable_queries = [
        golden for golden in dataset.queries if golden.expected_action == "answer"
    ]
    if not answerable_queries:
        raise ValueError("检索评测至少需要一条 expected_action=answer 的 Question")

    for golden in answerable_queries:
        started = time.perf_counter()
        error = None
        try:
            if unavailable_error:
                raise RuntimeError(unavailable_error)
            retrieved_ids = retriever.search(golden.query, method, top_k)
        except Exception as exc:
            retrieved_ids = []
            error = f"{type(exc).__name__}: {exc}"
            failures += 1
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        metrics = evaluate_query(
            retrieved_ids,
            golden.relevance,
            k_values,
        )
        metrics.update(
            _evidence_metrics(dataset, retrieved_ids, golden.evidence, k_values)
        )
        query_results.append(
            {
                "query_id": golden.query_id,
                "query": golden.query,
                "category": golden.category,
                "question_type": golden.question_type,
                "relevant_doc_ids": list(golden.relevance),
                "reference_answer": golden.reference_answer,
                "reference_claims": list(golden.reference_claims),
                "evidence": list(golden.evidence),
                "retrieved_doc_ids": retrieved_ids,
                "metrics": metrics,
                "latency_ms": latency_ms,
                "error": error,
            }
        )

    return {
        "metrics": aggregate_metrics(query_results),
        "categories": aggregate_by_category(query_results),
        "latency_ms": _latency_summary(latencies),
        "failures": failures,
        "failure_rate": failures / len(answerable_queries),
        "evaluated_queries": len(answerable_queries),
        "queries": query_results,
    }


def run_evaluation(
    dataset: EvaluationDataset,
    methods: Iterable[str] = DEFAULT_METHODS,
    *,
    top_k: int = 5,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    retriever: EvaluationRetriever | None = None,
) -> dict[str, Any]:
    """运行完整对照实验并输出可复现报告。"""
    selected_methods = tuple(dict.fromkeys(methods))
    unknown = set(selected_methods) - set(EvaluationRetriever.SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"不支持的方法: {', '.join(sorted(unknown))}")
    if not selected_methods:
        raise ValueError("至少选择一种检索方法")
    if top_k < max(k_values):
        raise ValueError("top_k 不能小于最大的指标 K 值")

    engine = retriever or EvaluationRetriever(dataset.documents)
    requires_embedding = any(method != "bm25" for method in selected_methods)
    setup_started = time.perf_counter()
    setup_error = None
    try:
        engine.prepare(selected_methods)
    except Exception as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    setup_ms = (time.perf_counter() - setup_started) * 1000

    method_reports = {}
    for method in selected_methods:
        method_reports[method] = evaluate_method(
            engine,
            dataset,
            method,
            top_k=top_k,
            k_values=k_values,
            unavailable_error=setup_error if method != "bm25" else None,
        )

    usage_snapshot = getattr(engine, "usage_snapshot", lambda: {})

    return {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset.name,
            "corpus_path": str(dataset.corpus_path),
            "queries_path": str(dataset.queries_path),
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
            "answerable_queries": sum(
                query.expected_action == "answer" for query in dataset.queries
            ),
            "unanswerable_queries": sum(
                query.expected_action != "answer" for query in dataset.queries
            ),
            "categories": sorted({query.category for query in dataset.queries}),
            "label_origins": sorted(
                {
                    str(query.metadata.get("label_origin", "unspecified"))
                    for query in dataset.queries
                }
            ),
            "label_completeness": sorted(
                {
                    str(query.metadata.get("label_completeness", "unspecified"))
                    for query in dataset.queries
                }
            ),
        },
        "configuration": {
            "models": engine.model_snapshot(),
            "retrieval": _retrieval_snapshot(top_k, k_values),
        },
        "index_setup": {
            "status": (
                "not_required"
                if not requires_embedding
                else "failed"
                if setup_error
                else "succeeded"
            ),
            "latency_ms": setup_ms,
            "error": setup_error,
        },
        "model_workload": usage_snapshot(),
        "methods": method_reports,
        "notice": (
            "结论应结合数据集 label_origin 与 label_completeness 解读；"
            "AI 自校验标签不等同于人工标注。"
        ),
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_k_values(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(item) for item in _parse_csv(value))))
    except ValueError:
        raise argparse.ArgumentTypeError("K 值必须是逗号分隔的整数") from None
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("K 值必须全部大于 0")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行通用 RAG 检索评测")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_DATA_DIR / "sample_corpus.jsonl",
        help="JSONL/JSON 语料文件",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_DATA_DIR / "sample_queries.jsonl",
        help="JSONL/JSON Golden Query 文件",
    )
    parser.add_argument("--dataset-name", help="报告中的数据集名称")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="逗号分隔：bm25,embedding,hybrid,hybrid_reranker",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k-values", type=_parse_k_values, default=DEFAULT_K_VALUES)
    parser.add_argument("--output", type=Path, help="JSON 报告输出路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    methods = _parse_csv(args.methods)
    unknown = set(methods) - set(EvaluationRetriever.SUPPORTED_METHODS)
    if unknown:
        parser.error(f"不支持的方法: {', '.join(sorted(unknown))}")
    if args.top_k <= 0:
        parser.error("top-k 必须大于 0")
    if args.top_k < max(args.k_values):
        parser.error("top-k 不能小于最大的指标 K 值")

    try:
        dataset = load_evaluation_dataset(
            args.corpus,
            args.queries,
            name=args.dataset_name,
        )
        report = run_evaluation(
            dataset,
            methods,
            top_k=args.top_k,
            k_values=args.k_values,
        )
    except (DatasetValidationError, ValueError) as exc:
        parser.error(str(exc))

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
