"""RAG 检索评估入口，运行真实算法而不是关键词 Mock。"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.eval.dataset import get_evaluation_corpus, get_golden_qa
from test.eval.real_retriever import EvaluationRetriever
from test.eval.retrieval_metrics import evaluate_retrieval


def evaluate_method(method: str, top_k: int = 10) -> Dict:
    """评估一种真实检索方法，并统计端到端查询延迟。"""
    retriever = EvaluationRetriever(get_evaluation_corpus(), method)
    latencies = []

    def retrieve(query: str):
        started = time.perf_counter()
        result = retriever.search(query, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        return result

    metrics = evaluate_retrieval(
        queries=get_golden_qa(),
        retrieval_fn=retrieve,
        k_values=[1, 3, 5, 10],
    )
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return {
        "metrics": metrics,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p95": ordered[p95_index] if ordered else 0.0,
        },
    }


def run_evaluation(methods: Iterable[str], top_k: int = 10) -> Dict:
    selected = list(dict.fromkeys(methods))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(get_golden_qa()),
        "corpus_size": len(get_evaluation_corpus()),
        "top_k": top_k,
        "methods": {method: evaluate_method(method, top_k) for method in selected},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行智语真实 RAG 检索评估")
    parser.add_argument(
        "--methods",
        default="bm25",
        help="逗号分隔：bm25,embedding,hybrid,hybrid_reranker",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - EvaluationRetriever.SUPPORTED_METHODS
    if unknown:
        parser.error(f"不支持的方法: {', '.join(sorted(unknown))}")

    result = run_evaluation(methods, args.top_k)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + os.linesep, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
