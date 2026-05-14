# -*- coding: utf-8 -*-
"""
一键运行所有 Benchmark
汇总检索、分块、清洗三项 benchmark 的结果

用法（需使用 agent_rag 环境）:
  # 方式一：激活环境后运行
  conda activate agent_rag
  python test/run_all_benchmarks.py

  # 方式二：直接指定 Python 路径
  C:/Users/ZHENGJUNHAO/anaconda3/envs/agent_rag/python.exe test/run_all_benchmarks.py

  # 可选参数
  python test/run_all_benchmarks.py --skip-retrieval   # 跳过耗时较长的检索 benchmark
  python test/run_all_benchmarks.py --verbose           # 详细输出
"""
import sys
import os
import time
import argparse
import json
import importlib.util
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, TEST_DIR)

# 绕过 backend/app/__init__.py 的重依赖链
_spec = importlib.util.spec_from_file_location("conftest", os.path.join(TEST_DIR, "conftest.py"))
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_conftest.setup_backend()


def _load_module(module_name: str, file_path: str):
    """动态加载模块（避免依赖 __init__.py）"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cleaning_benchmark(verbose: bool = False) -> dict:
    """运行清洗 benchmark"""
    mod = _load_module("benchmark_cleaning", os.path.join(TEST_DIR, "benchmark_cleaning.py"))
    benchmark = mod.CleaningBenchmark(verbose=verbose)
    return benchmark.run()


def run_chunking_benchmark(verbose: bool = False) -> dict:
    """运行分块 benchmark"""
    mod = _load_module("benchmark_chunking", os.path.join(TEST_DIR, "benchmark_chunking.py"))
    benchmark = mod.ChunkingBenchmark(verbose=verbose)
    return benchmark.run()


def run_retrieval_benchmark(top_k: int = 5, verbose: bool = False) -> dict:
    """运行检索 benchmark"""
    mod = _load_module("benchmark_retrieval", os.path.join(TEST_DIR, "benchmark_retrieval.py"))
    benchmark = mod.RetrievalBenchmark(top_k=top_k, verbose=verbose)
    return benchmark.run()


def generate_summary_report(retrieval_report: dict, chunking_report: dict, cleaning_report: dict) -> str:
    """生成文本摘要报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("  智语项目 Benchmark 汇总报告")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # --- 检索 Benchmark ---
    if retrieval_report:
        lines.append("\n[一] 检索方法对比")
        lines.append("-" * 50)
        methods = retrieval_report.get("methods", {})
        lines.append(f"{'方法':<22} {'命中率':>8} {'精确率':>8} {'召回率':>8} {'延迟':>8}")
        lines.append("-" * 56)
        for name, m in methods.items():
            lines.append(f"{name:<22} {m['hit_rate']:>7.1%} {m['avg_precision']:>7.1%} "
                        f"{m['avg_recall']:>7.1%} {m['avg_latency_ms']:>6.0f}ms")

        # 最佳方法
        best_method = max(methods.items(), key=lambda x: x[1]["hit_rate"])
        lines.append(f"\n  最佳方法: {best_method[0]} (命中率 {best_method[1]['hit_rate']:.1%})")

    # --- 分块 Benchmark ---
    if chunking_report:
        lines.append("\n[二] 分块策略对比")
        lines.append("-" * 50)
        strategies = chunking_report.get("strategies", {})
        lines.append(f"{'策略':<30} {'总块数':>6} {'均大小':>8} {'超长块':>6} {'标题对齐':>8}")
        lines.append("-" * 62)
        for name, data in strategies.items():
            s = data["summary"]
            lines.append(f"{name:<30} {s['total_chunks']:>6} {s['avg_chunk_size']:>7.0f} "
                        f"{s['total_oversized']:>6} {s['avg_header_aligned']:>7.0%}")

    # --- 清洗 Benchmark ---
    if cleaning_report:
        lines.append("\n[三] 数据清洗效果")
        lines.append("-" * 50)
        summary = cleaning_report.get("summary", {})
        lines.append(f"  噪声清除率: {summary.get('noise_reduction_rate', 0):.0%}")
        lines.append(f"  标题对齐提升: {summary.get('avg_header_aligned_improvement', 0):+.0%}")

        per_sample = cleaning_report.get("per_sample", {})
        for name, data in per_sample.items():
            ce = data["cleaning_effect"]
            imp = ce["improvements"]
            noise_removed = sum(v["removed"] for v in imp.values())
            lines.append(f"    {name}: 清除 {noise_removed} 个噪声指标")

    # --- 结论 ---
    lines.append("\n" + "=" * 70)
    lines.append("  关键结论")
    lines.append("=" * 70)

    if retrieval_report:
        methods = retrieval_report.get("methods", {})
        bm25_hr = methods.get("BM25", {}).get("hit_rate", 0)
        emb_hr = methods.get("Embedding", {}).get("hit_rate", 0)
        hybrid_hr = methods.get("Hybrid (RRF)", {}).get("hit_rate", 0)
        full_hr = methods.get("Hybrid+Reranker", {}).get("hit_rate", 0)

        lines.append(f"\n  1. 混合检索效果:")
        lines.append(f"     BM25: {bm25_hr:.1%} → Embedding: {emb_hr:.1%} → "
                    f"Hybrid: {hybrid_hr:.1%} → +Reranker: {full_hr:.1%}")

        bm25_lat = methods.get("BM25", {}).get("avg_latency_ms", 0)
        full_lat = methods.get("Hybrid+Reranker", {}).get("avg_latency_ms", 0)
        lines.append(f"     延迟代价: BM25 {bm25_lat:.0f}ms → Hybrid+Reranker {full_lat:.0f}ms "
                    f"({full_lat/bm25_lat:.1f}x)" if bm25_lat > 0 else "")

    if chunking_report:
        strategies = chunking_report.get("strategies", {})
        if "按标题分块 (项目方案)" in strategies and "固定长度分块 (1000字符)" in strategies:
            h_aligned = strategies["按标题分块 (项目方案)"]["summary"]["avg_header_aligned"]
            f_aligned = strategies["固定长度分块 (1000字符)"]["summary"]["avg_header_aligned"]
            lines.append(f"\n  2. 分块策略: 按标题分块标题对齐率 {h_aligned:.0%} vs 固定长度 {f_aligned:.0%}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="一键运行所有 Benchmark")
    parser.add_argument("--skip-retrieval", action="store_true", help="跳过检索 benchmark（耗时较长）")
    parser.add_argument("--top-k", type=int, default=5, help="检索 benchmark 的 top-k")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--output-dir", type=str, default="test", help="结果输出目录")
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()

    # 1. 数据清洗 Benchmark（快速，无需模型）
    print("\n" + "=" * 60)
    print("  [1/3] 数据清洗 Benchmark")
    print("=" * 60)
    t = time.time()
    cleaning_report = run_cleaning_benchmark(verbose=args.verbose)
    print(f"\n  清洗 benchmark 完成 ({time.time()-t:.1f}s)")

    # 2. 分块策略 Benchmark（快速，无需模型）
    print("\n" + "=" * 60)
    print("  [2/3] 分块策略 Benchmark")
    print("=" * 60)
    t = time.time()
    chunking_report = run_chunking_benchmark(verbose=args.verbose)
    print(f"\n  分块 benchmark 完成 ({time.time()-t:.1f}s)")

    # 3. 检索 Benchmark（耗时较长，需要加载模型）
    retrieval_report = {}
    if not args.skip_retrieval:
        print("\n" + "=" * 60)
        print("  [3/3] 检索方法 Benchmark（需要加载模型，预计 2-5 分钟）")
        print("=" * 60)
        t = time.time()
        retrieval_report = run_retrieval_benchmark(top_k=args.top_k, verbose=args.verbose)
        print(f"\n  检索 benchmark 完成 ({time.time()-t:.1f}s)")
    else:
        print("\n  [跳过] 检索 benchmark (--skip-retrieval)")

    total_elapsed = time.time() - total_start

    # 生成汇总报告
    summary = generate_summary_report(retrieval_report, chunking_report, cleaning_report)
    print(summary)

    # 保存所有结果
    full_report = {
        "benchmark_run": {
            "timestamp": datetime.now().isoformat(),
            "total_elapsed_seconds": total_elapsed,
            "top_k": args.top_k,
        },
        "retrieval": retrieval_report,
        "chunking": chunking_report,
        "cleaning": cleaning_report,
    }

    result_path = os.path.join(output_dir, "benchmark_results_all.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存到: {result_path}")

    # 保存文本摘要
    summary_path = os.path.join(output_dir, "benchmark_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"文本摘要已保存到: {summary_path}")

    print(f"\n总耗时: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
