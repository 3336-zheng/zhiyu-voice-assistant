# -*- coding: utf-8 -*-
"""
分块策略 Benchmark — 对比 按标题分块 vs 固定长度分块
衡量指标：块数、平均块大小、大小分布、超长块比例

环境: 需使用 agent_rag conda 环境

用法:
  conda activate agent_rag
  python test/benchmark_chunking.py
  python test/benchmark_chunking.py --verbose
"""
import sys
import os
import re
import argparse
import json
import importlib.util
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 绕过 backend/app/__init__.py 的重依赖链
_test_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("conftest", os.path.join(_test_dir, "conftest.py"))
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_conftest.setup_backend()


# ============================================================
# 固定长度分块（作为 baseline 对比）
# ============================================================

def split_fixed_length(content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    固定长度分块（简单 baseline）
    按字符数切分，支持重叠。
    """
    if len(content) <= chunk_size:
        return [content]

    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def split_fixed_length_by_sentence(content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    固定长度分块（按句子边界切分，更智能的 baseline）
    优先在句子末尾（。！？.!?）切分，避免截断句子。
    """
    if len(content) <= chunk_size:
        return [content]

    chunks = []
    start = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        if end < len(content):
            # 从 end 位置向前找句子边界
            sentence_end_pattern = re.compile(r'[。！？.!?]\s*')
            best_break = end
            # 在 [end - 200, end] 范围内找最近的句子边界
            search_start = max(start + chunk_size // 2, end - 200)
            for match in sentence_end_pattern.finditer(content, search_start, end):
                best_break = match.end()
            end = best_break

        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end - overlap) if end < len(content) else end

    return chunks


# ============================================================
# 测试文档
# ============================================================

def load_test_documents() -> Dict[str, str]:
    """加载 data/docs/ 下的所有 .md 文件"""
    docs_dir = os.path.join(PROJECT_ROOT, "data", "docs")
    documents = {}

    if not os.path.exists(docs_dir):
        print(f"[WARN] 文档目录不存在: {docs_dir}")
        return documents

    for filename in os.listdir(docs_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(docs_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    documents[filename] = content
            except Exception as e:
                print(f"[WARN] 读取文件失败 {filename}: {e}")

    return documents


# ============================================================
# 分块质量评估
# ============================================================

@dataclass
class ChunkStats:
    """分块统计信息"""
    chunk_count: int
    avg_size: float          # 平均字符数
    min_size: int
    max_size: int
    median_size: float
    std_size: float          # 标准差
    oversized_count: int     # 超过 1000 字符的块数
    oversized_ratio: float   # 超长块比例
    tiny_count: int          # 小于 50 字符的块数
    size_distribution: Dict[str, int]  # 大小分布区间


def compute_chunk_stats(chunks: List[str], max_chars: int = 1000) -> ChunkStats:
    """计算分块的统计信息"""
    if not chunks:
        return ChunkStats(
            chunk_count=0, avg_size=0, min_size=0, max_size=0,
            median_size=0, std_size=0, oversized_count=0, oversized_ratio=0,
            tiny_count=0, size_distribution={}
        )

    sizes = [len(c) for c in chunks]
    avg = sum(sizes) / len(sizes)
    variance = sum((s - avg) ** 2 for s in sizes) / len(sizes) if len(sizes) > 1 else 0
    std = variance ** 0.5

    sorted_sizes = sorted(sizes)
    median = sorted_sizes[len(sorted_sizes) // 2]

    oversized = sum(1 for s in sizes if s > max_chars)
    tiny = sum(1 for s in sizes if s < 50)

    # 大小分布区间
    distribution = Counter()
    for s in sizes:
        if s < 100:
            distribution["0-100"] += 1
        elif s < 300:
            distribution["100-300"] += 1
        elif s < 500:
            distribution["300-500"] += 1
        elif s < 800:
            distribution["500-800"] += 1
        elif s <= 1000:
            distribution["800-1000"] += 1
        else:
            distribution[">1000"] += 1

    return ChunkStats(
        chunk_count=len(chunks),
        avg_size=avg,
        min_size=min(sizes),
        max_size=max(sizes),
        median_size=median,
        std_size=std,
        oversized_count=oversized,
        oversized_ratio=oversized / len(chunks),
        tiny_count=tiny,
        size_distribution=dict(distribution),
    )


def evaluate_chunk_quality(chunks: List[str]) -> Dict[str, Any]:
    """
    评估分块质量（更细致的指标）

    检查：
    1. 标题完整性 — 每个块是否以标题行开始或包含标题
    2. 段落完整性 — 块是否在段落中间被截断
    3. 语义连贯性 — 相邻块之间是否有内容重叠
    """
    header_pattern = re.compile(r"^#{1,6}\s+")

    header_aligned = 0       # 以标题开头的块数
    contains_header = 0      # 包含标题的块数
    clean_start = 0          # 以完整句子开头的块数

    for chunk in chunks:
        lines = chunk.strip().split("\n")
        first_line = lines[0].strip() if lines else ""

        if header_pattern.match(first_line):
            header_aligned += 1

        if any(header_pattern.match(line.strip()) for line in lines):
            contains_header += 1

        # 检查是否以完整句子开头（不是被截断的句子）
        if first_line and (first_line[0] in "#-*>|0123456789" or
                          first_line.endswith(("。", "！", "？", ".", "!", "?", "：", ":")) or
                          len(first_line) < 5):
            clean_start += 1

    n = len(chunks) if chunks else 1
    return {
        "header_aligned_ratio": header_aligned / n,
        "contains_header_ratio": contains_header / n,
        "clean_start_ratio": clean_start / n,
    }


# ============================================================
# Benchmark 主逻辑
# ============================================================

class ChunkingBenchmark:
    """分块策略 Benchmark"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self) -> Dict[str, Any]:
        """运行完整 benchmark"""
        documents = load_test_documents()
        if not documents:
            print("[ERROR] 没有找到测试文档，请确保 data/docs/ 下有 .md 文件")
            return {}

        print(f"加载了 {len(documents)} 个测试文档:")
        for name, content in documents.items():
            print(f"  - {name} ({len(content)} 字符)")

        # 三种分块策略
        strategies = {
            "按标题分块 (项目方案)": lambda content: self._split_by_headers(content),
            "固定长度分块 (1000字符)": lambda content: split_fixed_length(content, 1000, 100),
            "固定长度+句子边界 (1000字符)": lambda content: split_fixed_length_by_sentence(content, 1000, 100),
        }

        all_results = {}

        for strategy_name, split_func in strategies.items():
            print(f"\n{'='*60}")
            print(f"  策略: {strategy_name}")
            print(f"{'='*60}")

            strategy_results = {}

            for doc_name, doc_content in documents.items():
                chunks = split_func(doc_content)
                stats = compute_chunk_stats(chunks)
                quality = evaluate_chunk_quality(chunks)

                result = {
                    "doc_name": doc_name,
                    "doc_size": len(doc_content),
                    "stats": stats.__dict__,
                    "quality": quality,
                }
                strategy_results[doc_name] = result

                if self.verbose:
                    print(f"\n  [{doc_name}] ({len(doc_content)} 字符)")
                    print(f"    块数: {stats.chunk_count}, 平均: {stats.avg_size:.0f}字符, "
                          f"最大: {stats.max_size}, 超长: {stats.oversized_count}({stats.oversized_ratio:.0%})")
                    print(f"    标题对齐: {quality['header_aligned_ratio']:.0%}, "
                          f"含标题: {quality['contains_header_ratio']:.0%}")

            # 汇总统计
            all_chunk_counts = [r["stats"]["chunk_count"] for r in strategy_results.values()]
            all_avg_sizes = [r["stats"]["avg_size"] for r in strategy_results.values()]
            all_oversized = [r["stats"]["oversized_count"] for r in strategy_results.values()]
            all_header_aligned = [r["quality"]["header_aligned_ratio"] for r in strategy_results.values()]

            summary = {
                "total_docs": len(strategy_results),
                "total_chunks": sum(all_chunk_counts),
                "avg_chunks_per_doc": sum(all_chunk_counts) / len(all_chunk_counts),
                "avg_chunk_size": sum(all_avg_sizes) / len(all_avg_sizes),
                "total_oversized": sum(all_oversized),
                "avg_header_aligned": sum(all_header_aligned) / len(all_header_aligned),
            }

            all_results[strategy_name] = {
                "summary": summary,
                "per_doc": strategy_results,
            }

            print(f"\n  汇总: {summary['total_chunks']} 块, "
                  f"平均 {summary['avg_chunks_per_doc']:.1f} 块/文档, "
                  f"平均块大小 {summary['avg_chunk_size']:.0f} 字符, "
                  f"超长块 {summary['total_oversized']} 个")

        return {
            "benchmark": "chunking",
            "timestamp": datetime.now().isoformat(),
            "strategies": {
                name: {
                    "summary": result["summary"],
                    "per_doc": {
                        dname: {k: v for k, v in dresult.items() if k != "chunks_sample"}
                        for dname, dresult in result["per_doc"].items()
                    }
                }
                for name, result in all_results.items()
            }
        }

    def _split_by_headers(self, content: str) -> List[str]:
        """使用项目的按标题分块策略"""
        from backend.app.services.doc_index_service import split_markdown_by_headers
        chunks_data = split_markdown_by_headers(content, "test.md")
        return [c["text"] for c in chunks_data]


# ============================================================
# 输出格式化
# ============================================================

def print_comparison_table(report: Dict[str, Any]):
    """打印对比表格"""
    strategies = report.get("strategies", {})

    print(f"\n{'='*70}")
    print(f"  分块策略对比结果")
    print(f"  时间: {report.get('timestamp', '')}")
    print(f"{'='*70}")

    # 汇总对比表
    print(f"\n{'策略':<30} {'总块数':>6} {'均块/文':>8} {'均大小':>8} {'超长块':>6} {'标题对齐':>8}")
    print("-" * 72)
    for name, data in strategies.items():
        s = data["summary"]
        print(f"{name:<30} {s['total_chunks']:>6} {s['avg_chunks_per_doc']:>7.1f} "
              f"{s['avg_chunk_size']:>7.0f} {s['total_oversized']:>6} {s['avg_header_aligned']:>7.0%}")

    # 逐文档对比
    print(f"\n{'='*70}")
    print("  逐文档对比")
    print(f"{'='*70}")

    # 收集所有文档名
    all_docs = set()
    for data in strategies.values():
        all_docs.update(data.get("per_doc", {}).keys())

    for doc_name in sorted(all_docs):
        print(f"\n  [{doc_name}]")
        for strat_name, data in strategies.items():
            doc_data = data.get("per_doc", {}).get(doc_name, {})
            if not doc_data:
                continue
            stats = doc_data.get("stats", {})
            print(f"    {strat_name:<30} 块数={stats.get('chunk_count', 0):>3}, "
                  f"均大小={stats.get('avg_size', 0):>6.0f}, "
                  f"最大={stats.get('max_size', 0):>5}, "
                  f"超长={stats.get('oversized_count', 0)}")


def save_report(report: Dict[str, Any], output_path: str):
    """保存结果到 JSON"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="分块策略 Benchmark")
    parser.add_argument("--verbose", action="store_true", help="显示每个文档的详细结果")
    parser.add_argument("--output", type=str, default="test/benchmark_results_chunking.json",
                        help="结果输出路径")
    args = parser.parse_args()

    benchmark = ChunkingBenchmark(verbose=args.verbose)
    report = benchmark.run()

    if report:
        print_comparison_table(report)
        save_report(report, os.path.join(PROJECT_ROOT, args.output))


if __name__ == "__main__":
    main()
