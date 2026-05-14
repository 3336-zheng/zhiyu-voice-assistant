# -*- coding: utf-8 -*-
"""
数据清洗效果 Benchmark — 对比清洗前后的 Markdown 质量
衡量指标：噪声行数、标题规范性、分块质量变化

环境: 需使用 agent_rag conda 环境

用法:
  conda activate agent_rag
  python test/benchmark_cleaning.py
  python test/benchmark_cleaning.py --verbose
"""
import sys
import os
import re
import argparse
import json
import importlib.util
from typing import List, Dict, Any
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 绕过 backend/app/__init__.py 的重依赖链
_test_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("conftest", os.path.join(_test_dir, "conftest.py"))
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_conftest.setup_backend()


# ============================================================
# 模拟噪声数据（模拟 PDF/Word 转换产生的问题）
# ============================================================

# 噪声样本 1：PDF 转换的页码和页眉页脚
DIRTY_SAMPLE_1 = """
# RAG 技术详解

第 1 页

RAG（Retrieval-Augmented Generation）是一种将检索与生成结合的技术。

第 2 页

## 数据准备

数据准备阶段包括文档加载和文本分块。

第 3 页

### 文本分块策略

文本分块是 RAG 中的关键步骤。

第 4 页

## 向量嵌入

将文本转换为向量表示。

Page 5

## 向量数据库

存储和检索向量的数据库系统。

Page 6

### ChromaDB

ChromaDB 是一个轻量级向量数据库。
"""

# 噪声样本 2：标题跳级 + 控制字符
DIRTY_SAMPLE_2 = """
# Agent 架构设计

Agent 是一种能够自主执行任务的系统。\x00\x01\x02

### 规划模块

规划模块负责任务分解。\x0b\x0c

###### 工具调用

工具调用模块扩展 Agent 的能力。\x7f

#### 记忆模块

记忆模块管理对话历史和长期知识。

## 反思机制
反思机制让 Agent 能够从错误中学习。
"""

# 噪声样本 3：重复页眉页脚 + 多余空行
DIRTY_SAMPLE_3 = """
# 机器学习基础

项目文档

机器学习是人工智能的一个分支。

项目文档

## 监督学习

监督学习使用标注数据训练模型。






项目文档

## 无监督学习

无监督学习不需要标注数据。




项目文档

## 强化学习

强化学习通过与环境交互学习最优策略。

项目文档
"""

# 噪声样本 4：混合噪声（最接近真实 PDF 转换结果）
DIRTY_SAMPLE_4 = """
# 智能语音助手项目文档

技术架构说明

## 第一章 项目概述

本项目是一个端侧智能语音笔记助手。\x00

第 1 页

## 第二章 技术栈

### 2.1 ASR 语音识别

使用 Whisper 模型进行语音转文字。

第 2 页

### 2.2 NLP 处理

使用 DeepSeek 大模型进行意图识别。

Page 3

### 2.3 RAG 检索

结合 BM25 和向量检索的混合方案。

技术架构说明

## 第三章 存储方案

采用 SQLite + ChromaDB + BM25 三级存储。

第 4 页

#### 3.1 SQLite

存储笔记元数据。

### 3.2 向量数据库

使用 ChromaDB 存储文档向量。

第 5 页

## 第四章 部署

端侧部署，无需云端依赖。
"""

# 噪声样本 5：行尾空格 + 混合编码
DIRTY_SAMPLE_5 = """
# 深度学习笔记   \x0b

神经网络基础\x00\x01

## CNN 卷积神经网络   \x0c

卷积层提取局部特征\x7f。
池化层降低特征维度。

## RNN 循环神经网络

RNN 处理序列数据。
LSTM 解决了长期依赖问题。

## Transformer

自注意力机制是 Transformer 的核心。
多头注意力增强了模型的表达能力。
"""

# 所有测试样本
DIRTY_SAMPLES = {
    "PDF页码噪声": DIRTY_SAMPLE_1,
    "控制字符+标题跳级": DIRTY_SAMPLE_2,
    "重复页眉+多余空行": DIRTY_SAMPLE_3,
    "混合噪声": DIRTY_SAMPLE_4,
    "行尾空格+控制字符": DIRTY_SAMPLE_5,
}


# ============================================================
# 清洗质量评估
# ============================================================

def count_noise_indicators(content: str) -> Dict[str, int]:
    """统计内容中的噪声指标"""
    lines = content.split("\n")

    # 页码模式
    page_pattern = re.compile(
        r"^\s*(第\s*\d+\s*页|Page\s*\d+|\d+\s*/\s*\d+|\-\s*\d+\s*\-)\s*$",
        re.MULTILINE
    )
    page_numbers = len(page_pattern.findall(content))

    # 控制字符
    control_chars = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", content))

    # 跳级标题
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    headers = []
    for line in lines:
        match = header_pattern.match(line.strip())
        if match:
            headers.append(len(match.group(1)))

    skipped_headers = 0
    for i in range(1, len(headers)):
        if headers[i] > headers[i - 1] + 1:
            skipped_headers += 1

    # 连续空行（3个以上）
    excessive_blank = len(re.findall(r"\n{3,}", content))

    # 行尾空格
    trailing_spaces = sum(1 for line in lines if line != line.rstrip())

    # 重复短行（出现 3+ 次的短行）
    from collections import Counter
    short_lines = [line.strip() for line in lines if 0 < len(line.strip()) <= 30 and not line.strip().startswith("#")]
    line_counts = Counter(short_lines)
    repeated_lines = sum(count - 1 for count in line_counts.values() if count >= 3)

    return {
        "page_numbers": page_numbers,
        "control_chars": control_chars,
        "skipped_headers": skipped_headers,
        "excessive_blank_lines": excessive_blank,
        "trailing_spaces": trailing_spaces,
        "repeated_short_lines": repeated_lines,
        "total_lines": len(lines),
        "total_chars": len(content),
    }


def evaluate_cleaning_effect(before: str, after: str) -> Dict[str, Any]:
    """评估清洗效果"""
    before_noise = count_noise_indicators(before)
    after_noise = count_noise_indicators(after)

    improvements = {}
    for key in before_noise:
        if key in ("total_lines", "total_chars"):
            continue
        b = before_noise[key]
        a = after_noise[key]
        if b > 0:
            improvements[key] = {
                "before": b,
                "after": a,
                "removed": b - a,
                "removal_rate": (b - a) / b,
            }
        else:
            improvements[key] = {"before": 0, "after": 0, "removed": 0, "removal_rate": 0}

    return {
        "before_stats": before_noise,
        "after_stats": after_noise,
        "char_reduction": before_noise["total_chars"] - after_noise["total_chars"],
        "line_reduction": before_noise["total_lines"] - after_noise["total_lines"],
        "improvements": improvements,
    }


def evaluate_chunk_impact(content_before: str, content_after: str) -> Dict[str, Any]:
    """评估清洗对分块质量的影响"""
    from backend.app.services.doc_index_service import split_markdown_by_headers

    chunks_before = split_markdown_by_headers(content_before, "test.md")
    chunks_after = split_markdown_by_headers(content_after, "test.md")

    def chunk_quality(chunks):
        if not chunks:
            return {"count": 0, "avg_size": 0, "header_aligned": 0}
        sizes = [len(c["text"]) for c in chunks]
        header_pattern = re.compile(r"^#{1,6}\s+")
        header_aligned = sum(
            1 for c in chunks
            if header_pattern.match(c["text"].strip().split("\n")[0])
        )
        return {
            "count": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "max_size": max(sizes),
            "header_aligned": header_aligned,
            "header_aligned_ratio": header_aligned / len(chunks),
        }

    q_before = chunk_quality(chunks_before)
    q_after = chunk_quality(chunks_after)

    return {
        "before": q_before,
        "after": q_after,
        "chunk_count_change": q_after["count"] - q_before["count"],
        "header_aligned_improvement": q_after["header_aligned_ratio"] - q_before["header_aligned_ratio"],
    }


# ============================================================
# Benchmark 主逻辑
# ============================================================

class CleaningBenchmark:
    """数据清洗效果 Benchmark"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self) -> Dict[str, Any]:
        """运行完整 benchmark"""
        from backend.app.services.doc_index_service import clean_markdown_for_chunking

        print(f"{'='*60}")
        print(f"  数据清洗效果 Benchmark")
        print(f"  测试样本数: {len(DIRTY_SAMPLES)}")
        print(f"{'='*60}")

        all_results = {}

        for sample_name, dirty_content in DIRTY_SAMPLES.items():
            print(f"\n--- 样本: {sample_name} ---")

            # 执行清洗
            clean_content = clean_markdown_for_chunking(dirty_content)

            # 评估清洗效果
            cleaning_effect = evaluate_cleaning_effect(dirty_content, clean_content)

            # 评估对分块的影响
            chunk_impact = evaluate_chunk_impact(dirty_content, clean_content)

            result = {
                "sample_name": sample_name,
                "original_size": len(dirty_content),
                "cleaned_size": len(clean_content),
                "cleaning_effect": cleaning_effect,
                "chunk_impact": chunk_impact,
                "cleaned_content_preview": clean_content[:500],
            }
            all_results[sample_name] = result

            # 打印摘要
            ce = cleaning_effect
            print(f"  字符: {ce['before_stats']['total_chars']} → {ce['after_stats']['total_chars']} "
                  f"(减少 {ce['char_reduction']})")
            print(f"  行数: {ce['before_stats']['total_lines']} → {ce['after_stats']['total_lines']}")

            # 噪声清除详情
            for noise_type, imp in ce["improvements"].items():
                if imp["before"] > 0:
                    print(f"  {noise_type}: {imp['before']} → {imp['after']} "
                          f"(清除率 {imp['removal_rate']:.0%})")

            # 分块影响
            ci = chunk_impact
            print(f"  分块: {ci['before']['count']} → {ci['after']['count']} 块, "
                  f"标题对齐: {ci['before']['header_aligned_ratio']:.0%} → {ci['after']['header_aligned_ratio']:.0%}")

            if self.verbose:
                print(f"\n  清洗后预览:")
                for line in clean_content[:300].split("\n")[:15]:
                    print(f"    {line}")

        # 汇总
        total_noise_before = sum(
            r["cleaning_effect"]["before_stats"]["page_numbers"] +
            r["cleaning_effect"]["before_stats"]["control_chars"] +
            r["cleaning_effect"]["before_stats"]["skipped_headers"] +
            r["cleaning_effect"]["before_stats"]["excessive_blank_lines"] +
            r["cleaning_effect"]["before_stats"]["trailing_spaces"]
            for r in all_results.values()
        )
        total_noise_after = sum(
            r["cleaning_effect"]["after_stats"]["page_numbers"] +
            r["cleaning_effect"]["after_stats"]["control_chars"] +
            r["cleaning_effect"]["after_stats"]["skipped_headers"] +
            r["cleaning_effect"]["after_stats"]["excessive_blank_lines"] +
            r["cleaning_effect"]["after_stats"]["trailing_spaces"]
            for r in all_results.values()
        )

        summary = {
            "total_samples": len(DIRTY_SAMPLES),
            "total_noise_before": total_noise_before,
            "total_noise_after": total_noise_after,
            "noise_reduction_rate": (total_noise_before - total_noise_after) / total_noise_before if total_noise_before > 0 else 0,
            "avg_header_aligned_improvement": sum(
                r["chunk_impact"]["header_aligned_improvement"] for r in all_results.values()
            ) / len(all_results),
        }

        print(f"\n{'='*60}")
        print(f"  汇总")
        print(f"{'='*60}")
        print(f"  总噪声指标: {total_noise_before} → {total_noise_after} (清除率 {summary['noise_reduction_rate']:.0%})")
        print(f"  标题对齐平均提升: {summary['avg_header_aligned_improvement']:+.0%}")

        return {
            "benchmark": "cleaning",
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "per_sample": {
                name: {
                    "original_size": r["original_size"],
                    "cleaned_size": r["cleaned_size"],
                    "cleaning_effect": r["cleaning_effect"],
                    "chunk_impact": r["chunk_impact"],
                }
                for name, r in all_results.items()
            },
        }


# ============================================================
# 输出格式化
# ============================================================

def print_summary_table(report: Dict[str, Any]):
    """打印汇总表格"""
    per_sample = report.get("per_sample", {})
    summary = report.get("summary", {})

    print(f"\n{'='*75}")
    print(f"  清洗效果对比表")
    print(f"  时间: {report.get('timestamp', '')}")
    print(f"{'='*75}")

    print(f"\n{'样本':<20} {'原文':>6} {'清洗后':>6} {'页码':>4} {'控符':>4} "
          f"{'跳级':>4} {'空行':>4} {'行尾':>4}")
    print("-" * 62)

    for name, data in per_sample.items():
        ce = data["cleaning_effect"]
        b = ce["before_stats"]
        a = ce["after_stats"]
        imp = ce["improvements"]
        print(f"{name:<20} {b['total_chars']:>5} {a['total_chars']:>5} "
              f"{imp['page_numbers']['removed']:>3} "
              f"{imp['control_chars']['removed']:>3} "
              f"{imp['skipped_headers']['removed']:>3} "
              f"{imp['excessive_blank_lines']['removed']:>3} "
              f"{imp['trailing_spaces']['removed']:>3}")

    print(f"\n  总噪声清除率: {summary.get('noise_reduction_rate', 0):.0%}")
    print(f"  标题对齐平均提升: {summary.get('avg_header_aligned_improvement', 0):+.0%}")


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
    parser = argparse.ArgumentParser(description="数据清洗效果 Benchmark")
    parser.add_argument("--verbose", action="store_true", help="显示清洗后内容预览")
    parser.add_argument("--output", type=str, default="test/benchmark_results_cleaning.json",
                        help="结果输出路径")
    args = parser.parse_args()

    benchmark = CleaningBenchmark(verbose=args.verbose)
    report = benchmark.run()

    if report:
        print_summary_table(report)
        save_report(report, os.path.join(PROJECT_ROOT, args.output))


if __name__ == "__main__":
    main()
