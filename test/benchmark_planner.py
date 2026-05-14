# -*- coding: utf-8 -*-
"""
Planner 意图识别准确率 Benchmark
测试 LLM 意图识别和规则匹配两种模式

环境: 需使用 agent_rag conda 环境

用法:
  conda activate agent_rag
  python test/benchmark_planner.py
  python test/benchmark_planner.py --verbose
  python test/benchmark_planner.py --mode rules    # 仅测试规则匹配
  python test/benchmark_planner.py --mode llm      # 仅测试 LLM
"""
import sys
import os
import time
import argparse
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 切换工作目录到项目根目录（确保 .env 文件能被读取）
os.chdir(PROJECT_ROOT)

# 绕过 backend/app/__init__.py 的重依赖链
_test_dir = os.path.dirname(os.path.abspath(__file__))
_spec = __import__('importlib').util.spec_from_file_location("conftest", os.path.join(_test_dir, "conftest.py"))
_conftest = __import__('importlib').util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_conftest.setup_backend()

from backend.app.agent.planner import Planner, get_planner
from backend.app.agent.models import IntentType


# ============================================================
# 测试用例定义
# ============================================================

@dataclass
class PlannerTestCase:
    """Planner 测试用例"""
    query: str                                    # 用户查询
    expected_intent: str                          # 期望的意图类型
    expected_keywords: List[str] = None           # 期望参数中包含的关键词
    category: str = ""                            # 测试类别
    description: str = ""                         # 描述
    is_asr_error: bool = False                    # 是否包含 ASR 错误


# 测试用例集合
TEST_CASES: List[PlannerTestCase] = [
    # --- 检索意图 ---
    PlannerTestCase(
        query="查找关于AI的笔记",
        expected_intent="search",
        expected_keywords=["AI"],
        category="检索",
        description="标准检索查询"
    ),
    PlannerTestCase(
        query="搜索会议记录",
        expected_intent="search",
        expected_keywords=["会议"],
        category="检索",
        description="搜索特定类型内容"
    ),
    PlannerTestCase(
        query="RAG是什么",
        expected_intent="search",
        expected_keywords=["RAG"],
        category="检索",
        description="知识查询"
    ),
    PlannerTestCase(
        query="向量数据库有哪些",
        expected_intent="search",
        expected_keywords=["向量数据库"],
        category="检索",
        description="列举类查询"
    ),

    # --- ASR 纠错场景 ---
    PlannerTestCase(
        query="帮我查走RAG的词类方式分快",
        expected_intent="search",
        expected_keywords=["RAG", "分块"],
        category="ASR纠错",
        description="走→找, 词类→词语, 分快→分块",
        is_asr_error=True
    ),
    PlannerTestCase(
        query="找一下有关向量数据库的笔",
        expected_intent="search",
        expected_keywords=["向量数据库"],
        category="ASR纠错",
        description="笔→笔记",
        is_asr_error=True
    ),
    PlannerTestCase(
        query="看看那个agent开发的坑",
        expected_intent="search",
        expected_keywords=["Agent", "开发"],
        category="ASR纠错",
        description="看看→去除, 坑→踩坑",
        is_asr_error=True
    ),
    PlannerTestCase(
        query="帮我查一下深度学习的资聊",
        expected_intent="search",
        expected_keywords=["深度学习"],
        category="ASR纠错",
        description="资聊→资料",
        is_asr_error=True
    ),

    # --- 创建笔记意图 ---
    PlannerTestCase(
        query="创建笔记标题是测试内容是Hello World",
        expected_intent="create_note",
        expected_keywords=["测试", "Hello World"],
        category="创建笔记",
        description="带标题和内容的创建"
    ),
    PlannerTestCase(
        query="新建笔记",
        expected_intent="create_note",
        category="创建笔记",
        description="简单创建"
    ),
    PlannerTestCase(
        query="记录一下今天的会议要点",
        expected_intent="create_note",
        expected_keywords=["会议"],
        category="创建笔记",
        description="口语化创建"
    ),
    PlannerTestCase(
        query="记一下明天要做的事情",
        expected_intent="create_note",
        category="创建笔记",
        description="口语化创建2"
    ),

    # --- 更新笔记意图 ---
    PlannerTestCase(
        query="更新笔记ID为5的内容改为新内容",
        expected_intent="update_note",
        expected_keywords=["5"],
        category="更新笔记",
        description="指定ID更新"
    ),
    PlannerTestCase(
        query="修改笔记标题",
        expected_intent="update_note",
        category="更新笔记",
        description="简单修改"
    ),

    # --- 删除笔记意图 ---
    PlannerTestCase(
        query="删除笔记ID为3",
        expected_intent="delete_note",
        expected_keywords=["3"],
        category="删除笔记",
        description="指定ID删除"
    ),
    PlannerTestCase(
        query="删掉这条笔记",
        expected_intent="delete_note",
        category="删除笔记",
        description="口语化删除"
    ),

    # --- 列出笔记意图 ---
    PlannerTestCase(
        query="列出所有笔记",
        expected_intent="list_notes",
        category="列出笔记",
        description="列出全部"
    ),
    PlannerTestCase(
        query="显示本周的笔记",
        expected_intent="list_notes",
        category="列出笔记",
        description="带时间范围"
    ),
    PlannerTestCase(
        query="笔记列表",
        expected_intent="list_notes",
        category="列出笔记",
        description="简单列表"
    ),

    # --- 时间查询意图 ---
    PlannerTestCase(
        query="现在几点",
        expected_intent="time_query",
        category="时间查询",
        description="查询时间"
    ),
    PlannerTestCase(
        query="今天星期几",
        expected_intent="time_query",
        category="时间查询",
        description="查询星期"
    ),
    PlannerTestCase(
        query="今天日期是多少",
        expected_intent="time_query",
        category="时间查询",
        description="查询日期"
    ),

    # --- 摘要意图 ---
    PlannerTestCase(
        query="总结关于项目的讨论",
        expected_intent="summarize",
        expected_keywords=["项目"],
        category="摘要",
        description="总结特定主题"
    ),
    PlannerTestCase(
        query="概括一下AI相关内容",
        expected_intent="summarize",
        expected_keywords=["AI"],
        category="摘要",
        description="概括类查询"
    ),

    # --- 创建MD文件意图 ---
    PlannerTestCase(
        query="创建一个md文件名叫会议记录",
        expected_intent="create_md",
        expected_keywords=["会议记录"],
        category="创建MD",
        description="指定文件名创建"
    ),
    PlannerTestCase(
        query="新建笔记文件",
        expected_intent="create_md",
        category="创建MD",
        description="简单创建MD"
    ),

    # --- 写入MD文件意图 ---
    PlannerTestCase(
        query="把这段话写进会议记录文件",
        expected_intent="write_md",
        expected_keywords=["会议记录"],
        category="写入MD",
        description="指定文件写入"
    ),
    PlannerTestCase(
        query="写入笔记文件内容是今天的会议要点",
        expected_intent="write_md",
        expected_keywords=["会议要点"],
        category="写入MD",
        description="带内容写入"
    ),
]


# ============================================================
# Benchmark 核心逻辑
# ============================================================

class PlannerBenchmark:
    """Planner 意图识别 Benchmark"""

    def __init__(self, mode: str = "both", verbose: bool = False):
        """
        Args:
            mode: 测试模式 - "llm", "rules", "both"
            verbose: 是否显示详细结果
        """
        self.mode = mode
        self.verbose = verbose
        self.planner = None

    def _init_planner(self):
        """初始化 Planner"""
        self.planner = get_planner()

    def _test_single_case(self, test_case: PlannerTestCase, use_llm: bool = True) -> Dict[str, Any]:
        """
        测试单个用例

        Args:
            test_case: 测试用例
            use_llm: 是否使用 LLM

        Returns:
            测试结果
        """
        start = time.time()

        try:
            if use_llm:
                plan = self.planner.plan(test_case.query)
            else:
                plan = self.planner._plan_with_rules(test_case.query)

            latency = (time.time() - start) * 1000  # 毫秒

            # 检查意图是否匹配
            actual_intent = plan.intent.value if hasattr(plan.intent, 'value') else str(plan.intent)
            intent_match = actual_intent == test_case.expected_intent

            # 检查关键词是否包含
            keyword_match = True
            matched_keywords = []
            missing_keywords = []

            if test_case.expected_keywords:
                # 从计划参数中提取实际值
                actual_values = self._extract_plan_values(plan)
                actual_text = " ".join(actual_values).lower()

                for kw in test_case.expected_keywords:
                    if kw.lower() in actual_text:
                        matched_keywords.append(kw)
                    else:
                        missing_keywords.append(kw)
                        keyword_match = False

            return {
                "query": test_case.query,
                "expected_intent": test_case.expected_intent,
                "actual_intent": actual_intent,
                "intent_match": intent_match,
                "keyword_match": keyword_match,
                "matched_keywords": matched_keywords,
                "missing_keywords": missing_keywords,
                "latency_ms": latency,
                "reasoning": plan.reasoning if hasattr(plan, 'reasoning') else "",
                "success": True,
            }

        except Exception as e:
            latency = (time.time() - start) * 1000
            return {
                "query": test_case.query,
                "expected_intent": test_case.expected_intent,
                "actual_intent": None,
                "intent_match": False,
                "keyword_match": False,
                "matched_keywords": [],
                "missing_keywords": test_case.expected_keywords or [],
                "latency_ms": latency,
                "error": str(e),
                "success": False,
            }

    def _extract_plan_values(self, plan) -> List[str]:
        """从计划中提取所有参数值"""
        values = []

        if plan.steps:
            for step in plan.steps:
                params = step.parameters if hasattr(step, 'parameters') else {}
                if isinstance(params, dict):
                    for v in params.values():
                        if isinstance(v, str):
                            values.append(v)
                        elif isinstance(v, list):
                            values.extend([str(item) for item in v])

        # 添加原始查询
        if hasattr(plan, 'original_query'):
            values.append(plan.original_query)

        # 添加推理
        if hasattr(plan, 'reasoning'):
            values.append(plan.reasoning)

        return values

    def run(self) -> Dict[str, Any]:
        """运行完整 benchmark"""
        self._init_planner()

        results = {}

        # 测试 LLM 模式
        if self.mode in ("llm", "both"):
            print(f"\n{'='*60}")
            print(f"  测试 LLM 意图识别")
            print(f"{'='*60}")
            results["llm"] = self._run_test_suite(use_llm=True)

        # 测试规则匹配模式
        if self.mode in ("rules", "both"):
            print(f"\n{'='*60}")
            print(f"  测试规则匹配意图识别")
            print(f"{'='*60}")
            results["rules"] = self._run_test_suite(use_llm=False)

        return {
            "benchmark": "planner_intent",
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "test_cases": len(TEST_CASES),
            "results": results,
        }

    def _run_test_suite(self, use_llm: bool) -> Dict[str, Any]:
        """运行测试套件"""
        mode_name = "LLM" if use_llm else "规则匹配"
        details = []
        intent_correct = 0
        keyword_correct = 0
        total_latency = 0
        success_count = 0

        # 按类别统计
        category_stats = {}

        for i, tc in enumerate(TEST_CASES):
            result = self._test_single_case(tc, use_llm=use_llm)
            details.append(result)

            # 统计
            if result["success"]:
                success_count += 1
                if result["intent_match"]:
                    intent_correct += 1
                if result["keyword_match"]:
                    keyword_correct += 1
                total_latency += result["latency_ms"]

            # 按类别统计
            cat = tc.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "intent_correct": 0, "keyword_correct": 0}
            category_stats[cat]["total"] += 1
            if result["intent_match"]:
                category_stats[cat]["intent_correct"] += 1
            if result["keyword_match"]:
                category_stats[cat]["keyword_correct"] += 1

            # 打印详细结果
            if self.verbose:
                status = "✓" if result["intent_match"] else "✗"
                kw_status = "✓" if result["keyword_match"] else "✗"
                print(f"  [{status}] [{kw_status}] '{tc.query[:30]}...' "
                      f"→ {result['actual_intent']} ({result['latency_ms']:.0f}ms)")
                if not result["intent_match"]:
                    print(f"        期望: {tc.expected_intent}, 实际: {result['actual_intent']}")
                if result.get("missing_keywords"):
                    print(f"        缺失关键词: {result['missing_keywords']}")

        n = len(TEST_CASES)
        intent_accuracy = intent_correct / n if n > 0 else 0
        keyword_accuracy = keyword_correct / n if n > 0 else 0
        avg_latency = total_latency / success_count if success_count > 0 else 0

        # 打印汇总
        print(f"\n  意图识别准确率: {intent_accuracy:.1%} ({intent_correct}/{n})")
        print(f"  关键词匹配准确率: {keyword_accuracy:.1%} ({keyword_correct}/{n})")
        print(f"  平均延迟: {avg_latency:.0f}ms")
        print(f"  成功率: {success_count/n:.1%} ({success_count}/{n})")

        # 按类别打印
        print(f"\n  按类别统计:")
        for cat, stats in sorted(category_stats.items()):
            cat_acc = stats["intent_correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"    {cat:<10} {cat_acc:.0%} ({stats['intent_correct']}/{stats['total']})")

        return {
            "intent_accuracy": intent_accuracy,
            "keyword_accuracy": keyword_accuracy,
            "avg_latency_ms": avg_latency,
            "success_rate": success_count / n,
            "intent_correct": intent_correct,
            "keyword_correct": keyword_correct,
            "success_count": success_count,
            "total": n,
            "category_stats": category_stats,
            "details": details,
        }


# ============================================================
# 输出格式化
# ============================================================

def print_comparison(report: Dict[str, Any]):
    """打印对比结果"""
    results = report.get("results", {})

    if len(results) == 2:
        print(f"\n{'='*70}")
        print(f"  Planner 意图识别 Benchmark 对比")
        print(f"  时间: {report.get('timestamp', '')}")
        print(f"  测试用例数: {report.get('test_cases', 0)}")
        print(f"{'='*70}")

        print(f"\n{'模式':<15} {'意图准确率':>10} {'关键词准确率':>12} {'平均延迟':>10} {'成功率':>8}")
        print("-" * 55)

        for mode_name, data in results.items():
            label = "LLM" if mode_name == "llm" else "规则匹配"
            print(f"{label:<15} {data['intent_accuracy']:>9.1%} {data['keyword_accuracy']:>11.1%} "
                  f"{data['avg_latency_ms']:>8.0f}ms {data['success_rate']:>7.1%}")

        # 分析差异
        llm_data = results.get("llm", {})
        rules_data = results.get("rules", {})

        if llm_data and rules_data:
            print(f"\n  分析:")
            if llm_data["intent_accuracy"] > rules_data["intent_accuracy"]:
                diff = llm_data["intent_accuracy"] - rules_data["intent_accuracy"]
                print(f"    LLM 意图准确率优于规则匹配 +{diff:.1%}")
            elif rules_data["intent_accuracy"] > llm_data["intent_accuracy"]:
                diff = rules_data["intent_accuracy"] - llm_data["intent_accuracy"]
                print(f"    规则匹配意图准确率优于 LLM +{diff:.1%}")
            else:
                print(f"    两者意图准确率相同")

            latency_diff = llm_data["avg_latency_ms"] - rules_data["avg_latency_ms"]
            print(f"    LLM 延迟比规则匹配高 {latency_diff:.0f}ms")

            # 按类别对比
            llm_cats = llm_data.get("category_stats", {})
            rules_cats = rules_data.get("category_stats", {})

            print(f"\n  按类别对比:")
            all_cats = set(list(llm_cats.keys()) + list(rules_cats.keys()))
            for cat in sorted(all_cats):
                llm_acc = llm_cats.get(cat, {}).get("intent_correct", 0) / llm_cats.get(cat, {}).get("total", 1)
                rules_acc = rules_cats.get(cat, {}).get("intent_correct", 0) / rules_cats.get(cat, {}).get("total", 1)
                winner = "LLM" if llm_acc > rules_acc else ("规则" if rules_acc > llm_acc else "平")
                print(f"    {cat:<10} LLM={llm_acc:.0%} 规则={rules_acc:.0%} → {winner}")


def print_missed_cases(report: Dict[str, Any]):
    """打印识别错误的用例"""
    results = report.get("results", {})

    for mode_name, data in results.items():
        label = "LLM" if mode_name == "llm" else "规则匹配"
        missed = [d for d in data.get("details", []) if not d.get("intent_match", False)]

        if missed:
            print(f"\n  {label} 识别错误 ({len(missed)} 个):")
            for d in missed:
                print(f"    - '{d['query'][:40]}'")
                print(f"      期望: {d['expected_intent']}, 实际: {d['actual_intent']}")


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
    parser = argparse.ArgumentParser(description="Planner 意图识别 Benchmark")
    parser.add_argument("--mode", type=str, default="both", choices=["llm", "rules", "both"],
                        help="测试模式: llm, rules, both")
    parser.add_argument("--verbose", action="store_true", help="显示每个用例的详细结果")
    parser.add_argument("--output", type=str, default="test/benchmark_results_planner.json",
                        help="结果输出路径")
    args = parser.parse_args()

    benchmark = PlannerBenchmark(mode=args.mode, verbose=args.verbose)
    report = benchmark.run()

    if report:
        print_comparison(report)
        print_missed_cases(report)
        save_report(report, os.path.join(PROJECT_ROOT, args.output))


if __name__ == "__main__":
    main()
