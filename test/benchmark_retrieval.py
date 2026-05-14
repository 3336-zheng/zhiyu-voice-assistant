# -*- coding: utf-8 -*-
"""
检索方法 Benchmark — 对比 BM25 / Embedding / Hybrid / Hybrid+Reranker
衡量指标：命中率(Hit Rate)、召回率(Recall)、精确率(Precision)、延迟(Latency)

前提: 数据库和向量库已初始化（服务已启动过至少一次，ChromaDB 和 BM25 已有数据）
环境: 需使用 agent_rag conda 环境

用法:
  conda activate agent_rag
  python test/benchmark_retrieval.py              # 运行完整 benchmark
  python test/benchmark_retrieval.py --top-k 10   # 指定 top-k
  python test/benchmark_retrieval.py --verbose     # 显示详细结果
"""
import sys
import os
import time
import argparse
import json
import importlib.util
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 绕过 backend/app/__init__.py 的重依赖链
_test_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("conftest", os.path.join(_test_dir, "conftest.py"))
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_conftest.setup_backend()

# ============================================================
# 测试用例定义
# ============================================================

@dataclass
class RetrievalTestCase:
    """检索测试用例"""
    query: str                                    # 查询文本
    relevant_doc_keywords: List[str]              # 相关文档应包含的关键词（用于判定命中）
    relevant_filenames: List[str]                 # 期望命中的文件名
    category: str = ""                            # 测试类别
    description: str = ""                         # 描述


# 基于 data/docs/ 中的实际文档设计测试用例
# 每个用例的 relevant_filenames 指定了该查询理论上应命中的文档
TEST_CASES: List[RetrievalTestCase] = [
    # --- 关键词匹配场景（BM25 优势） ---
    RetrievalTestCase(
        query="BM25分词和倒排索引的原理",
        relevant_doc_keywords=["BM25", "分词", "倒排", "索引"],
        relevant_filenames=["RAG面试问题汇总.md", "基于《all-in-rag》的RAG核心知识点.md"],
        category="关键词匹配",
        description="精确术语查询，BM25 应表现优秀"
    ),
    RetrievalTestCase(
        query="ChromaDB向量数据库配置和持久化",
        relevant_doc_keywords=["ChromaDB", "向量数据库", "持久化"],
        relevant_filenames=["RAG检索全流程思维导图.md", "基于《all-in-rag》的RAG核心知识点.md"],
        category="关键词匹配",
        description="专有名词查询"
    ),
    RetrievalTestCase(
        query="ReAct框架的思考行动观察循环",
        relevant_doc_keywords=["ReAct", "思考", "行动", "观察"],
        relevant_filenames=["Agent面试核心知识点汇总.md"],
        category="关键词匹配",
        description="Agent 概念精确查询"
    ),

    # --- 语义理解场景（Embedding 优势） ---
    RetrievalTestCase(
        query="如何让大模型减少胡说八道的问题",
        relevant_doc_keywords=["幻觉", "Hallucination", "RAG"],
        relevant_filenames=["RAG面试问题汇总.md", "基于《all-in-rag》的RAG核心知识点.md"],
        category="语义理解",
        description="口语化表达，需要语义理解映射到'幻觉'概念"
    ),
    RetrievalTestCase(
        query="怎样把很长的文章切分成小段来搜索",
        relevant_doc_keywords=["分块", "切分", "chunk"],
        relevant_filenames=["基于《all-in-rag》的RAG核心知识点.md", "RAG检索全流程思维导图.md"],
        category="语义理解",
        description="自然语言描述，需要映射到'文本分块'概念"
    ),
    RetrievalTestCase(
        query="多个搜索结果怎么合并排成一个列表",
        relevant_doc_keywords=["融合", "RRF", "排序"],
        relevant_filenames=["RAG检索全流程思维导图.md", "RAG面试问题汇总.md"],
        category="语义理解",
        description="口语化描述 RRF 融合排序"
    ),

    # --- 跨文档概念场景（Hybrid 应优势） ---
    RetrievalTestCase(
        query="RAG系统的完整开发流程从数据准备到检索生成",
        relevant_doc_keywords=["RAG", "检索", "生成", "数据准备"],
        relevant_filenames=[
            "RAG面试问题汇总.md",
            "基于《all-in-rag》的RAG核心知识点.md",
            "RAG检索全流程思维导图.md"
        ],
        category="跨文档",
        description="综合性查询，需要跨多个文档检索"
    ),
    RetrievalTestCase(
        query="Agent开发中遇到的坑和实际经验教训",
        relevant_doc_keywords=["Agent", "踩坑", "经验"],
        relevant_filenames=[
            "Agent应用开发实践踩坑与经验分享.md",
            "Agent面试核心知识点汇总.md"
        ],
        category="跨文档",
        description="经验类查询，需要同时命中踩坑文档和知识点文档"
    ),

    # --- 精确术语场景 ---
    RetrievalTestCase(
        query="BGE-M3嵌入模型和BGE-reranker重排序",
        relevant_doc_keywords=["BGE", "嵌入", "reranker", "重排序"],
        relevant_filenames=["RAG检索全流程思维导图.md", "RAG面试问题汇总.md"],
        category="精确术语",
        description="模型名称精确查询"
    ),
    RetrievalTestCase(
        query="Plan-and-Execute架构模式的拓扑排序并行执行",
        relevant_doc_keywords=["Plan", "Execute", "拓扑", "并行"],
        relevant_filenames=["Agent面试核心知识点汇总.md", "智语--端侧智能语音笔记助手（自研 Agent + RAG） - 项目文档.md"],
        category="精确术语",
        description="架构模式精确查询"
    ),

    # --- 混合难度场景 ---
    RetrievalTestCase(
        query="语义分块是怎么判断在哪里切开的",
        relevant_doc_keywords=["语义分块", "语义", "切分", "断点"],
        relevant_filenames=["基于《all-in-rag》的RAG核心知识点.md"],
        category="混合难度",
        description="口语化 + 专业概念混合"
    ),
    RetrievalTestCase(
        query="对话记忆和会话管理的上下文窗口",
        relevant_doc_keywords=["记忆", "会话", "上下文", "对话"],
        relevant_filenames=["Agent面试核心知识点汇总.md", "智语--端侧智能语音笔记助手（自研 Agent + RAG） - 项目文档.md"],
        category="混合难度",
        description="多概念组合查询"
    ),
]


# ============================================================
# Benchmark 核心逻辑
# ============================================================

class RetrievalBenchmark:
    """检索方法 Benchmark"""

    def __init__(self, top_k: int = 5, verbose: bool = False):
        self.top_k = top_k
        self.verbose = verbose
        self.results: Dict[str, List[Dict]] = {}

    def _init_services(self):
        """延迟初始化服务（避免 import 时就加载模型）"""
        # 切换工作目录到项目根目录（解决相对路径问题）
        os.chdir(PROJECT_ROOT)

        from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
        from backend.app.services.embedding_service import get_embedding_service
        from backend.app.services.bm25_service import get_bm25_service
        from backend.app.services.chroma_service import get_chroma_service
        from backend.app.services.reranker_service import get_reranker_service
        from backend.app.services.rrf_service import get_rrf_service
        from backend.app.services.doc_index_service import get_doc_index_service
        from backend.app.core.database import engine, Base
        from backend.app.models import Note, Audio  # 触发模型注册

        print("正在加载服务（首次加载模型可能需要 30-60 秒）...")
        start = time.time()

        # 确保数据库表存在
        Base.metadata.create_all(bind=engine)

        self.hybrid_service = get_hybrid_retrieval_service()
        self.embedding_service = get_embedding_service()
        self.bm25_service = get_bm25_service()
        self.chroma_service = get_chroma_service()
        self.reranker_service = get_reranker_service()
        self.rrf_service = get_rrf_service()

        # 同步文档索引：重建 BM25 + 索引 data/docs/ 中的文档到 ChromaDB
        print("  正在同步文档索引...")
        doc_index = get_doc_index_service()
        sync_result = doc_index.sync_docs()
        print(f"  同步完成: {sync_result}")

        elapsed = time.time() - start
        print(f"服务加载完成 ({elapsed:.1f}s)")
        print(f"  BM25 文档数: {self.bm25_service.get_document_count()}")
        print(f"  ChromaDB 向量数: {self.chroma_service.get_count()}")

    def _check_hit(self, result_ids: List[str], result_contents: List[str],
                   test_case: RetrievalTestCase) -> Tuple[bool, int]:
        """
        检查检索结果是否命中相关文档。

        判定逻辑：
        1. 结果的文件名在 relevant_filenames 中
        2. 或结果内容包含 relevant_doc_keywords 中的关键词

        Returns:
            (是否命中, 命中的相关文档数)
        """
        hit_count = 0

        for i, (doc_id, content) in enumerate(zip(result_ids, result_contents)):
            is_relevant = False

            # 检查文件名匹配
            for fname in test_case.relevant_filenames:
                stem = fname.replace(".md", "").replace(".txt", "")
                if stem in doc_id or doc_id.startswith(f"doc_{stem}"):
                    is_relevant = True
                    break

            # 检查关键词匹配（内容中包含 2+ 个关键词视为相关）
            if not is_relevant:
                keyword_hits = sum(1 for kw in test_case.relevant_doc_keywords if kw in content)
                if keyword_hits >= 2:
                    is_relevant = True

            if is_relevant:
                hit_count += 1

        return hit_count > 0, hit_count

    def _run_single_method(self, method_name: str, search_func, test_cases: List[RetrievalTestCase]) -> Dict[str, Any]:
        """运行单个检索方法的 benchmark"""
        hits = 0
        total_recall = 0.0
        total_precision = 0.0
        total_latency = 0.0
        details = []

        for tc in test_cases:
            # 计时
            start = time.time()
            try:
                raw_results = search_func(tc.query)
            except Exception as e:
                print(f"  [ERROR] {method_name} 查询失败: '{tc.query[:30]}...' - {e}")
                details.append({
                    "query": tc.query, "hit": False, "error": str(e), "latency": 0
                })
                continue
            latency = (time.time() - start) * 1000  # 毫秒

            # 提取结果 ID 和内容
            result_ids = [r.get("id", "") for r in raw_results]
            result_contents = [r.get("content", "") for r in raw_results]

            # 计算命中
            hit, hit_count = self._check_hit(result_ids, result_contents, tc)
            if hit:
                hits += 1

            # 精确率 = 命中数 / 返回数
            precision = hit_count / len(raw_results) if raw_results else 0
            # 召回率简化：假设相关文档总数 = len(relevant_filenames)，命中了 hit_count 个
            # 实际相关文档可能更多，这里用保守估计
            total_relevant = len(tc.relevant_filenames)
            recall = min(hit_count, total_relevant) / total_relevant if total_relevant > 0 else 0

            total_recall += recall
            total_precision += precision
            total_latency += latency

            detail = {
                "query": tc.query,
                "category": tc.category,
                "hit": hit,
                "hit_count": hit_count,
                "returned_count": len(raw_results),
                "precision": precision,
                "recall": recall,
                "latency_ms": latency,
                "top3_ids": result_ids[:3],
            }
            details.append(detail)

            if self.verbose:
                status = "HIT" if hit else "MISS"
                print(f"    [{status}] '{tc.query[:35]}...' "
                      f"P={precision:.2f} R={recall:.2f} {latency:.0f}ms "
                      f"→ {result_ids[:2]}")

        n = len(test_cases)
        return {
            "method": method_name,
            "hit_rate": hits / n if n > 0 else 0,
            "avg_precision": total_precision / n if n > 0 else 0,
            "avg_recall": total_recall / n if n > 0 else 0,
            "avg_latency_ms": total_latency / n if n > 0 else 0,
            "total_latency_ms": total_latency,
            "hits": hits,
            "total": n,
            "details": details,
        }

    def run(self) -> Dict[str, Any]:
        """运行完整 benchmark"""
        self._init_services()

        print(f"\n{'='*60}")
        print(f"  检索 Benchmark — top_k={self.top_k}, 测试用例={len(TEST_CASES)}")
        print(f"{'='*60}\n")

        from backend.app.core.database import SessionLocal
        db = SessionLocal()

        # 定义四种检索方法
        methods = {
            "BM25": lambda q: self._search_bm25(q, db),
            "Embedding": lambda q: self._search_embedding(q, db),
            "Hybrid (RRF)": lambda q: self._search_hybrid_no_rerank(q, db),
            "Hybrid+Reranker": lambda q: self._search_hybrid_full(q, db),
        }

        all_results = {}
        for method_name, search_func in methods.items():
            print(f"\n--- 测试方法: {method_name} ---")
            result = self._run_single_method(method_name, search_func, TEST_CASES)
            all_results[method_name] = result

            print(f"  命中率: {result['hit_rate']:.1%} ({result['hits']}/{result['total']})")
            print(f"  平均精确率: {result['avg_precision']:.2%}")
            print(f"  平均召回率: {result['avg_recall']:.2%}")
            print(f"  平均延迟: {result['avg_latency_ms']:.0f}ms")
            print(f"  总延迟: {result['total_latency_ms']:.0f}ms")

        db.close()

        # 按类别分析
        category_analysis = self._analyze_by_category(all_results)

        return {
            "benchmark": "retrieval",
            "timestamp": datetime.now().isoformat(),
            "top_k": self.top_k,
            "test_cases": len(TEST_CASES),
            "methods": {k: {kk: vv for kk, vv in v.items() if kk != "details"} for k, v in all_results.items()},
            "category_analysis": category_analysis,
            "details": {k: v["details"] for k, v in all_results.items()},
        }

    def _search_bm25(self, query: str, db) -> List[Dict]:
        """纯 BM25 检索"""
        results = self.hybrid_service.search_pure_bm25(query, top_k=self.top_k, db=db)
        return results

    def _search_embedding(self, query: str, db) -> List[Dict]:
        """纯 Embedding 检索"""
        results = self.hybrid_service.search_pure_embedding(query, top_k=self.top_k, db=db)
        return results

    def _search_hybrid_no_rerank(self, query: str, db) -> List[Dict]:
        """Hybrid BM25+Embedding+RRF（不含 Reranker）"""
        query_embedding = self.embedding_service.encode(query)

        bm25_results = self.bm25_service.search(query, top_k=20)
        embedding_results = self.chroma_service.search(query_embedding, top_k=20)

        if not bm25_results and not embedding_results:
            return []
        if not bm25_results:
            rrf_results = embedding_results[:self.top_k]
        elif not embedding_results:
            rrf_results = bm25_results[:self.top_k]
        else:
            rrf_results = self.rrf_service.fuse(bm25_results, embedding_results, top_k=self.top_k)

        # 获取详情
        return self._enrich_results(rrf_results, db)

    def _search_hybrid_full(self, query: str, db) -> List[Dict]:
        """完整 Hybrid 检索（BM25+Embedding+RRF+Reranker）"""
        results = self.hybrid_service.search_hybrid(query, top_k=self.top_k, db=db)
        return results

    def _enrich_results(self, rrf_results: List[Tuple[str, float]], db) -> List[Dict]:
        """将 RRF 结果转为带内容的字典列表"""
        from backend.app.models.note import Note

        enriched = []
        for doc_id, score in rrf_results:
            if doc_id.startswith("note_"):
                try:
                    note_id = int(doc_id.split("_", 1)[1])
                    note = db.query(Note).filter(Note.id == note_id).first()
                    if note:
                        enriched.append({
                            "id": doc_id,
                            "title": note.title or "",
                            "content": note.content or "",
                            "source_type": "note",
                        })
                except (ValueError, Exception):
                    pass
            elif doc_id.startswith("doc_"):
                # 从 BM25 corpus 获取，如果不在则查 ChromaDB
                content = self.bm25_service.corpus.get(doc_id, "")
                if not content:
                    try:
                        chunk_result = self.chroma_service.collection.get(
                            ids=[doc_id], include=["documents", "metadatas"]
                        )
                        if chunk_result["documents"]:
                            content = chunk_result["documents"][0]
                    except Exception:
                        pass
                enriched.append({
                    "id": doc_id,
                    "title": doc_id,
                    "content": content,
                    "source_type": "doc",
                })
        return enriched

    def _analyze_by_category(self, all_results: Dict) -> Dict:
        """按测试类别分析各方法表现"""
        categories = set(tc.category for tc in TEST_CASES)
        analysis = {}

        for cat in categories:
            cat_results = {}
            for method_name, result in all_results.items():
                cat_details = [d for d in result["details"] if d.get("category") == cat]
                if not cat_details:
                    continue
                cat_hits = sum(1 for d in cat_details if d.get("hit", False))
                cat_latencies = [d.get("latency_ms", 0) for d in cat_details]
                cat_results[method_name] = {
                    "hit_rate": cat_hits / len(cat_details),
                    "avg_latency_ms": sum(cat_latencies) / len(cat_latencies),
                    "count": len(cat_details),
                }
            analysis[cat] = cat_results

        return analysis


# ============================================================
# 输出格式化
# ============================================================

def print_summary(report: Dict[str, Any]):
    """打印格式化的 benchmark 结果摘要"""
    print(f"\n{'='*70}")
    print(f"  检索 Benchmark 结果摘要")
    print(f"  时间: {report['timestamp']}")
    print(f"  top_k: {report['top_k']}, 测试用例数: {report['test_cases']}")
    print(f"{'='*70}")

    # 主结果表
    methods = report["methods"]
    print(f"\n{'方法':<22} {'命中率':>8} {'精确率':>8} {'召回率':>8} {'平均延迟':>10}")
    print("-" * 60)
    for name, m in methods.items():
        print(f"{name:<22} {m['hit_rate']:>7.1%} {m['avg_precision']:>7.1%} "
              f"{m['avg_recall']:>7.1%} {m['avg_latency_ms']:>8.0f}ms")

    # 类别分析
    cat_analysis = report.get("category_analysis", {})
    if cat_analysis:
        print(f"\n{'='*60}")
        print("  按查询类别分析")
        print(f"{'='*60}")
        for cat, cat_data in cat_analysis.items():
            print(f"\n  [{cat}]")
            for method, stats in cat_data.items():
                print(f"    {method:<22} 命中率={stats['hit_rate']:.1%} "
                      f"延迟={stats['avg_latency_ms']:.0f}ms")

    # Miss 分析
    print(f"\n{'='*60}")
    print("  未命中用例分析")
    print(f"{'='*60}")
    for method_name, details in report.get("details", {}).items():
        misses = [d for d in details if not d.get("hit", False)]
        if misses:
            print(f"\n  {method_name} 未命中 {len(misses)} 个:")
            for d in misses:
                print(f"    - '{d['query'][:40]}...' (类别: {d.get('category', '?')})")


def save_report(report: Dict[str, Any], output_path: str):
    """保存 benchmark 结果到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="检索方法 Benchmark")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回的 top-k 数量")
    parser.add_argument("--verbose", action="store_true", help="显示每个查询的详细结果")
    parser.add_argument("--output", type=str, default="test/benchmark_results_retrieval.json",
                        help="结果输出路径")
    args = parser.parse_args()

    benchmark = RetrievalBenchmark(top_k=args.top_k, verbose=args.verbose)
    report = benchmark.run()

    print_summary(report)
    save_report(report, os.path.join(PROJECT_ROOT, args.output))


if __name__ == "__main__":
    main()
