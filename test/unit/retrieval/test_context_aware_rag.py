"""上下文感知 Query Rewrite 与 CRAG 评分测试。"""

import unittest
from unittest.mock import patch

from backend.app.agent.graph import (
    generate_node,
    grade_node,
    query_rewrite_node,
    retrieve_node,
)
from backend.app.agent.models import IntentType, Plan, PlanStep, ToolName
from backend.app.agent.responder import RESPONSE_GENERATION_PROMPT
from backend.app.services.retrieval.crag_grader_service import (
    CRAGGraderService,
    RelevanceGrade,
)
from backend.app.services.retrieval.query_rewrite_service import QueryRewriteService


class RecordingJSONLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class RecordingStructuredLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def structured_call(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.payload


class RecordingRewriteService:
    def __init__(self):
        self.kwargs = None

    def rewrite_query(self, query, **kwargs):
        self.kwargs = {"query": query, **kwargs}
        return [
            "Whisper 在中文方言语音识别上的表现如何？",
            "Whisper 方言识别准确率",
        ]


class RecordingHybridRetrieval:
    def __init__(self):
        self.kwargs = None

    def search_multi(self, queries, **kwargs):
        self.kwargs = {"queries": queries, **kwargs}
        return {"results": [], "stats": {"query_count": len(queries)}}


class RecordingExecutor:
    def __init__(self):
        self.hybrid_retrieval = RecordingHybridRetrieval()


class RecordingGrader:
    def __init__(self):
        self.kwargs = None

    def grade_documents(self, query, documents, **kwargs):
        self.kwargs = {"query": query, "documents": documents, **kwargs}
        return {
            "grade": RelevanceGrade.CORRECT,
            "scores": [
                {
                    "doc_id": 0,
                    "score": 0.93,
                    "verdict": "support",
                    "reason": "候选证据直接支持检索目标",
                }
            ],
            "reasoning": "候选证据支持检索目标",
            "max_score": 0.93,
            "coverage": "complete",
            "support_doc_ids": [0],
            "limited_support_doc_ids": [],
            "incorrect_doc_ids": [],
            "upper_threshold": 0.7,
            "lower_threshold": 0.3,
            "grading_failed": False,
        }


class StaticGrader:
    def __init__(self, payload):
        self.payload = payload
        self.refine_inputs = []

    def grade_documents(self, query, documents, **kwargs):
        return self.payload

    def refine_documents(self, query, documents, **kwargs):
        self.refine_inputs.append(documents)
        return "support 已确认核心事实；limited_support 仅补充未完整说明的边界。"


def make_plan():
    return Plan(
        goal="评估 Whisper 在中文方言语音识别上的表现",
        intent=IntentType.SEARCH,
        original_query="那它在方言上怎么样？",
        steps=[
            PlanStep(
                step_id=1,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                parameters={"query": "那它在方言上怎么样？", "top_k": 5},
                description="检索知识库",
            )
        ],
        estimated_steps=1,
        reasoning="结合上文识别 Whisper",
    )


def make_state():
    return {
        "query": "那它在方言上怎么样？",
        "context": [
            {"role": "user", "content": "Whisper 和 FunASR 有什么区别？"},
            {"role": "assistant", "content": "两者的架构和适用场景不同。"},
        ],
        "plan": make_plan(),
        "iter_count": 0,
        "max_iterations": 5,
        "search_results": [],
    }


class ContextAwareRewriteTestCase(unittest.TestCase):
    def test_rewrite_cache_reuses_non_contextual_query_and_refresh_can_bypass(self):
        service = QueryRewriteService()
        llm = RecordingJSONLLM(
            {
                "standalone_query": "SQLite 在智语中的职责是什么？",
                "queries": ["SQLite 主数据", "SQLite 运行记录"],
            }
        )
        service._llm_service = llm

        first = service.rewrite_query("SQLite 在智语中负责什么？")
        second = service.rewrite_query("SQLite 在智语中负责什么？")
        refreshed = service.rewrite_query(
            "SQLite 在智语中负责什么？",
            force_refresh=True,
        )

        self.assertEqual(first, second)
        self.assertEqual(refreshed, first)
        self.assertEqual(len(llm.calls), 2)

    def test_rewrite_prompt_contains_history_goal_and_returns_standalone_query(self):
        service = QueryRewriteService()
        llm = RecordingJSONLLM(
            {
                "standalone_query": "Whisper 在中文方言语音识别上的表现如何？",
                "queries": ["Whisper 方言识别准确率", "Whisper 中文方言 ASR"],
            }
        )
        service._llm_service = llm

        queries = service.rewrite_query(
            "那它在方言上怎么样？",
            context=make_state()["context"],
            goal=make_plan().goal,
            intent="search",
        )

        self.assertEqual(queries[0], "Whisper 在中文方言语音识别上的表现如何？")
        messages = llm.calls[0]["messages"]
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("Whisper 和 FunASR 有什么区别", prompt)
        self.assertIn("Planner 解析目标：评估 Whisper", prompt)
        self.assertEqual(llm.calls[0]["max_tokens"], 300)

    def test_rewrite_node_uses_standalone_query_for_retrieval_and_rerank(self):
        rewrite_service = RecordingRewriteService()
        executor = RecordingExecutor()
        state = make_state()

        with patch(
            "backend.app.agent.graph.get_query_rewrite_service",
            return_value=rewrite_service,
        ):
            rewritten_state = query_rewrite_node(state)
        with patch("backend.app.agent.graph.get_executor", return_value=executor):
            retrieve_node(rewritten_state)

        self.assertEqual(
            rewritten_state["retrieval_query"],
            "Whisper 在中文方言语音识别上的表现如何？",
        )
        self.assertEqual(rewrite_service.kwargs["context"], state["context"])
        self.assertEqual(
            executor.hybrid_retrieval.kwargs["original_query"],
            rewritten_state["retrieval_query"],
        )


class ContextAwareCRAGTestCase(unittest.TestCase):
    def test_responder_prompt_forbids_inference_beyond_evidence(self):
        self.assertIn("当前证据未说明", RESPONSE_GENERATION_PROMPT)
        self.assertIn("禁止使用“可能”“推测”“暗示”", RESPONSE_GENERATION_PROMPT)

    def test_crag_prompt_contains_resolved_intent_and_document_metadata(self):
        service = CRAGGraderService()
        llm = RecordingStructuredLLM(
            {
                "grade": "correct",
                "coverage": "complete",
                "scores": [
                    {"doc_id": 0, "score": 0.93, "reason": "正文直接支持"}
                ],
                "reasoning": "文档直接给出了方言识别实验结果",
            }
        )
        service._llm_service = llm

        result = service.grade_documents(
            "那它在方言上怎么样？",
            [
                {
                    "title": "Whisper 潮汕话微调实验",
                    "source_url": "/api/pages/12",
                    "content": "Whisper-small 经过 LoRA 微调后，潮汕话 CER 明显下降。",
                    "rerank_score": 0.91,
                }
            ],
            retrieval_query="Whisper 在中文方言语音识别上的表现如何？",
            goal="评估 Whisper 在中文方言语音识别上的表现",
            intent="search",
        )

        self.assertEqual(result["grade"], RelevanceGrade.CORRECT)
        prompt = "\n".join(
            message["content"] for message in llm.calls[0]["messages"]
        )
        self.assertIn("用户原始问题：那它在方言上怎么样", prompt)
        self.assertIn("消歧后的独立查询：Whisper", prompt)
        self.assertIn("Planner 结构化意图：search", prompt)
        self.assertIn("标题：Whisper 潮汕话微调实验", prompt)
        self.assertIn("来源：/api/pages/12", prompt)
        self.assertIn("Rerank 分数：0.91", prompt)
        self.assertIn("必须为每个候选文档恰好输出一次评分", prompt)
        self.assertIn("coverage", prompt)
        scores_schema = llm.calls[0]["parameters"]["properties"]["scores"]
        self.assertEqual(scores_schema["minItems"], 1)
        self.assertEqual(scores_schema["maxItems"], 1)
        self.assertEqual(result["max_score"], 0.93)
        self.assertEqual(result["support_doc_ids"], [0])

    def test_crag_uses_double_thresholds_coverage_and_ignores_model_grade(self):
        documents = [{"title": "证据", "content": "与问题相关"}]
        cases = [
            (0.70, "complete", RelevanceGrade.CORRECT, "support"),
            (0.50, "incomplete", RelevanceGrade.AMBIGUOUS, "limited_support"),
            (0.30, "none", RelevanceGrade.INCORRECT, "incorrect"),
        ]

        for score, coverage, expected_grade, expected_verdict in cases:
            with self.subTest(score=score):
                service = CRAGGraderService(upper_threshold=0.7, lower_threshold=0.3)
                llm = RecordingJSONLLM(
                    {
                        "grade": "incorrect",
                        "coverage": coverage,
                        "scores": [
                            {"doc_id": 0, "score": score, "reason": "测试"}
                        ],
                        "reasoning": "测试",
                    }
                )
                service._llm_service = llm

                result = service.grade_documents("问题", documents)

                self.assertEqual(result["grade"], expected_grade)
                self.assertEqual(result["max_score"], score)
                self.assertEqual(result["model_grade"], "incorrect")
                self.assertEqual(result["scores"][0]["verdict"], expected_verdict)

    def test_three_support_two_limited_support_use_coverage_for_overall_grade(self):
        documents = [
            {"title": f"证据 {index}", "content": f"正文 {index}"}
            for index in range(5)
        ]
        score_values = [0.90, 0.82, 0.74, 0.62, 0.45]

        for coverage, expected_grade in (
            ("complete", RelevanceGrade.CORRECT),
            ("incomplete", RelevanceGrade.AMBIGUOUS),
        ):
            with self.subTest(coverage=coverage):
                service = CRAGGraderService()
                service._llm_service = RecordingJSONLLM(
                    {
                        "grade": "correct",
                        "coverage": coverage,
                        "scores": [
                            {
                                "doc_id": index,
                                "score": score,
                                "reason": "测试证据",
                            }
                            for index, score in enumerate(score_values)
                        ],
                        "reasoning": "三个直接证据，两个部分证据",
                    }
                )

                result = service.grade_documents("复杂问题", documents)

                self.assertEqual(result["grade"], expected_grade)
                self.assertEqual(result["support_doc_ids"], [0, 1, 2])
                self.assertEqual(result["limited_support_doc_ids"], [3, 4])
                self.assertEqual(result["incorrect_doc_ids"], [])

    def test_crag_discards_invalid_and_out_of_range_scores(self):
        service = CRAGGraderService()
        service._llm_service = RecordingJSONLLM(
            {
                "grade": "correct",
                "coverage": "complete",
                "scores": [
                    {"doc_id": 99, "score": 1.0, "reason": "越界"},
                    {"doc_id": 0, "score": "not-a-number", "reason": "非法"},
                    {"doc_id": 0, "score": 1.2, "reason": "重复"},
                ],
                "reasoning": "无效输出",
            }
        )

        result = service.grade_documents("问题", [{"content": "证据"}])

        # 没有任何有效证据分数时必须 fail closed，禁止继续生成。
        self.assertEqual(result["grade"], RelevanceGrade.INCORRECT)
        self.assertIsNone(result["max_score"])
        self.assertEqual(result["scores"], [])
        self.assertTrue(result["grading_failed"])

    def test_crag_fails_closed_when_any_candidate_score_is_missing(self):
        service = CRAGGraderService()
        service._llm_service = RecordingJSONLLM(
            {
                "grade": "correct",
                "coverage": "complete",
                "scores": [
                    {"doc_id": 0, "score": 0.9, "reason": "只返回一个候选"}
                ],
                "reasoning": "漏掉第二个候选",
            }
        )

        result = service.grade_documents(
            "问题",
            [{"content": "证据一"}, {"content": "证据二"}],
        )

        self.assertEqual(result["grade"], RelevanceGrade.INCORRECT)
        self.assertTrue(result["grading_failed"])

    def test_grade_node_forwards_planner_semantics_to_crag(self):
        grader = RecordingGrader()
        state = make_state()
        state["retrieval_query"] = "Whisper 在中文方言语音识别上的表现如何？"
        state["search_results"] = [
            {
                "page_id": "whisper-page",
                "title": "Whisper 实验",
                "content": "Whisper 方言实验结果。",
                "rerank_score": 0.91,
            }
        ]

        with patch(
            "backend.app.agent.graph.get_crag_grader_service",
            return_value=grader,
        ):
            result = grade_node(state)

        self.assertEqual(result["relevance_grade"], "correct")
        self.assertEqual(grader.kwargs["retrieval_query"], state["retrieval_query"])
        self.assertEqual(grader.kwargs["goal"], state["plan"].goal)
        self.assertEqual(grader.kwargs["intent"], "search")
        self.assertEqual(result["search_results"][0]["crag_verdict"], "support")

    def test_grade_node_only_injects_support_when_coverage_is_complete(self):
        scores = [
            {"doc_id": 0, "score": 0.91, "verdict": "support", "reason": "直接"},
            {"doc_id": 1, "score": 0.82, "verdict": "support", "reason": "直接"},
            {"doc_id": 2, "score": 0.74, "verdict": "support", "reason": "直接"},
            {
                "doc_id": 3,
                "score": 0.60,
                "verdict": "limited_support",
                "reason": "部分",
            },
            {"doc_id": 4, "score": 0.20, "verdict": "incorrect", "reason": "无关"},
        ]
        grader = StaticGrader(
            {
                "grade": RelevanceGrade.CORRECT,
                "scores": scores,
                "reasoning": "support 已完整覆盖",
                "max_score": 0.91,
                "coverage": "complete",
                "upper_threshold": 0.7,
                "lower_threshold": 0.3,
                "grading_failed": False,
            }
        )
        state = make_state()
        state["search_results"] = [
            {
                "page_id": f"page-{index}",
                "title": f"文档 {index}",
                "content": f"正文 {index}",
                "rerank_score": 0.9 - index * 0.05,
            }
            for index in range(5)
        ]

        with patch(
            "backend.app.agent.graph.get_crag_grader_service",
            return_value=grader,
        ):
            result = grade_node(state)

        self.assertEqual(
            [item["page_id"] for item in result["search_results"]],
            ["page-0", "page-1", "page-2"],
        )
        self.assertEqual(result["crag_support_count"], 3)
        self.assertEqual(result["crag_limited_support_count"], 1)
        self.assertEqual(result["crag_incorrect_count"], 1)
        self.assertEqual(grader.refine_inputs, [])

    def test_grade_node_refines_limited_support_when_coverage_is_incomplete(self):
        scores = [
            {"doc_id": 0, "score": 0.91, "verdict": "support", "reason": "直接"},
            {"doc_id": 1, "score": 0.82, "verdict": "support", "reason": "直接"},
            {"doc_id": 2, "score": 0.74, "verdict": "support", "reason": "直接"},
            {
                "doc_id": 3,
                "score": 0.60,
                "verdict": "limited_support",
                "reason": "部分",
            },
            {
                "doc_id": 4,
                "score": 0.45,
                "verdict": "limited_support",
                "reason": "部分",
            },
        ]
        grader = StaticGrader(
            {
                "grade": RelevanceGrade.AMBIGUOUS,
                "scores": scores,
                "reasoning": "support 只覆盖部分问题",
                "max_score": 0.91,
                "coverage": "incomplete",
                "upper_threshold": 0.7,
                "lower_threshold": 0.3,
                "grading_failed": False,
            }
        )
        state = make_state()
        state["search_results"] = [
            {
                "page_id": f"page-{index}",
                "title": f"文档 {index}",
                "content": f"正文 {index}",
                "rerank_score": 0.9 - index * 0.05,
            }
            for index in range(5)
        ]

        with patch(
            "backend.app.agent.graph.get_crag_grader_service",
            return_value=grader,
        ):
            result = grade_node(state)

        self.assertEqual(len(result["search_results"]), 5)
        self.assertEqual(len(grader.refine_inputs), 1)
        self.assertEqual(
            [item["crag_verdict"] for item in grader.refine_inputs[0]],
            [
                "support",
                "support",
                "support",
                "limited_support",
                "limited_support",
            ],
        )
        self.assertIn("limited_support 仅补充", result["refined_content"])

    def test_insufficient_evidence_returns_without_calling_responder(self):
        class FailingResponder:
            @staticmethod
            def generate_response(*args, **kwargs):
                raise AssertionError("证据不足时不应调用生成模型")

        state = make_state()
        state["search_results"] = [
            {
                "page_id": "unrelated-page",
                "title": "无关页面",
                "content": "这段正文与用户问题无关。",
                "rerank_score": 0.1,
            }
        ]

        with patch(
            "backend.app.agent.graph.get_responder",
            return_value=FailingResponder(),
        ):
            result = generate_node(state)

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("没有足够证据", result["answer"])


if __name__ == "__main__":
    unittest.main()
