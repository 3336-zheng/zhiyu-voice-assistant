"""上下文感知 Query Rewrite 与 CRAG 评分测试。"""

import unittest
from unittest.mock import patch

from backend.app.agent.graph import grade_node, query_rewrite_node, retrieve_node
from backend.app.agent.models import IntentType, Plan, PlanStep, ToolName
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
            "scores": [],
            "reasoning": "候选证据支持检索目标",
        }


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
    def test_crag_prompt_contains_resolved_intent_and_document_metadata(self):
        service = CRAGGraderService()
        llm = RecordingJSONLLM(
            {
                "grade": "correct",
                "scores": [{"doc_id": 0, "grade": "correct", "score": 0.93}],
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
        self.assertIn("max_score >= 0.70 -> correct", prompt)
        self.assertIn("0.30 < max_score < 0.70 -> ambiguous", prompt)
        self.assertIn("max_score <= 0.30 -> incorrect", prompt)
        self.assertEqual(result["max_score"], 0.93)

    def test_crag_uses_double_thresholds_and_ignores_model_grade(self):
        documents = [{"title": "证据", "content": "与问题相关"}]
        cases = [
            (0.70, RelevanceGrade.CORRECT),
            (0.50, RelevanceGrade.AMBIGUOUS),
            (0.30, RelevanceGrade.INCORRECT),
        ]

        for score, expected_grade in cases:
            with self.subTest(score=score):
                service = CRAGGraderService(upper_threshold=0.7, lower_threshold=0.3)
                llm = RecordingJSONLLM(
                    {
                        "grade": "incorrect",
                        "scores": [{"doc_id": 0, "score": score}],
                        "reasoning": "测试",
                    }
                )
                service._llm_service = llm

                result = service.grade_documents("问题", documents)

                self.assertEqual(result["grade"], expected_grade)
                self.assertEqual(result["max_score"], score)
                self.assertEqual(result["model_grade"], "incorrect")

    def test_crag_discards_invalid_and_out_of_range_scores(self):
        service = CRAGGraderService()
        service._llm_service = RecordingJSONLLM(
            {
                "grade": "correct",
                "scores": [
                    {"doc_id": 99, "score": 1.0},
                    {"doc_id": 0, "score": "not-a-number"},
                    {"doc_id": 0, "score": 1.2},
                ],
                "reasoning": "无效输出",
            }
        )

        result = service.grade_documents("问题", [{"content": "证据"}])

        self.assertEqual(result["grade"], RelevanceGrade.AMBIGUOUS)
        self.assertIsNone(result["max_score"])
        self.assertEqual(result["scores"], [])

    def test_grade_node_forwards_planner_semantics_to_crag(self):
        grader = RecordingGrader()
        state = make_state()
        state["retrieval_query"] = "Whisper 在中文方言语音识别上的表现如何？"

        with patch(
            "backend.app.agent.graph.get_crag_grader_service",
            return_value=grader,
        ):
            result = grade_node(state)

        self.assertEqual(result["relevance_grade"], "correct")
        self.assertEqual(grader.kwargs["retrieval_query"], state["retrieval_query"])
        self.assertEqual(grader.kwargs["goal"], state["plan"].goal)
        self.assertEqual(grader.kwargs["intent"], "search")


if __name__ == "__main__":
    unittest.main()
