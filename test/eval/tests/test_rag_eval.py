"""通用 RAG 评测的核心契约测试。"""

import json

import pytest

from test.eval.dataset import DatasetValidationError, load_evaluation_dataset
from test.eval.rag_eval import _evidence_metrics, run_evaluation
from test.eval.retrieval_metrics import evaluate_query, ndcg_at_k


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_dataset_accepts_binary_and_graded_relevance(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        corpus,
        [
            {"id": "d1", "text": "第一篇正文"},
            {"doc_id": "d2", "content": "第二篇正文"},
        ],
    )
    _write_jsonl(
        queries,
        [
            {
                "id": "q1",
                "query": "问题一",
                "relevance": {"d1": 3, "d2": 1},
                "reference_claims": ["事实一"],
            },
            {"query_id": "q2", "query": "问题二", "relevant_doc_ids": ["d2"]},
        ],
    )

    dataset = load_evaluation_dataset(corpus, queries, name="fixture")

    assert dataset.name == "fixture"
    assert dataset.queries[0].relevance == {"d1": 3.0, "d2": 1.0}
    assert dataset.queries[0].reference_claims == ("事实一",)
    assert dataset.queries[1].relevance == {"d2": 1.0}


def test_dataset_rejects_missing_relevant_document(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(corpus, [{"id": "d1", "text": "正文"}])
    _write_jsonl(
        queries,
        [{"id": "q1", "query": "问题", "relevant_doc_ids": ["missing"]}],
    )

    with pytest.raises(DatasetValidationError, match="不存在的文档"):
        load_evaluation_dataset(corpus, queries)


def test_dataset_accepts_unanswerable_question_without_fake_evidence(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(corpus, [{"id": "d1", "text": "正文"}])
    _write_jsonl(
        queries,
        [
            {
                "id": "q1",
                "query": "不存在的配置是什么？",
                "question_type": "unanswerable",
                "expected_action": "reject",
            }
        ],
    )

    dataset = load_evaluation_dataset(corpus, queries)

    assert dataset.queries[0].relevance == {}
    assert dataset.queries[0].expected_action == "reject"


def test_dataset_rejects_fake_evidence_on_unanswerable_question(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(corpus, [{"id": "d1", "text": "正文"}])
    _write_jsonl(
        queries,
        [
            {
                "id": "q1",
                "query": "不存在的配置是什么？",
                "expected_action": "reject",
                "relevant_doc_ids": ["d1"],
            }
        ],
    )

    with pytest.raises(DatasetValidationError, match="不能包含"):
        load_evaluation_dataset(corpus, queries)


def test_metrics_support_graded_relevance():
    relevance = {"d1": 3.0, "d2": 1.0}

    ideal = ndcg_at_k(["d1", "d2"], relevance, 2)
    reversed_order = ndcg_at_k(["d2", "d1"], relevance, 2)
    metrics = evaluate_query(["d1", "noise"], relevance, [1, 2])

    assert ideal == pytest.approx(1.0)
    assert reversed_order < ideal
    assert metrics["hit@1"] == 1.0
    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 0.5


class _FakeRetriever:
    def __init__(self):
        self.prepared = ()

    def prepare(self, methods):
        self.prepared = tuple(methods)

    def search(self, query, method, top_k):
        if method == "embedding":
            raise RuntimeError("模型服务暂时不可用")
        return ["d1", "d2"][:top_k]

    def model_snapshot(self):
        return {"embedding_model": "fake", "reranker_model": "fake"}

    def usage_snapshot(self):
        return {"embedding_query_calls": 0}


def test_report_keeps_failure_and_configuration(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        corpus,
        [{"id": "d1", "text": "正文一"}, {"id": "d2", "text": "正文二"}],
    )
    _write_jsonl(
        queries,
        [{"id": "q1", "query": "问题", "relevant_doc_ids": ["d1"]}],
    )
    dataset = load_evaluation_dataset(corpus, queries, name="fixture")
    retriever = _FakeRetriever()

    report = run_evaluation(
        dataset,
        methods=("bm25", "embedding"),
        top_k=2,
        k_values=(1, 2),
        retriever=retriever,
    )

    assert retriever.prepared == ("bm25", "embedding")
    assert report["methods"]["bm25"]["metrics"]["hit@1"] == 1.0
    assert report["methods"]["embedding"]["failure_rate"] == 1.0
    assert report["methods"]["embedding"]["metrics"]["hit@1"] == 0.0
    assert report["configuration"]["retrieval"]["evaluation_top_k"] == 2
    assert report["index_setup"]["status"] == "succeeded"
    assert report["model_workload"]["embedding_query_calls"] == 0


class _FailedIndexRetriever(_FakeRetriever):
    def prepare(self, methods):
        raise RuntimeError("索引服务不可用")


def test_index_failure_does_not_hide_bm25_results(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(corpus, [{"id": "d1", "text": "正文"}])
    _write_jsonl(
        queries,
        [{"id": "q1", "query": "问题", "relevant_doc_ids": ["d1"]}],
    )
    dataset = load_evaluation_dataset(corpus, queries)

    report = run_evaluation(
        dataset,
        methods=("bm25", "embedding"),
        top_k=1,
        k_values=(1,),
        retriever=_FailedIndexRetriever(),
    )

    assert report["index_setup"]["status"] == "failed"
    assert report["methods"]["bm25"]["failure_rate"] == 0.0
    assert report["methods"]["embedding"]["failure_rate"] == 1.0


def test_evidence_metrics_use_exact_source_span(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "id": "parent-1",
                "text": "abcdefghij",
                "metadata": {
                    "source_path": "source.md",
                    "source_start_char": 0,
                    "source_end_char": 10,
                },
            },
            {
                "id": "parent-2",
                "text": "klmnopqrst",
                "metadata": {
                    "source_path": "source.md",
                    "source_start_char": 10,
                    "source_end_char": 20,
                },
            },
        ],
    )
    _write_jsonl(
        queries,
        [{"id": "q1", "query": "问题", "relevant_doc_ids": ["parent-1"]}],
    )
    dataset = load_evaluation_dataset(corpus, queries)

    metrics = _evidence_metrics(
        dataset,
        ["parent-1", "parent-2"],
        [{"source_id": "source.md", "start_char": 5, "end_char": 15}],
        (1, 2),
    )

    assert metrics["evidence_recall@1"] == 1.0
    assert metrics["evidence_coverage@1"] == 0.5
    assert metrics["evidence_coverage@2"] == 1.0


def test_default_retrieval_settings_match_fixed_engineering_profile():
    from backend.app.core.config import settings

    assert settings.rag_parent_chunk_chars == 1200
    assert settings.rag_parent_chunk_overlap_chars == 120
    assert settings.rag_child_chunk_chars == 500
    assert settings.rag_child_chunk_overlap_chars == 80
    assert settings.bm25_top_k == 20
    assert settings.embedding_top_k == 20
    assert settings.rrf_top_k == 30
    assert settings.rag_final_top_k == 5
    assert settings.retrieval_rerank_min_score == 0.35
    assert settings.retrieval_rerank_score_margin == 0.20
