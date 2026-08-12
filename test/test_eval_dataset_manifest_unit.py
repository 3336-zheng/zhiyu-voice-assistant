"""真实文档来源清单与统一评测集配额测试。"""

import json

import pytest

from test.eval.dataset_manifest import ManifestValidationError, audit_dataset_plan


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "rag.md").write_text("# RAG\n\n混合检索结合关键词与语义召回。", encoding="utf-8")
    manifest = tmp_path / "sources.json"
    plan = tmp_path / "plan.json"
    _write_json(
        manifest,
        {
            "dataset_name": "source-v2",
            "documents": [
                {
                    "path": "rag.md",
                    "domain": "rag",
                    "document_type": "technical_note",
                    "positive_question_target": 4,
                }
            ],
        },
    )
    _write_json(
        plan,
        {
            "dataset_name": "eval-v2",
            "total_questions": 5,
            "answerable_questions": 4,
            "unanswerable_questions": 1,
            "question_types": {
                "keyword": {"count": 1},
                "semantic_rewrite": {"count": 1},
                "multi_evidence": {"count": 1},
                "similar_concept": {"count": 1},
                "unanswerable": {"count": 1},
            },
            "domain_quotas": [
                {
                    "domain": "rag",
                    "keyword": 1,
                    "semantic_rewrite": 1,
                    "multi_evidence": 1,
                    "similar_concept": 1,
                    "unanswerable": 1,
                }
            ],
        },
    )
    return source_root, manifest, plan


def test_audit_dataset_plan_accepts_closed_quotas(tmp_path):
    source_root, manifest, plan = _fixture(tmp_path)

    report = audit_dataset_plan(
        source_root=source_root,
        source_manifest_path=manifest,
        question_plan_path=plan,
    )

    assert report["status"] == "passed"
    assert report["source_documents"] == 1
    assert report["answerable_questions"] == 4
    assert report["unanswerable_questions"] == 1


def test_audit_dataset_plan_rejects_sensitive_source(tmp_path):
    source_root, manifest, plan = _fixture(tmp_path)
    (source_root / "rag.md").write_text(
        "API_KEY=abcdefghijklmnop",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="敏感值"):
        audit_dataset_plan(
            source_root=source_root,
            source_manifest_path=manifest,
            question_plan_path=plan,
        )


def test_audit_dataset_plan_rejects_quota_mismatch(tmp_path):
    source_root, manifest, plan = _fixture(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["total_questions"] = 6
    _write_json(plan, payload)

    with pytest.raises(ManifestValidationError, match="total_questions"):
        audit_dataset_plan(
            source_root=source_root,
            source_manifest_path=manifest,
            question_plan_path=plan,
        )
