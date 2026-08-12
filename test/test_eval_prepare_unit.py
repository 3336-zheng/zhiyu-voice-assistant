"""真实评测数据准备工具的确定性逻辑测试。"""

from collections import Counter

from backend.app.core.config import settings
from test.eval.prepare_dataset import (
    SourceDocument,
    _allocate_types,
    _evidence_spans,
    _select_generation_chunks,
    build_corpus,
)


def _source(path: str, domain: str, target: int, text: str) -> SourceDocument:
    return SourceDocument(
        path=path,
        domain=domain,
        document_type="note",
        target=target,
        title=path,
        raw_text=text,
        normalized_text=text,
        sha256="hash",
        normalized_sha256="normalized-hash",
    )


def test_build_corpus_keeps_parent_child_and_source_offsets(monkeypatch):
    monkeypatch.setattr(settings, "rag_parent_chunk_chars", 80)
    monkeypatch.setattr(settings, "rag_parent_chunk_overlap_chars", 10)
    monkeypatch.setattr(settings, "rag_child_chunk_chars", 30)
    monkeypatch.setattr(settings, "rag_child_chunk_overlap_chars", 5)
    text = "# 标题\n\n" + "第一段内容。" * 10 + "\n\n" + "第二段内容。" * 10

    corpus = build_corpus([_source("a.md", "rag", 1, text)])
    parents = [item for item in corpus if item["metadata"]["chunk_level"] == "parent"]
    children = [item for item in corpus if item["metadata"]["chunk_level"] == "child"]

    assert len(parents) >= 2
    assert children
    for item in corpus:
        metadata = item["metadata"]
        source_span = text[metadata["source_start_char"]:metadata["source_end_char"]]
        assert "".join(source_span.split()) == "".join(item["text"].split())
        assert metadata["parent_chunk_id"] in {parent["id"] for parent in parents}


def test_allocate_types_closes_document_and_domain_quotas():
    sources = [_source("a.md", "rag", 3, "正文"), _source("b.md", "rag", 2, "正文")]
    plan = {
        "domain_quotas": [
            {
                "domain": "rag",
                "keyword": 1,
                "semantic_rewrite": 2,
                "multi_evidence": 1,
                "similar_concept": 1,
                "unanswerable": 1,
            }
        ]
    }

    corpus = [
        {
            "id": f"p-{index}",
            "text": "正文",
            "metadata": {
                "source_path": path,
                "chunk_level": "parent",
            },
        }
        for index, path in enumerate(("a.md", "a.md", "b.md", "b.md"))
    ]
    allocation = _allocate_types(sources, plan, corpus)

    assert len(allocation["a.md"]) == 3
    assert len(allocation["b.md"]) == 2
    assert Counter(allocation["a.md"] + allocation["b.md"]) == {
        "keyword": 1,
        "semantic_rewrite": 2,
        "multi_evidence": 1,
        "similar_concept": 1,
    }


def test_generation_chunk_selection_is_bounded_and_spread():
    chunks = [
        {"id": f"p-{index}", "text": str(index) * 500}
        for index in range(30)
    ]

    selected = _select_generation_chunks(chunks, 4, character_budget=4_000)

    assert len(selected) >= 4
    assert sum(len(item["text"]) for item in selected) <= 4_000
    assert int(selected[-1]["id"].split("-")[1]) > int(selected[0]["id"].split("-")[1])


def test_evidence_spans_keep_exact_source_offsets():
    text = "第一段是可引用的事实。\n\n第二段也包含一个可验证事实。"
    chunk = {
        "id": "parent-1",
        "text": text,
        "metadata": {"source_start_char": 10},
    }

    spans = _evidence_spans(chunk, "0123456789" + text)

    assert len(spans) == 2
    for span in spans:
        start = span["source_start_char"] - 10
        end = span["source_end_char"] - 10
        assert text[start:end] == span["text"]
