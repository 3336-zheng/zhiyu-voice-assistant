"""通用检索评测数据契约与加载器。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class DatasetValidationError(ValueError):
    """评测数据格式或引用关系不合法。"""


@dataclass(frozen=True)
class CorpusDocument:
    """一个可检索单元，可以是文档、页面或 Chunk。"""

    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenQuery:
    """查询、期望动作及其证据标注。"""

    query_id: str
    query: str
    relevance: dict[str, float]
    reference_answer: str = ""
    reference_claims: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    category: str = "default"
    question_type: str = "default"
    expected_action: str = "answer"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationDataset:
    """经过完整性校验的评测数据集。"""

    name: str
    documents: tuple[CorpusDocument, ...]
    queries: tuple[GoldenQuery, ...]
    corpus_path: Path
    queries_path: Path


def _read_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 或 JSON 数组，并保留可定位的错误信息。"""
    if not path.is_file():
        raise DatasetValidationError(f"评测文件不存在: {path}")

    try:
        if path.suffix.lower() == ".jsonl":
            records = []
            for line_number, raw_line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                line = raw_line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DatasetValidationError(
                        f"{path}:{line_number} 必须是 JSON 对象"
                    )
                records.append(value)
            return records

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise DatasetValidationError(f"{path} 必须是 JSON 对象数组")
        return value
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"{path}:{exc.lineno} 包含无效 JSON"
        ) from None


def _non_empty_string(record: dict[str, Any], keys: Iterable[str], label: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    expected = " 或 ".join(keys)
    raise DatasetValidationError(f"{label} 缺少非空字段 {expected}")


def _metadata(record: dict[str, Any], label: str) -> dict[str, Any]:
    value = record.get("metadata", {})
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{label} 的 metadata 必须是对象")
    return value


def load_corpus(path: Path) -> tuple[CorpusDocument, ...]:
    """加载语料；兼容 id/doc_id 与 text/content 两组字段名。"""
    documents = []
    seen_ids = set()
    for index, record in enumerate(_read_records(path), start=1):
        label = f"语料第 {index} 条"
        doc_id = _non_empty_string(record, ("id", "doc_id"), label)
        text = _non_empty_string(record, ("text", "content"), label)
        if doc_id in seen_ids:
            raise DatasetValidationError(f"语料包含重复文档 ID: {doc_id}")
        seen_ids.add(doc_id)
        title = record.get("title", "")
        if not isinstance(title, str):
            raise DatasetValidationError(f"{label} 的 title 必须是字符串")
        documents.append(
            CorpusDocument(
                doc_id=doc_id,
                text=text,
                title=title.strip(),
                metadata=_metadata(record, label),
            )
        )
    if not documents:
        raise DatasetValidationError("评测语料不能为空")
    return tuple(documents)


def _parse_relevance(
    record: dict[str, Any],
    label: str,
    *,
    required: bool = True,
) -> dict[str, float]:
    graded = record.get("relevance")
    if graded is not None:
        if not isinstance(graded, dict) or (required and not graded):
            requirement = "非空对象" if required else "对象"
            raise DatasetValidationError(f"{label} 的 relevance 必须是{requirement}")
        relevance = {}
        for doc_id, grade in graded.items():
            if (
                not isinstance(doc_id, str)
                or not doc_id.strip()
                or isinstance(grade, bool)
                or not isinstance(grade, (int, float))
                or float(grade) <= 0
            ):
                raise DatasetValidationError(
                    f"{label} 的 relevance 必须是 文档ID -> 正数相关性"
                )
            relevance[doc_id.strip()] = float(grade)
        return relevance

    relevant_ids = record.get("relevant_doc_ids")
    if relevant_ids is None and not required:
        return {}
    if not isinstance(relevant_ids, list) or (
        required and not relevant_ids
    ) or not all(isinstance(item, str) and item.strip() for item in relevant_ids):
        requirement = "非空 relevant_doc_ids 数组或 relevance 对象" if required else (
            "relevant_doc_ids 数组或 relevance 对象"
        )
        raise DatasetValidationError(
            f"{label} 必须提供{requirement}"
        )
    return {doc_id.strip(): 1.0 for doc_id in relevant_ids}


def load_queries(path: Path) -> tuple[GoldenQuery, ...]:
    """加载统一 Question；无答案样本不允许伪造证据与相关文档。"""
    queries = []
    seen_ids = set()
    for index, record in enumerate(_read_records(path), start=1):
        label = f"查询第 {index} 条"
        query_id = _non_empty_string(record, ("id", "query_id"), label)
        query = _non_empty_string(record, ("query",), label)
        if query_id in seen_ids:
            raise DatasetValidationError(f"查询包含重复 ID: {query_id}")
        seen_ids.add(query_id)
        category = record.get("category", "default")
        if not isinstance(category, str) or not category.strip():
            raise DatasetValidationError(f"{label} 的 category 必须是非空字符串")
        question_type = record.get("question_type", category)
        if not isinstance(question_type, str) or not question_type.strip():
            raise DatasetValidationError(f"{label} 的 question_type 必须是非空字符串")
        expected_action = record.get("expected_action", "answer")
        allowed_actions = {"answer", "reject", "correct_premise", "external_research"}
        if expected_action not in allowed_actions:
            raise DatasetValidationError(f"{label} 的 expected_action 不受支持")
        reference_answer = record.get("reference_answer", "")
        if not isinstance(reference_answer, str):
            raise DatasetValidationError(f"{label} 的 reference_answer 必须是字符串")
        reference_claims = record.get("reference_claims", [])
        if not isinstance(reference_claims, list) or not all(
            isinstance(item, str) and item.strip() for item in reference_claims
        ):
            raise DatasetValidationError(f"{label} 的 reference_claims 必须是字符串数组")
        evidence = record.get("evidence", [])
        if not isinstance(evidence, list) or not all(
            isinstance(item, dict) for item in evidence
        ):
            raise DatasetValidationError(f"{label} 的 evidence 必须是对象数组")
        relevance = _parse_relevance(
            record,
            label,
            required=expected_action == "answer",
        )
        if expected_action != "answer" and (
            relevance or reference_answer.strip() or reference_claims or evidence
        ):
            raise DatasetValidationError(
                f"{label} 是无答案样本，不能包含 relevance、reference_answer 或 evidence"
            )
        queries.append(
            GoldenQuery(
                query_id=query_id,
                query=query,
                relevance=relevance,
                reference_answer=reference_answer.strip(),
                reference_claims=tuple(item.strip() for item in reference_claims),
                evidence=tuple(evidence),
                category=category.strip(),
                question_type=question_type.strip(),
                expected_action=expected_action,
                metadata=_metadata(record, label),
            )
        )
    if not queries:
        raise DatasetValidationError("Golden Query 不能为空")
    return tuple(queries)


def load_evaluation_dataset(
    corpus_path: Path,
    queries_path: Path,
    *,
    name: str | None = None,
) -> EvaluationDataset:
    """加载数据集并验证 Golden Query 引用的文档均存在。"""
    documents = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    document_ids = {document.doc_id for document in documents}
    missing = sorted(
        {
            doc_id
            for query in queries
            for doc_id in query.relevance
            if doc_id not in document_ids
        }
    )
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise DatasetValidationError(
            f"Golden Query 引用了不存在的文档: {preview}{suffix}"
        )

    dataset_name = (name or queries_path.stem).strip()
    if not dataset_name:
        raise DatasetValidationError("数据集名称不能为空")
    return EvaluationDataset(
        name=dataset_name,
        documents=documents,
        queries=queries,
        corpus_path=corpus_path.resolve(),
        queries_path=queries_path.resolve(),
    )
