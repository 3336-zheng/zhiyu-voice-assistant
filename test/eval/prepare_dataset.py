"""从真实 Markdown 准备智语评测语料、Golden Question 与 Wiki 索引。"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.services.doc_index_service import (
    clean_markdown_for_chunking,
    split_markdown_by_headers,
)
from backend.app.services.llm_service import get_llm_service
from backend.app.services.page_index_service import split_parent_into_children
from backend.app.services.page_service import get_page_service
from test.eval.dataset import load_evaluation_dataset
from test.eval.dataset_manifest import (
    DEFAULT_QUESTION_PLAN,
    DEFAULT_SOURCE_MANIFEST,
    ManifestValidationError,
    audit_dataset_plan,
)


DEFAULT_OUTPUT_DIR = Path("data/eval/zhiyu-v2")
ANSWERABLE_TYPES = ("keyword", "semantic_rewrite", "multi_evidence", "similar_concept")
EXPECTED_NEGATIVE_ACTIONS = {"reject", "correct_premise", "external_research"}
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class DatasetPreparationError(RuntimeError):
    """真实评测数据无法安全生成或校验。"""


@dataclass(frozen=True)
class SourceDocument:
    path: str
    domain: str
    document_type: str
    target: int
    title: str
    raw_text: str
    normalized_text: str
    sha256: str
    normalized_sha256: str


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetPreparationError(f"JSON 根节点必须是对象: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _title_from_markdown(path: str, content: str) -> str:
    match = HEADER_PATTERN.search(content)
    return match.group(2).strip() if match else Path(path).stem


def load_sources(source_root: Path, manifest_path: Path) -> list[SourceDocument]:
    manifest = _load_json(manifest_path)
    sources = []
    for item in manifest["documents"]:
        relative_path = item["path"]
        raw_text = (source_root / relative_path).read_text(encoding="utf-8")
        normalized_text = clean_markdown_for_chunking(raw_text)
        sources.append(
            SourceDocument(
                path=relative_path,
                domain=item["domain"],
                document_type=item["document_type"],
                target=int(item["positive_question_target"]),
                title=_title_from_markdown(relative_path, raw_text),
                raw_text=raw_text,
                normalized_text=normalized_text,
                sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                normalized_sha256=hashlib.sha256(
                    normalized_text.encode("utf-8")
                ).hexdigest(),
            )
        )
    return sources


def _canonical_with_positions(text: str) -> tuple[str, list[int]]:
    canonical = []
    positions = []
    in_whitespace = False
    for index, character in enumerate(text):
        if character.isspace():
            if not in_whitespace:
                canonical.append(" ")
                positions.append(index)
            in_whitespace = True
            continue
        canonical.append(character)
        positions.append(index)
        in_whitespace = False
    return "".join(canonical).strip(), positions


def _locate_chunk_span(source_text: str, chunk_text: str, hint: int = 0) -> tuple[int, int]:
    direct = source_text.find(chunk_text, max(0, hint))
    if direct < 0:
        direct = source_text.find(chunk_text)
    if direct >= 0:
        return direct, direct + len(chunk_text)

    canonical_source, positions = _canonical_with_positions(source_text)
    canonical_chunk, _ = _canonical_with_positions(chunk_text)
    canonical_hint = bisect.bisect_left(positions, max(0, hint))
    start = canonical_source.find(canonical_chunk, canonical_hint)
    if start < 0:
        start = canonical_source.find(canonical_chunk)
    if start < 0:
        raise DatasetPreparationError("生产分块不能映射到规范化原文")
    end = start + len(canonical_chunk)
    return positions[start], positions[end - 1] + 1


def build_corpus(sources: Sequence[SourceDocument]) -> list[dict[str, Any]]:
    """复用生产分块器生成语料，并补充可回查的字符区间。"""
    records = []
    for source in sources:
        search_start = 0
        parents = split_markdown_by_headers(source.normalized_text, source.path)
        for parent_index, chunk in enumerate(parents):
            text = chunk["text"]
            try:
                actual_start, actual_end = _locate_chunk_span(
                    source.normalized_text,
                    text,
                    search_start,
                )
            except DatasetPreparationError as exc:
                raise DatasetPreparationError(
                    f"生产分块不能回查到规范化原文: {source.path}#{parent_index}"
                ) from exc
            search_start = actual_start + 1
            parent_id = _stable_id("parent", source.path, str(parent_index), text)
            metadata = {
                "source_path": source.path,
                "source_sha256": source.normalized_sha256,
                "raw_source_sha256": source.sha256,
                "source_start_char": actual_start,
                "source_end_char": actual_end,
                "domain": source.domain,
                "document_type": source.document_type,
                "section_path": chunk.get("section_title") or source.title,
                "chunk_level": "parent",
                "parent_chunk_id": parent_id,
                "chunk_index": parent_index,
            }
            records.append(
                {"id": parent_id, "title": source.title, "text": text, "metadata": metadata}
            )
            children = split_parent_into_children(
                text,
                settings.rag_child_chunk_chars,
                settings.rag_child_chunk_overlap_chars,
            )
            child_hint = actual_start
            for child_index, child in enumerate(children):
                child_start, child_end = _locate_chunk_span(
                    source.normalized_text,
                    child,
                    max(actual_start, child_hint - settings.rag_child_chunk_overlap_chars),
                )
                child_hint = child_start + 1
                child_id = f"{parent_id}:child:{child_index}"
                records.append(
                    {
                        "id": child_id,
                        "title": source.title,
                        "text": child,
                        "metadata": {
                            **metadata,
                            "source_start_char": child_start,
                            "source_end_char": child_end,
                            "chunk_level": "child",
                            "parent_chunk_id": parent_id,
                            "child_index": child_index,
                        },
                    }
                )
    return records


def _select_generation_chunks(
    chunks: Sequence[dict[str, Any]],
    question_count: int,
    *,
    character_budget: int = 12_000,
) -> list[dict[str, Any]]:
    """按文档顺序均匀选择信息量足够的父块，控制 LLM 输入规模。"""
    eligible = [item for item in chunks if len(item["text"].strip()) >= 80]
    if len(eligible) < question_count:
        eligible = list(chunks)
    target = min(len(eligible), max(question_count * 3, question_count + 2))
    if target <= 0:
        return []
    selected = []
    seen = set()
    for index in range(target):
        position = min(len(eligible) - 1, int((index + 0.5) * len(eligible) / target))
        item = eligible[position]
        if item["id"] not in seen:
            selected.append(item)
            seen.add(item["id"])
    while sum(len(item["text"]) for item in selected) > character_budget and len(selected) > question_count:
        selected.pop()
    return selected


def _evidence_spans(
    chunk: dict[str, Any],
    source_text: str,
    max_chars: int = 420,
) -> list[dict[str, Any]]:
    """把父块切成可由模型选择、但内容由代码保真的证据片段。"""
    text = chunk["text"]
    segments = [item for item in re.split(r"\n\s*\n", text) if item.strip()]
    spans = []
    search_start = 0
    for segment in segments:
        stripped = segment.strip()
        segment_start = text.find(stripped, search_start)
        if segment_start < 0:
            segment_start = text.find(stripped)
        if segment_start < 0:
            continue
        pieces = []
        if len(stripped) <= max_chars:
            pieces = [(0, len(stripped))]
        else:
            piece_start = 0
            while piece_start < len(stripped):
                hard_end = min(piece_start + max_chars, len(stripped))
                split_at = hard_end
                if hard_end < len(stripped):
                    sentence = max(
                        stripped.rfind(mark, piece_start + max_chars // 2, hard_end + 1)
                        for mark in ("。", "！", "？", ".", "!", "?", "\n")
                    )
                    if sentence >= piece_start:
                        split_at = sentence + 1
                pieces.append((piece_start, split_at))
                piece_start = split_at
        for piece_start, piece_end in pieces:
            piece = stripped[piece_start:piece_end].strip()
            if len(piece) < 8:
                continue
            relative_start = text.find(piece, segment_start + piece_start)
            if relative_start < 0:
                continue
            source_hint = chunk["metadata"]["source_start_char"] + relative_start
            try:
                source_start, source_end = _locate_chunk_span(
                    source_text,
                    piece,
                    source_hint,
                )
            except DatasetPreparationError:
                continue
            exact_quote = source_text[source_start:source_end]
            span_index = len(spans)
            spans.append(
                {
                    "span_id": f"{chunk['id']}:span:{span_index}",
                    "chunk_id": chunk["id"],
                    "text": exact_quote,
                    "source_start_char": source_start,
                    "source_end_char": source_end,
                }
            )
        search_start = segment_start + len(stripped)
    return spans


def _allocate_types(
    sources: Sequence[SourceDocument],
    plan: dict[str, Any],
    corpus: Sequence[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """将领域级类型配额稳定分配到每篇文档的正例配额。"""
    quotas = {item["domain"]: item for item in plan["domain_quotas"]}
    parent_counts = Counter(
        item["metadata"]["source_path"]
        for item in (corpus or ())
        if item["metadata"]["chunk_level"] == "parent"
    )
    allocation: dict[str, list[str]] = {}
    for domain in quotas:
        domain_sources = [source for source in sources if source.domain == domain]
        remaining = {
            question_type: int(quotas[domain][question_type])
            for question_type in ANSWERABLE_TYPES
        }
        if sum(source.target for source in domain_sources) != sum(remaining.values()):
            raise DatasetPreparationError(f"领域 {domain} 的来源与类型配额不一致")
        buckets: dict[str, list[str]] = defaultdict(list)
        slots = []
        for round_index in range(max(source.target for source in domain_sources)):
            slots.extend(source.path for source in domain_sources if source.target > round_index)
        for source_path in slots:
            candidates = [name for name, count in remaining.items() if count > 0]
            if parent_counts and parent_counts[source_path] < 2:
                candidates = [name for name in candidates if name != "multi_evidence"]
            if not candidates:
                raise DatasetPreparationError(f"{source_path} 没有可分配的 Question 类型")
            unseen = [name for name in candidates if name not in buckets[source_path]]
            pool = unseen or candidates
            chosen = max(pool, key=lambda name: (remaining[name], -ANSWERABLE_TYPES.index(name)))
            buckets[source_path].append(chosen)
            remaining[chosen] -= 1
        if any(remaining.values()):
            raise DatasetPreparationError(f"领域 {domain} 的 Question 类型未分配完: {remaining}")
        allocation.update(buckets)
    return allocation


def _parent_chunks_for_source(
    corpus: Sequence[dict[str, Any]], source_path: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in corpus
        if item["metadata"]["source_path"] == source_path
        and item["metadata"]["chunk_level"] == "parent"
    ]


def _function_result(
    messages: list[dict[str, str]],
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
    max_tokens: int = 4096,
    attempts: int = 3,
) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return get_llm_service().call_function(
                messages,
                name=name,
                description=description,
                parameters=parameters,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            error = exc
            print(
                f"结构化调用重试: function={name} attempt={attempt}/{attempts} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
    raise DatasetPreparationError(f"结构化模型调用失败: {error}") from error


def _checkpoint_path(checkpoint_dir: Path, kind: str, key: str) -> Path:
    return checkpoint_dir / f"{kind}-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.json"


def _load_question_checkpoint(
    path: Path,
    *,
    expected_count: int,
    expected_category: str,
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != expected_count:
        return None
    if any(
        not isinstance(item, dict) or item.get("category") != expected_category
        for item in value
    ):
        return None
    return value


def _generate_positive_for_source(
    source: SourceDocument,
    chunks: Sequence[dict[str, Any]],
    assigned_types: Sequence[str],
) -> list[dict[str, Any]]:
    generation_chunks = _select_generation_chunks(chunks, len(assigned_types))
    if not generation_chunks:
        raise DatasetPreparationError(f"{source.path} 没有足够的有效父块")
    evidence_spans = [
        span
        for chunk in generation_chunks
        for span in _evidence_spans(chunk, source.normalized_text)
    ]
    if not evidence_spans:
        raise DatasetPreparationError(f"{source.path} 没有可用证据片段")
    span_by_id = {item["span_id"]: item for item in evidence_spans}
    chunk_payload = []
    for chunk in generation_chunks:
        spans = [item for item in evidence_spans if item["chunk_id"] == chunk["id"]]
        if spans:
            chunk_payload.append(
                {
                    "chunk_id": chunk["id"],
                    "section": chunk["metadata"]["section_path"],
                    "evidence_spans": [
                        {"span_id": item["span_id"], "text": item["text"]}
                        for item in spans
                    ],
                }
            )
    prompt = {
        "task": "只根据给出的真实文档父块生成 RAG Golden Question",
        "source_path": source.path,
        "requirements": [
            f"严格生成 {len(assigned_types)} 条，question_type 顺序必须为 {list(assigned_types)}",
            "query 使用自然中文，不复制文档标题，不引入块外知识",
            "keyword 要包含原文关键术语；semantic_rewrite 要避免照抄原句",
            "multi_evidence 必须选择至少两个共同回答问题的 chunk_id",
            "similar_concept 要区分文档中的相近概念，不能虚构对比对象",
            "只能通过 evidence_span_ids 选择已有原文片段，不能自己填写或改写 claim",
            "reference_answer 只重述所选 evidence spans，不扩展事实",
            "不同问题覆盖不同知识点",
        ],
        "output_schema": {
            "questions": [
                {
                    "query": "...",
                    "question_type": "keyword|semantic_rewrite|multi_evidence|similar_concept",
                    "reference_answer": "...",
                    "evidence_span_ids": ["parent-...:span:0"],
                }
            ]
        },
        "chunks": chunk_payload,
    }
    result = _function_result(
        [
            {
                "role": "system",
                "content": "你是严格的 RAG 数据集标注器，只能使用输入证据。输出 JSON。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        name="submit_questions",
        description="提交基于真实文档证据生成的 Golden Question",
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "question_type": {"type": "string", "enum": list(ANSWERABLE_TYPES)},
                            "reference_answer": {"type": "string"},
                            "evidence_span_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "query",
                            "question_type",
                            "reference_answer",
                            "evidence_span_ids",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        },
    )
    questions = result.get("questions")
    if not isinstance(questions, list) or len(questions) != len(assigned_types):
        raise DatasetPreparationError(f"{source.path} 生成 Question 数量不符")
    prepared = []
    for index, (item, expected_type) in enumerate(zip(questions, assigned_types), start=1):
        if not isinstance(item, dict) or item.get("question_type") != expected_type:
            raise DatasetPreparationError(f"{source.path} 第 {index} 条类型不符")
        query = str(item.get("query") or "").strip()
        answer = str(item.get("reference_answer") or "").strip()
        evidence_span_ids = item.get("evidence_span_ids")
        if not query or not answer:
            raise DatasetPreparationError(f"{source.path} 第 {index} 条字段不完整")
        if not isinstance(evidence_span_ids, list) or not evidence_span_ids:
            raise DatasetPreparationError(f"{source.path} 第 {index} 条缺少证据片段")
        selected_spans = []
        for span_id in dict.fromkeys(evidence_span_ids):
            if span_id not in span_by_id:
                raise DatasetPreparationError(f"{source.path} 引用了未知证据片段")
            selected_spans.append(span_by_id[span_id])
        selected_chunk_ids = list(dict.fromkeys(item["chunk_id"] for item in selected_spans))
        if expected_type == "multi_evidence" and len(selected_chunk_ids) < 2:
            raise DatasetPreparationError(f"{source.path} 多证据问题不足两个父块")
        clean_claims = [item["text"] for item in selected_spans]
        evidence = []
        relevance = {chunk_id: 3 for chunk_id in selected_chunk_ids}
        for span in selected_spans:
            evidence.append(
                {
                    "source_id": source.path,
                    "source_sha256": source.normalized_sha256,
                    "raw_source_sha256": source.sha256,
                    "start_char": span["source_start_char"],
                    "end_char": span["source_end_char"],
                    "quote": span["text"],
                }
            )
        prepared.append(
            {
                "id": _stable_id("q", source.path, query),
                "query": query,
                "category": source.domain,
                "question_type": expected_type,
                "expected_action": "answer",
                "reference_answer": answer,
                "reference_claims": clean_claims,
                "relevance": relevance,
                "evidence": evidence,
                "metadata": {
                    "label_origin": "ai_generated_from_real_docs_program_validated",
                    "label_completeness": "positive_and_negative",
                    "source_path": source.path,
                },
            }
        )
    return prepared


def generate_positive_questions(
    sources: Sequence[SourceDocument],
    corpus: Sequence[dict[str, Any]],
    plan: dict[str, Any],
    *,
    attempts: int = 3,
    checkpoint_dir: Path | None = None,
) -> list[dict[str, Any]]:
    allocation = _allocate_types(sources, plan, corpus)
    all_questions = []
    for source in sources:
        checkpoint = (
            _checkpoint_path(checkpoint_dir, "positive", source.path)
            if checkpoint_dir
            else None
        )
        cached = (
            _load_question_checkpoint(
                checkpoint,
                expected_count=source.target,
                expected_category=source.domain,
            )
            if checkpoint
            else None
        )
        if cached is not None:
            all_questions.extend(cached)
            print(f"正例从检查点恢复: {source.path} ({len(cached)})", flush=True)
            continue
        error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                questions = _generate_positive_for_source(
                    source,
                    _parent_chunks_for_source(corpus, source.path),
                    allocation[source.path],
                )
                all_questions.extend(questions)
                if checkpoint:
                    _write_json(checkpoint, questions)
                error = None
                print(f"正例已生成: {source.path} ({len(questions)})", flush=True)
                break
            except Exception as exc:
                error = exc
                print(
                    f"正例生成重试: {source.path} attempt={attempt}/{attempts} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
        if error is not None:
            raise DatasetPreparationError(f"{source.path} 生成失败: {error}") from error
    return all_questions


def _domain_overview(sources: Sequence[SourceDocument], domain: str) -> list[dict[str, Any]]:
    overview = []
    for source in sources:
        if source.domain != domain:
            continue
        headings = [match.group(2).strip() for match in HEADER_PATTERN.finditer(source.normalized_text)]
        overview.append({"path": source.path, "title": source.title, "headings": headings[:80]})
    return overview


def generate_negative_questions(
    sources: Sequence[SourceDocument],
    plan: dict[str, Any],
    *,
    attempts: int = 3,
    checkpoint_dir: Path | None = None,
) -> list[dict[str, Any]]:
    questions = []
    for quota in plan["domain_quotas"]:
        domain = quota["domain"]
        count = int(quota["unanswerable"])
        checkpoint = (
            _checkpoint_path(checkpoint_dir, "negative", domain)
            if checkpoint_dir
            else None
        )
        cached = (
            _load_question_checkpoint(
                checkpoint,
                expected_count=count,
                expected_category=domain,
            )
            if checkpoint
            else None
        )
        if cached is not None:
            questions.extend(cached)
            print(f"负例从检查点恢复: {domain} ({len(cached)})", flush=True)
            continue
        accepted = []
        for attempt in range(1, attempts + 1):
            result = _function_result(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是 RAG 负例设计器。根据知识库目录生成与领域相关、但目录所示文档没有答案的"
                            "问题。不要生成冷僻百科题；优先错误前提、未实现配置或必须外部查证的时效信息。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "domain": domain,
                                "count": count,
                                "documents": _domain_overview(sources, domain),
                                "requirements": [
                                    "严格生成指定数量且问题不重复",
                                    "expected_action 只能是 reject、correct_premise、external_research",
                                    "correct_premise 必须是知识库能纠正的错误技术前提",
                                    "external_research 只用于合理且需要外部最新资料的问题",
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                name="submit_negative_questions",
                description="提交知识库无答案或错误前提问题",
                parameters={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "expected_action": {
                                        "type": "string",
                                        "enum": sorted(EXPECTED_NEGATIVE_ACTIONS),
                                    },
                                },
                                "required": ["query", "expected_action"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["questions"],
                    "additionalProperties": False,
                },
                max_tokens=2048,
            )
            generated = result.get("questions")
            if not isinstance(generated, list) or len(generated) != count:
                print(f"负例生成重试: {domain} attempt={attempt} 数量不符", flush=True)
                continue
            candidates = []
            seen_queries = set()
            for item in generated:
                query = str(item.get("query") or "").strip() if isinstance(item, dict) else ""
                action = item.get("expected_action") if isinstance(item, dict) else None
                normalized_query = re.sub(r"\s+", "", query).casefold()
                if (
                    not query
                    or action not in EXPECTED_NEGATIVE_ACTIONS
                    or normalized_query in seen_queries
                ):
                    continue
                seen_queries.add(normalized_query)
                candidates.append(
                    {
                        "query": query,
                        "expected_action": action,
                    }
                )
            if len(candidates) != count:
                print(f"负例生成重试: {domain} attempt={attempt} 字段或去重校验未通过", flush=True)
                continue
            for candidate in candidates:
                accepted.append(
                    {
                        "id": _stable_id("q-negative", domain, candidate["query"]),
                        "query": candidate["query"],
                        "category": domain,
                        "question_type": "unanswerable",
                        "expected_action": candidate["expected_action"],
                        "metadata": {
                            "label_origin": "ai_generated_from_real_doc_catalog",
                            "label_completeness": "positive_and_negative",
                        },
                    }
                )
            break
        if len(accepted) != count:
            raise DatasetPreparationError(
                f"领域 {domain} 仅生成 {len(accepted)}/{count} 条有效负例"
            )
        questions.extend(accepted)
        if checkpoint:
            _write_json(checkpoint, accepted)
        print(f"负例已生成: {domain} ({count})", flush=True)
    return questions


def _validate_question_set(
    questions: Sequence[dict[str, Any]],
    sources: Sequence[SourceDocument],
    corpus: Sequence[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    source_by_path = {source.path: source for source in sources}
    corpus_ids = {item["id"] for item in corpus}
    ids = [item["id"] for item in questions]
    normalized_queries = [re.sub(r"\s+", "", item["query"]).casefold() for item in questions]
    if len(ids) != len(set(ids)) or len(normalized_queries) != len(set(normalized_queries)):
        raise DatasetPreparationError("Question 包含重复 ID 或重复问题")
    for item in questions:
        if item["expected_action"] != "answer":
            continue
        for doc_id in item["relevance"]:
            if doc_id not in corpus_ids:
                raise DatasetPreparationError(f"Question 引用了未知 Chunk: {doc_id}")
        for evidence in item["evidence"]:
            source = source_by_path[evidence["source_id"]]
            if evidence["source_sha256"] != source.normalized_sha256:
                raise DatasetPreparationError("Question 的来源哈希不一致")
            start, end = evidence["start_char"], evidence["end_char"]
            if source.normalized_text[start:end] != evidence["quote"]:
                raise DatasetPreparationError("Question 的证据字符区间不能回查原文")
    type_counts = Counter(item["question_type"] for item in questions)
    expected_types = {key: int(value["count"]) for key, value in plan["question_types"].items()}
    if dict(type_counts) != expected_types:
        raise DatasetPreparationError(f"Question 类型配额不一致: {dict(type_counts)}")
    domain_counts = Counter(item["category"] for item in questions)
    expected_domains = {
        item["domain"]: sum(int(item[name]) for name in (*ANSWERABLE_TYPES, "unanswerable"))
        for item in plan["domain_quotas"]
    }
    if dict(domain_counts) != expected_domains:
        raise DatasetPreparationError(f"Question 领域配额不一致: {dict(domain_counts)}")
    return {
        "questions": len(questions),
        "answerable": sum(item["expected_action"] == "answer" for item in questions),
        "unanswerable": sum(item["expected_action"] != "answer" for item in questions),
        "question_types": dict(sorted(type_counts.items())),
        "domains": dict(sorted(domain_counts.items())),
    }


def import_wiki(sources: Sequence[SourceDocument], *, sync_index: bool) -> dict[str, Any]:
    db = SessionLocal()
    try:
        service = get_page_service(db)
        imported = 0
        deduplicated = 0
        for source in sources:
            result = service.upsert_page_by_source(
                title=source.title,
                content=source.raw_text,
                source_type="obsidian_eval_source",
                source_uri=f"obsidian:{source.path}",
                notebook="评测语料",
                tags=[source.domain, source.document_type],
                change_summary="同步真实评测语料",
                sync_index=False,
            )
            imported += not result["deduplicated"]
            deduplicated += result["deduplicated"]
        index_result = None
        if sync_index:
            index_result = service.process_pending_index_tasks(limit=max(100, len(sources)))
        return {
            "pages": len(sources),
            "written_or_updated": imported,
            "deduplicated": deduplicated,
            "index": index_result,
        }
    finally:
        db.close()


def prepare_dataset(
    *,
    source_root: Path,
    output_dir: Path,
    manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    question_plan_path: Path = DEFAULT_QUESTION_PLAN,
    generate_questions: bool = True,
    import_pages: bool = True,
    sync_index: bool = True,
) -> dict[str, Any]:
    audit = audit_dataset_plan(
        source_root=source_root,
        source_manifest_path=manifest_path,
        question_plan_path=question_plan_path,
    )
    sources = load_sources(source_root, manifest_path)
    corpus = build_corpus(sources)
    plan = _load_json(question_plan_path)
    output_dir = output_dir.resolve()
    checkpoint_dir = output_dir.parent / f".{output_dir.name}-checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wiki_result = import_wiki(sources, sync_index=sync_index) if import_pages else None
    if import_pages and sync_index:
        index = wiki_result.get("index") or {}
        if index.get("failed"):
            raise DatasetPreparationError(
                f"Wiki 索引存在 {index['failed']} 个失败任务，停止生成负例"
            )
        from backend.app.services.doc_index_service import get_doc_index_service

        get_doc_index_service().sync_docs()

    questions: list[dict[str, Any]] = []
    question_summary: dict[str, Any] | None = None
    if generate_questions:
        questions = generate_positive_questions(
            sources,
            corpus,
            plan,
            checkpoint_dir=checkpoint_dir,
        )
        questions.extend(
            generate_negative_questions(
                sources,
                plan,
                checkpoint_dir=checkpoint_dir,
            )
        )
        question_summary = _validate_question_set(questions, sources, corpus, plan)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        _write_jsonl(temporary / "corpus.jsonl", corpus)
        if generate_questions:
            _write_jsonl(temporary / "questions.jsonl", questions)
            load_evaluation_dataset(
                temporary / "corpus.jsonl",
                temporary / "questions.jsonl",
                name=plan["dataset_name"],
            )
        profile = {
            "schema_version": "2.0",
            "dataset_name": plan["dataset_name"],
            "source_dataset_name": audit["source_dataset_name"],
            "source_documents": len(sources),
            "source_characters": audit["source_characters"],
            "corpus_units": len(corpus),
            "parent_chunks": sum(item["metadata"]["chunk_level"] == "parent" for item in corpus),
            "child_chunks": sum(item["metadata"]["chunk_level"] == "child" for item in corpus),
            "questions": question_summary,
            "configuration": {
                "parent_chunk_chars": settings.rag_parent_chunk_chars,
                "parent_chunk_overlap_chars": settings.rag_parent_chunk_overlap_chars,
                "child_chunk_chars": settings.rag_child_chunk_chars,
                "child_chunk_overlap_chars": settings.rag_child_chunk_overlap_chars,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "llm_model": settings.llm_model,
            },
            "source_hashes": audit["source_hashes"],
        }
        _write_json(temporary / "profile.json", profile)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {"output_dir": str(output_dir), "profile": profile, "wiki": wiki_result}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备智语真实 RAG 评测数据")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--question-plan", type=Path, default=DEFAULT_QUESTION_PLAN)
    parser.add_argument("--skip-questions", action="store_true")
    parser.add_argument("--skip-wiki", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_dataset(
            source_root=args.source_root.resolve(),
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            question_plan_path=args.question_plan,
            generate_questions=not args.skip_questions,
            import_pages=not args.skip_wiki,
            sync_index=not args.skip_index,
        )
    except (DatasetPreparationError, ManifestValidationError) as exc:
        raise SystemExit(f"评测准备失败: {exc}") from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
