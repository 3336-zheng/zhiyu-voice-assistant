"""审计智语真实文档语料清单与统一评测集配额。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SOURCE_MANIFEST = Path(__file__).resolve().parent / "data" / "obsidian_sources.json"
DEFAULT_QUESTION_PLAN = Path(__file__).resolve().parent / "data" / "question_plan.json"
ANSWERABLE_TYPES = {"keyword", "semantic_rewrite", "multi_evidence", "similar_concept"}
UNANSWERABLE_TYPE = "unanswerable"
SENSITIVE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password|passwd)\b"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}",
        re.IGNORECASE,
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{12,}", re.IGNORECASE),
)


class ManifestValidationError(ValueError):
    """来源清单或 Question 配额不合法。"""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"无法读取 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestValidationError(f"JSON 根节点必须是对象: {path}")
    return value


def _safe_source_file(source_root: Path, relative_path: str) -> Path:
    root = source_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ManifestValidationError(f"来源路径越过根目录: {relative_path}") from None
    if candidate.suffix.lower() != ".md" or not candidate.is_file():
        raise ManifestValidationError(f"来源不是可读 Markdown: {relative_path}")
    return candidate


def _contains_sensitive_value(content: str) -> bool:
    return any(pattern.search(content) for pattern in SENSITIVE_PATTERNS)


def audit_dataset_plan(
    *,
    source_root: Path,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    question_plan_path: Path = DEFAULT_QUESTION_PLAN,
) -> dict[str, Any]:
    """验证来源安全性以及文档配额与 Question 配额是否闭合。"""
    manifest = _load_json(source_manifest_path)
    plan = _load_json(question_plan_path)
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ManifestValidationError("来源清单 documents 必须是非空数组")

    paths = set()
    domain_documents = Counter()
    domain_positive_targets = Counter()
    total_characters = 0
    source_hashes = {}
    for index, item in enumerate(documents, start=1):
        if not isinstance(item, dict):
            raise ManifestValidationError(f"来源清单第 {index} 项必须是对象")
        relative_path = str(item.get("path") or "").strip()
        domain = str(item.get("domain") or "").strip()
        document_type = str(item.get("document_type") or "").strip()
        target = item.get("positive_question_target")
        if not relative_path or relative_path in paths:
            raise ManifestValidationError(f"来源路径为空或重复: {relative_path or '<empty>'}")
        if not domain or not document_type:
            raise ManifestValidationError(f"来源缺少 domain/document_type: {relative_path}")
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise ManifestValidationError(f"来源正例配额必须是正整数: {relative_path}")
        path = _safe_source_file(source_root, relative_path)
        content = path.read_text(encoding="utf-8")
        if _contains_sensitive_value(content):
            raise ManifestValidationError(f"来源文件命中敏感值规则: {relative_path}")
        paths.add(relative_path)
        domain_documents[domain] += 1
        domain_positive_targets[domain] += target
        total_characters += len(content)
        source_hashes[relative_path] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    question_types = plan.get("question_types")
    domain_quotas = plan.get("domain_quotas")
    if not isinstance(question_types, dict) or not isinstance(domain_quotas, list):
        raise ManifestValidationError("Question 规划缺少 question_types 或 domain_quotas")
    expected_types = ANSWERABLE_TYPES | {UNANSWERABLE_TYPE}
    if set(question_types) != expected_types:
        raise ManifestValidationError("Question 类型必须严格包含五个预定义类型")
    type_targets = {}
    for name, item in question_types.items():
        count = item.get("count") if isinstance(item, dict) else None
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ManifestValidationError(f"Question 类型配额无效: {name}")
        type_targets[name] = count

    quota_by_type = Counter()
    quota_positive_by_domain = Counter()
    quota_unanswerable_by_domain = Counter()
    seen_domains = set()
    for item in domain_quotas:
        if not isinstance(item, dict):
            raise ManifestValidationError("domain_quotas 每项必须是对象")
        domain = str(item.get("domain") or "").strip()
        if not domain or domain in seen_domains:
            raise ManifestValidationError(f"领域为空或重复: {domain or '<empty>'}")
        seen_domains.add(domain)
        for question_type in expected_types:
            count = item.get(question_type)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ManifestValidationError(
                    f"领域 {domain} 的 {question_type} 配额必须是非负整数"
                )
            quota_by_type[question_type] += count
            if question_type in ANSWERABLE_TYPES:
                quota_positive_by_domain[domain] += count
            else:
                quota_unanswerable_by_domain[domain] += count

    if set(domain_documents) != seen_domains:
        raise ManifestValidationError("来源领域与 Question 配额领域不一致")
    if dict(quota_by_type) != type_targets:
        raise ManifestValidationError("按领域汇总的 Question 配额与类型总数不一致")
    if quota_positive_by_domain != domain_positive_targets:
        raise ManifestValidationError("来源正例配额与按领域 Question 正例配额不一致")

    answerable_total = sum(type_targets[name] for name in ANSWERABLE_TYPES)
    unanswerable_total = type_targets[UNANSWERABLE_TYPE]
    total_questions = answerable_total + unanswerable_total
    expected_totals = {
        "total_questions": total_questions,
        "answerable_questions": answerable_total,
        "unanswerable_questions": unanswerable_total,
    }
    for key, expected in expected_totals.items():
        if plan.get(key) != expected:
            raise ManifestValidationError(f"{key} 声明值与配额汇总不一致")

    return {
        "status": "passed",
        "dataset_name": str(plan.get("dataset_name") or ""),
        "source_dataset_name": str(manifest.get("dataset_name") or ""),
        "source_documents": len(documents),
        "source_characters": total_characters,
        "source_hashes": source_hashes,
        "domain_documents": dict(sorted(domain_documents.items())),
        "domain_positive_targets": dict(sorted(domain_positive_targets.items())),
        "domain_unanswerable_targets": dict(sorted(quota_unanswerable_by_domain.items())),
        "question_type_targets": dict(sorted(type_targets.items())),
        **expected_totals,
        "checks": [
            "source_path_boundary",
            "markdown_only",
            "duplicate_source_path",
            "sensitive_value_scan",
            "source_sha256",
            "domain_quota_consistency",
            "question_type_quota_consistency",
            "answerable_unanswerable_total_consistency",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计真实文档语料与统一评测集划分")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--question-plan", type=Path, default=DEFAULT_QUESTION_PLAN)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_dataset_plan(
            source_root=args.source_root,
            source_manifest_path=args.manifest,
            question_plan_path=args.question_plan,
        )
    except ManifestValidationError as exc:
        raise SystemExit(f"数据划分审计失败: {exc}") from None
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
