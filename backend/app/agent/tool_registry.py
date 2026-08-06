"""Agent 工具注册表和具体工具实现。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.agent.models import (
    CreateNoteParameters,
    SearchParameters,
    ToolName,
    UpdateNoteParameters,
)
from backend.app.models.wiki import ExternalResearchRun, WikiPageSource
from backend.app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
    get_hybrid_retrieval_service,
)
from backend.app.services.page_service import PageService, get_page_service

logger = logging.getLogger(__name__)


class AgentToolRegistry:
    """集中管理工具映射，并通过工厂延迟获取外部依赖。"""

    def __init__(
        self,
        retrieval_factory: Optional[Callable[[], HybridRetrievalService]] = None,
        page_factory: Optional[Callable[[Session], PageService]] = None,
        llm_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._retrieval_factory = retrieval_factory or get_hybrid_retrieval_service
        self._page_factory = page_factory or get_page_service
        self._llm_factory = llm_factory
        self._hybrid_retrieval: Optional[HybridRetrievalService] = None
        self.handlers = {
            ToolName.SEARCH_KNOWLEDGE_BASE: self.search_knowledge_base,
            ToolName.CREATE_NOTE: self.create_note,
            ToolName.UPDATE_NOTE: self.update_note,
            ToolName.DELETE_NOTE: self.delete_note,
            ToolName.LIST_NOTES: self.list_notes,
            ToolName.GET_CURRENT_TIME: self.get_current_time,
            ToolName.SUMMARIZE_TEXT: self.summarize_text,
        }

    @property
    def hybrid_retrieval(self) -> HybridRetrievalService:
        """首次使用检索工具时再加载本地模型。"""
        if self._hybrid_retrieval is None:
            self._hybrid_retrieval = self._retrieval_factory()
        return self._hybrid_retrieval

    def _get_llm(self) -> Any:
        if self._llm_factory is not None:
            return self._llm_factory()
        from backend.app.services.llm_service import get_llm_service

        return get_llm_service()

    def search_knowledge_base(self, parameters: Dict, db: Session) -> List[Dict]:
        """执行混合检索。"""
        params = SearchParameters(**parameters)
        return self.hybrid_retrieval.search_hybrid(
            query=params.query,
            top_k=params.top_k,
            db=db,
        )

    def create_note(self, parameters: Dict, db: Session) -> Dict:
        """通过 PageService 创建可版本化、可检索的 Wiki 页面。"""
        params = CreateNoteParameters(**parameters)
        service = self._page_factory(db)
        if params.research_run_id:
            run = db.get(ExternalResearchRun, params.research_run_id)
            if run is None or run.status not in {"completed", "save_pending", "saved"}:
                raise ValueError("外部研究任务不存在或状态不可保存")
            result = service.upsert_page_by_source(
                title=params.title,
                content=params.content,
                notebook=params.notebook,
                tags=params.tags,
                source_type="external_research",
                source_uri=f"external-research:{run.id}",
                change_summary="确认后保存外部研究草稿",
            )
            for source in run.sources:
                exists = (
                    db.query(WikiPageSource)
                    .filter(
                        WikiPageSource.page_id == result["id"],
                        WikiPageSource.research_source_id == source.id,
                    )
                    .first()
                )
                if not exists:
                    db.add(
                        WikiPageSource(
                            page_id=result["id"],
                            research_source_id=source.id,
                        )
                    )
            run.page_id = result["id"]
            run.status = "saved"
            db.commit()
            result["research_run_id"] = run.id
            return result

        return service.create_page(
            title=params.title,
            content=params.content,
            notebook=params.notebook,
            tags=params.tags,
            source_type="agent_note",
            change_summary="Agent 确认后创建页面",
        )

    def update_note(self, parameters: Dict, db: Session) -> Dict:
        """通过稳定 ID、标题或别名更新 Wiki 页面。"""
        params = UpdateNoteParameters(**parameters)
        service = self._page_factory(db)
        current = service.find_page(params.filename)
        return service.update_page(
            current["id"],
            expected_revision=current["revision"],
            title=params.title,
            content=params.content,
            tags=params.tags,
            change_summary="Agent 确认后更新页面",
        )

    def delete_note(self, parameters: Dict, db: Session) -> Dict:
        """通过 PageService 删除页面，历史版本继续保留。"""
        filename = parameters.get("filename")
        if not filename:
            raise ValueError("必须提供页面 ID、标题或别名")
        service = self._page_factory(db)
        current = service.find_page(filename)
        deleted = service.delete_page(
            current["id"],
            expected_revision=current["revision"],
        )
        deleted["deleted"] = True
        return deleted

    def list_notes(self, parameters: Dict, db: Session) -> List[Dict]:
        """列出统一 Wiki 中的活动页面。"""
        limit = parameters.get("limit", 20)
        date_from = parameters.get("date_from")
        date_to = parameters.get("date_to")
        result = self._page_factory(db).list_pages(offset=0, limit=1_000_000)
        filtered = []
        for page in result["items"]:
            modified_at = page.get("updated_at", "")
            if date_from and modified_at < date_from:
                continue
            if date_to and modified_at > date_to + "T23:59:59":
                continue
            filtered.append(page)
        return filtered[:limit]

    @staticmethod
    def get_current_time(parameters: Dict, db: Session = None) -> Dict:
        """返回当前本地时间。"""
        now = datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": weekdays[now.weekday()],
            "timestamp": int(now.timestamp()),
        }

    def summarize_text(self, parameters: Dict, db: Session = None, query: str = "") -> Dict:
        """严格基于工具输入调用 LLM 生成摘要。"""
        raw_content = parameters.get("content", "")
        if isinstance(raw_content, list):
            if raw_content and isinstance(raw_content[0], dict):
                filtered = [item for item in raw_content if item.get("rerank_score", 0) > 0.5]
                if filtered:
                    logger.info(
                        "[summarize_text] 分数过滤: %s -> %s 条",
                        len(raw_content),
                        len(filtered),
                    )
                    raw_content = filtered
            raw_content = "\n\n".join(
                item.get("content", item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
                for item in raw_content
            )
        elif isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
                if isinstance(parsed, list):
                    raw_content = "\n\n".join(
                        item.get("content", item.get("text", ""))
                        if isinstance(item, dict)
                        else str(item)
                        for item in parsed
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        if not raw_content or not raw_content.strip():
            return {"success": False, "summary": "", "error": "没有可总结的内容"}

        try:
            logger.info("[summarize_text] 输入内容长度: %s 字符", len(raw_content))
            topic_hint = f"用户查询主题：{query}\n\n" if query else ""
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个知识整理助手。你的任务是仅对用户提供的原始内容进行整理和总结。\n"
                        "严格要求：\n"
                        "1. 只使用下方提供的原始内容，禁止添加任何原始内容中没有的信息\n"
                        "2. 去除重复内容，按主题分类整理，使用 Markdown 标题层级\n"
                        "3. 保留原始内容中的关键信息、数据、参数、示例\n"
                        "4. 使用中文\n"
                        "5. 直接输出总结内容，不要添加固定前缀\n"
                        "6. 原始内容不足时只整理已有内容，不要补充\n"
                        "7. 只保留与用户查询主题直接相关的内容"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{topic_hint}请严格基于以下原始内容进行整理总结，"
                        f"只保留与主题相关的内容，不要添加任何额外知识：\n\n{raw_content}"
                    ),
                },
            ]
            summary = self._get_llm().chat(
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )
            logger.info("[summarize_text] LLM 返回长度: %s 字符", len(summary))
            return {"success": True, "summary": summary}
        except Exception as exc:
            logger.error("LLM 总结失败: %s", exc)
            return {"success": False, "summary": raw_content, "error": str(exc)}
