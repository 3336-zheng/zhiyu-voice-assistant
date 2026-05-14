"""
Plan-and-Execute Agent - Executor 执行器
执行计划中的工具调用，支持无依赖步骤并行执行
"""
import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor as ThreadExecutor
import logging

from sqlalchemy.orm import Session

from backend.app.agent.models import (
    Plan, PlanStep, ToolName, IntentType,
    ToolResult, ExecutionResult,
    SearchParameters, CreateNoteParameters, UpdateNoteParameters,
    DateRangeParameters, CreateMdParameters, WriteMdParameters
)
from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
from backend.app.services.chroma_service import get_chroma_service
from backend.app.services.bm25_service import get_bm25_service
from backend.app.services.embedding_service import get_embedding_service
from backend.app.agent.markdown_agent import get_markdown_agent
from backend.app.core.database import SessionLocal
from backend.app.models.note import Note

logger = logging.getLogger(__name__)


def result_to_log(step: PlanStep, result: ToolResult) -> str:
    """将步骤结果转为日志字符串"""
    status = "成功" if result.success else f"失败: {result.error_message}"
    ms = f" ({result.execution_time_ms}ms)" if result.execution_time_ms else ""
    return f"步骤 {step.step_id}: {step.description} → {status}{ms}"


class Executor:
    """
    Agent 执行器
    负责执行计划中的工具调用
    """

    def __init__(self):
        """初始化执行器"""
        self._thread_pool = ThreadExecutor(max_workers=4)
        self.tools = {
            ToolName.SEARCH_KNOWLEDGE_BASE: self.search_knowledge_base,
            ToolName.CREATE_NOTE: self.create_note,
            ToolName.UPDATE_NOTE: self.update_note,
            ToolName.DELETE_NOTE: self.delete_note,
            ToolName.LIST_NOTES: self.list_notes,
            ToolName.GET_CURRENT_TIME: self.get_current_time,
            ToolName.SEARCH_BY_DATE_RANGE: self.search_by_date_range,
            ToolName.GET_NOTE_DETAIL: self.get_note_detail,
            ToolName.CREATE_MD_FILE: self.create_md_file,
            ToolName.WRITE_MD_FILE: self.write_md_file,
            ToolName.SUMMARIZE_TEXT: self.summarize_text,
        }

        # 初始化服务
        self.hybrid_retrieval = get_hybrid_retrieval_service()
        self.chroma_service = get_chroma_service()
        self.bm25_service = get_bm25_service()
        self.embedding_service = get_embedding_service()
        self.markdown_agent = get_markdown_agent()

    def _build_waves(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """
        拓扑排序：将步骤按依赖关系分层（wave），同层步骤可并行执行

        Args:
            steps: 计划步骤列表

        Returns:
            List[List[PlanStep]]: 分层后的步骤列表
        """
        step_map = {s.step_id: s for s in steps}
        completed: Set[int] = set()
        remaining = set(step_map.keys())
        waves = []

        while remaining:
            # 找出所有依赖已满足的步骤
            wave = []
            for sid in list(remaining):
                step = step_map[sid]
                deps = step.depends_on or []
                if all(d in completed for d in deps):
                    wave.append(step)

            if not wave:
                # 剩余步骤有循环依赖，强制逐个执行
                logger.warning(f"检测到循环依赖或无法满足的依赖，剩余 {len(remaining)} 步骤串行执行")
                for sid in list(remaining):
                    waves.append([step_map[sid]])
                break

            waves.append(wave)
            for s in wave:
                completed.add(s.step_id)
                remaining.discard(s.step_id)

        return waves

    async def execute(self, plan: Plan, db: Session = None) -> ExecutionResult:
        """
        执行计划（无依赖步骤并行执行）

        Args:
            plan: 执行计划
            db: 数据库会话

        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        results = []
        execution_log = []
        final_data = {}

        # 管理数据库会话
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False

        try:
            waves = self._build_waves(plan.steps)
            logger.info(f"计划共 {len(plan.steps)} 步，分为 {len(waves)} 个执行波次")

            for wave_idx, wave in enumerate(waves):
                if len(wave) == 1:
                    # 单步骤直接执行
                    result = self._execute_step(wave[0], db, results, plan.original_query)
                    results.append(result)
                    execution_log.append(result_to_log(wave[0], result))
                    if result.success:
                        self._aggregate_data(final_data, wave[0].tool_name, result.result)
                else:
                    # 多步骤并行执行（每个线程使用独立 db session）
                    logger.info(f"波次 {wave_idx + 1}: {len(wave)} 个步骤并行执行")
                    loop = asyncio.get_event_loop()
                    futures = []
                    for step in wave:
                        futures.append(
                            loop.run_in_executor(
                                self._thread_pool,
                                self._execute_step_concurrent,
                                step, results, plan.original_query
                            )
                        )
                    wave_results = await asyncio.gather(*futures)
                    for result, step in zip(wave_results, wave):
                        results.append(result)
                        execution_log.append(result_to_log(step, result))
                        if result.success:
                            self._aggregate_data(final_data, step.tool_name, result.result)

            total_time = int((time.time() - start_time) * 1000)
            success = all(r.success for r in results)
            completed_steps = sum(1 for r in results if r.success)

            return ExecutionResult(
                plan=plan,
                results=results,
                completed_steps=completed_steps,
                total_steps=len(plan.steps),
                success=success,
                execution_log=execution_log,
                final_data=final_data
            )

        finally:
            if should_close:
                db.close()

    def _resolve_step_references(self, parameters: Dict, prev_results: List[ToolResult]) -> Dict:
        """
        解析参数中的步骤结果引用（如 $step_1_results）

        Args:
            parameters: 步骤参数
            prev_results: 已执行步骤的结果列表

        Returns:
            Dict: 解析后的参数
        """
        import copy
        resolved = copy.deepcopy(parameters)
        for key, value in resolved.items():
            if isinstance(value, str) and value.startswith("$step_") and value.endswith("_results"):
                # 提取步骤ID: $step_1_results → 1
                try:
                    ref_step_id = int(value.split("_")[1])
                    dep_result = next(
                        (r for r in prev_results if r.step_id == ref_step_id and r.success),
                        None
                    )
                    if dep_result and dep_result.result:
                        # 将检索结果格式化为文本
                        resolved[key] = self._format_search_results(dep_result.result)
                        logger.info(f"解析引用 {value} → 获取到 {len(resolved[key])} 字符, 前100字: {resolved[key][:100]}")
                    else:
                        resolved[key] = ""
                        logger.warning(f"引用 {value} 对应的步骤未成功执行")
                except (ValueError, IndexError):
                    logger.warning(f"无法解析引用: {value}")
        return resolved

    def _format_search_results(self, search_result: Any) -> str:
        """将检索结果格式化为可写入的文本"""
        if isinstance(search_result, dict):
            # summarize_text 工具返回的格式：{"success": True, "summary": "..."}
            if "summary" in search_result:
                return search_result["summary"]
            items = search_result.get("results", search_result.get("items", []))
            if isinstance(items, list):
                parts = []
                for i, item in enumerate(items, 1):
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        content = item.get("content", item.get("text", ""))
                        score = item.get("score", "")
                        parts.append(f"### {i}. {title}" if title else f"### {i}")
                        if content:
                            parts.append(content)
                        if score:
                            parts.append(f"*相关度: {score}*")
                        parts.append("")
                return "\n".join(parts)
            # 如果是简单字典，直接取内容
            return search_result.get("content", search_result.get("text", str(search_result)))
        if isinstance(search_result, list):
            parts = []
            for i, item in enumerate(search_result, 1):
                if isinstance(item, dict):
                    content = item.get("content", item.get("text", str(item)))
                    parts.append(f"### {i}\n{content}")
                else:
                    parts.append(f"### {i}\n{item}")
            return "\n\n".join(parts)
        return str(search_result)

    def _execute_step(self, step: PlanStep, db: Session, prev_results: List[ToolResult], original_query: str = "") -> ToolResult:
        """
        执行单个步骤（同步，在当前线程）

        Args:
            step: 计划步骤
            db: 数据库会话
            prev_results: 已有结果列表（用于依赖检查）
            original_query: 原始用户查询（传递给 summarize_text）

        Returns:
            ToolResult: 执行结果
        """
        step_start = time.time()
        logger.info(f"执行步骤 {step.step_id}: {step.tool_name.value}")

        # 检查依赖
        if step.depends_on:
            for dep_id in step.depends_on:
                dep_result = next((r for r in prev_results if r.step_id == dep_id and r.success), None)
                if not dep_result:
                    return ToolResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        success=False,
                        result=None,
                        error_message=f"依赖步骤 {dep_id} 未成功执行"
                    )

        try:
            tool_func = self.tools.get(step.tool_name)
            if not tool_func:
                return ToolResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    result=None,
                    error_message=f"未知工具: {step.tool_name.value}"
                )

            # 解析参数中的步骤结果引用（如 $step_1_results）
            resolved_params = self._resolve_step_references(step.parameters, prev_results)
            # summarize_text 需要原始查询用于相关性过滤
            if step.tool_name == ToolName.SUMMARIZE_TEXT:
                result = tool_func(resolved_params, db, query=original_query)
            else:
                result = tool_func(resolved_params, db)
            step_time = int((time.time() - step_start) * 1000)
            return ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=True,
                result=result,
                execution_time_ms=step_time
            )
        except Exception as e:
            step_time = int((time.time() - step_start) * 1000)
            logger.error(f"工具执行失败: {step.tool_name.value}, 错误: {e}")
            return ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=step_time
            )

    def _execute_step_concurrent(self, step: PlanStep, prev_results: List[ToolResult], original_query: str = "") -> ToolResult:
        """
        并发执行单个步骤（独立线程，使用独立 db session）
        """
        step_start = time.time()
        logger.info(f"[并发] 执行步骤 {step.step_id}: {step.tool_name.value}")

        # 检查依赖
        if step.depends_on:
            for dep_id in step.depends_on:
                dep_result = next((r for r in prev_results if r.step_id == dep_id and r.success), None)
                if not dep_result:
                    return ToolResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        success=False,
                        result=None,
                        error_message=f"依赖步骤 {dep_id} 未成功执行"
                    )

        db = SessionLocal()
        try:
            tool_func = self.tools.get(step.tool_name)
            if not tool_func:
                return ToolResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    result=None,
                    error_message=f"未知工具: {step.tool_name.value}"
                )

            # 解析参数中的步骤结果引用（如 $step_1_results）
            resolved_params = self._resolve_step_references(step.parameters, prev_results)
            # summarize_text 需要原始查询用于相关性过滤
            if step.tool_name == ToolName.SUMMARIZE_TEXT:
                result = tool_func(resolved_params, db, query=original_query)
            else:
                result = tool_func(resolved_params, db)
            step_time = int((time.time() - step_start) * 1000)
            return ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=True,
                result=result,
                execution_time_ms=step_time
            )
        except Exception as e:
            step_time = int((time.time() - step_start) * 1000)
            logger.error(f"[并发] 工具执行失败: {step.tool_name.value}, 错误: {e}")
            return ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=step_time
            )
        finally:
            db.close()

    def _aggregate_data(self, final_data: Dict, tool_name: ToolName, result: Any):
        """
        聚合工具执行结果到 final_data

        Args:
            final_data: 最终数据字典
            tool_name: 工具名称
            result: 工具返回结果
        """
        if tool_name == ToolName.SEARCH_KNOWLEDGE_BASE:
            final_data["search_results"] = result
        elif tool_name == ToolName.LIST_NOTES:
            final_data["notes_list"] = result
        elif tool_name == ToolName.GET_CURRENT_TIME:
            final_data["current_time"] = result
        elif tool_name == ToolName.CREATE_NOTE:
            final_data["created_note"] = result
        elif tool_name == ToolName.UPDATE_NOTE:
            final_data["updated_note"] = result
        elif tool_name == ToolName.DELETE_NOTE:
            final_data["deleted_note"] = result
        elif tool_name == ToolName.GET_NOTE_DETAIL:
            final_data["note_detail"] = result
        elif tool_name == ToolName.CREATE_MD_FILE:
            final_data["created_md"] = result
        elif tool_name == ToolName.WRITE_MD_FILE:
            final_data["written_md"] = result
        elif tool_name == ToolName.SUMMARIZE_TEXT:
            final_data["summarized_text"] = result

    # ==================== 工具方法 ====================

    def search_knowledge_base(self, parameters: Dict, db: Session) -> List[Dict]:
        """
        检索知识库

        Args:
            parameters: 检索参数
            db: 数据库会话

        Returns:
            List[Dict]: 检索结果
        """
        params = SearchParameters(**parameters)

        results = self.hybrid_retrieval.search_hybrid(
            query=params.query,
            top_k=params.top_k,
            db=db
        )

        return results

    def create_note(self, parameters: Dict, db: Session) -> Dict:
        """
        创建笔记

        Args:
            parameters: 创建参数
            db: 数据库会话

        Returns:
            Dict: 创建的笔记信息
        """
        params = CreateNoteParameters(**parameters)

        # 生成摘要（简单实现）
        if not params.summary and params.content:
            params.summary = params.content[:100] + "..." if len(params.content) > 100 else params.content

        # 生成嵌入向量
        embedding = self.embedding_service.encode(params.content)

        # 创建笔记
        note = Note(
            title=params.title,
            content=params.content,
            summary=params.summary,
            tags=params.tags or [],
            audio_id=params.audio_id,
            duration=0.0
        )

        db.add(note)
        db.commit()
        db.refresh(note)

        # 同步到 ChromaDB
        self.chroma_service.add_embedding(
            note_id=note.id,
            embedding=embedding,
            content=params.content,
            metadata={
                "title": params.title,
                "tags": params.tags or []
            }
        )

        # 更新 BM25 索引
        self.bm25_service.add_document(f"note_{note.id}", params.content, params.title)

        return {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "summary": note.summary,
            "tags": note.tags,
            "created_at": note.created_at.isoformat() if note.created_at else None
        }

    def update_note(self, parameters: Dict, db: Session) -> Dict:
        """
        更新笔记

        Args:
            parameters: 更新参数
            db: 数据库会话

        Returns:
            Dict: 更新后的笔记信息
        """
        params = UpdateNoteParameters(**parameters)

        note = db.query(Note).filter(Note.id == params.note_id).first()
        if not note:
            raise ValueError(f"笔记不存在: ID={params.note_id}")

        # 更新字段
        if params.title:
            note.title = params.title
        if params.content:
            note.content = params.content
            # 重新生成摘要
            note.summary = params.content[:100] + "..." if len(params.content) > 100 else params.content
        if params.tags:
            note.tags = params.tags
        if params.summary:
            note.summary = params.summary

        db.commit()
        db.refresh(note)

        # 更新 ChromaDB
        if params.content:
            self.chroma_service.update_embedding(
                note_id=note.id,
                embedding=self.embedding_service.encode(note.content),
                content=note.content,
                metadata={"title": note.title, "tags": note.tags}
            )

            # 更新 BM25 索引
            self.bm25_service.update_document(note.id, note.content, note.title)

        return {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "summary": note.summary,
            "tags": note.tags,
            "updated_at": note.updated_at.isoformat() if note.updated_at else None
        }

    def delete_note(self, parameters: Dict, db: Session) -> Dict:
        """
        删除笔记

        Args:
            parameters: 删除参数
            db: 数据库会话

        Returns:
            Dict: 删除结果
        """
        note_id = parameters.get("note_id")
        if not note_id:
            raise ValueError("必须提供笔记ID")

        note = db.query(Note).filter(Note.id == note_id).first()
        if not note:
            raise ValueError(f"笔记不存在: ID={note_id}")

        # 删除数据库记录
        db.delete(note)
        db.commit()

        # 删除 ChromaDB 向量
        self.chroma_service.delete_by_note_id(note_id)

        # 删除 BM25 索引
        self.bm25_service.remove_document(note_id)

        return {
            "deleted": True,
            "note_id": note_id,
            "title": note.title if note else None
        }

    def list_notes(self, parameters: Dict, db: Session) -> List[Dict]:
        """
        列出笔记

        Args:
            parameters: 列表参数
            db: 数据库会话

        Returns:
            List[Dict]: 笔记列表
        """
        limit = parameters.get("limit", 20)
        date_from = parameters.get("date_from")
        date_to = parameters.get("date_to")

        query = db.query(Note)

        # 应用日期过滤
        if date_from:
            from_date = datetime.fromisoformat(date_from)
            query = query.filter(Note.created_at >= from_date)
        if date_to:
            to_date = datetime.fromisoformat(date_to)
            query = query.filter(Note.created_at <= to_date)

        # 按时间倒序
        query = query.order_by(Note.created_at.desc())
        notes = query.limit(limit).all()

        return [
            {
                "id": note.id,
                "title": note.title,
                "summary": note.summary,
                "tags": note.tags,
                "created_at": note.created_at.isoformat() if note.created_at else None
            }
            for note in notes
        ]

    def get_current_time(self, parameters: Dict, db: Session = None) -> Dict:
        """
        获取当前时间

        Args:
            parameters: 空参数
            db: 数据库会话（未使用）

        Returns:
            Dict: 当前时间信息
        """
        now = datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": weekdays[now.weekday()],
            "timestamp": int(now.timestamp())
        }

    def search_by_date_range(self, parameters: Dict, db: Session) -> List[Dict]:
        """
        按日期范围搜索笔记

        Args:
            parameters: 搜索参数
            db: 数据库会话

        Returns:
            List[Dict]: 笔记列表
        """
        params = DateRangeParameters(**parameters)

        query = db.query(Note)

        if params.date_from:
            from_date = datetime.fromisoformat(params.date_from)
            query = query.filter(Note.created_at >= from_date)
        if params.date_to:
            to_date = datetime.fromisoformat(params.date_to)
            query = query.filter(Note.created_at <= to_date)

        # 如果有附加查询，先进行语义过滤
        if params.query:
            # 使用 hybrid retrieval 获取相关笔记 ID
            results = self.hybrid_retrieval.search_hybrid(
                query=params.query,
                top_k=50,
                db=db
            )
            note_ids = [r["id"] for r in results]
            query = query.filter(Note.id.in_(note_ids))

        notes = query.order_by(Note.created_at.desc()).all()

        return [
            {
                "id": note.id,
                "title": note.title,
                "summary": note.summary,
                "tags": note.tags,
                "created_at": note.created_at.isoformat() if note.created_at else None
            }
            for note in notes
        ]

    def get_note_detail(self, parameters: Dict, db: Session) -> Dict:
        """
        获取笔记详情

        Args:
            parameters: 参数
            db: 数据库会话

        Returns:
            Dict: 笔记详情
        """
        note_id = parameters.get("note_id")
        if not note_id:
            raise ValueError("必须提供笔记ID")

        note = db.query(Note).filter(Note.id == note_id).first()
        if not note:
            raise ValueError(f"笔记不存在: ID={note_id}")

        return {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "summary": note.summary,
            "tags": note.tags,
            "created_at": note.created_at.isoformat() if note.created_at else None,
            "updated_at": note.updated_at.isoformat() if note.updated_at else None,
            "audio_id": note.audio_id,
            "duration": note.duration
        }

    def create_md_file(self, parameters: Dict, db: Session = None) -> Dict:
        """
        创建 MD 文件

        Args:
            parameters: 创建参数
            db: 数据库会话（未使用）

        Returns:
            Dict: 创建结果
        """
        params = CreateMdParameters(**parameters)
        result = self.markdown_agent.create_md_file(
            filename=params.filename,
            title=params.title,
            content=params.content,
            directory=params.directory
        )
        return result

    def write_md_file(self, parameters: Dict, db: Session = None) -> Dict:
        """
        写入 MD 文件

        Args:
            parameters: 写入参数
            db: 数据库会话（未使用）

        Returns:
            Dict: 写入结果
        """
        params = WriteMdParameters(**parameters)
        result = self.markdown_agent.write_md_file(
            filename=params.filename,
            content=params.content,
            mode=params.mode,
            directory=params.directory
        )
        return result

    def summarize_text(self, parameters: Dict, db: Session = None, query: str = "") -> Dict:
        """
        使用 LLM 总结文本内容

        Args:
            parameters: {"content": "要总结的内容（字符串或检索结果列表）"}
            db: 数据库会话（未使用）
            query: 原始用户查询主题，用于相关性过滤

        Returns:
            Dict: {"success": True, "summary": "总结后的文本"}
        """
        raw_content = parameters.get("content", "")

        # 处理检索结果格式：提取每条的 content 字段拼接
        if isinstance(raw_content, list):
            # 按 rerank_score 过滤低分结果
            if raw_content and isinstance(raw_content[0], dict):
                filtered = [r for r in raw_content if r.get("rerank_score", 0) > 0.5]
                if filtered:
                    logger.info(f"[summarize_text] 分数过滤: {len(raw_content)} → {len(filtered)} 条")
                    raw_content = filtered
            texts = []
            for item in raw_content:
                if isinstance(item, dict):
                    texts.append(item.get("content", item.get("text", "")))
                else:
                    texts.append(str(item))
            raw_content = "\n\n".join(texts)
        elif isinstance(raw_content, str):
            # 尝试解析 JSON 字符串（Executor 传过来的可能是序列化后的）
            import json
            try:
                parsed = json.loads(raw_content)
                if isinstance(parsed, list):
                    texts = []
                    for item in parsed:
                        if isinstance(item, dict):
                            texts.append(item.get("content", item.get("text", "")))
                        else:
                            texts.append(str(item))
                    raw_content = "\n\n".join(texts)
            except (json.JSONDecodeError, TypeError):
                pass

        if not raw_content or not raw_content.strip():
            return {"success": False, "summary": "", "error": "没有可总结的内容"}

        # 调用 LLM 进行总结
        try:
            from backend.app.services.llm_service import get_llm_service
            llm = get_llm_service()

            logger.info(f"[summarize_text] 输入内容长度: {len(raw_content)} 字符")

            # 构建主题提示
            topic_hint = f"用户查询主题：{query}\n\n" if query else ""

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个知识整理助手。你的任务是**仅**对用户提供的原始内容进行整理和总结。\n"
                        "严格要求：\n"
                        "1. **只使用**下方提供的原始内容，禁止添加任何原始内容中没有的信息\n"
                        "2. 去除重复内容，按主题分类整理，使用 Markdown 标题层级\n"
                        "3. 保留原始内容中的关键信息、数据、参数、示例\n"
                        "4. 使用中文\n"
                        "5. 直接输出总结内容，不要加'以下是总结'等前缀\n"
                        "6. 如果原始内容不足以形成完整总结，只整理已有的内容，不要补充\n"
                        "7. 只保留与用户查询主题直接相关的内容，丢弃不相关的段落或知识点"
                    )
                },
                {
                    "role": "user",
                    "content": f"{topic_hint}请严格基于以下原始内容进行整理总结，只保留与主题相关的内容，不要添加任何额外知识：\n\n{raw_content}"
                }
            ]

            summary = llm.chat(messages=messages, temperature=0.3, max_tokens=2000)
            logger.info(f"[summarize_text] LLM 返回长度: {len(summary)} 字符, 前100字: {summary[:100]}")
            return {"success": True, "summary": summary}

        except Exception as e:
            logger.error(f"LLM 总结失败: {e}")
            return {"success": False, "summary": raw_content, "error": str(e)}


# 全局执行器实例
executor_instance = None


def get_executor() -> Executor:
    """获取执行器实例（单例模式）"""
    global executor_instance
    if executor_instance is None:
        executor_instance = Executor()
    return executor_instance
