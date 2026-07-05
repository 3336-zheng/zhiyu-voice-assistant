"""
Plan-and-Execute Agent - Executor 执行器（课堂学习场景聚焦版）
执行计划中的工具调用，支持无依赖步骤并行执行
"""
import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor as ThreadExecutor
import logging

from sqlalchemy.orm import Session

from backend.app.agent.models import (
    Plan, PlanStep, ToolName, IntentType,
    ToolResult, ExecutionResult,
    SearchParameters, CreateNoteParameters, UpdateNoteParameters
)
from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
from backend.app.agent.markdown_agent import get_markdown_agent
from backend.app.core.database import SessionLocal

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
            ToolName.SUMMARIZE_TEXT: self.summarize_text,
        }

        # 初始化服务
        self.hybrid_retrieval = get_hybrid_retrieval_service()
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
        创建笔记（写入 data/notes/ 作为 md 文件，不写入 SQLite/ChromaDB/BM25）

        Args:
            parameters: 创建参数
            db: 数据库会话（未使用）

        Returns:
            Dict: 创建的笔记信息
        """
        params = CreateNoteParameters(**parameters)

        # 生成文件名：优先用 title，否则用时间戳
        filename = params.title or f"笔记_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 写入 data/notes/ 目录
        result = self.markdown_agent.create_md_file(
            filename=filename,
            title=params.title,
            content=params.content,
            directory="data/notes"
        )

        if result.get("success"):
            # 同步索引到 ChromaDB + BM25（使笔记可被搜索）
            try:
                from backend.app.services.doc_index_service import get_doc_index_service
                get_doc_index_service().index_doc(result.get("file_path"))
            except Exception as e:
                logger.warning(f"笔记 {result.get('filename')} 索引失败（不影响创建）: {e}")

            return {
                "id": None,
                "title": params.title,
                "content": params.content,
                "file_path": result.get("file_path"),
                "filename": result.get("filename"),
                "created_at": result.get("created_at")
            }
        else:
            raise ValueError(f"创建笔记文件失败: {result.get('error')}")

    def update_note(self, parameters: Dict, db: Session) -> Dict:
        """
        更新笔记（覆写 data/notes/ 下的 md 文件）

        Args:
            parameters: 更新参数
            db: 数据库会话（未使用）

        Returns:
            Dict: 更新后的笔记信息
        """
        params = UpdateNoteParameters(**parameters)

        # 先读取原文件内容，确认文件存在
        read_result = self.markdown_agent.read_md_file(params.filename, directory="data/notes")
        if not read_result.get("success"):
            raise ValueError(f"笔记文件不存在: {params.filename}")

        # 构建新内容：保留原标题行，覆写正文
        new_content = params.content or ""
        if params.title:
            new_content = f"# {params.title}\n\n{new_content}"

        # 覆写文件
        result = self.markdown_agent.write_md_file(
            filename=params.filename,
            content=new_content,
            mode="overwrite",
            directory="data/notes"
        )

        if result.get("success"):
            # 同步更新索引
            try:
                from backend.app.services.doc_index_service import get_doc_index_service
                get_doc_index_service().index_doc(result.get("file_path"))
            except Exception as e:
                logger.warning(f"笔记 {params.filename} 索引更新失败: {e}")

            return {
                "filename": params.filename,
                "title": params.title,
                "content": new_content,
                "updated_at": result.get("written_at")
            }
        else:
            raise ValueError(f"更新笔记文件失败: {result.get('error')}")

    def delete_note(self, parameters: Dict, db: Session) -> Dict:
        """
        删除笔记（删除 data/notes/ 下的 md 文件）

        Args:
            parameters: 删除参数（filename 字段）
            db: 数据库会话（未使用）

        Returns:
            Dict: 删除结果
        """
        filename = parameters.get("filename")
        if not filename:
            raise ValueError("必须提供文件名")

        file_path = self.markdown_agent._get_file_path(filename, directory="data/notes")

        if not os.path.exists(file_path):
            raise ValueError(f"笔记文件不存在: {filename}")

        os.remove(file_path)
        logger.info(f"删除笔记文件: {file_path}")

        # 清理索引
        try:
            from backend.app.services.doc_index_service import get_doc_index_service
            get_doc_index_service().remove_doc(filename if filename.endswith(".md") else f"{filename}.md")
        except Exception as e:
            logger.warning(f"笔记 {filename} 索引清理失败: {e}")

        return {
            "deleted": True,
            "filename": filename,
            "file_path": file_path
        }

    def list_notes(self, parameters: Dict, db: Session) -> List[Dict]:
        """
        列出笔记（读取 data/notes/ 目录下的 md 文件）

        Args:
            parameters: 列表参数（limit, date_from, date_to）
            db: 数据库会话（未使用）

        Returns:
            List[Dict]: 笔记列表
        """
        limit = parameters.get("limit", 20)
        date_from = parameters.get("date_from")
        date_to = parameters.get("date_to")

        result = self.markdown_agent.list_md_files(directory="data/notes")
        files = result.get("files", [])

        # 按文件修改时间过滤
        filtered = []
        for f in files:
            mtime = f.get("modified_at", "")
            if date_from and mtime < date_from:
                continue
            if date_to and mtime > date_to + "T23:59:59":
                continue
            filtered.append(f)

        # 限制数量
        filtered = filtered[:limit]

        return [
            {
                "id": None,
                "filename": f["filename"],
                "title": f["filename"].replace(".md", ""),
                "summary": "",
                "tags": [],
                "created_at": f.get("modified_at"),
                "file_path": f.get("file_path")
            }
            for f in filtered
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
