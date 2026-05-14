"""
Plan-and-Execute Agent - Responder 回复生成器
支持 LLM 增强的智能回复生成
"""
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.app.agent.models import (
    Plan, ExecutionResult, AgentResponse, IntentType
)


logger = logging.getLogger(__name__)

# LLM 回复生成的系统提示
RESPONSE_GENERATION_PROMPT = """你是一个智能笔记助手的回复生成器。根据用户的查询和系统执行结果，生成自然、友好的回复。

要求：
1. 回复应该简洁、清晰、有条理
2. 如果是检索结果，列出关键信息并标注相关度
3. 如果是操作结果，确认操作成功并提供关键信息
4. 如果有错误，友好地解释问题并提供建议
5. 使用中文回复
6. 适当使用 markdown 格式增强可读性"""


class Responder:
    """
    Agent 回复生成器
    支持 LLM 增强的智能回复生成
    """

    def __init__(self):
        """初始化回复生成器"""
        self._llm_service = None

    @property
    def llm_service(self):
        """延迟加载 LLM 服务"""
        if self._llm_service is None:
            try:
                from backend.app.services.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            except Exception as e:
                logger.warning(f"LLM 服务加载失败，将使用模板回复: {e}")
        return self._llm_service

    def generate_response(
        self,
        user_query: str,
        plan: Plan,
        execution_result: ExecutionResult,
        context: List[Dict[str, str]] = None
    ) -> AgentResponse:
        """
        生成 Agent 最终回复

        Args:
            user_query: 用户原始查询
            plan: 执行计划
            execution_result: 执行结果
            context: 对话上下文（可选）

        Returns:
            AgentResponse: 最终响应
        """
        start_time = time.time()

        # 优先使用 LLM 生成回复
        if self.llm_service:
            try:
                response_text = self._generate_with_llm(user_query, plan, execution_result, context)
            except Exception as e:
                logger.warning(f"LLM 回复生成失败，降级到模板: {e}")
                response_text = self._generate_with_template(plan, execution_result, user_query)
        else:
            response_text = self._generate_with_template(plan, execution_result, user_query)

        execution_time = int((time.time() - start_time) * 1000)
        sources = self._extract_sources(execution_result)

        return AgentResponse(
            query=user_query,
            response=response_text,
            plan=plan,
            execution_result=execution_result,
            sources=sources,
            confidence=1.0 if execution_result.success else 0.5,
            timestamp=datetime.now(),
            execution_time_ms=execution_time
        )

    def _generate_with_llm(
        self,
        user_query: str,
        plan: Plan,
        result: ExecutionResult,
        context: List[Dict[str, str]] = None
    ) -> str:
        """
        使用 LLM 生成回复

        Args:
            user_query: 用户查询
            plan: 执行计划
            result: 执行结果
            context: 对话上下文

        Returns:
            str: 生成的回复
        """
        # 构建执行结果摘要
        result_summary = self._build_result_summary(plan, result)

        messages = [
            {"role": "system", "content": RESPONSE_GENERATION_PROMPT}
        ]

        # 添加对话上下文
        if context:
            for msg in context[-4:]:  # 最近 2 轮对话
                messages.append(msg)

        messages.append({
            "role": "user",
            "content": f"""用户原始查询: {user_query}
润色后的查询: {self._get_polished_query(plan)}

执行意图: {plan.intent.value}
执行状态: {'成功' if result.success else '失败'}

执行结果:
{result_summary}

请根据以上信息生成回复。
注意：如果原始查询和润色后的查询不同，请在回复开头简要说明您理解的查询意图（如"您想查询的是XXX"）。"""
        })

        return self.llm_service.chat(messages=messages, temperature=0.7, max_tokens=1000)

    def _build_result_summary(self, plan: Plan, result: ExecutionResult) -> str:
        """构建执行结果摘要"""
        lines = []

        if plan.intent == IntentType.SEARCH:
            search_results = result.final_data.get("search_results", [])
            if search_results:
                lines.append(f"找到 {len(search_results)} 条相关笔记：")
                for i, item in enumerate(search_results[:5], 1):
                    title = item.get("title", "无标题")
                    score = item.get("rerank_score", 0)
                    content = item.get("content", "")[:100]
                    lines.append(f"{i}. {title} (相关度: {score:.2f})")
                    lines.append(f"   内容: {content}...")
            else:
                lines.append("未找到相关内容。")

        elif plan.intent == IntentType.CREATE_NOTE:
            created = result.final_data.get("created_note")
            if created:
                lines.append(f"笔记创建成功")
                lines.append(f"标题: {created.get('title')}")
                lines.append(f"ID: {created.get('id')}")

        elif plan.intent == IntentType.UPDATE_NOTE:
            updated = result.final_data.get("updated_note")
            if updated:
                lines.append(f"笔记更新成功")
                lines.append(f"标题: {updated.get('title')}")
                lines.append(f"ID: {updated.get('id')}")

        elif plan.intent == IntentType.DELETE_NOTE:
            deleted = result.final_data.get("deleted_note")
            if deleted:
                lines.append(f"笔记删除成功")
                lines.append(f"ID: {deleted.get('note_id')}")

        elif plan.intent == IntentType.LIST_NOTES:
            notes = result.final_data.get("notes_list", [])
            if notes:
                lines.append(f"共 {len(notes)} 条笔记：")
                for note in notes[:10]:
                    lines.append(f"- {note.get('title')} (ID: {note.get('id')})")
            else:
                lines.append("暂无笔记。")

        elif plan.intent == IntentType.TIME_QUERY:
            time_info = result.final_data.get("current_time")
            if time_info:
                lines.append(f"当前时间: {time_info.get('date')} {time_info.get('weekday')} {time_info.get('time')}")

        elif plan.intent == IntentType.SUMMARIZE:
            search_results = result.final_data.get("search_results", [])
            if search_results:
                lines.append(f"找到 {len(search_results)} 条相关内容用于总结")
                for i, item in enumerate(search_results[:5], 1):
                    lines.append(f"{i}. {item.get('title')}: {item.get('content', '')[:80]}...")

        elif plan.intent == IntentType.CREATE_MD:
            created_md = result.final_data.get("created_md")
            if created_md:
                if created_md.get("success"):
                    lines.append(f"MD文件创建成功")
                    lines.append(f"文件名: {created_md.get('filename')}")
                    lines.append(f"路径: {created_md.get('file_path')}")
                else:
                    lines.append(f"MD文件创建失败: {created_md.get('error')}")

        elif plan.intent == IntentType.WRITE_MD:
            written_md = result.final_data.get("written_md")
            if written_md:
                if written_md.get("success"):
                    lines.append(f"内容写入MD文件成功")
                    lines.append(f"文件名: {written_md.get('filename')}")
                    lines.append(f"模式: {written_md.get('mode')}")
                else:
                    lines.append(f"写入MD文件失败: {written_md.get('error')}")

        return "\n".join(lines) if lines else "执行完成。"

    def _get_polished_query(self, plan: Plan) -> str:
        """从 plan 的 SearchParameters 中提取润色后的查询"""
        if plan.intent == IntentType.SEARCH and plan.steps:
            params = plan.steps[0].parameters
            return params.get("query", plan.original_query)
        return plan.original_query

    def _generate_with_template(
        self,
        plan: Plan,
        result: ExecutionResult,
        user_query: str
    ) -> str:
        """
        使用模板生成回复（降级方案）

        Args:
            plan: 执行计划
            result: 执行结果
            user_query: 用户查询

        Returns:
            str: 生成的回复
        """
        if plan.intent == IntentType.SEARCH:
            return self._template_search_response(plan, result)
        elif plan.intent == IntentType.CREATE_NOTE:
            return self._template_create_note_response(plan, result)
        elif plan.intent == IntentType.UPDATE_NOTE:
            return self._template_update_note_response(plan, result)
        elif plan.intent == IntentType.DELETE_NOTE:
            return self._template_delete_note_response(plan, result)
        elif plan.intent == IntentType.LIST_NOTES:
            return self._template_list_notes_response(plan, result)
        elif plan.intent == IntentType.TIME_QUERY:
            return self._template_time_query_response(plan, result)
        elif plan.intent == IntentType.SUMMARIZE:
            return self._template_summarize_response(plan, result, user_query)
        elif plan.intent == IntentType.CREATE_MD:
            return self._template_create_md_response(plan, result)
        elif plan.intent == IntentType.WRITE_MD:
            return self._template_write_md_response(plan, result)
        else:
            return self._template_default_response(plan, result)

    def _template_search_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：检索回复"""
        search_results = result.final_data.get("search_results", [])

        if not search_results:
            return "未找到相关内容。"

        if not result.success:
            return "检索过程中出现错误，请稍后重试。"

        polished = self._get_polished_query(plan)
        lines = []
        if polished != plan.original_query:
            lines.append(f"为您查询「{polished}」，找到 {len(search_results)} 条相关内容：")
        else:
            lines.append(f"为您找到 {len(search_results)} 条相关内容：")
        lines.append("")

        for i, item in enumerate(search_results, 1):
            title = item.get("title", "无标题")
            content = item.get("content", "")
            score = item.get("rerank_score", 0)

            if len(content) > 150:
                content = content[:150] + "..."

            lines.append(f"{i}. **{title}** (相关度: {score:.2f})")
            lines.append(f"   {content}")
            lines.append("")

        return "\n".join(lines)

    def _template_create_note_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：创建笔记回复"""
        if not result.success:
            return "创建笔记失败，请检查输入内容。"

        created_note = result.final_data.get("created_note")
        if not created_note:
            return "创建笔记成功，但无法获取详情。"

        return f"笔记创建成功！\n\n标题：{created_note.get('title', '无标题')}\nID：{created_note.get('id')}"

    def _template_update_note_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：更新笔记回复"""
        if not result.success:
            error_msg = result.results[-1].error_message if result.results else "未知错误"
            return f"更新笔记失败：{error_msg}"

        updated_note = result.final_data.get("updated_note")
        if not updated_note:
            return "更新笔记成功。"

        return f"笔记更新成功！\n\n标题：{updated_note.get('title', '无标题')}\nID：{updated_note.get('id')}"

    def _template_delete_note_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：删除笔记回复"""
        if not result.success:
            error_msg = result.results[-1].error_message if result.results else "未知错误"
            return f"删除笔记失败：{error_msg}"

        deleted_note = result.final_data.get("deleted_note")
        if deleted_note:
            return f"已删除笔记：{deleted_note.get('title', 'ID=' + str(deleted_note.get('note_id')))}"
        return "笔记已删除。"

    def _template_list_notes_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：列出笔记回复"""
        notes_list = result.final_data.get("notes_list", [])

        if not notes_list:
            return "未找到任何笔记。"

        lines = [f"共 {len(notes_list)} 条笔记：", ""]

        for note in notes_list:
            title = note.get("title", "无标题")
            created = note.get("created_at", "")[:10]
            tags = note.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""

            lines.append(f"- **{title}** (ID: {note.get('id')}){tag_str}")
            lines.append(f"  创建于: {created}")
            lines.append("")

        return "\n".join(lines)

    def _template_time_query_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：时间查询回复"""
        time_info = result.final_data.get("current_time")
        if not time_info:
            return "无法获取当前时间。"

        date = time_info.get("date", "")
        weekday = time_info.get("weekday", "")
        current_time = time_info.get("time", "")

        return f"现在是 {date} {weekday} {current_time}"

    def _template_summarize_response(self, plan: Plan, result: ExecutionResult, user_query: str) -> str:
        """模板：摘要回复"""
        search_results = result.final_data.get("search_results", [])

        if not search_results:
            return "未找到可总结的内容。"

        topic = plan.original_query.replace("总结", "").replace("摘要", "").strip()
        total_content = sum(len(r.get("content", "")) for r in search_results)

        lines = [f"关于「{topic}」的内容总结：", ""]
        lines.append(f"共汇总 {len(search_results)} 条笔记，总字数约 {total_content} 字。")
        lines.append("")

        lines.append("**主要内容：**")
        for i, item in enumerate(search_results[:5], 1):
            title = item.get("title", "无标题")
            content = item.get("content", "")[:100]
            if len(item.get("content", "")) > 100:
                content += "..."
            lines.append(f"{i}. {title}：{content}")

        lines.append("")
        lines.append("如需查看完整内容，请告诉我具体笔记编号。")

        return "\n".join(lines)

    def _template_create_md_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：创建MD文件回复"""
        created_md = result.final_data.get("created_md")

        if not created_md:
            return "创建MD文件失败，未获取到结果。"

        if not created_md.get("success"):
            return f"创建MD文件失败：{created_md.get('error', '未知错误')}"

        filename = created_md.get("filename", "")
        file_path = created_md.get("file_path", "")

        return f"MD文件创建成功！\n\n**文件名**：{filename}\n**路径**：{file_path}"

    def _template_write_md_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：写入MD文件回复"""
        written_md = result.final_data.get("written_md")

        if not written_md:
            return "写入MD文件失败，未获取到结果。"

        if not written_md.get("success"):
            return f"写入MD文件失败：{written_md.get('error', '未知错误')}"

        filename = written_md.get("filename", "")
        mode = written_md.get("mode", "")
        is_new = written_md.get("is_new", False)

        mode_text = "追加" if mode == "append" else "覆盖"
        status_text = "新文件" if is_new else "已有文件"

        return f"内容已写入MD文件！\n\n**文件名**：{filename}\n**写入模式**：{mode_text}\n**状态**：{status_text}"

    def _template_default_response(self, plan: Plan, result: ExecutionResult) -> str:
        """模板：默认回复"""
        if result.success:
            return "操作已完成。"
        else:
            error_msg = ""
            for r in result.results:
                if not r.success and r.error_message:
                    error_msg = r.error_message
                    break
            return f"操作失败：{error_msg or '未知错误'}"

    def _extract_sources(self, result: ExecutionResult) -> Optional[List[Dict]]:
        """提取引用来源"""
        search_results = result.final_data.get("search_results", [])
        if not search_results:
            return None

        return [
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "score": r.get("rerank_score"),
                "source_type": r.get("source_type", "note"),
                "filename": r.get("filename"),
                "section_title": r.get("section_title"),
            }
            for r in search_results
        ]


# 全局回复生成器实例
responder_instance = None


def get_responder() -> Responder:
    """获取回复生成器实例（单例模式）"""
    global responder_instance
    if responder_instance is None:
        responder_instance = Responder()
    return responder_instance
