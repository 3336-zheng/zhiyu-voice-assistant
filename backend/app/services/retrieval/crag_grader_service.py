"""
CRAG (Corrective RAG) 评分服务
检索后用 LLM 给文档打相关性分，根据结果决定是否改写重检
"""
import logging
from typing import List, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RelevanceGrade(str, Enum):
    """相关性等级"""
    CORRECT = "correct"  # 高相关，直接生成
    AMBIGUOUS = "ambiguous"  # 部分相关，需要精炼
    INCORRECT = "incorrect"  # 不相关，需要改写重检


class CRAGGraderService:
    """
    CRAG 评分服务
    用 LLM 给召回文档打相关性分
    """

    def __init__(self):
        """初始化"""
        self._llm_service = None

    @property
    def llm_service(self):
        """延迟加载 LLM 服务"""
        if self._llm_service is None:
            try:
                from backend.app.services.ai.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            except Exception as e:
                logger.warning(f"LLM 服务加载失败: {e}")
        return self._llm_service

    def grade_documents(
        self,
        query: str,
        documents: List[Dict]
    ) -> Dict:
        """
        评估文档相关性

        Args:
            query: 用户查询
            documents: 检索到的文档列表

        Returns:
            Dict: {
                "grade": RelevanceGrade,
                "scores": List[Dict],  # 每个文档的评分
                "reasoning": str  # 评分理由
            }
        """
        if not documents:
            return {
                "grade": RelevanceGrade.INCORRECT,
                "scores": [],
                "reasoning": "没有检索到任何文档"
            }

        if not self.llm_service:
            logger.warning("LLM 服务不可用，跳过 CRAG 评分")
            return {
                "grade": RelevanceGrade.CORRECT,
                "scores": [{"doc_id": i, "grade": "correct", "score": 0.5} for i in range(len(documents))],
                "reasoning": "LLM 服务不可用，默认返回 correct"
            }

        try:
            # 构建文档摘要
            doc_summaries = []
            for i, doc in enumerate(documents[:5]):  # 只评估前5个文档
                content = doc.get("content", "")[:200]  # 截取前200字
                doc_summaries.append(f"文档{i+1}: {content}")

            doc_text = "\n\n".join(doc_summaries)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个文档相关性评估助手。请评估以下文档与用户查询的相关性。\n\n"
                        "评分标准：\n"
                        "- correct: 文档与查询高度相关，包含回答查询所需的信息\n"
                        "- ambiguous: 文档与查询部分相关，可能需要结合其他信息\n"
                        "- incorrect: 文档与查询不相关，无法帮助回答查询\n\n"
                        "请以 JSON 格式返回：\n"
                        "{\n"
                        '  "grade": "correct/ambiguous/incorrect",\n'
                        '  "scores": [\n'
                        '    {"doc_id": 0, "grade": "correct/ambiguous/incorrect", "score": 0.0-1.0},\n'
                        "    ...\n"
                        "  ],\n"
                        '  "reasoning": "评分理由"\n'
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": f"用户查询：{query}\n\n文档列表：\n{doc_text}\n\n请评估相关性："
                }
            ]

            response = self.llm_service.chat_json(
                messages=messages,
                temperature=0.1
            )

            # 解析结果
            grade_str = response.get("grade", "incorrect")
            scores = response.get("scores", [])
            reasoning = response.get("reasoning", "")

            # 映射等级
            grade_map = {
                "correct": RelevanceGrade.CORRECT,
                "ambiguous": RelevanceGrade.AMBIGUOUS,
                "incorrect": RelevanceGrade.INCORRECT
            }
            grade = grade_map.get(grade_str, RelevanceGrade.INCORRECT)

            logger.info(f"CRAG 评分完成: grade={grade.value}, 理由='{reasoning[:50]}...'")

            return {
                "grade": grade,
                "scores": scores,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error(f"CRAG 评分失败: {e}")
            # 失败时默认返回 ambiguous，触发重试
            return {
                "grade": RelevanceGrade.AMBIGUOUS,
                "scores": [],
                "reasoning": f"评分失败: {str(e)}"
            }

    def refine_documents(
        self,
        query: str,
        documents: List[Dict]
    ) -> str:
        """
        知识精炼：从文档中提取与查询相关的关键信息

        Args:
            query: 用户查询
            documents: 检索到的文档列表

        Returns:
            str: 精炼后的关键信息
        """
        if not documents:
            return ""

        if not self.llm_service:
            logger.warning("LLM 服务不可用，跳过知识精炼")
            return "\n\n".join([doc.get("content", "")[:200] for doc in documents[:3]])

        try:
            # 构建文档内容
            doc_contents = []
            for i, doc in enumerate(documents[:5]):
                content = doc.get("content", "")[:300]
                doc_contents.append(f"文档{i+1}:\n{content}")

            doc_text = "\n\n".join(doc_contents)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个知识精炼助手。请从文档中提取与用户查询直接相关的关键信息。\n"
                        "要求：\n"
                        "1. 只提取与查询相关的信息\n"
                        "2. 去除无关内容\n"
                        "3. 保持信息的完整性\n"
                        "4. 使用中文"
                    )
                },
                {
                    "role": "user",
                    "content": f"用户查询：{query}\n\n文档内容：\n{doc_text}\n\n请提取关键信息："
                }
            ]

            refined = self.llm_service.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )

            logger.info(f"知识精炼完成: 输入文档数={len(documents)}, 输出长度={len(refined)}")
            return refined

        except Exception as e:
            logger.error(f"知识精炼失败: {e}")
            # 失败时返回原文档摘要
            return "\n\n".join([doc.get("content", "")[:200] for doc in documents[:3]])


# 全局实例
crag_grader_service = None


def get_crag_grader_service() -> CRAGGraderService:
    """获取 CRAG 评分服务实例（单例模式）"""
    global crag_grader_service
    if crag_grader_service is None:
        crag_grader_service = CRAGGraderService()
    return crag_grader_service
