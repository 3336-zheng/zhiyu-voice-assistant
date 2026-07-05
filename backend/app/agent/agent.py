"""
Plan-and-Execute Agent 主类（LangGraph 版本）
整合 Planner、Executor、Responder，支持多轮对话记忆
使用 LangGraph 状态机替代手写 if-else 编排
"""
import time
import logging
from typing import Optional, List, Dict
from datetime import datetime
import uuid

from sqlalchemy.orm import Session

from backend.app.agent.models import AgentResponse
from backend.app.agent.planner import get_planner
from backend.app.agent.executor import get_executor
from backend.app.agent.responder import get_responder
from backend.app.core.database import SessionLocal

logger = logging.getLogger(__name__)


class PlanExecuteAgent:
    """
    Plan-and-Execute Agent
    主执行流程：Plan → Execute → Respond
    支持多轮对话记忆
    """

    def __init__(self):
        """初始化 Agent"""
        self.planner = get_planner()
        self.executor = get_executor()
        self.responder = get_responder()
        self._memory_service = None

    @property
    def memory_service(self):
        """延迟加载记忆服务"""
        if self._memory_service is None:
            try:
                from backend.app.services.memory_service import get_memory_service
                self._memory_service = get_memory_service()
            except Exception as e:
                logger.warning(f"记忆服务加载失败: {e}")
        return self._memory_service

    def _get_session_id(self, session_id: Optional[str] = None) -> str:
        """
        获取或生成会话 ID

        Args:
            session_id: 可选的会话 ID

        Returns:
            str: 会话 ID
        """
        if session_id:
            return session_id
        return f"session_{uuid.uuid4().hex[:16]}"

    async def run(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        db: Session = None
    ) -> AgentResponse:
        """
        执行 Agent 主流程（使用 LangGraph 图状态机）

        Args:
            user_query: 用户输入
            session_id: 会话 ID（可选，用于多轮对话）
            db: 数据库会话

        Returns:
            AgentResponse: Agent 响应
        """
        start_time = time.time()
        session_id = self._get_session_id(session_id)

        logger.info(f"Agent 开始处理查询: '{user_query[:50]}...'" if len(user_query) > 50 else f"Agent 开始处理查询: '{user_query}'")
        logger.info(f"会话 ID: {session_id}")

        # 管理数据库会话
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False

        try:
            # 获取对话上下文
            context = []
            if self.memory_service:
                try:
                    context = self.memory_service.get_history(session_id, db=db)
                except Exception as e:
                    logger.warning(f"获取对话历史失败: {e}")

            # 使用 LangGraph 图状态机执行
            from backend.app.agent.graph import get_agent_graph
            graph = get_agent_graph()

            # 构建初始状态
            initial_state = {
                "query": user_query,
                "session_id": session_id,
                "context": context,
                "plan": None,
                "search_results": None,
                "execution_results": None,
                "answer": None,
                "sources": None,
                "confidence": 0.0,
                "iter_count": 0,
                "max_iterations": 5,
                "error": None
            }

            # 执行图
            logger.info("执行 LangGraph 图状态机...")
            final_state = graph.invoke(initial_state)

            # 提取结果
            answer = final_state.get("answer", "抱歉，无法生成答案。")
            sources = final_state.get("sources", [])
            confidence = final_state.get("confidence", 0.0)
            error = final_state.get("error")

            if error:
                logger.warning(f"图执行过程中出现错误: {error}")

            total_time = int((time.time() - start_time) * 1000)

            # 构建响应
            response = AgentResponse(
                query=user_query,
                response=answer,
                session_id=session_id,
                sources=sources,
                confidence=confidence,
                timestamp=datetime.now(),
                execution_time_ms=total_time
            )

            # 保存对话历史
            if self.memory_service:
                try:
                    # 保存用户消息
                    self.memory_service.add_message(
                        session_id=session_id,
                        role="user",
                        content=user_query,
                        db=db
                    )

                    # 保存助手回复
                    self.memory_service.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=answer,
                        metadata={"sources": sources},
                        db=db
                    )

                    # 检查是否需要摘要压缩
                    self.memory_service.summarize_if_needed(session_id, db=db)
                except Exception as e:
                    logger.warning(f"保存对话历史失败: {e}")

            logger.info(f"Agent 处理完成，总耗时 {total_time}ms")
            return response

        except Exception as e:
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            return AgentResponse(
                query=user_query,
                response=f"抱歉，处理您的请求时出现错误：{str(e)}",
                confidence=0.0,
                timestamp=datetime.now(),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

        finally:
            if should_close:
                db.close()

    async def chat(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        db: Session = None
    ) -> str:
        """
        简化版对话接口，只返回回复文本

        Args:
            user_query: 用户输入
            session_id: 会话 ID
            db: 数据库会话

        Returns:
            str: 回复文本
        """
        response = await self.run(user_query, session_id, db)
        return response.response

    def clear_session(self, session_id: str, db: Session = None) -> bool:
        """
        清除会话历史

        Args:
            session_id: 会话 ID
            db: 数据库会话

        Returns:
            bool: 是否成功
        """
        if self.memory_service:
            return self.memory_service.clear_session(session_id, db)
        return False

    def list_sessions(self, db: Session = None) -> List[Dict]:
        """
        列出所有会话

        Args:
            db: 数据库会话

        Returns:
            List[Dict]: 会话列表
        """
        if self.memory_service:
            return self.memory_service.list_sessions(db)
        return []


# 全局 Agent 实例
agent_instance = None


def get_agent() -> PlanExecuteAgent:
    """获取 Agent 实例（单例模式）"""
    global agent_instance
    if agent_instance is None:
        agent_instance = PlanExecuteAgent()
    return agent_instance
