"""
Plan-and-Execute Agent 主类
整合 Planner、Executor、Responder，支持多轮对话记忆
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
        执行 Agent 主流程

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

            # Step 1: Plan - 分析意图，生成计划
            logger.info("Step 1: 生成执行计划...")
            plan = self.planner.plan(user_query, context)
            logger.info(f"计划生成完成：意图={plan.intent.value}, 步骤数={len(plan.steps)}")

            # Step 2: Execute - 执行计划
            logger.info("Step 2: 执行计划...")
            execution_result = await self.executor.execute(plan, db)
            logger.info(f"计划执行完成：成功步骤={execution_result.completed_steps}/{execution_result.total_steps}")

            # Step 3: Respond - 生成回复
            logger.info("Step 3: 生成回复...")
            response = self.responder.generate_response(user_query, plan, execution_result, context)

            total_time = int((time.time() - start_time) * 1000)
            response.execution_time_ms = total_time

            # 保存对话历史
            if self.memory_service:
                try:
                    # 保存用户消息
                    self.memory_service.add_message(
                        session_id=session_id,
                        role="user",
                        content=user_query,
                        intent=plan.intent.value,
                        db=db
                    )

                    # 保存助手回复
                    self.memory_service.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=response.response,
                        intent=plan.intent.value,
                        metadata={"sources": response.sources},
                        db=db
                    )

                    # 检查是否需要摘要压缩
                    self.memory_service.summarize_if_needed(session_id, db=db)
                except Exception as e:
                    logger.warning(f"保存对话历史失败: {e}")

            logger.info(f"Agent 处理完成，总耗时 {total_time}ms")

            # 将 session_id 附加到响应
            response_dict = response.dict()
            response_dict["session_id"] = session_id
            return AgentResponse(**response_dict)

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
