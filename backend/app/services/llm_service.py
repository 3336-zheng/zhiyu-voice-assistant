"""
LLM 服务
使用 OpenAI 兼容 SDK 调用 DeepSeek 等大模型
支持 Langfuse 可观测
"""
import json
import logging
from typing import List, Dict, Any, Optional, Generator
from openai import OpenAI
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM 服务
    使用 OpenAI 兼容接口调用大模型
    """

    def __init__(self):
        """初始化 LLM 服务"""
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_url,
        )
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

        # Langfuse 可观测
        self._langfuse = None
        self._init_langfuse()

    def _init_langfuse(self):
        """初始化 Langfuse（如果配置了）"""
        try:
            langfuse_host = getattr(settings, 'langfuse_host', None)
            langfuse_public_key = getattr(settings, 'langfuse_public_key', None)
            langfuse_secret_key = getattr(settings, 'langfuse_secret_key', None)

            if langfuse_host and langfuse_public_key and langfuse_secret_key:
                from langfuse import Langfuse
                self._langfuse = Langfuse(
                    host=langfuse_host,
                    public_key=langfuse_public_key,
                    secret_key=langfuse_secret_key
                )
                logger.info("Langfuse 可观测已启用")
            else:
                logger.debug("Langfuse 未配置，跳过")
        except Exception as e:
            logger.warning(f"Langfuse 初始化失败: {e}")
            self._langfuse = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        response_format: Dict = None,
        trace_name: str = None
    ) -> str:
        """
        发送对话请求

        Args:
            messages: 对话消息列表，格式: [{"role": "user", "content": "..."}]
            temperature: 温度参数（可选）
            max_tokens: 最大生成 token 数（可选）
            response_format: 响应格式（可选）
            trace_name: Langfuse trace 名称（可选）

        Returns:
            str: 模型回复内容
        """
        import time
        start_time = time.time()

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            duration_ms = int((time.time() - start_time) * 1000)

            logger.debug(f"LLM 调用成功，回复长度: {len(content)}，耗时: {duration_ms}ms")

            # Langfuse trace
            if self._langfuse and trace_name:
                try:
                    self._langfuse.trace(
                        name=trace_name,
                        input=messages[-1].get("content", "") if messages else "",
                        output=content,
                        metadata={
                            "model": self.model,
                            "temperature": temperature or self.temperature,
                            "max_tokens": max_tokens or self.max_tokens,
                            "duration_ms": duration_ms,
                            "tokens_used": response.usage.total_tokens if response.usage else None
                        }
                    )
                except Exception as e:
                    logger.warning(f"Langfuse trace 失败: {e}")

            return content

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise RuntimeError(f"LLM 服务调用失败: {str(e)}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        发送对话请求并返回 JSON 格式结果

        Args:
            messages: 对话消息列表
            temperature: 温度参数

        Returns:
            Dict: 解析后的 JSON 结果
        """
        content = self.chat(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if json_match:
                return json.loads(json_match.group(1).strip())
            raise RuntimeError(f"无法解析 LLM 返回的 JSON: {content[:200]}")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> Generator[str, None, None]:
        """
        流式对话请求

        Args:
            messages: 对话消息列表
            temperature: 温度参数（可选）
            max_tokens: 最大生成 token 数（可选）

        Yields:
            str: 增量 token
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": True
            }

            stream = self.client.chat.completions.create(**kwargs)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            yield f"[错误] {str(e)}"

    def generate_summary(self, content: str, max_length: int = 200) -> str:
        """
        生成文本摘要

        Args:
            content: 待摘要的文本
            max_length: 摘要最大长度

        Returns:
            str: 生成的摘要
        """
        messages = [
            {
                "role": "system",
                "content": f"你是一个专业的文本摘要助手。请对以下文本生成简洁的摘要，不超过{max_length}字。"
            },
            {
                "role": "user",
                "content": content
            }
        ]

        return self.chat(messages=messages, max_tokens=500)

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        从文本中提取关键词

        Args:
            text: 输入文本
            top_k: 返回关键词数量

        Returns:
            List[str]: 关键词列表
        """
        messages = [
            {
                "role": "system",
                "content": f"请从以下文本中提取{top_k}个最重要的关键词，以JSON数组格式返回。"
            },
            {
                "role": "user",
                "content": text
            }
        ]

        result = self.chat_json(messages=messages)
        if isinstance(result, list):
            return result[:top_k]
        elif isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v[:top_k]
        return []


# 全局服务实例
llm_service_instance = None


def get_llm_service() -> LLMService:
    """获取 LLM 服务实例（单例模式）"""
    global llm_service_instance
    if llm_service_instance is None:
        llm_service_instance = LLMService()
    return llm_service_instance
