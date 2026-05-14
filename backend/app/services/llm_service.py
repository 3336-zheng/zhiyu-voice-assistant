"""
LLM 服务
使用 OpenAI 兼容 SDK 调用 DeepSeek 等大模型
"""
import json
import logging
from typing import List, Dict, Any, Optional
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

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        response_format: Dict = None
    ) -> str:
        """
        发送对话请求

        Args:
            messages: 对话消息列表，格式: [{"role": "user", "content": "..."}]
            temperature: 温度参数（可选）
            max_tokens: 最大生成 token 数（可选）
            response_format: 响应格式（可选）

        Returns:
            str: 模型回复内容
        """
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
            logger.debug(f"LLM 调用成功，回复长度: {len(content)}")
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
