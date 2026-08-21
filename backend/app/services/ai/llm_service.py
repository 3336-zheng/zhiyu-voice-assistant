"""
LLM 服务
使用 OpenAI 兼容 SDK 通过 Vercel AI Gateway 调用大模型
支持 Langfuse 可观测
"""
import json
import logging
import time
from typing import List, Dict, Any, Optional, Generator
from openai import OpenAI
from backend.app.core.config import settings
from backend.app.core.observability import get_request_id, record_model_usage, timed_stage
from backend.app.agent.events import AgentRunCancelled

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
            timeout=settings.llm_timeout_seconds,
        )
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
        self.fallback_client = None
        self.fallback_model = settings.llm_fallback_model.strip()
        # disabled 避免已知不兼容网关在首个请求上浪费一次往返；auto 仍可探测。
        mode = settings.llm_response_format_mode
        self._json_response_format_supported: Optional[bool] = (
            None if mode == "auto" else mode == "enabled"
        )
        if settings.llm_fallback_enabled and self.fallback_model:
            self.fallback_client = OpenAI(
                api_key=settings.llm_fallback_api_key or settings.llm_api_key,
                base_url=settings.llm_fallback_api_url or settings.llm_api_url,
                timeout=settings.llm_timeout_seconds,
            )

        # Langfuse 可观测
        self._langfuse = None
        self._init_langfuse()

    def _init_langfuse(self):
        """初始化 Langfuse（如果配置了）"""
        try:
            langfuse_host = getattr(settings, 'langfuse_host', None)
            langfuse_public_key = getattr(settings, 'langfuse_public_key', None)
            langfuse_secret_key = getattr(settings, 'langfuse_secret_key', None)

            if (
                settings.observability_enabled
                and langfuse_host
                and langfuse_public_key
                and langfuse_secret_key
            ):
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

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """只对连接、超时、限流和服务端错误执行故障转移。"""
        status_code = getattr(exc, "status_code", None)
        if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
            return True
        return type(exc).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        }

    @staticmethod
    def _usage_values(response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return prompt_tokens, completion_tokens, total_tokens

    @staticmethod
    def _estimated_cost(
        prompt_tokens: int,
        completion_tokens: int,
        fallback_used: bool,
    ) -> float:
        if fallback_used:
            input_rate = settings.llm_fallback_input_cost_per_million
            output_rate = settings.llm_fallback_output_cost_per_million
        else:
            input_rate = settings.llm_input_cost_per_million
            output_rate = settings.llm_output_cost_per_million
        return (
            prompt_tokens * input_rate + completion_tokens * output_rate
        ) / 1_000_000

    def _record_attempt(
        self,
        *,
        model: str,
        provider: str,
        duration_ms: float,
        fallback_used: bool,
        response: Any = None,
        error: Optional[Exception] = None,
        operation: Optional[str] = None,
        finish_reason: Optional[str] = None,
        token_budget: Optional[int] = None,
    ) -> None:
        prompt_tokens, completion_tokens, total_tokens = self._usage_values(response)
        resolved_finish_reason = finish_reason or self._finish_reason(response)
        record_model_usage(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=self._estimated_cost(
                prompt_tokens,
                completion_tokens,
                fallback_used,
            ),
            duration_ms=duration_ms,
            fallback_used=fallback_used,
            success=error is None,
            error_type=type(error).__name__ if error else None,
            operation=operation,
            finish_reason=resolved_finish_reason,
            truncated=resolved_finish_reason == "length",
            token_budget=token_budget,
            context_window_tokens=settings.llm_context_window_tokens,
        )

    @staticmethod
    def _finish_reason(response: Any) -> Optional[str]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        value = getattr(choices[0], "finish_reason", None)
        return str(value) if value else None

    def _chat_completion(
        self,
        kwargs: Dict[str, Any],
        *,
        model: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> tuple[Any, str, bool]:
        primary_model = (model or "").strip() or self.model
        attempts = [(self.client, primary_model, "primary", False)]
        if self.fallback_client is not None:
            attempts.append((self.fallback_client, self.fallback_model, "fallback", True))

        last_error: Optional[Exception] = None
        for index, (client, model, provider, fallback_used) in enumerate(attempts):
            attempt_kwargs = {**kwargs, "model": model}
            started = time.perf_counter()
            try:
                with timed_stage(f"llm.{provider}"):
                    response = client.chat.completions.create(**attempt_kwargs)
                duration_ms = (time.perf_counter() - started) * 1000
                self._record_attempt(
                    model=model,
                    provider=provider,
                    duration_ms=duration_ms,
                    fallback_used=fallback_used,
                    response=response,
                    operation=operation,
                    token_budget=attempt_kwargs.get("max_tokens"),
                )
                return response, model, fallback_used
            except AgentRunCancelled:
                raise
            except Exception as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                self._record_attempt(
                    model=model,
                    provider=provider,
                    duration_ms=duration_ms,
                    fallback_used=fallback_used,
                    error=exc,
                    operation=operation,
                    token_budget=attempt_kwargs.get("max_tokens"),
                )
                last_error = exc
                can_fallback = index == 0 and len(attempts) > 1 and self._is_retryable_error(exc)
                if not can_fallback:
                    break
                logger.warning("主模型暂时不可用，切换备用模型: %s", type(exc).__name__)
        raise last_error or RuntimeError("LLM 调用失败")

    def _trace_langfuse(
        self,
        trace_name: Optional[str],
        messages: List[Dict[str, str]],
        content: str,
        model: str,
        duration_ms: int,
        response: Any,
        fallback_used: bool,
    ) -> None:
        if not self._langfuse or not trace_name:
            return
        try:
            capture_content = settings.observability_capture_content
            prompt_tokens, completion_tokens, total_tokens = self._usage_values(response)
            metadata = {
                "request_id": get_request_id(),
                "model": model,
                "duration_ms": duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "fallback_used": fallback_used,
                "content_captured": capture_content,
            }
            if hasattr(self._langfuse, "start_generation"):
                generation = self._langfuse.start_generation(
                    name=trace_name,
                    input=messages[-1].get("content", "") if messages and capture_content else None,
                    output=content if capture_content else None,
                    metadata=metadata,
                    model=model,
                    usage_details={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    },
                )
                generation.end()
            else:
                self._langfuse.trace(
                    name=trace_name,
                    input=messages[-1].get("content", "") if messages and capture_content else None,
                    output=content if capture_content else None,
                    metadata=metadata,
                )
        except Exception as exc:
            logger.warning("Langfuse trace 失败: %s", exc)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        response_format: Dict = None,
        trace_name: str = None,
        model: str = None,
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
        start_time = time.time()

        try:
            kwargs = {
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response, used_model, fallback_used = self._chat_completion(
                kwargs,
                model=model,
                operation=trace_name,
            )
            content = response.choices[0].message.content or ""
            duration_ms = int((time.time() - start_time) * 1000)
            finish_reason = self._finish_reason(response)

            logger.debug(f"LLM 调用成功，回复长度: {len(content)}，耗时: {duration_ms}ms")
            if finish_reason == "length":
                logger.warning(
                    "LLM 输出达到 Token 上限: operation=%s, model=%s, length=%s",
                    trace_name or "chat",
                    used_model,
                    len(content),
                )

            self._trace_langfuse(
                trace_name,
                messages,
                content,
                used_model,
                duration_ms,
                response,
                fallback_used,
            )

            return content

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise RuntimeError(f"LLM 服务调用失败: {str(e)}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        trace_name: str = None,
    ) -> Dict[str, Any]:
        """
        发送对话请求并返回 JSON 格式结果

        Args:
            messages: 对话消息列表
            temperature: 温度参数
            max_tokens: 最大生成 Token 数

        Returns:
            Dict: 解析后的 JSON 结果
        """
        response_format = {"type": "json_object"}
        if self._json_response_format_supported is not False:
            try:
                content = self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    model=model,
                    trace_name=trace_name,
                )
                self._json_response_format_supported = True
            except RuntimeError as exc:
                if not self._is_response_format_rejection(exc):
                    raise
                self._json_response_format_supported = False
                logger.warning(
                    "当前 LLM 网关不支持 response_format，降级为提示词约束 JSON: %s",
                    str(exc)[:240],
                )
                content = self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    trace_name=trace_name,
                )
        else:
            content = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                trace_name=trace_name,
            )

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if json_match:
                return json.loads(json_match.group(1).strip())
            raise RuntimeError(f"无法解析 LLM 返回的 JSON: {content[:200]}")

    @staticmethod
    def _is_response_format_rejection(exc: Exception) -> bool:
        """识别网关不支持 response_format 的请求错误。"""
        message = str(exc).lower()
        if "response_format" not in message:
            return False
        return any(
            marker in message
            for marker in ("invalid", "unsupported", "not support", "unknown")
        )

    def call_function(
        self,
        messages: List[Dict[str, str]],
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        temperature: float = 0,
        max_tokens: int = None,
        model: str = None,
        trace_name: str = None,
    ) -> Dict[str, Any]:
        """强制调用一个 OpenAI 兼容函数，并返回经过 JSON 解析的参数。"""
        function_name = (name or "").strip()
        if not function_name or not isinstance(parameters, dict):
            raise ValueError("Function Calling 需要函数名和 JSON Schema")
        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": function_name},
            },
        }
        try:
            response, _, _ = self._chat_completion(
                kwargs,
                model=model,
                operation=trace_name,
            )
            tool_calls = response.choices[0].message.tool_calls or []
            if len(tool_calls) != 1 or tool_calls[0].function.name != function_name:
                raise RuntimeError("模型未按要求返回唯一函数调用")
            arguments = json.loads(tool_calls[0].function.arguments)
            if not isinstance(arguments, dict):
                raise RuntimeError("Function Calling 参数根节点必须是对象")
            return arguments
        except json.JSONDecodeError as exc:
            raise RuntimeError("Function Calling 返回了无效 JSON 参数") from exc

    def structured_call(
        self,
        messages: List[Dict[str, str]],
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        temperature: float = 0,
        max_tokens: int = None,
        model: str = None,
        trace_name: str = None,
    ) -> Dict[str, Any]:
        """优先使用 Function Calling，网关不兼容时回退到 JSON 输出。"""
        mode = settings.llm_structured_output_mode
        if mode != "json":
            try:
                return self.call_function(
                    messages,
                    name=name,
                    description=description,
                    parameters=parameters,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    trace_name=trace_name,
                )
            except AgentRunCancelled:
                raise
            except Exception as exc:
                if mode == "function_call":
                    raise
                logger.warning(
                    "Function Calling 失败，降级为 JSON 输出: operation=%s, error=%s",
                    trace_name or name,
                    str(exc)[:240],
                )

        schema_instruction = {
            "role": "system",
            "content": (
                "严格返回一个符合下列 JSON Schema 的 JSON 对象，不要输出代码块或额外说明：\n"
                + json.dumps(parameters, ensure_ascii=False)
            ),
        }
        return self.chat_json(
            [schema_instruction, *messages],
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            trace_name=trace_name,
        )

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        trace_name: str = None,
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
        primary_model = (model or "").strip() or self.model
        attempts = [(self.client, primary_model, "primary", False)]
        if self.fallback_client is not None:
            attempts.append((self.fallback_client, self.fallback_model, "fallback", True))

        for index, (client, model, provider, fallback_used) in enumerate(attempts):
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            started = time.perf_counter()
            emitted = False
            final_chunk = None
            finish_reason = None
            try:
                stream = client.chat.completions.create(**kwargs)
                for chunk in stream:
                    final_chunk = chunk
                    if chunk.choices and getattr(chunk.choices[0], "finish_reason", None):
                        finish_reason = chunk.choices[0].finish_reason
                    if chunk.choices and getattr(chunk.choices[0].delta, "content", None):
                        emitted = True
                        yield chunk.choices[0].delta.content
                self._record_attempt(
                    model=model,
                    provider=provider,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    fallback_used=fallback_used,
                    response=final_chunk,
                    operation=trace_name,
                    finish_reason=finish_reason,
                    token_budget=kwargs.get("max_tokens"),
                )
                if finish_reason == "length":
                    logger.warning(
                        "LLM 流式输出达到 Token 上限: operation=%s, model=%s",
                        trace_name or "stream_chat",
                        model,
                    )
                return
            except AgentRunCancelled:
                raise
            except Exception as exc:
                self._record_attempt(
                    model=model,
                    provider=provider,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    fallback_used=fallback_used,
                    error=exc,
                    operation=trace_name,
                    token_budget=kwargs.get("max_tokens"),
                )
                can_fallback = (
                    not emitted
                    and index == 0
                    and len(attempts) > 1
                    and self._is_retryable_error(exc)
                )
                if can_fallback:
                    logger.warning("主模型流尚未开始，切换备用模型: %s", type(exc).__name__)
                    continue
                logger.error("LLM 流式调用失败: %s", exc)
                raise RuntimeError(f"LLM 流式服务调用失败: {exc}") from exc

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
