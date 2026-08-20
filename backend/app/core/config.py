"""
应用配置管理
"""
import json
import os
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # 应用配置
    app_name: str = "智语端侧智能语音笔记助手"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8337

    # 数据库配置
    database_url: str = "sqlite:///data/database/notes.db"

    # 模型路径配置（需在 .env 中填写实际路径）
    whisper_model_path: str = ""
    embedding_provider: Literal["local", "openai_compatible"] = "local"
    embedding_model_path: str = ""
    embedding_api_key: str = ""
    embedding_api_url: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = Field(default=0, ge=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=2048)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_max_retries: int = Field(default=2, ge=0, le=10)
    reranker_provider: Literal["local", "rerank_compatible"] = "local"
    reranker_model_path: str = ""
    reranker_api_key: str = ""
    reranker_api_url: str = ""
    reranker_model: str = ""
    reranker_timeout_seconds: float = Field(default=30.0, gt=0)

    # 文件存储配置
    upload_dir: str = "data/uploads"
    wiki_pages_dir: str = "data/wiki/pages"
    wiki_attachments_dir: str = "data/wiki/attachments"
    wiki_exports_dir: str = "data/wiki/exports"
    backup_dir: str = "data/backups"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: str = ".wav,.mp3,.flac,.ogg,.webm"  # 从 .env 读取为字符串

    # ChromaDB 向量数据库配置
    chroma_persist_path: str = "data/database/chromadb"
    chroma_collection_name: str = "notes"

    # 混合检索配置（新增）
    rrf_k: float = 60.0  # RRF 融合常数
    rrf_top_k: int = 30  # 兼容旧混合检索入口的融合候选数
    bm25_top_k: int = 20  # BM25 检索数量
    embedding_top_k: int = 20  # Embedding 检索数量
    rag_rerank_candidate_top_k: int = Field(default=12, ge=5, le=100)
    retrieval_rerank_min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    retrieval_rerank_score_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    retrieval_max_workers: int = Field(default=8, ge=2, le=32)

    # RAG v2 配置。关闭后继续使用原有的逐查询完整检索链路。
    rag_v2_enabled: bool = True
    rag_parent_child_enabled: bool = True
    rag_parent_chunk_chars: int = Field(default=1200, ge=200, le=20_000)
    rag_parent_chunk_overlap_chars: int = Field(default=120, ge=0, le=5_000)
    rag_child_chunk_chars: int = Field(default=500, ge=80, le=10_000)
    rag_child_chunk_overlap_chars: int = Field(default=80, ge=0, le=2_000)
    rag_context_token_budget: int = 3000
    rag_final_top_k: int = 5

    # Agent 配置（新增）
    agent_max_iterations: int = 5  # Agent 最大执行轮数
    agent_run_timeout_seconds: float = 180.0
    agent_event_buffer_size: int = 2000
    agent_run_retention_seconds: int = 3600
    agent_tool_context_token_budget: int = 4000
    agent_plan_max_steps: int = Field(default=6, ge=1, le=20)
    agent_max_replans: int = Field(default=1, ge=0, le=3)

    # ASR 引擎配置
    asr_provider: Literal["whisper", "dashscope", "mimo"] = "whisper"
    asr_timeout_seconds: float = Field(default=180.0, gt=0, le=3_600)
    audio_normalize_timeout_seconds: float = Field(default=120.0, gt=0, le=3_600)
    audio_probe_timeout_seconds: float = Field(default=15.0, gt=0, le=300)

    # 百炼 DashScope ASR 配置
    dashscope_asr_api_key: str = ""
    dashscope_asr_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_asr_model: str = "paraformer-realtime-v2"

    # 小米 MiMo ASR 配置
    mimo_asr_api_key: str = ""
    mimo_asr_api_url: str = "https://api.xiaomimimo.com/v1"
    mimo_asr_model: str = "mimo-v2.5-asr"
    mimo_asr_request_timeout_seconds: float = Field(default=120.0, gt=0, le=3_600)
    mimo_asr_max_base64_bytes: int = Field(default=10 * 1024 * 1024, ge=1_024)

    # LLM 配置（Vercel AI Gateway 的 OpenAI 兼容接口）
    llm_api_key: str = ""
    llm_api_url: str = "https://ai-gateway.vercel.sh/v1"
    llm_model: str = "deepseek/deepseek-v4-flash"
    # 各阶段可独立选用低延迟模型；空值时回退到 LLM_MODEL。
    llm_planner_model: str = "openai/gpt-4.1-mini"
    llm_query_rewrite_model: str = "openai/gpt-4.1-nano"
    llm_crag_model: str = "openai/gpt-4.1-mini"
    llm_responder_model: str = "openai/gpt-4.1-mini"
    llm_max_tokens: int = Field(default=1024, ge=128, le=16_000)
    llm_response_max_tokens: int = Field(default=1024, ge=128, le=8_000)
    llm_context_window_tokens: int = Field(default=16_000, ge=4_096)
    llm_temperature: float = 0.7
    llm_timeout_seconds: float = 60.0
    llm_fallback_enabled: bool = False
    llm_fallback_api_key: str = ""
    llm_fallback_api_url: str = ""
    llm_fallback_model: str = ""
    llm_input_cost_per_million: float = 0.0
    llm_output_cost_per_million: float = 0.0
    llm_fallback_input_cost_per_million: float = 0.0
    llm_fallback_output_cost_per_million: float = 0.0
    llm_response_format_mode: Literal["disabled", "auto", "enabled"] = "disabled"
    llm_structured_output_mode: Literal["auto", "function_call", "json"] = "auto"

    # 对话记忆配置（新增）
    memory_max_history: int = 20  # 最大保留对话轮数
    memory_summary_threshold: int = 10  # 超过此轮数时触发摘要压缩
    memory_context_token_budget: int = Field(default=3_000, ge=256)
    memory_summary_token_budget: int = Field(default=600, ge=128)
    memory_summary_trigger_tokens: int = Field(default=4_000, ge=256)
    memory_summary_input_token_budget: int = Field(default=6_000, ge=512)

    # 会话过期清理配置
    session_ttl_hours: int = 24  # 会话过期时间（小时），超过此时间未更新的会话将被清理
    cleanup_interval_minutes: int = 60  # 清理任务执行间隔（分钟）

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "data/logs/app.log"
    log_error_file: str = "data/logs/error.log"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1_048_576)
    log_backup_count: int = Field(default=7, ge=1, le=90)

    # Langfuse 可观测配置（可选）
    langfuse_host: str = ""  # Langfuse 服务地址
    langfuse_public_key: str = ""  # Langfuse 公钥
    langfuse_secret_key: str = ""  # Langfuse 私钥
    observability_enabled: bool = True
    observability_capture_content: bool = False
    observability_trace_api_enabled: bool = True
    observability_trace_allow_remote: bool = False
    otel_enabled: bool = False
    otel_service_name: str = "zhiyu-wiki"
    otel_exporter_endpoint: str = ""

    # CORS 配置
    cors_origins: str = "*"  # 逗号分隔的域名列表，如 "https://example.com,https://app.example.com"

    # 性能配置
    max_concurrent_requests: int = 10
    timeout_seconds: int = 300

    # Wiki 索引 worker 配置
    wiki_index_poll_seconds: float = 5.0
    wiki_index_batch_size: int = 5
    wiki_index_max_backoff_seconds: int = 300

    # 旧 data/docs 兼容索引；新部署默认只使用 Wiki 主数据。
    sync_legacy_docs_on_startup: bool = False

    # 可信问答证据门禁
    evidence_min_score: float = 0.35
    evidence_min_sources: int = 1

    # CRAG 证据相关性双阈值。最终等级由后端根据最高有效证据分数裁决。
    crag_upper_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    crag_lower_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # 低于该分数直接拒答；介于该值和下阈值之间时最多恢复检索一次。
    crag_recovery_min_score: float = Field(default=0.1, ge=0.0, le=1.0)

    # 简单只读查询快速路径。
    fast_path_enabled: bool = True
    fast_path_max_query_chars: int = Field(default=160, ge=20, le=2_000)
    fast_path_rerank_min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    fast_path_min_sources: int = Field(default=1, ge=1, le=20)

    # 远程查询短 TTL 缓存，索引内容变化时由 collection count 参与检索缓存键。
    query_embedding_cache_ttl_seconds: float = Field(default=60.0, ge=0.0, le=3_600)
    query_embedding_cache_max_entries: int = Field(default=256, ge=0, le=10_000)
    query_rewrite_cache_ttl_seconds: float = Field(default=300.0, ge=0.0, le=3_600)
    query_rewrite_cache_max_entries: int = Field(default=256, ge=0, le=10_000)
    retrieval_cache_ttl_seconds: float = Field(default=20.0, ge=0.0, le=3_600)
    retrieval_cache_max_entries: int = Field(default=128, ge=0, le=10_000)
    crag_cache_ttl_seconds: float = Field(default=60.0, ge=0.0, le=3_600)
    crag_cache_max_entries: int = Field(default=128, ge=0, le=10_000)

    # MCP 外部研究。默认关闭，仅允许显式配置的 stdio Server 与两个只读工具。
    mcp_research_enabled: bool = False
    mcp_server_label: str = "external-research"
    mcp_server_command: str = ""
    mcp_server_args_json: str = "[]"
    mcp_server_env_json: str = "{}"
    mcp_search_tool: str = "web_search"
    mcp_fetch_tool: str = "fetch_page"
    mcp_search_query_arg: str = "query"
    mcp_search_limit_arg: str = "count"
    mcp_fetch_url_arg: str = "url"
    mcp_max_queries: int = 2
    mcp_max_sources: int = 5
    mcp_max_content_chars: int = 12_000
    mcp_timeout_seconds: float = 30.0
    mcp_total_timeout_seconds: float = 90.0

    @model_validator(mode="after")
    def validate_crag_thresholds(self):
        """保证 CRAG 的双阈值形成明确的三段区间。"""
        if self.crag_lower_threshold >= self.crag_upper_threshold:
            raise ValueError("CRAG_LOWER_THRESHOLD 必须小于 CRAG_UPPER_THRESHOLD")
        if self.crag_recovery_min_score > self.crag_lower_threshold:
            raise ValueError("CRAG_RECOVERY_MIN_SCORE 不能高于 CRAG_LOWER_THRESHOLD")
        return self

    def get_cors_origins(self) -> list:
        """获取 CORS 允许的源列表"""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",")]

    def get_allowed_extensions(self) -> list:
        """获取允许的文件扩展名列表"""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    def get_mcp_server_args(self) -> list[str]:
        """解析 MCP Server 参数，并拒绝非字符串值。"""
        values = json.loads(self.mcp_server_args_json or "[]")
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("MCP_SERVER_ARGS_JSON 必须是字符串数组")
        return values

    def get_mcp_server_env(self) -> dict[str, str]:
        """只向 MCP 子进程传递显式配置的环境变量。"""
        values = json.loads(self.mcp_server_env_json or "{}")
        if not isinstance(values, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in values.items()
        ):
            raise ValueError("MCP_SERVER_ENV_JSON 必须是字符串键值对象")
        return values

    def mcp_research_available(self) -> bool:
        """只有开关、命令和两个工具名齐全时才向用户提供外部研究。"""
        return bool(
            self.mcp_research_enabled
            and self.mcp_server_command.strip()
            and self.mcp_search_tool.strip()
            and self.mcp_fetch_tool.strip()
        )

    def get_upload_dir(self) -> str:
        """获取上传目录的绝对路径"""
        upload_path = Path(self.upload_dir)
        if not upload_path.is_absolute():
            upload_path = Path(os.path.abspath(self.upload_dir))
        upload_path.mkdir(parents=True, exist_ok=True)
        return str(upload_path)


# 全局配置实例
settings = Settings()
