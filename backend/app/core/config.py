"""
应用配置管理
"""
import json
import os
from typing import Literal

from pydantic import Field
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
    rrf_top_k: int = 30  # 融合后候选数
    bm25_top_k: int = 30  # BM25 检索数量
    embedding_top_k: int = 30  # Embedding 检索数量

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

    # ASR 引擎配置（whisper=本地, dashscope=百炼API）
    asr_provider: str = "whisper"  # 默认使用本地 Whisper

    # 百炼 DashScope ASR 配置
    dashscope_asr_api_key: str = ""
    dashscope_asr_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_asr_model: str = "paraformer-realtime-v2"

    # LLM 配置（Vercel AI Gateway 的 OpenAI 兼容接口）
    llm_api_key: str = ""
    llm_api_url: str = "https://ai-gateway.vercel.sh/v1"
    llm_model: str = "deepseek/deepseek-v4-flash"
    llm_max_tokens: int = 2048
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

    # Langfuse 可观测配置（可选）
    langfuse_host: str = ""  # Langfuse 服务地址
    langfuse_public_key: str = ""  # Langfuse 公钥
    langfuse_secret_key: str = ""  # Langfuse 私钥
    observability_enabled: bool = True
    observability_capture_content: bool = False
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
