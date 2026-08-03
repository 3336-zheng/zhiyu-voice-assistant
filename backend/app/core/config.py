"""
应用配置管理
"""
import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
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
    embedding_model_path: str = ""
    reranker_model_path: str = ""

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
    rrf_top_k: int = 10  # 融合后候选数
    bm25_top_k: int = 20  # BM25 检索数量
    embedding_top_k: int = 20  # Embedding 检索数量

    # Agent 配置（新增）
    agent_max_iterations: int = 5  # Agent 最大执行轮数

    # ASR 引擎配置（whisper=本地, dashscope=百炼API）
    asr_provider: str = "whisper"  # 默认使用本地 Whisper

    # 百炼 DashScope ASR 配置
    dashscope_asr_api_key: str = ""
    dashscope_asr_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_asr_model: str = "paraformer-realtime-v2"

    # LLM 配置（OpenAI 兼容接口）
    llm_api_key: str = ""
    llm_api_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7

    # 对话记忆配置（新增）
    memory_max_history: int = 20  # 最大保留对话轮数
    memory_summary_threshold: int = 10  # 超过此轮数时触发摘要压缩

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

    # CORS 配置
    cors_origins: str = "*"  # 逗号分隔的域名列表，如 "https://example.com,https://app.example.com"

    # 性能配置
    max_concurrent_requests: int = 10
    timeout_seconds: int = 300

    # Wiki 索引 worker 配置
    wiki_index_poll_seconds: float = 5.0
    wiki_index_batch_size: int = 5
    wiki_index_max_backoff_seconds: int = 300

    # 可信问答证据门禁
    evidence_min_score: float = 0.35
    evidence_min_sources: int = 1

    def get_cors_origins(self) -> list:
        """获取 CORS 允许的源列表"""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_allowed_extensions(self) -> list:
        """获取允许的文件扩展名列表"""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    def get_upload_dir(self) -> str:
        """获取上传目录的绝对路径"""
        upload_path = Path(self.upload_dir)
        if not upload_path.is_absolute():
            upload_path = Path(os.path.abspath(self.upload_dir))
        upload_path.mkdir(parents=True, exist_ok=True)
        return str(upload_path)


# 全局配置实例
settings = Settings()
