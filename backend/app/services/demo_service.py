"""面试演示数据初始化服务。"""

import math
import struct
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models import Audio
from .page_service import PageService

DEMO_AUDIO_FILENAME = "zhiyu-demo-lesson.wav"
DEMO_SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "RAG 通过外部知识检索增强大模型回答。"},
    {"start": 2.0, "end": 4.0, "text": "BM25 擅长关键词匹配，向量检索负责语义召回。"},
    {"start": 4.0, "end": 6.0, "text": "RRF 融合两路排序，再由 Reranker 进行精排。"},
    {"start": 6.0, "end": 8.0, "text": "证据不足时系统应拒绝生成推测性答案。"},
]

DEMO_PAGES = [
    {
        "slug": "rag-overview",
        "title": "RAG 检索增强生成",
        "tags": ["RAG", "LLM"],
        "content": """# RAG 检索增强生成

[00:00-00:02] RAG 在生成答案前检索外部知识，让回答能够引用可核验的来源。

## 核心流程

1. 将问题转换为检索请求。
2. 从知识库召回相关片段。
3. 将证据交给大模型生成答案。
4. 返回答案及来源。

进一步阅读：[[混合检索流水线]]、[[可信问答与证据门禁]]。
""",
    },
    {
        "slug": "hybrid-retrieval",
        "title": "混合检索流水线",
        "tags": ["BM25", "Embedding", "RRF", "Reranker"],
        "content": """# 混合检索流水线

[00:02-00:04] BM25 对专有名词和精确关键词敏感，Embedding 更擅长语义相似表达。

[00:04-00:06] 系统并行召回两路候选，使用 RRF 合并排名，再由 BGE Reranker 精排。

这种组合兼顾关键词召回、语义召回和最终排序质量，是 [[RAG 检索增强生成]] 的检索基础。
""",
    },
    {
        "slug": "evidence-gate",
        "title": "可信问答与证据门禁",
        "tags": ["可信问答", "Evidence Gate"],
        "content": """# 可信问答与证据门禁

[00:06-00:08] 检索结果为空、来源不足或精排分数低于阈值时，系统不会调用 LLM 猜测答案。

## 输出约束

- 返回结构化的证据状态和原因。
- 保留页面、版本、章节和稳定 Chunk ID。
- 对课堂知识同时提供原始录音时间点。

相关实现依赖 [[混合检索流水线]]。
""",
    },
]


def _write_demo_audio(path: Path) -> None:
    """生成一段体积很小的 WAV，便于演示媒体时间定位。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8_000
    duration_seconds = 8
    frames = bytearray()
    for index in range(sample_rate * duration_seconds):
        second = index // sample_rate
        frequency = 330 + second * 20
        sample = int(2_500 * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(frames)


def initialize_demo_data(
    db: Session,
    *,
    pages_dir: Optional[str] = None,
    upload_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """幂等创建演示音频、转写时间片和 Wiki 页面。"""
    resolved_upload_dir = Path(upload_dir or settings.upload_dir).resolve()
    audio_path = resolved_upload_dir / DEMO_AUDIO_FILENAME
    if not audio_path.exists():
        _write_demo_audio(audio_path)

    transcription = " ".join(segment["text"] for segment in DEMO_SEGMENTS)
    audio = db.query(Audio).filter(Audio.filename == DEMO_AUDIO_FILENAME).one_or_none()
    created_audio = audio is None
    if audio is None:
        audio = Audio(filename=DEMO_AUDIO_FILENAME)
        db.add(audio)
    audio.original_filename = DEMO_AUDIO_FILENAME
    audio.file_path = str(audio_path)
    audio.file_size = audio_path.stat().st_size
    audio.duration = 8.0
    audio.language = "zh"
    audio.transcription = transcription
    audio.transcription_segments = DEMO_SEGMENTS
    db.commit()
    db.refresh(audio)

    page_service = PageService(db, pages_dir=pages_dir or settings.wiki_pages_dir)
    pages = []
    created_pages = 0
    for definition in DEMO_PAGES:
        page = page_service.upsert_page_by_source(
            title=definition["title"],
            content=definition["content"],
            notebook="智语演示",
            tags=definition["tags"],
            source_type="class_audio",
            source_uri=f"audio:{audio.id}#demo:{definition['slug']}",
            change_summary="初始化面试演示数据",
        )
        if not page["deduplicated"] and page["revision"] == 1:
            created_pages += 1
        pages.append(
            {
                "id": page["id"],
                "title": page["title"],
                "revision": page["revision"],
                "deduplicated": page["deduplicated"],
                "index_status": page["index_status"],
            }
        )

    return {
        "audio": {
            "id": audio.id,
            "filename": audio.filename,
            "created": created_audio,
            "segments": len(DEMO_SEGMENTS),
        },
        "pages": pages,
        "created_pages": created_pages,
        "message": "演示数据已初始化；索引任务将由后台 Worker 处理。",
    }
