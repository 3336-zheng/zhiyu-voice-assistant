"""音频探测、规范化与上传产物管理。"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


class AudioProcessingError(RuntimeError):
    """音频无法读取或转换。"""


class AudioTooLargeError(AudioProcessingError):
    """原始文件或规范化产物超过大小限制。"""


@dataclass(frozen=True)
class AudioArtifact:
    """规范化后可持久化的音频信息。"""

    filename: str
    file_path: str
    file_size: int
    duration: float


def _find_media_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable

    python_dir = Path(sys.executable).parent
    candidates = [
        python_dir / name,
        python_dir / f"{name}.exe",
        python_dir / "Library" / "bin" / f"{name}.exe",
        python_dir.parent / "Library" / "bin" / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise AudioProcessingError(f"找不到 {name}，请安装 ffmpeg 工具集")


def probe_duration(file_path: str | Path, timeout_seconds: float = 15.0) -> float:
    """使用 ffprobe 读取媒体容器中的真实时长。"""
    command = [
        _find_media_tool("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=duration",
        "-of",
        "json",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError("读取音频时长超时") from exc

    if result.returncode != 0:
        logger.warning("ffprobe 读取失败: %s", result.stderr[:300])
        raise AudioProcessingError("无法读取音频文件")

    try:
        payload = json.loads(result.stdout)
        candidates = [payload.get("format", {}).get("duration")]
        candidates.extend(stream.get("duration") for stream in payload.get("streams", []))
        duration = next(float(value) for value in candidates if value not in (None, "N/A"))
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise AudioProcessingError("音频文件缺少有效时长") from exc

    if not math.isfinite(duration) or duration <= 0:
        raise AudioProcessingError("音频时长无效")
    return duration


def _normalize(source_path: Path, output_path: Path, timeout_seconds: float) -> None:
    command = [
        _find_media_tool("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError("音频规范化超时") from exc

    if result.returncode != 0:
        logger.warning("ffmpeg 规范化失败: %s", result.stderr[:300])
        raise AudioProcessingError("音频格式无效或无法转换")


def normalize_upload(
    original_name: str,
    content: bytes,
    upload_dir: str | Path,
    max_file_size: int,
    timeout_seconds: float,
    probe_timeout_seconds: float,
) -> AudioArtifact:
    """将任意受支持音频统一保存为 16kHz 单声道 16-bit PCM WAV。"""
    if not content:
        raise AudioProcessingError("上传的文件内容为空")
    if len(content) > max_file_size:
        raise AudioTooLargeError("原始音频超过大小限制")

    directory = Path(upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower() or ".audio"
    token = uuid.uuid4().hex
    source_path = directory / f".{token}.source{suffix}"
    temporary_output = directory / f".{token}.normalized.wav"
    final_path = directory / f"{stem}.wav"

    try:
        source_path.write_bytes(content)
        probe_duration(source_path, probe_timeout_seconds)
        _normalize(source_path, temporary_output, timeout_seconds)

        output_size = temporary_output.stat().st_size
        if output_size > max_file_size:
            raise AudioTooLargeError("规范化后的音频超过大小限制")
        if output_size <= 44:
            raise AudioProcessingError("规范化后的音频为空")

        duration = probe_duration(temporary_output, probe_timeout_seconds)
        os.replace(temporary_output, final_path)
        return AudioArtifact(
            filename=final_path.name,
            file_path=str(final_path.resolve()),
            file_size=output_size,
            duration=duration,
        )
    finally:
        source_path.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)
