"""
Whisper 转录性能测试脚本
直接运行即可，自动测试 data/uploads/ 下的音频文件
"""
import os
import sys
import time
import json
import psutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置项 - 按需修改
AUDIO_DIR = str(Path(__file__).parent.parent / "data" / "uploads")  # 音频文件目录
LANGUAGE = "zh"             # 语言代码
USE_VAD = True              # 是否使用VAD过滤
COMPUTE_TYPE = "int8"       # 计算精度: int8 / float16 / float32


def get_gpu_memory():
    """获取GPU显存使用情况（MB）"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024 / 1024,
                "cached": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
    except Exception:
        pass
    return None


def get_system_memory():
    """获取系统内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return {
        "rss": process.memory_info().rss / 1024 / 1024,
        "vms": process.memory_info().vms / 1024 / 1024,
        "percent": process.memory_percent(),
    }


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.2f}s"


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}")


def test_whisper(audio_path, model, language, use_vad):
    """
    测试单个音频文件的转录性能

    Args:
        audio_path: 音频文件路径
        model: 已加载的WhisperModel实例
        language: 语言代码
        use_vad: 是否使用VAD过滤
    """
    import torch
    import librosa

    file_size = os.path.getsize(audio_path) / 1024 / 1024  # MB
    print(f"\n[信息] 音频文件: {audio_path}")
    print(f"[信息] 文件大小: {file_size:.2f} MB")

    # 获取音频时长
    audio_duration = librosa.get_duration(path=audio_path)
    print(f"[信息] 音频时长: {format_time(audio_duration)} ({audio_duration:.2f}秒)")

    # 获取初始内存状态
    initial_memory = get_system_memory()
    initial_gpu = get_gpu_memory()

    # 预热GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 执行转录
    print("[转录] 开始...")
    transcribe_start = time.time()

    vad_params = {"min_silence_duration_ms": 500} if use_vad else None
    segments, info = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        vad_filter=use_vad,
        vad_parameters=vad_params
    )

    # 迭代获取所有segments
    segments_list = []
    segment_times = []
    for i, seg in enumerate(segments):
        segments_list.append(seg)
        segment_times.append({
            "index": i,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "duration": seg.end - seg.start,
        })

    transcribe_time = time.time() - transcribe_start

    # 获取转录后内存状态
    after_memory = get_system_memory()
    after_gpu = get_gpu_memory()

    # 计算指标
    rtf = transcribe_time / audio_duration if audio_duration > 0 else 0
    speed_ratio = audio_duration / transcribe_time if transcribe_time > 0 else 0
    mem_delta = after_memory['rss'] - initial_memory['rss']
    gpu_delta = after_gpu['allocated'] - (initial_gpu['allocated'] if initial_gpu else 0) if after_gpu else 0

    # 输出结果
    print_separator("测试结果")
    print(f"[核心指标]")
    print(f"  转录耗时: {format_time(transcribe_time)}")
    print(f"  音频时长: {format_time(audio_duration)}")
    print(f"  RTF (实时率): {rtf:.4f} (越小越好，<1表示快于实时)")
    print(f"  速度倍率: {speed_ratio:.2f}x (越大越好，>1表示快于实时)")

    print(f"\n[语言检测]")
    print(f"  检测语言: {info.language}")
    print(f"  语言概率: {info.language_probability:.4f}")

    print(f"\n[内存使用]")
    print(f"  进程内存增量: {mem_delta:.1f} MB")
    if after_gpu:
        print(f"  GPU显存峰值: {after_gpu['max_allocated']:.1f} MB")
        print(f"  GPU显存增量: {gpu_delta:.1f} MB")

    print(f"\n[Segment 统计]")
    print(f"  总段数: {len(segments_list)}")
    if segments_list:
        seg_durations = [s['duration'] for s in segment_times]
        print(f"  平均段长: {sum(seg_durations)/len(seg_durations):.2f}s")
        print(f"  最短段: {min(seg_durations):.2f}s")
        print(f"  最长段: {max(seg_durations):.2f}s")

    full_text = " ".join([seg.text.strip() for seg in segments_list])
    print(f"\n[转录文本]")
    print(f"  总字数: {len(full_text)}")
    print(f"  内容: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")

    print(f"\n[Segment 详情]")
    for seg in segment_times:
        print(f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['text']}")

    return {
        "file": audio_path,
        "file_size_mb": file_size,
        "audio_duration": audio_duration,
        "transcribe_time": transcribe_time,
        "rtf": rtf,
        "speed_ratio": speed_ratio,
        "language": info.language,
        "language_probability": info.language_probability,
        "segment_count": len(segments_list),
        "text_length": len(full_text),
        "transcription": full_text,
        "memory_delta_mb": mem_delta,
        "gpu_peak_mb": after_gpu['max_allocated'] if after_gpu else 0,
        "gpu_delta_mb": gpu_delta,
        "segments": segment_times,
    }


def main():
    print_separator("Whisper 转录性能测试")

    import torch
    from faster_whisper import WhisperModel

    # GPU 信息
    print_separator("硬件环境")
    print(f"[GPU] CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GPU] 设备: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA 版本: {torch.version.cuda}")
        print(f"[GPU] 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
    print(f"[系统] CPU 核心数: {psutil.cpu_count()}")
    print(f"[系统] 内存总量: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")

    # 获取模型路径
    try:
        from backend.app.core.config import settings
        model_path = settings.whisper_model_path
    except Exception:
        print("[错误] 无法从配置获取模型路径")
        return 1

    print_separator("模型加载")
    print(f"[模型] 路径: {model_path}")
    print(f"[模型] 精度: {COMPUTE_TYPE}")
    print(f"[模型] 设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 加载模型
    load_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_path, device=device, compute_type=COMPUTE_TYPE)
    load_time = time.time() - load_start
    print(f"[模型] 加载耗时: {format_time(load_time)}")

    # 获取音频文件列表
    audio_dir = Path(AUDIO_DIR)
    if not audio_dir.exists():
        print(f"[错误] 音频目录不存在: {AUDIO_DIR}")
        return 1

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"[错误] 未找到WAV音频文件: {AUDIO_DIR}")
        return 1

    print_separator("开始测试")
    print(f"[配置] 语言: {LANGUAGE}")
    print(f"[配置] VAD过滤: {USE_VAD}")
    print(f"[配置] 测试文件数: {len(audio_files)}")

    # 逐个测试
    results = []
    for audio_path in audio_files:
        result = test_whisper(str(audio_path), model, LANGUAGE, USE_VAD)
        if result:
            results.append(result)

    # 汇总统计
    if results:
        print_separator("汇总统计")
        avg_rtf = sum(r['rtf'] for r in results) / len(results)
        avg_speed = sum(r['speed_ratio'] for r in results) / len(results)
        total_audio = sum(r['audio_duration'] for r in results)
        total_time = sum(r['transcribe_time'] for r in results)

        print(f"[统计] 测试文件数: {len(results)}")
        print(f"[统计] 总音频时长: {format_time(total_audio)}")
        print(f"[统计] 总转录耗时: {format_time(total_time)}")
        print(f"[统计] 平均 RTF: {avg_rtf:.4f}")
        print(f"[统计] 平均速度倍率: {avg_speed:.2f}x")

        # 保存结果
        output_file = str(Path(__file__).parent.parent / "whisper_test_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "language": LANGUAGE,
                    "use_vad": USE_VAD,
                    "compute_type": COMPUTE_TYPE,
                    "device": device,
                    "model_path": model_path,
                    "model_load_time": load_time,
                },
                "summary": {
                    "file_count": len(results),
                    "total_audio_duration": total_audio,
                    "total_transcribe_time": total_time,
                    "avg_rtf": avg_rtf,
                    "avg_speed_ratio": avg_speed,
                },
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[保存] 详细结果已保存到: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
