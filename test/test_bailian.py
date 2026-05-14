"""
百炼 DashScope ASR API 连通性测试（使用官方 SDK）
直接运行：python test/test_bailian.py
"""
import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置项 - 按需修改
AUDIO_DIR = str(Path(__file__).parent.parent / "data" / "uploads")
LANGUAGE = "zh"


def print_separator(title=""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}")


def format_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.2f}s"


def load_config():
    """加载百炼 API 配置"""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    try:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env", override=True)

        from backend.app.core.config import settings
        return {
            "api_key": settings.dashscope_asr_api_key,
            "model": settings.dashscope_asr_model,
        }
    except Exception as e:
        print(f"[警告] 从配置加载失败: {e}")
        return {
            "api_key": os.environ.get("DASHSCOPE_ASR_API_KEY", ""),
            "model": os.environ.get("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v2"),
        }


def check_sdk():
    """检查 dashscope SDK 是否可用"""
    print_separator("SDK 检查")
    try:
        import importlib.metadata
        ver = importlib.metadata.version("dashscope")
        print(f"[OK] dashscope SDK 已安装，版本: {ver}")
        return True
    except ImportError:
        print("[错误] dashscope SDK 未安装，请执行: pip install dashscope")
        return False


def run_transcribe(audio_path, api_key, model):
    """使用 DashScope SDK 转录单个音频"""
    import librosa
    from dashscope.audio.asr import Recognition, RecognitionCallback

    file_size = os.path.getsize(audio_path) / 1024
    print(f"\n[信息] 音频文件: {audio_path}")
    print(f"[信息] 文件大小: {file_size:.1f} KB")

    try:
        audio_duration = librosa.get_duration(path=audio_path)
        print(f"[信息] 音频时长: {format_time(audio_duration)} ({audio_duration:.2f}秒)")
    except Exception as e:
        print(f"[警告] 无法获取音频时长: {e}")
        audio_duration = 0

    # 检测格式
    ext = os.path.splitext(audio_path)[1].lower().strip(".")
    fmt = {"wav": "wav", "mp3": "mp3", "flac": "flac"}.get(ext, "wav")

    # 获取采样率
    try:
        sr = int(librosa.get_samplerate(audio_path))
    except Exception:
        sr = 16000

    print(f"[请求] 模型: {model}, 格式: {fmt}, 采样率: {sr}")

    # 回调类
    class Callback(RecognitionCallback):
        def __init__(self):
            self.sentences = []
        def on_event(self, result):
            s = result.get_sentence()
            if s:
                self.sentences.extend(s) if isinstance(s, list) else self.sentences.append(s)

    start_time = time.time()
    try:
        import dashscope
        dashscope.api_key = api_key

        cb = Callback()
        rec = Recognition(model=model, callback=cb, format=fmt, sample_rate=sr)
        result = rec.call(audio_path)
        elapsed = time.time() - start_time

        print_separator("转录结果")
        print(f"[HTTP] 状态码: {result.status_code}")
        print(f"[耗时] 总耗时: {format_time(elapsed)}")

        if result.status_code == 200:
            sentences = result.get_sentence() or cb.sentences
            texts = []
            segments = []
            for sent in sentences:
                if isinstance(sent, dict):
                    t = sent.get("text", "")
                    texts.append(t)
                    segments.append({
                        "start": (sent.get("begin_time", 0) or 0) / 1000,
                        "end": (sent.get("end_time", 0) or 0) / 1000,
                        "text": t,
                    })
                elif isinstance(sent, str):
                    texts.append(sent)

            full_text = "".join(texts)
            print(f"[转录] 文本长度: {len(full_text)} 字")
            print(f"[转录] 内容:\n  {full_text}")

            if audio_duration > 0:
                print(f"\n[性能] RTF: {elapsed / audio_duration:.4f}")
                print(f"[性能] 速度倍率: {audio_duration / elapsed:.2f}x")

            if segments:
                print(f"\n[分段] 共 {len(segments)} 段:")
                for seg in segments:
                    print(f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['text']}")

            return {
                "status": "success", "text": full_text,
                "elapsed": elapsed, "audio_duration": audio_duration,
                "rtf": elapsed / audio_duration if audio_duration > 0 else 0,
                "segments": segments,
            }
        else:
            error_msg = result.message or f"状态码: {result.status_code}"
            print(f"[错误] {error_msg}")
            return {"status": "error", "code": result.status_code, "error": error_msg}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[错误] 异常 ({format_time(elapsed)}): {e}")
        return {"status": "error", "error": str(e)}


def main():
    print_separator("百炼 DashScope ASR 测试（SDK 模式）")

    # 1. 检查 SDK
    if not check_sdk():
        return 1

    # 2. 加载配置
    cfg = load_config()
    if not cfg["api_key"]:
        print("[错误] API Key 未配置，请在 .env 中设置 DASHSCOPE_ASR_API_KEY")
        return 1
    print(f"[配置] 模型: {cfg['model']}")
    print(f"[配置] API Key: {cfg['api_key'][:8]}...{cfg['api_key'][-4:]}")

    # 3. 查找音频文件
    audio_dir = Path(AUDIO_DIR)
    if not audio_dir.exists():
        print(f"\n[错误] 音频目录不存在: {AUDIO_DIR}")
        return 1

    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        print(f"\n[错误] 未找到 WAV 音频文件: {AUDIO_DIR}")
        return 1

    # 4. 逐个转录
    print_separator("开始转录测试")
    print(f"[配置] 文件数: {len(files)}")

    results = []
    for af in files:
        results.append(run_transcribe(str(af), cfg["api_key"], cfg["model"]))

    # 5. 汇总
    print_separator("汇总统计")
    success = [r for r in results if r["status"] == "success"]
    fail = [r for r in results if r["status"] != "success"]
    print(f"[统计] 总文件数: {len(results)}, 成功: {len(success)}, 失败: {len(fail)}")

    if success:
        avg_rtf = sum(r["rtf"] for r in success) / len(success)
        total_audio = sum(r["audio_duration"] for r in success)
        total_time = sum(r["elapsed"] for r in success)
        print(f"[统计] 总音频时长: {format_time(total_audio)}")
        print(f"[统计] 总转录耗时: {format_time(total_time)}")
        print(f"[统计] 平均 RTF: {avg_rtf:.4f}")

    # 保存结果
    output_file = str(Path(__file__).parent / "bailian_test_result.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"model": cfg["model"], "api_key": f"{cfg['api_key'][:8]}...{cfg['api_key'][-4:]}"},
            "summary": {"total": len(results), "success": len(success), "fail": len(fail)},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 结果已保存到: {output_file}")

    if fail:
        print(f"\n[结论] {len(fail)} 个文件转录失败")
        return 1
    print("\n[结论] 全部测试通过，百炼 ASR 调用正常")
    return 0


def test_bailian_asr():
    """pytest 入口"""
    assert main() == 0, "百炼 ASR 测试失败，请查看上方输出"


if __name__ == "__main__":
    sys.exit(main())
