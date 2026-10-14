"""音频规范化、ASR 调度与 MiMo Provider 测试。"""

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from test.support.db import make_test_engine
from backend.app.models import Audio
from backend.app.services.ingestion.asr_service import (
    ASRInputError,
    ASRProviderNotFoundError,
    resolve_provider,
)
from backend.app.services.ingestion.audio_processing import (
    AudioTooLargeError,
    normalize_upload,
)
from backend.app.services.ingestion.mimo_asr_service import MiMoASRService
from backend.app.tasks.asr_tasks import FAILURE_CODES, asr_job_id, transcribe_audio


def asr_result(text: str = "测试转写") -> dict:
    return {
        "transcription": text,
        "segments": [],
        "language": "zh",
        "duration": 1.0,
        "transcribe_time": 0.1,
        "rtf": 0.1,
    }


class AsrTaskTestCase(unittest.IsolatedAsyncioTestCase):
    """转写任务的行为。

    执行调度（去重、超时、并发）现在归 arq 队列管，这里测的是搬过去之后
    任务本身：写回音频表、失败时返回结构化结果而不是抛异常
    （任务在另一个进程里跑，异常要跨进程序列化才能传回 API）。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        engine = make_test_engine()
        self.Session = sessionmaker(bind=engine)
        self.audio_file = self.root / "sample.wav"
        self.audio_file.write_bytes(b"RIFF....")

        db = self.Session()
        db.add(Audio(id=7, filename="sample.wav", file_path=str(self.audio_file)))
        db.commit()
        db.close()

        self._session_patch = patch(
            "backend.app.tasks.asr_tasks.SessionLocal", self.Session
        )
        self._session_patch.start()

    def tearDown(self):
        self._session_patch.stop()
        self._tmp.cleanup()

    async def test_转写成功会写回音频表(self):
        with patch(
            "backend.app.services.ingestion.asr_service.transcribe_sync",
            return_value=asr_result("识别出来的内容"),
        ):
            outcome = await transcribe_audio({}, 7, "mimo", "auto")

        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["result"]["transcription"], "识别出来的内容")

        db = self.Session()
        try:
            audio = db.get(Audio, 7)
            self.assertEqual(audio.transcription, "识别出来的内容")
        finally:
            db.close()

    async def test_音频不存在时返回结构化结果而不是抛异常(self):
        # 任务在另一个进程里跑，抛异常得跨进程序列化才能传回 API，
        # 而自定义异常类在两端版本不一致时会变成难查的解析错误。
        outcome = await transcribe_audio({}, 999, "mimo", "auto")
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["code"], "not_found")

    async def test_转写失败时不写入半截数据(self):
        with patch(
            "backend.app.services.ingestion.asr_service.transcribe_sync",
            side_effect=RuntimeError("上游挂了"),
        ):
            outcome = await transcribe_audio({}, 7, "mimo", "auto")

        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["code"], "execution")
        db = self.Session()
        try:
            self.assertIsNone(db.get(Audio, 7).transcription)
        finally:
            db.close()

    async def test_失败信息不泄露上游细节(self):
        with patch(
            "backend.app.services.ingestion.asr_service.transcribe_sync",
            side_effect=RuntimeError("upstream leaked api_key=secret"),
        ):
            outcome = await transcribe_audio({}, 7, "mimo", "auto")

        self.assertEqual(outcome["message"], "语音转写服务暂时不可用")
        self.assertNotIn("secret", outcome["message"])

    async def test_已移除的本地_provider_被拒绝(self):
        # whisper 已移除。调用里再出现它必须立刻报错，而不是悄悄回退到某个
        # 远程 provider 跑起来——静默降级会让人以为还在用本地模型。
        with self.assertRaises(ASRProviderNotFoundError):
            resolve_provider("whisper")

    async def test_job_id_按音频维度(self):
        # 同一个音频同时只允许一个转写任务，靠 job_id 的唯一性实现；
        # 换成进程内集合的话多实例下每个进程各有一份，拦不住别的实例。
        self.assertEqual(asr_job_id(7), "asr:7")
        self.assertNotEqual(asr_job_id(7), asr_job_id(8))

    async def test_每种失败码都有对应的_HTTP_状态(self):
        # worker 新增一种失败而 API 没跟上时，会静默落进默认的 502，
        # 用户看到「服务不可用」但其实是他自己的输入有问题。这条守住两者的同步。
        from backend.app.api.ingestion.audio import ASR_FAILURE_STATUS

        missing = [
            code for code in FAILURE_CODES
            if code != "execution" and code not in ASR_FAILURE_STATUS
        ]
        self.assertEqual(missing, [], f"这些失败码没有对应的 HTTP 状态: {missing}")


class AudioProcessingTestCase(unittest.TestCase):
    def test_normalized_file_size_is_checked_again(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "backend.app.services.ingestion.audio_processing.probe_duration",
                    return_value=1.0,
                ),
                patch(
                    "backend.app.services.ingestion.audio_processing._normalize",
                    side_effect=lambda _source, output, _timeout: output.write_bytes(b"x" * 101),
                ),
            ):
                with self.assertRaises(AudioTooLargeError):
                    normalize_upload(
                        "sample.mp3",
                        b"source",
                        directory,
                        max_file_size=100,
                        timeout_seconds=1,
                        probe_timeout_seconds=1,
                    )
            self.assertEqual(list(Path(directory).iterdir()), [])


class FakeMiMoResponse:
    ok = True
    status_code = 200

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": "  小米转写结果  "}}]}


class MiMoASRServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "mimo_asr_api_key": settings.mimo_asr_api_key,
            "mimo_asr_api_url": settings.mimo_asr_api_url,
            "mimo_asr_max_base64_bytes": settings.mimo_asr_max_base64_bytes,
        }
        settings.mimo_asr_api_key = "test-key"
        settings.mimo_asr_api_url = "https://api.xiaomimimo.com/v1"
        settings.mimo_asr_max_base64_bytes = 10 * 1024 * 1024

    def tearDown(self):
        for key, value in self.previous.items():
            setattr(settings, key, value)

    def test_request_and_response_follow_mimo_contract(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(b"wav-bytes")
            audio_file.flush()
            with (
                patch(
                    "backend.app.services.ingestion.mimo_asr_service.probe_duration",
                    return_value=2.0,
                ),
                patch(
                    "backend.app.services.ingestion.mimo_asr_service.requests.post",
                    return_value=FakeMiMoResponse(),
                ) as post,
            ):
                result = MiMoASRService().transcribe(audio_file.name, language="zh")

        self.assertEqual(result["transcription"], "小米转写结果")
        self.assertEqual(result["segments"], [])
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["api-key"], "test-key")
        self.assertEqual(kwargs["json"]["model"], "mimo-v2.5-asr")
        self.assertEqual(kwargs["json"]["asr_options"], {"language": "zh"})
        audio_data = kwargs["json"]["messages"][0]["content"][0]["input_audio"]["data"]
        self.assertTrue(audio_data.startswith("data:audio/wav;base64,"))

    def test_base64_limit_is_reported_as_input_error(self):
        settings.mimo_asr_max_base64_bytes = 24
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(b"wav-bytes")
            audio_file.flush()
            with patch(
                "backend.app.services.ingestion.mimo_asr_service.probe_duration",
                return_value=2.0,
            ):
                with self.assertRaises(ASRInputError):
                    MiMoASRService().transcribe(audio_file.name)


if __name__ == "__main__":
    unittest.main()
