"""音频规范化、ASR 调度与 MiMo Provider 测试。"""

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.core.config import settings
from backend.app.services.ingestion.asr_service import (
    ASRAlreadyRunningError,
    ASRExecutionError,
    ASRExecutionService,
    ASRInputError,
    ASRTimeoutError,
)
from backend.app.services.ingestion.audio_processing import (
    AudioTooLargeError,
    normalize_upload,
)
from backend.app.services.ingestion.mimo_asr_service import MiMoASRService


def asr_result(text: str = "测试转写") -> dict:
    return {
        "transcription": text,
        "segments": [],
        "language": "zh",
        "duration": 1.0,
        "transcribe_time": 0.1,
        "rtf": 0.1,
    }


class ASRExecutionServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous_key = settings.mimo_asr_api_key
        self.previous_timeout = settings.asr_timeout_seconds
        settings.mimo_asr_api_key = "test-key"
        settings.asr_timeout_seconds = 1.0

    def tearDown(self):
        settings.mimo_asr_api_key = self.previous_key
        settings.asr_timeout_seconds = self.previous_timeout

    async def test_same_audio_id_cannot_run_twice(self):
        started = threading.Event()
        release = threading.Event()

        def blocking_transcribe(*_args):
            started.set()
            release.wait(1)
            return asr_result()

        service = ASRExecutionService()
        with patch(
            "backend.app.services.ingestion.asr_service._transcribe_sync",
            side_effect=blocking_transcribe,
        ):
            first = asyncio.create_task(service.transcribe(7, "audio.wav", "mimo"))
            await asyncio.to_thread(started.wait, 1)
            with self.assertRaises(ASRAlreadyRunningError):
                await service.transcribe(7, "audio.wav", "mimo")
            release.set()
            await first

    async def test_timeout_keeps_audio_claimed_until_thread_finishes(self):
        settings.asr_timeout_seconds = 0.01

        def slow_transcribe(*_args):
            time.sleep(0.08)
            return asr_result()

        service = ASRExecutionService()
        with patch(
            "backend.app.services.ingestion.asr_service._transcribe_sync",
            side_effect=slow_transcribe,
        ):
            with self.assertRaises(ASRTimeoutError):
                await service.transcribe(8, "audio.wav", "mimo")
            with self.assertRaises(ASRAlreadyRunningError):
                await service.transcribe(8, "audio.wav", "mimo")
            await asyncio.sleep(0.1)

    async def test_whisper_requests_are_serialized(self):
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def tracked_transcribe(*_args):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return asr_result()

        with tempfile.TemporaryDirectory() as model_dir:
            previous_path = settings.whisper_model_path
            settings.whisper_model_path = model_dir
            try:
                service = ASRExecutionService()
                with patch(
                    "backend.app.services.ingestion.asr_service._transcribe_sync",
                    side_effect=tracked_transcribe,
                ):
                    await asyncio.gather(
                        service.transcribe(1, "one.wav", "whisper"),
                        service.transcribe(2, "two.wav", "whisper"),
                    )
            finally:
                settings.whisper_model_path = previous_path
        self.assertEqual(max_active, 1)

    async def test_provider_error_is_not_exposed(self):
        service = ASRExecutionService()
        with patch(
            "backend.app.services.ingestion.asr_service._transcribe_sync",
            side_effect=RuntimeError("upstream leaked api_key=secret"),
        ):
            with self.assertRaises(ASRExecutionError) as context:
                await service.transcribe(9, "audio.wav", "mimo")
        self.assertEqual(str(context.exception), "语音转写服务暂时不可用")
        self.assertNotIn("secret", str(context.exception))


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
