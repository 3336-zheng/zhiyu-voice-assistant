"""文档转换与结构重排接口测试。

三个接口都不写库（commit 除外），所以这里重点测边界：格式拦截、编码错误、
空内容、超长、模型返回异常。这些出错方式都是静默的——用户会拿到空页面或
半篇文档，而界面上看不出来。

重排本身是 LLM 调用，测不了「排得好不好」（那是提示词的事，靠预览人工判断），
所以把 llm 换成假的，只验接口把参数传对了、把异常转成了合适的 HTTP 状态码。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend.app.api.ingestion import transform as transform_api
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.services.ingestion.doc_restructure_service import DocRestructureService
from backend.app.services.wiki.page_service import PageService
from test.support.db import TestDatabaseUnavailable, make_test_engine


class FakeLLM:
    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


class FlakyIndexer:
    """索引替身：接口只关心它被调用，不关心索引真的建了没有。"""

    def index_page(self, *args, **kwargs):
        return True


class TransformAPITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls._engine_probe = make_test_engine()
        except TestDatabaseUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls._engine_probe.dispose()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.engine = make_test_engine()
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        app = FastAPI()
        app.include_router(transform_api.router, prefix="/api/transform")

        def override_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        def override_page_service():
            db = self.session_factory()
            try:
                yield PageService(
                    db,
                    pages_dir=str(self.root / "pages"),
                    index_service=FlakyIndexer(),
                )
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[transform_api.page_service] = override_page_service
        self.client = TestClient(app)
        self.previous = {
            "doc_restructure_max_chars": settings.doc_restructure_max_chars,
            "doc_restructure_max_tokens": settings.doc_restructure_max_tokens,
        }

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        for key, value in self.previous.items():
            setattr(settings, key, value)
        self.temporary.cleanup()

    def _upload(self, name: str, data: bytes):
        return self.client.post(
            "/api/transform/convert",
            files={"file": (name, data, "application/octet-stream")},
        )

    # ---------- convert ----------

    def test_不支持的格式被拒绝(self):
        response = self._upload("图表.png", b"\x89PNG")
        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持的文件格式", response.json()["detail"])

    def test_md_直接返回原文不经过转换(self):
        response = self._upload("笔记.md", "# 标题\n\n正文".encode("utf-8"))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["content"], "# 标题\n\n正文")
        self.assertFalse(body["converted"])
        self.assertEqual(body["title"], "笔记")

    def test_txt_同样直接返回(self):
        response = self._upload("草稿.txt", "纯文本内容".encode("utf-8"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["converted"])

    def test_非utf8的文本文件明确报错(self):
        """GBK 编码的文件不能当乱码读进来，那样正文全是问号。"""
        response = self._upload("gbk.txt", "中文内容".encode("gbk"))
        self.assertEqual(response.status_code, 422)
        self.assertIn("UTF-8", response.json()["detail"])

    def test_空白文件被拒绝(self):
        response = self._upload("空.md", b"   \n\n  ")
        self.assertEqual(response.status_code, 422)
        self.assertIn("内容为空", response.json()["detail"])

    def test_pdf走转换并标出converted(self):
        with patch(
            "backend.app.services.ingestion.doc_convert_service.get_converter",
            return_value=lambda path: "# 从 PDF 转来的",
        ):
            response = self._upload("论文.pdf", b"%PDF-1.4 fake")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["converted"])
        self.assertEqual(body["content"], "# 从 PDF 转来的")

    def test_转换失败返回422而不是500(self):
        """转换失败是文件本身的问题，用户能自己换一个，不该报成服务端故障。"""
        def boom(path):
            raise ValueError("不是有效的 PDF")

        with patch(
            "backend.app.services.ingestion.doc_convert_service.get_converter",
            return_value=boom,
        ):
            response = self._upload("坏的.pdf", b"garbage")

        self.assertEqual(response.status_code, 422)
        self.assertIn("文档转换失败", response.json()["detail"])

    # ---------- restructure ----------

    def test_重排返回原文与结果的对照(self):
        llm = FakeLLM(reply="# 加好结构的正文")
        with patch.object(
            DocRestructureService, "llm", new_callable=lambda: property(lambda self: llm)
        ):
            response = self.client.post(
                "/api/transform/restructure",
                json={"content": "没有结构的正文"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["content"], "# 加好结构的正文")
        # 前端用它做左右对比，缺了原文就没法比。
        self.assertEqual(body["original"], "没有结构的正文")
        self.assertEqual(body["original_chars"], 7)

    def test_空内容被schema拦住(self):
        response = self.client.post("/api/transform/restructure", json={"content": ""})
        self.assertEqual(response.status_code, 422)

    def test_超长内容返回422并提示拆分(self):
        settings.doc_restructure_max_chars = 50
        with patch.object(
            DocRestructureService, "llm",
            new_callable=lambda: property(lambda self: FakeLLM(reply="x")),
        ):
            response = self.client.post(
                "/api/transform/restructure",
                json={"content": "字" * 60},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("拆分", response.json()["detail"])

    def test_模型报错返回502(self):
        llm = FakeLLM(error=RuntimeError("网关超时"))
        with patch.object(
            DocRestructureService, "llm", new_callable=lambda: property(lambda self: llm)
        ):
            response = self.client.post(
                "/api/transform/restructure",
                json={"content": "正文"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("重排失败", response.json()["detail"])

    # ---------- commit ----------

    def test_写入知识库并返回页面id(self):
        response = self.client.post(
            "/api/transform/commit",
            json={"title": "转换结果", "content": "# 标题\n\n正文", "source_uri": "probe:1"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["page_id"])
        self.assertEqual(body["title"], "转换结果")
        self.assertFalse(body["deduplicated"])

    def test_相同来源重复提交被幂等跳过(self):
        payload = {"title": "转换结果", "content": "# 标题", "source_uri": "probe:same"}
        first = self.client.post("/api/transform/commit", json=payload).json()
        second = self.client.post("/api/transform/commit", json=payload).json()

        self.assertFalse(first["deduplicated"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(first["page_id"], second["page_id"])


if __name__ == "__main__":
    unittest.main()
