"""文档结构重排服务测试。

重排本身是 LLM 调用，测不了「排得好不好」——那是提示词的事，靠预览人工判断。
这里测的是模型之外的边界：空输入、超长输入、模型返回裹了围栏、模型返回空。
这些出错方式都是静默的（用户会拿到空页面或半篇文档），必须有测兜住。
"""

import unittest
from unittest.mock import patch

from backend.app.core.config import settings
from backend.app.services.ingestion.doc_restructure_service import (
    RESTRUCTURE_PROMPT,
    DocRestructureService,
    strip_code_fence,
)


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


class StripCodeFenceTestCase(unittest.TestCase):
    """模型爱把整篇裹进代码围栏，这个后处理必须可靠。"""

    def test_剥掉整篇围栏(self):
        wrapped = "```markdown\n# 标题\n\n正文\n```"
        self.assertEqual(strip_code_fence(wrapped), "# 标题\n\n正文")

    def test_剥掉无语言标记的围栏(self):
        self.assertEqual(strip_code_fence("```\n# 标题\n```"), "# 标题")

    def test_没有围栏时原样返回(self):
        self.assertEqual(strip_code_fence("# 标题\n\n正文"), "# 标题\n\n正文")

    def test_正文自带代码块时不被误剥(self):
        """正文里有代码块时，整篇不是「单一围栏」结构，不能动它。

        这条是防过度剥离：` ```python ... ``` ` 出现在中间时，如果正则贪心，
        会把开头到结尾之间全部当成围栏内容。
        """
        content = "# 标题\n\n```python\nprint(1)\n```\n\n结尾"
        self.assertEqual(strip_code_fence(content), content)

    def test_空输入返回空串(self):
        self.assertEqual(strip_code_fence(""), "")
        self.assertEqual(strip_code_fence(None), "")


class DocRestructureServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "doc_restructure_max_chars": settings.doc_restructure_max_chars,
            "doc_restructure_max_tokens": settings.doc_restructure_max_tokens,
            "doc_restructure_model": settings.doc_restructure_model,
        }

    def tearDown(self):
        for key, value in self.previous.items():
            setattr(settings, key, value)

    def test_空内容报错(self):
        service = DocRestructureService(llm=FakeLLM(reply="x"))
        with self.assertRaises(ValueError) as ctx:
            service.restructure("   \n  ")
        self.assertIn("为空", str(ctx.exception))

    def test_超长内容拒绝而不是静默截断(self):
        """静默截断会让用户拿到半篇文档，预览里未必看得出来。"""
        settings.doc_restructure_max_chars = 100
        service = DocRestructureService(llm=FakeLLM(reply="x"))
        with self.assertRaises(ValueError) as ctx:
            service.restructure("字" * 101)
        self.assertIn("拆分", str(ctx.exception))

    def test_模型返回空时报错而不是返回空页面(self):
        service = DocRestructureService(llm=FakeLLM(reply=""))
        with self.assertRaises(RuntimeError) as ctx:
            service.restructure("正文内容")
        self.assertIn("没有返回内容", str(ctx.exception))

    def test_调用参数带上独立的最大输出长度(self):
        """不能用全局 llm_max_tokens（1024），重排会被截断成半篇。"""
        settings.doc_restructure_max_tokens = 12_000
        llm = FakeLLM(reply="# 标题")
        service = DocRestructureService(llm=llm)

        service.restructure("原文内容")

        self.assertEqual(llm.calls[0]["max_tokens"], 12_000)
        self.assertEqual(llm.calls[0]["temperature"], 0.1)

    def test_配了专用模型就用它_否则交给_llm_service_默认(self):
        llm = FakeLLM(reply="# 标题")
        service = DocRestructureService(llm=llm)

        settings.doc_restructure_model = "openai/gpt-4.1-mini"
        service.restructure("原文")
        self.assertEqual(llm.calls[0]["model"], "openai/gpt-4.1-mini")

        settings.doc_restructure_model = ""
        service.restructure("原文")
        # 空串要传 None 而不是 ""，否则 llm_service 那边 (model or "") 也会
        # 回退到默认，但传空串语义上不清楚。
        self.assertIsNone(llm.calls[1]["model"])

    def test_提示词包含不许改内容的约束(self):
        """这几个约束是「重排但不改写」的全部依据，删了就失去意义。"""
        for keyword in ["不要增删", "不要改写", "不要纠错"]:
            self.assertIn(keyword, RESTRUCTURE_PROMPT)
        self.assertIn("{content}", RESTRUCTURE_PROMPT)

    def test_原文被填进提示词(self):
        llm = FakeLLM(reply="# 标题")
        service = DocRestructureService(llm=llm)

        service.restructure("这是需要重排的原文")

        self.assertIn("这是需要重排的原文", llm.calls[0]["messages"][0]["content"])

    def test_返回值带实际使用的模型(self):
        llm = FakeLLM(reply="# 标题")
        service = DocRestructureService(llm=llm)

        settings.doc_restructure_model = "openai/gpt-4.1-mini"
        result = service.restructure("原文")

        self.assertEqual(result["model"], "openai/gpt-4.1-mini")
        self.assertEqual(result["content"], "# 标题")

    def test_模型异常原样抛出(self):
        service = DocRestructureService(llm=FakeLLM(error=RuntimeError("网关超时")))
        with self.assertRaises(RuntimeError):
            service.restructure("原文")


if __name__ == "__main__":
    unittest.main()
