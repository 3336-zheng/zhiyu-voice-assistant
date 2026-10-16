"""数字守卫测试。

它只做一件事：判断一次处理有没有改坏原文里的数字。用在 PDF 解析（两条路径
择优）和 LLM 重排（预览时提示用户）两处。

**为什么只盯数字**：文字可以同义替换、可以调语序，这些不一定算错；数字改了
就一定是错的。所以它是「改动是否越界」里最可靠、最该自动检出的信号。
"""

import unittest

from backend.app.services.ingestion.text_compare import (
    normalize_for_compare,
    number_multiset,
    numbers_changed,
)


class NormalizeTestCase(unittest.TestCase):
    def test_全角数字转半角(self):
        """中文文档里全角很常见。不归一化的话「原文全角、输出半角」会被
        误判成数字被改过。"""
        self.assertEqual(normalize_for_compare("２０２６"), "2026")
        self.assertEqual(normalize_for_compare("１２．５"), "12.5")

    def test_半角保持原样(self):
        self.assertEqual(normalize_for_compare("2026"), "2026")

    def test_空输入不报错(self):
        self.assertEqual(normalize_for_compare(""), "")
        self.assertEqual(normalize_for_compare(None), "")


class NumberMultisetTestCase(unittest.TestCase):
    def test_提取各种写法(self):
        text = "准确率 1.5，召回 0.85，共 1,000 条，比例 3:1，涨 20%"
        self.assertEqual(
            number_multiset(text),
            {"1.5": 1, "0.85": 1, "1,000": 1, "3:1": 1, "20%": 1},
        )

    def test_小数算一个整体(self):
        """拆开比对的话 `1.5` 和 `1 5` 会得到相同的多重集，正好漏掉要检出的改动。"""
        self.assertEqual(number_multiset("1.5"), {"1.5": 1})
        self.assertNotEqual(number_multiset("1.5"), number_multiset("1 5"))

    def test_用多重集不是集合(self):
        """`1.5` 出现两次改成一次也是改动，集合看不出来。"""
        self.assertEqual(number_multiset("1.5 和 1.5"), {"1.5": 2})
        self.assertNotEqual(number_multiset("1.5 1.5"), number_multiset("1.5"))

    def test_负号三种写法都认(self):
        for dash in ["-", "−"]:
            with self.subTest(dash=dash):
                self.assertEqual(number_multiset(f"{dash}5"), {f"{dash}5": 1})

    def test_全角输入被归一化(self):
        self.assertEqual(number_multiset("２０２６"), number_multiset("2026"))


class NumbersChangedTestCase(unittest.TestCase):
    def test_数字没变时返回假(self):
        changed, diff = numbers_changed("准确率 1.5", "**准确率** 1.5")
        self.assertFalse(changed)
        self.assertEqual(diff["missing"], [])
        self.assertEqual(diff["added"], [])

    def test_只加结构不算数字变动(self):
        """重排的正常结果是「文字一样、只是多了标记」，不该报警。"""
        original = "第一步做 A。第二步做 B。第三步 3 秒完成。"
        restructured = "- 第一步做 A\n- 第二步做 B\n- 第三步 **3 秒**完成\n"
        changed, _ = numbers_changed(original, restructured)
        self.assertFalse(changed)

    def test_小数点被拆开能检出(self):
        changed, diff = numbers_changed("准确率 1.5", "准确率 1 5")
        self.assertTrue(changed)
        self.assertIn("1.5", diff["missing"])
        self.assertIn("1", diff["added"])
        self.assertIn("5", diff["added"])

    def test_日期少一位能检出(self):
        changed, diff = numbers_changed("2026年9月18日", "2026年9月8日")
        self.assertTrue(changed)
        self.assertIn("18", diff["missing"])
        self.assertIn("8", diff["added"])

    def test_数字被删掉能检出(self):
        changed, diff = numbers_changed("共 3 步", "共 步")
        self.assertTrue(changed)
        self.assertIn("3", diff["missing"])

    def test_数字被凭空加上能检出(self):
        changed, diff = numbers_changed("共 步", "共 3 步")
        self.assertTrue(changed)
        self.assertIn("3", diff["added"])

    def test_全角半角混用不算变动(self):
        """原文全角、模型输出半角，是排版归一化，不是改内容。"""
        changed, _ = numbers_changed("共２０２６条", "共 2026 条")
        self.assertFalse(changed)

    def test_差异明细带上计数(self):
        changed, diff = numbers_changed("1 1 1", "1")
        self.assertTrue(changed)
        self.assertEqual(diff["missing_count"], 2)
        self.assertEqual(diff["added_count"], 0)

    def test_差异明细截断到十个(self):
        """完整差异对用户没有更多信息，界面上也放不下。"""
        original = " ".join(str(i) for i in range(30))
        changed, diff = numbers_changed(original, "")
        self.assertTrue(changed)
        self.assertEqual(len(diff["missing"]), 10)
        self.assertEqual(diff["missing_count"], 30)

    def test_空文本比对不报错(self):
        self.assertEqual(numbers_changed("", "")[0], False)
        self.assertEqual(numbers_changed("1.5", "")[0], True)


if __name__ == "__main__":
    unittest.main()
