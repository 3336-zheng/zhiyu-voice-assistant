"""PDF 解析测试。

底层从 pdfplumber 换成 pypdfium2 之后，这几个坑都实际踩过，且表现都是静默的
（输出看着像正常文本，只是少字、少空格、顺序乱），所以必须有测试兜住：

  空格丢失      PDF 里空格的 box 宽度是 0，按宽度过滤会把「账单编号 JVMEWEI」
                 粘成「账单编号JVMEWEI」
  行列错乱      按坐标全局排序会毁掉 pypdfium2 还原好的多栏阅读顺序
  不可见字符    软连字符等混在正文里，检索和标题匹配都对不上
  过度合并      所有行粘成一段，「每行一个字段」的文档没法看
  扫描件        没有文字层时返回空串，用户以为程序坏了

用假的 pdfium 对象而不是真 PDF 文件：仓库里不放二进制样本，而这里要验的是
「拿到字符之后怎么处理」，字符序列本身就是最好的输入。
"""

import unittest
from unittest.mock import patch

from backend.app.services.ingestion.doc_convert_service import (
    PARAGRAPH_GAP_RATIO,
    DocumentParseError,
    convert_pdf_to_markdown,
    _merge_paragraph_lines,
)


def glyph(char, height=12.0, top=0.0):
    """描述一个字符：内容、字号（box 高度）、行位置（从上往下）。"""
    return (char, height, top)


class FakeTextPage:
    """pypdfium2 的 PdfTextPage 替身。

    `get_charbox` 的返回值要能反映出 height 和 top：真实实现返回
    (x0, y0, x1, y1) 且 y 轴从下往上，所以 y0 = -top-height、y1 = -top。
    """

    def __init__(self, glyphs):
        self._glyphs = glyphs
        self.closed = False

    def count_chars(self):
        return len(self._glyphs)

    def get_text_range(self, index, count=1):
        return "".join(item[0] for item in self._glyphs[index:index + count])

    def get_charbox(self, index):
        _, height, top = self._glyphs[index]
        # 空格这类字符高度为 0，真实 pdfium 也返回退化 box。
        return (0.0, -top - height, max(height, 0.0), -top)

    def close(self):
        self.closed = True


class FakePage:
    def __init__(self, textpage):
        self._textpage = textpage

    def get_textpage(self):
        return self._textpage

    def close(self):
        pass


class FakeDocument:
    def __init__(self, pages):
        self._pages = pages

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        pass


def parse(glyphs, pages=1):
    """把一组字符喂给 convert_pdf_to_markdown（单个文本页）。"""
    textpage = FakeTextPage(glyphs)
    document = FakeDocument([FakePage(textpage)] * pages)
    with patch("pypdfium2.PdfDocument", return_value=document):
        return convert_pdf_to_markdown("ignored.pdf")


def lines_of(*specs):
    """把 (文本, 字号, 行位置) 展开成字符序列，行尾自动补换行符。

    行内的空格用 height=0 表示，与真实 PDF 一致。
    """
    glyphs = []
    for text, height, top in specs:
        for char in text:
            glyphs.append(glyph(char, 0.0 if char == " " else height, top))
        glyphs.append(glyph("\r", 0.0, top))
        glyphs.append(glyph("\n", 0.0, top))
    return glyphs


class ArtifactCleanupTestCase(unittest.TestCase):
    """PDF 文本层混进来的不可见字符必须清掉。"""

    def test_软连字符被清除(self):
        out = parse(lines_of(("2664￾1095", 12.0, 0.0)))
        self.assertNotIn("￾", out)

    def test_零宽字符被清除(self):
        out = parse(lines_of(("正​文", 12.0, 0.0)))
        self.assertNotIn("​", out)
        self.assertIn("正文", out)

    def test_正常文本不受影响(self):
        out = parse(lines_of(("正常文本", 12.0, 0.0)))
        self.assertIn("正常文本", out)


class WhitespaceTestCase(unittest.TestCase):
    """空格必须保住——它在 PDF 里的 box 宽度是 0，容易被当成空字符过滤掉。"""

    def test_行内空格被保留(self):
        out = parse(lines_of(("账单编号 JVMEWIEU", 12.0, 0.0)))
        self.assertIn("账单编号 JVMEWIEU", out)

    def test_连续多个空格被保留(self):
        out = parse(lines_of(("a   b", 12.0, 0.0)))
        self.assertIn("a   b", out)

    def test_纯空格行不会产出空标题(self):
        out = parse(lines_of(("   ", 12.0, 0.0), ("正文", 12.0, 20.0)))
        self.assertNotIn("#   ", out)


class HeadingTestCase(unittest.TestCase):
    """字号推标题层级——这是智语原有的逻辑，换库后必须仍然生效。"""

    def test_字号明显大的行成为标题(self):
        out = parse(lines_of(
            ("第一章 概述", 24.0, 0.0),
            ("正文第一段", 12.0, 40.0),
            ("正文第二段", 12.0, 60.0),
            ("正文第三段", 12.0, 80.0),
        ))
        self.assertIn("# 第一章 概述", out)

    def test_正文不会被误判成标题(self):
        out = parse(lines_of(
            ("第一章", 24.0, 0.0),
            ("这是一段正文", 12.0, 40.0),
            ("这是另一段正文", 12.0, 60.0),
            ("第三段正文", 12.0, 80.0),
        ))
        self.assertNotIn("# 这是一段正文", out)

    def test_无标题文档输出普通段落(self):
        out = parse(lines_of(("只有正文内容", 12.0, 0.0)))
        self.assertNotIn("#", out)

    def test_正文基准取众数而不是最大字号(self):
        """正文行数远多于标题行时，基准落在正文字号上。

        这是个**已知边界**：如果文档只有一行标题和一行正文，两个字号出现
        次数平局，Counter.most_common 会取先出现的那个（标题），于是标题
        识别失效。真实文档里正文行不可能这么少，所以不额外处理。
        """
        out = parse(lines_of(
            ("大标题", 20.0, 0.0),
            ("正文甲", 10.0, 40.0),
            ("正文乙", 10.0, 55.0),
            ("正文丙", 10.0, 70.0),
        ))
        self.assertIn("# 大标题", out)


class ParagraphTestCase(unittest.TestCase):
    """行距决定分段——所有行粘成一段会让「每行一个字段」的文档没法看。"""

    def test_行距大时分成两段(self):
        out = parse(lines_of(
            ("第一段第一行", 12.0, 0.0),
            ("第一段第二行", 12.0, 14.0),      # 行距 14 / 12 ≈ 1.17，同段
            ("第二段内容", 12.0, 120.0),       # 行距 106 / 12 ≈ 8.8，分段
        ))
        self.assertIn("\n\n", out)
        blocks = out.split("\n\n")
        self.assertEqual(len(blocks), 2)
        self.assertIn("第一段第一行", blocks[0])
        self.assertIn("第一段第二行", blocks[0])
        self.assertIn("第二段内容", blocks[1])

    def test_行距小时合成一段(self):
        out = parse(lines_of(
            ("上句结尾", 12.0, 0.0),
            ("下句开始", 12.0, 14.0),
        ))
        self.assertEqual(len(out.split("\n\n")), 1)

    def test_中文行之间不插空格(self):
        out = parse(lines_of(("这是上半句", 12.0, 0.0), ("这是下半句", 12.0, 14.0)))
        self.assertIn("这是上半句这是下半句", out)

    def test_英文行之间补空格(self):
        """不补的话 inter + national 会粘成 international。"""
        out = parse(lines_of(("inter", 12.0, 0.0), ("national", 12.0, 14.0)))
        self.assertIn("inter national", out)


class MergeParagraphLinesTestCase(unittest.TestCase):
    """直接测拼接函数，不经过 PDF 层。"""

    def _line(self, text, top, size=12.0):
        return {"text": text, "max_size": size, "top": top}

    def test_空列表返回空串(self):
        self.assertEqual(_merge_paragraph_lines([]), "")

    def test_缺少坐标时保守合并(self):
        """拿不到 top 就不拆——宁可少分段，也不要把一段正文拆碎。"""
        lines = [{"text": "甲乙"}, {"text": "丙丁"}]
        self.assertEqual(_merge_paragraph_lines(lines), "甲乙丙丁")

    def test_缺少字号时不拆(self):
        lines = [{"text": "甲", "top": 0.0}, {"text": "乙", "top": 999.0}]
        self.assertEqual(_merge_paragraph_lines(lines), "甲乙")

    def test_阈值边界(self):
        size = 10.0
        gap_at_limit = size * PARAGRAPH_GAP_RATIO
        lines = [
            self._line("上", 0.0, size),
            self._line("下", gap_at_limit, size),
        ]
        # 恰好等于阈值时不分段（用 > 判断，不是 >=）。
        self.assertEqual(_merge_paragraph_lines(lines), "上下")


class ScannedDocumentTestCase(unittest.TestCase):
    def test_没有文字层时给出可操作的报错(self):
        """扫描件返回空串会让用户以为程序坏了，要明确说明原因和下一步。"""
        with patch("pypdfium2.PdfDocument", return_value=FakeDocument([FakePage(FakeTextPage([]))])):
            with self.assertRaises(DocumentParseError) as ctx:
                convert_pdf_to_markdown("scan.pdf")
        message = str(ctx.exception)
        self.assertIn("文字层", message)
        self.assertIn("OCR", message)


class ResourceCleanupTestCase(unittest.TestCase):
    def test_文本页被显式关闭(self):
        """不关的话大文档下内存涨得很快。"""
        textpage = FakeTextPage(lines_of(("内容", 12.0, 0.0)))
        document = FakeDocument([FakePage(textpage)])
        with patch("pypdfium2.PdfDocument", return_value=document):
            convert_pdf_to_markdown("ignored.pdf")
        self.assertTrue(textpage.closed)


if __name__ == "__main__":
    unittest.main()
