"""
文档格式转换服务
将 PDF、Word (.docx/.doc) 转换为 Markdown 文本

**当前没有调用者**，这是有意的：知识库上传已收窄成只接受 .md/.txt（见
api/wiki/documents.py），PDF/Word 要经过独立的「文档转换」入口，
那个入口还没做，做完后由它调用这里。

不要因为「搜不到引用」就删掉——转换逻辑（尤其是按字号推标题层级的
_font_size_to_heading_level 那一套）是这个项目里最难重写的部分之一。
"""
from __future__ import annotations
import re
import logging
from typing import Optional

try:
    from docx.text.paragraph import Paragraph
except ImportError:
    Paragraph = None

logger = logging.getLogger(__name__)


# PDF 文本层里混进来的不可见字符。­ 是软连字符（英文换行断词用的），
# ​-‏ 是各种零宽符，﻿ 是 BOM，￾/￿ 是非法码位。
# 它们在正文里看不见，但会让检索、标题匹配、文本对比全都对不上，所以删掉。
#
# 这组字符取自 WeKnora 的 docreader（_PDF_ARTIFACT_RE）——实测智语的样本里
# 确实会撞上，收据的「2664-1095」被输出成「2664￾1095」。
_PDF_ARTIFACT_RE = re.compile(r"[\u00ad\u200b-\u200f\ufeff\ufffe\uffff]")


class DocumentParseError(ValueError):
    """文档无法解析，且原因是用户能理解、能处理的（如扫描件没有文字层）。

    单独一个类型是为了让上层把「文件本身的问题」和「程序出错了」分开报——
    前者该提示用户换文件，后者才该 500。
    """


def convert_pdf_to_markdown(filepath: str) -> str:
    """
    将 PDF 文件转换为 Markdown 文本。
    利用字符级排版信息（字号、坐标）通用识别文档结构。

    转换步骤：
    1. 提取每个字符的坐标和字号
    2. 按 Y 坐标聚合为行，记录每行的最大字号
    3. 统计正文字号，字号大于正文 → 映射为 Markdown 标题
    4. 检测列表项（•、-、数字编号开头）
    5. 按行间距合并段落

    **底层用 pypdfium2 而不是 pdfplumber**：实测同一份收据，pdfplumber 的输出
    含 36 个 NUL 字符、日期月份的数字被吃掉（`2026年9月18日` → `2026918`）、
    页眉跑到正文末尾、双栏内容交织。pypdfium2 这些一个都没有——它自带阅读
    顺序还原和多栏切分，这是它和 pdfplumber 的根本差别，不是调参能补的。
    智语上面那套「字号推标题层级」的逻辑没有变，只是换了取字符的库。

    Args:
        filepath: PDF 文件路径

    Returns:
        str: Markdown 格式文本
    """
    import pypdfium2 as pdfium

    all_lines = []  # 跨页收集所有行

    document = pdfium.PdfDocument(filepath)
    try:
        line_count = 0
        for page in document:
            textpage = page.get_textpage()
            lines = _extract_page_lines(textpage)
            line_count += len(lines)
            if lines:
                all_lines.extend(lines)
            # 文本页对象不显式关会一路攒着，大文档下内存涨得很快。
            textpage.close()
            page.close()
    finally:
        document.close()

    if line_count == 0:
        # 扫描件（整页是图片、没有文字层）走这里。明确告诉用户原因，
        # 而不是返回空字符串让他以为程序坏了。
        raise DocumentParseError(
            "这份 PDF 没有可提取的文字层，可能是扫描件或纯图片 PDF。"
            "请先用 OCR 工具转换成可选中文本的 PDF。"
        )

    if not all_lines:
        return ""

    # 计算正文字号（众数）
    body_size = _get_body_font_size(all_lines)

    # 逐行分类并生成 Markdown
    md_parts = []
    # 累积普通文本行。存整个行字典而不是字符串——分段要用 top 算行距。
    paragraph_buffer = []

    def flush_paragraph():
        if paragraph_buffer:
            md_parts.append(_merge_paragraph_lines(paragraph_buffer))
            paragraph_buffer.clear()

    for line in all_lines:
        text = line["text"].strip()
        if not text:
            flush_paragraph()
            continue

        # 检测列表项
        if re.match(r"^[•\-\*]\s+", text) or re.match(r"^\d+[\.\)]\s+", text):
            flush_paragraph()
            # 无序列表
            if re.match(r"^[•\-\*]\s+", text):
                text = re.sub(r"^[•\-\*]\s+", "- ", text)
            md_parts.append(text)
            continue

        # 检测标题：字号明显大于正文
        heading_level = _font_size_to_heading_level(line["max_size"], body_size)
        if heading_level:
            flush_paragraph()
            md_parts.append(f"{'#' * heading_level} {text}")
            continue

        # 普通文本 → 累积为段落
        paragraph_buffer.append({**line, "text": text})

    flush_paragraph()

    return "\n\n".join(md_parts)


# 行距超过「行高 × 这个比值」就认为是新段落。
#
# 实测当前样本的分布（间距 ÷ 行高）：
#   同段落内换行   收据 1.57~2.39   简历 1.18~1.85
#   跨段落/区块    收据 4.26、4.57、10.31   简历 2.11、2.51
# 3.0 取在两组之间，两边都留了余量。定得太小会把正文拆碎，太大则所有行粘成
# 一段——收据那类「每行一个字段」的文档粘起来就没法看了。
PARAGRAPH_GAP_RATIO = 3.0


def _merge_paragraph_lines(lines: list) -> str:
    """把连续的文本行拼成段落，行距大的地方断开。

    拼接时中文行之间不插空格——插了会变成「这是 一段 中文」。只有行尾和
    下一行开头都是 ASCII 字母/数字时才补空格（英文换行的场景），否则
    `inter` + `national` 会粘成 `international`。
    """
    if not lines:
        return ""

    def merge_block(block: list) -> str:
        merged = block[0]["text"]
        for item in block[1:]:
            previous = merged[-1] if merged else ""
            following = item["text"][0] if item["text"] else ""
            if previous and following and previous.isascii() and following.isascii():
                merged = f"{merged} {item['text']}"
            else:
                merged = f"{merged}{item['text']}"
        return merged

    blocks = []
    current = [lines[0]]
    for line in lines[1:]:
        previous = current[-1]
        gap = None
        if previous.get("top") is not None and line.get("top") is not None:
            gap = abs(line["top"] - previous["top"])
        # 拿不到坐标或字号时保守处理：当作同段落，不轻易拆。
        if (
            gap is not None
            and previous.get("max_size")
            and gap > previous["max_size"] * PARAGRAPH_GAP_RATIO
        ):
            blocks.append(merge_block(current))
            current = [line]
        else:
            current.append(line)
    blocks.append(merge_block(current))

    return "\n\n".join(blocks)


def _extract_page_lines(textpage) -> list:
    """按 pypdfium2 给出的阅读顺序取行。

    **不要按坐标重新排序**。pypdfium2 已经做过阅读顺序还原：多栏 PDF 里它会
    先给完左栏再给右栏，这个顺序体现在字符的输出次序上，行分隔是 `\\r\\n`。
    如果这里再按 top 坐标全局排一遍，左右栏就会按 y 交错在一起——正好把它
    还原好的顺序毁掉。旧版（pdfplumber）需要排序，是因为那个库只按坐标吐
    字符、没有顺序概念。

    bold 信息拿不到：pdfium 的 C API 有字体查询，但 pypdfium2 的 Python 绑定
    没有暴露。所以旧版那条「加粗短行当小标题」的规则一并去掉了——换库换来
    阅读顺序和正确的标点，代价是这一条启发式规则，值。

    Args:
        textpage: pypdfium2 的 PdfTextPage

    Returns:
        list[dict]: [{text, max_size}, ...]
    """
    lines = []
    current = []

    for index in range(textpage.count_chars()):
        char = textpage.get_text_range(index, 1)
        if char in "\r\n":
            if current:
                lines.append(_build_line(current))
                current = []
            continue
        # 空格必须留着，而且不能做宽度检查：PDF 里的空格往往只靠「下一个字符
        # 的 x 偏移」表示，它自己的 box 宽度是 0。按宽度过滤会把所有空格吃掉，
        # 「账单编号 JVMEWIEU-0001」会粘成「账单编号JVMEWIEU-0001」。
        if char.isspace():
            current.append({"text": char, "size": 0})
            continue
        # 其余零宽字符（有些 PDF 用它们做隐藏文本层）才是要丢的，否则会把
        # 行高统计拉到 0。
        x0, y0, x1, y1 = textpage.get_charbox(index)
        height = y1 - y0
        if height <= 0 or (x1 - x0) <= 0:
            continue
        # top 取负号：pypdfium2 的 y 轴从页面底部往上，转成「从上往下」的
        # 直观方向，和 _extract_page_lines 里断行的顺序一致。
        current.append({"text": char, "size": height, "top": -y1})

    if current:
        lines.append(_build_line(current))

    return lines


def _build_line(chars: list) -> dict:
    """将一行字符组装为行信息字典"""
    raw_text = "".join(c.get("text", "") for c in chars)
    # 清洗放在这里而不是取字符时：line 的 text 是唯一对外的东西，
    # 在这里清一次就够，size / top 的统计不受影响。
    text = _PDF_ARTIFACT_RE.sub("", raw_text).strip()
    sizes = [c.get("size", 12) for c in chars if c.get("text", "").strip()]
    max_size = max(sizes) if sizes else 12
    # 行位置取该行第一个有实际坐标的字符。空格没有坐标（见 _extract_page_lines），
    # 所以行首是空格时不能拿它当基准。
    tops = [c["top"] for c in chars if "top" in c]
    return {"text": text, "max_size": max_size, "top": min(tops) if tops else None}


def _get_body_font_size(lines: list) -> float:
    """
    统计正文字号（出现频率最高的字号）。
    只统计合理范围内的字号（排除极小的注释和极大的标题）。
    """
    from collections import Counter

    sizes = []
    for line in lines:
        if line.get("is_table"):
            continue
        size = line.get("max_size", 0)
        if 8 <= size <= 24:
            sizes.append(round(size, 1))

    if not sizes:
        return 12.0

    counter = Counter(sizes)
    return counter.most_common(1)[0][0]


def _font_size_to_heading_level(font_size: float, body_size: float) -> int:
    """
    根据字号与正文字号的比值映射标题层级。
    返回 0 表示不是标题。
    """
    if body_size <= 0 or font_size <= 0:
        return 0

    ratio = font_size / body_size

    if ratio >= 1.8:
        return 1
    elif ratio >= 1.4:
        return 2
    elif ratio >= 1.15:
        return 3
    return 0


def convert_docx_to_markdown(filepath: str) -> str:
    """
    将 Word (.docx) 文件转换为 Markdown 文本。
    保留标题层级、段落、列表、表格。

    Args:
        filepath: Word 文件路径

    Returns:
        str: Markdown 格式文本
    """
    from docx import Document
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    doc = Document(filepath)
    md_lines = []

    # 遍历文档元素（段落和表格按文档顺序交错）
    for element in iter_block_items(doc):
        if isinstance(element, Paragraph):
            md_line = _paragraph_to_markdown(element)
            if md_line is not None:
                md_lines.append(md_line)
        elif isinstance(element, DocxTable):
            md_table = _docx_table_to_markdown(element)
            if md_table:
                md_lines.append("")  # 表格前后留空行
                md_lines.append(md_table)
                md_lines.append("")

    return "\n".join(md_lines)


# ---- 内部辅助函数 ----

def _table_to_markdown(table: list) -> str:
    """
    将二维列表转为 Markdown 表格。
    第一行为表头。
    """
    if not table or len(table) < 1:
        return ""

    # 清洗单元格
    cleaned = []
    for row in table:
        cleaned.append([str(cell).strip() if cell else "" for cell in row])

    headers = cleaned[0]
    rows = cleaned[1:]

    # 计算列数（取最大列数）
    max_cols = max(len(r) for r in cleaned)
    # 补齐列数
    headers = headers + [""] * (max_cols - len(headers))
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _paragraph_to_markdown(para: Paragraph) -> Optional[str]:
    """
    将 Word 段落转为 Markdown 行。
    支持标题（Heading 1-6）、普通段落、列表项。
    """
    text = para.text.strip()
    if not text:
        return ""  # 空行保留，用于段落分隔

    style_name = para.style.name.lower() if para.style else ""

    # 标题样式映射
    if "heading" in style_name:
        # 提取标题级别：Heading 1 -> #, Heading 2 -> ##, ...
        level_match = re.search(r"heading\s*(\d)", style_name)
        if level_match:
            level = int(level_match.group(1))
            level = min(level, 6)
            return f"{'#' * level} {text}"

    # 列表项
    if "list" in style_name:
        # 尝试判断有序/无序
        if "number" in style_name or "ordered" in style_name:
            return f"1. {text}"
        return f"- {text}"

    # 普通段落
    return text


def iter_block_items(doc):
    """
    按文档顺序遍历段落和表格（python-docx 不保证 document.paragraphs 包含表格）。
    参考 python-docx 官方示例。
    """
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, doc)


def _docx_table_to_markdown(table) -> str:
    """将 python-docx Table 对象转为 Markdown 表格"""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(cells)

    if not rows:
        return ""

    return _table_to_markdown(rows)


def get_converter(filepath: str):
    """
    根据文件扩展名返回对应的转换函数。

    Args:
        filepath: 文件路径

    Returns:
        转换函数，或 None（不支持的格式）
    """
    lower = filepath.lower()
    if lower.endswith(".pdf"):
        return convert_pdf_to_markdown
    elif lower.endswith(".docx"):
        return convert_docx_to_markdown
    return None
