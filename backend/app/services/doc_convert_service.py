"""
文档格式转换服务
将 PDF、Word (.docx/.doc) 转换为 Markdown 文本
"""
from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(filepath: str) -> str:
    """
    将 PDF 文件转换为 Markdown 文本。
    利用字符级排版信息（字号、加粗、坐标）通用识别文档结构。

    转换步骤：
    1. 提取每个字符的坐标、字号、字体信息
    2. 按 Y 坐标聚合为行，记录每行的最大字号和加粗状态
    3. 统计正文字号，字号大于正文 → 映射为 Markdown 标题
    4. 检测列表项（•、-、数字编号开头）
    5. 按行间距合并段落

    Args:
        filepath: PDF 文件路径

    Returns:
        str: Markdown 格式文本
    """
    import pdfplumber

    all_lines = []  # 跨页收集所有行

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            for table in (tables or []):
                if table and len(table) >= 2:
                    all_lines.append({
                        "text": _table_to_markdown(table),
                        "max_size": 0,
                        "all_bold": False,
                        "is_table": True,
                    })

            # 用 page.chars 提取字符级信息
            chars = page.chars
            if not chars:
                continue

            lines = _group_chars_to_lines(chars)
            all_lines.extend(lines)

    if not all_lines:
        return ""

    # 计算正文字号（众数）
    body_size = _get_body_font_size(all_lines)

    # 逐行分类并生成 Markdown
    md_parts = []
    paragraph_buffer = []  # 累积普通文本行，合并为段落

    def flush_paragraph():
        if paragraph_buffer:
            md_parts.append(" ".join(paragraph_buffer))
            paragraph_buffer.clear()

    for line in all_lines:
        # 表格直接输出
        if line.get("is_table"):
            flush_paragraph()
            md_parts.append(line["text"])
            continue

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

        # 检测加粗独立短行（< 50 字符，上下有空行或边界）→ 疑似小标题
        if line["all_bold"] and len(text) < 50:
            flush_paragraph()
            md_parts.append(f"## {text}")
            continue

        # 普通文本 → 累积为段落
        paragraph_buffer.append(text)

    flush_paragraph()

    return "\n\n".join(md_parts)


def _group_chars_to_lines(chars: list) -> list:
    """
    将 PDF 字符按 Y 坐标聚合为行。
    每行记录：文本内容、最大字号、是否全部加粗。

    Args:
        chars: pdfplumber page.chars 列表

    Returns:
        list[dict]: [{text, max_size, all_bold}, ...]
    """
    if not chars:
        return []

    # 按 top 坐标排序
    sorted_chars = sorted(chars, key=lambda c: (c.get("top", 0), c.get("x0", 0)))

    lines = []
    current_line_chars = [sorted_chars[0]]

    for char in sorted_chars[1:]:
        prev = current_line_chars[-1]
        # 同一行判断：top 坐标差值 < 字号的一半
        threshold = max(prev.get("size", 12), char.get("size", 12)) * 0.5
        if abs(char.get("top", 0) - prev.get("top", 0)) < threshold:
            current_line_chars.append(char)
        else:
            lines.append(_build_line(current_line_chars))
            current_line_chars = [char]

    if current_line_chars:
        lines.append(_build_line(current_line_chars))

    return lines


def _build_line(chars: list) -> dict:
    """将一行字符组装为行信息字典"""
    text = "".join(c.get("text", "") for c in chars).strip()
    sizes = [c.get("size", 12) for c in chars if c.get("text", "").strip()]
    max_size = max(sizes) if sizes else 12
    all_bold = all(
        "bold" in c.get("fontname", "").lower()
        for c in chars if c.get("text", "").strip()
    )
    return {"text": text, "max_size": max_size, "all_bold": all_bold}


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
