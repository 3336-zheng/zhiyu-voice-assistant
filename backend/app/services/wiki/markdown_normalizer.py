"""Wiki Markdown 标题清洗工具。"""

import re


# 标题中的链接正文是页面可读名称，目标地址不应进入标题或章节元数据。
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((?:[^()\\]|\\.)*\)")
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]+)\]\((?:[^()\\]|\\.)*\)")
_HEADING_PATTERN = re.compile(r"^(\s{0,3}#{1,6})(\s+)(.*?)(\s*)$")


def clean_markdown_link_label(value: str) -> str:
    """将 Markdown 链接或图片转换为可读标签，保留普通文本。"""
    text = str(value or "").strip()
    previous = None
    while text != previous:
        previous = text
        text = _MARKDOWN_IMAGE_PATTERN.sub(r"\1", text)
        text = _MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_page_title(value: str) -> str:
    """清洗页面标题，但不改变标题之外的正文链接。"""
    return clean_markdown_link_label(value)


def normalize_heading_links(content: str) -> str:
    """只清洗 Markdown 标题行，正文中的外部链接保持不变。"""
    lines = []
    fence = None
    for line in str(content or "").split("\n"):
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            lines.append(line)
            continue
        if fence is not None:
            lines.append(line)
            continue
        match = _HEADING_PATTERN.match(line)
        if not match:
            lines.append(line)
            continue
        prefix, spacing, title, trailing = match.groups()
        cleaned_title = clean_markdown_link_label(title)
        lines.append(f"{prefix}{spacing}{cleaned_title}{trailing}")
    return "\n".join(lines)
