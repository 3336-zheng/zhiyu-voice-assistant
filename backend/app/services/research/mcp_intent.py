"""识别用户是否明确要求使用 MCP 做外部查询。"""

import re


_NEGATION_PATTERN = re.compile(
    r"(?:不要|不用|无需|不使用|别用|don't use|do not use)\s*(?:调用|使用|通过)?\s*(?:mcp|firecrawl|联网|外部搜索)",
    re.IGNORECASE,
)
_EXPLICIT_MCP_PATTERN = re.compile(
    r"(?:使用|调用|通过|用|让[^，。！？]*调用|请[^，。！？]*用)\s*"
    r"(?:一下\s*)?(?:mcp|firecrawl)"
    r"|(?:use|call|via)\s+(?:mcp|firecrawl)"
    r"|(?:mcp|firecrawl)\s*(?:查询|搜索|检索|抓取|查一下|查最新)"
    r"|(?:联网|网络|外部|网页)\s*(?:查询|搜索|检索|查一下|查最新|研究)",
    re.IGNORECASE,
)


def is_explicit_mcp_request(query: str) -> bool:
    """只对明确提到 MCP、Firecrawl 或联网研究的请求启用自动外部查询。"""
    if not isinstance(query, str):
        return False
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized or _NEGATION_PATTERN.search(normalized):
        return False
    return bool(_EXPLICIT_MCP_PATTERN.search(normalized))
