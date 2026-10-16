"""识别用户是否明确要求使用 MCP 做外部查询。"""

import re


_NEGATION_PATTERN = re.compile(
    r"(?:不要|不用|无需|不使用|别用|don't use|do not use)\s*"
    r"(?:调用|使用|通过|用)?\s*"
    # 目标词要覆盖下面所有会命中的说法，否则「不要上网查」会被正向规则捞回来。
    r"(?:mcp|firecrawl|联网|上网|网上|网络|外网|外部|网页|互联网)",
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
# 指向外部的词与搜索动作词。
#
# 两者各自单独出现都不足以判定意图：「网上的资料」可能只是在描述某篇本地
# 笔记，「查一下」在本地检索里同样常见。只有同现才说明用户想去外面找。
# 拆成两个正则而不是写一个交叉模式，是因为「帮我上网查一下」中间隔着字，
# 用空白匹配的相邻模式覆盖不到。
_WEB_PATTERN = re.compile(
    r"联网|上网|网上|网络|外网|外部|网页|互联网",
    re.IGNORECASE,
)
_SEARCH_ACTION_PATTERN = re.compile(
    r"搜一下|搜搜|搜索|搜|找找|找一下|查一下|查查|查询|检索|查最新|研究",
    re.IGNORECASE,
)


def is_explicit_mcp_request(query: str) -> bool:
    """判断用户是否明确要求联网/外部查询。

    两条路径：点名 MCP、Firecrawl，或者说的是「上网搜一下」这类日常说法。
    后者宽松一些，但要求外部词和搜索动词同时出现。
    """
    if not isinstance(query, str):
        return False
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized or _NEGATION_PATTERN.search(normalized):
        return False
    if _EXPLICIT_MCP_PATTERN.search(normalized):
        return True
    return bool(
        _WEB_PATTERN.search(normalized) and _SEARCH_ACTION_PATTERN.search(normalized)
    )
