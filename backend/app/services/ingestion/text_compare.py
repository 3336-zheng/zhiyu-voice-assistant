"""文本比对工具：判断一次处理有没有改坏原文里的数字。

用在两处：

    PDF 解析    pypdfium2 的版面重建有可能把 `1.5` 拆成 `1 5`、把负号吃掉。
    LLM 重排    模型被要求「只加结构不改内容」，但实际会顺手改数字。

**为什么只盯数字**：文字可以同义替换、可以调语序，这些不一定算错；数字改了就
一定是错的——`1.5` 和 `1 5` 是两个不同的值，`2026年9月18日` 少一位就是另一个
日期。所以数字是「改动是否越界」里最可靠、也最该自动检出的信号。

**只报不拦**：调用方拿到结果后自己决定——PDF 那边用来在两条解析路径里择优，
重排那边用来在预览界面上提示用户。这里不做拦截，因为它不知道上下文
（比如 `1 5` 在原文里本来就分开写的）。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# 数字连同可能紧跟的单位符号。负号要认三种写法：ASCII 的 -、全角 −、
# 以及排版常见的连字符 -。
#
# 小数、千分位、日期和比例（1.5 / 1,000 / 2026/09 / 3:1）都算一个整体——
# 拆开比对的话 `1.5` 和 `1 5` 会得到相同的多重集，正好漏掉要检出的那种改动。
_NUMBER_PATTERN = re.compile(r"[+\-−]?\d+(?:[.,:/]\d+)*(?:[%‰])?")


def normalize_for_compare(text: str) -> str:
    """归一化后再比对。

    NFKC 把全角数字（２０２６）转成半角（2026）。中文文档里全角很常见，
    不归一化的话「原文全角、输出半角」会被误判成数字被改过。
    """
    return unicodedata.normalize("NFKC", text or "")


def number_multiset(text: str) -> Counter:
    """文本里所有数字的多重集。

    用多重集而不是集合：`1.5` 出现两次改成一次也是改动，集合看不出来。
    """
    return Counter(_NUMBER_PATTERN.findall(normalize_for_compare(text)))


def numbers_changed(original: str, candidate: str) -> tuple[bool, dict]:
    """比对两段文本的数字。

    返回 (是否变化, 明细)。明细里 missing 是原文有、候选没有的，added 是反过来的，
    各取前几个做示例——只给数量的话用户不知道该去哪里核对。
    """
    before = number_multiset(original)
    after = number_multiset(candidate)
    if before == after:
        return False, {"missing": [], "added": []}

    missing = sorted((before - after).elements())
    added = sorted((after - before).elements())
    return True, {
        # 截断是为了界面上好展示，完整差异对用户没有更多信息。
        "missing": missing[:10],
        "added": added[:10],
        "missing_count": sum((before - after).values()),
        "added_count": sum((after - before).values()),
    }
