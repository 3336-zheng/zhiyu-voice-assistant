"""文档结构重排：用 LLM 给正文补上 Markdown 结构，但不改写内容。

**为什么单独一个服务**：知识库上传只接受 .md/.txt，PDF/Word 要先转换成
Markdown 再进来。但这个项目里 PDF 转换是按字号猜标题层级的（见
doc_convert_service），转出来的结构经常是「一堆没有层级的段落」；
手写的 .md 也常有不分标题、不列清单的情况。这段服务负责把这类正文
整理成结构清晰的 Markdown，再交给索引。

**不做内容校验，靠预览把关**：重排结果会以「原文 / 重排后」左右对比呈现给
用户，由用户决定是否采用。所以这里只负责「按提示词尽力做好」，不额外写
一套相似度校验去拦截——那套校验要么宽松到没用，要么严格到误杀正常重排，
而用户看一眼就知道对不对。

**边界**：不写库、不建索引。它只接收文本、返回文本，落库由调用方负责。
"""

from __future__ import annotations

import logging
import re

from backend.app.core.config import settings
from backend.app.services.ingestion.text_compare import numbers_changed

logger = logging.getLogger(__name__)


RESTRUCTURE_PROMPT = """你是 Markdown 排版编辑。任务是把用户给的纯文本整理成结构清晰的 Markdown。

**你只做排版，不做改写。** 具体的：

必须做的：
1. 按内容逻辑划分章节，用 `#` `##` `###` 标出层级（最多三级，不要更多）
2. 把并列的条目整理成无序或有序列表
3. 把明显的表格数据整理成 Markdown 表格
4. 给关键术语、结论加 `**加粗**`，方便扫读
5. 段落之间留空行，长段落按语义拆开

绝对不能做的：
1. **不要增删任何实质内容**——不补充原文没有的解释、例子、总结
2. **不要改写说法**——保留原文措辞，不要换成「更好的表达」
3. **不要合并或拆分原本的意思**——一段话讲三件事，还是三件事
4. **不要纠错**——原文的技术判断、数据、结论哪怕是错的也原样保留
5. 不要加原文没有的标题文字（比如自己总结一个「概述」章节）

允许的小调整：
- 补全明显缺失的标点
- 修正明显的错别字
- 去掉行首多余的空格

格式要求：
- 直接输出 Markdown 正文，**不要用 ``` 代码围栏把整篇包起来**
- 原文里已有的代码块、引用块原样保留，其中的内容一个字都不要动
- 只输出整理后的正文，不要输出「以下是整理结果」这类说明性文字

原文：

{content}"""


# 输出被 ``` 包起来是模型最常见的越界行为，剥掉它。
_FENCE_WRAPPER = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """剥掉整篇被代码围栏包裹的情况。

    提示词里明确说了不要包，但模型仍常这么干。只在「整篇就是一个围栏」时剥——
    如果正文自身就含代码块，那它不会是单一围栏结构，正则不会命中，原样返回。
    """
    match = _FENCE_WRAPPER.match(text or "")
    return match.group(1).strip() if match else (text or "").strip()


class DocRestructureService:
    """把正文整理成结构清晰的 Markdown。"""

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            from backend.app.services.ai.llm_service import get_llm_service

            self._llm = get_llm_service()
        return self._llm

    def restructure(self, content: str) -> dict:
        """重排正文。返回 {content, model, truncated}。

        输入过长时**明确报错而不是静默截断**：截掉一半再重排，用户拿到的
        是半篇文档，而且预览里未必看得出来。宁可让他先手动拆分。
        """
        original = content or ""
        if not original.strip():
            raise ValueError("内容为空，无法重排")

        limit = settings.doc_restructure_max_chars
        if len(original) > limit:
            raise ValueError(
                f"文档约 {len(original)} 字，超过单次重排上限 {limit} 字。"
                "请先拆分后分次处理。"
            )

        model = settings.doc_restructure_model.strip() or None
        raw = self.llm.chat(
            messages=[{
                "role": "user",
                "content": RESTRUCTURE_PROMPT.format(content=original),
            }],
            max_tokens=settings.doc_restructure_max_tokens,
            temperature=0.1,
            model=model,
            trace_name="doc.restructure",
        )
        restructured = strip_code_fence(raw)
        if not restructured:
            raise RuntimeError("模型没有返回内容，请重试")

        # 数字守卫：提示词里写了「不要改写内容」，但模型仍可能顺手改数字。
        # 这里只报不拦——重排结果本来就要给用户左右对比，把「数字被改过」
        # 显式标出来比自动回退更有用（用户可能是故意改的）。
        numbers_moved, number_diff = numbers_changed(original, restructured)
        if numbers_moved:
            logger.warning(
                "重排改动了数字: 原文独有 %s，重排后独有 %s",
                number_diff["missing"][:5],
                number_diff["added"][:5],
            )

        logger.info(
            "文档重排完成: 原文 %s 字 -> 重排后 %s 字",
            len(original),
            len(restructured),
        )
        return {
            "content": restructured,
            "model": model or settings.llm_model,
            "numbers_changed": numbers_moved,
            "number_diff": number_diff,
        }


def get_doc_restructure_service() -> DocRestructureService:
    return DocRestructureService()
