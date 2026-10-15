import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Obsidian 的嵌入语法：`![[文件名]]`，通常是图片。
//
// 智语没有图片存储能力，所以这些嵌入指向的文件根本不在。原来的处理里
// 它会被下面的 wikiLinksToMarkdown 当成页面链接，渲染成指向
// 「Pasted image 20260421163755.png」这种标题的按钮，点了只会报
// 「链接目标尚不存在」——既不是图，也不像个链接。
//
// 这里把它先摘出来、换成一段明确的文字说明。顺序很重要：必须先处理
// 带 `!` 的嵌入，否则下面那条正则会把 `!` 后面那截吃掉。
const EMBED_PATTERN = /!\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]/g

function embedsToPlaceholders(content) {
  return content.replace(
    EMBED_PATTERN,
    (_, target) => `[图片未导入：${target.trim()}](#embed-missing)`,
  )
}

function wikiLinksToMarkdown(content) {
  return content.replace(
    /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/g,
    (_, target, label) => `[${label || target}](#wiki=${encodeURIComponent(target.trim())})`,
  )
}

function plainMath(value) {
  return value
    .replace(/\\text\{([^{}]*)\}/g, '$1')
    .replace(/\\operatorname\{([^{}]*)\}/g, '$1')
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)')
    .replace(/\\sum/g, 'Σ')
    .replace(/\\cdot/g, '·')
    .replace(/\\times/g, '×')
    .replace(/\\le/g, '≤')
    .replace(/\\ge/g, '≥')
    .replace(/\\rightarrow/g, '→')
    .replace(/\\left|\\right/g, '')
    .replace(/[{}]/g, '')
    .replace(/\\([a-zA-Z]+)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()
}

function normalizeMath(content) {
  let normalized = content
  normalized = normalized.replace(/\$\$([\s\S]*?)\$\$/g, (_, formula) => '\n> 公式：' + plainMath(formula) + '\n')
  normalized = normalized.replace(/\\\[([\s\S]*?)\\\]/g, (_, formula) => '\n> 公式：' + plainMath(formula) + '\n')
  normalized = normalized.replace(/\\\(([^\n]*?)\\\)/g, (_, formula) => String.fromCharCode(96) + plainMath(formula) + String.fromCharCode(96))
  normalized = normalized.replace(
    /^\s*\[\s*((?:[^\n]*\\(?:text|operatorname|sum|frac|cdot|times|le|ge|rightarrow)[^\n]*))\s*\]\s*$/gm,
    (_, formula) => '> 公式：' + plainMath(formula),
  )
  return normalized
}

function stripCitationAppendix(content) {
  const marker = /(?:^|\n)\s*(?:#{1,6}\s*)?引用来源\s*(?:\n|$)/m
  const match = marker.exec(content)
  return match ? content.slice(0, match.index).trimEnd() : content
}

export default function MarkdownView({ content, onWikiLink, stripCitationAppendix: removeCitationAppendix = false }) {
  const source = removeCitationAppendix ? stripCitationAppendix(content || '') : content || ''
  const normalized = normalizeMath(source)
  const rootRef = useRef(null)

  // 给渲染出来的标题编号，供右侧「页面目录」跳转。
  //
  // 在 DOM 上做而不是在 markdown 源码上做：源码那边要自己解析 `{#id}` 语法，
  // 而 normalizeMath 已经改过一遍文本，两边编号容易对不上。这里直接在
  // 最终 DOM 里按顺序取，所见即所得——页面目录也是照着 h1~h4 列的。
  //
  // 编号用「第几个标题」而不是行号：normalizeMath 会把公式展开成多行，
  // 行号与 WikiWorkspace.headingsFrom 那边算出来的对不上；而它不动标题，
  // 所以标题的数量和顺序两侧必然一致。
  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    root.querySelectorAll('h1, h2, h3, h4').forEach((node, index) => {
      node.id = `heading-${index + 1}`
    })
  }, [normalized])

  return (
    <div className="markdown" ref={rootRef}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children, ...props }) {
            if (href === '#embed-missing') {
              return <span className="embed-missing">{children}</span>
            }
            if (href?.startsWith('#wiki=')) {
              const title = decodeURIComponent(href.slice(6))
              return (
                <button className="wiki-link" type="button" onClick={() => onWikiLink?.(title)}>
                  {children}
                </button>
              )
            }
            return <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>
          },
        }}
      >
        {wikiLinksToMarkdown(embedsToPlaceholders(normalized))}
      </ReactMarkdown>
    </div>
  )
}
