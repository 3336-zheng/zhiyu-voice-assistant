import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

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

  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children, ...props }) {
            if (href?.startsWith('#wiki=')) {
              const title = decodeURIComponent(href.slice(6))
              return (
                <button className="wiki-link" type="button" onClick={() => onWikiLink?.(title)}>
                  {children}
                </button>
              )
            }
            return <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>
          }
        }}
      >
        {wikiLinksToMarkdown(normalized)}
      </ReactMarkdown>
    </div>
  )
}
