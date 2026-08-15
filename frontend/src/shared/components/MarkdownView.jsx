import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function wikiLinksToMarkdown(content) {
  return content.replace(
    /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/g,
    (_, target, label) => `[${label || target}](#wiki=${encodeURIComponent(target.trim())})`,
  )
}

export default function MarkdownView({ content, onWikiLink }) {
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
        {wikiLinksToMarkdown(content || '')}
      </ReactMarkdown>
    </div>
  )
}
