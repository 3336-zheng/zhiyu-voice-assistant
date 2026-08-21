import {
  Activity,
  BookOpenCheck,
  Check,
  ChevronDown,
  ExternalLink,
  Globe2,
  Headphones,
  LoaderCircle,
  MessageSquareText,
  Save,
  ThumbsDown,
  X,
} from 'lucide-react'
import AnswerFeedbackPanel from './AnswerFeedbackPanel'
import MarkdownView from '../../../shared/components/MarkdownView'
import { formatCost, formatTime, stageLabel } from '../utils/agentPresentation'

function citationLabel(value) {
  return String(value || '')
    .replace(/\\([\[\]\(\)])/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\s*>\s*/g, ' · ')
    .replace(/\s+/g, ' ')
    .trim()
}

function orderedSources(sources) {
  const seen = new Set()
  return [...(sources || [])]
    .sort((left, right) => {
      const leftScore = Number(left.rerank_score ?? left.score ?? 0)
      const rightScore = Number(right.rerank_score ?? right.score ?? 0)
      return rightScore - leftScore
    })
    .filter((source) => {
      const key = source.chunk_id || source.id || (
        String(source.page_id || '') + ':' + String(source.section_path || source.section_title || '')
      )
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
}

function sourcePageId(source) {
  if (source?.page_id) return String(source.page_id)
  if (source?.pageId) return String(source.pageId)
  const sourceUrlId = source?.source_url?.match(/\/api\/pages\/([^/?#]+)/)?.[1]
  if (sourceUrlId) return sourceUrlId
  const chunkId = String(source?.chunk_id || source?.id || '')
  return chunkId.match(/^page:([0-9a-f-]{36}):/)?.[1] || null
}

function sourceTitle(source) {
  const candidates = [
    source?.title,
    source?.page_title,
    source?.filename?.replace(/\.md$/i, ''),
    source?.source_uri?.split('/').pop(),
    source?.section_path,
    source?.section_title,
  ]
  return candidates.map(citationLabel).find(Boolean) || '知识库页面'
}

export default function AgentMessage({
  message,
  onPrepareSave,
  onResearch,
  onResolveAction,
  onCancelFeedback,
  onConfirmFeedback,
  onFlagResponse,
  onRetryFeedback,
  onOpenSource,
}) {
  const citationSources = orderedSources(message.sources)

  return (
    <article id={`message-${message.id}`} className={`message ${message.role}`}>
      <div className="message-role">{message.role === 'user' ? '你' : <><MessageSquareText size={14} /> 智语</>}</div>
      <div className="message-body">
        {message.role === 'assistant' ? <MarkdownView content={message.content} stripCitationAppendix /> : message.content}
        {message.role === 'assistant' && message.evidenceStatus === 'insufficient' && (
          <div className="evidence-warning" role="status">
            <strong>证据不足，未生成推测性答案</strong>
            <span>{message.evidenceReason || '请补充 Wiki 页面或缩小查询范围。'}</span>
            {message.externalResearchAvailable && !message.externalResearch && (
              <button
                className="research-button"
                type="button"
                disabled={message.researchStatus === 'loading'}
                onClick={() => onResearch(message.id, message.originalQuery)}
              >
                {message.researchStatus === 'loading'
                  ? <><LoaderCircle size={15} className="spin" /> 正在查找</>
                  : <><Globe2 size={15} /> 查找外部资料</>}
              </button>
            )}
          </div>
        )}
        {message.externalResearch && (
          <section className="external-research" aria-label="外部研究结果">
            <div className="external-research-heading">
              <div><Globe2 size={16} /><strong>外部研究</strong></div>
              <button
                className="secondary-button"
                type="button"
                disabled={['loading', 'pending', 'saved'].includes(message.researchSaveStatus)}
                onClick={() => onPrepareSave(message.id, message.externalResearch.run_id)}
              >
                {message.researchSaveStatus === 'loading'
                  ? <LoaderCircle size={15} className="spin" />
                  : <Save size={15} />}
                {message.researchSaveStatus === 'saved' ? '已保存' : '保存为 Wiki'}
              </button>
            </div>
            <MarkdownView content={message.externalResearch.answer || ''} />
            <div className="external-source-list">
              {message.externalResearch.sources.map((source, index) => (
                <a key={source.id} href={source.url} target="_blank" rel="noreferrer">
                  <span>[{index + 1}] {source.title}</span>
                  <ExternalLink size={13} />
                </a>
              ))}
            </div>
          </section>
        )}
        {message.confirmationRequired && (
          <div className="action-preview">
            <div className="action-preview-title"><BookOpenCheck size={16} /> 等待确认的知识变更</div>
            {message.preview.map((step) => (
              <div className="action-step" key={step.step_id}>
                <strong>{step.description}</strong>
                <pre>{JSON.stringify(step.parameters, null, 2)}</pre>
              </div>
            ))}
            <div className="action-buttons">
              <button className="primary-button" type="button" onClick={() => onResolveAction(message.id, message.pendingActionId, 'confirm')}><Check size={16} /> 确认执行</button>
              <button className="secondary-button" type="button" onClick={() => onResolveAction(message.id, message.pendingActionId, 'cancel')}><X size={16} /> 取消</button>
            </div>
          </div>
        )}
        {citationSources.length > 0 && (
          <div className="citation-list">
            <span>引用来源</span>
            {citationSources.map((source) => {
              const pageId = sourcePageId(source)
              return (
                <div className="citation-item" key={source.chunk_id || source.id}>
                  <a
                    href={source.source_url || '#'}
                    target={pageId ? undefined : '_blank'}
                    rel={pageId ? undefined : 'noreferrer'}
                    onClick={(event) => {
                      if (pageId && onOpenSource) {
                        event.preventDefault()
                        onOpenSource(pageId)
                      }
                    }}
                  >
                    <strong>{sourceTitle(source)}</strong>
                    <small>{citationLabel(source.section_path || source.section_title) || '页面正文'}</small>
                  </a>
                  {source.audio_url && (
                    <a className="audio-citation" href={source.audio_url} target="_blank" rel="noreferrer">
                      <Headphones size={13} /> 原始录音 {formatTime(source.audio_start)}
                    </a>
                  )}
                </div>
              )
            })}
          </div>
        )}
        {message.role === 'assistant' && (message.timeline?.length > 0 || message.retrievalStats) && (
          <details className="run-details">
            <summary>
              <span><Activity size={14} /> 执行详情</span>
              <span>{message.executionTimeMs || 0} ms <ChevronDown size={14} /></span>
            </summary>
            <div className="run-detail-body">
              {message.retrievalStats && (
                <div className="retrieval-metrics">
                  <span><small>查询</small><strong>{message.retrievalStats.query_count || 0}</strong></span>
                  <span><small>召回</small><strong>{(message.retrievalStats.bm25_hits || 0) + (message.retrievalStats.embedding_hits || 0)}</strong></span>
                  <span><small>融合</small><strong>{message.retrievalStats.fused_candidates || 0}</strong></span>
                  <span><small>精排</small><strong>{message.retrievalStats.reranked_candidates || 0}</strong></span>
                  <span><small>上下文</small><strong>{message.retrievalStats.context_tokens || 0} tokens</strong></span>
                </div>
              )}
              <div className="execution-timeline">
                {message.timeline.filter((item) => item.stage !== 'agent.total').map((item, index) => (
                  <div key={`${item.stage}-${index}`}>
                    <i className={item.status === 'failed' ? 'failed' : ''} />
                    <span>{stageLabel(item.stage)}</span>
                    <strong>{Number(item.duration_ms || 0).toFixed(1)} ms</strong>
                  </div>
                ))}
              </div>
              <div className="run-footnote">
                <span>{message.modelUsage?.total_tokens || 0} tokens · {formatCost(message.modelUsage?.estimated_cost)}</span>
                {message.modelUsage?.fallback_used && <strong>已切换备用模型</strong>}
                {message.requestId && <code>{message.requestId}</code>}
              </div>
            </div>
          </details>
        )}
        {message.role === 'assistant' && message.requestId && message.status === 'complete' && !message.feedback && (
          <div className="message-feedback-action">
            <button type="button" onClick={() => onFlagResponse(message)}><ThumbsDown size={14} /> 回答有问题</button>
          </div>
        )}
        {message.feedback && (
          <AnswerFeedbackPanel
            feedback={message.feedback}
            onCancel={() => onCancelFeedback(message.id, message.feedback.id)}
            onConfirm={() => onConfirmFeedback(message.id, message.feedback.id)}
            onRetry={() => onRetryFeedback(message.id, message.feedback.id)}
          />
        )}
      </div>
    </article>
  )
}
