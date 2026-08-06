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
  X,
} from 'lucide-react'
import MarkdownView from './MarkdownView'
import { formatCost, formatTime, stageLabel } from '../utils/agentPresentation'

export default function AgentMessage({
  message,
  onPrepareSave,
  onResearch,
  onResolveAction,
}) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-role">{message.role === 'user' ? '你' : <><MessageSquareText size={14} /> 智语</>}</div>
      <div className="message-body">
        {message.role === 'assistant' ? <MarkdownView content={message.content} /> : message.content}
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
        {message.sources?.length > 0 && (
          <div className="citation-list">
            <span>引用来源</span>
            {message.sources.map((source) => (
              <div className="citation-item" key={source.chunk_id || source.id}>
                <a href={source.source_url || '#'} target="_blank" rel="noreferrer">
                  <strong>{source.title}</strong>
                  <small>{source.section_path || source.section_title || '页面正文'}</small>
                </a>
                {source.audio_url && (
                  <a className="audio-citation" href={source.audio_url} target="_blank" rel="noreferrer">
                    <Headphones size={13} /> 原始录音 {formatTime(source.audio_start)}
                  </a>
                )}
              </div>
            ))}
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
      </div>
    </article>
  )
}
