import { useMemo, useState } from 'react'
import {
  ArrowUp,
  BookOpenCheck,
  Check,
  ExternalLink,
  Globe2,
  Headphones,
  LoaderCircle,
  MessageSquareText,
  RotateCcw,
  Save,
  Sparkles,
  X,
} from 'lucide-react'
import { api } from '../api'
import MarkdownView from '../components/MarkdownView'

function getSessionId() {
  const key = 'zhiyu_react_session_id'
  let value = localStorage.getItem(key)
  if (!value) {
    value = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    localStorage.setItem(key, value)
  }
  return value
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0))
  const minutes = Math.floor(total / 60)
  const remainder = total % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

export default function AskWorkspace({ notify }) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(getSessionId)

  const sessionLabel = useMemo(() => sessionId.replace('session_', '').slice(0, 10), [sessionId])

  async function sendQuery(event) {
    event?.preventDefault()
    const text = query.trim()
    if (!text || loading) return
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: text }
    setMessages((previous) => [...previous, userMessage])
    setQuery('')
    setLoading(true)
    try {
      const data = await api('/agent/chat/', {
        method: 'POST',
        body: JSON.stringify({ query: text, session_id: sessionId }),
      })
      setMessages((previous) => [...previous, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.response,
        sources: data.sources || [],
        evidenceStatus: data.evidence_status,
        evidenceScore: data.evidence_score,
        evidenceSourceCount: data.evidence_source_count,
        evidenceReason: data.evidence_reason,
        externalResearchAvailable: data.external_research_available,
        originalQuery: text,
        confirmationRequired: data.confirmation_required,
        pendingActionId: data.pending_action_id,
        preview: data.action_preview || [],
        status: data.confirmation_required ? 'pending' : 'complete',
      }])
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setLoading(false)
    }
  }

  async function runExternalResearch(messageId, messageQuery) {
    setMessages((previous) => previous.map((message) => (
      message.id === messageId ? { ...message, researchStatus: 'loading' } : message
    )))
    try {
      const data = await api('/agent/research/', {
        method: 'POST',
        body: JSON.stringify({ query: messageQuery, session_id: sessionId }),
      })
      setMessages((previous) => previous.map((message) => (
        message.id === messageId
          ? { ...message, researchStatus: 'complete', externalResearch: data }
          : message
      )))
    } catch (error) {
      setMessages((previous) => previous.map((message) => (
        message.id === messageId ? { ...message, researchStatus: 'failed' } : message
      )))
      notify(error.message, 'error')
    }
  }

  async function prepareResearchSave(messageId, runId) {
    setMessages((previous) => previous.map((message) => (
      message.id === messageId ? { ...message, researchSaveStatus: 'loading' } : message
    )))
    try {
      const data = await api(`/agent/research/${runId}/prepare-save`, {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      })
      setMessages((previous) => previous.map((message) => {
        if (message.id !== messageId) return message
        return {
          ...message,
          researchSaveStatus: 'pending',
          confirmationRequired: data.confirmation_required,
          pendingActionId: data.pending_action_id,
          preview: data.action_preview || [],
          status: 'pending',
        }
      }))
    } catch (error) {
      setMessages((previous) => previous.map((message) => (
        message.id === messageId ? { ...message, researchSaveStatus: 'failed' } : message
      )))
      notify(error.message, 'error')
    }
  }

  async function resolveAction(messageId, actionId, action) {
    try {
      const data = await api(`/agent/actions/${actionId}/${action}`, {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      })
      setMessages((previous) => previous.map((message) => {
        if (message.id !== messageId) return message
        return {
          ...message,
          status: action === 'confirm' ? 'complete' : 'cancelled',
          content: action === 'confirm' ? data.response : '知识变更已取消。',
          confirmationRequired: false,
          researchSaveStatus: action === 'confirm' ? 'saved' : 'cancelled',
        }
      }))
      notify(action === 'confirm' ? '知识变更已执行' : '知识变更已取消', 'success')
    } catch (error) {
      notify(error.message, 'error')
    }
  }

  function resetSession() {
    localStorage.removeItem('zhiyu_react_session_id')
    setSessionId(getSessionId())
    setMessages([])
  }

  return (
    <div className="ask-workspace">
      <header className="workspace-titlebar">
        <div><span className="eyebrow">可信问答</span><h1>基于你的 Wiki 寻找答案</h1></div>
        <button className="secondary-button" type="button" onClick={resetSession}><RotateCcw size={15} /> 新会话 <span className="session-code">#{sessionLabel}</span></button>
      </header>

      <div className="conversation">
        {messages.length === 0 && (
          <div className="ask-empty">
            <div className="ask-mark"><Sparkles size={24} /></div>
            <h2>从已有知识开始</h2>
            <div className="suggestion-grid">
              {['总结最近的 RAG 笔记', '比较两篇页面中的不同观点', '找出尚未解决的课堂疑问'].map((item) => (
                <button key={item} type="button" onClick={() => setQuery(item)}>{item}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
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
                      onClick={() => runExternalResearch(message.id, message.originalQuery)}
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
                      onClick={() => prepareResearchSave(message.id, message.externalResearch.run_id)}
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
                    <button className="primary-button" type="button" onClick={() => resolveAction(message.id, message.pendingActionId, 'confirm')}><Check size={16} /> 确认执行</button>
                    <button className="secondary-button" type="button" onClick={() => resolveAction(message.id, message.pendingActionId, 'cancel')}><X size={16} /> 取消</button>
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
            </div>
          </article>
        ))}
        {loading && <div className="answer-loading"><LoaderCircle size={18} className="spin" /> 正在检索并检查证据</div>}
      </div>

      <form className="ask-composer" onSubmit={sendQuery}>
        <textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            sendQuery()
          }
        }} placeholder="询问知识、比较页面，或交代一个知识整理任务…" rows="3" />
        <button className="send-button" type="submit" disabled={!query.trim() || loading} title="发送"><ArrowUp size={19} /></button>
      </form>
    </div>
  )
}
