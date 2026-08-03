import { useMemo, useState } from 'react'
import {
  ArrowUp,
  BookOpenCheck,
  Check,
  LoaderCircle,
  MessageSquareText,
  RotateCcw,
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
                </div>
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
                    <a key={source.chunk_id || source.id} href={source.source_url || '#'} target="_blank" rel="noreferrer">
                      <strong>{source.title}</strong>
                      <small>{source.section_path || source.section_title || '页面正文'} · v{source.page_revision || '-'}</small>
                    </a>
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
