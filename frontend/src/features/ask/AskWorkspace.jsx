import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowUp,
  History,
  LoaderCircle,
  MessageSquareText,
  RotateCcw,
  Search,
  Server,
  Sparkles,
  Square,
  X,
} from 'lucide-react'
import { api } from '../../shared/api/client'
import Modal from '../../shared/components/Modal'
import AgentMessage from './components/AgentMessage'
import McpStatusPanel from './components/McpStatusPanel'
import useAgentRun from './hooks/useAgentRun'

export default function AskWorkspace({ notify }) {
  const [mcpOpen, setMcpOpen] = useState(false)
  const [mcpStatus, setMcpStatus] = useState(null)
  const [mcpLoading, setMcpLoading] = useState(false)
  const [sessionQuery, setSessionQuery] = useState('')
  const [sessions, setSessions] = useState([])
  const [sessionsLoading, setSessionsLoading] = useState(true)
  const [sessionOpening, setSessionOpening] = useState(false)
  const [feedbackMessage, setFeedbackMessage] = useState(null)
  const [feedbackCategory, setFeedbackCategory] = useState('knowledge_missing')
  const [feedbackNote, setFeedbackNote] = useState('')
  const [feedbackPageId, setFeedbackPageId] = useState('')
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false)
  const sessionRequestSequence = useRef(0)
  const {
    activeRunId,
    cancelAnswerFeedback,
    confirmAnswerFeedback,
    historyVersion,
    liveStage,
    loadSession,
    loading,
    messages,
    prepareResearchSave,
    query,
    resetSession,
    reportAnswerFeedback,
    resolveAction,
    runExternalResearch,
    retryAnswerFeedback,
    sendQuery,
    sessionId,
    sessionLabel,
    setQuery,
    stopRun,
  } = useAgentRun(notify)

  const feedbackPages = useMemo(() => {
    const pages = new Map()
    for (const source of feedbackMessage?.sources || []) {
      if (source.page_id && !pages.has(source.page_id)) {
        pages.set(source.page_id, { id: source.page_id, title: source.title || source.page_id })
      }
    }
    return [...pages.values()]
  }, [feedbackMessage])

  const loadSessions = useCallback(async () => {
    const requestSequence = sessionRequestSequence.current + 1
    sessionRequestSequence.current = requestSequence
    setSessionsLoading(true)
    try {
      const params = new URLSearchParams({ limit: '100' })
      if (sessionQuery.trim()) params.set('query', sessionQuery.trim())
      const data = await api(`/agent/sessions/?${params}`)
      if (requestSequence !== sessionRequestSequence.current) return
      setSessions(data.sessions || [])
    } catch (error) {
      if (requestSequence !== sessionRequestSequence.current) return
      notify(error.message, 'error')
    } finally {
      if (requestSequence === sessionRequestSequence.current) setSessionsLoading(false)
    }
  }, [notify, sessionQuery])

  useEffect(() => {
    const timer = window.setTimeout(loadSessions, 180)
    return () => window.clearTimeout(timer)
  }, [historyVersion, loadSessions])

  async function openSession(session) {
    if (sessionOpening) return
    setSessionOpening(true)
    try {
      await loadSession(session.session_id)
      if (session.matched_message_id) {
        window.requestAnimationFrame(() => {
          document.getElementById(`message-history-${session.matched_message_id}`)?.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
          })
        })
      }
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setSessionOpening(false)
    }
  }

  async function loadMcpStatus(checkHealth = false) {
    setMcpLoading(true)
    try {
      const data = await api(checkHealth ? '/agent/mcp/health' : '/agent/mcp/status')
      setMcpStatus(data)
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setMcpLoading(false)
    }
  }

  function toggleMcpPanel() {
    const nextOpen = !mcpOpen
    setMcpOpen(nextOpen)
    if (nextOpen && !mcpStatus) loadMcpStatus(false)
  }

  function openFeedback(message) {
    setFeedbackMessage(message)
    setFeedbackCategory('knowledge_missing')
    setFeedbackNote('')
    setFeedbackPageId('')
  }

  function chooseFeedbackCategory(category) {
    setFeedbackCategory(category)
    if (category === 'content_outdated' && feedbackPages.length === 1) {
      setFeedbackPageId(feedbackPages[0].id)
    } else if (category !== 'content_outdated') {
      setFeedbackPageId('')
    }
  }

  async function submitFeedback() {
    if (!feedbackMessage?.requestId) return
    setFeedbackSubmitting(true)
    try {
      await reportAnswerFeedback(feedbackMessage.id, feedbackMessage.requestId, {
        category: feedbackCategory,
        user_note: feedbackNote.trim() || null,
        target_page_id: feedbackPageId || null,
      })
      setFeedbackMessage(null)
    } finally {
      setFeedbackSubmitting(false)
    }
  }

  return (
    <div className="ask-workspace">
      <div className="ask-layout">
        <aside className="conversation-browser">
          <div className="conversation-browser-heading">
            <div><span className="eyebrow">历史对话</span><strong>{sessions.length} 个会话</strong></div>
            <History size={17} />
          </div>
          <label className="browser-search conversation-search">
            <Search size={15} />
            <input
              value={sessionQuery}
              onChange={(event) => setSessionQuery(event.target.value)}
              placeholder="搜索对话标题和正文"
              aria-label="搜索历史对话"
            />
            {sessionQuery && (
              <button type="button" title="清空搜索" aria-label="清空对话搜索" onClick={() => setSessionQuery('')}>
                <X size={14} />
              </button>
            )}
          </label>
          <div className="session-list">
            {sessionsLoading && <div className="inline-loading"><LoaderCircle size={16} className="spin" /> 加载中</div>}
            {!sessionsLoading && sessions.length === 0 && <div className="empty-inline">没有匹配的对话</div>}
            {sessions.map((session) => (
              <button
                key={session.session_id}
                className={`${session.session_id === sessionId ? 'session-row active' : 'session-row'}${session.match_snippet ? ' with-snippet' : ''}`}
                type="button"
                disabled={loading || sessionOpening}
                onClick={() => openSession(session)}
              >
                <MessageSquareText size={15} />
                <span>
                  <strong>{session.title || '新对话'}</strong>
                  <small>{session.message_count || 0} 条消息 · {formatDate(session.updated_at)}</small>
                  {session.match_snippet && <em>{session.match_snippet}</em>}
                </span>
              </button>
            ))}
          </div>
        </aside>

        <main className="ask-main">
          <header className="workspace-titlebar">
            <div><span className="eyebrow">可信问答</span><h1>基于你的 Wiki 寻找答案</h1></div>
            <div className="ask-title-actions">
              <button className="secondary-button" type="button" onClick={toggleMcpPanel} aria-expanded={mcpOpen}><Server size={15} /> MCP</button>
              <button className="secondary-button" type="button" onClick={resetSession} disabled={loading}><RotateCcw size={15} /> 新会话 <span className="session-code">#{sessionLabel}</span></button>
            </div>
          </header>

          {mcpOpen && <McpStatusPanel loading={mcpLoading} status={mcpStatus} onRefresh={() => loadMcpStatus(true)} />}

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
              <AgentMessage
                key={message.id}
                message={message}
                onPrepareSave={prepareResearchSave}
                onResearch={runExternalResearch}
                onResolveAction={resolveAction}
                onCancelFeedback={cancelAnswerFeedback}
                onConfirmFeedback={confirmAnswerFeedback}
                onFlagResponse={openFeedback}
                onRetryFeedback={retryAnswerFeedback}
              />
            ))}
            {loading && <div className="answer-loading"><LoaderCircle size={18} className="spin" /> {liveStage}</div>}
          </div>

          <form className="ask-composer" onSubmit={sendQuery}>
            <textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                sendQuery()
              }
            }} placeholder="询问知识、比较页面，或交代一个知识整理任务…" rows="3" />
            {loading ? (
              <button className="send-button stop-button" type="button" disabled={!activeRunId} onClick={stopRun} title="停止生成" aria-label="停止生成"><Square size={17} /></button>
            ) : (
              <button className="send-button" type="submit" disabled={!query.trim()} title="发送" aria-label="发送"><ArrowUp size={19} /></button>
            )}
          </form>
        </main>
      </div>

      {feedbackMessage && (
        <Modal
          title="标记回答问题"
          onClose={() => !feedbackSubmitting && setFeedbackMessage(null)}
          actions={(
            <>
              <button className="secondary-button" type="button" disabled={feedbackSubmitting} onClick={() => setFeedbackMessage(null)}>取消</button>
              <button
                className="primary-button"
                type="button"
                disabled={feedbackSubmitting || (feedbackCategory === 'content_outdated' && !feedbackPageId)}
                onClick={submitFeedback}
              >
                {feedbackSubmitting ? <><LoaderCircle size={15} className="spin" /> 正在生成草稿</> : '保存并生成草稿'}
              </button>
            </>
          )}
        >
          <div className="feedback-form">
            <div className="field-label">问题类型</div>
            <div className="feedback-category-grid" role="group" aria-label="回答问题类型">
              {[
                ['knowledge_missing', '知识缺失'],
                ['content_outdated', '内容过期'],
                ['citation_error', '引用错误'],
                ['answer_irrelevant', '回答不相关'],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={feedbackCategory === value}
                  className={feedbackCategory === value ? 'active' : ''}
                  onClick={() => chooseFeedbackCategory(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            {feedbackCategory === 'content_outdated' && (
              <label>
                需要修订的页面
                <select value={feedbackPageId} onChange={(event) => setFeedbackPageId(event.target.value)}>
                  <option value="">请选择引用页面</option>
                  {feedbackPages.map((page) => <option key={page.id} value={page.id}>{page.title}</option>)}
                </select>
              </label>
            )}
            <label>
              补充说明
              <textarea value={feedbackNote} onChange={(event) => setFeedbackNote(event.target.value)} maxLength="1000" rows="4" />
            </label>
          </div>
        </Modal>
      )}
    </div>
  )
}

function formatDate(value) {
  if (!value) return '刚刚'
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit' }).format(new Date(value))
}
