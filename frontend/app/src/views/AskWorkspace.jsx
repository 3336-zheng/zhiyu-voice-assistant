import { useState } from 'react'
import {
  ArrowUp,
  LoaderCircle,
  RotateCcw,
  Server,
  Sparkles,
  Square,
} from 'lucide-react'
import { api } from '../api'
import AgentMessage from '../components/AgentMessage'
import McpStatusPanel from '../components/McpStatusPanel'
import useAgentRun from '../hooks/useAgentRun'

export default function AskWorkspace({ notify }) {
  const [mcpOpen, setMcpOpen] = useState(false)
  const [mcpStatus, setMcpStatus] = useState(null)
  const [mcpLoading, setMcpLoading] = useState(false)
  const {
    activeRunId,
    liveStage,
    loading,
    messages,
    prepareResearchSave,
    query,
    resetSession,
    resolveAction,
    runExternalResearch,
    sendQuery,
    sessionLabel,
    setQuery,
    stopRun,
  } = useAgentRun(notify)

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

  return (
    <div className="ask-workspace">
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
    </div>
  )
}
