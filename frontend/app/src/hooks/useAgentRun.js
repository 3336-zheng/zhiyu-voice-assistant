import { useMemo, useState } from 'react'
import { api, streamSse } from '../api'
import { stageLabel } from '../utils/agentPresentation'

function getSessionId() {
  const key = 'zhiyu_react_session_id'
  let value = localStorage.getItem(key)
  if (!value) {
    value = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    localStorage.setItem(key, value)
  }
  return value
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

export default function useAgentRun(notify) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(getSessionId)
  const [activeRunId, setActiveRunId] = useState(null)
  const [liveStage, setLiveStage] = useState('准备运行')

  const sessionLabel = useMemo(
    () => sessionId.replace('session_', '').slice(0, 10),
    [sessionId],
  )

  async function sendQuery(event) {
    event?.preventDefault()
    const text = query.trim()
    if (!text || loading) return
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: text }
    const assistantId = crypto.randomUUID()
    setMessages((previous) => [...previous, userMessage, {
      id: assistantId,
      role: 'assistant',
      content: '',
      originalQuery: text,
      status: 'running',
      sources: [],
      timeline: [],
    }])
    setQuery('')
    setLoading(true)
    setLiveStage('准备运行')
    let terminalReceived = false
    let currentRunId = null
    let lastSequence = 0
    let reconnectCount = 0
    let firstConnection = true

    const handleAgentEvent = (eventPayload, metadata) => {
      const eventType = eventPayload.type
      const data = eventPayload.data || {}
      const sequence = Number(eventPayload.sequence || metadata.eventId || 0)
      lastSequence = Math.max(lastSequence, sequence)
      if (eventPayload.run_id) {
        currentRunId = eventPayload.run_id
        setActiveRunId(eventPayload.run_id)
      }
      if (eventType === 'stage_started') setLiveStage(stageLabel(data.stage || '运行中'))
      if (eventType === 'tool_started') setLiveStage(`调用 ${data.tool_name || '工具'}`)
      if (['run_completed', 'run_error', 'run_cancelled'].includes(eventType)) {
        terminalReceived = true
      }

      setMessages((previous) => previous.map((message) => {
        if (message.id !== assistantId) return message
        if (eventType === 'token') {
          return { ...message, content: `${message.content}${data.content || ''}` }
        }
        if (eventType === 'run_error') {
          return {
            ...message,
            content: message.content || data.message || '运行失败。',
            status: 'failed',
          }
        }
        if (eventType === 'run_cancelled') {
          return {
            ...message,
            content: message.content || '本次生成已停止。',
            status: 'cancelled',
          }
        }
        if (eventType !== 'run_completed') return message

        const response = data.response || {}
        return {
          ...message,
          content: response.response || message.content,
          sources: response.sources || [],
          evidenceStatus: response.evidence_status,
          evidenceScore: response.evidence_score,
          evidenceSourceCount: response.evidence_source_count,
          evidenceReason: response.evidence_reason,
          externalResearchAvailable: response.external_research_available,
          confirmationRequired: response.confirmation_required,
          pendingActionId: response.pending_action_id,
          preview: response.action_preview || [],
          status: response.confirmation_required ? 'pending' : 'complete',
          requestId: response.request_id,
          executionTimeMs: response.execution_time_ms,
          timeline: response.timeline || [],
          retrievalStats: response.retrieval_stats,
          modelUsage: response.model_usage,
        }
      }))
    }

    try {
      while (!terminalReceived) {
        const path = firstConnection
          ? '/agent/chat/stream/'
          : `/agent/runs/${encodeURIComponent(currentRunId)}/events?session_id=${encodeURIComponent(sessionId)}&after_sequence=${lastSequence}`
        const options = firstConnection
          ? {
              method: 'POST',
              body: JSON.stringify({ query: text, session_id: sessionId }),
            }
          : { headers: { 'Last-Event-ID': String(lastSequence) } }
        let streamError = null
        try {
          await streamSse(path, options, handleAgentEvent, (response) => {
            const headerRunId = response.headers.get('X-Agent-Run-ID')
            if (headerRunId) {
              currentRunId = headerRunId
              setActiveRunId(headerRunId)
            }
          })
        } catch (error) {
          streamError = error
        }

        if (terminalReceived) break
        if (!currentRunId || reconnectCount >= 3) {
          throw streamError || new Error('流式连接提前结束，请重新发送问题。')
        }
        reconnectCount += 1
        firstConnection = false
        setLiveStage('正在恢复连接')
        await wait(300 * reconnectCount)
      }
    } catch (error) {
      setMessages((previous) => previous.map((message) => (
        message.id === assistantId && !terminalReceived
          ? { ...message, content: message.content || error.message, status: 'failed' }
          : message
      )))
      notify(error.message, 'error')
    } finally {
      setLoading(false)
      setActiveRunId(null)
      setLiveStage('准备运行')
    }
  }

  async function stopRun() {
    if (!activeRunId) return
    setLiveStage('正在停止')
    try {
      await api(`/agent/runs/${activeRunId}/cancel`, {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      })
    } catch (error) {
      notify(error.message, 'error')
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

  return {
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
  }
}
