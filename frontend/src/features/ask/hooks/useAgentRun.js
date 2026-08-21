import { useMemo, useRef, useState } from 'react'
import { api, streamSse } from '../../../shared/api/client'
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

export default function useAgentRun(notify, { allowExternalResearch = false } = {}) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(getSessionId)
  const [activeRunId, setActiveRunId] = useState(null)
  const [liveStage, setLiveStage] = useState('准备运行')
  const [historyVersion, setHistoryVersion] = useState(0)
  const sessionIdRef = useRef(sessionId)

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
              body: JSON.stringify({
                query: text,
                session_id: sessionId,
                allow_external_research: allowExternalResearch,
              }),
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
      setHistoryVersion((version) => version + 1)
    }
  }

  async function loadSession(nextSessionId) {
    if (loading || !nextSessionId) return null
    const data = await api(`/agent/sessions/${encodeURIComponent(nextSessionId)}/messages`)
    let previousUserQuery = ''
    const restored = (data.messages || []).map((message) => {
      const metadata = message.metadata || {}
      if (message.role === 'user') previousUserQuery = message.content
      return {
        id: `history-${message.id}`,
        role: message.role,
        content: message.content,
        originalQuery: message.role === 'assistant' ? previousUserQuery : undefined,
        sources: metadata.sources || [],
        evidenceStatus: metadata.evidence_status,
        evidenceScore: metadata.evidence_score,
        evidenceSourceCount: metadata.evidence_source_count,
        evidenceReason: metadata.evidence_reason,
        externalResearchAvailable: metadata.external_research_available,
        confirmationRequired: metadata.confirmation_required,
        pendingActionId: metadata.pending_action_id,
        preview: metadata.action_preview || [],
        requestId: metadata.request_id,
        feedbackId: metadata.feedback_id,
        status: metadata.confirmation_required ? 'pending' : 'complete',
        createdAt: message.created_at,
      }
    })
    await Promise.all(restored.map(async (message) => {
      if (!message.feedbackId) return
      try {
        message.feedback = await api(
          `/agent/feedback/${encodeURIComponent(message.feedbackId)}?session_id=${encodeURIComponent(nextSessionId)}`,
        )
      } catch {
        message.feedback = null
      }
    }))
    localStorage.setItem('zhiyu_react_session_id', nextSessionId)
    sessionIdRef.current = nextSessionId
    setSessionId(nextSessionId)
    setMessages(restored)
    setQuery('')
    return data
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

  function updateFeedback(messageId, feedback) {
    setMessages((previous) => previous.map((message) => (
      message.id === messageId ? { ...message, feedback } : message
    )))
  }

  async function pollAnswerFeedback(messageId, feedbackId, feedbackSessionId) {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      await wait(1000)
      try {
        const data = await api(
          `/agent/feedback/${encodeURIComponent(feedbackId)}?session_id=${encodeURIComponent(feedbackSessionId)}`,
        )
        if (sessionIdRef.current !== feedbackSessionId) return
        updateFeedback(messageId, data)
        if (['resolved', 'retest_failed', 'index_failed'].includes(data.status)) {
          if (data.status === 'resolved') {
            notify('知识已修订，原问题复测完成', 'success')
            setHistoryVersion((version) => version + 1)
          }
          return
        }
      } catch (error) {
        notify(error.message, 'error')
        return
      }
    }
  }

  async function reportAnswerFeedback(messageId, requestId, values) {
    let feedbackId = null
    updateFeedback(messageId, { status: 'reported' })
    try {
      const created = await api('/agent/feedback/', {
        method: 'POST',
        body: JSON.stringify({
          request_id: requestId,
          session_id: sessionId,
          ...values,
        }),
      })
      feedbackId = created.id
      updateFeedback(messageId, { ...created, status: 'researching' })
      const prepared = await api(`/agent/feedback/${encodeURIComponent(created.id)}/prepare`, {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      })
      updateFeedback(messageId, prepared)
      notify('纠错草稿已生成，请确认后写入', 'success')
      return prepared
    } catch (error) {
      if (feedbackId) {
        try {
          const failed = await api(
            `/agent/feedback/${encodeURIComponent(feedbackId)}?session_id=${encodeURIComponent(sessionId)}`,
          )
          updateFeedback(messageId, failed)
        } catch {
          updateFeedback(messageId, { id: feedbackId, status: 'draft_failed', error: error.message })
        }
      } else {
        updateFeedback(messageId, null)
      }
      notify(error.message, 'error')
      throw error
    }
  }

  async function confirmAnswerFeedback(messageId, feedbackId) {
    const feedbackSessionId = sessionId
    updateFeedback(messageId, { id: feedbackId, status: 'writing' })
    try {
      const data = await api(`/agent/feedback/${encodeURIComponent(feedbackId)}/confirm`, {
        method: 'POST',
        body: JSON.stringify({ session_id: feedbackSessionId }),
      })
      if (sessionIdRef.current !== feedbackSessionId) return
      updateFeedback(messageId, data)
      if (data.status === 'retesting') pollAnswerFeedback(messageId, feedbackId, feedbackSessionId)
      else if (data.status === 'index_failed') notify(data.error || '索引失败', 'error')
    } catch (error) {
      notify(error.message, 'error')
      const data = await api(
        `/agent/feedback/${encodeURIComponent(feedbackId)}?session_id=${encodeURIComponent(sessionId)}`,
      ).catch(() => ({ id: feedbackId, status: 'write_failed', error: error.message }))
      updateFeedback(messageId, data)
    }
  }

  async function retryAnswerFeedback(messageId, feedbackId) {
    const feedbackSessionId = sessionId
    try {
      const data = await api(`/agent/feedback/${encodeURIComponent(feedbackId)}/retry`, {
        method: 'POST',
        body: JSON.stringify({ session_id: feedbackSessionId }),
      })
      if (sessionIdRef.current !== feedbackSessionId) return
      updateFeedback(messageId, data)
      if (data.status === 'retesting') pollAnswerFeedback(messageId, feedbackId, feedbackSessionId)
    } catch (error) {
      notify(error.message, 'error')
    }
  }

  async function cancelAnswerFeedback(messageId, feedbackId) {
    try {
      const data = await api(`/agent/feedback/${encodeURIComponent(feedbackId)}/cancel`, {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      })
      updateFeedback(messageId, data)
    } catch (error) {
      notify(error.message, 'error')
    }
  }

  function resetSession() {
    localStorage.removeItem('zhiyu_react_session_id')
    const nextSessionId = getSessionId()
    sessionIdRef.current = nextSessionId
    setSessionId(nextSessionId)
    setMessages([])
    setHistoryVersion((version) => version + 1)
  }

  return {
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
  }
}
