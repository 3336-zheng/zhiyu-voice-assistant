import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Clock3,
  LoaderCircle,
  RefreshCw,
  Search,
  ServerCog,
} from 'lucide-react'
import { api } from '../../shared/api/client'

function formatTime(timestamp) {
  if (!timestamp) return '—'
  return new Date(timestamp * 1000).toLocaleString('zh-CN', { hour12: false })
}

function formatDuration(value) {
  const duration = Number(value || 0)
  return duration >= 1000 ? `${(duration / 1000).toFixed(2)} s` : `${duration.toFixed(1)} ms`
}

function statusLabel(status) {
  const labels = {
    completed: '完成',
    failed: '失败',
    timed_out: '超时',
    cancelled: '已停止',
    running: '运行中',
    pending: '等待中',
  }
  return labels[status] || status || '未知'
}

export default function DiagnosticsWorkspace({ notify }) {
  const [requests, setRequests] = useState([])
  const [requestId, setRequestId] = useState('')
  const [trace, setTrace] = useState(null)
  const [loading, setLoading] = useState(false)
  const [listLoading, setListLoading] = useState(true)

  const loadRecent = useCallback(async () => {
    setListLoading(true)
    try {
      const data = await api('/api/observability/requests?limit=40')
      setRequests(data.requests || [])
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setListLoading(false)
    }
  }, [notify])

  useEffect(() => {
    loadRecent()
  }, [loadRecent])

  async function loadTrace(selectedId = requestId) {
    const normalized = selectedId.trim()
    if (!normalized) return
    setLoading(true)
    try {
      const data = await api(`/api/observability/requests/${encodeURIComponent(normalized)}`)
      setTrace(data)
      setRequestId(normalized)
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setLoading(false)
    }
  }

  const timeline = useMemo(() => trace?.timeline || [], [trace])
  const modelUsage = trace?.model_usage || {}

  return (
    <div className="diagnostics-workspace">
      <header className="workspace-titlebar">
        <div><span className="eyebrow">LOCAL OBSERVABILITY</span><h1>运行追踪</h1></div>
        <button className="secondary-button" type="button" onClick={loadRecent} disabled={listLoading}>
          {listLoading ? <LoaderCircle size={16} className="spin" /> : <RefreshCw size={16} />} 刷新
        </button>
      </header>

      <div className="diagnostics-layout">
        <aside className="trace-browser">
          <div className="trace-browser-heading"><Activity size={17} /><strong>近期请求</strong><span>{requests.length}</span></div>
          <div className="trace-list">
            {requests.map((item) => (
              <button
                key={item.request_id}
                className={trace?.request_id === item.request_id ? 'trace-row active' : 'trace-row'}
                type="button"
                onClick={() => loadTrace(item.request_id)}
              >
                <span className={`trace-state ${item.status}`}>{statusLabel(item.status)}</span>
                <strong>{item.method} {item.path}</strong>
                <small>{formatDuration(item.total_ms)} · {formatTime(item.completed_at)}</small>
                {item.error_code && <code>{item.error_code}</code>}
              </button>
            ))}
            {!listLoading && requests.length === 0 && <div className="trace-list-empty">暂无运行记录</div>}
          </div>
        </aside>

        <main className="trace-detail">
          <form className="trace-search" onSubmit={(event) => { event.preventDefault(); loadTrace() }}>
            <Search size={17} />
            <input value={requestId} onChange={(event) => setRequestId(event.target.value)} placeholder="输入 Request ID" aria-label="Request ID" />
            <button className="primary-button" type="submit" disabled={!requestId.trim() || loading}>
              {loading ? <LoaderCircle size={16} className="spin" /> : <Search size={16} />} 查询
            </button>
          </form>

          {!trace ? (
            <div className="trace-empty"><ServerCog size={30} /><strong>选择一条运行记录</strong></div>
          ) : (
            <div className="trace-content">
              <section className="trace-summary-band">
                <div><span>状态</span><strong className={`trace-status-text ${trace.status}`}>{statusLabel(trace.status)}</strong></div>
                <div><span>总耗时</span><strong>{formatDuration(trace.total_ms)}</strong></div>
                <div><span>请求</span><strong>{trace.method} {trace.path}</strong></div>
                <div><span>模型调用</span><strong>{modelUsage.call_count || 0} 次</strong></div>
              </section>

              <section className="trace-identity">
                <span>Request ID</span><code>{trace.request_id}</code><small>{formatTime(trace.completed_at)}</small>
              </section>

              {trace.error && (
                <section className="trace-error-band">
                  <AlertTriangle size={18} />
                  <div><strong>{trace.error.error_code || 'UNKNOWN_ERROR'}</strong><span>{trace.error.message || '请求执行失败'}</span></div>
                  <em>{trace.error.retryable ? '可重试' : '需检查'}</em>
                </section>
              )}

              <section className="trace-timeline-section">
                <div className="section-heading"><Clock3 size={17} /><h2>执行时间线</h2><span>{timeline.length} 个事件</span></div>
                <div className="trace-timeline">
                  {timeline.map((item, index) => (
                    <div className={`trace-stage ${item.status || 'completed'}`} key={`${item.stage}-${index}`}>
                      <i />
                      <div><strong>{item.stage}</strong>{item.error_code && <code>{item.error_code}</code>}</div>
                      <span>{item.duration_ms == null ? statusLabel(item.status) : formatDuration(item.duration_ms)}</span>
                    </div>
                  ))}
                  {timeline.length === 0 && <div className="trace-list-empty">该请求没有阶段事件</div>}
                </div>
              </section>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
