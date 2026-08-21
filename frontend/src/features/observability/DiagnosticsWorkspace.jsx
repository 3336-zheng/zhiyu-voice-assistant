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
    success: '完成',
    failed: '失败',
    timed_out: '超时',
    cancelled: '已停止',
    cancelling: '停止中',
    running: '运行中',
    pending: '等待中',
  }
  return labels[status] || status || '未知'
}

function evidenceLabel(status) {
  const labels = {
    sufficient: '证据充分',
    insufficient: '证据不足',
    not_applicable: '未评估',
  }
  return labels[status] || status || '未评估'
}

function scoreLabel(value) {
  if (value === null || value === undefined || value === '') return '—'
  const score = Number(value)
  if (!Number.isFinite(score)) return '—'
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`
}

function rerankScoreLabel(value) {
  if (value === null || value === undefined || value === '') return '—'
  const score = Number(value)
  return Number.isFinite(score) ? score.toFixed(3) : '—'
}

function tokenCount(value) {
  if (value === null || value === undefined || value === '') return '—'
  const count = Number(value)
  return Number.isFinite(count) ? count.toLocaleString('zh-CN') : '—'
}

function stageLabel(stage) {
  const normalizedStage = String(stage || '').split('#')[0]
  const labels = {
    'agent.plan': '规划',
    'agent.replan': '重新规划',
    'llm.primary': '主模型调用',
    'llm.fallback': '备用模型调用',
    'agent.query_rewrite': '查询改写',
    'agent.query_rewrite.hyde': 'HyDE 假设答案',
    'retrieval.embedding': '向量召回',
    'retrieval.bm25': 'BM25 召回',
    'retrieval.recall': '并行召回',
    'retrieval.fusion': 'RRF 融合',
    'retrieval.fetch_chunks': '读取证据',
    'retrieval.rerank': '在线精排',
    'agent.retrieve': '知识检索',
    'agent.crag_grade': 'CRAG 可信度评估',
    'agent.crag_refine': 'CRAG 证据精炼',
    'agent.evidence': '证据门禁',
    'agent.generation': '答案生成',
    'agent.generation.stream': '流式答案生成',
    'memory.summary': '记忆摘要压缩',
    'agent.total': '查询总耗时',
  }
  const suffix = String(stage || '').includes('#') ? ` #${String(stage).split('#')[1]}` : ''
  return `${labels[normalizedStage] || normalizedStage || '未知阶段'}${suffix}`
}

const resultLabels = {
  intent: '意图',
  goal: '目标',
  steps: '计划步骤',
  mode: '执行模式',
  planner_skipped: '跳过规划',
  retrieval_query: '检索查询',
  rewritten_queries: '改写查询',
  query_count: '查询数',
  hits: '命中数',
  bm25_hits: 'BM25 命中',
  embedding_hits: '向量命中',
  fused_candidates: '融合候选',
  fetched_candidates: '读取候选',
  reranked_candidates: '精排候选',
  candidates: '候选数',
  selected: '保留数',
  selected_documents: '保留文档',
  rerank_score: 'Rerank 分数',
  rrf_score: 'RRF 分数',
  model_grade: '模型分类',
  coverage: '覆盖度',
  max_score: '最高分',
  upper_threshold: '上阈值',
  lower_threshold: '下阈值',
  support_count: '直接支持',
  limited_support_count: '部分支持',
  incorrect_count: '不相关',
  accepted_count: '纳入证据',
  grading_failed: '评分失败',
  skipped: '跳过评分',
  confidence: '置信度',
  evidence_status: '证据状态',
  evidence_score: '证据分数',
  evidence_source_count: '证据来源',
  evidence_reason: '证据说明',
  answer_generated: '生成答案',
  source_count: '来源数量',
  operation: '操作',
  model: '模型',
  total_tokens: '总 Token',
  prompt_tokens: '输入 Token',
  completion_tokens: '输出 Token',
  context_tokens: '上下文 Token',
  tool_context_tokens: '工具上下文 Token',
  tool_context_budget: '工具上下文预算',
  tool_steps: '工具步骤',
  token_budget: '输出预算',
  token_remaining: '输出剩余',
  input_budget: '输入预算',
  input_remaining: '输入剩余',
  success: '调用成功',
  finish_reason: '结束原因',
  total_ms: '总耗时',
  recording_note: '记录说明',
}

const scoreKeys = new Set(['score', 'max_score', 'upper_threshold', 'lower_threshold', 'confidence', 'evidence_score'])

function resultEntries(result) {
  if (!result || typeof result !== 'object' || Array.isArray(result)) return []
  return Object.entries(result).filter(([key, value]) => key !== 'status' && value !== undefined && value !== null)
}

function formatResultValue(value, key) {
  if (key === 'selected_documents' && Array.isArray(value)) {
    return value.map((item, index) => {
      const title = item?.title || item?.section_title || '知识库页面'
      const rrf = item?.rrf_score == null ? '' : ` · RRF ${rerankScoreLabel(item.rrf_score)}`
      return `${index + 1}. ${title} · Rerank ${rerankScoreLabel(item?.score)}${rrf}`
    }).join('；') || '—'
  }
  if (scoreKeys.has(key) && Number.isFinite(Number(value))) return scoreLabel(value)
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (Array.isArray(value)) {
    return value.every((item) => typeof item === 'string')
      ? value.join('、') || '—'
      : JSON.stringify(value)
  }
  if (value && typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function resultSummary(item) {
  const result = item.result
  if (!result || typeof result !== 'object') return '旧记录仅保存耗时'
  const stage = item.stage || ''
  if (stage === 'agent.plan') {
    return `${result.intent || '未知意图'} · ${Array.isArray(result.steps) ? `${result.steps.length} 步` : '步骤未保存'}`
  }
  if (stage === 'agent.query_rewrite') {
    const queryCount = Array.isArray(result.rewritten_queries)
      ? result.rewritten_queries.length
      : (result.query_count ?? 0)
    return `${queryCount} 个查询 · ${result.retrieval_query || '改写文本未保存'}`
  }
  if (stage === 'retrieval.embedding' || stage === 'retrieval.bm25') {
    return `命中 ${result.hits ?? 0}`
  }
  if (stage === 'retrieval.recall' || stage === 'agent.retrieve') {
    const context = result.token_budget == null
      ? ''
      : ` · 上下文 ${tokenCount(result.context_tokens)}/${tokenCount(result.token_budget)} Token`
    return `BM25 ${result.bm25_hits ?? 0} · 向量 ${result.embedding_hits ?? 0} · RRF ${result.fused_candidates ?? 0}${context}`
  }
  if (stage === 'retrieval.fusion') return `融合候选 ${result.fused_candidates ?? 0}`
  if (stage === 'retrieval.fetch_chunks') return `读取候选 ${result.fetched_candidates ?? 0}`
  if (stage === 'retrieval.rerank') {
    const selected = (result.selected_documents || []).map((item) => (
      `${item.title || item.section_title || '知识库页面'} (${rerankScoreLabel(item.score)})`
    )).join('、')
    return `候选 ${result.candidates ?? 0} · 保留 ${result.selected ?? 0}${selected ? ` · ${selected}` : ''}`
  }
  if (stage === 'agent.crag_grade') {
    if (result.skipped) return '已跳过评分 · 高可信证据'
    const counts = result.support_count == null || result.limited_support_count == null
      ? '分类数量未保存'
      : `支持 ${result.support_count} · 部分 ${result.limited_support_count}`
    return `最高分 ${scoreLabel(result.max_score)} · ${counts}`
  }
  if (stage === 'agent.evidence') {
    if (result.recording_note && result.evidence_status == null) return result.recording_note
    return `${evidenceLabel(result.evidence_status)} · ${scoreLabel(result.evidence_score)}`
  }
  if (stage === 'agent.generation') {
    const generated = result.answer_generated == null
      ? '生成状态未保存'
      : (result.answer_generated ? '已生成' : '未生成')
    const sources = result.source_count == null ? '来源数未保存' : `来源 ${result.source_count}`
    return `${generated} · ${sources}`
  }
  if (stage.startsWith('llm.')) {
    const usage = result.token_budget == null
      ? `${tokenCount(result.total_tokens)} Token`
      : `输出 ${tokenCount(result.completion_tokens)}/${tokenCount(result.token_budget)} Token`
    return `${result.model || '未知模型'} · ${usage}`
  }
  if (stage === 'agent.total') return formatDuration(result.total_ms)
  const entries = resultEntries(result)
  return entries.slice(0, 2).map(([key, value]) => `${resultLabels[key] || key} ${formatResultValue(value, key)}`).join(' · ') || '旧记录仅保存耗时'
}

function TraceResult({ result }) {
  const entries = resultEntries(result)
  if (!entries.length) return <div className="trace-result-empty">旧记录仅保存了该阶段耗时</div>
  return (
    <div className="trace-result-grid">
      {entries.map(([key, value]) => (
        <div className={`trace-result-field ${key === 'selected_documents' ? 'trace-result-field-wide' : ''}`} key={key}>
          <span>{resultLabels[key] || key}</span>
          <strong>{formatResultValue(value, key)}</strong>
        </div>
      ))}
    </div>
  )
}

function budgetRatio(used, budget) {
  const normalizedUsed = Number(used)
  const normalizedBudget = Number(budget)
  if (!Number.isFinite(normalizedUsed) || !Number.isFinite(normalizedBudget) || normalizedBudget <= 0) return 0
  return Math.min(100, Math.max(0, (normalizedUsed / normalizedBudget) * 100))
}

function BudgetUsage({ used, budget, remaining, truncated = false }) {
  const hasBudget = budget !== null && budget !== undefined
  return (
    <div className="trace-budget-usage">
      <span><strong>{tokenCount(used)}</strong>{hasBudget ? ` / ${tokenCount(budget)} Token` : ' Token'}</span>
      {hasBudget && <i><b style={{ width: `${budgetRatio(used, budget)}%` }} /></i>}
      <small>{hasBudget ? `剩余 ${tokenCount(remaining)}${truncated ? ' · 已截断' : ''}` : (truncated ? '已截断' : '未设置预算')}</small>
    </div>
  )
}

function ModelCallBudget({ calls, usageCalls }) {
  if (!calls.length) return <div className="trace-budget-empty">本次没有模型调用</div>
  return (
    <div className="trace-budget-rows">
      {calls.map((call, index) => {
        const usage = usageCalls[index] || {}
        return (
          <div className="trace-budget-row trace-model-call-row" key={`${call.stage}-${index}`}>
            <div className="trace-budget-row-title">
              <strong>{stageLabel(call.stage)}</strong>
              <small>{usage.model || '模型未记录'}{usage.duration_ms == null ? '' : ` · ${formatDuration(usage.duration_ms)}`}</small>
            </div>
            <div><span>实际输入</span><strong>{tokenCount(call.input_tokens)} Token</strong></div>
            <div><span>输出使用</span><strong>{tokenCount(call.output_tokens)} / {tokenCount(call.budget)}</strong></div>
            <div><span>输出剩余</span><strong>{tokenCount(call.remaining)} Token</strong></div>
            <div><span>合计</span><strong>{tokenCount(call.total_used)} Token</strong></div>
          </div>
        )
      })}
    </div>
  )
}

function ModelContextBudget({ contexts }) {
  const entries = Object.entries(contexts || {})
  if (!entries.length) return <div className="trace-budget-empty">旧记录没有保存阶段上下文拆分</div>
  return (
    <div className="trace-budget-rows">
      {entries.map(([stage, context]) => (
        <div className="trace-budget-row trace-context-row" key={stage}>
          <div className="trace-budget-row-title">
            <strong>{stageLabel(stage)}</strong>
            <small>{context.legacy_estimate ? '根据历史调用记录推导' : '发送前上下文估算'}</small>
          </div>
          <BudgetUsage
            used={context.used_tokens}
            budget={context.input_budget}
            remaining={context.remaining}
            truncated={context.truncated}
          />
          <div className="trace-context-breakdown">
            <span>系统 <strong>{tokenCount(context.system_tokens)}</strong></span>
            <span>历史摘要 <strong>{tokenCount(context.summary_tokens)}</strong></span>
            <span>近期消息 <strong>{tokenCount(context.recent_tokens)}</strong></span>
            <span>当前任务 <strong>{tokenCount(context.current_tokens)}</strong></span>
            <span>输出预留 <strong>{tokenCount(context.output_reserved_tokens)}</strong></span>
            <span>丢弃消息 <strong>{tokenCount(context.dropped_recent_messages)}</strong></span>
          </div>
        </div>
      ))}
    </div>
  )
}

function ContextSourceBudget({ rag, tool, responseSummary, memory }) {
  const toolSteps = tool.steps || []
  return (
    <div className="trace-context-sources">
      <div className="trace-context-source">
        <div><strong>RAG 证据</strong><small>{rag.selected_results ?? 0} 条结果</small></div>
        <BudgetUsage used={rag.used} budget={rag.budget} remaining={rag.remaining} truncated={rag.truncated} />
      </div>
      <div className="trace-context-source">
        <div><strong>工具结果</strong><small>{toolSteps.length} 个步骤 · 单步 {tokenCount(tool.per_step)} Token</small></div>
        <BudgetUsage used={tool.used} budget={tool.budget} remaining={tool.remaining} truncated={toolSteps.some((item) => item.truncated)} />
      </div>
      {responseSummary && (
        <div className="trace-context-source">
          <div><strong>工具响应摘要</strong><small>截断前估算 {tokenCount(responseSummary.estimated_tokens)} Token</small></div>
          <BudgetUsage
            used={responseSummary.used_tokens}
            budget={responseSummary.token_budget}
            remaining={Math.max(0, Number(responseSummary.token_budget || 0) - Number(responseSummary.used_tokens || 0))}
            truncated={responseSummary.truncated}
          />
        </div>
      )}
      <div className="trace-memory-budget">
        <span>对话历史 <strong>{tokenCount(memory.history_budget)}</strong></span>
        <span>摘要保留 <strong>{tokenCount(memory.summary_budget)}</strong></span>
        <span>压缩触发 <strong>{tokenCount(memory.summary_trigger)}</strong></span>
        <span>摘要输入 <strong>{tokenCount(memory.summary_input_budget)}</strong></span>
      </div>
      {toolSteps.length > 0 && (
        <div className="trace-tool-step-list">
          {toolSteps.map((step) => (
            <div key={`${step.step_id}-${step.tool}`}>
              <span>步骤 {step.step_id} · {step.tool}</span>
              <strong>{tokenCount(step.used)} / {tokenCount(step.budget)} Token{step.truncated ? ' · 已截断' : ''}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function DiagnosticsWorkspace({ notify }) {
  const [requests, setRequests] = useState([])
  const [conversationQuery, setConversationQuery] = useState('')
  const [trace, setTrace] = useState(null)
  const [expandedStage, setExpandedStage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [listLoading, setListLoading] = useState(true)

  const loadRecent = useCallback(async (search = '') => {
    setListLoading(true)
    try {
      const params = new URLSearchParams({ limit: '40' })
      if (search.trim()) params.set('conversation', search.trim())
      const data = await api(`/api/observability/requests?${params}`)
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

  async function loadTrace(selectedId, resetExpanded = true) {
    const normalized = selectedId.trim()
    if (!normalized) return
    setLoading(true)
    try {
      const data = await api(`/api/observability/requests/${encodeURIComponent(normalized)}`)
      setTrace(data)
      if (resetExpanded) setExpandedStage(null)
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setLoading(false)
    }
  }

  async function refreshWorkspace() {
    await loadRecent(conversationQuery)
    if (trace?.request_id) await loadTrace(trace.request_id, false)
  }

  const timeline = useMemo(() => trace?.timeline || [], [trace])
  const modelUsage = trace?.model_usage || {}
  const retrievalStats = trace?.retrieval_stats || {}
  const cragStats = retrievalStats.crag || {}
  const tokenBudget = trace?.token_budget || {}
  const outputBudget = tokenBudget.output || {}
  const inputBudget = tokenBudget.input || {}
  const ragBudget = tokenBudget.contexts?.rag || {}
  const toolBudget = tokenBudget.contexts?.tool || {}
  const modelContexts = tokenBudget.contexts?.model || {}
  const responseSummaryBudget = tokenBudget.contexts?.response_summary || null
  const memoryBudget = tokenBudget.contexts?.memory || {}
  const modelCalls = tokenBudget.calls || []
  const usageCalls = modelUsage.calls || []

  return (
    <div className="diagnostics-workspace">
      <header className="workspace-titlebar">
        <div><span className="eyebrow">LOCAL OBSERVABILITY</span><h1>运行追踪</h1></div>
        <button className="secondary-button" type="button" onClick={refreshWorkspace} disabled={listLoading || loading}>
          {listLoading || loading ? <LoaderCircle size={16} className="spin" /> : <RefreshCw size={16} />} 刷新
        </button>
      </header>

      <div className="diagnostics-layout">
        <aside className="trace-browser">
          <div className="trace-browser-heading"><Activity size={17} /><strong>近期查询</strong><span>{requests.length}</span></div>
          <div className="trace-list">
            {requests.map((item) => (
              <button
                key={item.request_id}
                className={trace?.request_id === item.request_id ? 'trace-row active' : 'trace-row'}
                type="button"
                onClick={() => loadTrace(item.request_id)}
              >
                <span className={`trace-state ${item.status}`}>{statusLabel(item.status)}</span>
                <strong>{item.conversation_title || '新对话'}</strong>
                <small>{item.query || '未记录问题'} · {evidenceLabel(item.evidence_status)} · {formatDuration(item.total_ms)} · {formatTime(item.completed_at)}</small>
                {item.error_code && <code>{item.error_code}</code>}
              </button>
            ))}
            {!listLoading && requests.length === 0 && <div className="trace-list-empty">暂无运行记录</div>}
          </div>
        </aside>

        <main className="trace-detail">
          <form className="trace-search" onSubmit={(event) => { event.preventDefault(); loadRecent(conversationQuery) }}>
            <Search size={17} />
            <input value={conversationQuery} onChange={(event) => setConversationQuery(event.target.value)} placeholder="搜索对话名称" aria-label="搜索对话名称" />
            <button className="primary-button" type="submit" disabled={listLoading}>
              {listLoading ? <LoaderCircle size={16} className="spin" /> : <Search size={16} />} 查询
            </button>
          </form>

          {!trace ? (
            <div className="trace-empty"><ServerCog size={30} /><strong>选择一条查询记录</strong></div>
          ) : (
            <div className="trace-content">
              <section className="trace-summary-band">
                <div><span>状态</span><strong className={`trace-status-text ${trace.status}`}>{statusLabel(trace.status)}</strong></div>
                <div><span>可信度</span><strong>{scoreLabel(trace.answer_quality?.evidence_score ?? trace.evidence_score)} <em>{evidenceLabel(trace.answer_quality?.evidence_status ?? trace.evidence_status)}</em></strong></div>
                <div><span>证据来源</span><strong>{trace.answer_quality?.evidence_source_count ?? trace.evidence_source_count ?? 0} 个页面</strong></div>
                <div><span>总耗时</span><strong>{formatDuration(trace.total_ms)}</strong></div>
              </section>

              <section className="trace-identity">
                <span>对话名称</span><strong>{trace.conversation_title || '新对话'}</strong><small>{formatTime(trace.completed_at)}</small>
              </section>
              <section className="trace-identity">
                <span>本次问题</span><strong>{trace.query || '未记录问题'}</strong><small>{trace.intent || 'Agent 查询'}</small>
              </section>
              <section className="trace-identity trace-request-id">
                <span>Request ID</span><code>{trace.request_id}</code><small>用于定位本次运行</small>
              </section>

              {trace.error && (
                <section className="trace-error-band">
                  <AlertTriangle size={18} />
                  <div><strong>{trace.error.error_code || 'UNKNOWN_ERROR'}</strong><span>{trace.error.message || '请求执行失败'}</span></div>
                  <em>{trace.error.retryable ? '可重试' : '需检查'}</em>
                </section>
              )}

              <section className="trace-retrieval">
                <div className="section-heading"><Activity size={17} /><h2>检索摘要</h2></div>
                <div className="trace-metric-grid">
                  <div><span>查询变体</span><strong>{retrievalStats.query_count || 0}</strong></div>
                  <div><span>BM25 召回</span><strong>{retrievalStats.bm25_hits || 0}</strong></div>
                  <div><span>向量召回</span><strong>{retrievalStats.embedding_hits || 0}</strong></div>
                  <div><span>RRF 候选</span><strong>{retrievalStats.fused_candidates || 0}</strong></div>
                  <div><span>精排候选</span><strong>{retrievalStats.reranked_candidates || 0}</strong></div>
                  <div><span>CRAG 最高分</span><strong>{cragStats.skipped ? '已跳过' : scoreLabel(cragStats.max_score)}</strong></div>
                </div>
              </section>

              <section className="trace-token-budget">
                <div className="section-heading"><Activity size={17} /><h2>Token 预算</h2><span>{tokenBudget.model?.call_count || modelUsage.call_count || 0} 次模型调用</span></div>
                <div className="trace-metric-grid trace-token-budget-grid">
                  <div><span>模型总 Token</span><strong>{tokenCount(tokenBudget.model?.total_used ?? modelUsage.total_tokens)} Token</strong></div>
                  <div><span>实际输入</span><strong>{tokenCount(inputBudget.used)} Token</strong></div>
                  <div><span>累计输入容量</span><strong>{tokenCount(inputBudget.budget)} Token</strong></div>
                  <div><span>实际输出</span><strong>{tokenCount(outputBudget.used)} / {tokenCount(outputBudget.budget)}</strong></div>
                  <div><span>输出剩余</span><strong>{tokenCount(outputBudget.remaining)} Token</strong></div>
                  <div><span>RAG 证据</span><strong>{ragBudget.budget == null ? '—' : `${tokenCount(ragBudget.used)} / ${tokenCount(ragBudget.budget)}`}</strong></div>
                  <div><span>工具结果</span><strong>{toolBudget.budget ? `${tokenCount(toolBudget.used)} / ${tokenCount(toolBudget.budget)}` : '未使用'}</strong></div>
                  <div><span>上下文截断</span><strong>{[ragBudget.truncated, ...Object.values(modelContexts).map((item) => item.truncated), ...(toolBudget.steps || []).map((item) => item.truncated)].filter(Boolean).length} 处</strong></div>
                </div>
                <div className="trace-budget-details">
                  <details open>
                    <summary><span>模型调用使用量</span><small>接口返回的真实 Token</small></summary>
                    <ModelCallBudget calls={modelCalls} usageCalls={usageCalls} />
                  </details>
                  <details>
                    <summary><span>阶段上下文组成</span><small>系统、摘要、历史与当前任务</small></summary>
                    <ModelContextBudget contexts={modelContexts} />
                  </details>
                  <details>
                    <summary><span>RAG、工具与记忆</span><small>独立上下文预算</small></summary>
                    <ContextSourceBudget rag={ragBudget} tool={toolBudget} responseSummary={responseSummaryBudget} memory={memoryBudget} />
                  </details>
                </div>
                <p className="trace-token-budget-note">模型调用显示网关返回的真实用量；阶段上下文是发送前估算。各次模型窗口、RAG 和工具预算相互独立，不合并成一个可共享余额。</p>
              </section>

              <section className="trace-timeline-section">
                <div className="section-heading"><Clock3 size={17} /><h2>执行时间线</h2><span>{timeline.length} 个事件</span></div>
                <div className="trace-timeline">
                  {timeline.map((item, index) => (
                    <details className={`trace-stage ${item.status || 'completed'}`} key={`${item.stage}-${index}`} open={expandedStage === `${item.stage}-${index}`}>
                      <summary onClick={(event) => {
                        event.preventDefault()
                        setExpandedStage((current) => current === `${item.stage}-${index}` ? null : `${item.stage}-${index}`)
                      }}>
                        <i />
                        <div className="trace-stage-heading">
                          <strong>{stageLabel(item.stage)}</strong>
                          <small>{resultSummary(item)}</small>
                          {item.error_code && <code>{item.error_code}</code>}
                        </div>
                        <span className="trace-stage-duration">{item.duration_ms == null ? '—' : formatDuration(item.duration_ms)}</span>
                      </summary>
                      <div className="trace-stage-result"><TraceResult result={item.result} /></div>
                    </details>
                  ))}
                  {timeline.length === 0 && <div className="trace-list-empty">该请求没有阶段事件</div>}
                </div>
              </section>
              <section className="trace-usage">
                <div><span>模型调用</span><strong>{modelUsage.call_count || 0} 次</strong></div>
                <div><span>Token</span><strong>{modelUsage.total_tokens || 0}</strong></div>
                <div><span>输入 / 输出</span><strong>{modelUsage.prompt_tokens || 0} / {modelUsage.completion_tokens || 0}</strong></div>
              </section>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
