const stageLabels = {
  'agent.plan': '规划',
  'agent.route': '路由',
  'agent.query_rewrite': '查询改写',
  'agent.retrieve': '检索',
  'agent.crag_grade': '证据校正',
  'agent.evidence': '证据门禁',
  'agent.generation': '答案生成',
  'agent.total': '总耗时',
  'retrieval.embedding': '向量化',
  'retrieval.recall': '混合召回',
  'retrieval.fusion': '统一融合',
  'retrieval.fetch_chunks': '父块读取',
  'retrieval.rerank': '统一精排',
  'llm.primary': '主模型',
  'llm.fallback': '备用模型',
}

export function stageLabel(stage) {
  return stageLabels[stage] || stage.replace(/^agent\.|^retrieval\.|^llm\./, '')
}

export function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0))
  const minutes = Math.floor(total / 60)
  const remainder = total % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

export function formatCost(value) {
  const amount = Number(value) || 0
  return amount > 0 ? `$${amount.toFixed(6)}` : '未配置单价'
}
