import { RefreshCw } from 'lucide-react'

export default function McpStatusPanel({ loading, status, onRefresh }) {
  return (
    <section className="mcp-status-panel" aria-label="MCP 状态">
      <div className="mcp-status-main">
        <span className={`status-dot ${status?.status === 'healthy' ? 'healthy' : ''}`} />
        <div>
          <strong>{status?.server_label || '外部研究服务'}</strong>
          <small>{loading ? '正在读取状态' : status?.status || (status?.available ? '已配置' : '未启用')}</small>
        </div>
      </div>
      <div className="mcp-tool-map">
        <span>Search <strong>{status?.tools?.search || '-'}</strong></span>
        <span>Fetch <strong>{status?.tools?.fetch || '-'}</strong></span>
      </div>
      <button className="icon-button" type="button" title="检查 MCP 连接" aria-label="检查 MCP 连接" disabled={loading || !status?.available} onClick={onRefresh}>
        <RefreshCw size={16} className={loading ? 'spin' : ''} />
      </button>
    </section>
  )
}
