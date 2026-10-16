import { useCallback, useState } from 'react'
import {
  Activity,
  BookOpen,
  BrainCircuit,
  Menu,
  MessageSquareText,
  Mic2,
  Network,
  FolderInput,
  X,
} from 'lucide-react'
import AskWorkspace from '../features/ask/AskWorkspace'
import CaptureWorkspace from '../features/capture/CaptureWorkspace'
import DiagnosticsWorkspace from '../features/observability/DiagnosticsWorkspace'
import GraphWorkspace from '../features/graph/GraphWorkspace'
import TransformWorkspace from '../features/transform/TransformWorkspace'
import WikiWorkspace from '../features/wiki/WikiWorkspace'

const views = {
  wiki: { label: '知识库', icon: BookOpen },
  ask: { label: '可信问答', icon: MessageSquareText },
  capture: { label: '课堂沉淀', icon: Mic2 },
  transform: { label: '文档转换', icon: FolderInput },
  graph: { label: '知识图谱', icon: Network },
  diagnostics: { label: '运行追踪', icon: Activity },
}

export default function App() {
  const [activeView, setActiveView] = useState('wiki')
  const [wikiTarget, setWikiTarget] = useState(null)
  const [toast, setToast] = useState(null)
  const [mobileNav, setMobileNav] = useState(false)

  const notify = useCallback((message, type = 'success') => {
    setToast({ message, type })
    window.setTimeout(() => setToast(null), 3200)
  }, [])

  function openView(view) {
    setActiveView(view)
    setMobileNav(false)
  }

  function openWikiPage(pageId, section = null) {
    if (!pageId) return
    // section 是引用里的章节路径（「页面标题 > 一级 > 二级」），带上它才能
    // 落到被引用的那一段而不是文档开头。没有就照旧只开页面。
    setWikiTarget({ pageId, section, openedAt: Date.now() })
    setActiveView('wiki')
    setMobileNav(false)
  }

  return (
    <div className="app-shell">
      <aside className={mobileNav ? 'app-sidebar open' : 'app-sidebar'}>
        <div className="brand">
          <div className="brand-mark"><BrainCircuit size={22} /></div>
          <div><strong>智语</strong><span>PERSONAL AI WIKI</span></div>
        </div>
        <nav className="primary-nav" aria-label="主导航">
          {Object.entries(views).map(([key, item]) => {
            const Icon = item.icon
            return <button key={key} type="button" className={activeView === key ? 'active' : ''} onClick={() => openView(key)}><Icon size={18} /><span>{item.label}</span></button>
          })}
        </nav>
        <div className="sidebar-status"><i /> 本地知识服务</div>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <button
            className="mobile-menu"
            type="button"
            aria-label={mobileNav ? '关闭导航' : '打开导航'}
            title={mobileNav ? '关闭导航' : '打开导航'}
            onClick={() => setMobileNav(!mobileNav)}
          >
            {mobileNav ? <X size={20} /> : <Menu size={20} />}
          </button>
          <div className="topbar-context">{views[activeView].label}</div>
          <div className="topbar-badge">Agent Runtime <span>online</span></div>
        </header>

        <div className="view-container">
          {activeView === 'wiki' && <WikiWorkspace notify={notify} openPage={wikiTarget} />}
          {activeView === 'ask' && <AskWorkspace notify={notify} onOpenSource={openWikiPage} />}
          {activeView === 'capture' && <CaptureWorkspace notify={notify} onSaved={() => openView('wiki')} />}
          {activeView === 'transform' && <TransformWorkspace notify={notify} onSaved={openWikiPage} />}
          {activeView === 'graph' && <GraphWorkspace notify={notify} openPage={openWikiPage} />}
          {activeView === 'diagnostics' && <DiagnosticsWorkspace notify={notify} />}
        </div>
      </div>

      {mobileNav && <button className="nav-scrim" type="button" aria-label="关闭导航" onClick={() => setMobileNav(false)} />}
      {toast && <div className={`app-toast ${toast.type}`} role="status">{toast.message}</div>}
    </div>
  )
}
