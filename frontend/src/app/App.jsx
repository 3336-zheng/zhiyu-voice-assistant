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

// 导航分两组：上半是围绕知识的日常循环（存 → 看 → 问），下半是导入、探索、
// 诊断这些外围工具。
//
// 六项挤成一组时，日常用的那几个会被淹没——用户得逐个读完才知道哪个是自己
// 要的。两组之间留一段弹性空白，位置本身就是分组信号。
const navGroups = [
  {
    id: 'flow',
    label: '知识工作流',
    views: {
      wiki: { label: '知识库', icon: BookOpen },
      ask: { label: '可信问答', icon: MessageSquareText },
      capture: { label: '课堂沉淀', icon: Mic2 },
    },
  },
  {
    id: 'tools',
    label: '工具',
    views: {
      transform: { label: '文档转换', icon: FolderInput },
      graph: { label: '知识图谱', icon: Network },
      diagnostics: { label: '运行追踪', icon: Activity },
    },
  },
]

// 扁平索引，给顶栏标题和初始视图校验用。分组是渲染层的事，
// 按 key 查模块的地方不该关心它属于哪一组。
const views = Object.assign({}, ...navGroups.map((group) => group.views))

function getInitialView() {
  // 刷新后停在上次那个模块。存下来的值可能因为改版而失效（模块改名或被删），
  // 不校验的话 views[activeView].label 会读 undefined.label，整个页面白屏。
  const saved = localStorage.getItem('zhiyu_active_view')
  return saved && saved in views ? saved : 'wiki'
}

export default function App() {
  const [activeView, setActiveView] = useState(getInitialView)
  const [wikiTarget, setWikiTarget] = useState(null)
  const [toast, setToast] = useState(null)
  const [mobileNav, setMobileNav] = useState(false)

  const notify = useCallback((message, type = 'success') => {
    setToast({ message, type })
    window.setTimeout(() => setToast(null), 3200)
  }, [])

  function openView(view) {
    setActiveView(view)
    localStorage.setItem('zhiyu_active_view', view)
    setMobileNav(false)
  }

  function openWikiPage(pageId, section = null) {
    if (!pageId) return
    // section 是引用里的章节路径（「页面标题 > 一级 > 二级」），带上它才能
    // 落到被引用的那一段而不是文档开头。没有就照旧只开页面。
    setWikiTarget({ pageId, section, openedAt: Date.now() })
    // 引用跳转过去也算一次模块切换——下次刷新应该还在知识库。
    openView('wiki')
  }

  return (
    <div className="app-shell">
      <aside className={mobileNav ? 'app-sidebar open' : 'app-sidebar'}>
        <div className="brand">
          <div className="brand-mark"><BrainCircuit size={22} /></div>
          <div><strong>智语</strong><span>PERSONAL AI WIKI</span></div>
        </div>
        {navGroups.map((group, index) => (
          <div key={group.id} className={`nav-group${index > 0 ? ' secondary' : ''}`}>
            {/* 可见的分组标题。之前的分组信号只是两组之间的空白——空白一多，
                看起来像没排好的漏风而不是「这里分了两类」。写上名字就不靠猜了。 */}
            <span className="nav-group-label">{group.label}</span>
            <nav className="primary-nav" aria-label={group.label}>
              {Object.entries(group.views).map(([key, item]) => {
                const Icon = item.icon
                return <button key={key} type="button" className={activeView === key ? 'active' : ''} onClick={() => openView(key)}><Icon size={18} /><span>{item.label}</span></button>
              })}
            </nav>
          </div>
        ))}
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
