import { useCallback, useState } from 'react'
import {
  BookOpen,
  BrainCircuit,
  Command,
  Menu,
  MessageSquareText,
  Mic2,
  Search,
  X,
} from 'lucide-react'
import AskWorkspace from './views/AskWorkspace'
import CaptureWorkspace from './views/CaptureWorkspace'
import WikiWorkspace from './views/WikiWorkspace'

const views = {
  wiki: { label: '知识库', icon: BookOpen },
  ask: { label: '可信问答', icon: MessageSquareText },
  capture: { label: '课堂沉淀', icon: Mic2 },
}

export default function App() {
  const [activeView, setActiveView] = useState('wiki')
  const [searchQuery, setSearchQuery] = useState('')
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
          {activeView === 'wiki' ? (
            <label className="global-search"><Search size={17} /><input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="搜索标题、标签或别名" /><kbd><Command size={12} /> K</kbd></label>
          ) : <div className="topbar-context">{views[activeView].label}</div>}
          <div className="topbar-badge">Agent Runtime <span>online</span></div>
        </header>

        <div className="view-container">
          {activeView === 'wiki' && <WikiWorkspace searchQuery={searchQuery} setSearchQuery={setSearchQuery} notify={notify} />}
          {activeView === 'ask' && <AskWorkspace notify={notify} />}
          {activeView === 'capture' && <CaptureWorkspace notify={notify} onSaved={() => openView('wiki')} />}
        </div>
      </div>

      {mobileNav && <button className="nav-scrim" type="button" aria-label="关闭导航" onClick={() => setMobileNav(false)} />}
      {toast && <div className={`app-toast ${toast.type}`} role="status">{toast.message}</div>}
    </div>
  )
}
