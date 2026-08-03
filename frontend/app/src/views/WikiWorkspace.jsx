import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArchiveRestore,
  BookOpen,
  Clock3,
  Download,
  Edit3,
  FilePlus2,
  FileText,
  History,
  Link2,
  LoaderCircle,
  MoreHorizontal,
  Network,
  Plus,
  RefreshCw,
  Save,
  Search,
  Tag,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { api, download } from '../api'
import MarkdownView from '../components/MarkdownView'
import Modal from '../components/Modal'

const emptyDraft = { title: '', notebook: '', tags: '', aliases: '', content: '' }

function formatDate(value) {
  if (!value) return ''
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function headingsFrom(content) {
  return (content || '').split('\n').flatMap((line, index) => {
    const match = /^(#{1,4})\s+(.+)$/.exec(line)
    return match ? [{ level: match[1].length, title: match[2], key: `${index}-${match[2]}` }] : []
  })
}

export default function WikiWorkspace({ searchQuery, setSearchQuery, notify }) {
  const [pages, setPages] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [current, setCurrent] = useState(null)
  const [links, setLinks] = useState({ outgoing: [], backlinks: [] })
  const [revisions, setRevisions] = useState([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(emptyDraft)
  const [notebookFilter, setNotebookFilter] = useState('')
  const [tagFilter, setTagFilter] = useState('')
  const [modal, setModal] = useState(null)
  const uploadRef = useRef(null)

  const loadPages = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ page_size: '100' })
      if (searchQuery) params.set('query', searchQuery)
      if (notebookFilter) params.set('notebook', notebookFilter)
      if (tagFilter) params.set('tag', tagFilter)
      const data = await api(`/api/pages?${params}`)
      setPages(data.results || [])
      setSelectedId((previous) => {
        if (previous && data.results?.some((page) => page.id === previous)) return previous
        return data.results?.[0]?.id || null
      })
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setLoading(false)
    }
  }, [notebookFilter, notify, searchQuery, tagFilter])

  const loadPage = useCallback(async (pageId) => {
    if (!pageId) {
      setCurrent(null)
      return
    }
    try {
      const [page, linkData, revisionData] = await Promise.all([
        api(`/api/pages/${pageId}`),
        api(`/api/pages/${pageId}/links`),
        api(`/api/pages/${pageId}/revisions`),
      ])
      setCurrent(page)
      setLinks(linkData)
      setRevisions(revisionData.revisions || [])
      setDraft({
        title: page.title,
        notebook: page.notebook || '',
        tags: (page.tags || []).join(', '),
        aliases: (page.aliases || []).join(', '),
        content: page.content || '',
      })
      setEditing(false)
    } catch (error) {
      notify(error.message, 'error')
    }
  }, [notify])

  useEffect(() => {
    const timer = window.setTimeout(loadPages, 180)
    return () => window.clearTimeout(timer)
  }, [loadPages])

  useEffect(() => {
    loadPage(selectedId)
  }, [loadPage, selectedId])

  const notebooks = useMemo(
    () => [...new Set(pages.map((page) => page.notebook).filter(Boolean))].sort(),
    [pages],
  )
  const tags = useMemo(
    () => [...new Set(pages.flatMap((page) => page.tags || []))].sort(),
    [pages],
  )
  const outline = useMemo(() => headingsFrom(current?.content), [current?.content])

  async function savePage() {
    if (!current || !draft.title.trim()) return
    setSaving(true)
    try {
      const updated = await api(`/api/pages/${current.id}`, {
        method: 'PUT',
        body: JSON.stringify({
          expected_revision: current.revision,
          title: draft.title.trim(),
          notebook: draft.notebook.trim() || null,
          tags: draft.tags.split(',').map((item) => item.trim()).filter(Boolean),
          aliases: draft.aliases.split(',').map((item) => item.trim()).filter(Boolean),
          content: draft.content,
          change_summary: '在 React Wiki 工作台中编辑',
        }),
      })
      setCurrent(updated)
      setEditing(false)
      await loadPages()
      await loadPage(updated.id)
      notify('页面已保存', 'success')
    } catch (error) {
      notify(error.message, 'error')
      if (error.message.includes('版本冲突')) await loadPage(current.id)
    } finally {
      setSaving(false)
    }
  }

  async function createPage() {
    if (!draft.title.trim()) return
    setSaving(true)
    try {
      const created = await api('/api/pages', {
        method: 'POST',
        body: JSON.stringify({
          title: draft.title.trim(),
          notebook: draft.notebook.trim() || null,
          tags: draft.tags.split(',').map((item) => item.trim()).filter(Boolean),
          content: draft.content,
        }),
      })
      setModal(null)
      await loadPages()
      setSelectedId(created.id)
      notify('页面已创建', 'success')
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  async function deletePage() {
    if (!current) return
    setSaving(true)
    try {
      await api(`/api/pages/${current.id}?expected_revision=${current.revision}`, { method: 'DELETE' })
      setModal(null)
      setCurrent(null)
      setSelectedId(null)
      await loadPages()
      notify('页面已删除，历史版本仍然保留', 'success')
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  async function rollback(revision) {
    if (!current) return
    setSaving(true)
    try {
      const restored = await api(`/api/pages/${current.id}/rollback`, {
        method: 'POST',
        body: JSON.stringify({ target_revision: revision, expected_revision: current.revision }),
      })
      setModal(null)
      await loadPage(restored.id)
      await loadPages()
      notify(`已恢复版本 ${revision}`, 'success')
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    try {
      const result = await api('/api/documents/upload', { method: 'POST', body: form })
      await loadPages()
      setSelectedId(result.page_id)
      notify(result.message, 'success')
    } catch (error) {
      notify(error.message, 'error')
    }
  }

  async function importLegacy(source) {
    try {
      const result = await api('/api/pages/import-legacy', {
        method: 'POST',
        body: JSON.stringify({ source, sync_index: true }),
      })
      await loadPages()
      notify(`已导入 ${result.imported} 个页面，跳过 ${result.skipped} 个`, 'success')
    } catch (error) {
      notify(error.message, 'error')
    }
  }

  async function openWikiLink(title) {
    const match = pages.find((page) =>
      [page.title, ...(page.aliases || [])].some((name) => name.toLowerCase() === title.toLowerCase()),
    )
    if (match) setSelectedId(match.id)
    else notify(`链接目标尚不存在：${title}`, 'error')
  }

  return (
    <div className="wiki-layout">
      <aside className="wiki-browser">
        <div className="browser-toolbar">
          <div>
            <span className="eyebrow">个人 Wiki</span>
            <strong>{pages.length} 个页面</strong>
          </div>
          <button
            className="icon-button primary-icon"
            type="button"
            title="新建页面"
            onClick={() => { setDraft(emptyDraft); setModal('create') }}
          >
            <Plus size={18} />
          </button>
        </div>

        <div className="filter-block">
          <div className="filter-label"><BookOpen size={14} /> 笔记本</div>
          <button className={!notebookFilter ? 'filter active' : 'filter'} onClick={() => setNotebookFilter('')}>全部页面</button>
          {notebooks.map((notebook) => (
            <button key={notebook} className={notebookFilter === notebook ? 'filter active' : 'filter'} onClick={() => setNotebookFilter(notebook)}>{notebook}</button>
          ))}
        </div>

        {tags.length > 0 && (
          <div className="filter-block">
            <div className="filter-label"><Tag size={14} /> 标签</div>
            <div className="tag-cloud">
              {tags.map((tag) => (
                <button key={tag} className={tagFilter === tag ? 'tag active' : 'tag'} onClick={() => setTagFilter(tagFilter === tag ? '' : tag)}>#{tag}</button>
              ))}
            </div>
          </div>
        )}

        <div className="page-list-header">
          <span>最近修改</span>
          {(notebookFilter || tagFilter) && (
            <button className="text-button" type="button" onClick={() => { setNotebookFilter(''); setTagFilter('') }}><X size={13} /> 清除</button>
          )}
        </div>
        <div className="page-list">
          {loading && <div className="inline-loading"><LoaderCircle size={16} className="spin" /> 加载中</div>}
          {!loading && pages.length === 0 && <div className="empty-inline">没有匹配的页面</div>}
          {pages.map((page) => (
            <button key={page.id} className={selectedId === page.id ? 'page-row active' : 'page-row'} type="button" onClick={() => setSelectedId(page.id)}>
              <FileText size={16} />
              <span><strong>{page.title}</strong><small>{page.notebook || page.source_type} · {formatDate(page.updated_at)}</small></span>
              {page.index_status === 'failed' && <i className="status-mark error" title={page.index_error}>!</i>}
            </button>
          ))}
        </div>

        <div className="browser-footer">
          <button className="icon-button" type="button" title="上传文档" onClick={() => uploadRef.current?.click()}><Upload size={17} /></button>
          <input ref={uploadRef} hidden type="file" accept=".md,.txt,.pdf,.docx" onChange={uploadDocument} />
          <button className="icon-button" type="button" title="导入旧笔记" onClick={() => importLegacy('notes')}><ArchiveRestore size={17} /></button>
          <button className="icon-button" type="button" title="导出 Wiki" onClick={() => download('/api/pages/export', 'zhiyu-wiki-export.zip').catch((error) => notify(error.message, 'error'))}><Download size={17} /></button>
          <button className="icon-button" type="button" title="重建索引" onClick={() => api('/api/pages/reindex', { method: 'POST' }).then(() => notify('索引任务已入队', 'success')).catch((error) => notify(error.message, 'error'))}><RefreshCw size={17} /></button>
        </div>
      </aside>

      <main className="page-workspace">
        {!current && !loading && (
          <div className="workspace-empty">
            <Network size={34} />
            <h2>建立第一条知识连接</h2>
            <button className="primary-button" type="button" onClick={() => { setDraft(emptyDraft); setModal('create') }}><FilePlus2 size={17} /> 新建页面</button>
          </div>
        )}
        {current && (
          <>
            <header className="page-header">
              <div className="breadcrumbs"><span>{current.notebook || '未分类'}</span><span>/</span><span>v{current.revision}</span></div>
              <div className="page-actions">
                {editing ? (
                  <>
                    <button className="secondary-button" type="button" onClick={() => { setEditing(false); loadPage(current.id) }}>取消</button>
                    <button className="primary-button" type="button" disabled={saving} onClick={savePage}><Save size={16} /> {saving ? '保存中' : '保存'}</button>
                  </>
                ) : (
                  <>
                    <button className="secondary-button" type="button" onClick={() => setEditing(true)}><Edit3 size={16} /> 编辑</button>
                    <button className="icon-button danger-icon" type="button" title="删除页面" onClick={() => setModal('delete')}><Trash2 size={17} /></button>
                  </>
                )}
              </div>
            </header>

            {editing ? (
              <div className="page-editor">
                <input className="title-input" value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} aria-label="页面标题" />
                <div className="metadata-grid">
                  <label>笔记本<input value={draft.notebook} onChange={(event) => setDraft({ ...draft, notebook: event.target.value })} /></label>
                  <label>标签<input value={draft.tags} onChange={(event) => setDraft({ ...draft, tags: event.target.value })} placeholder="RAG, LLM" /></label>
                  <label>别名<input value={draft.aliases} onChange={(event) => setDraft({ ...draft, aliases: event.target.value })} /></label>
                </div>
                <textarea className="markdown-editor" value={draft.content} onChange={(event) => setDraft({ ...draft, content: event.target.value })} aria-label="Markdown 内容" />
              </div>
            ) : (
              <article className="page-document">
                <h1>{current.title}</h1>
                <div className="page-meta-row">
                  {(current.tags || []).map((tag) => <span key={tag} className="tag">#{tag}</span>)}
                  <span><Clock3 size={13} /> {formatDate(current.updated_at)}</span>
                  <span className={`index-pill ${current.index_status}`}>{current.index_status === 'ready' ? '已索引' : current.index_status === 'pending' ? '待索引' : current.index_status === 'failed' ? '索引失败' : current.index_status}</span>
                </div>
                <MarkdownView content={current.content} onWikiLink={openWikiLink} />
              </article>
            )}
          </>
        )}
      </main>

      <aside className="context-panel">
        <section>
          <h3><MoreHorizontal size={15} /> 页面目录</h3>
          {outline.length === 0 ? <span className="muted">暂无标题</span> : outline.map((item) => (
            <div key={item.key} className="outline-item" style={{ paddingLeft: `${(item.level - 1) * 10}px` }}>{item.title}</div>
          ))}
        </section>
        <section>
          <h3><Link2 size={15} /> 反向链接 <span>{links.backlinks.length}</span></h3>
          {links.backlinks.length === 0 ? <span className="muted">暂无页面引用这里</span> : links.backlinks.map((item) => (
            <button className="context-link" type="button" key={item.page_id} onClick={() => setSelectedId(item.page_id)}>{item.title}</button>
          ))}
        </section>
        <section>
          <h3><Network size={15} /> 页面链接 <span>{links.outgoing.length}</span></h3>
          {links.outgoing.map((item) => (
            <button className={item.resolved ? 'context-link' : 'context-link unresolved'} type="button" key={item.target_title} onClick={() => item.target_page_id && setSelectedId(item.target_page_id)}>{item.target_title}</button>
          ))}
        </section>
        <section>
          <h3><History size={15} /> 历史版本 <span>{revisions.length}</span></h3>
          {revisions.slice(0, 8).map((revision) => (
            <button className="revision-row" type="button" key={revision.revision} onClick={() => revision.revision !== current?.revision && setModal({ type: 'rollback', revision: revision.revision })}>
              <span>v{revision.revision}</span><small>{formatDate(revision.created_at)}</small>
            </button>
          ))}
        </section>
      </aside>

      {modal === 'create' && (
        <Modal
          title="新建 Wiki 页面"
          onClose={() => setModal(null)}
          actions={<><button className="secondary-button" onClick={() => setModal(null)}>取消</button><button className="primary-button" disabled={saving || !draft.title.trim()} onClick={createPage}><Plus size={16} /> 创建</button></>}
        >
          <div className="form-stack">
            <label>标题<input autoFocus value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} /></label>
            <label>笔记本<input value={draft.notebook} onChange={(event) => setDraft({ ...draft, notebook: event.target.value })} /></label>
            <label>标签<input value={draft.tags} onChange={(event) => setDraft({ ...draft, tags: event.target.value })} placeholder="用逗号分隔" /></label>
            <label>正文<textarea rows="8" value={draft.content} onChange={(event) => setDraft({ ...draft, content: event.target.value })} /></label>
          </div>
        </Modal>
      )}
      {modal === 'delete' && (
        <Modal title="删除页面" onClose={() => setModal(null)} actions={<><button className="secondary-button" onClick={() => setModal(null)}>取消</button><button className="danger-button" disabled={saving} onClick={deletePage}><Trash2 size={16} /> 删除</button></>}>
          <p>将删除“{current?.title}”的当前页面和检索索引，历史版本仍可用于恢复。</p>
        </Modal>
      )}
      {modal?.type === 'rollback' && (
        <Modal title={`恢复版本 ${modal.revision}`} onClose={() => setModal(null)} actions={<><button className="secondary-button" onClick={() => setModal(null)}>取消</button><button className="primary-button" disabled={saving} onClick={() => rollback(modal.revision)}><ArchiveRestore size={16} /> 恢复</button></>}>
          <p>历史内容会作为新版本保存，不会覆盖或删除当前版本记录。</p>
        </Modal>
      )}
    </div>
  )
}
