import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArchiveRestore,
  BookOpen,
  Clock3,
  Download,
  Edit3,
  FilePlus2,
  FileText,
  Link2,
  LoaderCircle,
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
import { api, download } from '../../shared/api/client'
import MarkdownView from '../../shared/components/MarkdownView'
import Modal from '../../shared/components/Modal'

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
  let ordinal = 0
  // 围栏代码块里的 `# 注释` 不是标题，必须跳过。
  //
  // 不跳的话目录项会比正文真实标题多，而锚点是按「第几个标题」编的号——
  // 从第一个假标题起全部错位，后面的目录项指向不存在的 id，点了没反应。
  // 这正是「有些标题能点、有些点不动」的原因：代码块越靠前，错位越多。
  let fence = null
  return (content || '').split('\n').flatMap((line, index) => {
    // 围栏标记：三个及以上反引号或波浪号开头。记录用的是哪种、多长，
    // 只有同种且不短于它的才闭合——markdown 的规则就是这样。
    const fenceMatch = /^\s*(`{3,}|~{3,})/.exec(line)
    if (fenceMatch) {
      const marker = fenceMatch[1]
      if (fence === null) {
        fence = marker
      } else if (marker[0] === fence[0] && marker.length >= fence.length) {
        fence = null
      }
      return []
    }
    if (fence !== null) return []

    const match = /^(#{1,4})\s+(.+)$/.exec(line)
    if (!match) return []
    // 锚点用「第几个标题」而不是行号或标题文字：
    // - 行号不行——MarkdownView 渲染的是经 normalizeMath 处理过的内容，
    //   公式会被展开成多行，两边的行号对不上；
    // - 标题文字不行——可能重复、含空格和中文标点。
    // 标题数量与顺序在两侧一致（规范化不动标题），序号是唯一稳定的依据。
    ordinal += 1
    return [{
      level: match[1].length,
      title: match[2],
      key: `${index}-${match[2]}`,
      anchor: `heading-${ordinal}`,
    }]
  })
}

export default function WikiWorkspace({ notify, openPage }) {
  const openPageId = openPage?.pageId || null
  const [pages, setPages] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedId, setSelectedId] = useState(null)
  const [current, setCurrent] = useState(null)
  const [links, setLinks] = useState({ outgoing: [], backlinks: [] })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(emptyDraft)
  const [notebookFilter, setNotebookFilter] = useState('')
  const [tagFilter, setTagFilter] = useState('')
  const [modal, setModal] = useState(null)
  // 索引迟迟没完成时给个说明。索引由独立 worker 异步做，
  // 它没在跑的时候保存会一直停在「待索引」，不说一声用户只能干等。
  const [indexStalled, setIndexStalled] = useState(false)
  const uploadRef = useRef(null)
  const pageRequestSequence = useRef(0)
  // 索引轮询的世代号。切换页面、再次保存、组件卸载都要让它失效，
  // 否则旧轮询回来会把 current 覆盖成另一个页面的数据。
  const indexPollSequence = useRef(0)

  const loadPages = useCallback(async () => {
    const requestSequence = pageRequestSequence.current + 1
    pageRequestSequence.current = requestSequence
    setLoading(true)
    try {
      const params = new URLSearchParams({ page_size: '100' })
      if (searchQuery) params.set('query', searchQuery)
      if (notebookFilter) params.set('notebook', notebookFilter)
      if (tagFilter) params.set('tag', tagFilter)
      const data = await api(`/api/pages?${params}`)
      if (requestSequence !== pageRequestSequence.current) return
      setPages(data.results || [])
      setSelectedId((previous) => {
        // 来源跳转必须优先于列表默认项，即使目标页被当前搜索或筛选条件隐藏。
        if (openPageId) return openPageId
        if (previous && data.results?.some((page) => page.id === previous)) return previous
        return data.results?.[0]?.id || null
      })
    } catch (error) {
      if (requestSequence !== pageRequestSequence.current) return
      notify(error.message, 'error')
    } finally {
      if (requestSequence === pageRequestSequence.current) setLoading(false)
    }
  }, [notebookFilter, notify, openPageId, searchQuery, tagFilter])

  const loadPage = useCallback(async (pageId) => {
    // 换页面就让在途的索引轮询作废——它回来时会 setCurrent，
    // 把刚切过去的那页覆盖成上一页的数据。
    indexPollSequence.current += 1
    setIndexStalled(false)
    if (!pageId) {
      setCurrent(null)
      return
    }
    try {
      const [page, linkData] = await Promise.all([
        api(`/api/pages/${pageId}`),
        api(`/api/pages/${pageId}/links`),
      ])
      setCurrent(page)
      setLinks(linkData)
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

  // 轮询页面索引状态，直到不再是 pending。
  //
  // 保存接口返回时索引还没跑完（索引由独立 worker 异步做，几百毫秒起），
  // 那时 index_status 是 pending，再也没有人回查——用户看到「待索引」
  // 一直不变，只能手动刷新。这里把它盯到终态为止。
  //
  // 用世代号而不是「页面 id 是否还匹配」来判废：再次保存同一页面时 id 不变，
  // 只比 id 会让两次轮询同时活着，后回来的那个可能带着更旧的状态。
  const pollIndexStatus = useCallback(async (pageId) => {
    indexPollSequence.current += 1
    const sequence = indexPollSequence.current
    setIndexStalled(false)
    // 索引是几百毫秒级的短活，30 秒还没到终态基本就是 worker 没在跑。
    for (let attempt = 0; attempt < 30; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 1000))
      if (sequence !== indexPollSequence.current) return
      try {
        const page = await api(`/api/pages/${pageId}`)
        if (sequence !== indexPollSequence.current) return
        // 只在状态真的变了才 setCurrent。每轮无条件更新会让整个
        // MarkdownView 每秒重渲染一次，长文档下这笔开销白花。
        setCurrent((previous) => (
          previous?.index_status === page.index_status && previous?.index_error === page.index_error
            ? previous
            : page
        ))
        // ready 和 failed 都是终态，都该停——失败的要让用户看到「索引失败」，
        // 而不是一直转「待索引」。
        if (page.index_status !== 'pending') return
      } catch {
        // 单次查询失败不终止轮询：可能是瞬时网络抖动，下一轮再试。
        // 连续失败到次数用尽会走下面的超时分支。
      }
    }
    if (sequence === indexPollSequence.current) setIndexStalled(true)
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(loadPages, 180)
    return () => window.clearTimeout(timer)
  }, [loadPages])

  // 卸载时让在途的索引轮询作废。它回来还会 setCurrent / setIndexStalled，
  // 对已经卸载的组件调用等于内存泄漏，也会让 React 报警告。
  useEffect(() => () => { indexPollSequence.current += 1 }, [])

  useEffect(() => {
    loadPage(selectedId)
  }, [loadPage, selectedId])

  useEffect(() => {
    if (openPageId) setSelectedId(openPageId)
  }, [openPageId])

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
      // 放在 loadPage 之后：它会把 current 重置成刚保存的 pending 状态，
      // 先起轮询的话第一轮读到的还是旧数据。
      void pollIndexStatus(updated.id)
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
      notify('页面已删除', 'success')
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

        <label className="browser-search">
          <Search size={15} />
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="搜索文档名称和正文"
            aria-label="搜索知识库文档"
          />
          {searchQuery && (
            <button type="button" title="清空搜索" aria-label="清空知识库搜索" onClick={() => setSearchQuery('')}>
              <X size={14} />
            </button>
          )}
        </label>

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
            <button key={page.id} className={`${selectedId === page.id ? 'page-row active' : 'page-row'}${page.match_snippet ? ' with-snippet' : ''}`} type="button" onClick={() => setSelectedId(page.id)}>
              <FileText size={16} />
              <span>
                <strong>{page.title}</strong>
                <small>{page.notebook || page.source_type} · {formatDate(page.updated_at)}</small>
                {page.match_snippet && <em>{page.match_snippet}</em>}
              </span>
              {page.index_status === 'failed' && <i className="status-mark error" title={page.index_error}>!</i>}
            </button>
          ))}
        </div>

        <div className="browser-footer">
          <button className="icon-button" type="button" title="上传 Markdown 或纯文本" onClick={() => uploadRef.current?.click()}><Upload size={17} /></button>
          <input ref={uploadRef} hidden type="file" accept=".md,.txt" onChange={uploadDocument} />
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
              <div className="breadcrumbs"><span>{current.notebook || '未分类'}</span></div>
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
                  {indexStalled && current.index_status === 'pending' && (
                    <span className="muted" title="索引由后台服务异步完成；它没在运行时，保存会一直停在这一步。">
                      索引耗时偏长，请确认后台服务已启动
                    </span>
                  )}
                </div>
                <MarkdownView content={current.content} onWikiLink={openWikiLink} />
              </article>
            )}
          </>
        )}
      </main>

      <aside className="context-panel">
        <section>
          <h3>页面目录</h3>
          {outline.length === 0 ? <span className="muted">暂无标题</span> : outline.map((item) => (
            <button
              key={item.key}
              type="button"
              className="outline-item"
              style={{ paddingLeft: `${(item.level - 1) * 10}px` }}
              onClick={() => document.getElementById(item.anchor)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
            >{item.title}</button>
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
          {links.outgoing.length === 0 ? <span className="muted">暂无页面链接</span> : links.outgoing.map((item) => (
            <button className={item.resolved ? 'context-link' : 'context-link unresolved'} type="button" key={item.target_title} onClick={() => item.target_page_id && setSelectedId(item.target_page_id)}>{item.target_title}</button>
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
          <p>将删除“{current?.title}”及其检索索引。</p>
        </Modal>
      )}
    </div>
  )
}
