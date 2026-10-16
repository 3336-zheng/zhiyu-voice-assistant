import { useState } from 'react'
import {
  AlertTriangle,
  ArrowRight,
  Check,
  FileText,
  LoaderCircle,
  Sparkles,
  Upload,
} from 'lucide-react'
import { api } from '../../shared/api/client'
import MarkdownView from '../../shared/components/MarkdownView'

// 能走「先转换」这一步的格式。其余按纯文本读——但这里只提示，真正的拦截在后端，
// 因为 accept 属性在拖拽和「所有文件」下拦不住。
const CONVERT_SUFFIXES = ['.pdf', '.docx']

function suffixOf(name) {
  const index = String(name || '').lastIndexOf('.')
  return index < 0 ? '' : String(name).slice(index).toLowerCase()
}

export default function TransformWorkspace({ notify, onSaved }) {
  const [filename, setFilename] = useState('')
  const [title, setTitle] = useState('')
  const [original, setOriginal] = useState('')
  const [restructured, setRestructured] = useState('')
  const [model, setModel] = useState('')
  // 重排后数字被改动过时记下来，在对比区顶部提示。不阻断写入——
  // 用户看着左右两侧自己判断，可能是合理的调整。
  const [numberWarning, setNumberWarning] = useState(null)
  // converting: 转换/读取中；restructuring: 重排中；committing: 写入中
  const [busy, setBusy] = useState('')
  const [savedPageId, setSavedPageId] = useState(null)

  const neededConversion = CONVERT_SUFFIXES.includes(suffixOf(filename))
  const canRestructure = Boolean(original.trim()) && !busy
  const canCommit = Boolean(original.trim()) && Boolean(title.trim()) && !busy

  function reset() {
    setFilename('')
    setTitle('')
    setOriginal('')
    setRestructured('')
    setModel('')
    setNumberWarning(null)
    setSavedPageId(null)
  }

  async function handleFile(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setBusy('converting')
    setSavedPageId(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const result = await api('/api/transform/convert', { method: 'POST', body: form })
      setFilename(result.filename)
      setTitle(result.title)
      setOriginal(result.content)
      // 换文件后旧的重排结果必须清掉，否则会和新原文对不上。
      setRestructured('')
      setModel('')
      setNumberWarning(null)
      notify(result.converted ? '已转换，可以重排结构了' : '已读取，可以重排结构了', 'success')
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setBusy('')
    }
  }

  async function restructure() {
    setBusy('restructuring')
    try {
      const result = await api('/api/transform/restructure', {
        method: 'POST',
        body: JSON.stringify({ content: original }),
      })
      setRestructured(result.content)
      setModel(result.model)
      setNumberWarning(result.numbers_changed ? result.number_diff : null)
      notify('重排完成，请对比左右两侧', 'success')
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setBusy('')
    }
  }

  async function commit() {
    setBusy('committing')
    try {
      // 用户确认的是右侧（重排后）那份；没重排过就写原文。
      const content = restructured || original
      const result = await api('/api/transform/commit', {
        method: 'POST',
        body: JSON.stringify({ title: title.trim(), content, notebook: null }),
      })
      setSavedPageId(result.page_id)
      notify(result.message, 'success')
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="transform-workspace">
      <header className="workspace-titlebar">
        <div>
          <span className="eyebrow">文档转换</span>
          <h1>把文档整理成结构化 Markdown</h1>
        </div>
        <div className="transform-actions">
          {original && (
            <button className="secondary-button" type="button" disabled={Boolean(busy)} onClick={reset}>
              重新开始
            </button>
          )}
          <label className="primary-button transform-upload">
            <Upload size={15} /> 选择文件
            <input
              hidden
              type="file"
              accept=".md,.txt,.pdf,.docx"
              disabled={Boolean(busy)}
              onChange={handleFile}
            />
          </label>
        </div>
      </header>

      {!original && (
        <div className="transform-empty">
          <FileText size={30} />
          <strong>选择一个文件开始</strong>
          <p>
            PDF 和 Word 会先转换成 Markdown；Markdown 和纯文本直接读取。
            转换后的内容可以再做一次<strong>结构重排</strong>，重排由模型完成，
            只加结构不改内容，结果会在右侧对比显示，确认后才会写入知识库。
          </p>
        </div>
      )}

      {original && (
        <>
          <div className="transform-toolbar">
            <label className="transform-title">
              页面标题
              <input
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="写入知识库时使用的标题"
                disabled={Boolean(busy)}
              />
            </label>
            <div className="transform-steps">
              {/* 走到这一步说明文件已经读进来了，所以第一格恒为完成态。 */}
              <span className="done">
                <Check size={13} /> {neededConversion ? '转换' : '读取'}
              </span>
              <ArrowRight size={13} />
              <span className={restructured ? 'done' : ''}>
                {restructured ? <Check size={13} /> : null} 结构重排
              </span>
              <ArrowRight size={13} />
              <span className={savedPageId ? 'done' : ''}>
                {savedPageId ? <Check size={13} /> : null} 写入知识库
              </span>
            </div>
          </div>

            {numberWarning && (
              <div className="transform-warning" role="status">
                <AlertTriangle size={15} />
                <div>
                  <strong>重排改动了数字，请对照左右两侧确认</strong>
                  <span>
                    原文有而重排后没有：
                    {numberWarning.missing.length ? numberWarning.missing.join('、') : '无'}
                    {numberWarning.added.length ? ` ｜ 重排后新增：${numberWarning.added.join('、')}` : ''}
                  </span>
                </div>
              </div>
            )}
          <div className="transform-compare">
            <section className="transform-pane">
              <header>
                <strong>原始内容</strong>
                <span>{original.length} 字</span>
              </header>
              {/* 原文可编辑：转换出错时用户能直接改，不必去别处修文件再传。 */}
              <textarea
                className="transform-source"
                value={original}
                onChange={(event) => setOriginal(event.target.value)}
                disabled={Boolean(busy)}
              />
            </section>

            <section className="transform-pane">
              <header>
                <strong>重排后</strong>
                {restructured
                  ? <span>{restructured.length} 字 · {model}</span>
                  : <span>尚未重排</span>}
              </header>
              {restructured ? (
                <div className="transform-preview">
                  <MarkdownView content={restructured} />
                </div>
              ) : (
                <div className="transform-placeholder">
                  <Sparkles size={22} />
                  <span>点下面的「结构重排」生成结果</span>
                </div>
              )}
            </section>
          </div>

          <div className="transform-footer">
            <button
              className="secondary-button"
              type="button"
              disabled={!canRestructure}
              onClick={restructure}
            >
              {busy === 'restructuring'
                ? <LoaderCircle size={15} className="spin" />
                : <Sparkles size={15} />}
              {restructured ? '重新重排' : '结构重排'}
            </button>
            <button
              className="primary-button"
              type="button"
              disabled={!canCommit}
              onClick={commit}
            >
              {busy === 'committing' ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}
              {restructured ? '采用重排结果并写入' : '写入知识库'}
            </button>
            {savedPageId && (
              <button className="text-button" type="button" onClick={() => onSaved?.(savedPageId)}>
                去知识库查看
              </button>
            )}
          </div>
        </>
      )}
    </div>
  )
}
