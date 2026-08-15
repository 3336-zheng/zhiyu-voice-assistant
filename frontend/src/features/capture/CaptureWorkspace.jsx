import { useRef, useState } from 'react'
import {
  Check,
  FileAudio,
  FileUp,
  LoaderCircle,
  Mic,
  Pause,
  Play,
  Save,
  Square,
  WandSparkles,
} from 'lucide-react'
import { api } from '../../shared/api/client'
import MarkdownView from '../../shared/components/MarkdownView'

export default function CaptureWorkspace({ notify, onSaved }) {
  const [title, setTitle] = useState('')
  const [audioId, setAudioId] = useState(null)
  const [transcript, setTranscript] = useState('')
  const [segments, setSegments] = useState([])
  const [summary, setSummary] = useState('')
  const [provider, setProvider] = useState('whisper')
  const [stage, setStage] = useState('idle')
  const [recording, setRecording] = useState(false)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const fileRef = useRef(null)

  async function uploadAudio(file) {
    if (!file) return
    setStage('uploading')
    const form = new FormData()
    form.append('file', file, file.name || 'recording.webm')
    try {
      const result = await api('/audio/upload/', { method: 'POST', body: form })
      setAudioId(result.audio_id)
      setTranscript('')
      setSegments([])
      setSummary('')
      setStage('uploaded')
      notify('音频已上传', 'success')
    } catch (error) {
      setStage('idle')
      notify(error.message, 'error')
    }
  }

  async function toggleRecording() {
    if (recording) {
      recorderRef.current?.stop()
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (event) => chunksRef.current.push(event.data)
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop())
        setRecording(false)
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], `课堂录音_${Date.now()}.webm`, { type: 'audio/webm' })
        await uploadAudio(file)
      }
      recorderRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch (error) {
      notify(`无法开始录音：${error.message}`, 'error')
    }
  }

  async function transcribeAudio() {
    if (!audioId) return
    setStage('transcribing')
    try {
      const result = await api(`/audio/transcribe/${audioId}?provider=${provider}`, { method: 'POST' })
      setTranscript(result.transcription || '')
      setSegments(result.segments || [])
      setStage('transcribed')
      notify('语音转写完成', 'success')
    } catch (error) {
      setStage('uploaded')
      notify(error.message, 'error')
    }
  }

  async function generateSummary() {
    if (!transcript.trim()) return
    setStage('summarizing')
    try {
      const result = await api('/summary/generate', {
        method: 'POST',
        body: JSON.stringify({ content: transcript, title: title || null, segments }),
      })
      setSummary(result.summary)
      setStage('preview')
      notify('课堂笔记已生成，请确认后保存', 'success')
    } catch (error) {
      setStage('transcribed')
      notify(error.message, 'error')
    }
  }

  async function saveSummary() {
    if (!summary.trim()) return
    setStage('saving')
    const pageTitle = title.trim() || `课堂笔记_${new Date().toLocaleDateString('zh-CN').replaceAll('/', '-')}`
    try {
      const result = await api('/summary/save', {
        method: 'POST',
        body: JSON.stringify({
          title: pageTitle,
          filename: pageTitle,
          content: summary,
          audio_id: audioId,
        }),
      })
      setStage('saved')
      notify('课堂笔记已保存并建立索引', 'success')
      onSaved?.(result.page_id)
    } catch (error) {
      setStage('preview')
      notify(error.message, 'error')
    }
  }

  const busy = ['uploading', 'transcribing', 'summarizing', 'saving'].includes(stage)

  return (
    <div className="capture-workspace">
      <header className="workspace-titlebar">
        <div><span className="eyebrow">课堂知识沉淀</span><h1>从声音到可检索的 Wiki 页面</h1></div>
        <div className="stage-indicator">
          {['音频', '转写', '整理', '确认'].map((item, index) => <span key={item} className={stage === 'saved' || index <= ['idle', 'uploaded', 'transcribed', 'preview'].indexOf(stage) ? 'done' : ''}>{index + 1} {item}</span>)}
        </div>
      </header>

      <div className="capture-grid">
        <section className="capture-source">
          <div className="section-heading"><FileAudio size={17} /><h2>语音来源</h2></div>
          <label className="field-label">课程标题<input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="例如：机器学习第 3 讲" /></label>
          <div className={recording ? 'recorder active' : 'recorder'}>
            <div className="waveform" aria-hidden="true">{Array.from({ length: 32 }, (_, index) => <i key={index} style={{ height: `${12 + ((index * 17) % 36)}px` }} />)}</div>
            <button
              className={recording ? 'record-button active' : 'record-button'}
              type="button"
              aria-label={recording ? '停止录音' : '开始录音'}
              title={recording ? '停止录音' : '开始录音'}
              onClick={toggleRecording}
            >
              {recording ? <Square size={20} /> : <Mic size={21} />}
            </button>
            <strong>{recording ? '正在录音' : audioId ? '音频已就绪' : '开始课堂录音'}</strong>
          </div>
          <div className="capture-actions-row">
            <button className="secondary-button" type="button" onClick={() => fileRef.current?.click()}><FileUp size={16} /> 上传音频</button>
            <input ref={fileRef} hidden type="file" accept="audio/*,.webm" onChange={(event) => uploadAudio(event.target.files?.[0])} />
            <select value={provider} onChange={(event) => setProvider(event.target.value)} aria-label="转写引擎"><option value="whisper">本地 Whisper</option><option value="dashscope">DashScope</option></select>
            <button className="primary-button" type="button" disabled={!audioId || busy} onClick={transcribeAudio}>{stage === 'transcribing' ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />} 转写</button>
          </div>
        </section>

        <section className="capture-transcript">
          <div className="section-heading"><Pause size={17} /><h2>转写文本</h2><span>{transcript.length} 字</span></div>
          <textarea value={transcript} onChange={(event) => {
            setTranscript(event.target.value)
            setSegments([])
          }} placeholder="转写内容会显示在这里，也可以直接粘贴课堂文字…" />
          <button className="primary-button align-end" type="button" disabled={!transcript.trim() || busy} onClick={generateSummary}>{stage === 'summarizing' ? <LoaderCircle size={16} className="spin" /> : <WandSparkles size={16} />} 整理为课堂笔记</button>
        </section>
      </div>

      <section className="summary-preview">
        <div className="section-heading"><Check size={17} /><h2>保存前预览</h2>{summary && <span>可编辑</span>}</div>
        {!summary ? <div className="preview-empty">完成转写与整理后，结构化笔记会出现在这里。</div> : (
          <div className="preview-split">
            <textarea value={summary} onChange={(event) => setSummary(event.target.value)} aria-label="课堂笔记 Markdown" />
            <div className="preview-render"><MarkdownView content={summary} /></div>
          </div>
        )}
        <button className="primary-button align-end" type="button" disabled={!summary.trim() || busy || stage === 'saved'} onClick={saveSummary}>{stage === 'saving' ? <LoaderCircle size={16} className="spin" /> : <Save size={16} />} {stage === 'saved' ? '已保存' : '确认并保存到 Wiki'}</button>
      </section>
    </div>
  )
}
