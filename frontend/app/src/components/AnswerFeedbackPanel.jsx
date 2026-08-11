import {
  Check,
  CircleAlert,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  X,
} from 'lucide-react'
import MarkdownView from './MarkdownView'

const progressLabels = {
  reported: '反馈已保存',
  researching: '正在核验外部资料',
  writing: '正在写入知识库',
  indexing: '正在更新检索索引',
  retesting: '正在使用原问题复测',
}

const failedStatuses = new Set(['draft_failed', 'write_failed', 'index_failed', 'retest_failed'])

export default function AnswerFeedbackPanel({ feedback, onCancel, onConfirm, onRetry }) {
  if (!feedback) return null

  if (progressLabels[feedback.status]) {
    return (
      <section className="answer-feedback-panel" aria-live="polite">
        <div className="feedback-status-line">
          <LoaderCircle size={16} className="spin" />
          <strong>{progressLabels[feedback.status]}</strong>
        </div>
      </section>
    )
  }

  if (feedback.status === 'pending_confirmation') {
    return (
      <section className="answer-feedback-panel">
        <div className="feedback-panel-heading">
          <div><span>纠错草稿</span><strong>{feedback.draft?.title}</strong></div>
          <small>{feedback.category_label}</small>
        </div>
        <details className="feedback-draft">
          <summary>查看草稿正文</summary>
          <MarkdownView content={feedback.draft?.content || ''} />
        </details>
        <div className="action-buttons">
          <button className="primary-button" type="button" onClick={onConfirm}><Check size={16} /> 确认写入并复测</button>
          <button className="secondary-button" type="button" onClick={onCancel}><X size={16} /> 取消</button>
        </div>
      </section>
    )
  }

  if (feedback.status === 'resolved') {
    return (
      <section className="answer-feedback-panel resolved">
        <div className="feedback-panel-heading">
          <div><span>自动复测完成</span><strong>{feedback.category_label}</strong></div>
          <small><Check size={13} /> 已重新索引</small>
        </div>
        <div className="feedback-comparison">
          <div><span>原回答</span><MarkdownView content={feedback.before?.answer || ''} /></div>
          <div><span>复测回答</span><MarkdownView content={feedback.retest?.answer || ''} /></div>
        </div>
        {feedback.retest?.request_id && <code className="feedback-request-id">{feedback.retest.request_id}</code>}
      </section>
    )
  }

  if (failedStatuses.has(feedback.status)) {
    const retryable = ['index_failed', 'retest_failed'].includes(feedback.status)
    return (
      <section className="answer-feedback-panel failed" role="status">
        <div className="feedback-status-line"><CircleAlert size={16} /><strong>{feedback.error || '纠错流程执行失败'}</strong></div>
        {retryable && <button className="secondary-button" type="button" onClick={onRetry}><RefreshCw size={15} /> 重试失败阶段</button>}
      </section>
    )
  }

  if (feedback.status === 'cancelled') {
    return (
      <section className="answer-feedback-panel cancelled">
        <div className="feedback-status-line"><RotateCcw size={15} /><strong>纠错草稿已取消</strong></div>
      </section>
    )
  }

  return null
}
