import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { LoaderCircle, RefreshCw, SlidersHorizontal } from 'lucide-react'
import { api } from '../../shared/api/client'

// 节点颜色按 notebook 分。现在只有两个笔记本，再加的话用哈希兜底，
// 免得新笔记本全是一个颜色分不出来。
const NOTEBOOK_COLORS = ['#2f7f69', '#a8622c', '#4a6fa5', '#8a5a94', '#7a7a3f']

function colorForNode(node, notebooks) {
  const index = notebooks.indexOf(node.notebook)
  return NOTEBOOK_COLORS[(index < 0 ? notebooks.length : index) % NOTEBOOK_COLORS.length]
}

export default function GraphWorkspace({ notify, openPage }) {
  const [graph, setGraph] = useState(null)
  const [loading, setLoading] = useState(true)
  // 阈值由服务端给默认值，用户拖滑块后覆盖。语料一变合适的值就变，
  // 做成可调比在代码里猜一个数可靠。
  const [threshold, setThreshold] = useState(null)
  const [applied, setApplied] = useState(null)
  const graphRef = useRef(null)
  // 画布尺寸靠 ResizeObserver 跟着容器走，写死宽高在窄屏会被裁掉。
  const wrapperRef = useRef(null)
  const [size, setSize] = useState({ width: 800, height: 600 })

  const load = useCallback(async (value) => {
    setLoading(true)
    try {
      const query = value === null || value === undefined ? '' : `?threshold=${value}`
      const data = await api(`/api/graph${query}`)
      setGraph(data)
      setThreshold((previous) => (previous === null ? data.threshold : previous))
      setApplied(data.threshold)
    } catch (error) {
      notify(error.message, 'error')
    } finally {
      setLoading(false)
    }
  }, [notify])

  useEffect(() => { load(null) }, [load])

  useEffect(() => {
    const element = wrapperRef.current
    if (!element) return undefined
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect
      if (box) setSize({ width: box.width, height: box.height })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  // 力导向图的节点对象是原地被引擎改写的（x/y/vx/vy），不能直接把它喂给
  // 渲染——数据变了但引用没变，React 不会重画。这里每次拿到新数据都造一份。
  const data = useMemo(() => {
    if (!graph) return { nodes: [], links: [] }
    return {
      nodes: graph.nodes.map((node) => ({ ...node })),
      links: graph.edges.map((edge) => ({
        source: edge.source,
        target: edge.target,
        type: edge.type,
        weight: edge.weight,
      })),
    }
  }, [graph])

  const notebooks = useMemo(
    () => [...new Set((graph?.nodes || []).map((node) => node.notebook))].sort(),
    [graph],
  )
  const maxDegree = useMemo(
    () => Math.max(1, ...(graph?.nodes || []).map((node) => node.degree)),
    [graph],
  )

  // 节点半径按度数缩放。开方是为了压一压差距——线性缩放会让高度数节点
  // 大到盖住邻居，低度数的又小得看不见。
  const radiusOf = useCallback(
    (node) => 3 + 9 * Math.sqrt(node.degree / maxDegree),
    [maxDegree],
  )

  const drawNode = useCallback((node, ctx, globalScale) => {
    const radius = radiusOf(node)
    const color = colorForNode(node, notebooks)
    ctx.beginPath()
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI)
    // 没索引的页面没有向量，画成空心：能看出它在那儿，也知道它还没连上。
    if (node.indexed) {
      ctx.fillStyle = color
      ctx.fill()
    } else {
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    // 缩得太小的时候不画标题，否则字全糊成一团。
    if (globalScale > 1.2) {
      ctx.font = `${11 / globalScale}px sans-serif`
      ctx.fillStyle = '#3b4642'
      ctx.textAlign = 'center'
      ctx.fillText(node.title.slice(0, 18), node.x, node.y + radius + 10 / globalScale)
    }
  }, [notebooks, radiusOf])

  return (
    <div className="graph-workspace">
      <header className="workspace-titlebar">
        <div>
          <span className="eyebrow">知识图谱</span>
          <h1>页面之间的语义关联</h1>
        </div>
        <div className="graph-actions">
          <button className="secondary-button" type="button" disabled={loading} onClick={() => load(applied)}>
            <RefreshCw size={15} /> 刷新
          </button>
        </div>
      </header>

      <div className="graph-toolbar">
        <SlidersHorizontal size={15} />
        <label>
          相似度门槛
          <input
            type="range"
            min="0.6"
            max="0.95"
            step="0.01"
            value={threshold ?? 0.77}
            onChange={(event) => setThreshold(Number(event.target.value))}
            onMouseUp={() => load(threshold)}
            onTouchEnd={() => load(threshold)}
            disabled={loading}
          />
          <strong>{(threshold ?? 0.77).toFixed(2)}</strong>
        </label>
        <span className="graph-legend">
          <i className="legend-dot" /> 节点大小 = 关联数
          <i className="legend-line" /> 连线 = 语义相近
          <i className="legend-ring" /> 空心 = 尚未索引
        </span>
        {graph && (
          <span className="graph-counts">
            {graph.nodes.length} 个页面 · {graph.edges.length} 条关联
          </span>
        )}
      </div>

      <div className="graph-canvas" ref={wrapperRef}>
        {loading && <div className="graph-loading"><LoaderCircle size={18} className="spin" /> 正在计算页面关联…</div>}
        {!loading && graph?.nodes.length === 0 && (
          <div className="graph-loading">知识库还没有页面。</div>
        )}
        {!loading && graph && graph.nodes.length > 0 && (
          <ForceGraph2D
            ref={graphRef}
            graphData={data}
            width={size.width}
            height={size.height}
            nodeRelSize={1}
            nodeCanvasObject={drawNode}
            nodePointerAreaPaint={(node, color, ctx) => {
              // 命中区域比画出来的圆大一点，不然小节点很难点中。
              ctx.beginPath()
              ctx.arc(node.x, node.y, radiusOf(node) + 4, 0, 2 * Math.PI)
              ctx.fillStyle = color
              ctx.fill()
            }}
            linkColor={(link) => (link.type === 'link' ? '#2f7f69' : 'rgba(120,132,126,0.42)')}
            linkLineDash={(link) => (link.type === 'link' ? [] : [3, 3])}
            linkWidth={(link) => (link.type === 'link' ? 1.6 : 0.6 + link.weight * 1.2)}
            onNodeClick={(node) => openPage?.(node.id)}
            cooldownTicks={120}
          />
        )}
      </div>
    </div>
  )
}
