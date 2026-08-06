export async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json'
  }
  const response = await fetch(path, { ...options, headers })
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text()
  if (!response.ok) {
    const detail = typeof payload === 'object' ? payload.detail : payload
    throw new Error(detail || `请求失败（${response.status}）`)
  }
  return payload
}

export async function streamSse(path, options = {}, onEvent = () => {}, onOpen = () => {}) {
  const headers = { ...(options.headers || {}) }
  if (options.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json'
  const response = await fetch(path, { ...options, headers })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail || `请求失败（${response.status}）`)
  }
  if (!response.body) throw new Error('浏览器不支持流式响应')
  onOpen(response)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  function processBlock(rawBlock) {
    const block = rawBlock.replaceAll('\r', '')
    if (!block || block.startsWith(':')) return
    const dataLines = []
    let eventType = 'message'
    let eventId = ''
    block.split('\n').forEach((line) => {
      if (line.startsWith('event:')) eventType = line.slice(6).trim()
      if (line.startsWith('id:')) eventId = line.slice(3).trim()
      if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
    })
    if (!dataLines.length) return
    const payload = JSON.parse(dataLines.join('\n'))
    onEvent(payload, { eventType, eventId })
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      processBlock(buffer.slice(0, boundary))
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')
    }
    if (done) break
  }
  if (buffer.trim()) processBlock(buffer)
}

export async function download(path, filename) {
  const response = await fetch(path)
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail || `下载失败（${response.status}）`)
  }
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}
