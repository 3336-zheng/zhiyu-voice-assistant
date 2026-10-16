/**
 * 校验「页面目录」与正文标题一一对应。
 *
 * 目录项（WikiWorkspace.headingsFrom）和正文标题（MarkdownView 渲染出的 h1~h4）
 * 是两套独立解析：前者是逐行正则，后者是 ReactMarkdown。两边对不上时，目录项
 * 的锚点会指向不存在的 id——表现是「有些标题点了能跳，有些点了没反应」，而且
 * 不报错。
 *
 * 实际踩过一次：headingsFrom 没跳过围栏代码块，把代码里的 `# 注释` 当成了标题。
 * 25 个页面里 12 个错位，某个页面目录列 34 条、正文只有 26 个标题。
 *
 * 两层检查：
 *   1. 内置用例——不依赖任何外部数据，CI 里跑的就是它。覆盖围栏代码块、波浪号
 *      围栏、嵌套代码块里的标题等已知坑。
 *   2. 真实语料——data/wiki/pages 存在时才跑（它被 gitignore，CI 上没有）。
 *
 * 纯前端逻辑没法用后端的 pytest 覆盖，所以单独放在这里，由
 * `npm run check:headings` 调用。
 */
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { gfm } from 'micromark-extension-gfm'
import { gfmFromMarkdown } from 'mdast-util-gfm'

const here = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(here, '..')
const pagesDir = resolve(frontendRoot, '../data/wiki/pages')

/** 从 WikiWorkspace.jsx 里抽出 headingsFrom，保证测的是真实现而不是重写的副本。 */
function loadHeadingsFrom() {
  const source = readFileSync(
    join(frontendRoot, 'src/features/wiki/WikiWorkspace.jsx'),
    'utf8',
  )
  const start = source.indexOf('function headingsFrom(content) {')
  if (start < 0) throw new Error('WikiWorkspace.jsx 里找不到 headingsFrom')
  // 按花括号配平截出整个函数体。
  let depth = 0
  for (let i = source.indexOf('{', start); i < source.length; i += 1) {
    if (source[i] === '{') depth += 1
    else if (source[i] === '}') {
      depth -= 1
      if (depth === 0) {
        return new Function(`${source.slice(start, i + 1)}; return headingsFrom`)()
      }
    }
  }
  throw new Error('headingsFrom 括号不配平')
}

/** 用真解析器取正文标题的文本序列。 */
function renderedHeadings(markdown) {
  const tree = fromMarkdown(markdown, {
    extensions: [gfm()],
    mdastExtensions: [gfmFromMarkdown()],
  })
  const titles = []
  const walk = (node) => {
    if (node.type === 'heading' && node.depth <= 4) {
      titles.push((node.children || []).map((child) => child.value ?? '').join(''))
    }
    for (const child of node.children || []) walk(child)
  }
  walk(tree)
  return titles
}

/** 剥掉 YAML front matter——接口返回的 content 不含它，比对要同口径。 */
function stripFrontMatter(text) {
  if (!text.trimStart().startsWith('---')) return text
  const parts = text.split('---')
  return parts.length >= 3 ? parts.slice(2).join('---') : text
}

// 源码标题里的 `code` 在渲染后是不带反引号的纯文本，比对前统一去掉。
const normalize = (title) => title.replace(/`/g, '')

function compare(markdown) {
  const headingsFrom = loadHeadingsFrom()
  const fromSource = headingsFrom(markdown).map((item) => normalize(item.title))
  const fromRendered = renderedHeadings(markdown).map(normalize)
  const same = fromSource.length === fromRendered.length
    && fromSource.every((title, index) => title === fromRendered[index])
  if (same) return null
  const firstDiff = fromSource.findIndex((title, i) => title !== fromRendered[i])
  return {
    sourceCount: fromSource.length,
    renderedCount: fromRendered.length,
    firstDiff: firstDiff < 0 ? fromSource.length : firstDiff,
    sourceTitle: fromSource[firstDiff],
    renderedTitle: fromRendered[firstDiff],
  }
}

// 内置用例：每条的「期望正文标题」就是它该解析出来的结果。
const FIXTURES = [
  {
    name: '代码块里的 # 注释不算标题',
    markdown: [
      '# 真标题',
      '',
      '```python',
      '# 这是注释，不是标题',
      'def f(): pass',
      '```',
      '',
      '## 第二个真标题',
    ].join('\n'),
    expected: ['真标题', '第二个真标题'],
  },
  {
    name: '波浪号围栏同样要跳过',
    markdown: ['# A', '~~~', '# 注释', '~~~', '# B'].join('\n'),
    expected: ['A', 'B'],
  },
  {
    name: '围栏内的井号层级不干扰编号',
    markdown: [
      '## 一',
      '```bash',
      '# 注释1',
      '## 注释2',
      '```',
      '## 二',
    ].join('\n'),
    expected: ['一', '二'],
  },
  {
    name: '普通正文里的标题正常识别',
    markdown: ['# 一级', '正文', '## 二级', '### 三级', '#### 四级'].join('\n'),
    expected: ['一级', '二级', '三级', '四级'],
  },
  {
    name: '未闭合的围栏会吃掉后面的标题（markdown 的实际行为）',
    markdown: ['# 可见', '```', '# 在围栏里', '## 也在围栏里'].join('\n'),
    expected: ['可见'],
  },
]

function main() {
  let failed = 0

  console.log('内置用例：')
  for (const fixture of FIXTURES) {
    const diff = compare(fixture.markdown)
    const rendered = renderedHeadings(fixture.markdown).map(normalize)
    const expectedOk = rendered.length === fixture.expected.length
      && rendered.every((title, i) => title === fixture.expected[i])
    if (!diff && expectedOk) {
      console.log(`  ✓ ${fixture.name}`)
    } else {
      failed += 1
      console.error(`  ✗ ${fixture.name}`)
      if (!expectedOk) {
        console.error(`      markdown 解析结果与预期不符：${JSON.stringify(rendered)}`)
      } else {
        console.error(`      目录与正文不一致：目录 ${diff.sourceCount} vs 正文 ${diff.renderedCount}`)
      }
    }
  }

  if (existsSync(pagesDir)) {
    const files = readdirSync(pagesDir).filter((name) => name.endsWith('.md'))
    const mismatches = []
    for (const name of files) {
      const diff = compare(stripFrontMatter(readFileSync(join(pagesDir, name), 'utf8')))
      if (diff) mismatches.push({ name, ...diff })
    }
    console.log(`\n真实语料（${pagesDir}）：`)
    if (mismatches.length === 0) {
      console.log(`  ✓ ${files.length} 个页面全部一致`)
    } else {
      failed += mismatches.length
      console.error(`  ✗ ${mismatches.length}/${files.length} 个页面不一致`)
      for (const item of mismatches) {
        console.error(`      ${item.name}：目录 ${item.sourceCount} vs 正文 ${item.renderedCount}`)
        console.error(`        第 ${item.firstDiff + 1} 个起错位：目录=${JSON.stringify(item.sourceTitle)} 正文=${JSON.stringify(item.renderedTitle)}`)
      }
    }
  } else {
    console.log(`\n真实语料：跳过（${pagesDir} 不存在，它被 gitignore）`)
  }

  return failed === 0 ? 0 : 1
}

process.exit(main())

