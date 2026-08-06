# 智语

智语是一个面向个人学习与面试展示的本地优先 AI Wiki。它把课堂录音、文档和零散笔记沉淀为可维护的 Markdown 知识库，并通过可追溯 RAG、LangGraph Agent 和确认式写入完成可信问答与持续整理。

项目关注的不是聊天界面本身，而是知识从采集到复用的完整生命周期：

```text
语音 / 文档采集 -> Wiki 主数据 -> 异步索引 -> 证据检索
-> Agent 推理 -> 证据门禁 -> 回答或受控外部研究 -> 确认入库
```

## 核心能力

- React Wiki 工作台：Markdown 页面、标签、笔记本、别名、Wiki Link 与反向链接。
- 本地优先数据模型：Markdown 保存当前正文，SQLite 保存元数据、版本、链接、任务与运行记录。
- RAG v2：父子分块、多查询统一 RRF、父块回填、单次 Reranker 和 Token 预算。
- 可信问答：保留 CRAG 纠错与证据门禁，证据不足时拒绝生成推测性答案。
- Agent 安全闭环：知识写入先预览、再确认；重复确认不会产生重复副作用。
- 可恢复运行时：单次真实流式生成、类型化事件、断线续传、停止、总超时和终态回放。
- 受控 MCP 研究：仅允许配置的 Search/Fetch 工具，公网来源经过校验并确认后写入 Wiki。
- 可靠索引：持久化索引任务、指数退避、失败重试与服务重启恢复。
- 模型网关：主备模型故障转移、Token/成本统计；流开始后禁止静默切换模型。
- 可观测性：请求 ID、执行时间线、检索统计、模型用量，以及可选 Langfuse/OpenTelemetry。
- 增量记忆：摘要只处理新增历史，原始对话始终保留。

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React 19、Vite 6、React Markdown、Lucide React |
| API | Python 3.11、FastAPI、Uvicorn、Pydantic |
| 数据 | SQLite、SQLAlchemy、UTF-8 Markdown |
| 检索 | BM25、BGE Embedding、RRF、BGE Reranker、ChromaDB |
| Agent | LangGraph 1.x、Plan-and-Execute、CRAG、MCP Python SDK |
| 模型 | OpenAI 兼容接口、主备模型网关、faster-whisper / DashScope |
| 观测 | Request ID、Server-Timing、Langfuse、OpenTelemetry |
| 工程 | Pytest、GitHub Actions、Docker 多阶段构建 |

数据库访问当前使用 SQLAlchemy 同步 Session；FastAPI 路由和后台 Worker 使用异步编排。这个选择适合单用户本地部署，也明确保留了未来迁移 PostgreSQL 与异步驱动的边界。

## 架构

```text
React 工作台
    |
    v
FastAPI + Request Context
    |
    +-- PageService -------- Markdown / Revision / Link / Index Task
    |                              |
    |                              v
    |                       Wiki Index Worker
    |                       Parent / Child Chunks
    |                       BM25 + ChromaDB
    |
    +-- Agent Runtime ----- Run State -> Typed Events -> SSE Resume
    |       |
    |       v
    |   LangGraph Agent --- Query Rewrite -> Unified RRF -> Reranker
    |                       -> Token Budget -> CRAG -> Evidence Gate
    |                       -> Single-pass LLM Stream
    |
    +-- MCP Research ------- Search / Fetch 白名单 -> 来源快照 -> 确认入库
    +-- Observability ------ Agent Run / Timeline / Usage / Langfuse / OTel
```

后端按“API 聚合、运行时、Agent 编排、执行调度、工具实现、领域服务”分层：Agent 路由按 Run、确认动作、外部研究、会话和检索拆分，`Executor` 只处理依赖图与执行控制，具体工具由可注入的注册表管理。前端问答页同样拆为页面编排、运行状态 Hook 和消息/MCP 展示组件，流式协议与视觉样式保持独立。

父块沿用稳定 ID，子块只负责提高召回粒度：

```text
page:{page_id}:revision:{revision}:chunk:{index}
page:{page_id}:revision:{revision}:chunk:{index}:child:{child_index}
```

命中子块后会先折叠回父块，再执行统一融合和一次精排。因此生成模型获得完整上下文，引用仍指向稳定父块。

## 数据一致性

```text
Markdown + SQLite = 需要备份的业务主数据
ChromaDB + BM25   = 可以重建的派生索引
```

页面保存成功后立即返回，索引任务由 Worker 异步处理。索引失败不会回滚或删除正文，而是持久化错误并退避重试。FastAPI `lifespan` 统一负责 schema 迁移、任务恢复、Worker 启停和退出清理。

当前 schema 版本为 `6`，迁移只新增字段或表，不删除已有数据。活动 Run 的事件保存在进程内，完成、失败、取消或超时后将状态与终态事件批量写入 SQLite；服务重启会明确收敛遗留 Run，不伪装成可继续执行。

## 快速开始

环境要求：Python 3.11+、Node.js 20+、ffmpeg，以及本地 Embedding/Reranker 模型或相应服务配置。

```bash
conda create -n zhiyu python=3.11 -y
conda activate zhiyu
conda install -c conda-forge ffmpeg -y
pip install -r requirements.txt

cd frontend
npm install
npm run build
cd ..

cp .env.example .env
python main.py
```

应用默认运行在 `http://127.0.0.1:8337`。开发前端可在 `frontend` 目录运行 `npm run dev`。

最小模型配置：

```env
EMBEDDING_MODEL_PATH=/absolute/path/to/bge-model
RERANKER_MODEL_PATH=/absolute/path/to/reranker-model
LLM_API_KEY=your_api_key
LLM_API_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

RAG v2、模型故障转移、MCP、Langfuse 与 OpenTelemetry 均有独立开关，完整配置见 [.env.example](.env.example)。关闭 `RAG_V2_ENABLED` 可回退到原有逐查询检索链路。

从旧版本升级并启用父子分块后，应在 Wiki 工作台执行一次“重建索引”，让已有页面生成子块；页面正文和历史版本不会被改写。

## 验证

```bash
python -m pytest -q
python -m pip check

cd frontend
npm run build
npm audit --omit=dev
```

测试覆盖页面版本冲突、索引失败重试、Agent 事件顺序与断线回放、会话并发冲突、取消、超时、服务重启恢复、重复确认、父子分块、统一融合、Token 预算、模型故障转移、增量摘要、MCP 安全边界、备份恢复和音频溯源。

## 安全边界

- 写操作必须经过确认；外部研究不会自动写入知识库。
- MCP 只调用部署时配置的 Search/Fetch 工具，不接受运行时任意工具名。
- MCP 子进程只接收显式配置的环境变量；状态面不返回命令、参数或密钥。
- 外部 URL 拒绝本机、私网、保留地址、嵌入凭证和非 HTTP(S) 协议。
- Langfuse/OTel 默认关闭，默认不采集提示词和答案正文；导出失败不影响请求。
- 上传、音频访问和备份恢复均进行路径边界校验。

## 项目边界

- 当前面向单用户、本机或可信内网，不包含多租户认证与细粒度权限系统。
- SQLite 和进程内 Worker 适合个人项目规模；高并发部署应迁移 PostgreSQL、异步数据库驱动和独立任务队列。
- Agent 活动事件只保存在当前进程，适合单实例部署；多实例需要共享运行状态和事件总线。
- ChromaDB 采用嵌入式持久化客户端，不对外开放独立服务端口。
- 本地 Embedding、Reranker 和 Whisper 需要有效模型目录；缺失时页面管理仍可用，相关能力会降级。

## 文档

- [系统架构](docs/architecture.md)
- [与最早 main 的演进对比](docs/project-evolution.md)
- [Markdown 主数据决策](docs/adr/0001-markdown-source-of-truth.md)
- [持久化索引任务决策](docs/adr/0002-persistent-index-tasks.md)
- [可信 Agent 决策](docs/adr/0003-trusted-agent.md)
- [受控 MCP 研究决策](docs/adr/0004-controlled-mcp-research.md)

## License

MIT License
