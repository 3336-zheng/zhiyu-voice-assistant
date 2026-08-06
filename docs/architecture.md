# 智语架构说明

## 目标与边界

智语面向单用户、本地优先的课堂知识管理场景。系统优先保证知识可读、可迁移和可恢复；模型、向量索引和后台任务均不得成为页面写入的单点故障。

当前不解决多租户、高并发写入和跨节点任务调度。需要这些能力时，可将 SQLite、进程内 Worker 和本地文件存储分别迁移到 PostgreSQL、独立任务队列和对象存储。

## 模块结构

```mermaid
flowchart LR
    UI["React 工作台"] --> API["FastAPI API"]
    API --> PAGE["PageService 门面"]
    PAGE --> FILE["WikiFileStore"]
    PAGE --> REV["WikiRevisionService"]
    PAGE --> LINK["WikiLinkService"]
    PAGE --> TASK["WikiIndexTaskService"]
    FILE --> MD["Markdown 主数据"]
    REV --> DB["SQLite / SQLAlchemy"]
    LINK --> DB
    TASK --> DB
    TASK --> WORKER["Wiki Index Worker"]
    WORKER --> SEARCH["BM25 / ChromaDB"]
    API --> RUNTIME["Agent Runtime"]
    RUNTIME --> AGENT["LangGraph Agent"]
    RUNTIME --> DB
    AGENT --> RETRIEVAL["RAG v2 与证据门禁"]
    RETRIEVAL --> SEARCH
    AGENT --> GATEWAY["主备模型网关"]
    GATEWAY --> LLM["OpenAI 兼容模型"]
    AGENT --> RESEARCH["外部研究编排"]
    RESEARCH --> MCP["可信 stdio MCP Server"]
    MCP --> WEB["公开资料"]
    RESEARCH --> DB
    RESEARCH --> PAGE
    API --> AUDIO["Whisper / DashScope ASR"]
    AUDIO --> DB
    API --> OBS["请求时间线与运行统计"]
    OBS --> DB
    OBS -.-> TRACE["Langfuse / OpenTelemetry"]
```

`PageService` 只负责编排跨资源事务和提供稳定 API；文件、版本、链接和索引任务由独立服务承担。ChromaDB 和 BM25 均为派生数据，可以从 Markdown 与 SQLite 元数据重建。

核心目录遵循单向依赖和稳定聚合入口：

| 模块 | 职责 | 依赖边界 |
| --- | --- | --- |
| `backend/app/api/agent.py` | 聚合 Agent 子路由 | 不承载业务流程 |
| `backend/app/api/agent_*` | Run、确认、研究、会话和检索协议 | 调用 Agent 与领域服务 |
| `backend/app/services/agent_runtime_service.py` | Run 状态、事件、会话锁和终态持久化 | 不实现具体 Agent 工具 |
| `backend/app/agent/graph.py` | LangGraph 节点与状态转换 | 编排检索、证据和生成能力 |
| `backend/app/agent/executor.py` | 依赖图、并发波次、取消和结果聚合 | 通过工具注册表执行步骤 |
| `backend/app/agent/tool_registry.py` | 工具映射与 Wiki、检索、摘要工具 | 通过工厂延迟加载外部依赖 |
| `backend/app/services/hybrid_retrieval_service.py` | 召回、融合、精排与上下文装配 | 构造参数可注入，默认使用生产单例 |
| `frontend/app/src/hooks/useAgentRun.js` | SSE、续传、停止、会话与确认状态 | 不负责页面布局 |
| `frontend/app/src/components/AgentMessage.jsx` | 回答、引用、研究和运行详情展示 | 不直接发起 API 请求 |
| `frontend/app/src/views/AskWorkspace.jsx` | 页面编排与输入交互 | 组合 Hook 和展示组件 |

这种拆分保留了原有导入入口与 URL，便于渐进迁移；测试可通过构造参数注入替身，不再依赖 `__new__` 或全局单例内部状态。当前 Agent 的 Planner、Graph 和 Responder 仍属于复杂度较高的核心模块，后续只有在业务规则继续增长时才值得进一步按节点或策略拆分。

## 页面写入

1. 校验标题、正文、版本号和页面 ID。
2. 原子写入带 YAML Front Matter 的 Markdown。
3. 在同一数据库事务中保存元数据、完整版本快照和索引任务。
4. 重建 Wiki Link 与反向链接。
5. API 立即返回，后台 Worker 异步更新 BM25 和 ChromaDB。

数据库提交失败时恢复原 Markdown；模型不可用时保留主数据并将任务置为 `failed`，按指数退避重试。

## 可信问答

```text
Plan -> Query Rewrite -> 多查询 BM25 / Embedding 召回
     -> 子块折叠为父块 -> 单次 RRF -> 单次 Reranker
     -> Token Budget -> CRAG Grade -> Evidence Gate -> Generate
```

索引同时保存稳定父块和细粒度子块。子块参与召回，命中后折叠回父块；不同改写查询的稀疏、稠密结果全部收集后只执行一次统一融合和一次精排，避免旧链路为每个查询重复加载候选和精排。最终上下文按 Token 预算装配，最后一个父块允许截断，但来源标识不会丢失。

`RAG_V2_ENABLED` 是总回退开关，关闭后恢复原逐查询完整检索；`RAG_PARENT_CHILD_ENABLED` 可以独立关闭父子分块。CRAG、证据门禁、稳定引用和 MCP 确认流程不受开关影响。

证据门禁在生成前执行。无结果、来源不足或分数低于阈值时返回结构化拒答，不调用 LLM 生成推测性答案。每个来源保留页面、版本、章节、稳定 Chunk ID；课堂音频页面额外返回转写时间范围和媒体链接。

## 外部研究

外部研究是证据门禁后的显式分支，不是检索图的自动回退。用户触发后，模型生成有限数量的检索词，MCP 客户端只连接配置的 stdio Server，并只调用预先声明的搜索和抓取工具。

```text
本地证据不足 -> 用户触发 -> 查询生成 -> MCP 搜索/抓取
-> URL 与内容校验 -> 带引用回答和 Wiki 草稿
-> 用户确认 -> PageService -> 异步索引
```

外部资料经过以下边界后才进入模型：

1. URL 仅允许 HTTP(S)，拒绝凭证、本机、私网、链路本地和保留地址。
2. 来源按规范化 URL 和正文哈希去重，并限制来源数与正文总量。
3. 外部正文被转义并标记为不可信证据，模型不得执行其中的指令。
4. 研究任务和来源快照先写入 SQLite；研究结果不会直接创建页面。
5. 用户确认后，页面通过 `PageService` 写入，并由 `wiki_page_sources` 建立来源关系。

MCP Server 是部署信任边界。应用不会向子进程传递完整环境变量，也不会调用运行时返回的任意工具；Server 自身仍需负责 HTTP 重定向、DNS 重绑定和网络出口控制。

## Agent 写入

写操作采用两阶段协议：首次请求只持久化计划和预览，确认接口才执行工具。完成结果写回 `agent_pending_actions`；重复确认直接返回首次结果，避免重复副作用。

## Agent 运行时

每次流式请求只创建一个模型生成过程。模型流片段同时用于前端输出和最终回答组装，不再为了流式展示重复调用模型。事件统一包含 Run ID、会话 ID、严格递增的序号、时间和类型，覆盖阶段、工具、Token 与终态。

同一会话只允许一个活动 Run。浏览器断线后可从最后事件序号继续订阅；取消使用协作式信号，总超时负责停止后续编排并将状态收敛为 `timed_out`。工具结果的原始值保留给业务与审计，进入模型上下文和步骤引用前才按 Token 预算截断。

活动事件保存在进程内，避免为每个 Token 写 SQLite。Run 进入完成、失败、取消或超时后，运行状态和非 Token 事件批量持久化，终态事件包含完整回答，因此重连不依赖逐 Token 数据。服务重启时，遗留的 `pending`、`running` 和 `cancelling` Run 会标记为失败并补写可回放的终态事件。

当前实现定位于单进程本地部署。Python 线程中的外部 SDK 调用无法被强制终止，只能在调用返回后响应取消信号；总超时保证应用状态收敛和后续步骤停止，不等同于操作系统级线程终止。多实例部署需要将活动状态、事件流和会话锁迁移到共享基础设施。

## 生命周期与可观测性

FastAPI `lifespan` 统一执行 schema 迁移、Agent Run 与索引任务恢复、Worker 启动和退出清理。每个 HTTP 请求生成或继承 `X-Request-ID`，响应返回 `Server-Timing`、Agent 执行时间线、检索统计和模型用量。相同数据脱敏后写入 `agent_runs`，用于复盘单次请求。

LLM 网关记录主模型和备用模型的 Token、耗时与估算成本。仅连接错误、超时、429 和 5xx 可以触发故障转移；流式输出一旦开始就不再切换，避免把两个模型的内容拼接成同一回答。Langfuse 和 OpenTelemetry 默认关闭，正文采集默认关闭，导出异常不会影响请求。

## 对话记忆

会话摘要使用 `summary_message_id` 作为增量游标。每次只摘要尚未覆盖且不属于最近窗口的消息，旧摘要作为输入继续更新；原始 `conversation_messages` 不删除。模型上下文由“累计摘要 + 摘要游标之后的最近消息”组成，兼顾上下文预算和审计完整性。

## 关键决策

- [ADR-0001：Markdown 作为知识主数据](adr/0001-markdown-source-of-truth.md)
- [ADR-0002：持久化异步索引任务](adr/0002-persistent-index-tasks.md)
- [ADR-0003：证据门禁与确认式 Agent](adr/0003-trusted-agent.md)
- [ADR-0004：受控 MCP 外部研究](adr/0004-controlled-mcp-research.md)
