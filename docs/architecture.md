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
    API --> AGENT["LangGraph Agent"]
    AGENT --> RETRIEVAL["混合检索与证据门禁"]
    RETRIEVAL --> SEARCH
    AGENT --> RESEARCH["外部研究编排"]
    RESEARCH --> MCP["可信 stdio MCP Server"]
    MCP --> WEB["公开资料"]
    RESEARCH --> DB
    RESEARCH --> PAGE
    API --> AUDIO["Whisper / DashScope ASR"]
    AUDIO --> DB
```

`PageService` 只负责编排跨资源事务和提供稳定 API；文件、版本、链接和索引任务由独立服务承担。ChromaDB 和 BM25 均为派生数据，可以从 Markdown 与 SQLite 元数据重建。

## 页面写入

1. 校验标题、正文、版本号和页面 ID。
2. 原子写入带 YAML Front Matter 的 Markdown。
3. 在同一数据库事务中保存元数据、完整版本快照和索引任务。
4. 重建 Wiki Link 与反向链接。
5. API 立即返回，后台 Worker 异步更新 BM25 和 ChromaDB。

数据库提交失败时恢复原 Markdown；模型不可用时保留主数据并将任务置为 `failed`，按指数退避重试。

## 可信问答

```text
Plan → Query Rewrite → BM25 / Embedding → RRF → Reranker
     → CRAG Grade → Evidence Gate → Generate
```

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

## 生命周期与可观测性

FastAPI `lifespan` 统一执行 schema 迁移、索引恢复、Worker 启动和退出清理。每个 HTTP 请求生成或继承 `X-Request-ID`，最终日志记录查询改写、召回、融合、精排、证据判断和生成耗时；响应同时返回 `Server-Timing`。

## 关键决策

- [ADR-0001：Markdown 作为知识主数据](adr/0001-markdown-source-of-truth.md)
- [ADR-0002：持久化异步索引任务](adr/0002-persistent-index-tasks.md)
- [ADR-0003：证据门禁与确认式 Agent](adr/0003-trusted-agent.md)
- [ADR-0004：受控 MCP 外部研究](adr/0004-controlled-mcp-research.md)
