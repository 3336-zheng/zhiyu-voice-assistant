# 内置只读 MCP Server 与模型并发闸门设计

日期：2026-10-17
状态：待实现

## 1. 意图与背景

智语目前已经具备完整的 RAG 问答、Agent 与 Wiki 能力，但这两项能力只在应用内部闭环：

- 外部 AI 工具（Claude Code、Cursor 等）无法直接检索智语积累的知识；
- API、索引、转写、Agent 四个进程各自调用模型网关，进程之间互不知道对方的并发量，
  高峰期可能同时打出超过供应商限制的请求，触发 429 并让用户看到「模型调用失败」。

本次做两件事：

1. 增加一个**只读 MCP Server**，把知识库以标准 MCP 工具暴露给外部 AI 客户端。
2. 增加一个**Redis 分布式模型并发闸门**，按模型维度限制所有进程合计的并发调用数。

Lite 零依赖模式经评估不做：当前 MySQL + Redis 架构在并发写入、任务持久化和
崩溃恢复上明显强于 SQLite + 进程内队列，资源开销（约 300–500 MB）是必要代价。

## 2. 目标

- 外部 MCP 客户端能通过三个工具检索与读取智语知识库。
- MCP 工具严格只读，不产生任何页面写入或索引写入。`search_wiki` 复用在线混合检索，
  因此会将查询文本发送到配置的 Embedding / Rerank 端点；`read_page`、`list_pages`
  不产生外部请求。
- LLM、Embedding、Rerank 三类在线模型调用受同一套跨进程并发上限约束。
- Redis 不可用时闸门降级放行，不阻断问答与索引。
- 现有 API、Agent、Worker 行为在默认配置下保持不变。

## 3. 非目标

- 不支持 MCP 写操作（创建、修改、删除页面）。
- 不支持多租户、按用户隔离的知识库范围。
- 不做 HTTP/SSE 形式的 MCP 端点（只做 stdio，见 5.1）。
- 不做跨模型的自适应限流（例如根据 429 动态调低上限）。
- 不改动现有鉴权、CORS 或前端。
- 不把并发闸门扩展到非模型外部调用（如 MCP 外部研究、ASR）。

## 4. 约束

- Python 3.11、MCP SDK 2.x（已安装 2.0.0）、redis-py 5.x。
- 项目使用同步 SQLAlchemy Session；MCP 工具是异步入口，数据库操作必须放进线程。
- 后台 Worker 与 API 都通过 `get_*_service()` 单例调用模型，闸门应挂在这些单例的
  调用路径上，而不是散落到各业务模块。
- 测试使用真实 MySQL 的独立测试库；闸门测试使用假 Redis 客户端，不依赖真实 Redis。
- 新增配置必须有安全默认值，且写进 `.env.example`。

## 5. 设计

### 5.1 MCP 传输方式

**选择 stdio，不做 HTTP。**

理由：

- 智语定位是单用户、本地优先，stdio 不需要额外监听端口、鉴权或 TLS。
- Claude Code / Cursor 原生支持 stdio，配置只需一条命令。
- HTTP 挂载需要处理 MCP SDK 的 SessionManager 生命周期、反向代理和 Token 校验，
  收益与复杂度不成比例。

启动方式：

```bash
python -m backend.app.mcp_server
```

客户端配置示例（在启动参数里指向项目根目录）：

```json
{
  "mcpServers": {
    "zhiyu": {
      "command": "<conda 环境 python 路径>",
      "args": ["-m", "backend.app.mcp_server"],
      "cwd": "<项目根目录>"
    }
  }
}
```

MCP 进程直接读取项目 `.env` 并连接 MySQL / Chroma。它不启动 FastAPI，也不消费队列。

### 5.2 工具契约

三个工具全部只读，工具名与参数：

| 工具 | 参数 | 返回 |
| --- | --- | --- |
| `search_wiki` | `query: str`，`top_k: int = 5` | 混合检索结果列表：`page_id`、`title`、`section_title`、`content`、`score`、`source_url`、`tags` |
| `read_page` | `page_id_or_title: str` | 页面元数据与完整 Markdown 正文 |
| `list_pages` | `notebook?: str`、`tag?: str`、`query?: str`、`limit: int = 20` | 页面摘要列表与总数 |

行为约定：

- `search_wiki` 调用现有 `HybridRetrievalService.search_hybrid`（BM25 + 向量 + RRF），
  不触发 Query Rewrite / CRAG / LLM，避免外部工具一次检索就产生模型费用。
- `read_page` 调用 `PageService.find_page`，支持 UUID、`UUID.md`、标题和别名。
- `list_pages` 调用 `PageService.list_pages`，只返回摘要，不返回正文。
- 返回正文长度受 `MCP_SERVER_MAX_CONTENT_CHARS` 限制（默认 8000），超出截断并附
  `truncated: true`。
- 所有工具捕获领域异常并返回 MCP 工具错误文本，不把堆栈或数据库连接信息暴露给客户端。

### 5.3 模块结构

```
backend/app/mcp_server/
├── __init__.py
├── __main__.py      # python -m backend.app.mcp_server 入口，run("stdio")
├── server.py        # 构建 MCPServer，注册工具
└── tools.py         # WikiMCPTools：注入式服务封装，便于单测
```

`tools.py` 不直接依赖全局单例，而是接收两个工厂函数：

```python
class WikiMCPTools:
    def __init__(
        self,
        *,
        page_service_factory: Callable[[Session], PageService] = get_page_service,
        retrieval_factory: Callable[[], HybridRetrievalService] = get_hybrid_retrieval_service,
        session_factory: Callable[[], Session] = SessionLocal,
    ): ...
```

每个工具方法内部用 `asyncio.to_thread` 执行同步数据库/检索逻辑，并在 `finally`
中关闭 Session。这样 MCP 事件循环不会被数据库或 Embedding 调用阻塞。

### 5.4 并发闸门

#### 键与上限

Redis 键：`zhiyu:model_gate:{kind}:{provider}:{model}`

- `kind`：`llm` / `embedding` / `rerank`
- `provider`：现有服务里的 `provider_name` 或 `primary` / `fallback`
- `model`：实际调用的模型名

默认上限：

| 类型 | 默认值 | 依据 |
| --- | --- | --- |
| LLM | 4 | 与 `arq_agent_max_jobs=4` 对齐 |
| Embedding | 8 | 索引批处理需要更高吞吐 |
| Rerank | 4 | 跟随检索并发 |

配置项（均可通过 `.env` 覆盖）：

```
MODEL_GATE_ENABLED=true
MODEL_GATE_LLM_MAX_CONCURRENCY=4
MODEL_GATE_EMBEDDING_MAX_CONCURRENCY=8
MODEL_GATE_RERANK_MAX_CONCURRENCY=4
MODEL_GATE_WAIT_TIMEOUT_SECONDS=30
MODEL_GATE_LEASE_SECONDS=300
```

#### 算法

使用 Redis ZSET 实现带租约的分布式信号量：

- 获取：删除分数早于 `now - lease` 的成员（持有者崩溃留下的僵尸许可），
  若当前成员数小于上限，则 `ZADD` 自己的唯一成员并返回许可；
  否则 `sleep(50ms)` 后重试，直到 `wait_timeout`。
- 释放：`ZREM` 自己的成员。
- 获取/释放都用 Lua 脚本或 pipeline 保证原子性。
- 成员值：`{pid}:{thread_id}:{uuid4}`。

#### 调用点

- `LLMService._chat_completion`：每次尝试（primary / fallback）分别获取许可，
  用完即释放。
- `LLMService.stream_chat`：许可覆盖整个流式迭代周期；`GeneratorExit` /
  取消时也要释放（用 `try/finally`）。
- `OpenAICompatibleEmbeddingBackend.encode_batch`：每个批次获取一次许可。
- `RerankCompatibleBackend.rerank`：每次请求获取一次许可。

#### 降级

- Redis 连接 / 执行异常时：记录一条限频警告，**不获取许可直接放行**。
- `MODEL_GATE_ENABLED=false` 时完全跳过，行为与当前一致。
- 等待超时：记录警告后放行（宁可让请求自己去撞 429，也不要让用户一直卡住）。

### 5.5 可观测性

- 闸门等待耗时记入 `timed_stage("model_gate.wait")`，沿用现有遥测。
- 新增日志字段：`kind`、`provider`、`model`、`waited_ms`、`degraded`。
- 不记录 Prompt、正文或 API Key。

## 6. 测试策略

单元测试（不依赖真实 Redis）：

- `test/unit/core/test_model_gate.py`
  - 未达上限时立即获取并释放；
  - 达到上限时等待，释放后可获取；
  - 租约过期后僵尸许可被清理；
  - Redis 异常时降级放行并只告警；
  - `MODEL_GATE_ENABLED=false` 时不访问 Redis。
- `test/unit/mcp_server/test_tools.py`
  - `search_wiki` 返回精简后的结果并遵守 `top_k`；
  - `read_page` 支持标题与 ID，正文按上限截断；
  - `list_pages` 传递筛选参数并只返回摘要；
  - 领域异常转换为工具错误，不泄露堆栈。

测试使用假服务对象，不连接 MySQL。

集成验证（手工，不写入自动化测试）：

```bash
# 1. 启动 MySQL / Redis 与应用
# 2. 在 Claude Code 中配置 stdio MCP
# 3. 调用 search_wiki / read_page / list_pages
# 4. 关闭 Redis，确认问答仍可工作且日志出现降级告警
```

## 7. 文档更新

- `.env.example`：新增闸门配置段与说明。
- `README.md`：新增「MCP Server」与「模型并发闸门」两节。
- `docs/architecture/overview.md`：在部署拓扑旁补充「所有模型调用经过 Redis 闸门」一句，
  并说明闸门是 Redis 的又一个消费者。

## 8. 留给实现者的决定

- Redis ZSET 与 Lua 脚本的具体写法。
- 工具返回的 JSON 字段命名细节（遵循 MCP 客户端展示习惯）。
- 单测中假 Redis 客户端的构造方式。
- 日志限频的实现方式（时间戳 + 线程锁即可）。
