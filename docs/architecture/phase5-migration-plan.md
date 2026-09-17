# Phase 5 分布式改造实施计划

- 分支：`feat/phase5-distributed`
- 基线：`ca84ca4`（从 `codex/evose-ui-style` 切出，非 `main`——`main` 落后 6 个提交，后端有 34 个文件差异）
- 决策依据：[ADR-0006](../adr/0006-distributed-runtime.md)

## 目标架构

```
                  前端 (Nginx 静态托管)
                          │ HTTP / SSE
                  zhiyu-api × N  (FastAPI，无状态)
                  │        │        │
          ┌───────┘        │        └───────┐
       MySQL 8          Redis 7         ChromaDB
       业务数据       队列/事件/锁/信号      向量检索
          ▲               ▲   ▲               ▲
          └───────┬───────┘   └───────┬───────┘
      arq-worker-default          arq-worker-asr
      索引/摘要，max_jobs=10       Whisper，max_jobs=1
      可 scale                     不可 scale（独占模型）
```

## 阶段依赖

```
Stage 0 ──┬─→ Stage 1 ──→ Stage 2 ──┬──→ Stage 5 ──→ Stage 6 ──→ Stage 7
          ├─→ Stage 3 (可并行) ──────┤
          └─→ Stage 4 (可并行) ──────┘
                                          Stage 8（可选，独立）
```

- Stage 0 是所有阶段前置
- Stage 3、4 不依赖数据迁移，可与 Stage 2 并行
- Stage 6 必须在 Stage 5 之后（arq 基础设施先就位）
- Stage 7 必须在 Stage 5、6 之后

---

## Stage 0 — 容器骨架 `S`

**改动**
- `docker-compose.yml`（27 行 → 约 120 行）
- `requirements.txt`

**内容**
- 新增 `mysql:8.0`、`redis:7-alpine`、`chromadb/chroma` 三个 service
- 每个 service 配 `healthcheck`，`zhiyu` 用 `depends_on: condition: service_healthy`
- 数据卷：`mysql_data`、`redis_data`、`chroma_data` 独立持久化
- `requirements.txt` 增加 `pymysql`、`redis`、`arq`、`alembic`，并为 SQLAlchemy 固定版本

**验收**：`docker compose up` 后三个新容器 healthy，`zhiyu` 仍跑 SQLite，功能不受影响。

---

## Stage 1 — 数据库层改造 `M`

**改动**
- `backend/app/core/database.py`（21 行 → 约 70 行）
- `backend/app/core/lifecycle.py:20`
- `backend/app/core/config.py`
- 新增 `alembic/` 目录与基线迁移
- `backend/app/core/schema.py` 归档
- `test/unit/core/test_schema_v6.py` 随 `schema.py` 一起归档

**内容**
1. `database.py` 按 dialect 分支（仅约 10 行）：
   - SQLite（测试路径）：`connect_args={"check_same_thread": False}`
   - MySQL（生产路径）：`charset=utf8mb4` + 显式连接池 `pool_size` / `max_overflow` / `pool_timeout` / `pool_recycle`
2. 删除 `lifecycle.py:20` 的 `settings.database_url.replace("sqlite:///", "")` 硬编码，否则 MySQL 下静默失效
3. 建立 Alembic 基线，每个迁移必须提供 `upgrade` 与 `downgrade`（现有 `schema.py` 只能前进）
4. `schema.py:149` 的 `id INTEGER PRIMARY KEY AUTOINCREMENT` 改为复合主键 `(page_id, research_source_id)`——该表 `wiki_page_sources` 已有同字段唯一约束 `uq_wiki_page_source`，改动无损

**不做**：不开启 SQLite WAL。生产不跑 SQLite，WAL 仅对单机多线程读写争抢有价值，测试是单线程的。

**验收**：`DATABASE_URL` 分别指向 SQLite 与 MySQL 都能启动服务并跑通单元测试。

---

## Stage 2 — 数据迁移 `M`

**改动**：新增 `scripts/migrate_sqlite_to_mysql.py`

**内容**
- 按 ORM 逐表读写，**不使用 SQL dump**（dump 携带 SQLite 方言）
- 迁移顺序：无外键依赖表 → 有外键依赖表
- 迁移后做行数比对 + 关键字段抽样比对
- 原 `data/notes.db` 保留为只读冷备，不再被代码引用

**验收**：两库行数一致，前端四个 tab 数据显示正常。

---

## Stage 3 — BM25 索引外置 `S/M`（可与 Stage 2 并行）

**改动**
- `backend/app/services/retrieval/bm25_service.py`（新增 `save_snapshot` / `load_snapshot`）
- `backend/app/core/lifecycle.py:55-59`

**内容**
- `self.corpus` 序列化后存入 Redis，启动时优先 `load_snapshot`，失败才回退全量重建
- 写入路径（`add` / `update` / `remove`）加 Redis 锁 + 版本号，避免多实例互相覆盖
- 检索算法与分词逻辑一行不改

**验收**：重启进程后 `get_stats()` 文档数不变，启动耗时明显下降。

---

## Stage 4 — Chroma server 模式 + 收封装泄漏 `S`

**改动**
- `backend/app/services/retrieval/chroma_service.py`
- `backend/app/services/retrieval/hybrid_retrieval_service.py:109`、`:323`
- `backend/app/services/retrieval/hybrid_retrieval_service.py:254`、`:269`、`:310`、`:637`

**内容**
1. **先修封装泄漏**：`hybrid_retrieval_service` 两处直接访问 `self.chroma_service.collection.get(...)` 与 `.count()`，必须收敛为 `chroma_service` 的方法，否则更换 client 类型会失败
2. `PersistentClient` 改为工厂函数，按配置返回 `PersistentClient` 或 `HttpClient`
3. 工厂内增加向量库地址 SSRF 校验（参考 WeKnora `internal/container/engine_factory.go` 的 `validateRuntimeVectorStoreAddresses`）——现有外部研究已做 URL 校验，但向量库地址未做
4. 顺手修复 4 处 per-request 线程池：`:254`、`:269`、`:310`、`:637` 目前都是 `with ThreadPoolExecutor(...)`，检索是高频路径，每次请求创建销毁线程池是纯浪费，改为服务实例级持有

**验收**：embedded 与 server 两种模式下检索结果一致。

---

## Stage 5 — 索引与 ASR 迁移到 arq `L`

**改动**
- 新增 `backend/app/tasks/`：`worker.py`、`index_tasks.py`、`asr_tasks.py`
- 删除 `backend/app/services/wiki/wiki_index_worker.py`（全文 40 行）
- `backend/app/services/ingestion/asr_service.py:110`
- `backend/app/core/lifecycle.py`
- `docker-compose.yml` 增加两个 worker service

**内容**
1. 写页面时直接 `enqueue_job('index_page', page_id)`，**取消 5 秒轮询**
2. ASR 用 job_id 做幂等去重，替代 `asr_service.py:110` 的 `_running_audio_ids` 进程内集合
3. 两个队列配置：
   - `default`：`max_jobs=10`，可 scale
   - `asr`：`max_jobs=1`，独占 Whisper 模型，不可 scale
   - `asr` 队列的 `max_jobs=1` 同时替代 `asr_service.py:112` 的 `self._whisper_slot = asyncio.Semaphore(1)`。该信号量只在单进程内有效，多实例下每个进程各有一个，无法保证全局单并发，本地模型会被同时加载多份
4. 保留 `WikiIndexTask` 表作为任务状态记录（ADR-0002 的退避与重试语义不变），arq 只接管调度

**验收**：`docker compose up --scale arq-worker-default=2`，提交 10 个索引任务无重复执行。

---

## Stage 6 — AgentRuntimeService 分布式化 `XL`

**改动**：`backend/app/services/runtime/agent_runtime_service.py`（605 行，本次改造最难的一块）

| 现状 | 改为 |
| --- | --- |
| `RuntimeRun` 内存事件列表 | Redis Stream `run:{id}:events`，SSE 用 `XREAD` + `last_id` 断点续传 |
| `threading.Event` 取消信号 | Redis key `run:{id}:cancel`，执行循环轮询 |
| `_active_sessions` 会话锁 | Redis `SET NX EX` + 心跳续期 |
| `runtime_instance` 全局单例 | 单例保留，状态全部外置 |
| `observability.py:33` 的 `_recent_traces`（进程内 OrderedDict，上限 500 条） | Redis，否则多实例下 trace 查询变成按实例抽奖：请求落在实例 A，查询被路由到实例 B 就返回 None |

**已知坑**：改造后 `agent_runs` 表的 `timeline` / `retrieval_stats` / `model_usage` 三个字段会**静默变空**——它们依赖 `core/observability.py` 的请求级上下文，arq worker 是独立进程，没有 FastAPI 请求上下文。必须显式将 context 作为任务参数传入。

**验收**：实例 A 发起对话，实例 B 能读到完整事件流；在 B 上点取消，A 上的执行立即停止。

---

## Stage 7 — 多实例上线 `M`

**改动**
- `backend/app/core/lifecycle.py`
- `backend/app/services/runtime/agent_runtime_service.py` 的恢复逻辑
- `agent_runs` 表新增 `heartbeat_at` 字段

**内容**
1. 移除 `lifecycle.py` 中所有常驻后台任务的 `asyncio.create_task(...)`（已在 Stage 5 迁走）
2. **修复恢复逻辑误杀**：现状是启动时把所有 `pending`/`running`/`cancelling` 标记失败。多实例下，实例 A 重启会杀掉实例 B、C 正在运行的 Run。

```python
# 参考 WeKnora internal/container/reset_pending_tasks.go 的 distributed 参数思路
def recover_interrupted_runs(db, distributed: bool, stale_cutoff: datetime):
    q = db.query(AgentRun).filter(AgentRun.status.in_(RUNNING_STATUSES))
    if distributed:
        q = q.filter(AgentRun.heartbeat_at < stale_cutoff)   # 只回收心跳过期的
    stuck_ids = [r.id for r in q.all()]                      # 先 SELECT 出 ID
    if not stuck_ids:
        return
    db.query(AgentRun).filter(                               # 再按 ID 更新
        AgentRun.id.in_(stuck_ids),
        AgentRun.status.in_(RUNNING_STATUSES),               # 二次校验，防 SELECT/UPDATE 间隙误伤
    ).update({...}, synchronize_session=False)
```

3. worker 执行期间定期续期 `heartbeat_at`，`stale_cutoff` 才有判定依据

**验收**：`--scale zhiyu-api=3`，滚动重启其中一个实例，另外两个实例上正在运行的 Run 不受影响。

---

## Stage 8 — 可选：拆分文档解析服务 `L`

参考 WeKnora 的 `docreader/`（独立 Python gRPC 服务）。将 pdfplumber / python-docx / faster-whisper 拆为独立服务，主 API 镜像不再安装 torch。

前七阶段完成前不启动。

---

## 跨阶段事项

### 进程内 TTL/LRU 缓存（Stage 7 之后，优先级低）

四处进程内缓存，多实例下**不影响正确性，只影响成本**：

| 位置 | 缓存内容 |
| --- | --- |
| `services/ai/embedding_service.py` | Embedding 查询结果 |
| `services/retrieval/query_rewrite_service.py` | Query Rewrite 结果 |
| `services/retrieval/hybrid_retrieval_service.py` | 检索结果 |
| `services/retrieval/crag_grader_service.py` | CRAG 评分 |

N 个实例各存一份，同一查询打到不同实例会重算，命中率降至 1/N，Embedding 与 LLM 调用费用相应上升。缓存未命中只是重算，不产生错误结果，因此优先级低于 Stage 6。

迁移时缓存键需保留现有的模型、索引版本、证据版本和策略参数维度，避免跨实例读到过期结果。

### 线程池治理（Stage 5 之后）

现状：26 处 `asyncio.to_thread` 共用 asyncio 默认池（`min(32, cpu+4)`），池中混有慢任务（ASR 转写、LLM 调用、BM25 全量重建）与快任务（DB persist）。慢任务会饿死快任务。

Stage 5 后 ASR 与 wiki index 的 `to_thread` 会随迁移消失，届时再新增 `core/executors.py` 按用途分池（retrieval / db / llm），避免现在做的工作被后续阶段删掉。

**必须计算的约束**：线程总数上限 ≤ DB 连接池容量。默认池 32 线程 vs Stage 1 的 `pool_size=10 + max_overflow=20 = 30`，会打爆连接池，表现为随机的 `QueuePool limit ... connection timed out`，且难以复现。此数值需在 Stage 1 定池大小时一并确定。

### 测试与生产方言不一致

单元测试跑 SQLite（11 个文件用 `Base.metadata.create_all`，方言无关，零改动），生产跑 MySQL。已知会出问题的差异：

- 字符串比较大小写：SQLite `LIKE` 默认不区分大小写，MySQL 取决于 collation
- 事务隔离级别：MySQL 默认 REPEATABLE READ，SQLite 为 SERIALIZABLE
- VARCHAR 长度：SQLite 忽略长度限制，MySQL 会截断或报错
- 索引 key 长度：MySQL utf8mb4 下有上限，长文本字段建索引会失败

对策：新增 `test/integration/mysql/`，仅覆盖上述四类方言敏感点与 Alembic 迁移连通性，CI 启动容器执行。单元测试继续用 SQLite 保持秒级。

### 技术债清理（任意阶段插入）

- 根目录 11 个 LeetCode 练习文件（未跟踪状态）——移出项目目录
- `frontend/legacy/` 4 个死 HTML 文件——删除
- `package-lock.json` 与 `pnpm-lock.yaml` 同时入库——二选一
- `frontend/src/styles/index.css` 3562 行，多于 JSX 总量 2734 行——拆分
- 前端 4 个 tab 无路由（`App.jsx:27` 单个 `useState`）——刷新丢失状态，引入 react-router

## 待确认

- conda 环境：`/Users/evose/miniconda3/envs` 下有 `base`、`Evose`、`zhiyu`，拟使用 `zhiyu`
- 起始阶段：建议 Stage 0 + Stage 1 合并推进，拆为两次提交
