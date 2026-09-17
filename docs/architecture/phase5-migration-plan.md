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

## Stage 0 — 容器骨架 `S` ✅ 已完成

**改动**
- `docker-compose.yml`（27 行 → 约 130 行）
- `requirements.txt`、`.env.example`

**内容**
- 新增 `mysql:8.0`、`redis:7-alpine`、`chromadb/chroma:1.5.9` 三个 service
- 每个 service 配 `healthcheck`，`zhiyu` 用 `depends_on: condition: service_healthy`
- 数据卷：`mysql_data`、`redis_data`、`chroma_data` 独立持久化
- `requirements.txt` 增加 `pymysql`、`cryptography`、`redis`、`arq`、`alembic`，并为 SQLAlchemy 与 chromadb 固定版本区间

**端口映射的双地址视角**

应用既可能直接跑在宿主机（本地开发），也可能跑在 compose 网络内（部署），两者看到的服务地址不同：

| 服务 | 宿主机视角（`.env`） | 容器内视角（compose `environment` 覆盖） |
|---|---|---|
| MySQL | `127.0.0.1:3316` | `mysql:3306` |
| Redis | `127.0.0.1:6382` | `redis:6379` |
| Chroma | `127.0.0.1:8001` | `chroma:8000` |

映射端口刻意避开 3306 / 6379 / 8000，因为开发机上常有其他项目占用。**一律绑定 `127.0.0.1`**，写成 `3316:3306` 等价于 `0.0.0.0`，会把无认证或弱口令的数据库暴露到局域网。

**执行期修正**

| 项 | 问题 | 处理 |
|---|---|---|
| chromadb 版本 | `requirements.txt` 原写 `>=0.5.0` 无上限，实际安装到 1.5.9 | 镜像与依赖同步锁到 `1.5.9` / `>=1.5.9,<1.6` |
| chroma healthcheck | 1.x 核心已用 Rust 重写，镜像内无 `python`/`curl`/`wget`/`nc` | 改用 bash 内建 `/dev/tcp` 做 TCP 探测 |
| API 路径 | 1.x 的 `/api/v1` 已返回 `410 Gone` | 健康检查不绑定 API 路径版本 |

**验收结果**：三容器全部 healthy；MySQL 8.0.46 / utf8mb4 / `max_connections=300`，中文与 emoji 写入往返正常；Redis AOF 已启用；Chroma `/api/v2/heartbeat` 正常、持久化目录为 `/data`；三个镜像均有原生 arm64，无需 `platform: linux/amd64`。

---

## Stage 1 — 数据库层改造 `M` ✅ 已完成

**改动**
- `backend/app/core/database.py`（24 行 → 62 行），新增 `is_sqlite()` 与 `_create_engine()`
- `backend/app/core/lifecycle.py:15` `_initialize_database()` 改为方言双分支
- `backend/app/core/config.py` 新增 12 个字段
- `backend/app/models/wiki.py` 两处唯一约束改为前缀索引
- 新增 `alembic.ini` / `alembic/` 与基线迁移 `913259aa108b`
- 删除 `backend/app/core/schema.py` 与 `test/unit/core/test_schema_v6.py`
- 新增 `test/unit/core/test_alembic_migration.py`

**内容**
1. `database.py` 按方言分支：SQLite 保持原样，MySQL 走 `pool_size` / `max_overflow` / `pool_timeout` / `pool_recycle` / `pool_pre_ping` + `charset=utf8mb4`
2. `_initialize_database()` 在非 SQLite 下直接返回、不执行任何 DDL——多实例并发建表会互相撞车，且 MySQL 的 DDL 不在事务内，中途失败会留下半完成的库
3. Alembic 基线含 14 张表，`upgrade` / `downgrade` 双向可用
4. `alembic/env.py` 从应用配置读取地址，不在 `alembic.ini` 中存第二份；写入前对 `%` 转义，避免 configparser 把 URL 编码的密码当作插值语法

**执行期发现的三处计划偏差**

| 原计划 | 实际 | 原因 |
|---|---|---|
| SQLite 加 `check_same_thread=False` | 不加 | SQLAlchemy 2.0 对文件型 SQLite 默认已用 QueuePool 并自行处理跨线程；实测跨线程访问正常。该建议源自 1.3 时代 |
| `schema.py:149` 改复合主键 | 不需要 | 问题只存在于裸 SQL，ORM 用的是方言无关的 `autoincrement=True`，`schema.py` 归档后自然消失 |
| 未预见 | 两处唯一约束改前缀索引 | 见下 |

**未预见的阻断：InnoDB 索引键长度上限**

utf8mb4 下每字符最多 4 字节，InnoDB 索引键上限 3072 字节，即被索引的 `VARCHAR` 最长 768 字符。SQLite 无此限制，因此该问题只在切到 MySQL 时才暴露，报错为 `1071 Specified key was too long`。

| 表.列 | 类型 | 原约束 | 所需字节 |
|---|---|---|---|
| `external_research_sources.url` | String(2048) | `uq_..._url (run_id, url)` | 8192 |
| `wiki_pages.file_path` | String(1024) | 列级 `unique=True` | 4096 |

处理方式：改为 `Index(..., unique=True, mysql_length=700)` 前缀索引，**列容量一字未改**，无数据截断风险；`mysql_length` 在 SQLite 上自动忽略。`url` 的唯一约束本就只是第二道防御——`external_research_service.py:206` 的 `_normalize_sources` 已用 `seen_urls` 完整去重，且 `run_id` 每次为新 UUID。

**同时证实 `schema.py` 无法用于 MySQL**（这是必须换 Alembic 而非沿用的硬理由）：

| 位置 | 写法 | MySQL 实测 |
|---|---|---|
| `schema.py:333` | `CREATE INDEX IF NOT EXISTS` | 语法错误 1064 |
| `schema.py:149` | `INTEGER PRIMARY KEY AUTOINCREMENT` | 语法错误 1064 |

**不做**：不开启 SQLite WAL。生产不跑 SQLite，WAL 仅对单机多线程读写争抢有价值，测试是单线程的。

**验收结果**
- MySQL 空库 `alembic upgrade head` 成功，建出 15 张表（14 业务 + `alembic_version`），全部 `utf8mb4_unicode_ci`，两个前缀索引 `sub_part=700`
- MySQL 连接池实测 `pool_size=10` / `max_overflow=20` / `pre_ping=True` / `recycle=3600`，连接字符集 `utf8mb4`
- 现有 SQLite 库（175 行真实数据）经新建表逻辑后行数不变
- `test/unit` 102 项 + `test/integration` 4 项全部通过

---

## Stage 2 — 数据迁移 `M` ✅ 已完成

**改动**：新增 `scripts/migrate_sqlite_to_mysql.py`

**内容**
- 按 ORM 逐表读写，**不使用 SQL dump**（dump 携带 SQLite 方言）
- 迁移顺序：无外键依赖表 → 有外键依赖表
- 迁移后做行数比对 + 关键字段抽样比对
- 原 `data/notes.db` 保留为只读冷备，不再被代码引用

**执行期修正**

- **DATETIME 精度**：MySQL 的 `DATETIME` 默认零精度，会把微秒直接截断。新增 `backend/app/models/types.py` 的 `DateTimeMs`，用 `with_variant` 只在 MySQL 方言下换成 `DATETIME(6)`，SQLite 行为不变；配套迁移改了 29 列。Alembic 的 autogenerate 检测不到 fsp 差异（即使开 `compare_type=True` 也只得到空迁移），这个迁移只能手写。
- **校验的浮点容差**：JSON 列里的浮点数在两库之间存在 1 ULP（相对误差约 2e-16）的差异——SQLite 侧走 Python `json.dumps` 的最短往返表示，MySQL 原生 JSON 列用自己的 double↔文本算法。校验改成递归比对，**只对浮点放宽到 1e-12，其余类型（含 datetime）一律严格相等**。全量比对 167 行，非浮点差异 0 处。
- **AUTO_INCREMENT 校验误报**：`information_schema` 的统计列是缓存值，`information_schema_stats_expiry` 默认 86400 秒，表空时读到的 1 会返回一整天。校验前设 `SET SESSION information_schema_stats_expiry = 0`。
- **清空目标表用 DELETE 而非 TRUNCATE**：被外键引用的表 TRUNCATE 会被 InnoDB 拒绝（错误 1701）；DELETE 不重置 AUTO_INCREMENT，而迁移本就要显式写入原 id。

**验收**：两库行数一致（14 表 167 行），前端四个 tab 数据显示正常，`test/unit` + `test/integration` 全绿。

---

## Stage 3 — BM25 索引跨实例一致 `S/M` ✅ 已完成

> 原计划是「把词表序列化进 Redis，省掉启动时的全量重建」。实测后这个前提不成立，方案改为只同步版本号。下面是改后的内容。

**改动**
- 新增 `backend/app/core/index_version.py`
- `backend/app/services/retrieval/bm25_service.py`（加锁 + `ensure_fresh` / `mark_synced`）
- `backend/app/services/retrieval/chroma_service.py`（`mark_index_changed` 改为递增全局版本，删除进程内 `get_generation`）
- `backend/app/services/retrieval/hybrid_retrieval_service.py`（四个检索入口在检索前刷新索引）
- `backend/app/services/ingestion/doc_index_service.py`（重建过程加锁、记录已同步版本、Chroma 拉取失败改为抛出）

**为什么不存快照**

973 个 chunk 全量重建实测 **399 ms**，拆开是：从 Chroma 拉取 175 ms + jieba 分词 216 ms + `BM25Okapi` 构造 8 ms。快照只能省掉分词那一段，净省约 211 ms，代价却是序列化格式兼容、快照体积、pickle 反序列化的安全面，以及写入侧的分布式锁。在这个数据量下是负收益。等 chunk 数涨到一万左右（分词约 2.2 秒）再考虑。

**真正要解决的两个问题**

1. **多实例索引不一致**：`_generation` 计数器存在进程内存里。实例 A 写入后只更新自己的计数器，B 和 C 毫不知情，继续用陈旧索引返回旧结果——不报错，只是搜不到新内容。
2. **BM25 的静默数据错位**：`doc_id_list[i]` 与 `tokenized_corpus[i]` 靠下标一一对应，而 `remove_document` 的两次删除不是原子的。并发下错位后，BM25 算出「第 5 篇最相关」，去 `doc_id_list` 取第 5 个却拿到别的文档。同样不抛异常，只是安静地返回错误结果。

**内容**
- Redis 上放一个计数器 `zhiyu:index:version`，索引变更时 `INCR`，各实例检索前比对，落后就自己全量重建
- 版本号带来源前缀（`r:` 来自 Redis，`l:` 来自进程内），从 Redis 模式掉到降级模式时版本串必然不同，必然触发一次重建；比较用 `!=` 而非 `>`，防止 Redis 被清空后计数器倒退导致漏重建
- Redis 不可用时降级为进程内计数器：连接超时 0.3 秒（这个调用在检索关键路径上），失败后 5 秒退避窗口内不再重试，warning 只打一次
- `BM25Service` 加两把锁：`_lock` 保护增删改与检索取快照，`_rebuild_lock` 单独保护全量重建（重建期间检索照常读旧索引，只有最后替换数据的一小段互斥）。取锁顺序固定 `_rebuild_lock → _lock`
- `search()` 改为锁内取一致快照、锁外算分：`BM25Okapi` 重建时是整个替换的，拿到的引用不会被原地改；`doc_id_list` 浅拷贝。两者同一时刻取出，下标对应关系必然一致
- 写入侧不需要逐点埋点：`mark_index_changed()` 是所有索引变更的唯一汇聚点
- 检索算法、分词逻辑、混合检索流程一行未改
- **不新增任何缓存**。现有 4 个 TTLCache（query 改写、检索结果、query embedding、CRAG 评分）缓存的都是昂贵的 LLM/embedding 调用，全部保留在进程内，不搬 Redis

**已知取舍**：写入的那个实例自己也会被版本号带动重建一次，哪怕它刚用 `add_document` 增量更新过。本可以让写入方顺手把自己标成已同步，但那要求精确区分「这次版本变化是我造成的」和「别的实例也同时改了」，判断错一次就是永久漏数据。多花的几百毫秒远小于写入本身（embedding + Chroma 落盘）的耗时。

**验收**（已通过）
- 新增 18 个测试，`test/unit` + `test/integration` 共 124 passed
- 并发用例做过反向验证：把锁换成空操作后，下标错位用例立刻失败（裸并发测试撞不到那个只有几条字节码宽的窗口，需要人为撑开）
- 端到端跨进程验证：另一个进程调 `mark_index_changed()` 让 Redis 版本 1→2，运行中的应用下一次检索日志显示「本地=r:1 全局=r:2」并自动重建
- 降级验证：停掉 Redis 后连续三次检索均 HTTP 200，warning 只出现一次；恢复 Redis 后日志显示「索引版本号已恢复使用 Redis」并重建一次

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
