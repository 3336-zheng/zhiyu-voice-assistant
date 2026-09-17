# ADR-0006：分布式运行时改造

- 状态：已提议
- 日期：2026-09-16
- 关联：扩展 ADR-0002（持久化异步索引任务）

## 背景

当前部署形态为单进程单容器，多处运行时状态绑定在进程内存中，无法横向扩容：

| 位置 | 进程内状态 |
| --- | --- |
| `services/retrieval/bm25_service.py:30` | `self.corpus: Dict[str, str]`，纯内存无持久化 |
| `services/runtime/agent_runtime_service.py` | `RuntimeRun` 事件列表、`threading.Event` 取消信号、`_active_sessions` 会话锁 |
| `services/ingestion/asr_service.py:110` | `_running_audio_ids` 去重集合 |
| `services/wiki/wiki_index_worker.py` | 5 秒轮询数据库，无跨进程任务抢占租约 |

ADR-0002 已预判此问题：「当前 Worker 不支持跨节点竞争；扩展到多实例时应迁移到具备租约或确认语义的独立队列。」

## 决策

**算法与功能不变，仅将进程内状态外置。**

1. 业务库由 SQLite 迁移到 MySQL 8，生产环境只保留 MySQL 单一方言；SQLite 仅作为单元测试的轻量替身继续存在。
2. 引入 Redis，一个组件承担四个职责：任务队列（经 arq）、事件流（Streams）、取消信号（key）、分布式锁（`SET NX EX`）。
3. 后台任务由进程内协程迁移到 arq worker 进程，索引任务改为写入时直接入队，取消 5 秒轮询。
4. ChromaDB 保留，仅由 `PersistentClient` 改为 `HttpClient` 独立容器。
5. 混合检索算法（BM25 + 向量 + RRF + Rerank）不做任何改动，仅将 BM25 索引存储外置到 Redis。

### 明确不做

- 不引入鉴权与多租户：项目定位为单用户。
- 不更换向量库为 pgvector：换库需重跑全量 embedding 并重新标定召回。
- 不用 MySQL FULLTEXT 替换 BM25：中文分词质量不可退。
- 不维护 MySQL/SQLite 双生产方言：参考项目 WeKnora 维护双方言是因其存在桌面版形态，本项目无此需求。

## 结果

- API 进程变为无状态，可 `--scale N` 横向扩容。
- 索引延迟由平均 2.5 秒（轮询周期一半）降至接近 0。
- 部署复杂度上升：容器由 1 个增加到 6 个，需要 healthcheck 编排。
- 运维成本上升：MySQL 与 Redis 需要备份与监控，原先只需备份单个 `.db` 文件。
- 测试与生产方言不一致，需要补充一组 MySQL 集成测试覆盖方言敏感点。
