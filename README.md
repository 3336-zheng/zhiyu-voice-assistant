# 智语

智语是一个本地优先的个人 AI Wiki，面向课堂学习场景提供录音转写、结构化笔记、版本化页面、混合检索和可信问答。

## 能力边界

- React Wiki 工作台：Markdown 页面、笔记本、标签、别名、Wiki Link、反向链接和版本回滚。
- 统一页面服务：Markdown 是主数据，SQLite 保存元数据、版本、链接和索引任务。
- 异步索引：页面写入与 BM25/Embedding/ChromaDB 索引解耦，失败任务自动退避并可恢复。
- Agent 确认式写入：创建、修改、删除等高影响操作先生成预览，确认后幂等执行。
- 证据门禁：无召回或相关性不足时返回结构化“证据不足”，不调用模型生成推测性答案。
- 课堂沉淀：录音或上传音频，经 Whisper/DashScope 转写后生成 Wiki 页面。
- 数据安全：非破坏性 schema 迁移、数据库与 Wiki 文件备份、受保护恢复。

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React 19、Vite 6、React Markdown、Lucide React |
| API | Python 3.11、FastAPI、Uvicorn |
| 数据 | SQLite、SQLAlchemy、UTF-8 Markdown |
| 检索 | BM25、BGE Embedding、RRF、BGE Reranker、ChromaDB |
| Agent | LangGraph、Plan-and-Execute、多轮会话 |
| 语音 | faster-whisper 或 DashScope |
| 部署 | Docker 多阶段构建、Docker Compose |

## 架构

```text
React 工作台
    |
    v
FastAPI
    |
    +-- PageService ---------- Markdown 主数据 + SQLite 元数据
    |       |
    |       +---------------- WikiIndexTask 持久化任务
    |                              |
    |                              v
    |                       Wiki Index Worker
    |                       BM25 / Embedding / ChromaDB
    |
    +-- Agent Runtime -------- 检索、证据门禁、确认式写入
    +-- Audio / Summary ------- 转写、课堂笔记预览和保存
```

页面索引分块 ID 固定为：

```text
page:{page_id}:revision:{revision}:chunk:{index}
```

## 安装与运行

### 环境要求

- Python 3.11+
- Node.js 20+
- ffmpeg
- 本地 Whisper、Embedding、Reranker 模型，或相应云端配置
- OpenAI 兼容的 LLM API

### 安装依赖

```bash
conda create -n zhiyu python=3.11 -y
conda activate zhiyu
conda install -c conda-forge ffmpeg -y
pip install -r requirements.txt

cd frontend
npm install
cd ..
cp .env.example .env
```

`.env` 至少需要配置：

```env
WHISPER_MODEL_PATH=/absolute/path/to/whisper-model
EMBEDDING_MODEL_PATH=/absolute/path/to/bge-model
RERANKER_MODEL_PATH=/absolute/path/to/reranker-model
ASR_PROVIDER=whisper
LLM_API_KEY=your_api_key
LLM_API_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 生产模式

FastAPI 优先托管 `frontend/dist`：

```bash
cd frontend && npm run build
cd ..
python main.py
```

访问：

- 应用：`http://127.0.0.1:8337`
- Swagger：`http://127.0.0.1:8337/api/docs`
- 健康检查：`http://127.0.0.1:8337/health`
- 模型状态：`http://127.0.0.1:8337/health/models`

### 前端开发模式

终端一启动后端，终端二启动 Vite：

```bash
# 终端一，项目根目录
python main.py

# 终端二，frontend 目录
npm run dev
```

Vite 默认运行在 `http://127.0.0.1:5173`，并代理 `/api`、`/agent`、`/audio`、`/summary`、`/notes` 和 `/health`。

## Wiki API

基础路径：`/api/pages`。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/pages` | 创建页面并加入索引队列 |
| `GET` | `/api/pages` | 按标题、标签、笔记本分页查询 |
| `GET` | `/api/pages/{page_id}` | 读取页面 |
| `PUT` | `/api/pages/{page_id}` | 基于 `expected_revision` 更新页面 |
| `DELETE` | `/api/pages/{page_id}` | 软删除页面，保留历史版本 |
| `GET` | `/api/pages/{page_id}/links` | 出链和反向链接 |
| `GET` | `/api/pages/{page_id}/revisions` | 版本列表 |
| `GET` | `/api/pages/{page_id}/diff` | 两个版本的差异 |
| `POST` | `/api/pages/{page_id}/rollback` | 从历史版本创建新版本 |
| `POST` | `/api/pages/import-legacy` | 幂等导入旧笔记或文档 |
| `GET` | `/api/pages/export` | 导出 Wiki ZIP |
| `POST` | `/api/pages/reindex` | 全量生成索引任务 |
| `POST` | `/api/pages/index-tasks/retry` | 立即重试失败任务 |

创建页面：

```bash
curl -X POST http://127.0.0.1:8337/api/pages \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "RAG 基础",
    "content": "# RAG\n\n检索增强生成。",
    "notebook": "AI",
    "tags": ["RAG", "LLM"],
    "aliases": ["检索增强生成"]
  }'
```

写入接口立即返回页面，但 `index_status` 初始通常为 `pending`。后台 worker 默认每 5 秒轮询，每批处理 5 个任务。模型不可用时，页面仍然保存，任务进入 `failed` 并按指数退避重试。

更新与回滚：

```bash
curl -X PUT http://127.0.0.1:8337/api/pages/PAGE_ID \
  -H 'Content-Type: application/json' \
  -d '{"expected_revision": 1, "content": "更新后的正文"}'

curl 'http://127.0.0.1:8337/api/pages/PAGE_ID/diff?from_revision=1&to_revision=2'

curl -X POST http://127.0.0.1:8337/api/pages/PAGE_ID/rollback \
  -H 'Content-Type: application/json' \
  -d '{"target_revision": 1, "expected_revision": 2}'
```

`expected_revision` 不匹配时返回 `409 Conflict`，调用方必须重新读取页面。非法参数返回 `422`，页面或版本不存在返回 `404`。

## Agent 与证据门禁

Agent 写入采用两阶段确认：

```bash
curl -X POST http://127.0.0.1:8337/agent/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"query": "创建一篇标题为 RAG 复习提纲的笔记", "session_id": "demo"}'
```

写入请求返回 `confirmation_required=true`、`pending_action_id` 和 `action_preview`，确认或取消：

```bash
curl -X POST http://127.0.0.1:8337/agent/actions/ACTION_ID/confirm \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo"}'

curl -X POST http://127.0.0.1:8337/agent/actions/ACTION_ID/cancel \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo"}'
```

问答响应新增字段：

| 字段 | 含义 |
| --- | --- |
| `evidence_status` | `sufficient`、`insufficient` 或 `not_applicable` |
| `evidence_score` | 最高重排分数，可能为空 |
| `evidence_source_count` | 去重后的来源数量 |
| `evidence_reason` | 证据评估原因 |
| `sources` | 页面、版本、章节、来源 URI 和稳定 chunk ID |

当 `evidence_status=insufficient` 时，系统不会继续调用 LLM 生成推测性答案。

## 备份与恢复

备份包含 SQLite 快照、Wiki 页面、附件、上传文件和兼容目录。备份创建不会锁定或删除业务数据：

```bash
python scripts/backup.py create --output-dir data/backups
```

恢复是显式写入操作，必须确认；默认不覆盖已有文件：

```bash
python scripts/backup.py restore data/backups/zhiyu-backup-*.zip \
  --target-root /absolute/path/to/project \
  --confirm
```

需要覆盖时额外传入 `--overwrite`。恢复过程会校验 ZIP 路径，拒绝目录穿越和未知备份格式。

## 数据与迁移

| 数据 | 位置 |
| --- | --- |
| 当前 Wiki 页面 | `data/wiki/pages/` |
| 页面附件和导出 | `data/wiki/attachments/`、`data/wiki/exports/` |
| SQLite 数据库 | `data/database/notes.db` |
| 备份 | `data/backups/` |
| 向量索引 | `data/database/chromadb/` |

启动时只执行非破坏性 schema 迁移，当前 schema 版本为 `2`，迁移记录保存在 `schema_migrations`。应用不会自动删除旧表；历史目录需要通过 `/api/pages/import-legacy` 显式迁移。

## Docker

```bash
docker compose up --build
```

容器通过 `./data` 持久化数据库、页面、上传文件、索引和日志；模型目录由 `.env` 配置并以只读方式挂载。

## 验证

```bash
cd frontend && npm run build
cd ..
python -m compileall -q backend scripts main.py
python -m unittest discover -s test -p 'test_*_unit.py' -v
git diff --check
```

当前单元测试覆盖页面版本与冲突、Wiki 链接、索引任务、Agent 确认、异步入队、证据门禁、备份恢复和路径安全。

## 已知限制

- Embedding、Reranker 和本地 Whisper 需要有效的本地模型目录；缺失时页面 CRUD 仍可用，相关索引或转写功能降级。
- 当前服务面向单用户本机部署；若暴露到局域网，应配置受限 CORS、API Key 和反向代理限流。
- `data/database/chromadb` 是派生数据，可以通过 `/api/pages/reindex` 从 Wiki 主数据重建。

## License

MIT License
