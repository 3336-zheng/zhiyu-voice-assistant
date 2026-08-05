# 智语

智语是一个面向个人学习场景的本地优先 AI Wiki。系统将课堂录音和文档沉淀为可维护的长期知识，通过混合检索寻找证据，并由 Agent 完成知识整理、可信问答和确认式写入。

项目重点不是增加一个聊天入口，而是建立完整的知识生命周期：

```text
语音与文档采集 -> Wiki 组织与版本管理 -> RAG 证据检索
-> Agent 规划与执行 -> 来源追溯与持续复习
```

## 核心能力

- React Wiki 工作台：Markdown 页面、笔记本、标签、别名、Wiki Link 和反向链接。
- 统一页面服务：Markdown 是主数据，SQLite 保存元数据、历史版本、链接和索引任务。
- 异步索引：页面写入与 BM25/Embedding/ChromaDB 索引解耦，失败任务自动退避并可恢复。
- Agent 确认式写入：创建、修改、删除等高影响操作先生成预览，确认后幂等执行。
- 证据门禁：无召回或相关性不足时返回结构化“证据不足”，不调用模型生成推测性答案。
- 受控外部研究：本地证据不足时可显式调用白名单 MCP 搜索与抓取工具，保留来源快照，确认后再沉淀为 Wiki。
- 课堂沉淀：Whisper/DashScope 分段转写，回答来源可回溯到原音频时间点。
- 请求可观测：统一 `request_id`，记录查询改写、召回、精排、证据判断和生成耗时。
- 数据安全：非破坏性 schema 迁移、数据库与 Wiki 文件备份、路径校验和受保护恢复。

## 核心设计

### 主数据与派生索引

Markdown 保存 Wiki 当前正文，SQLite 保存页面元数据、历史版本、链接和任务状态。ChromaDB 与 BM25 只承担检索职责，属于可以从主数据重建的派生索引。

```text
Markdown + SQLite = 需要保护和备份的知识状态
ChromaDB + BM25   = 可以重建的检索索引
```

### 写入与最终一致性

页面写入、版本快照和索引任务由统一页面服务编排。页面保存成功后立即返回，模型索引由后台 Worker 异步处理；索引失败不会删除知识正文，而是记录失败原因、指数退避并在服务重启后继续恢复。

### 可信 Agent

读取任务可以直接执行，创建、修改和删除等高影响操作先生成变更预览，用户确认后才调用确定性工具。执行结果持久化，重复确认不会产生重复副作用。问答在生成前检查证据，无召回或相关性不足时明确拒答。

本地证据不足时，系统只向用户提供外部研究入口，不会自动联网。外部研究由模型生成检索词，但只能调用配置白名单中的 MCP 搜索和抓取工具；结果经过公网 URL 校验、去重、长度限制和提示注入隔离后形成带引用草稿。草稿必须再次确认才会写入 Wiki，页面同时关联研究来源快照。

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React 19、Vite 6、React Markdown、Lucide React |
| API | Python 3.11、FastAPI、Uvicorn、Pydantic |
| 数据 | SQLite、SQLAlchemy、UTF-8 Markdown |
| 检索 | BM25、BGE Embedding、RRF、BGE Reranker、ChromaDB |
| Agent | LangGraph 1.x、Plan-and-Execute、MCP Python SDK、多轮会话 |
| 语音 | faster-whisper 或 DashScope |
| 可观测性 | Request ID、结构化阶段耗时、Server-Timing |
| 部署 | Docker 多阶段构建、Docker Compose |

## 系统架构

```text
React 工作台
    |
    v
FastAPI
    |
    +-- PageService ---------- 稳定编排门面
    |       +-- WikiFileStore / Revision / Link
    |       +---------------- WikiIndexTask 持久化任务
    |                              |
    |                              v
    |                       Wiki Index Worker
    |                       BM25 / Embedding / ChromaDB
    |
    +-- Agent Runtime -------- 检索、证据门禁、确认式写入
    |       +---------------- 受限 MCP 外部研究与来源追溯
    +-- Audio / Summary ------- 转写、课堂笔记预览和保存
```

页面索引分块 ID 固定为：

```text
page:{page_id}:revision:{revision}:chunk:{index}
```

模块职责和关键决策见[架构说明](docs/architecture.md)与[架构决策记录](docs/adr/)。

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 22+
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

外部研究默认关闭。启用时只配置可信的 stdio MCP Server，并明确指定只读搜索、抓取工具及其参数名；完整配置项见 `.env.example`。MCP 子进程只接收 `MCP_SERVER_ENV_JSON` 中显式声明的环境变量，不继承应用的完整配置。

### 构建并运行

FastAPI 优先托管 `frontend/dist`：

```bash
cd frontend && npm run build
cd ..
python main.py
```

应用默认运行在 `http://127.0.0.1:8337`。

### 前端开发模式

终端一启动后端，终端二启动 Vite：

```bash
# 终端一，项目根目录
python main.py

# 终端二，frontend 目录
npm run dev
```

Vite 默认运行在 `http://127.0.0.1:5173`，开发服务器将业务请求代理到 FastAPI。旧笔记和文档入口仅作为兼容层保留，新功能统一进入页面服务；旧目录自动索引默认关闭，避免继续扩大双写范围。

## 可靠性与安全

- 页面更新使用版本号进行乐观并发控制，防止覆盖用户刚刚完成的修改。
- 索引任务持久化保存状态、尝试次数、错误和下次执行时间。
- FastAPI `lifespan` 统一执行 schema 迁移、任务恢复、Worker 启停和退出清理。
- 每次请求生成或继承 Request ID，并记录查询改写、召回、融合、精排、证据判断和生成耗时。
- 上传文件名、音频访问路径和备份恢复路径均进行边界校验。
- ChromaDB 以嵌入式 `PersistentClient` 运行，不对外暴露独立的未认证服务端口。
- 配置模板只保留占位符，真实模型路径和密钥不进入版本库。
- MCP 默认关闭且不接受动态工具名；外部 URL 会拒绝本机、私网、保留地址、嵌入凭证和非 HTTP(S) 协议。
- 外部正文按不可信数据处理，进入模型前执行总量截断和边界转义；模型输出不会绕过用户确认直接写入知识库。

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

启动时只执行非破坏性 schema 迁移，当前 schema 版本为 `4`，迁移记录保存在 `schema_migrations`。应用不会自动删除旧表；历史目录通过显式兼容导入流程迁移。外部研究任务、来源快照及页面来源关系保存在 SQLite 中，正文仍由 Markdown 承担主数据角色。

## Docker

```bash
docker compose up --build
```

容器通过 `./data` 持久化数据库、页面、上传文件、索引和日志；模型目录由 `.env` 配置并以只读方式挂载。

## 工程验证

```bash
cd frontend && npm run build
cd ..
python -m compileall -q backend scripts main.py
python -m unittest discover -s test -p 'test_*_unit.py' -v
python -m unittest discover -s test -p 'test_*_integration.py' -v
python test/eval/rag_eval.py --methods bm25
git diff --check
```

需要本地 Embedding 和 Reranker 模型时，可运行完整对照：

```bash
python test/eval/rag_eval.py \
  --methods bm25,embedding,hybrid,hybrid_reranker \
  --output test/eval/latest_results.json
```

测试覆盖页面版本与冲突、Wiki 链接、索引失败重试、重启恢复、Agent 重复确认、证据门禁、MCP 超时与私网拦截、外部来源确认入库、备份恢复、音频溯源和路径安全。CI 会执行后端测试、无模型评估、React 构建、Python 与 Node 依赖审计以及 Docker 构建。

当前无模型 BM25 基线：

| 指标 | 结果 |
| --- | ---: |
| Hit@1 | 0.80 |
| Hit@3 | 1.00 |
| MRR | 0.8889 |
| NDCG@5 | 0.9230 |

## 项目结构

```text
backend/app/
├── api/                 FastAPI 业务入口与兼容层
├── agent/               LangGraph 状态图、规划、执行和响应
├── core/                配置、数据库、生命周期、迁移和可观测性
├── models/              SQLAlchemy 数据模型
└── services/            页面、检索、语音、索引和备份服务

frontend/app/            React Wiki 工作台
docs/                    架构说明与 ADR
scripts/                 备份和恢复脚本
test/                    单元测试、API 集成测试和检索评估
data/                    本地运行数据，不进入版本库
```

## 已知限制

- Embedding、Reranker 和本地 Whisper 需要有效的本地模型目录；缺失时页面 CRUD 仍可用，相关索引或转写功能降级。
- 当前服务面向单用户本机部署；若暴露到局域网，应配置受限 CORS、API Key 和反向代理限流。
- ChromaDB 当前上游版本存在仅影响未认证 HTTP 服务模式的安全公告；本项目只使用嵌入式客户端，并在 CI 中保留单项、可追踪的临时例外。
- `data/database/chromadb` 是派生数据，可以从 Wiki 主数据重新生成。
- MCP Server 属于部署时信任边界，必须由使用者审查并限制其网络权限；应用会校验请求 URL，但无法替代 Server 对重定向和 DNS 重绑定的防护。

## 设计文档

- [系统架构](docs/architecture.md)
- [ADR-0001：Markdown 作为知识主数据](docs/adr/0001-markdown-source-of-truth.md)
- [ADR-0002：持久化异步索引任务](docs/adr/0002-persistent-index-tasks.md)
- [ADR-0003：可信问答与确认式 Agent](docs/adr/0003-trusted-agent.md)
- [ADR-0004：受控 MCP 外部研究](docs/adr/0004-controlled-mcp-research.md)

## License

MIT License
