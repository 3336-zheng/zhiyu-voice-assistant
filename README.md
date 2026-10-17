<div align="center">

# 智语 · Zhiyu

**本地优先的个人知识工作台**

把语音、文档和笔记沉淀为可维护知识库，再用可信问答和受控研究帮助知识持续复用与更新。

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=111827" alt="React 19" />
  <img src="https://img.shields.io/badge/LangGraph-Agent-111827?style=flat-square" alt="LangGraph Agent" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="MIT License" />
</p>

</div>

## 产品简介

智语面向需要长期整理资料的个人用户。它将课堂录音、PDF/DOCX、Markdown 和零散笔记统一收纳为可编辑 Wiki，并在回答时给出可回溯的知识来源。

智语的重点不是让模型“自由发挥”，而是让知识可以维护、回答可以核验、操作可以确认。外部研究、页面修改和知识纠错都不会绕过用户确认直接写入。

## 你可以用它做什么

- 整理课程录音、技术资料和个人笔记。
- 在自己的知识库中搜索标题、正文和历史对话。
- 获得带来源的可信回答，快速回到对应页面。
- 在本地资料不足时，通过受控的 Search/Fetch 获取外部资料。
- 标记回答中的知识缺失、内容过期或引用错误，确认修订后自动复测。

## 核心能力

| 能力 | 产品价值 |
| --- | --- |
| **统一知识库** | Markdown 保存正文，MySQL 管理版本、链接、会话和索引状态，资料可读、可备份、可迁移。 |
| **可信问答** | BM25、向量检索、RRF、Rerank 和证据门禁共同筛选答案来源；证据不足时明确拒答。 |
| **智能执行** | Agent 根据当前任务选择搜索、总结、页面操作等能力，后端负责参数校验、权限、预算和确认。 |
| **受控外部研究** | 本地知识不足时才启用 MCP Search/Fetch，外部内容先形成研究草稿，再由用户决定是否写入。 |
| **可恢复运行** | 长任务支持流式输出、断线续传、取消、超时和重启后的终态恢复。 |
| **知识纠错闭环** | 回答反馈、证据快照、修订草稿、确认写入、重新索引和原问题复测形成完整闭环。 |
| **可靠语音采集** | 支持 DashScope 和 MiMo 两种云端转写；统一音频格式、记录时间戳，并限制重复转录。 |
| **本地优先与隐私** | 数据和索引默认保存在本机；日志脱敏，在线模型和外部研究按配置显式启用。 |

## 技术底座

| 领域 | 技术 |
| --- | --- |
| 前端 | React 19、Vite、React Markdown、Lucide React、SSE |
| 后端 | Python 3.11、FastAPI、Uvicorn、Pydantic v2、LangGraph |
| 数据 | MySQL 8、Redis 7、SQLAlchemy、Markdown、YAML Front Matter |
| 检索 | BM25、Jieba、ChromaDB、RRF、Embedding、Rerank、CRAG |
| 任务 | arq、Redis 队列（索引 / 语音转写 / Agent 运行三条队列各一个 worker） |
| Agent | Plan-and-Execute、工具注册表、JSON Schema、有限 Replan、MCP Python SDK |
| 采集 | ffmpeg/ffprobe、DashScope ASR、MiMo ASR、pdfplumber、python-docx |
| 运维 | 结构化 JSON 日志、Request ID、运行追踪、Server-Timing、Pytest、Docker |

业务数据在 MySQL 8，SQLAlchemy 使用同步会话；异步主要用于 FastAPI、SSE、Agent/MCP 编排和后台 Worker。这个配置适合单用户本地部署。

## 开始使用

环境要求：Python 3.11+、Node.js 20+、ffmpeg、Docker（跑 MySQL / Redis / Chroma），以及兼容的在线模型服务。

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
```

### 启动

应用分四个进程：一个 API，三个后台 worker（索引、语音转写、Agent 运行）。worker 各消费一个队列，分开是因为时长差一个量级——混在一起的话，上传一段长音频会让后面排着的索引任务一直等。

**必须四个都起来**。只跑 API 的话，页面能打开，但提问不会有任何反应（Agent 运行没人执行）。

```bash
# ① 三个基础服务
docker compose up -d mysql redis chroma

# ② API（一个终端）
python run_api.py

# ③ 三个 worker（另一个终端）
python run_workers.py
```

应用默认地址为 `http://127.0.0.1:8337`。模型、Embedding、Rerank、ASR 和 MCP 配置统一放在 `.env`，不配置模型时，知识库管理等非模型功能仍可使用。

### 用 Docker 跑全部

依赖和应用一起容器化，一条命令：

```bash
docker compose up -d
```

数据库 schema 由 Alembic 管理，首次启动或升级前执行 `alembic upgrade head`。

## 开发与测试

测试连的是**真实 MySQL** 上的独立库 `zhiyu_test`，不是内存库——这样测试和生产跑在同一种方言上。代价是**跑测试前必须先起 MySQL**：

```bash
docker compose up -d mysql

# 新环境首次需要建测试库（已有的数据卷不会自动执行初始化脚本）
bash scripts/init_test_db.sh

python -m pytest test/
```

每个测试用例都会重建整张 schema，所以**测试之间没有残留**，但也**必须串行跑**——共用同一个库时并行会互相清表。

## 运行管理

前端“运行追踪”可以查看查询可信度、阶段耗时、模型用量、Token 预算、RAG 证据和错误码。服务重启后，已结束的 Agent 运行仍可从 MySQL 中恢复。

日志默认写入 `data/logs/app.log` 和 `data/logs/error.log`，每个文件达到 `10 MB` 后轮转，各保留 `7` 个备份，单类日志最多约 `80 MB`。日志按文件大小控制，不按记录条数控制；密钥、Prompt、Wiki 正文、转写全文和外部网页正文不会写入日志。

## MCP Server（只读）

智语可以把知识库以标准 MCP 工具暴露给 Claude Code、Cursor 等外部 AI 工具，仅提供 `search_wiki`、`read_page`、`list_pages` 三个只读能力，不写库、不调用 LLM。

`search_wiki` 复用应用内的混合检索，因此会调用配置的在线 Embedding / Rerank 端点，查询文本会发送到该端点；这两个端点未配置时检索能力不可用。`read_page` 与 `list_pages` 只读本地数据。

```bash
# .env 中启用
MCP_SERVER_ENABLED=true

# 启动（stdio，通常由 MCP 客户端拉起）
python -m backend.app.mcp_server
```

客户端配置（路径按实际环境替换）：

```json
{
  "mcpServers": {
    "zhiyu": {
      "command": "<python 路径>",
      "args": ["-m", "backend.app.mcp_server"],
      "cwd": "<项目根目录>"
    }
  }
}
```

## 模型并发闸门

API 与三个 worker 会同时调用模型网关。闸门用 Redis 按 `provider + model` 限制所有进程合计的并发数，避免触发供应商 429：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_GATE_ENABLED` | `true` | 总开关 |
| `MODEL_GATE_LLM_MAX_CONCURRENCY` | `4` | LLM 并发上限 |
| `MODEL_GATE_EMBEDDING_MAX_CONCURRENCY` | `8` | Embedding 并发上限 |
| `MODEL_GATE_RERANK_MAX_CONCURRENCY` | `4` | Rerank 并发上限 |
| `MODEL_GATE_WAIT_TIMEOUT_SECONDS` | `30` | 等待许可的最长时间，超时放行 |
| `MODEL_GATE_LEASE_SECONDS` | `300` | 许可租约，进程崩溃后自动回收 |

Redis 不可用时闸门自动降级放行，问答与索引不受影响。

## 数据与隐私

- Markdown 正文和 MySQL 里的业务数据是**要备份的部分**；BM25、ChromaDB 是可以随时重建的派生索引，不用备。数据库用 `mysqldump` 导出，正文目录直接打包。
- API Key 只放在本地 `.env`，不会提交到仓库。
- 启用在线 Embedding、Rerank、LLM 或 MCP 时，相关查询和候选内容会发送到配置的服务端点。
- 当前定位是单用户、本机或可信内网部署，不含多租户权限。部署形态是**单实例多进程**（一个 API + 三个 worker）；多实例所需的代码能力保留着，但没有启用。

## 产品文档

- [系统架构](docs/architecture/overview.md)
- [分布式改造总结](docs/architecture/phase5-retrospective.md)
- [日志与运行追踪](docs/runbooks/observability.md)
- [RAG 评测设计](docs/eval/01-evaluation-design.md)
- [RAG 评测流程](docs/eval/02-evaluation-workflow.md)
- [评测代码说明](test/eval/README.md)

关键决策记录：

- [ADR-0001：Markdown 作为知识主数据](docs/adr/0001-markdown-source-of-truth.md)
- [ADR-0002：持久化异步索引任务](docs/adr/0002-persistent-index-tasks.md)
- [ADR-0003：证据门禁与确认式 Agent](docs/adr/0003-trusted-agent.md)
- [ADR-0004：受控 MCP 外部研究](docs/adr/0004-controlled-mcp-research.md)
- [ADR-0005：可配置检索模型与向量空间隔离](docs/adr/0005-configurable-retrieval-models.md)
- [ADR-0006：分布式运行时改造](docs/adr/0006-distributed-runtime.md)

## License

[MIT License](LICENSE)
