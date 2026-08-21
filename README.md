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
| **统一知识库** | Markdown 保存正文，SQLite 管理版本、链接、会话和索引状态，资料可读、可备份、可迁移。 |
| **可信问答** | BM25、向量检索、RRF、Rerank 和证据门禁共同筛选答案来源；证据不足时明确拒答。 |
| **智能执行** | Agent 根据当前任务选择搜索、总结、页面操作等能力，后端负责参数校验、权限、预算和确认。 |
| **受控外部研究** | 本地知识不足时才启用 MCP Search/Fetch，外部内容先形成研究草稿，再由用户决定是否写入。 |
| **可恢复运行** | 长任务支持流式输出、断线续传、取消、超时和重启后的终态恢复。 |
| **知识纠错闭环** | 回答反馈、证据快照、修订草稿、确认写入、重新索引和原问题复测形成完整闭环。 |
| **可靠语音采集** | 支持本地 Whisper、DashScope 和 MiMo；统一音频格式、记录时间戳，并限制重复转录和本地并发。 |
| **本地优先与隐私** | 数据和索引默认保存在本机；日志脱敏，在线模型和外部研究按配置显式启用。 |

## 技术底座

| 领域 | 技术 |
| --- | --- |
| 前端 | React 19、Vite、React Markdown、Lucide React、SSE |
| 后端 | Python 3.11、FastAPI、Uvicorn、Pydantic v2、LangGraph |
| 数据 | SQLite、SQLAlchemy、Markdown、YAML Front Matter |
| 检索 | BM25、Jieba、ChromaDB、RRF、Embedding、Rerank、CRAG |
| Agent | Plan-and-Execute、工具注册表、JSON Schema、有限 Replan、MCP Python SDK |
| 采集 | ffmpeg/ffprobe、faster-whisper、DashScope ASR、MiMo ASR、pdfplumber、python-docx |
| 运维 | 结构化 JSON 日志、Request ID、运行追踪、Server-Timing、Pytest、Docker |

数据库当前使用 SQLite 与同步 SQLAlchemy；异步主要用于 FastAPI、SSE、Agent/MCP 编排和索引 Worker。这个配置适合单用户本地部署。

## 开始使用

环境要求：Python 3.11+、Node.js 20+、ffmpeg，以及本地模型或兼容的在线模型服务。

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

应用默认地址为 `http://127.0.0.1:8337`。模型、Embedding、Rerank、ASR 和 MCP 配置统一放在 `.env`，不配置模型时，知识库管理等非模型功能仍可使用。

## 运行管理

前端“运行追踪”可以查看查询可信度、阶段耗时、模型用量、Token 预算、RAG 证据和错误码。服务重启后，已结束的 Agent 运行仍可从 SQLite 中恢复。

日志默认写入 `data/logs/app.log` 和 `data/logs/error.log`，每个文件达到 `10 MB` 后轮转，各保留 `7` 个备份，单类日志最多约 `80 MB`。日志按文件大小控制，不按记录条数控制；密钥、Prompt、Wiki 正文、转写全文和外部网页正文不会写入日志。

## 数据与隐私

- Markdown 和 SQLite 是可备份的业务数据；BM25、ChromaDB 是可以重建的派生索引。
- API Key 只放在本地 `.env`，不会提交到仓库。
- 启用在线 Embedding、Rerank、LLM 或 MCP 时，相关查询和候选内容会发送到配置的服务端点。
- 当前定位是单用户、本机或可信内网部署，不包含多租户权限和多实例状态共享。

## 产品文档

- [系统架构](docs/architecture/overview.md)
- [日志与运行追踪](docs/runbooks/observability.md)
- [RAG 评测设计](docs/eval/01-evaluation-design.md)
- [RAG 评测流程](docs/eval/02-evaluation-workflow.md)
- [评测代码说明](test/eval/README.md)
- [Markdown 主数据决策](docs/adr/0001-markdown-source-of-truth.md)
- [可信 Agent 决策](docs/adr/0003-trusted-agent.md)
- [受控 MCP 研究决策](docs/adr/0004-controlled-mcp-research.md)

## License

[MIT License](LICENSE)
