# 智语 - 端侧智能语音笔记助手

端侧部署的智能语音笔记助手，集成语音转写、Plan-and-Execute Agent 多轮对话、混合检索（BM25 + Embedding + RRF + Reranker）、纪要生成和文档管理，支持本地 Whisper 与云端百炼 ASR 双引擎切换。

GitHub: https://github.com/3336-zheng/zhiyu-voice-assistant

---

## 功能亮点

- **双 ASR 引擎** -- 本地 Whisper（faster-whisper / CTranslate2 量化）与阿里百炼 DashScope API，按需切换
- **Plan-and-Execute Agent** -- LLM 驱动的意图识别 + 多步规划 + 工具执行，支持检索、笔记 CRUD、摘要总结、MD 文件生成等 11 种意图
- **四阶段混合检索** -- BM25 与 Embedding 并行检索，RRF 融合排序，BGE-reranker 精排，检索精度显著优于单一方法
- **多轮对话记忆** -- SQLite 持久化会话历史，支持摘要压缩和过期清理
- **纪要总结** -- 一键将转录文字生成会议纪要 / 课堂笔记 / 通用笔记，支持预览编辑后保存并自动索引
- **文档管理** -- 支持上传 md / txt / pdf / docx，自动分块索引到向量库
- **前端三合一** -- 纯 HTML/CSS/JS 单页应用，问题检索 + 纪要总结 + 文档管理三个页面

---

## 系统架构

```
用户语音/文本
      |
      v
  FastAPI 后端 (端口 8337)
      |
      +-- 音频模块: ffmpeg 转码 -> Whisper/DashScope ASR -> 转录文本
      |
      +-- Agent 模块: Planner(LLM 意图识别) -> Executor(工具调度) -> Responder(答案生成)
      |       |
      |       +-- 混合检索: BM25(jieba分词) + Embedding(BGE) + RRF融合 + Reranker(BGE-reranker-v2-m3)
      |       |
      |       +-- 笔记工具: 创建/更新/删除/列出/详情 (Markdown 文件)
      |       |
      |       +-- 时间查询 / 摘要总结
      |
      +-- 纪要模块: LLM 生成纪要 -> 预览 -> 保存到 data/docs/ -> 自动索引
      |
      +-- 文档模块: 上传/删除文档 -> 分块 -> ChromaDB + BM25 双索引
      |
      +-- 记忆模块: SQLite 会话表 + 消息历史 + 摘要压缩 + 过期清理
      |
      v
  纯前端 (HTML/CSS/JS) -- 问题检索 / 纪要总结 / 文档管理
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Python 3.11 + FastAPI + Uvicorn |
| 数据库 | SQLite (SQLAlchemy ORM) |
| 向量存储 | ChromaDB |
| 关键词检索 | rank-bm25 + jieba 分词 |
| Embedding 模型 | BGE (BAAI) |
| Reranker 模型 | BGE-reranker-v2-m3 |
| ASR (本地) | faster-whisper (CTranslate2 量化) |
| ASR (云端) | 阿里百炼 DashScope paraformer-realtime-v2 |
| LLM | OpenAI 兼容接口 (默认 DeepSeek) |
| 音频处理 | ffmpeg + librosa |
| 文档解析 | pdfplumber / python-docx |
| 前端 | 原生 HTML / CSS / JavaScript |
| 容器化 | Docker + Docker Compose |

---

## 快速开始

### 环境要求

- Python 3.11+
- ffmpeg（音频转码必需，需在 PATH 中）
- 本地模型文件（Whisper、Embedding、Reranker，路径在 `.env` 中配置）
- LLM API Key（DeepSeek / OpenAI 兼容接口）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/3336-zheng/zhiyu-voice-assistant.git
cd zhiyu-voice-assistant

# 2. 创建虚拟环境（推荐 conda）
conda create -n zhiyu python=3.11 -y
conda activate zhiyu

# 3. 安装 ffmpeg
conda install -c conda-forge ffmpeg -y

# 4. 安装 Python 依赖
pip install -r requirements.txt

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env，填入模型路径和 API Key
```

### 配置 .env

在项目根目录创建 `.env` 文件，参考以下关键配置项：

```env
# 模型路径（替换为你的实际路径）
WHISPER_MODEL_PATH=C:/models/whisper-small-finetune-ct2--int8
EMBEDDING_MODEL_PATH=D:/models/BGE
RERANKER_MODEL_PATH=D:/models/BGE-reranker-v2-m3

# ASR 引擎 (whisper 或 dashscope)
ASR_PROVIDER=whisper

# 百炼 ASR（仅 asr_provider=dashscope 时需要）
DASHSCOPE_ASR_API_KEY=your_api_key_here

# LLM 配置
LLM_API_KEY=your_deepseek_api_key
LLM_API_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 启动服务

```bash
python main.py
```

服务默认监听 `http://localhost:8337`，前端页面直接访问该地址即可。

API 文档地址：`http://localhost:8337/api/docs`

---

## Docker 部署

### 使用 Docker Compose（推荐）

```bash
# 确保 .env 文件已配置好
docker compose up --build
```

服务启动后访问 `http://localhost:8337`。

### 数据持久化

Docker Compose 会自动挂载以下目录/文件：

| 宿主机路径 | 容器路径 | 说明 |
|-----------|---------|------|
| `./.env` | `/app/.env` | 环境变量（只读） |
| `./data` | `/app/data` | 数据库、上传文件、向量库、日志 |
| 模型目录（.env 中配置） | `/app/models/*` | Whisper / Embedding / Reranker 模型（只读） |

### 单独构建 Docker 镜像

```bash
docker build -t zhiyu-assistant .
docker run -p 8337:8337 --env-file .env -v ./data:/app/data zhiyu-assistant
```

---

## 项目结构

```
.
├── main.py                          # 应用入口
├── requirements.txt                 # Python 依赖
├── Dockerfile                       # Docker 镜像定义
├── docker-compose.yml               # Docker Compose 编排
├── .env.example                     # 环境变量模板
│
├── backend/
│   └── app/
│       ├── __init__.py              # FastAPI 应用初始化、路由注册、中间件
│       ├── core/
│       │   ├── config.py            # 配置管理（pydantic-settings）
│       │   └── database.py          # SQLAlchemy 数据库引擎
│       ├── api/
│       │   ├── __init__.py          # 路由导出
│       │   ├── agent.py             # Agent 对话 / 混合检索 / 会话管理
│       │   ├── audio.py             # 音频上传 / 转录 / 润色
│       │   ├── notes.py             # 笔记 CRUD（md 文件）
│       │   ├── docs.py              # 文档管理（上传 / 索引）
│       │   ├── summary.py           # 纪要生成 / 保存
│       │   └── health.py            # 健康检查
│       ├── agent/
│       │   ├── models.py            # Agent 数据模型（意图、工具、计划、响应）
│       │   ├── planner.py           # Plan-and-Execute 规划器
│       │   ├── executor.py          # 工具执行器
│       │   ├── responder.py         # 响应生成器
│       │   ├── agent.py             # Agent 主控
│       │   └── markdown_agent.py    # Markdown 文件操作代理
│       ├── services/
│       │   ├── whisper_service.py   # ASR 服务（Whisper / DashScope 工厂）
│       │   ├── llm_service.py       # LLM 服务（OpenAI 兼容接口）
│       │   ├── embedding_service.py # BGE Embedding 服务
│       │   ├── bm25_service.py      # BM25 关键词检索
│       │   ├── reranker_service.py  # BGE-reranker 精排
│       │   ├── rrf_service.py       # RRF 融合排序
│       │   ├── hybrid_retrieval_service.py  # 混合检索整合
│       │   ├── retrieval_service.py # 基础检索服务
│       │   ├── chroma_service.py    # ChromaDB 向量存储
│       │   ├── doc_index_service.py # 文档索引（分块 + 向量化 + BM25）
│       │   └── memory_service.py    # 对话记忆（SQLite）
│       └── models/
│           ├── __init__.py          # SQLAlchemy 模型导出
│           └── audio.py             # Audio 表模型
│
├── frontend/
│   ├── index.html                   # 问题检索页
│   ├── summary.html                 # 纪要总结页
│   ├── docs.html                    # 文档管理页
│   └── style.css                    # 全局样式
│
├── data/                            # 运行时数据（已 gitignore）
│   ├── database/                    # SQLite 数据库
│   ├── uploads/                     # 上传的音频文件
│   ├── notes/                       # 笔记 md 文件
│   ├── docs/                        # 文档 md/txt 文件
│   └── logs/                        # 应用日志
│
└── test/                            # 测试与基准
    ├── benchmark_retrieval.py       # 检索基准测试
    └── benchmark_results_retrieval.json
```

---

## API 端点概览

所有端点的请求/响应 schema 可在 `http://localhost:8337/api/docs` (Swagger UI) 中查看。

### Agent 智能助手 (`/agent`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/agent/chat/` | Agent 多轮对话（支持 session_id 持久化记忆） |
| POST | `/agent/search/` | 混合检索（BM25 + Embedding + RRF + Reranker） |
| POST | `/agent/compare/` | 对比三种检索方式的结果 |
| GET | `/agent/sessions/` | 列出所有会话 |
| DELETE | `/agent/sessions/{session_id}` | 清除指定会话 |
| POST | `/agent/sessions/cleanup` | 手动清理过期会话 |
| GET | `/agent/sessions/stats` | 获取会话统计信息 |

### 音频管理 (`/audio`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/audio/upload/` | 上传音频文件（自动转码为 16kHz WAV） |
| POST | `/audio/transcribe/{audio_id}` | 转录音频（支持 `?provider=whisper\|dashscope`） |
| POST | `/audio/polish/{audio_id}` | 润色转录文本（去口头禅、补标点、纠错） |
| DELETE | `/audio/{audio_id}` | 删除音频文件 |
| GET | `/audio/asr-providers` | 获取可用 ASR 引擎列表 |

### 笔记管理 (`/notes`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/notes/create-file` | 创建笔记 md 文件 |
| GET | `/notes/list` | 分页列出笔记 |
| GET | `/notes/{filename}` | 获取笔记详情 |
| PUT | `/notes/{filename}` | 编辑笔记 |
| DELETE | `/notes/{filename}` | 删除笔记 |
| GET | `/notes/search/?query=xxx` | 检索相关笔记 |

### 纪要总结 (`/summary`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/summary/generate` | 生成纪要（仅预览，不存储） |
| POST | `/summary/save` | 保存纪要到文档目录并自动索引 |

### 文档管理 (`/api/documents`)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/documents/list` | 获取文档列表 |
| POST | `/api/documents/upload` | 上传文档（md / txt / pdf / docx），自动索引 |
| DELETE | `/api/documents/{filename}` | 删除文档 |

### 健康检查 (`/health`)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health/` | 服务健康状态 |

---

## 配置说明

所有配置项均可通过 `.env` 文件设置，由 `pydantic-settings` 自动加载。

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `APP_NAME` | 智语端侧智能语音笔记助手 | 应用名称 |
| `HOST` | 0.0.0.0 | 监听地址 |
| `PORT` | 8337 | 监听端口 |
| `DEBUG` | true | 调试模式（启用热重载） |
| `DATABASE_URL` | sqlite:///data/database/notes.db | SQLite 数据库路径 |
| `WHISPER_MODEL_PATH` | -- | Whisper 模型路径（本地 ASR 必填） |
| `EMBEDDING_MODEL_PATH` | -- | BGE Embedding 模型路径 |
| `RERANKER_MODEL_PATH` | -- | BGE-reranker 模型路径 |
| `ASR_PROVIDER` | whisper | ASR 引擎：`whisper`（本地）或 `dashscope`（云端） |
| `DASHSCOPE_ASR_API_KEY` | -- | 百炼 DashScope API Key |
| `DASHSCOPE_ASR_MODEL` | paraformer-realtime-v2 | 百炼 ASR 模型名 |
| `LLM_API_KEY` | -- | LLM API Key（必填） |
| `LLM_API_URL` | https://api.deepseek.com/v1 | LLM API 地址 |
| `LLM_MODEL` | deepseek-chat | LLM 模型名 |
| `LLM_MAX_TOKENS` | 2048 | LLM 最大输出 token 数 |
| `LLM_TEMPERATURE` | 0.7 | LLM 生成温度 |
| `RRF_K` | 60.0 | RRF 融合常数 |
| `RRF_TOP_K` | 10 | RRF 融合后候选数 |
| `BM25_TOP_K` | 20 | BM25 检索数量 |
| `EMBEDDING_TOP_K` | 20 | Embedding 检索数量 |
| `AGENT_MAX_ITERATIONS` | 5 | Agent 最大执行轮数 |
| `MEMORY_MAX_HISTORY` | 20 | 最大保留对话轮数 |
| `MEMORY_SUMMARY_THRESHOLD` | 10 | 触发摘要压缩的轮数阈值 |
| `SESSION_TTL_HOURS` | 24 | 会话过期时间（小时） |
| `CLEANUP_INTERVAL_MINUTES` | 60 | 过期会话清理间隔（分钟） |
| `MAX_FILE_SIZE` | 52428800 | 上传文件大小限制（字节，默认 50MB） |
| `ALLOWED_EXTENSIONS` | .wav,.mp3,.flac,.ogg,.webm | 允许的音频格式 |
| `LOG_LEVEL` | INFO | 日志级别 |

---

## 开发指南

### 本地开发

```bash
# 启动开发服务器（自动热重载）
python main.py
```

`DEBUG=true` 时 Uvicorn 会启用 `reload`，代码修改后自动重启。

### API 调试

启动后访问 Swagger UI：`http://localhost:8337/api/docs`

### 对话示例

Agent 支持自然语言交互，以下是一些示例：

```
# 检索知识库
"查找关于 RAG 的笔记"

# 创建笔记
"创建笔记标题是会议记录内容是今天的讨论要点"

# 总结并写入文档
"总结关于向量数据库的内容写成md文档"

# 按日期搜索
"查看上周的笔记"

# 纪要操作
"生成一份会议纪要"   -> 使用 /summary/generate
"保存这份纪要"       -> 使用 /summary/save
```

### 检索基准测试

```bash
python test/benchmark_retrieval.py
```

测试结果保存在 `test/benchmark_results_retrieval.json`。

---

## License

MIT License

Copyright (c) 2026 3336-zheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
