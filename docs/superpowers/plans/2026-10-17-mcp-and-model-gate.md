# 内置只读 MCP Server 与模型并发闸门 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为智语增加一个 stdio 只读 MCP Server，并用 Redis 分布式信号量约束所有进程合计的模型并发调用。

**Architecture:** MCP Server 是独立进程（`python -m backend.app.mcp_server`），只复用现有 `PageService` 与 `HybridRetrievalService`，不启动 FastAPI、不消费队列。并发闸门是一个独立的 Redis ZSET 信号量模块，挂在 LLM / Embedding / Reranker 三个服务的实际调用点，Redis 异常时降级放行。

**Tech Stack:** Python 3.11、MCP SDK 2.0.0（`mcp.server.mcpserver.MCPServer`）、redis-py 5.x、SQLAlchemy 2.x、pytest/unittest、FastAPI（仅复用其服务层）。

**Spec:** `docs/superpowers/specs/2026-10-17-mcp-and-model-gate-design.md`

## Global Constraints

- 中文注释；代码文件 UTF-8 无 BOM。
- 测试使用真实 MySQL 测试库；闸门测试使用假 Redis 客户端，不依赖真实 Redis。
- 新增配置字段必须加进 `backend/app/core/config.py` 的 `Settings`（`extra="forbid"`），并同步 `.env.example`。
- MCP 工具严格只读；不得触发 LLM、不得写库、不得访问外部网络。
- 闸门默认值：LLM=4、Embedding=8、Rerank=4；`MODEL_GATE_ENABLED=true`；等待超时 30 秒；租约 300 秒。
- 默认配置下现有行为不变；Redis 故障时闸门降级放行，不抛错。
- 每个工具方法用 `asyncio.to_thread` 执行同步逻辑，并在 `finally` 关闭 Session。
- 提交信息使用 Conventional Commits（`feat:` / `test:` / `docs:`）。

## Review Focus

1. **MCP 客户端传入空查询或超大 `top_k`** —— 期望返回空列表或按上限截断，而不是异常或超长输出。
2. **`read_page` 传入不存在/有歧义的标题** —— 期望返回可读的工具错误文本，而不是堆栈。
3. **Redis 在闸门获取中途断开** —— 期望立即降级放行，问答不中断，日志出现限频告警。
4. **流式 LLM 在生成中被取消** —— 期望闸门许可被释放，不发生泄漏导致后续请求永久排队。
5. **闸门达到上限且长时间无释放** —— 期望租约到期清理僵尸成员，等待方最终拿到许可或超时放行。

---

## File Structure

| 文件 | 动作 | 职责 |
| --- | --- | --- |
| `backend/app/core/config.py` | 修改 | 新增 MCP Server 与 `MODEL_GATE_*` 配置字段 |
| `.env.example` | 修改 | 同步新增配置及说明 |
| `backend/app/core/model_gate.py` | 创建 | Redis ZSET 分布式信号量：`ModelGate`、`acquire_model_slot`、`reset_for_tests` |
| `backend/app/services/ai/llm_service.py` | 修改 | `_chat_completion` 与 `stream_chat` 接入闸门 |
| `backend/app/services/ai/embedding_service.py` | 修改 | `encode_batch` 每批接入闸门 |
| `backend/app/services/ai/reranker_service.py` | 修改 | `rerank` 接入闸门 |
| `backend/app/mcp_server/__init__.py` | 创建 | 包导出 |
| `backend/app/mcp_server/tools.py` | 创建 | 三个只读工具的注入式实现 |
| `backend/app/mcp_server/server.py` | 创建 | 构建 `MCPServer` 并注册工具 |
| `backend/app/mcp_server/__main__.py` | 创建 | `python -m backend.app.mcp_server` 入口 |
| `test/unit/core/test_model_gate.py` | 创建 | 闸门单测 |
| `test/unit/mcp_server/test_tools.py` | 创建 | MCP 工具单测 |
| `README.md` | 修改 | 新增 MCP Server 与闸门使用说明 |
| `docs/architecture/overview.md` | 修改 | 补充模型调用经闸门的说明 |

---

### Task 1: 配置字段

**Files:**
- Modify: `backend/app/core/config.py`（模型网关配置段之后）
- Modify: `.env.example`（`RERANKER_*` 段之后）
- Test: `test/unit/core/test_model_gate.py`（仅先建文件验证配置可读）

**Interfaces:**
- Produces:
  - `settings.mcp_server_enabled: bool = False`
  - `settings.mcp_server_max_content_chars: int = 8000`
  - `settings.model_gate_enabled: bool = True`
  - `settings.model_gate_llm_max_concurrency: int = 4`
  - `settings.model_gate_embedding_max_concurrency: int = 8`
  - `settings.model_gate_rerank_max_concurrency: int = 4`
  - `settings.model_gate_wait_timeout_seconds: float = 30.0`
  - `settings.model_gate_lease_seconds: int = 300`

- [ ] **Step 1: 在 `Settings` 中新增字段**

在 `reranker_timeout_seconds` 之后加入：

```python
    # 内置只读 MCP Server（stdio）。默认关闭，避免与外部研究用的 MCP 客户端混淆。
    mcp_server_enabled: bool = False
    mcp_server_max_content_chars: int = Field(default=8_000, ge=500, le=100_000)

    # 分布式模型并发闸门：按 provider + model 限制所有进程合计的并发调用数。
    # 默认值与 arq_agent_max_jobs 对齐；Redis 异常时降级放行，不阻断业务。
    model_gate_enabled: bool = True
    model_gate_llm_max_concurrency: int = Field(default=4, ge=1, le=128)
    model_gate_embedding_max_concurrency: int = Field(default=8, ge=1, le=256)
    model_gate_rerank_max_concurrency: int = Field(default=4, ge=1, le=128)
    model_gate_wait_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    model_gate_lease_seconds: int = Field(default=300, ge=10, le=3600)
```

- [ ] **Step 2: 在 `.env.example` 中同步配置**

在 `RERANKER_TIMEOUT_SECONDS=30` 之后加入：

```
# 内置只读 MCP Server（stdio）。启动方式：python -m backend.app.mcp_server
MCP_SERVER_ENABLED=false
MCP_SERVER_MAX_CONTENT_CHARS=8000

# 分布式模型并发闸门。限制 API 与各 worker 合计的并发模型调用数。
# Redis 不可用时自动降级放行，不会阻断问答。
MODEL_GATE_ENABLED=true
MODEL_GATE_LLM_MAX_CONCURRENCY=4
MODEL_GATE_EMBEDDING_MAX_CONCURRENCY=8
MODEL_GATE_RERANK_MAX_CONCURRENCY=4
MODEL_GATE_WAIT_TIMEOUT_SECONDS=30
MODEL_GATE_LEASE_SECONDS=300
```

- [ ] **Step 3: 写测试验证配置默认值**

创建 `test/unit/core/test_model_gate.py`：

```python
"""分布式模型并发闸门的单元测试。"""

import unittest

from backend.app.core.config import settings


class ModelGateConfigTest(unittest.TestCase):
    def test_默认值与设计一致(self):
        self.assertTrue(settings.model_gate_enabled)
        self.assertEqual(settings.model_gate_llm_max_concurrency, 4)
        self.assertEqual(settings.model_gate_embedding_max_concurrency, 8)
        self.assertEqual(settings.model_gate_rerank_max_concurrency, 4)
        self.assertEqual(settings.model_gate_wait_timeout_seconds, 30.0)
        self.assertEqual(settings.model_gate_lease_seconds, 300)
```

- [ ] **Step 4: 运行测试**

Run: `python -m pytest test/unit/core/test_model_gate.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/config.py .env.example test/unit/core/test_model_gate.py
git commit -m "feat: add MCP server and model gate configuration"
```

---

### Task 2: Redis 分布式信号量

**Files:**
- Create: `backend/app/core/model_gate.py`
- Test: `test/unit/core/test_model_gate.py`

**Interfaces:**
- Consumes: `settings.model_gate_*`（Task 1）
- Produces:
  - `class ModelGate`
  - `ModelGate(kind: str, provider: str, model: str, *, client_factory=None)`
  - `ModelGate.acquire(timeout_seconds: float | None = None) -> bool`
  - `ModelGate.release() -> None`
  - `ModelGate.__enter__() / __exit__(...)` —— 上下文管理器，`acquire` 返回 False 时仍放行（降级语义）
  - `acquire_model_slot(kind, provider, model)` —— 上下文管理器便捷函数
  - `reset_for_tests()` —— 清空进程内客户端缓存
  - `MODEL_GATE_KEY = "zhiyu:model_gate:{kind}:{provider}:{model}"`

- [ ] **Step 1: 写失败测试**

在 `test/unit/core/test_model_gate.py` 追加：

```python
import threading
import time
from unittest.mock import patch

from backend.app.core import model_gate
from backend.app.core.model_gate import ModelGate


class FakeRedis:
    """只实现闸门用到的 zset / expire / eval 子集。"""

    def __init__(self):
        self.zsets = {}
        self.expires = {}
        self.fail = False
        self._lock = threading.Lock()

    def ping(self):
        if self.fail:
            raise ConnectionError("redis down")
        return True

    def zremrangebyscore(self, key, low, high):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            members = self.zsets.get(key, {})
            removed = [m for m, s in members.items() if float(low) <= s <= float(high)]
            for m in removed:
                del members[m]
            return len(removed)

    def zcard(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return len(self.zsets.get(key, {}))

    def zadd(self, key, mapping):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            self.zsets.setdefault(key, {}).update(mapping)
        return 1

    def zrem(self, key, member):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            self.zsets.get(key, {}).pop(member, None)
        return 1

    def expire(self, key, seconds):
        if self.fail:
            raise ConnectionError("redis down")
        self.expires[key] = seconds
        return True


class ModelGateTest(unittest.TestCase):
    def setUp(self):
        model_gate.reset_for_tests()

    def tearDown(self):
        model_gate.reset_for_tests()

    def _gate(self, fake, limit=2, kind="llm"):
        return ModelGate(
            kind,
            "primary",
            "test-model",
            client_factory=lambda: fake,
            max_concurrency=limit,
            wait_timeout_seconds=1.0,
            lease_seconds=60,
        )

    def test_未达上限时可获取并释放(self):
        fake = FakeRedis()
        gate = self._gate(fake)
        self.assertTrue(gate.acquire())
        self.assertEqual(fake.zcard(gate.key), 1)
        gate.release()
        self.assertEqual(fake.zcard(gate.key), 0)

    def test_达到上限后在释放前拿不到许可(self):
        fake = FakeRedis()
        first = self._gate(fake, limit=1)
        second = self._gate(fake, limit=1)
        self.assertTrue(first.acquire())
        self.assertFalse(second.acquire(timeout_seconds=0.2))
        first.release()
        self.assertTrue(second.acquire(timeout_seconds=1.0))
        second.release()

    def test_租约过期后僵尸许可被清理(self):
        fake = FakeRedis()
        gate = self._gate(fake, limit=1)
        key = gate.key
        fake.zsets[key] = {"dead-holder": time.time() - 120}
        self.assertTrue(gate.acquire(timeout_seconds=1.0))
        self.assertNotIn("dead-holder", fake.zsets[key])
        gate.release()

    def test_redis异常时降级放行(self):
        fake = FakeRedis()
        fake.fail = True
        gate = self._gate(fake, limit=1)
        self.assertTrue(gate.acquire())
        gate.release()  # 释放失败也不能抛

    def test_关闭开关时不访问redis(self):
        fake = FakeRedis()
        gate = self._gate(fake)
        with patch.object(model_gate.settings, "model_gate_enabled", False):
            gate.acquire()
            self.assertEqual(fake.zsets, {})

    def test_上下文管理器释放许可(self):
        fake = FakeRedis()
        gate = self._gate(fake, limit=1)
        with gate:
            self.assertEqual(fake.zcard(gate.key), 1)
        self.assertEqual(fake.zcard(gate.key), 0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/unit/core/test_model_gate.py -v`
Expected: FAIL（`ModuleNotFoundError: backend.app.core.model_gate` 或 `ImportError: ModelGate`）

- [ ] **Step 3: 实现 `backend/app/core/model_gate.py`**

实现要点：

- `MODEL_GATE_KEY = "zhiyu:model_gate:{kind}:{provider}:{model}"`；模型名做 `str.strip()`，空值用 `"unknown"`。
- 进程级同步 `redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=0.3, socket_timeout=2.0)`，用 `threading.Lock` 缓存；`ping()` 失败时抛错，由调用方降级。
- `acquire()`：
  1. `settings.model_gate_enabled` 为假 → 返回 True，不触碰 Redis。
  2. 计算 `limit`：`kind == "llm"` → `settings.model_gate_llm_max_concurrency`；`embedding` → `settings.model_gate_embedding_max_concurrency`；`rerank` → `settings.model_gate_rerank_max_concurrency`；其他 → 1。
  3. 循环直到 `deadline = monotonic() + timeout`：`zremrangebyscore(key, 0, now - lease)` → `zcard(key)`；若小于 limit 则 `zadd(key, {member: now})` 并 `expire(key, lease)`，返回 True；否则 `time.sleep(0.05)`。
  4. 超时返回 False（调用方**仍放行**，见上下文管理器语义）。
  5. 任何 Redis 异常：记录**限频** warning（同一 gate 60 秒最多一条，用 `time.monotonic()` + `threading.Lock`），返回 True。
- `release()`：若未持有许可直接返回；`zrem` 失败只记 debug 日志，不抛。
- 成员值：`f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"`。
- `__enter__` 调 `acquire()` 并保存 `self._held = True`；`__exit__` 调 `release()`。`acquire` 返回 False 时也把 `_held` 设为 False，`release` 无操作。
- `acquire_model_slot(kind, provider, model)` 返回 `ModelGate(kind, provider, model)`。
- `reset_for_tests()` 关闭并清空缓存的客户端。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/unit/core/test_model_gate.py -v`
Expected: PASS（7 个用例）

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/model_gate.py test/unit/core/test_model_gate.py
git commit -m "feat: add redis-backed distributed model concurrency gate"
```

---

### Task 3: LLM 接入闸门

**Files:**
- Modify: `backend/app/services/ai/llm_service.py`（`_chat_completion`、`stream_chat`）
- Test: `test/unit/ai/test_llm_model_gate.py`（新建）

**Interfaces:**
- Consumes: `acquire_model_slot(kind, provider, model)`（Task 2）
- Produces: 无新公开接口；`LLMService` 行为不变，仅内部多一层许可。

- [ ] **Step 1: 写失败测试**

创建 `test/unit/ai/test_llm_model_gate.py`：

```python
"""LLM 服务接入并发闸门的测试：获取与释放必须成对。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.app.services.ai.llm_service import LLMService


def _fake_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class LLMGateTest(unittest.TestCase):
    def setUp(self):
        with patch("backend.app.services.ai.llm_service.OpenAI"):
            self.service = LLMService()

    def test_chat_获取并释放许可(self):
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        self.service.client.chat.completions.create.return_value = _fake_response()
        with patch(
            "backend.app.services.ai.llm_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            self.assertEqual(self.service.chat([{"role": "user", "content": "hi"}]), "ok")
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[0], "llm")
        gate.__exit__.assert_called_once()

    def test_主模型失败切备用时各持一次许可(self):
        fallback = MagicMock()
        self.service.fallback_client = fallback
        self.service.fallback_model = "fallback-model"
        self.service.client.chat.completions.create.side_effect = TimeoutError("boom")
        fallback.chat.completions.create.return_value = _fake_response("fallback")
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        with patch(
            "backend.app.services.ai.llm_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            self.assertEqual(self.service.chat([{"role": "user", "content": "hi"}]), "fallback")
        self.assertEqual(factory.call_count, 2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/unit/ai/test_llm_model_gate.py -v`
Expected: FAIL（`AttributeError: ... acquire_model_slot` 未被调用）

- [ ] **Step 3: 在 LLM 调用点接入闸门**

- `import`：`from backend.app.core.model_gate import acquire_model_slot`。
- `_chat_completion`：把 `client.chat.completions.create(**attempt_kwargs)` 包进
  `with acquire_model_slot("llm", provider, model):`（`provider` 用循环里的 `"primary"/"fallback"`，
  `model` 用本次尝试的模型名）。
- `stream_chat`：同样包住 `stream = client.chat.completions.create(**kwargs)` 与整段
  `for chunk in stream:` 迭代。用 `with` 保证 `GeneratorExit` / 取消 / 异常时释放。
- 不要给 `chat_json` / `call_function` / `structured_call` 再加一层——它们都走
  `_chat_completion` 或 `chat`，重复加锁会导致同一进程内自锁。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest test/unit/ai/test_llm_model_gate.py -v`
Expected: PASS

- [ ] **Step 5: 回归 LLM 相关测试**

Run: `python -m pytest test/unit/ai -v`
Expected: PASS（原有测试不受影响）

- [ ] **Step 6: 提交**

```bash
git add backend/app/services/ai/llm_service.py test/unit/ai/test_llm_model_gate.py
git commit -m "feat: gate llm calls with distributed concurrency limit"
```

---

### Task 4: Embedding 与 Rerank 接入闸门

**Files:**
- Modify: `backend/app/services/ai/embedding_service.py`（`OpenAICompatibleEmbeddingBackend.encode_batch`）
- Modify: `backend/app/services/ai/reranker_service.py`（`RerankCompatibleBackend.rerank`）
- Test: `test/unit/ai/test_embedding_rerank_gate.py`（新建）

**Interfaces:**
- Consumes: `acquire_model_slot`（Task 2）
- Produces: 无新公开接口。

- [ ] **Step 1: 写失败测试**

创建 `test/unit/ai/test_embedding_rerank_gate.py`：

```python
"""Embedding / Rerank 接入并发闸门的测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.app.services.ai.embedding_service import OpenAICompatibleEmbeddingBackend
from backend.app.services.ai.reranker_service import RerankCompatibleBackend


class EmbeddingGateTest(unittest.TestCase):
    def test_每个批次获取一次许可(self):
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
        )
        backend = OpenAICompatibleEmbeddingBackend(
            api_key="k", api_url="https://example.com/v1",
            model="m", batch_size=1, client=client,
        )
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        with patch(
            "backend.app.services.ai.embedding_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            backend.encode_batch(["a", "b"])
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args.args[0], "embedding")


class RerankGateTest(unittest.TestCase):
    def test_请求前获取许可(self):
        session = MagicMock()
        session.post.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {"results": [{"index": 0, "relevance_score": 0.9}]},
        )
        backend = RerankCompatibleBackend(
            api_key="k", api_url="https://example.com/rerank",
            model="m", session=session,
        )
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        with patch(
            "backend.app.services.ai.reranker_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            backend.rerank("q", ["a"], 1)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[0], "rerank")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/unit/ai/test_embedding_rerank_gate.py -v`
Expected: FAIL（`acquire_model_slot` 未被调用）

- [ ] **Step 3: 接入闸门**

- `embedding_service.py`：`from backend.app.core.model_gate import acquire_model_slot`；
  在 `encode_batch` 的 `for start in range(...)` 循环体内，把
  `self.client.embeddings.create(**params)` 包进
  `with acquire_model_slot("embedding", self.provider_name, self.model_name):`。
- `reranker_service.py`：同样的 import；把
  `self.session.post(...)` 包进
  `with acquire_model_slot("rerank", self.provider_name, self.model_name):`。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest test/unit/ai/test_embedding_rerank_gate.py -v`
Expected: PASS

- [ ] **Step 5: 回归**

Run: `python -m pytest test/unit/ai -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add backend/app/services/ai/embedding_service.py backend/app/services/ai/reranker_service.py test/unit/ai/test_embedding_rerank_gate.py
git commit -m "feat: gate embedding and rerank calls"
```

---

### Task 5: MCP 工具实现

**Files:**
- Create: `backend/app/mcp_server/__init__.py`
- Create: `backend/app/mcp_server/tools.py`
- Test: `test/unit/mcp_server/test_tools.py`（新建）

**Interfaces:**
- Consumes: `PageService.find_page`、`PageService.list_pages`、
  `HybridRetrievalService.search_hybrid`、`settings.mcp_server_max_content_chars`
- Produces:
  - `class WikiMCPTools`
  - `WikiMCPTools(*, page_service_factory=None, retrieval_factory=None, session_factory=None)`
  - `async WikiMCPTools.search_wiki(query: str, top_k: int = 5) -> dict`
  - `async WikiMCPTools.read_page(page_id_or_title: str) -> dict`
  - `async WikiMCPTools.list_pages(notebook: str | None = None, tag: str | None = None, query: str | None = None, limit: int = 20) -> dict`
  - 所有方法返回 `{"ok": bool, ...}`；失败时 `{"ok": False, "error": "<可读文本>"}`

- [ ] **Step 1: 写失败测试**

创建 `test/unit/mcp_server/test_tools.py`：

```python
"""MCP 只读工具的单元测试：不连接 MySQL，全部使用假服务。"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from backend.app.mcp_server.tools import WikiMCPTools
from backend.app.services.wiki.page_errors import PageNotFoundError


def _run(coro):
    return asyncio.run(coro)


class WikiMCPToolsTest(unittest.TestCase):
    def _tools(self, retrieval=None, page=None, session=None):
        return WikiMCPTools(
            page_service_factory=lambda _db: page,
            retrieval_factory=lambda: retrieval,
            session_factory=lambda: session or MagicMock(),
        )

    def test_search_wiki_精简结果并遵守top_k(self):
        retrieval = MagicMock()
        retrieval.search_hybrid.return_value = [
            {"page_id": "p1", "title": "T1", "section_title": "S1",
             "content": "x" * 100, "score": 0.9, "tags": ["a"],
             "source_url": "/api/pages/p1"},
        ]
        tools = self._tools(retrieval=retrieval)
        result = _run(tools.search_wiki("查询", top_k=3))
        self.assertTrue(result["ok"])
        self.assertEqual(retrieval.search_hybrid.call_args.kwargs["top_k"], 3)
        self.assertEqual(result["results"][0]["page_id"], "p1")

    def test_search_wiki_空查询返回空结果(self):
        retrieval = MagicMock()
        tools = self._tools(retrieval=retrieval)
        result = _run(tools.search_wiki("   "))
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"], [])
        retrieval.search_hybrid.assert_not_called()

    def test_search_wiki_top_k超限被截断(self):
        retrieval = MagicMock()
        retrieval.search_hybrid.return_value = []
        tools = self._tools(retrieval=retrieval)
        _run(tools.search_wiki("q", top_k=999))
        self.assertEqual(retrieval.search_hybrid.call_args.kwargs["top_k"], 50)

    def test_read_page_正文超限截断(self):
        page = MagicMock()
        page.find_page.return_value = {
            "id": "p1", "title": "标题", "status": "active",
            "revision": 2, "tags": [], "content": "字" * 20_000,
        }
        tools = self._tools(page=page)
        result = _run(tools.read_page("标题"))
        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["content"]), 8000)

    def test_read_page_不存在时返回可读错误(self):
        page = MagicMock()
        page.find_page.side_effect = PageNotFoundError("页面不存在")
        tools = self._tools(page=page)
        result = _run(tools.read_page("没有这页"))
        self.assertFalse(result["ok"])
        self.assertIn("不存在", result["error"])
        self.assertNotIn("Traceback", result["error"])

    def test_list_pages_只返回摘要字段(self):
        page = MagicMock()
        page.list_pages.return_value = {
            "total": 1, "items": [
                {"id": "p1", "title": "T", "notebook": "n", "tags": [],
                 "status": "active", "revision": 1, "updated_at": "x",
                 "content": "正文不应出现"},
            ],
        }
        tools = self._tools(page=page)
        result = _run(tools.list_pages(notebook="n", limit=5))
        self.assertTrue(result["ok"])
        self.assertEqual(result["total"], 1)
        self.assertNotIn("content", result["pages"][0])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/unit/mcp_server/test_tools.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 `tools.py`**

实现要点：

- 构造参数默认值在 `__init__` 内延迟 import（避免导入 `backend.app.mcp_server` 时初始化
  模型服务）：
  ```python
  from backend.app.core.database import SessionLocal
  from backend.app.services.retrieval.hybrid_retrieval_service import get_hybrid_retrieval_service
  from backend.app.services.wiki.page_service import get_page_service
  ```
- 每个公开方法都是 `async def`，内部：
  ```python
  def _run() -> dict:
      db = self._session_factory()
      try:
          ...  # 同步逻辑
      finally:
          db.close()
  return await asyncio.to_thread(_run)
  ```
- `search_wiki`：`query` 去空白；空 → `{"ok": True, "results": []}`；
  `top_k = max(1, min(int(top_k), 50))`；调用 `search_hybrid(query, top_k=top_k, db=db)`
  （注意现有接口需要 `db`，传入线程内新建的 Session）；把每条结果裁剪为
  `page_id/title/section_title/content/snippet/score/tags/source_url`。
- `read_page`：`page_service.find_page(reference)`；正文按
  `settings.mcp_server_max_content_chars` 截断并设 `truncated`；返回
  `id/title/status/revision/tags/notebook/content/truncated/file_path 不返回`。
- `list_pages`：`limit = max(1, min(int(limit), 100))`；调用
  `page_service.list_pages(notebook=..., tag=..., query=..., offset=0, limit=limit)`；
  每项只保留 `id/title/notebook/tags/status/revision/updated_at`。
- 异常处理：`PageNotFoundError` / `AmbiguousPageError` / `PageValidationError` /
  `EmbeddingProviderError` / `RerankerProviderError` → `{"ok": False, "error": str(exc)}`；
  其他异常 → `{"ok": False, "error": "工具执行失败"}` 并 `logger.exception`。
- `__init__.py` 导出 `WikiMCPTools`。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest test/unit/mcp_server/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/mcp_server/__init__.py backend/app/mcp_server/tools.py test/unit/mcp_server/test_tools.py
git commit -m "feat: add read-only wiki tools for mcp server"
```

---

### Task 6: MCP Server 入口

**Files:**
- Create: `backend/app/mcp_server/server.py`
- Create: `backend/app/mcp_server/__main__.py`
- Test: `test/unit/mcp_server/test_server.py`（新建）

**Interfaces:**
- Consumes: `WikiMCPTools`（Task 5）、`settings.mcp_server_enabled`
- Produces:
  - `create_server(tools: WikiMCPTools | None = None) -> MCPServer`
  - 三个工具名：`search_wiki`、`read_page`、`list_pages`
  - `python -m backend.app.mcp_server` 启动 stdio

- [ ] **Step 1: 写失败测试**

创建 `test/unit/mcp_server/test_server.py`：

```python
"""MCP Server 组装测试：工具注册与开关。"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.mcp_server.server import create_server


class CreateServerTest(unittest.TestCase):
    def test_注册三个只读工具(self):
        tools = MagicMock()
        tools.search_wiki = AsyncMock(return_value={"ok": True, "results": []})
        tools.read_page = AsyncMock(return_value={"ok": True})
        tools.list_pages = AsyncMock(return_value={"ok": True, "pages": []})
        server = create_server(tools)
        names = {tool.name for tool in _list_tools(server)}
        self.assertEqual(names, {"search_wiki", "read_page", "list_pages"})


def _list_tools(server):
    import asyncio
    return asyncio.run(server.list_tools())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/unit/mcp_server/test_server.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 `server.py` 与 `__main__.py`**

`server.py`：

```python
"""构建只读 MCP Server：三个工具全部委托给 WikiMCPTools。"""

from mcp.server.mcpserver import MCPServer

from backend.app.mcp_server.tools import WikiMCPTools
from backend.app.core.config import settings


def create_server(tools: WikiMCPTools | None = None) -> MCPServer:
    tools = tools or WikiMCPTools()
    server = MCPServer(
        name="zhiyu",
        version=settings.app_version,
        instructions="智语本地知识库的只读检索工具，不提供写入能力。",
    )

    @server.tool()
    async def search_wiki(query: str, top_k: int = 5) -> dict:
        """在智语知识库中做混合检索（BM25 + 向量）。返回带页面来源的片段。"""
        return await tools.search_wiki(query, top_k)

    @server.tool()
    async def read_page(page_id_or_title: str) -> dict:
        """按页面 ID、标题或别名读取完整 Markdown 正文。"""
        return await tools.read_page(page_id_or_title)

    @server.tool()
    async def list_pages(
        notebook: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> dict:
        """分页列出知识库页面摘要，可按笔记本、标签或关键词筛选。"""
        return await tools.list_pages(notebook, tag, query, limit)

    return server
```

`__main__.py`：

```python
"""stdio 启动入口：python -m backend.app.mcp_server"""

import logging
import sys

from backend.app.core.config import settings
from backend.app.mcp_server.server import create_server


def main() -> int:
    if not settings.mcp_server_enabled:
        print("MCP Server 未启用：请在 .env 中设置 MCP_SERVER_ENABLED=true", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    create_server().run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 运行测试**

Run: `python -m pytest test/unit/mcp_server/test_server.py -v`
Expected: PASS

- [ ] **Step 5: 手工验证开关**

Run: `MCP_SERVER_ENABLED=false python -m backend.app.mcp_server`
Expected: 打印未启用提示，退出码 1（不阻塞、不启动 stdio）

- [ ] **Step 6: 提交**

```bash
git add backend/app/mcp_server/server.py backend/app/mcp_server/__main__.py test/unit/mcp_server/test_server.py
git commit -m "feat: add stdio mcp server entrypoint"
```

---

### Task 7: 文档与整体回归

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/overview.md`

**Interfaces:**
- Consumes: 前六个任务的全部产物。

- [ ] **Step 1: 更新 `README.md`**

在「运行管理」之后新增两节：

```markdown
## MCP Server（只读）

智语可以把知识库以标准 MCP 工具暴露给 Claude Code、Cursor 等外部 AI 工具，
仅提供 `search_wiki`、`read_page`、`list_pages` 三个只读能力，不写库、不调用 LLM。

```bash
# .env 中启用
MCP_SERVER_ENABLED=true

# 启动（stdio，由 MCP 客户端拉起）
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

API 与三个 worker 会同时调用模型网关。闸门用 Redis 按 `provider + model`
限制所有进程合计的并发数，避免触发供应商 429：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_GATE_ENABLED` | `true` | 总开关 |
| `MODEL_GATE_LLM_MAX_CONCURRENCY` | `4` | LLM 并发上限 |
| `MODEL_GATE_EMBEDDING_MAX_CONCURRENCY` | `8` | Embedding 并发上限 |
| `MODEL_GATE_RERANK_MAX_CONCURRENCY` | `4` | Rerank 并发上限 |
| `MODEL_GATE_WAIT_TIMEOUT_SECONDS` | `30` | 等待许可的最长时间，超时放行 |
| `MODEL_GATE_LEASE_SECONDS` | `300` | 许可租约，进程崩溃后自动回收 |

Redis 不可用时闸门自动降级放行，问答与索引不受影响。
```

- [ ] **Step 2: 更新 `docs/architecture/overview.md`**

在「部署形态」的架构图之后补一段：

```markdown
所有在线模型调用（LLM / Embedding / Rerank）都经过 Redis 上的分布式并发闸门
（`zhiyu:model_gate:*`）。闸门按 `provider + model` 计数，限制的是四个应用进程
的合计并发，而不是单进程并发。Redis 故障时闸门降级放行。
```

在「模块结构」表格中 `services/ai/*` 行后补一行：

```markdown
| `backend/app/core/model_gate.py` | Redis ZSET 分布式信号量，按模型限制跨进程并发 | 不决定模型选择，不重试；Redis 异常降级放行 |
```

- [ ] **Step 3: 全量回归**

Run: `docker compose up -d mysql && python -m pytest test/unit -v`
Expected: PASS

- [ ] **Step 4: 手工验证 MCP stdio**

1. `.env` 设 `MCP_SERVER_ENABLED=true`。
2. 在 Claude Code 中按 README 配置 stdio MCP。
3. 调用 `list_pages` → 应返回页面摘要；调用 `search_wiki` → 应返回带 `page_id` 的片段；
   调用 `read_page` 传入一个标题 → 应返回正文。
4. 停止 Redis（`docker compose stop redis`），再发起一次问答 → 应正常完成，日志出现
   `模型闸门` 降级告警；恢复 Redis。

- [ ] **Step 5: 提交**

```bash
git add README.md docs/architecture/overview.md
git commit -m "docs: document mcp server and model concurrency gate"
```

---

## Self-Review

- **Spec coverage:** 5.1 传输（Task 6）、5.2 工具契约（Task 5）、5.3 模块结构（Task 5/6）、
  5.4 闸门（Task 1/2/3/4）、5.5 可观测性（Task 2 限频告警 + 现有 `timed_stage`）、
  6 测试（各任务）、7 文档（Task 7）。无遗漏。
- **Step scan:** 每个实现步骤都给了签名、文件位置与具体数值；算法只在闸门一处给出。
- **Type consistency:** `acquire_model_slot(kind, provider, model)` 在三处调用一致；
  `WikiMCPTools` 方法签名在 Task 5 定义、Task 6 使用一致。
- **Review Focus:** 空查询/超限 top_k（Task 5 测试）、页面不存在（Task 5 测试）、
  Redis 断开（Task 2 测试）、流式取消释放（Task 3 `with` 保证，测试覆盖
  `__exit__` 调用）、租约回收（Task 2 测试）。
- **Proportion:** 计划长度约为设计的 1.5 倍，属于实现决策密集的正常范围。
