# 智语 RAG 评测

评测目标不是搜索最优 Chunk 或 Top-K，而是验证固定工程配置能否支撑可信知识问答：正确证据能否召回并前置，最终回答是否忠于证据，知识库无答案时能否拒答、纠正前提或进入 MCP 外部研究。

## 数据划分

评测由两种文件组成，但它们不是训练集和测试集：

```text
真实文档语料库
  私有 Markdown，全部进入同一个检索索引

统一 Question 评测集
  可回答问题 + 无答案/错误前提问题
```

- `data/obsidian_sources.json`：真实文档来源、知识领域、文档类型和正例配额。
- `data/question_plan.json`：Question 类型、领域配额、评测范围和生成规则。
- `dataset_manifest.py`：检查来源路径、Markdown 类型、敏感值、文件哈希和配额闭合。

来源覆盖 RAG 与检索、Agent、Embedding 与向量库、Python 后端、工程实践和智语项目设计。面试问题汇总、旧评测报告和命中真实凭据规则的文档不进入语料库。

Question 分为五类，实际配额由本地规划文件决定：

| 类型 | 验证能力 |
| --- | --- |
| `keyword` | BM25 对专有名词、缩写和配置名的检索 |
| `semantic_rewrite` | Embedding 对自然语言改写的语义召回 |
| `multi_evidence` | 多事实、多片段证据覆盖 |
| `similar_concept` | RRF 与 Rerank 对相近概念的区分 |
| `unanswerable` | CRAG/Evidence Gate 的拒答、纠错和 MCP 路由 |

所有 Question 保存在同一个文件中，通过 `question_type` 和 `expected_action` 区分，不再拆开发集、测试集或参数调优集。

## Question 契约

可回答问题必须包含原文证据：

```json
{
  "id": "q-rag-001",
  "query": "替换向量模型后为什么要重建索引？",
  "category": "embedding_vector_store",
  "question_type": "semantic_rewrite",
  "expected_action": "answer",
  "reference_answer": "仅依据原文证据整理的参考答案",
  "reference_claims": ["旧向量与新查询向量不在同一语义空间"],
  "relevance": {"parent-chunk-id": 3},
  "evidence": [
    {
      "source_id": "真实相对路径.md",
      "source_sha256": "...",
      "start_char": 1200,
      "end_char": 1500,
      "quote": "原文精确子串"
    }
  ]
}
```

无答案问题不得伪造参考答案、相关文档或证据：

```json
{
  "id": "q-negative-001",
  "query": "智语当前 pgvector 的 HNSW 参数是什么？",
  "category": "zhiyu_project",
  "question_type": "unanswerable",
  "expected_action": "correct_premise"
}
```

允许的 `expected_action` 为 `answer`、`reject`、`correct_premise` 和 `external_research`。通用检索评测器只计算 `answer` 样本；其余样本交给证据门禁和 Agent 路由评测。

## 固定配置

```env
RAG_PARENT_CHUNK_CHARS=1200
RAG_PARENT_CHUNK_OVERLAP_CHARS=120
RAG_CHILD_CHUNK_CHARS=500
RAG_CHILD_CHUNK_OVERLAP_CHARS=80
BM25_TOP_K=30
EMBEDDING_TOP_K=30
RRF_K=60
RRF_TOP_K=30
RAG_FINAL_TOP_K=5
RAG_CONTEXT_TOKEN_BUDGET=3000
```

Markdown 优先按标题和段落切分，上述字符数只用于超长章节兜底。固定配置用于控制评测变量；只有失败案例出现稳定模式时才调整参数。

## 评测范围

检索层对所有可回答问题运行四级消融：

```text
BM25
Embedding
BM25 + Embedding + RRF
BM25 + Embedding + RRF + Rerank
```

主指标为 Evidence Recall@5、MRR 和检索 P95。端到端 Faithfulness 与无答案问题的门禁决策准确率单独评测，不与检索指标混合。

NDCG、Hit@K、Precision@K、模型调用量等可以保留在 JSON 诊断结果中，但不作为 README 或简历主指标。

## 清单审计

```bash
python -m test.eval.dataset_manifest \
  --source-root /path/to/obsidian \
  --manifest /path/to/source-manifest.json \
  --question-plan /path/to/question-plan.json \
  --output data/eval/dataset-plan-audit.json
```

审计失败时命令返回非零状态，并指出来源越界、文件不存在、敏感值命中或配额不一致。审计只读取本地文档，不调用 LLM、Embedding 或 Rerank。

## 检索评测

先执行一次可重复的数据准备命令。该命令会完成来源审计、按固定父子分块冻结语料、通过当前 LLM 的 Function Calling 生成 Question、校验证据 ID 与字符区间，并将来源幂等写入 Wiki 后建立在线 Embedding 索引：

```bash
python -m test.eval.prepare_dataset \
  --source-root /path/to/obsidian \
  --manifest /path/to/source-manifest.json \
  --question-plan /path/to/question-plan.json \
  --output-dir data/eval/current
```

正式文件只有 `corpus.jsonl`、`questions.jsonl` 和 `profile.json`。任何来源、分块参数或模型配置变化都应重新生成，不手工拼接旧评测产物。命令只在整套数据通过校验后替换输出目录；Wiki 通过稳定 `source_uri` 幂等写入，重复执行不会制造副本。`--skip-questions`、`--skip-wiki` 和 `--skip-index` 仅用于定位准备阶段的问题，不用于正式评测。

正式 Question 生成并冻结后，使用通用入口运行：

```bash
python -m test.eval.rag_eval \
  --corpus data/eval/current/corpus.jsonl \
  --queries data/eval/current/questions.jsonl \
  --dataset-name local-evaluation \
  --methods bm25,embedding,hybrid,hybrid_reranker \
  --top-k 5 \
  --k-values 1,3,5 \
  --output data/eval/current/retrieval-report.json
```

模型调用异常计入失败率。报告只记录模型名称、检索参数和调用工作量，不记录 API Key 或网关地址。Question 由 AI 基于真实证据生成并由程序校验，参考事实必须可回查原文，不能对外宣称人工标注。
