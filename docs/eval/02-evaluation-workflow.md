# RAG 评测执行流程

## 目录职责

```text
test/eval/
├── data/                      # 可公开的小型示例与评测规划
├── dataset_manifest.py        # 来源、安全与配额审计
├── prepare_dataset.py         # 语料和 Golden Question 生成
├── dataset.py                 # 数据契约与加载
├── real_retriever.py          # 真实检索适配器
├── retrieval_metrics.py       # 指标计算
└── rag_eval.py                # 通用评测入口

data/eval/                     # 本地生成数据，不提交 Git
```

## 一、准备来源清单

来源清单记录相对于知识库根目录的 Markdown 路径、领域、文档类型和正例配额。不要把 API Key、Token、SSH 配置、账号资料、旧评测报告或面试题库加入来源。

题型规划定义各领域的 `keyword`、`semantic_rewrite`、`multi_evidence`、`similar_concept` 和 `unanswerable` 配额，使问题分布由配置控制，而不是由模型自由决定。

## 二、审计来源

```bash
python -m test.eval.dataset_manifest \
  --source-root /path/to/knowledge-base \
  --manifest /path/to/source-manifest.json \
  --question-plan /path/to/question-plan.json \
  --output data/eval/dataset-plan-audit.json
```

审计检查路径边界、Markdown 类型、重复来源、敏感值、文件哈希和配额闭合。该阶段不调用在线模型。

## 三、生成数据集

```bash
python -m test.eval.prepare_dataset \
  --source-root /path/to/knowledge-base \
  --manifest /path/to/source-manifest.json \
  --question-plan /path/to/question-plan.json \
  --output-dir data/eval/current
```

准备工具依次执行：

1. 规范化 Markdown；
2. 复用生产父子分块策略；
3. 记录稳定 Chunk ID、标题路径和来源字符区间；
4. 使用 Function Calling 生成证据级 Question；
5. 校验证据、题型配额和重复项；
6. 幂等写入 Wiki 并建立索引；
7. 全部成功后原子替换正式输出目录。

工具使用文档级检查点。在线调用中断后可以继续执行，不需要重新生成已完成来源。

## 四、检查本地产物

```text
data/eval/current/
├── corpus.jsonl
├── questions.jsonl
└── profile.json
```

- `corpus.jsonl` 保存真实父子分块；
- `questions.jsonl` 保存问题、期望动作和证据标签；
- `profile.json` 保存来源哈希、分块参数和模型快照。

这些文件可能包含私人知识内容，因此默认由 `.gitignore` 排除。

## 五、运行四级检索评测

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

只有 `expected_action=answer` 的问题参与检索指标计算。无答案样本需要交给 Evidence Gate 和 Agent 路由评测，不能当作普通召回失败。

报告包含配置快照、汇总指标、分类指标、逐题结果、延迟、调用量和失败率，但不保存 API Key。

## 六、重跑规则

以下变化需要重新生成语料与 Question：

- 来源文档或来源清单变化；
- 分块策略变化；
- 证据生成规则变化。

以下变化只需要重新运行检索评测：

- 更换 Embedding 或 Rerank；
- 修改召回、RRF、Top-K 或排序算法。

更换 Embedding 模型或维度后必须重新向量化全部语料，因为新旧模型不在同一向量空间。更换 Rerank 不需要重建向量索引。

## 七、公开前检查

```bash
git check-ignore data/eval/current/questions.jsonl
git status --short
```

确认以下内容未被跟踪：

- 原始知识库；
- 完整语料和 Golden Question；
- 逐题检索报告；
- `.env`、数据库和向量索引；
- API Key、Token 和本机绝对路径。
