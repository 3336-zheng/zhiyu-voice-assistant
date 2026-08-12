# RAG 评测结果记录模板

本文档只提供记录结构，不包含私人语料规模、真实模型成绩、费用或逐题结果。实际报告保存在本地 `data/eval/`。

## 实验信息

| 项目 | 记录 |
| --- | --- |
| 数据集版本 | `<local-version>` |
| 来源哈希 | 见本地 `profile.json` |
| 分块配置 | `<parent/overlap, child/overlap>` |
| Embedding | `<provider/model/dimensions>` |
| Rerank | `<provider/model>` |
| 评测时间 | `<date>` |

## 检索结果

| 链路 | Hit@5 | Evidence Recall@5 | MRR | P95 | 失败率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |
| Embedding | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |
| BM25 + Embedding + RRF | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |
| BM25 + Embedding + RRF + Rerank | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |

## 消融结论

按实际报告回答：

1. Embedding 相比 BM25 改善了哪些题型？
2. RRF 是否提高了证据覆盖？
3. Rerank 是否改善 Hit@1 或 MRR？
4. 在线模型增加了多少 P95 延迟？
5. 是否存在稳定失败的领域或题型？

不要只写“效果更好”，应同时记录绝对值、百分点变化和延迟代价。

## 成本

根据本地报告的 `model_workload` 和实际服务价格计算：

```text
Embedding 成本 = 实际计费 Token × 每 Token 价格
Rerank 成本 = 实际搜索次数 × 每次搜索价格
```

最终账单以服务商记录为准，不将本地估算写成精确成本。

## 结论边界

- Hit@K 是检索命中率，不是回答准确率；
- AI 生成并经程序校验的数据不能称为人工标注；
- 当前结果只适用于对应来源哈希和配置；
- 检索命中不代表最终回答完全忠于证据；
- Faithfulness 和无答案门禁需要单独评测。

## 简历表述模板

> 基于真实知识文档构建证据级 RAG 评测集，完成 BM25、Embedding、RRF 与 Rerank 四级消融；使用 Hit@K、Evidence Recall@K、MRR 和 P95 同时衡量召回、排序与延迟，并根据在线模型失败场景设计逐级降级链路。

只有在愿意公开且能够复现真实数字时，才在简历中补充具体指标。
