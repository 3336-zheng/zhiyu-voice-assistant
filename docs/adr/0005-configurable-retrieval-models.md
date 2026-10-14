# ADR-0005：可配置检索模型与向量空间隔离

- 状态：已采用
- 日期：2026-08-11
- 更新：2026-10-14 —— 本地模型后端（BGE）已移除，只保留在线 Provider

## 背景

智语最初直接加载本地 BGE Embedding 和 BGE Reranker。该方式适合离线部署，但要求设备具备足够的内存或显存，也无法使用企业模型网关提供的在线检索模型。

Embedding 与 Rerank 在数据边界上并不相同：Embedding 同时决定索引向量和查询向量的语义空间，Rerank 只处理已经召回的文本候选。

如果在同一 Chroma 集合中混用不同 Embedding 模型，即使维度相同，向量也不可比；维度不同时则会直接写入失败。查询失败时自动切换 Embedding 同样会造成查询向量与存量索引不匹配。

## 决策

保留 `EmbeddingService` 和 `RerankerService` 作为稳定门面，调用方继续使用 `encode`、`encode_queries`、`encode_documents` 和 `rerank`，门面按部署配置选择实现。

**当前只保留在线实现**——本地模型后端已移除，理由与移除本地 whisper 一致：多进程下每个进程各加载一份模型，内存和显存都不划算；代码里用来限并发的信号量又只在单进程内有效，拦不住跨进程。

- Embedding 的 `provider` 只有 `openai_compatible` 一个取值；Rerank 的只有 `rerank_compatible`。
- **这两个字段保留不删**：它们参与向量集合名的哈希计算。删掉字段就等于改变了哈希输入，会生成一个全新的集合，已有向量看起来像"消失"了。这是「配置项兼作数据标识」的典型陷阱。
- 在线 Embedding 使用批量 OpenAI Embeddings 协议，校验结果数量与维度；空文本不发送给模型，结果按原输入位置补回零向量。
- 在线 Rerank 使用 `model`、`query`、`documents`、`top_n` 与 `return_documents` 请求契约，兼容 Vercel、Jina 和 SiliconFlow 的 Cohere/Jina 风格协议。
- Embedding 不进行跨模型自动降级；索引失败交给持久化索引任务重试。
- Embedding Profile 由 Provider、端点、模型和维度计算，映射到独立 Chroma 集合；API Key 不参与标识。
- 切换 Embedding 后由部署者显式重建索引，避免未经确认产生批量在线费用。
- Rerank 可以独立切换，不需要重建索引，但证据阈值需要结合模型分数分布校准。

模型端点属于部署配置而不是用户输入。端点必须为 HTTP(S)，URL 不允许携带凭证；API Key 只从服务端环境加载，日志和健康检查不输出密钥、端点响应正文或内部异常详情。

## 结果

- 没有本地 GPU 的环境可以通过模型网关完成索引与精排。
- 不同 Embedding 模型和维度不会污染同一向量集合，切换回旧配置仍可访问旧集合。
- 改造后的集合名形如 `notes-ae3b14f344b5`，与原 `notes` 是两个不同的集合——升级时旧集合仍在，但检索走的是新集合。
- 在线模型会增加网络延迟、调用成本和数据出站风险，必须使用可信端点。
- 新 Profile 首次启用时检索集合为空，需要显式重建索引。
- 移除本地后端后，升级时 `.env` 里残留的 `EMBEDDING_MODEL_PATH` / `RERANKER_MODEL_PATH` 必须删掉——配置模型是 `extra="forbid"`，留着会导致启动失败。
- 未实现企业级多端点负载均衡、配额和计费；这些能力不属于当前单用户部署边界。
