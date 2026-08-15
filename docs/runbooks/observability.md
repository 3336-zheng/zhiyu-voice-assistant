# 日志与运行追踪

## 定位入口

开发环境优先从前端“运行追踪”页面定位问题。选择近期请求或输入 Request ID 后，可以查看请求状态、总耗时、阶段时间线、稳定错误码和模型调用次数。追踪接口默认只允许本机访问；远程部署前应先增加认证，不建议直接开放。

文件日志用于进一步排查：

| 位置 | 内容 |
| --- | --- |
| `data/logs/app.log` | 单行 JSON 应用日志，包含正常请求与业务事件 |
| `data/logs/error.log` | 仅记录 ERROR 及以上事件 |
| 终端标准错误 | 适合本地开发时实时观察 |

## 排查流程

1. 在失败响应或前端消息中取得 Request ID。
2. 在“运行追踪”中查询该 ID，确认失败阶段、错误码、耗时和是否可重试。
3. 在 `app.log` 与 `error.log` 中按 `request_id` 检索同一链路。
4. 根据 `component`、`operation` 和 `error_code` 定位到对应业务域。
5. 对可重试的模型、检索或索引故障先确认上游状态；对版本冲突和参数错误修正输入后再请求。

示例：

```bash
rg '"request_id":"your-request-id"' data/logs/app.log data/logs/error.log
rg '"error_code":"RERANK_PROVIDER_ERROR"' data/logs/error.log
```

## 关键配置

```env
LOG_LEVEL=INFO
LOG_FILE=data/logs/app.log
LOG_ERROR_FILE=data/logs/error.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=7

OBSERVABILITY_ENABLED=true
OBSERVABILITY_CAPTURE_CONTENT=false
OBSERVABILITY_TRACE_API_ENABLED=true
OBSERVABILITY_TRACE_ALLOW_REMOTE=false
```

日志按文件大小轮转。保持 `OBSERVABILITY_CAPTURE_CONTENT=false` 时，追踪只记录统计和阶段状态；API Key、Prompt、Wiki 正文、转写全文及外部网页正文不应进入日志。Langfuse 与 OpenTelemetry 是可选导出层，默认关闭，导出异常不会影响主请求。

## 错误边界

稳定错误码用于分类，不替代底层异常。常见类别包括模型供应商、Embedding、Rerank、检索、MCP、Agent 执行、索引和页面版本冲突。界面只展示脱敏后的摘要，完整堆栈仅保存在本机错误日志中。

当前运行追踪的内存快照适合近期请求；已结束的 Agent Run 会持久化终态、时间线、检索统计和模型用量，服务重启后仍可查询。普通 HTTP 请求的近期时间线不会跨重启保留。
