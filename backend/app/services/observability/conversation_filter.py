"""运行追踪的「对话可见性」过滤。

抽成独立函数是为了能被测试直接调：过滤条件是 SQL 层的 LIKE 转义加子查询
并集，塞在路由函数里就只能靠起服务打接口来验。

**判据是两个条件的并集**，单独任何一个都会漏：

    session_ 前缀   前端生成的真实会话 ID 固定是这个格式（见 useAgentRun 的
                    getSessionId），而 bench-*/perf-*/codex-* 这类脚本 ID
                    一个都不符合。但将来若有别的入口用别的 ID 格式，这个就漏。
    有真实消息       现在问一句落两条消息，8 月那版不落——所以只看消息会把
                    那批真人历史一起滤掉，只看 ID 则漏掉新入口。

两个条件是**平级**的：子查询里不能再套前缀条件。最初就是这么写错的——
子查询也加了 `session_` 过滤，于是「有消息」分支被自己筛空，并集实际只剩
前缀那一半，静默少了一半会话（test_conversation_filter 里有用例钉住）。
"""

from __future__ import annotations

from sqlalchemy import or_, select

from backend.app.models.conversation import ConversationMessage
from backend.app.models.observability import AgentRun

# 下划线在 LIKE 里是单字符通配符，必须转义——不转义的话 "sessionXabc"
# 也会被当成真实会话放进来。
SESSION_ID_PREFIX = "session\\_%"
ESCAPE = "\\"


def apply_conversation_visibility_filter(query, *, include_empty: bool):
    """给对话列表查询加上可见性过滤。

    query 必须是以 AgentRun 为主表的查询（过滤条件建在 AgentRun.session_id 上）。

    **必须接住返回值**：SQLAlchemy 的 `Query.filter()` 返回新对象，不原地改。
    丢掉返回值等于过滤没生效，而且不会报错——最初就踩了这一点。
    """
    if include_empty:
        return query
    return query.filter(
        or_(
            AgentRun.session_id.like(SESSION_ID_PREFIX, escape=ESCAPE),
            AgentRun.session_id.in_(select(ConversationMessage.session_id).distinct()),
        )
    )
