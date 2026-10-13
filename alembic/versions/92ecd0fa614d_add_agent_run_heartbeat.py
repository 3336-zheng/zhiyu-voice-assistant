"""agent_runs 增加 heartbeat_at 心跳字段

Revision ID: 92ecd0fa614d
Revises: 5740db79fbb4
Create Date: 2026-10-13

Stage 7 用这个字段解决「多实例下启动恢复误伤」的问题。

原先的恢复逻辑是「把数据库里所有 pending/running/cancelling 的运行标记为失败」。
单进程时这是对的——进程重启了，它管的运行确实都死了。但 Agent 执行搬到 worker
之后（Stage 6d），多个 worker 同时存在，其中一个重启就会把**别的 worker 正在跑的
对话一起杀掉**。要区分「我上次跑挂的」和「别人正在跑的」，需要一个能反映
「执行者还活着」的信号，这就是 heartbeat_at。

**为什么不复用 updated_at**：状态变更、字段回写都会刷新它，判断不出执行者是
否还在续期。必须是一个只由心跳更新的独立字段。

**这条迁移不能在 SQLite 上整体跳过**（与 5740 那条 DATETIME 精度迁移不同）：
SQLite 支持 ADD COLUMN，而且 test_alembic_migration 会在 SQLite 上执行迁移后
用 compare_metadata 校验结构一致性——跳过的话会报「迁移缺了 heartbeat_at 列」。
所以这里按方言选类型，而不是按方言决定做不做。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision: str = '92ecd0fa614d'
down_revision: Union[str, Sequence[str], None] = '5740db79fbb4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_agent_runs_heartbeat_at"


def upgrade() -> None:
    """加列、建索引，并把已有行的心跳初始化成它们最后的活动时间。"""
    is_mysql = op.get_bind().dialect.name == "mysql"
    # MySQL 要 6 位小数秒，与 DateTimeMs 的定义保持一致；SQLite 用普通 DateTime
    column_type = mysql.DATETIME(fsp=6) if is_mysql else sa.DateTime(timezone=True)

    op.add_column(
        "agent_runs",
        sa.Column("heartbeat_at", column_type, nullable=True),
    )
    op.create_index(INDEX_NAME, "agent_runs", ["heartbeat_at"])

    # 已有行按「最后活动时间」初始化。
    #
    # 依赖 updated_at，而历史数据里它有可能是 NULL（这套表早期没有统一写入）。
    # 这些行的 heartbeat_at 会保持 NULL——不影响功能：它们全是终态记录，
    # 而回收逻辑只扫 running / cancelling；即使状态字段有偏差，NULL 会被当作
    # 「已过期」处理，那对一条早就结束的运行也是正确结论。
    op.execute(
        "UPDATE agent_runs SET heartbeat_at = updated_at WHERE heartbeat_at IS NULL"
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="agent_runs")
    op.drop_column("agent_runs", "heartbeat_at")
