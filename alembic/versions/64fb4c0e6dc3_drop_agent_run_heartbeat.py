"""移除 agent_runs 的 heartbeat_at 心跳字段

Revision ID: 64fb4c0e6dc3
Revises: 92ecd0fa614d
Create Date: 2026-10-14

回退 Stage 7 加的运行心跳。

那个字段是为多实例服务的：多个 worker 并存时，启动恢复需要区分「这条运行
是我上次跑挂的」和「是别的 worker 正在跑的」。做法是执行期间每 30 秒写一次
心跳，恢复时只回收心跳过期的运行。

项目最终定位是单实例（API 一份进程、三种 worker 各一份），这个前提不成立：
worker 只有一个，它启动了就等于「数据库里所有 running/cancelling 都是它上次
跑挂的」，直接全部回收就是正确行为，不需要额外判据。心跳因此在这个部署形态
下只是每 30 秒一次的无效写入。

原来的 92ecd0fa614d 已经把这个列加到线上的库里了，所以这里要显式删掉，
而不是把那条迁移删掉了事——迁移是历史，已经执行过的不该改写。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision: str = '64fb4c0e6dc3'
down_revision: Union[str, Sequence[str], None] = '92ecd0fa614d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_agent_runs_heartbeat_at"


def upgrade() -> None:
    """删索引、删列。"""
    op.drop_index(INDEX_NAME, table_name="agent_runs")
    op.drop_column("agent_runs", "heartbeat_at")


def downgrade() -> None:
    """把列和索引加回来，已有行的心跳置空。

    不回填历史值：心跳的唯一用途是判断执行者是否还活着，而能走到这里的
    都是已经结束的运行，给它们编一个心跳时间没有意义，还可能误导排查。
    """
    is_mysql = op.get_bind().dialect.name == "mysql"
    # MySQL 要 6 位小数秒，与 DateTimeMs 的定义保持一致；SQLite 用普通 DateTime
    column_type = mysql.DATETIME(fsp=6) if is_mysql else sa.DateTime(timezone=True)

    op.add_column(
        "agent_runs",
        sa.Column("heartbeat_at", column_type, nullable=True),
    )
    op.create_index(INDEX_NAME, "agent_runs", ["heartbeat_at"])
