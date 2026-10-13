"""把 DATETIME 精度提升到微秒（仅 MySQL）

Revision ID: 5740db79fbb4
Revises: 913259aa108b
Create Date: 2026-09-17 15:56:58.962141

MySQL 的 DATETIME 默认精度为秒，写入带微秒的值会四舍五入到整秒。
本项目有多处记录在同一秒内生成（wiki_pages、wiki_page_revisions、
wiki_index_tasks、wiki_page_links），而 page_service 按 updated_at 排序、
wiki_index_task_service 按 created_at 取任务，精度降到秒之后先后顺序不再确定。

三个写这条迁移时才发现的约束：

1. 这条迁移**不是** autogenerate 生成的。Alembic 的类型比较不识别 DATETIME 的
   fsp 差异，即使开了 compare_type=True，对着已建好的库跑 autogenerate 得到的
   也是一条空迁移。同理 test_alembic_migration.py 的 compare_metadata 断言
   也拦不住这类精度漂移。下面 29 个列是脚本扫描 metadata 后写死的。

2. 每个 alter_column 必须带 existing_nullable：MySQL 的 MODIFY COLUMN 是整列
   定义全量替换，漏掉它会把 NOT NULL 约束一并抹掉。

3. 默认值的精度必须与列精度一致。带 DEFAULT now() 的 6 个列如果只改列类型，
   MySQL 会报 1067 Invalid default value——now() 是零精度的 CURRENT_TIMESTAMP，
   配不上 DATETIME(6)。所以要同时把默认值改成 CURRENT_TIMESTAMP(6)。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision: str = '5740db79fbb4'
down_revision: Union[str, Sequence[str], None] = '913259aa108b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """把所有 DATETIME 列改为 DATETIME(6)。"""
    # SQLite 本身就保存微秒，不需要改；而且它不支持 ALTER COLUMN，
    # 执行到这里会直接报错。测试库走的正是 SQLite，所以必须挡住。
    if op.get_bind().dialect.name != "mysql":
        return

    op.alter_column(
        "agent_pending_actions", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "agent_pending_actions", "expires_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "agent_pending_actions", "completed_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "agent_runs", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    op.alter_column(
        "agent_runs", "updated_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    op.alter_column(
        "agent_runs", "completed_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "answer_feedbacks", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "answer_feedbacks", "updated_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "answer_feedbacks", "completed_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "audios", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    op.alter_column(
        "audios", "updated_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "conversation_messages", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    op.alter_column(
        "conversations", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    op.alter_column(
        "conversations", "updated_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "retrievals", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    op.alter_column(
        "wiki_index_tasks", "next_attempt_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "wiki_index_tasks", "locked_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "wiki_index_tasks", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_index_tasks", "updated_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_pages", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_pages", "updated_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_pages", "deleted_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "external_research_runs", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "external_research_runs", "completed_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=True,
    )
    op.alter_column(
        "wiki_page_links", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_page_revisions", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "external_research_sources", "retrieved_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "external_research_sources", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_page_sources", "created_at",
        type_=mysql.DATETIME(fsp=6),
        existing_type=mysql.DATETIME(),
        existing_nullable=False,
    )


def downgrade() -> None:
    """退回秒级精度。已写入的微秒会被四舍五入丢弃，不可逆。"""
    if op.get_bind().dialect.name != "mysql":
        return

    op.alter_column(
        "agent_pending_actions", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "agent_pending_actions", "expires_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "agent_pending_actions", "completed_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "agent_runs", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP(6)"),
    )
    op.alter_column(
        "agent_runs", "updated_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP(6)"),
    )
    op.alter_column(
        "agent_runs", "completed_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "answer_feedbacks", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "answer_feedbacks", "updated_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "answer_feedbacks", "completed_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "audios", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP(6)"),
    )
    op.alter_column(
        "audios", "updated_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "conversation_messages", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP(6)"),
    )
    op.alter_column(
        "conversations", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP(6)"),
    )
    op.alter_column(
        "conversations", "updated_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "retrievals", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
        server_default=sa.text("CURRENT_TIMESTAMP"),
        existing_server_default=sa.text("CURRENT_TIMESTAMP(6)"),
    )
    op.alter_column(
        "wiki_index_tasks", "next_attempt_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "wiki_index_tasks", "locked_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "wiki_index_tasks", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_index_tasks", "updated_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_pages", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_pages", "updated_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_pages", "deleted_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "external_research_runs", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "external_research_runs", "completed_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=True,
    )
    op.alter_column(
        "wiki_page_links", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_page_revisions", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "external_research_sources", "retrieved_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "external_research_sources", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
    op.alter_column(
        "wiki_page_sources", "created_at",
        type_=mysql.DATETIME(),
        existing_type=mysql.DATETIME(fsp=6),
        existing_nullable=False,
    )
