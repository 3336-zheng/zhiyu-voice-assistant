"""校验 Alembic 迁移链与 ORM 元数据保持一致。

迁移链必须在**真实的 MySQL** 上跑通一遍：MySQL 的 DDL 不是事务性的，
中途失败会留下一个半完成的库；而且它的索引键长度上限（3072 字节）和
utf8mb4 组合起来会限制索引列的长度，这些都不是纯逻辑能验出来的。
"""

import unittest
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from backend.app import models  # noqa: F401  导入以触发全部 ORM 表注册
from backend.app.core.database import Base
from test.support.db import build_test_database_url

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _reset_schema(url: str) -> None:
    """把测试库清成真正的空库。

    `drop_all` 清不掉 `alembic_version`（它不在 ORM 元数据里），而这三个用例
    都要从"从零升级"开始——上一个用例 downgrade 到 base 之后未必留下干净状态。
    """
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
        Base.metadata.drop_all(engine)
    finally:
        engine.dispose()


class AlembicBaselineTest(unittest.TestCase):
    """迁移必须能从空库建出与 ORM 定义完全一致的结构。"""

    def setUp(self) -> None:
        self.url = build_test_database_url()
        _reset_schema(self.url)
        self.config = Config(str(PROJECT_ROOT / "alembic.ini"))
        self.config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
        # env.py 只在地址为空时才从应用配置注入，这里显式指定以打到测试库
        self.config.set_main_option("sqlalchemy.url", self.url)

    def tearDown(self) -> None:
        _reset_schema(self.url)

    def test_upgrade_creates_all_orm_tables(self) -> None:
        command.upgrade(self.config, "head")
        engine = create_engine(self.url)
        try:
            actual = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()
        self.assertEqual(set(Base.metadata.tables) | {"alembic_version"}, actual)

    def test_migration_matches_orm_metadata(self) -> None:
        """迁移执行完毕后不应再存在待生成的差异。

        改了 ORM 却忘记生成迁移时这里会失败。没有这道断言，
        测试库（走 create_all，永远跟着 ORM）与生产库（走 Alembic）
        会悄悄分叉，直到线上执行到缺失的列才暴露。
        """
        command.upgrade(self.config, "head")
        engine = create_engine(self.url)
        try:
            with engine.connect() as connection:
                context = MigrationContext.configure(
                    connection, opts={"compare_type": True}
                )
                diff = compare_metadata(context, Base.metadata)
        finally:
            engine.dispose()
        self.assertEqual(
            [],
            diff,
            f"ORM 与迁移链不一致，需执行 alembic revision --autogenerate：{diff}",
        )

    def test_downgrade_removes_all_tables(self) -> None:
        """回滚必须干净，否则失败的发布无法退回上一版本。"""
        command.upgrade(self.config, "head")
        command.downgrade(self.config, "base")
        engine = create_engine(self.url)
        try:
            remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
        finally:
            engine.dispose()
        self.assertEqual(set(), remaining)


if __name__ == "__main__":
    unittest.main()
