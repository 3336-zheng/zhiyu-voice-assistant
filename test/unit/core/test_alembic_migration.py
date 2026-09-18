"""校验 Alembic 迁移链与 ORM 元数据保持一致。

取代原先的 test_schema_v6.py：schema.py 的手写迁移链依赖 SQLite 专有语法
（CREATE INDEX IF NOT EXISTS、INTEGER PRIMARY KEY AUTOINCREMENT），
在 MySQL 上会直接报 1064 语法错误，已由 Alembic 取代。
"""

import tempfile
import unittest
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from backend.app import models  # noqa: F401  导入以触发全部 ORM 表注册
from backend.app.core.database import Base

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AlembicBaselineTest(unittest.TestCase):
    """迁移必须能从空库建出与 ORM 定义完全一致的结构。"""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.url = f"sqlite:///{self._tmpdir.name}/probe.db"
        self.config = Config(str(PROJECT_ROOT / "alembic.ini"))
        self.config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
        # env.py 只在地址为空时才从应用配置注入，这里显式指定以打到临时库
        self.config.set_main_option("sqlalchemy.url", self.url)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

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
        开发库（SQLite 走 create_all，永远跟着 ORM）与生产库（MySQL 走 Alembic）
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
