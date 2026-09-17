"""Alembic 迁移环境。

数据库地址不写在 alembic.ini 里，而是统一从应用配置读取，
避免 .env 与 alembic.ini 两处各存一份、迁移打到错误的库上。
"""

from logging.config import fileConfig
import os
import sys

from sqlalchemy import engine_from_config, pool

from alembic import context

# alembic 由项目根执行，但 sys.path 未必包含项目根，需显式加入才能导入 backend 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.core.config import settings  # noqa: E402
from backend.app.core.database import Base  # noqa: E402
from backend.app import models  # noqa: E402,F401  导入以触发全部 ORM 表注册

config = context.config

# 调用方（例如测试）已显式指定地址时不覆盖，便于把迁移打到临时库上验证。
# alembic.ini 由 configparser 解析，值中的 % 会被当作插值语法，
# 数据库密码经 URL 编码后常含 %，不转义会在读取配置时抛 InterpolationSyntaxError。
if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 文本，不连接数据库。"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库并执行迁移。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # 检测列类型变更，否则改了字段类型 autogenerate 会静默跳过
            compare_type=True,
            # SQLite 不支持 ALTER COLUMN / DROP CONSTRAINT，
            # batch 模式通过"建新表-拷数据-换名"绕开，对 MySQL 无副作用
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
