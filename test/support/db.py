"""测试用的数据库工具。

测试连的是**真实的 MySQL 服务**上的一个独立库（默认 `zhiyu_test`），不是内存库。
这样做是为了让测试和生产跑在同一种方言上——MySQL 特有的行为（字符集协商、
utf8mb4 下的索引键长度上限、JSON 字段语义）能在测试里暴露，而不是等上线才炸。

**代价**：跑测试前必须先起 MySQL 容器：

    docker compose up -d mysql

容器没起时，`make_test_engine()` 会抛出一个说人话的错误，而不是让 40 个测试
各自甩一堆 ConnectionRefusedError。

**每个测试类拿到一个干净的空库**：`create_all` 之前先 `drop_all`，所以测试之间
没有残留。因为表结构是现建的，测试不依赖 `alembic upgrade`（迁移链的完整性由
`test_alembic_migration.py` 单独把关）。

**必须串行跑**：所有测试共用同一个库，如果以后引入 pytest-xdist 并行，多个进程
会互相 `drop_all` 打架。届时要么给每个 worker 分配不同的库名，要么改成
"建一次库 + 事务回滚"的模式。
"""

import os
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url, URL
from sqlalchemy.pool import NullPool

from backend.app.core.database import Base
from backend.app import models  # noqa: F401  导入以触发全部 ORM 表注册

DEFAULT_TEST_DATABASE = "zhiyu_test"


class TestDatabaseUnavailable(RuntimeError):
    """测试库连不上。"""


def build_test_database_url() -> str:
    """测试库的连接串。

    优先取 TEST_DATABASE_URL；没设时从应用的 DATABASE_URL 推导——
    只把库名换掉，其余（host / port / 账号 / 密码 / charset）原样保留，
    所以 .env 里怎么配的，测试就跟着怎么连，不用维护第二份凭据。
    """
    explicit = os.getenv("TEST_DATABASE_URL")
    if explicit:
        return explicit

    from backend.app.core.config import settings

    url = make_url(settings.database_url)
    if url.get_backend_name() != "mysql":
        raise TestDatabaseUnavailable(
            f"测试只支持 MySQL，当前 DATABASE_URL 指向 {url.get_backend_name()}。"
        )
    return url.set(database=DEFAULT_TEST_DATABASE).render_as_string(hide_password=False)


def make_test_engine() -> Engine:
    """建一个连测试库、并已清空重建全部表的引擎。

    用 NullPool：每个测试类各自建引擎，几百个连接如果被池缓存着不放，
    会很快耗光 MySQL 的 max_connections。用完必须 `engine.dispose()`。

    `drop_all` 清不掉 `alembic_version`——它不在 ORM 元数据里。这里显式删掉，
    免得迁移测试在"上一次升级过的库"上跑，看不到真实的从零升级过程。
    """
    url = build_test_database_url()
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
    except Exception as exc:  # noqa: BLE001 — 转成能看懂的话再抛
        engine.dispose()
        raise TestDatabaseUnavailable(
            f"连不上测试库 {_safe(url)}。\n"
            "跑测试前请先起 MySQL：docker compose up -d mysql\n"
            f"原始错误：{exc}"
        ) from exc
    return engine


def _safe(url: str) -> str:
    """把连接串里的密码藏掉再拼进报错信息。"""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 — 拼错了也不该盖住原始错误
        return "<无法解析的连接串>"
