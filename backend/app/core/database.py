"""
数据库连接管理

只对接 MySQL。SQLite 已在分布式改造中移除：它是嵌入式单文件库，多个进程
同时写同一个文件会锁竞争甚至损坏，而应用现在会跑出 4 个进程。schema 一律
由 Alembic 管理，应用启动不执行任何 DDL（见 lifecycle）。
"""
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings


def _create_engine() -> Engine:
    """构建 MySQL 引擎。

    MySQL 是独立进程，每次建连都要经过 TCP 握手与认证，耗时以十毫秒计，
    必须复用连接，否则高并发下连接创建本身就会成为瓶颈。
    """
    url = make_url(settings.database_url)
    connect_args: dict = {}
    if url.get_backend_name() == "mysql":
        # 不显式指定时，部分环境会协商成 utf8mb3，导致 emoji 与少数生僻字写入失败
        connect_args["charset"] = "utf8mb4"

    return create_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        # 取连接前先发一次探活。pool_recycle 只能回收"按时间推算可能已失效"的连接，
        # 防不住网络抖动、服务端重启、中间代理提前断开导致的静默失效，
        # 两者必须并用，缺一都会在低峰期后的第一个请求上抛连接错误。
        pool_pre_ping=True,
        connect_args=connect_args,
    )


# 创建数据库引擎
engine = _create_engine()

# 创建基类
Base = declarative_base()

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()