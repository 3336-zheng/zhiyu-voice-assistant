"""
数据库连接管理
"""
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings


def is_sqlite() -> bool:
    """当前是否运行在 SQLite 上。建表策略与迁移方式都依赖这个判断。"""
    return make_url(settings.database_url).get_backend_name() == "sqlite"


def _create_engine() -> Engine:
    """按数据库方言构建引擎。

    SQLite 是进程内直接读写的文件，打开代价是微秒级，没有"建立连接"这回事，
    连接池参数对它没有意义；SQLAlchemy 2.0 对文件型 SQLite 默认已使用 QueuePool
    并自行处理跨线程访问，因此这里不传任何额外参数，保持既有行为不变。

    MySQL 是独立进程，每次建连都要经过 TCP 握手与认证，耗时以十毫秒计，
    必须复用连接，否则高并发下连接创建本身就会成为瓶颈。
    """
    url = make_url(settings.database_url)
    if url.get_backend_name() == "sqlite":
        return create_engine(settings.database_url, echo=settings.debug)

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