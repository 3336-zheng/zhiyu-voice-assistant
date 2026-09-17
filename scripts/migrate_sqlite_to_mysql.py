"""把 SQLite 里的业务数据迁移到 MySQL。

用法：
    # 先看一眼要搬什么，不写任何数据
    python scripts/migrate_sqlite_to_mysql.py --dry-run

    # 正式迁移（目标库必须已经执行过 alembic upgrade head）
    python scripts/migrate_sqlite_to_mysql.py

    # 目标库已有数据时先清空再迁（危险，需要同时传 --confirm）
    python scripts/migrate_sqlite_to_mysql.py --truncate --confirm

设计说明：

- 走 SQLAlchemy Core + ORM 元数据，不导 SQL。mysqldump / sqlite3 .dump 产出的
  语句带方言（AUTOINCREMENT、双引号标识符等），MySQL 不认；而 Core 读出来的是
  Python 对象，写回去时由目标方言自行渲染，两边的类型差异交给 SQLAlchemy 处理。

- 也不走 ORM 对象。ORM 会触发 relationship 加载、维护 identity map，在这里纯属
  开销，还可能因为 before_insert 之类的事件钩子改写数据。

- 表顺序直接取 metadata.sorted_tables，它已经是按外键依赖排好的拓扑序，
  不需要手工维护先后关系。

- 只搬 ORM 里声明过的表。源库里的 schema_migrations 是已删除的 schema.py
  留下的，不在 metadata 中，会被自动跳过。
"""

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import Table, create_engine, func, select, text
from sqlalchemy.engine import Connection, Engine, make_url

from backend.app import models  # noqa: F401  导入以触发全部 ORM 表注册
from backend.app.core.config import settings
from backend.app.core.database import Base

DEFAULT_SOURCE = "sqlite:///data/database/notes.db"
# 单批行数。批次太大时 pymysql 拼出的语句会超过 max_allowed_packet（默认 64 MB），
# 500 行对本项目的数据量（最宽的表是带 JSON 的 agent_runs）远在安全范围内。
BATCH_SIZE = 500
# 每张表抽样比对的行数
SAMPLE_SIZE = 5
# JSON 列里浮点数的相对容差。
#
# 两边的浮点数会在最后一位上不一样：SQLite 侧 SQLAlchemy 用 Python 的 json.dumps
# 存成文本，写的是最短往返表示，读回来逐位还原；MySQL 侧是原生 JSON 列，
# double 与文本之间的转换走的是 MySQL 自己的算法，末位可能与 Python 差一个 ULP
# （double 在该数量级上能表示的最小间隔）。
#
# 迁移时实测：全部差异恰好是 1.00 ULP，相对误差约 2e-16，集中在检索的 rrf_score 上。
# 那是写进可观测性记录的中间分数，不参与任何后续计算，而实际分值之间的间距在
# 1e-2 量级，差一个 ULP 不可能改变排序。
#
# 容差只对浮点生效，其余类型（含 datetime）一律要求严格相等。
FLOAT_TOLERANCE = 1e-12


class MigrationError(RuntimeError):
    """迁移前置条件不满足，或迁移后校验未通过。"""


def _count(connection: Connection, table: Table) -> int:
    return connection.execute(select(func.count()).select_from(table)).scalar_one()


def _read_all(connection: Connection, table: Table) -> list[dict[str, Any]]:
    """按主键排序读出整张表。

    排序是为了让两边的抽样比对取到同一批行，否则不带 ORDER BY 时
    返回顺序由存储引擎决定，SQLite 与 InnoDB 未必一致。
    """
    order = list(table.primary_key.columns) or list(table.columns)
    rows = connection.execute(select(table).order_by(*order)).mappings().all()
    return [dict(row) for row in rows]


def _check_target_empty(connection: Connection) -> list[str]:
    """返回目标库中已有数据的表名。"""
    return [t.name for t in Base.metadata.sorted_tables if _count(connection, t) > 0]


def _truncate_all(connection: Connection) -> None:
    """按依赖倒序清空目标表。

    不用 TRUNCATE：被外键引用的表执行 TRUNCATE 会被 InnoDB 直接拒绝
    （错误 1701），而 DELETE 可以，代价是不重置 AUTO_INCREMENT 计数器——
    这恰好是我们要的，因为迁移会显式写入原来的 id。
    """
    for table in reversed(Base.metadata.sorted_tables):
        connection.execute(table.delete())


def _copy_table(
    source: Connection, target: Connection, table: Table
) -> tuple[int, int]:
    """把一张表从源库搬到目标库，返回（源行数，写入行数）。"""
    rows = _read_all(source, table)
    if not rows:
        return 0, 0

    written = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        # executemany：单条 INSERT 带多组参数，比逐行执行少一个数量级的往返
        target.execute(table.insert(), batch)
        written += len(batch)
    return len(rows), written


def _verify_counts(source: Connection, target: Connection) -> list[str]:
    """逐表比对行数，返回不一致的描述。"""
    problems = []
    for table in Base.metadata.sorted_tables:
        want, got = _count(source, table), _count(target, table)
        if want != got:
            problems.append(f"{table.name}: 源 {want} 行，目标 {got} 行")
    return problems


def _diff(want: Any, got: Any, path: str) -> list[str]:
    """递归比较两个值，返回不一致的叶子节点描述。

    递归是因为 JSON 列里嵌了好几层——直接用 != 比整个 dict 只能知道
    「不一样」，打印出来是上万字符，定位不到是哪个字段。
    """
    if isinstance(want, float) and isinstance(got, float):
        if want != got:
            scale = max(abs(want), abs(got))
            if abs(want - got) > FLOAT_TOLERANCE * max(scale, 1.0):
                return [f"{path}: 源 {want!r}，目标 {got!r}"]
        return []
    if type(want) is not type(got):
        return [f"{path}: 类型不同，源 {type(want).__name__}，目标 {type(got).__name__}"]
    if isinstance(want, dict):
        if set(want) != set(got):
            missing = set(want) - set(got)
            extra = set(got) - set(want)
            return [f"{path}: 键集合不同，源多出 {missing or '无'}，目标多出 {extra or '无'}"]
        return [p for k in want for p in _diff(want[k], got[k], f"{path}.{k}")]
    if isinstance(want, list):
        if len(want) != len(got):
            return [f"{path}: 长度不同，源 {len(want)}，目标 {len(got)}"]
        return [
            p
            for i, (x, y) in enumerate(zip(want, got))
            for p in _diff(x, y, f"{path}[{i}]")
        ]
    if want != got:
        return [f"{path}: 源 {want!r:.100}，目标 {got!r:.100}"]
    return []


def _verify_samples(source: Connection, target: Connection) -> list[str]:
    """抽样做全字段比对，返回不一致的描述。

    行数相同不代表内容相同——类型转换出错（比如 JSON 被存成字符串字面量、
    datetime 丢精度）不会改变行数。
    """
    problems = []
    for table in Base.metadata.sorted_tables:
        src_rows = _read_all(source, table)[:SAMPLE_SIZE]
        if not src_rows:
            continue
        dst_rows = _read_all(target, table)[:SAMPLE_SIZE]
        for index, (want, got) in enumerate(zip(src_rows, dst_rows)):
            for column, value in want.items():
                problems += _diff(
                    value, got.get(column), f"{table.name} 第 {index + 1} 行 {column}"
                )
    return problems


def _verify_autoincrement(target: Connection) -> list[str]:
    """检查自增计数器有没有跟上已写入的最大 id。

    迁移时显式带了 id，MySQL 8 会把 AUTO_INCREMENT 推到 max(id)+1，
    但这是实现行为不是协议保证。计数器落后的话，下一次插入会撞主键。
    """
    if target.dialect.name != "mysql":
        return []

    # information_schema 里的 auto_increment 是缓存的统计信息，不是实时值。
    # MySQL 8 的 information_schema_stats_expiry 默认 86400 秒，也就是说
    # 表还空着的时候读过一次，接下来一整天读到的都是那个 1，哪怕中途已经
    # 插进去几万行。这里把本会话的过期时间设为 0，强制每次现查存储引擎。
    target.execute(text("SET SESSION information_schema_stats_expiry = 0"))

    query = text(
        "SELECT auto_increment FROM information_schema.tables "
        "WHERE table_schema = DATABASE() AND table_name = :name"
    )

    problems = []
    for table in Base.metadata.sorted_tables:
        pk = list(table.primary_key.columns)
        if len(pk) != 1 or not _is_integer_column(pk[0]):
            continue
        max_id = target.execute(select(func.max(pk[0]))).scalar()
        if max_id is None:
            continue
        next_id = target.execute(query, {"name": table.name}).scalar()
        if next_id is not None and next_id <= max_id:
            problems.append(
                f"{table.name}: AUTO_INCREMENT={next_id}，但已有 id 最大到 {max_id}"
            )
    return problems


def _is_integer_column(column) -> bool:
    """判断一个列是不是整数列。

    python_type 对部分自定义类型没有实现，会抛 NotImplementedError，
    这里当成「不是整数」处理即可——那种列本来也不会是自增主键。
    """
    try:
        return issubclass(column.type.python_type, int)
    except NotImplementedError:
        return False


def _describe(engine: Engine) -> str:
    """打印连接信息时抹掉密码。"""
    return engine.url.render_as_string(hide_password=True)


def migrate(source_url: str, target_url: str, *, truncate: bool, dry_run: bool) -> int:
    source_engine = create_engine(source_url)
    target_engine = create_engine(target_url)

    print(f"源  : {_describe(source_engine)}")
    print(f"目标: {_describe(target_engine)}")
    print()

    try:
        with source_engine.connect() as source:
            plan = [(t, _count(source, t)) for t in Base.metadata.sorted_tables]
            total = sum(n for _, n in plan)

            print("待迁移的表（已按外键依赖排序）：")
            for table, n in plan:
                print(f"  {table.name:<30} {n:>6} 行")
            print(f"  {'合计':<28} {total:>6} 行")
            print()

            if dry_run:
                print("--dry-run，未写入任何数据。")
                return 0

            with target_engine.begin() as target:
                occupied = _check_target_empty(target)
                if occupied and not truncate:
                    raise MigrationError(
                        "目标库以下表已有数据，拒绝执行："
                        + "、".join(occupied)
                        + "\n重复迁移会造成主键冲突或数据翻倍。"
                        "确认要覆盖请加 --truncate --confirm。"
                    )
                if occupied:
                    print(f"清空目标库 {len(occupied)} 张有数据的表…")
                    _truncate_all(target)

                for table, _ in plan:
                    read, written = _copy_table(source, target, table)
                    if read:
                        print(f"  {table.name:<30} 搬运 {written:>6} 行")
                print()

            # 校验放在事务提交之后，验的是落库后的真实结果
            with target_engine.connect() as target:
                problems = _verify_counts(source, target)
                problems += _verify_samples(source, target)
                problems += _verify_autoincrement(target)

        if problems:
            print("校验未通过：")
            for item in problems:
                print(f"  ✗ {item}")
            raise MigrationError(f"共 {len(problems)} 处不一致")

        print(f"校验通过：{len(plan)} 张表、{total} 行，行数与抽样内容一致。")
        return 0
    finally:
        source_engine.dispose()
        target_engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="把 SQLite 业务数据迁移到 MySQL")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"源数据库地址，默认 {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="目标数据库地址，默认取配置里的 DATABASE_URL",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写数据")
    parser.add_argument(
        "--truncate", action="store_true", help="目标表已有数据时先清空"
    )
    parser.add_argument("--confirm", action="store_true", help="确认执行 --truncate")
    args = parser.parse_args()

    target_url = args.target or settings.database_url
    if make_url(target_url).get_backend_name() == "sqlite":
        parser.error(
            f"目标库仍是 SQLite（{target_url}）。"
            "请用 --target 显式指定 MySQL 地址，或先把 .env 的 DATABASE_URL 改过去。"
        )
    if args.truncate and not args.confirm:
        parser.error("--truncate 会删除目标库数据，请同时传入 --confirm")

    try:
        return migrate(
            args.source, target_url, truncate=args.truncate, dry_run=args.dry_run
        )
    except MigrationError as exc:
        print(f"\n迁移失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
