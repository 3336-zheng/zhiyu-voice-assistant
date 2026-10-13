"""跨方言的共享列类型。"""

from sqlalchemy import DateTime
from sqlalchemy.dialects import mysql

# MySQL 的 DATETIME 默认精度为秒，写入带微秒的值会四舍五入到整秒
# （注意是四舍五入不是截断，.6 秒会进位到下一秒）。
#
# 本项目有多处记录在同一秒内生成：wiki_pages 的 created_at / updated_at、
# wiki_page_revisions.created_at、wiki_index_tasks.created_at、
# wiki_page_links.created_at。而 page_service.py 按 updated_at 排序列出页面、
# wiki_index_task_service.py 按 created_at 取任务，精度降到秒之后
# 这些记录的先后顺序就不再确定。
#
# 因此对 MySQL 显式要求 6 位小数秒。fsp 是 MySQL 的术语
# （fractional seconds precision），取值 0-6，6 即微秒。
#
# with_variant 只在方言匹配时生效，SQLite 仍用原本的 DateTime，
# 行为与改动前完全一致。
DateTimeMs = DateTime(timezone=True).with_variant(mysql.DATETIME(fsp=6), "mysql")
