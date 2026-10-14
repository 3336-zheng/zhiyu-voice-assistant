#!/usr/bin/env bash
# 建测试库 zhiyu_test 并授权给应用账号。
#
# docker/mysql/init/ 里那份 SQL 只在数据卷**首次初始化**时自动执行。
# 已经跑过的环境要补这一步，否则测试会连不上库——`make_test_engine()`
# 会给出明确提示，但手动跑一次这个脚本更省事。
#
# 用法：bash scripts/init_test_db.sh
set -euo pipefail

CONTAINER="${MYSQL_CONTAINER:-zhiyu-mysql}"
DATABASE="${TEST_DATABASE_NAME:-zhiyu_test}"

# 从 .env 取 root 密码和应用账号，避免在脚本里写死凭据
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

: "${MYSQL_ROOT_PASSWORD:?请在 .env 中设置 MYSQL_ROOT_PASSWORD}"
: "${MYSQL_USER:=zhiyu}"

echo "在容器 ${CONTAINER} 上创建 ${DATABASE} 并授权给 ${MYSQL_USER}..."
docker exec -i "${CONTAINER}" mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" <<SQL
CREATE DATABASE IF NOT EXISTS \`${DATABASE}\`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON \`${DATABASE}\`.* TO '${MYSQL_USER}'@'%';
FLUSH PRIVILEGES;
SQL

echo "完成。现在可以跑：python -m pytest test/"
