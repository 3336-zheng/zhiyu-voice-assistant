-- 测试库初始化。
--
-- 测试连的是真实 MySQL 上的 zhiyu_test 库（见 test/support/db.py），而不是
-- 内存库——这样测试和生产跑在同一种方言上，MySQL 特有的行为（字符集协商、
-- utf8mb4 下的索引键长度上限、JSON 字段语义）能在测试里暴露。
--
-- 这个脚本只在**数据卷首次初始化**时执行一次。已有的环境要手动跑一次：
--
--   docker compose exec mysql mysql -uroot -p"$MYSQL_ROOT_PASSWORD" \
--     -e "CREATE DATABASE IF NOT EXISTS zhiyu_test CHARACTER SET utf8mb4
--         COLLATE utf8mb4_unicode_ci;
--         GRANT ALL PRIVILEGES ON zhiyu_test.* TO 'zhiyu'@'%';
--         FLUSH PRIVILEGES;"
--
-- 或直接用 scripts/init_test_db.sh。

CREATE DATABASE IF NOT EXISTS zhiyu_test
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- 应用账号要能在测试库里建表、删表——测试每个用例都会重建整个 schema。
GRANT ALL PRIVILEGES ON zhiyu_test.* TO 'zhiyu'@'%';

FLUSH PRIVILEGES;
