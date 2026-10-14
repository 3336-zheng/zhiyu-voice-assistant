"""测试的全局配置。

**必须在导入任何应用代码之前设好环境变量**——`Settings` 在 config 模块导入时
就实例化了（模块底部那句 `settings = Settings()`），晚一步就来不及。

为什么需要这个：CI 上没有 `.env` 文件（它被 gitignore，不会随代码 checkout）。
`database_url` 是必填字段，缺了它 `Settings` 校验直接失败，随后所有 import 到
settings 的模块全崩——表现是几十个 collection error，而不是一个清楚的报错。

判断依据是**项目根有没有 `.env` 文件**，不是环境变量在不在：本地开发时
`DATABASE_URL` 写在 `.env` 里，进程环境里本来就没有这个变量，所以光看
`os.environ` 区分不出「本地有 .env」和「CI 没有 .env」。

有 `.env` → 什么都不做，让 pydantic-settings 按它原本的优先级去读。
没有 `.env` → 补一个能连上的默认值；CI 的 workflow 会用 env 覆盖成它自己的地址。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if not (PROJECT_ROOT / ".env").exists():
    # 兜底：CI 环境。用这里没有真实数据，只求 Settings 能实例化、测试能连上
    # 测试库——测试每个用例都会重建 schema，不需要库里有任何东西。
    os.environ.setdefault(
        "DATABASE_URL",
        "mysql+pymysql://root:root@127.0.0.1:3306/zhiyu_test?charset=utf8mb4",
    )
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
