"""本地开发入口：启动 API 服务。

和 `python main.py` 做的事情完全一样，只是名字更明确——项目现在有四个进程要起
（API + 三个 worker），用文件名区分比记参数清楚：

    python run_api.py        API 服务
    python run_workers.py    三个后台 worker

**不重复实现启动逻辑**：直接委托给 main.py 的 `__main__` 分支。启动参数
（host / port / reload）只在 main.py 里写一份，改一处就够。

Docker 部署不走这里：镜像的 CMD 是 `python main.py`，compose 里只有 `zhiyu`
一个 API 服务。

依赖的三个基础服务要先起来：
    docker compose up -d mysql redis chroma
"""

import runpy
import sys
from pathlib import Path

# 保证从项目根目录 import 得到 main（用 `python run_api.py` 从别处调用时也成立）
sys.path.insert(0, str(Path(__file__).resolve().parent))


if __name__ == "__main__":
    # run_name="__main__" 让 main.py 执行它的 `if __name__ == "__main__"` 分支，
    # 也就是 uvicorn.run(...)。
    runpy.run_module("main", run_name="__main__")
