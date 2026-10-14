"""本地开发入口：启动全部后台 worker。

智语有三个 worker，各消费一个队列：

    arq-worker        索引    几百毫秒的短活
    arq-worker-asr    语音转写  最长 180 秒
    arq-worker-agent  Agent 运行 最长 180 秒，交互式

分开是因为时长差一个量级——混在一起的话，你上传一段长音频，后面排着的索引
任务得等它转完，页面会一直显示"索引中"。

Docker 部署时这三个由 compose 各自起一个容器（`restart: unless-stopped` 负责
崩溃重启），不需要这个脚本。**这个脚本只服务本地开发**：不用 Docker 跑应用时，
一条命令把三个都拉起来。

用法：
    python run_workers.py

按 Ctrl+C 会依次停掉三个子进程。任意一个异常退出时会连带停掉其余两个并以
非零码退出——不要留一个"半死不活"的进程组，那样排查起来更费劲。

一个注意点：这是**父进程管子进程**，不是把三个队列塞进一个进程。arq 的 Worker
只接受一个 queue_name，一个进程监听多队列它原生不支持；而且合并会让"worker 崩了
不影响 API"这个隔离失效。所以这里用 subprocess 拉起三个独立进程，跟 Docker 里的
拓扑完全一致——只是本地少开两个终端。
"""

import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import settings

ROOT = Path(__file__).resolve().parent

# 三个 worker 的启动目标。顺序即启动顺序。
WORKERS = [
    ("index", "backend.app.tasks.worker.WorkerSettings", 0),
    ("asr", "backend.app.tasks.worker.AsrWorkerSettings", 0),
    ("agent", "backend.app.tasks.worker.AgentWorkerSettings", 0),
]

# 颜色只在终端有意义；不支持的终端会自动忽略转义序列。
COLORS = {
    "index": "\033[36m",   # 青
    "asr": "\033[35m",     # 紫
    "agent": "\033[33m",   # 黄
}
RESET = "\033[0m"


@dataclass
class Worker:
    """一个被托管的 worker 子进程。"""

    name: str
    spec: str
    process: subprocess.Popen


def _spawn(name: str, spec: str) -> Worker:
    """拉起一个 worker 子进程。

    子进程的 stdout/stderr 直接继承本进程——这样日志的排版和日志级别处理
    跟单独跑 `arq xxxSettings` 完全一致，不做二次加工。

    队列名从应用配置里读（settings.arq_*_queue），不写死：worker 内部也用
    同一个配置去连 Redis，两处一旦写死就可能对不上，改配置时会踩坑。
    """
    cmd = [sys.executable, "-m", "arq", spec]
    process = subprocess.Popen(cmd, cwd=str(ROOT))
    print(
        f"{COLORS.get(name, '')}[{name}]{RESET} 已启动 "
        f"(pid {process.pid})  队列 {_queue_of(name)}",
        flush=True,
    )
    return Worker(name=name, spec=spec, process=process)


def _queue_of(name: str) -> str:
    """这个 worker 消费哪个队列——从配置读，和 worker 自身用的是同一份。"""
    return {
        "index": settings.arq_index_queue,
        "asr": settings.arq_asr_queue,
        "agent": settings.arq_agent_queue,
    }[name]


def _stop_all(workers: list[Worker]) -> None:
    """停掉所有子进程：先发 SIGTERM 让它优雅退出，超时再 SIGKILL。

    arq 收到 SIGTERM 会等手头的任务收尾（`stop_grace_period` 那个语义），
    直接 SIGKILL 会把正在跑的索引/转写/对话切成半个。
    """
    alive = [w for w in workers if w.process.poll() is None]
    for worker in alive:
        worker.process.terminate()

    deadline = time.monotonic() + settings.arq_shutdown_grace_seconds
    for worker in alive:
        remaining = max(deadline - time.monotonic(), 0)
        try:
            worker.process.wait(timeout=remaining)
            print(f"{COLORS.get(worker.name, '')}[{worker.name}]{RESET} 已停止", flush=True)
        except subprocess.TimeoutExpired:
            print(
                f"{COLORS.get(worker.name, '')}[{worker.name}]{RESET} "
                f"超过 {settings.arq_shutdown_grace_seconds}s 没退出，强制结束",
                flush=True,
            )
            worker.process.kill()
            worker.process.wait()


def main() -> int:
    workers = [_spawn(name, spec) for name, spec, _ in WORKERS]

    stopping = False

    def request_stop(signum, _frame):
        """收到 Ctrl+C / SIGTERM 时只记下意图，真正停放在主循环里做。

        信号处理函数里做 wait() 有风险（可能重入）；标记一下交给主循环更稳。
        """
        nonlocal stopping
        if not stopping:
            stopping = True
            print(f"\n收到信号 {signum}，正在停止所有 worker……", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    exit_code = 0
    while not stopping:
        time.sleep(0.5)
        for worker in workers:
            code = worker.process.poll()
            if code is not None:
                # 有一个意外退出：不让剩下的继续单独跑，那样状态会变得难判断。
                print(
                    f"{COLORS.get(worker.name, '')}[{worker.name}]{RESET} "
                    f"已退出（返回码 {code}），停止其余 worker",
                    file=sys.stderr,
                    flush=True,
                )
                stopping = True
                exit_code = code or 1
                break

    _stop_all(workers)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
