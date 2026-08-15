"""
智语端侧智能语音笔记助手主应用
直接使用 backend app 作为主应用
"""
import sys
import os

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app
from backend.app.core.config import settings


def kill_port_conflict(port: int):
    """杀掉占用指定端口的 node.exe 进程（Figma MCP 等）"""
    if sys.platform != 'win32':
        return
    import subprocess
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f':{port}' in line and 'LISTENING' in line:
                pid = line.strip().split()[-1]
                # 检查是否是 node.exe
                task = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}'],
                    capture_output=True, text=True, timeout=5
                )
                if 'node.exe' in task.stdout:
                    print(f'[端口清理] 杀掉 node.exe [PID {pid}]，释放端口 {port}')
                    subprocess.run(['taskkill', '/PID', pid, '/F'],
                                   capture_output=True, timeout=5)
    except Exception:
        pass  # 清理失败不影响启动


if __name__ == "__main__":
    kill_port_conflict(settings.port)
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,
    )
