"""stdio 启动入口：python -m backend.app.mcp_server

MCP 客户端（Claude Code / Cursor）会以子进程方式拉起本模块，通过 stdin/stdout
走 MCP 协议。日志必须写到 stderr——stdout 是协议通道，混入任何输出都会破坏握手。
"""

import logging
import sys

from backend.app.core.config import settings
from backend.app.mcp_server.server import create_server


def main() -> int:
    if not settings.mcp_server_enabled:
        print(
            "MCP Server 未启用：请在 .env 中设置 MCP_SERVER_ENABLED=true",
            file=sys.stderr,
        )
        return 1
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    create_server().run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
