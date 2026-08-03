"""智语数据备份和恢复命令行工具。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.backup_service import BackupValidationError, create_backup, restore_backup


def main() -> int:
    parser = argparse.ArgumentParser(description="备份或恢复智语 Wiki 数据")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="创建 ZIP 备份")
    create.add_argument("--output-dir", help="备份输出目录")

    restore = subparsers.add_parser("restore", help="恢复 ZIP 备份")
    restore.add_argument("archive", help="ZIP 备份路径")
    restore.add_argument("--target-root", help="恢复目标项目根目录")
    restore.add_argument("--overwrite", action="store_true", help="允许覆盖已有文件")
    restore.add_argument("--confirm", action="store_true", help="确认执行恢复")

    args = parser.parse_args()
    try:
        if args.command == "create":
            result = create_backup(args.output_dir)
        else:
            if not args.confirm:
                parser.error("恢复属于写入操作，请同时传入 --confirm")
            result = restore_backup(args.archive, args.target_root, overwrite=args.overwrite)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BackupValidationError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
