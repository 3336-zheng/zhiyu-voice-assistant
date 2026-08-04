#!/usr/bin/env python3
"""智语面试演示数据命令。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def initialize() -> dict:
    from backend.app import models as _models  # noqa: F401
    from backend.app.core.database import Base, SessionLocal, engine
    from backend.app.core.schema import ensure_schema
    from backend.app.services.demo_service import initialize_demo_data

    Base.metadata.create_all(bind=engine)
    ensure_schema(engine)
    db = SessionLocal()
    try:
        return initialize_demo_data(db)
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化智语面试演示数据")
    parser.add_argument("command", choices=["init"], help="幂等初始化演示数据")
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(initialize(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
