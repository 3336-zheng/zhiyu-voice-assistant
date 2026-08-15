"""数据库和 Wiki 主数据备份、恢复服务。"""

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from backend.app.core.config import settings


class BackupValidationError(ValueError):
    """备份文件或恢复目标不合法。"""


def _database_path() -> Path:
    """解析当前 SQLite 数据库路径。"""
    prefix = "sqlite:///"
    if not settings.database_url.startswith(prefix):
        raise BackupValidationError("备份目前只支持 SQLite 数据库")
    path = Path(settings.database_url[len(prefix) :])
    return path if path.is_absolute() else Path.cwd() / path


def _included_roots() -> list[tuple[str, Path]]:
    return [
        ("wiki", Path(settings.wiki_pages_dir).parent),
        ("uploads", Path(settings.upload_dir)),
        ("notes", Path("data/notes")),
        ("docs", Path("data/docs")),
    ]


def create_backup(output_dir: str | None = None) -> Dict[str, Any]:
    """创建包含 SQLite 快照、Wiki 文件和附件的 ZIP 备份。"""
    database_path = _database_path()
    if not database_path.exists():
        raise BackupValidationError(f"数据库不存在: {database_path}")

    destination = Path(output_dir or settings.backup_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    backup_name = f"zhiyu-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.zip"
    archive_path = destination / backup_name

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temporary:
        snapshot_path = Path(temporary.name)
    try:
        source = sqlite3.connect(str(database_path))
        target = sqlite3.connect(str(snapshot_path))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()

        manifest = {
            "format": "zhiyu-backup",
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database": "database/notes.db",
            "roots": [name for name, path in _included_roots() if path.exists()],
        }
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot_path, "database/notes.db")
            for name, root in _included_roots():
                if not root.exists():
                    continue
                for file_path in root.rglob("*"):
                    if file_path.is_file():
                        archive_name = Path(name) / file_path.relative_to(root)
                        archive.write(file_path, str(archive_name))
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        snapshot_path.unlink(missing_ok=True)

    return {"path": str(archive_path), "filename": archive_path.name, "manifest": manifest}


def restore_backup(archive_path: str, target_root: str | None = None, *, overwrite: bool = False) -> Dict[str, Any]:
    """恢复备份；调用方必须显式确认，且压缩包路径不能逃逸目标目录。"""
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise BackupValidationError(f"备份文件不存在: {archive}")
    root = Path(target_root or Path.cwd()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        try:
            manifest = json.loads(source.read("manifest.json"))
        except (KeyError, json.JSONDecodeError) as exc:
            raise BackupValidationError("备份缺少有效 manifest.json") from exc
        if manifest.get("format") != "zhiyu-backup":
            raise BackupValidationError("不支持的备份格式")

        members = []
        for member in source.infolist():
            if member.is_dir():
                continue
            destination = (root / member.filename).resolve()
            if os.path.commonpath([str(root), str(destination)]) != str(root):
                raise BackupValidationError(f"备份包含越界路径: {member.filename}")
            if destination.exists() and not overwrite:
                raise BackupValidationError(f"目标文件已存在，请使用 --overwrite: {destination}")
            members.append((member, destination))

        for member, destination in members:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as source_file, destination.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)

    return {"restored": len(members), "target_root": str(root), "manifest": manifest}
