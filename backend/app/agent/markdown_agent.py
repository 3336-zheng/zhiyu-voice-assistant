"""旧 Markdown 文件写入器，仅为历史代码兼容保留。

新页面写入统一使用 PageService；本模块不再接入 API 或 Agent 执行器。
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 默认笔记存储目录
DEFAULT_NOTES_DIR = "data/notes"


class MarkdownAgent:
    """
    Markdown 文件管理 Agent
    支持创建和写入 .md 文件
    """

    def __init__(self, notes_dir: str = None):
        """
        初始化 Markdown Agent

        Args:
            notes_dir: 笔记存储目录，默认为 data/notes
        """
        self.notes_dir = notes_dir or DEFAULT_NOTES_DIR
        self._ensure_directory(self.notes_dir)

    def _ensure_directory(self, directory: str):
        """确保目录存在"""
        Path(directory).mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, filename: str, directory: str = None) -> str:
        """
        获取完整的文件路径

        Args:
            filename: 文件名
            directory: 目标目录（可选）

        Returns:
            str: 完整文件路径
        """
        # 处理文件名
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        # 清理文件名中的非法字符
        filename = self._sanitize_filename(filename)

        # 确定目录
        target_dir = directory or self.notes_dir
        self._ensure_directory(target_dir)

        return os.path.join(target_dir, filename)

    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名中的非法字符

        Args:
            filename: 原始文件名

        Returns:
            str: 清理后的文件名
        """
        # Windows 文件名非法字符
        illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        return filename

    def create_md_file(
        self,
        filename: str,
        title: str = None,
        content: str = None,
        directory: str = None
    ) -> Dict:
        """
        创建 MD 文件

        Args:
            filename: 文件名（不含扩展名）
            title: 标题（可选，作为文件首行）
            content: 初始内容（可选）
            directory: 目标目录（可选）

        Returns:
            Dict: 创建结果
        """
        file_path = self._get_file_path(filename, directory)

        # 检查文件是否已存在
        if os.path.exists(file_path):
            logger.warning(f"文件已存在: {file_path}")
            return {
                "success": False,
                "error": f"文件已存在: {os.path.basename(file_path)}",
                "file_path": file_path
            }

        # 构建文件内容
        file_content = ""
        if title:
            file_content = f"# {title}\n\n"
        if content:
            file_content += content

        # 写入文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

            logger.info(f"创建 MD 文件成功: {file_path}")
            return {
                "success": True,
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "title": title,
                "content_length": len(file_content),
                "created_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"创建 MD 文件失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def write_md_file(
        self,
        filename: str,
        content: str,
        mode: str = "append",
        directory: str = None
    ) -> Dict:
        """
        写入 MD 文件

        Args:
            filename: 文件名
            content: 要写入的内容
            mode: 写入模式，append（追加）或 overwrite（覆盖）
            directory: 目标目录（可选）

        Returns:
            Dict: 写入结果
        """
        file_path = self._get_file_path(filename, directory)

        # 检查文件是否存在
        file_exists = os.path.exists(file_path)

        try:
            # 确定写入模式
            if mode == "overwrite" or not file_exists:
                write_mode = 'w'
                # 如果是覆盖模式或新文件，添加时间戳头部
                if not content.startswith("#"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    content = f"## {timestamp}\n\n{content}"
            else:
                write_mode = 'a'
                # 追加模式，添加分隔线和时间戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                content = f"\n\n---\n\n## {timestamp}\n\n{content}"

            with open(file_path, write_mode, encoding='utf-8') as f:
                f.write(content)

            # 获取文件大小
            file_size = os.path.getsize(file_path)

            logger.info(f"写入 MD 文件成功: {file_path}")
            return {
                "success": True,
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "mode": mode,
                "content_length": len(content),
                "file_size": file_size,
                "is_new": not file_exists,
                "written_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"写入 MD 文件失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def list_md_files(self, directory: str = None) -> Dict:
        """
        列出目录下的 MD 文件

        Args:
            directory: 目标目录（可选）

        Returns:
            Dict: 文件列表
        """
        target_dir = directory or self.notes_dir

        if not os.path.exists(target_dir):
            return {
                "success": True,
                "directory": target_dir,
                "files": [],
                "count": 0
            }

        md_files = []
        for filename in os.listdir(target_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(target_dir, filename)
                stat = os.stat(file_path)
                md_files.append({
                    "filename": filename,
                    "file_path": file_path,
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        # 按修改时间排序
        md_files.sort(key=lambda x: x["modified_at"], reverse=True)

        return {
            "success": True,
            "directory": target_dir,
            "files": md_files,
            "count": len(md_files)
        }

    def read_md_file(self, filename: str, directory: str = None) -> Dict:
        """
        读取 MD 文件内容

        Args:
            filename: 文件名
            directory: 目标目录（可选）

        Returns:
            Dict: 文件内容
        """
        file_path = self._get_file_path(filename, directory)

        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"文件不存在: {os.path.basename(file_path)}",
                "file_path": file_path
            }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "success": True,
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "content": content,
                "content_length": len(content)
            }
        except Exception as e:
            logger.error(f"读取 MD 文件失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


# 全局 Agent 实例
markdown_agent_instance = None


def get_markdown_agent() -> MarkdownAgent:
    """获取 Markdown Agent 实例（单例模式）"""
    global markdown_agent_instance
    if markdown_agent_instance is None:
        markdown_agent_instance = MarkdownAgent()
    return markdown_agent_instance
