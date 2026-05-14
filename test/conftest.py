# -*- coding: utf-8 -*-
"""
Benchmark 共享配置 — 模块注入，绕过 backend/app/__init__.py 的重依赖链。
在 import backend.app.services 之前调用 setup_backend() 即可。
"""
import sys
import os
import types
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

_backend_ready = False


def setup_backend():
    """
    预注入 backend 命名空间，避免触发 backend/app/__init__.py（它会加载 FastAPI、挂载静态文件等）。
    只加载 benchmark 需要的模块：config、database、services。
    """
    global _backend_ready
    if _backend_ready:
        return

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # 创建空的包模块，阻止 Python 执行真实的 __init__.py
    dummy_packages = [
        "backend", "backend.app", "backend.app.core",
        "backend.app.services", "backend.app.models",
        "backend.app.agent", "backend.app.api",
    ]
    for name in dummy_packages:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(BACKEND_DIR, *name.split(".")[1:])]
            sys.modules[name] = mod

    def _load(full_name: str, file_path: str):
        """加载单个模块文件到 sys.modules"""
        spec = importlib.util.spec_from_file_location(full_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # 按依赖顺序加载
    _load("backend.app.core.config", os.path.join(BACKEND_DIR, "app", "core", "config.py"))
    _load("backend.app.core.database", os.path.join(BACKEND_DIR, "app", "core", "database.py"))

    # 加载所有 models 子模块
    models_dir = os.path.join(BACKEND_DIR, "app", "models")
    for fname in os.listdir(models_dir):
        if fname.endswith(".py") and fname != "__init__.py":
            mod_name = fname[:-3]
            _load(f"backend.app.models.{mod_name}", os.path.join(models_dir, fname))

    # 加载 models/__init__.py 以导出 Note, Audio 等类
    _load("backend.app.models", os.path.join(models_dir, "__init__.py"))

    # 加载所有 services
    services_dir = os.path.join(BACKEND_DIR, "app", "services")
    for fname in os.listdir(services_dir):
        if fname.endswith("_service.py"):
            svc_name = fname[:-3]  # 去掉 .py
            _load(f"backend.app.services.{svc_name}", os.path.join(services_dir, fname))

    _backend_ready = True
