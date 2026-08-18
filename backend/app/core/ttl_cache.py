"""进程内短 TTL 缓存，适合缓存远程模型查询结果。"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from threading import RLock
from typing import Generic, Optional, TypeVar


Key = TypeVar("Key")
Value = TypeVar("Value")


class TTLCache(Generic[Key, Value]):
    """带容量上限和过期时间的线程安全 LRU 缓存。"""

    def __init__(self, ttl_seconds: float, max_entries: int) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_entries = max(0, int(max_entries))
        self._items: OrderedDict[Key, tuple[float, Value]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Key) -> Optional[Value]:
        """读取未过期值；返回副本避免调用方修改缓存内容。"""
        if self.ttl_seconds <= 0 or self.max_entries <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return copy.deepcopy(value)

    def set(self, key: Key, value: Value) -> None:
        """写入值并淘汰最旧条目。"""
        if self.ttl_seconds <= 0 or self.max_entries <= 0:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """清空缓存。"""
        with self._lock:
            self._items.clear()
