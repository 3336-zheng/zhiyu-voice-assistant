"""全局索引版本号的降级与退避行为测试。

重点不在「Redis 能用时能读到数字」，而在 Redis 出问题时检索不会被拖垮：
必须立刻降级到进程内计数器，而且在退避窗口内不再反复尝试连接。
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from backend.app.core import index_version


class FakeRedis:
    """够用的假客户端：只实现被用到的 ping / get / incr。"""

    def __init__(self):
        self.value = None
        self.incr_calls = 0

    def ping(self):
        return True

    def get(self, key):
        return self.value

    def incr(self, key):
        self.incr_calls += 1
        self.value = str(int(self.value or 0) + 1)
        return int(self.value)


class IndexVersionTest(unittest.TestCase):
    def setUp(self):
        index_version.reset_for_tests()

    def tearDown(self):
        index_version.reset_for_tests()

    def test_redis_可用时返回带_r_前缀的版本(self):
        fake = FakeRedis()
        fake.value = "7"
        with patch("redis.Redis.from_url", return_value=fake):
            self.assertEqual(index_version.current(), "r:7")
            index_version.bump()
            self.assertEqual(fake.incr_calls, 1)
            self.assertEqual(index_version.current(), "r:8")

    def test_键不存在时当作零(self):
        with patch("redis.Redis.from_url", return_value=FakeRedis()):
            self.assertEqual(index_version.current(), "r:0")

    def test_连不上_redis_时降级为进程内计数器(self):
        with patch("redis.Redis.from_url", side_effect=OSError("连接被拒绝")):
            self.assertEqual(index_version.current(), "l:0")
            index_version.bump()
            self.assertEqual(index_version.current(), "l:1")

    def test_降级后在退避窗口内不再重试连接(self):
        # 没有退避的话，Redis 挂掉期间每次检索都要白等一次连接超时。
        connect = MagicMock(side_effect=OSError("连接被拒绝"))
        with patch("redis.Redis.from_url", connect):
            for _ in range(10):
                index_version.current()
        self.assertEqual(connect.call_count, 1)

    def test_退避到期后会重新尝试连接(self):
        connect = MagicMock(side_effect=OSError("连接被拒绝"))
        with patch("redis.Redis.from_url", connect):
            index_version.current()
            # 直接把退避截止时间拨到过去，避免测试真的等 5 秒
            index_version._index_version._retry_after = time.monotonic() - 1
            index_version.current()
        self.assertEqual(connect.call_count, 2)

    def test_读写中途出错也会降级而不是抛给调用方(self):
        fake = FakeRedis()
        fake.value = "3"
        with patch("redis.Redis.from_url", return_value=fake):
            self.assertEqual(index_version.current(), "r:3")
            fake.get = MagicMock(side_effect=OSError("连接断开"))
            # 检索路径上不允许抛异常，只能降级
            self.assertEqual(index_version.current(), "l:0")

    def test_降级期间的_bump_仍然让本地版本递增(self):
        # 本地计数器是降级模式下唯一的版本来源，它不递增的话
        # 本进程写入后自己的缓存都不会失效。
        with patch("redis.Redis.from_url", side_effect=OSError("连接被拒绝")):
            before = index_version.current()
            index_version.bump()
            self.assertNotEqual(index_version.current(), before)

    def test_从_redis_模式掉到降级模式时版本串必然变化(self):
        # 前缀存在的意义就在这里：两种来源的计数器互不相干，
        # 不加前缀的话 "3" 和 "3" 会被误判成同一个版本，重建就被漏掉了。
        fake = FakeRedis()
        fake.value = "0"
        with patch("redis.Redis.from_url", return_value=fake):
            redis_version = index_version.current()
        index_version._index_version._client = None
        index_version._index_version._retry_after = 0.0
        with patch("redis.Redis.from_url", side_effect=OSError("连接被拒绝")):
            local_version = index_version.current()
        self.assertNotEqual(redis_version, local_version)


if __name__ == "__main__":
    unittest.main()
