"""Wiki 页面服务共享异常。"""


class PageServiceError(Exception):
    """页面服务基础异常。"""


class PageNotFoundError(PageServiceError):
    """页面不存在。"""


class PageConflictError(PageServiceError):
    """页面版本冲突。"""


class PageValidationError(PageServiceError):
    """页面输入或 Front Matter 不合法。"""


class AmbiguousPageError(PageServiceError):
    """标题或别名匹配到多个页面。"""
