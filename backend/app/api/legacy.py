"""旧 API 兼容层的统一响应标识。"""

from fastapi import Response


def mark_legacy_response(response: Response) -> None:
    """通过标准响应头提示调用方迁移到统一 Wiki API。"""
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</api/pages>; rel="successor-version"'
    response.headers["Warning"] = '299 - "Legacy API; migrate to /api/pages"'
