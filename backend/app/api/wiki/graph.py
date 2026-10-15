"""知识图谱 API。

放在 /api/graph 而不是 /api/pages/graph：后者会被 GET /api/pages/{page_id}
抢走——FastAPI 按注册顺序匹配，`graph` 会被当成一个 page_id。独立前缀
不会因为将来有人调路由顺序而失效。
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.services.wiki.graph_service import GraphService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
@router.get("/")
def build_graph(
    threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="语义边相似度门槛，不传用服务端配置",
    ),
    db: Session = Depends(get_db),
):
    """返回页面知识图谱的节点与边。

    threshold 可以由前端传：页面语料一变，合适的值就变了，让用户先看到
    效果再定默认值，比在代码里猜一个数可靠。
    """
    try:
        return GraphService(db).build(threshold=threshold)
    except Exception as exc:  # noqa: BLE001 — 图谱是只读视图，失败不该 500 静默
        logger.exception("构建知识图谱失败")
        raise HTTPException(status_code=503, detail=f"知识图谱暂不可用: {exc}") from exc
