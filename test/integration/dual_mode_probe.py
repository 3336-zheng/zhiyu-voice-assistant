"""采集 ChromaService 在当前 CHROMA_MODE 下的行为指纹，写成 JSON 供跨模式比对。

只走 ChromaService 的公开方法——Stage 4 改的就是这层边界，
再往上（RRF、reranker）两种模式共用同一份代码，比了也只会引入浮点噪声。

查询向量用固定种子生成而不是跑真实 embedding：同一个模型对同一句话的输出本身
就有浮点抖动，那点抖动会盖住我真正要找的差异。固定向量让两次运行的输入位级相同。
"""

import hashlib
import json
import random
import sys

from backend.app.core.config import settings
from backend.app.services.retrieval.chroma_service import ChromaService

DIMENSION = 4096
QUERY_SEEDS = [11, 22, 33, 44, 55]
TOP_K = 20


def make_query_vector(seed: int):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)]


def digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def main(out_path: str):
    service = ChromaService()
    report = {
        "mode": settings.chroma_mode,
        "collection": service.collection_name,
        "count": service.get_count(),
    }

    # 1. 向量检索：doc_id 序列与分数，这是验收标准的正主
    searches = []
    for seed in QUERY_SEEDS:
        hits = service.search(make_query_vector(seed), top_k=TOP_K)
        searches.append({
            "seed": seed,
            "ids": [doc_id for doc_id, _ in hits],
            # 分数留 9 位小数：既能发现真实的排序/距离差异，
            # 又不会因为最后一两位的浮点表示差异误报
            "scores": [round(score, 9) for _, score in hits],
        })
    report["searches"] = searches

    # 2. 全量拉取：server 模式下这是最可能踩响应体上限 / 需要分页的地方
    chunks = service.get_all_chunks(["doc", "wiki_page"])
    report["all_chunks_count"] = len(chunks)
    report["all_chunks_digest"] = digest(
        "|".join(
            f"{c['id']}:{digest(c['content'])}:{json.dumps(c['metadata'], sort_keys=True, ensure_ascii=False)}"
            for c in sorted(chunks, key=lambda c: c["id"])
        )
    )
    report["all_chunks_has_embeddings_key"] = any("embedding" in c for c in chunks[:5])

    # 3. 元数据拉取
    metas = service.get_doc_metadatas()
    report["doc_metadatas_count"] = len(metas)
    report["doc_metadatas_digest"] = digest(
        "|".join(sorted(json.dumps(m, sort_keys=True, ensure_ascii=False) for m in metas))
    )

    # 4. 按 ID 批量取：取前 50 个 id，验证内容与 metadata 逐字节一致
    sample_ids = sorted(c["id"] for c in chunks)[:50]
    by_ids = service.get_chunks_by_ids(sample_ids)
    report["by_ids_count"] = len(by_ids)
    report["by_ids_digest"] = digest(
        "|".join(
            f"{k}:{digest(v['content'])}:{json.dumps(v['metadata'], sort_keys=True, ensure_ascii=False)}"
            for k, v in sorted(by_ids.items())
        )
    )

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"[{report['mode']}] count={report['count']} "
          f"all_chunks={report['all_chunks_count']} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
