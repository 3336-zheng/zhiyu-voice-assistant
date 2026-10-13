"""比对 two-mode 探针输出，给 Stage 4 验收一个可复现的判定。

用容差而不是逐位比较分数：embedded 走进程内传值，server 走 HTTP JSON，
后者把分数序列化成 f32 再传回来，末位必然有约 1e-7 的相对误差。
真正要守住的是「同一批查询召回同一批 doc_id、且排序一致」——
分数只要在浮点精度内一致即可，逐位比较会把表示差异误报成结果差异。
"""

import json
import sys


def relative_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


def compare(left: dict, right: dict, score_tolerance: float = 1e-5) -> list[str]:
    problems: list[str] = []

    for field in ("collection", "count", "all_chunks_count",
                  "doc_metadatas_count", "by_ids_count"):
        if left[field] != right[field]:
            problems.append(f"{field}: {left[field]} != {right[field]}")

    # 逐字节摘要必须完全一致：内容、元数据、ID 集合任何一处不同都会改变它
    for field in ("all_chunks_digest", "doc_metadatas_digest", "by_ids_digest"):
        if left[field] != right[field]:
            problems.append(f"{field} 不一致：{left[field]} != {right[field]}")

    for left_search, right_search in zip(left["searches"], right["searches"]):
        seed = left_search["seed"]
        if left_search["ids"] != right_search["ids"]:
            # 顺序不同就是真问题——RRF 融合与精排都依赖召回顺序
            problems.append(
                f"seed={seed} doc_id 序列不一致:\n"
                f"  embedded={left_search['ids']}\n"
                f"  server  ={right_search['ids']}"
            )
            continue
        for rank, (a, b) in enumerate(zip(left_search["scores"], right_search["scores"])):
            error = relative_error(a, b)
            if error > score_tolerance:
                problems.append(
                    f"seed={seed} 第 {rank} 名分数超出容差: {a} vs {b}（相对误差 {error:.2e}）"
                )
    return problems


def main(embedded_path: str, server_path: str) -> int:
    with open(embedded_path, encoding="utf-8") as handle:
        embedded = json.load(handle)
    with open(server_path, encoding="utf-8") as handle:
        server = json.load(handle)

    problems = compare(embedded, server)

    print(f"embedded: count={embedded['count']} all_chunks={embedded['all_chunks_count']} "
          f"doc_metadatas={embedded['doc_metadatas_count']} by_ids={embedded['by_ids_count']}")
    print(f"server  : count={server['count']} all_chunks={server['all_chunks_count']} "
          f"doc_metadatas={server['doc_metadatas_count']} by_ids={server['by_ids_count']}")
    print(f"检索组数: {len(embedded['searches'])}")

    # 顺带报一下分数最大偏差，说明「一致」到底一致到什么程度
    worst = 0.0
    for left_search, right_search in zip(embedded["searches"], server["searches"]):
        for a, b in zip(left_search["scores"], right_search["scores"]):
            worst = max(worst, relative_error(a, b))
    print(f"分数最大相对偏差: {worst:.2e}（f32 序列化精度量级，不影响排序）")

    if problems:
        print(f"\n❌ 验收未通过，{len(problems)} 处不一致：")
        for problem in problems[:10]:
            print(f"  - {problem}")
        return 1

    print("\n✅ 验收通过：doc_id 序列、内容摘要、元数据摘要、计数全部一致")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
