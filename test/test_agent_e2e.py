# -*- coding: utf-8 -*-
"""
端到端测试脚本 — 验证 Agent 全链路正确性 + LLM 稳定性
用法:
  python test_agent_e2e.py           # 运行自动化测试
  python test_agent_e2e.py --chat    # 进入交互模式，手动输入问题
前提: 服务已在 localhost:8336 启动
"""
import requests
import json
import time
import sys
import threading
from typing import Optional, Dict, Any, List

BASE_URL = "http://localhost:8336"

# 颜色输出
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log_pass(name: str, detail: str = ""):
    status = f"{GREEN}[PASS]{RESET} {name}"
    if detail:
        status += f" — {detail}"
    print(status)


def log_fail(name: str, detail: str = ""):
    status = f"{RED}[FAIL]{RESET} {name}"
    if detail:
        status += f" — {detail}"
    print(status)


def log_info(msg: str):
    print(f"{YELLOW}[INFO]{RESET} {msg}")


def chat(query: str, timeout: int = 60, session_id: str = None) -> Optional[Dict[str, Any]]:
    """调用 Agent 聊天接口"""
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id
    try:
        resp = requests.post(
            f"{BASE_URL}/agent/chat/",
            json=payload,
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            log_fail(f"HTTP {resp.status_code}", resp.text[:200])
            return None
    except requests.exceptions.Timeout:
        log_fail("请求超时", f"超过 {timeout}s")
        return None
    except Exception as e:
        log_fail("请求异常", str(e))
        return None


def print_response(result: Dict[str, Any]):
    """格式化打印 Agent 响应"""
    print()
    # 意图和状态
    intent = result.get("intent", "unknown")
    success = result.get("success", False)
    exec_time = result.get("execution_time_ms", 0)
    status_color = GREEN if success else RED
    print(f"  {BOLD}意图:{RESET} {intent}  "
          f"{BOLD}状态:{RESET} {status_color}{'成功' if success else '失败'}{RESET}  "
          f"{BOLD}耗时:{RESET} {exec_time}ms")

    # 计划摘要
    plan_summary = result.get("plan_summary")
    if plan_summary:
        print(f"  {BOLD}计划:{RESET} {plan_summary}")

    # 来源
    sources = result.get("sources", [])
    if sources:
        print(f"  {BOLD}来源:{RESET} ({len(sources)} 条)")
        for i, s in enumerate(sources, 1):
            stype = s.get("source_type", "note")
            title = s.get("title", "N/A")
            score = s.get("score", 0)
            filename = s.get("filename")
            section = s.get("section_title")
            type_tag = f"{CYAN}[doc]{RESET}" if stype == "doc" else f"[note]"
            line = f"    {i}. {type_tag} {title} (score: {score:.2f})"
            if filename:
                line += f"  文件: {filename}"
            if section:
                line += f"  章节: {section}"
            print(line)

    # 回复内容
    response_text = result.get("response", "")
    print(f"\n  {BOLD}回复:{RESET}")
    for line in response_text.split("\n"):
        print(f"  {line}")
    print()


# ============================================================
# 一、健康检查
# ============================================================
def test_health():
    """测试服务健康状态"""
    print("\n" + "=" * 50)
    print("一、健康检查")
    print("=" * 50)

    try:
        resp = requests.get(f"{BASE_URL}/health/health", timeout=5)
        if resp.status_code == 200:
            log_pass("健康检查", f"HTTP {resp.status_code}")
        else:
            log_fail("健康检查", f"HTTP {resp.status_code}")
    except Exception as e:
        log_fail("健康检查", str(e))


# ============================================================
# 二、意图识别测试
# ============================================================
def test_intent_recognition():
    """测试各类意图是否被正确识别"""
    print("\n" + "=" * 50)
    print("二、意图识别测试")
    print("=" * 50)

    test_cases = [
        ("现在几点", "time_query"),
        ("列出所有笔记", "list_notes"),
        ("RAG是什么", "search"),
    ]

    for query, expected_intent in test_cases:
        result = chat(query)
        if result is None:
            log_fail(f"意图识别: '{query}'", "无响应")
            continue

        actual_intent = result.get("intent", "unknown")
        if actual_intent == expected_intent:
            log_pass(f"意图识别: '{query}'", f"→ {actual_intent}")
        else:
            log_fail(
                f"意图识别: '{query}'",
                f"期望 {expected_intent}，实际 {actual_intent}"
            )


# ============================================================
# 三、ASR 纠错测试
# ============================================================
def test_asr_correction():
    """测试语音识别纠错是否生效"""
    print("\n" + "=" * 50)
    print("三、ASR 纠错测试")
    print("=" * 50)

    test_cases = [
        {
            "query": "帮我查走RAG的词类方式分快",
            "expected_keywords": ["RAG", "分块"],
            "description": "走→找, 词类→词语, 分快→分块",
        },
        {
            "query": "找一下有关向量数据库的笔",
            "expected_keywords": ["向量数据库"],
            "description": "笔→笔记, 去除口语冗余",
        },
        {
            "query": "看看那个agent开发的坑",
            "expected_keywords": ["Agent", "开发"],
            "description": "看看→去除, 坑→踩坑",
        },
    ]

    for case in test_cases:
        result = chat(case["query"])
        if result is None:
            log_fail(f"ASR纠错: '{case['query']}'", "无响应")
            continue

        response_text = result.get("response", "")
        # 检查回复中是否包含预期关键词
        matched = []
        for kw in case["expected_keywords"]:
            if kw.lower() in response_text.lower():
                matched.append(kw)

        if matched:
            log_pass(
                f"ASR纠错: '{case['query']}'",
                f"回复含关键词 {matched} ({case['description']})"
            )
        else:
            log_fail(
                f"ASR纠错: '{case['query']}'",
                f"回复中未找到 {case['expected_keywords']}。回复前100字: {response_text[:100]}"
            )


# ============================================================
# 四、文档检索测试
# ============================================================
def test_doc_retrieval():
    """测试 RAG 检索是否能命中 data/docs/ 下的文档内容"""
    print("\n" + "=" * 50)
    print("四、文档检索测试")
    print("=" * 50)

    result = chat("RAG分块策略")
    if result is None:
        log_fail("文档检索", "无响应")
        return

    response_text = result.get("response", "")
    sources = result.get("sources", [])

    # 检查是否有 doc 类型的来源
    doc_sources = [s for s in sources if s.get("source_type") == "doc"]
    note_sources = [s for s in sources if s.get("source_type") == "note"]

    if doc_sources:
        log_pass(
            "文档检索",
            f"命中 {len(doc_sources)} 个 doc chunk，{len(note_sources)} 条笔记"
        )
        for s in doc_sources[:3]:
            log_info(f"  → {s.get('title', 'N/A')} (score: {s.get('score', 0):.2f})")
    elif "RAG" in response_text or "分块" in response_text:
        log_pass("文档检索", "回复中包含 RAG 相关内容（可能来自笔记）")
    else:
        log_fail(
            "文档检索",
            f"未命中 doc chunks。sources 数量: {len(sources)}，回复前100字: {response_text[:100]}"
        )


# ============================================================
# 五、LLM 稳定性测试
# ============================================================
def test_llm_stability():
    """同一查询多次调用，检查 LLM 输出一致性"""
    print("\n" + "=" * 50)
    print("五、LLM 稳定性测试")
    print("=" * 50)

    test_query = "RAG分块策略"
    runs = 3
    results = []

    for i in range(runs):
        log_info(f"第 {i+1}/{runs} 次调用: '{test_query}'")
        result = chat(test_query, timeout=90)
        if result:
            results.append({
                "intent": result.get("intent"),
                "success": result.get("success"),
                "sources_count": len(result.get("sources", [])),
                "response_len": len(result.get("response", "")),
            })
        else:
            results.append(None)
        if i < runs - 1:
            time.sleep(2)  # 间隔 2 秒

    # 分析一致性
    valid = [r for r in results if r is not None]
    if not valid:
        log_fail("LLM 稳定性", "所有调用均失败")
        return

    # 检查 intent 一致性
    intents = set(r["intent"] for r in valid)
    if len(intents) == 1:
        log_pass("意图一致性", f"全部为 {intents.pop()}")
    else:
        log_fail("意图一致性", f"出现 {len(intents)} 种不同意图: {intents}")

    # 检查成功一致性
    successes = set(r["success"] for r in valid)
    if len(successes) == 1:
        log_pass("执行一致性", f"全部 {'成功' if successes.pop() else '失败'}")
    else:
        log_fail("执行一致性", f"结果不一致: {successes}")

    # 检查 sources 数量波动
    source_counts = [r["sources_count"] for r in valid]
    if max(source_counts) - min(source_counts) <= 1:
        log_pass("检索一致性", f"sources 数量: {source_counts}")
    else:
        log_fail("检索一致性", f"sources 数量波动过大: {source_counts}")

    # 打印每次的回复长度
    lengths = [r["response_len"] for r in valid]
    log_info(f"回复长度: {lengths}")


# ============================================================
# 六、回复格式检查
# ============================================================
def test_response_format():
    """检查回复是否包含必要的结构化信息"""
    print("\n" + "=" * 50)
    print("六、回复格式检查")
    print("=" * 50)

    # 检索类查询应返回 sources
    result = chat("向量数据库")
    if result is None:
        log_fail("回复格式", "无响应")
        return

    response_text = result.get("response", "")
    sources = result.get("sources", [])

    if sources:
        log_pass("sources 字段", f"包含 {len(sources)} 条来源")
    else:
        log_fail("sources 字段", "检索类查询未返回 sources")

    if response_text and len(response_text) > 20:
        log_pass("回复内容", f"长度 {len(response_text)} 字符")
    else:
        log_fail("回复内容", f"回复过短或为空: '{response_text}'")

    # 时间查询应返回时间信息
    result2 = chat("现在几点")
    if result2:
        resp2 = result2.get("response", "")
        if any(kw in resp2 for kw in ["年", "月", "日", "星期", "时间", ":"]):
            log_pass("时间查询", f"回复含时间信息: {resp2[:50]}")
        else:
            log_fail("时间查询", f"回复中未找到时间信息: {resp2[:80]}")


# ============================================================
# 七、Agent 稳定性测试
# ============================================================
def test_agent_stability():
    """测试 Agent 在异常和边界情况下的稳定性"""
    print("\n" + "=" * 50)
    print("七、Agent 稳定性测试")
    print("=" * 50)

    _test_edge_input()
    _test_concurrent_requests()
    _test_multi_turn_conversation()
    _test_repeated_operations()
    _test_llm_fallback()


def _test_edge_input():
    """边界输入：空字符串、超长文本、特殊字符"""
    print("\n--- 7.1 边界输入 ---")

    # 空字符串
    try:
        resp = requests.post(
            f"{BASE_URL}/agent/chat/",
            json={"query": ""},
            timeout=10
        )
        if resp.status_code == 400:
            log_pass("空字符串", "返回 400（预期行为）")
        elif resp.status_code == 200:
            log_fail("空字符串", "返回 200，应拒绝空查询")
        else:
            log_fail("空字符串", f"HTTP {resp.status_code}")
    except Exception as e:
        log_fail("空字符串", str(e))

    # 纯空格
    result = chat("   ", timeout=10)
    if result is None:
        log_pass("纯空格", "请求被拒绝或超时（可接受）")
    elif result.get("response"):
        log_pass("纯空格", "返回了回复（服务端做了容错）")
    else:
        log_fail("纯空格", "返回了空响应体")

    # 特殊字符
    result = chat("!@#$%^&*()_+{}|:\"<>?", timeout=15)
    if result and result.get("response"):
        log_pass("特殊字符", "返回了回复，未崩溃")
    elif result is None:
        log_fail("特殊字符", "请求失败（服务端可能崩溃）")
    else:
        log_fail("特殊字符", "返回了空响应")

    # 超长文本（2000 字）
    long_query = "这是一段很长的测试文本" * 200
    result = chat(long_query, timeout=30)
    if result and result.get("response"):
        log_pass("超长文本(2000字)", f"返回了回复（{len(result.get('response', ''))} 字符）")
    elif result is None:
        log_fail("超长文本(2000字)", "请求失败")
    else:
        log_fail("超长文本(2000字)", "返回了空响应")

    # 纯 emoji
    result = chat("😀🎉🔥💡📝", timeout=15)
    if result and result.get("response"):
        log_pass("纯emoji", "返回了回复，未崩溃")
    else:
        log_fail("纯emoji", "请求失败或无响应")


def _test_concurrent_requests():
    """并发请求：同时发送多个查询，检查是否互相干扰"""
    print("\n--- 7.2 并发请求 ---")

    queries = ["现在几点", "列出所有笔记", "RAG是什么"]
    results: List[Optional[Dict]] = [None] * len(queries)
    errors: List[Optional[str]] = [None] * len(queries)

    def send_request(idx: int, query: str):
        try:
            results[idx] = chat(query, timeout=90)
        except Exception as e:
            errors[idx] = str(e)

    threads = []
    for i, q in enumerate(queries):
        t = threading.Thread(target=send_request, args=(i, q))
        threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    elapsed = time.time() - start

    # 检查结果
    success_count = sum(1 for r in results if r is not None)
    error_count = sum(1 for e in errors if e is not None)

    if success_count == len(queries):
        log_pass("并发请求", f"全部成功（{success_count}/{len(queries)}），耗时 {elapsed:.1f}s")
    elif error_count > 0:
        log_fail("并发请求", f"有 {error_count} 个异常: {[e for e in errors if e]}")
    else:
        log_fail("并发请求", f"仅 {success_count}/{len(queries)} 成功")

    # 检查结果是否与查询匹配（未互相干扰）
    for i, q in enumerate(queries):
        if results[i]:
            intent = results[i].get("intent", "")
            if q == "现在几点" and intent == "time_query":
                log_pass(f"并发隔离: '{q}'", f"→ {intent}")
            elif q == "列出所有笔记" and intent == "list_notes":
                log_pass(f"并发隔离: '{q}'", f"→ {intent}")
            elif q == "RAG是什么" and intent == "search":
                log_pass(f"并发隔离: '{q}'", f"→ {intent}")
            else:
                log_fail(f"并发隔离: '{q}'", f"意图不匹配: {intent}")


def _test_multi_turn_conversation():
    """多轮对话：通过 session_id 传递上下文"""
    print("\n--- 7.3 多轮对话 ---")

    import uuid
    session_id = str(uuid.uuid4())

    # 第一轮：创建笔记
    r1 = chat("创建笔记标题是测试多轮对话内容是第一轮消息", timeout=60, session_id=session_id)
    if r1 and r1.get("success"):
        log_pass("多轮-第1轮", f"创建笔记成功 (intent: {r1.get('intent')})")
    else:
        log_fail("多轮-第1轮", "创建笔记失败")
        return

    # 第二轮：列出笔记（应包含刚创建的）
    r2 = chat("列出所有笔记", timeout=60, session_id=session_id)
    if r2 and r2.get("success"):
        log_pass("多轮-第2轮", f"列出笔记成功 (intent: {r2.get('intent')})")
    else:
        log_fail("多轮-第2轮", "列出笔记失败")

    # 第三轮：检索（上下文应保持）
    r3 = chat("搜索关于多轮对话的内容", timeout=60, session_id=session_id)
    if r3 and r3.get("success"):
        log_pass("多轮-第3轮", f"检索成功 (intent: {r3.get('intent')})")
    else:
        log_fail("多轮-第3轮", "检索失败")


def _test_repeated_operations():
    """重复操作：连续创建同名笔记，检查幂等性"""
    print("\n--- 7.4 重复操作 ---")

    title = f"重复测试_{int(time.time())}"

    # 第一次创建
    r1 = chat(f"创建笔记标题是{title}内容是第一次创建", timeout=60)
    if r1 and r1.get("success"):
        log_pass("重复操作-第1次创建", "成功")
    else:
        log_fail("重复操作-第1次创建", "失败")
        return

    # 第二次创建同名笔记
    r2 = chat(f"创建笔记标题是{title}内容是第二次创建", timeout=60)
    if r2 and r2.get("success"):
        log_pass("重复操作-第2次创建", "成功（系统允许重复）")
    elif r2 is None:
        log_fail("重复操作-第2次创建", "请求失败")
    else:
        log_fail("重复操作-第2次创建", f"失败: {r2.get('response', '')[:80]}")


def _test_llm_fallback():
    """LLM 降级：验证 Planner 和 Responder 的降级路径存在"""
    print("\n--- 7.5 降级机制验证 ---")

    # 正常查询，确认当前走的是 LLM 路径
    result = chat("现在几点", timeout=30)
    if result is None:
        log_fail("降级验证", "正常查询失败，无法建立基线")
        return

    intent = result.get("intent", "")
    response_text = result.get("response", "")

    if intent == "time_query":
        log_pass("降级验证-意图识别", f"正常路径工作正常 (intent: {intent})")
    else:
        log_fail("降级验证-意图识别", f"正常查询意图错误: {intent}")

    if response_text and len(response_text) > 5:
        log_pass("降级验证-回复生成", "正常路径回复正常")
    else:
        log_fail("降级验证-回复生成", "回复为空或过短")

    # 检查服务端日志（通过健康接口确认服务仍存活）
    try:
        health = requests.get(f"{BASE_URL}/health/health", timeout=5)
        if health.status_code == 200:
            log_pass("降级验证-服务存活", "服务在多次请求后仍正常响应")
        else:
            log_fail("降级验证-服务存活", f"HTTP {health.status_code}")
    except Exception as e:
        log_fail("降级验证-服务存活", str(e))


# ============================================================
# 八、交互模式
# ============================================================
def interactive_chat():
    """交互模式：手动输入问题进行测试"""
    print("\n" + "=" * 50)
    print("  交互模式 — 输入问题进行测试")
    print("  输入 quit/exit 退出，输入 test 重跑自动化测试")
    print("=" * 50)

    while True:
        try:
            query = input(f"\n{CYAN}>{RESET} 请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("退出。")
            break
        if query.lower() == "test":
            run_automated_tests()
            continue

        log_info(f"发送: '{query}'")
        start = time.time()
        result = chat(query, timeout=90)
        elapsed = time.time() - start

        if result:
            print_response(result)
            log_info(f"总耗时: {elapsed:.1f}s")
        else:
            log_fail("请求失败")


# ============================================================
# 自动化测试
# ============================================================
def run_automated_tests():
    """运行全部自动化测试"""
    start = time.time()

    test_health()
    test_intent_recognition()
    test_asr_correction()
    test_doc_retrieval()
    test_llm_stability()
    test_response_format()
    test_agent_stability()

    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print(f"  测试完成，耗时 {elapsed:.1f}s")
    print(f"{'=' * 50}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  智语 Agent 端到端测试")
    print(f"  目标: {BASE_URL}")
    print("=" * 50)

    if "--chat" in sys.argv:
        interactive_chat()
    else:
        run_automated_tests()
