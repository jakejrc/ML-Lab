# -*- coding: utf-8 -*-
"""
Jedi 代码补全后端
为代码沙箱提供 Python 语义级代码补全
"""

import jedi
from functools import lru_cache
import time
import re


# ── 缓存配置 ──
_CACHE_MAXSIZE = 100       # 最大缓存条目数
_CACHE_TTL = 15.0          # 缓存有效期（秒）
_cache = {}                 # {code_hash: (timestamp, [completions])}


def _make_key(code, line, col):
    """生成缓存 key（忽略行尾空白差异）"""
    normalized = code.rstrip()
    return hash((normalized, line, col))


def _clean_completions(completions, max_items=30):
    """清理并格式化补全结果"""
    results = []
    seen = set()
    for c in completions:
        if c.name in seen:
            continue
        seen.add(c.name)
        if len(results) >= max_items:
            break

        item = {
            "label": c.name,
            "type": _jedi_type_to_cm(c.type),
            "detail": c.full_name or c.name,
            "info": _get_doc_preview(c),
        }
        results.append(item)
    return results


def _jedi_type_to_cm(jedi_type):
    """将 Jedi 类型映射为 CM6 补全图标类型"""
    mapping = {
        "function": "function",
        "class": "class",
        "module": "module",
        "keyword": "keyword",
        "statement": "keyword",
        "import": "keyword",
        "param": "variable",
        "property": "property",
        "instance": "variable",
        "int": "number",
        "float": "number",
        "str": "string",
        "bool": "keyword",
        "list": "variable",
        "dict": "variable",
        "tuple": "variable",
        "NoneType": "keyword",
        "type": "type",
    }
    return mapping.get(jedi_type, "text")


def _get_doc_preview(c, max_len=80):
    """获取文档预览"""
    try:
        doc = c.docstring()
        if doc:
            # 只取第一行
            first_line = doc.split("\n")[0].strip()
            if len(first_line) > max_len:
                return first_line[:max_len] + "..."
            return first_line
    except Exception:
        pass
    return ""


def _warm_cache():
    """预热缓存：提前解析常用库"""
    warmup_code = "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.ensemble import RandomForestClassifier"
    try:
        script = jedi.Script(warmup_code)
        # 在 import 行之后补全
        script.complete(4, 30)
    except Exception:
        pass


def get_completions(code, line, col):
    """获取 Jedi 补全结果（带缓存）"""
    key = _make_key(code, line, col)

    # 检查缓存
    now = time.time()
    if key in _cache:
        ts, results = _cache[key]
        if now - ts < _CACHE_TTL:
            return results

    try:
        # 创建 Jedi Script
        script = jedi.Script(code)
        completions = script.complete(line, col)
        results = _clean_completions(completions)

        # 更新缓存
        _cache[key] = (now, results)
        # 清理过期缓存
        if len(_cache) > _CACHE_MAXSIZE * 1.5:
            _clean_expired(now)

        return results

    except Exception as e:
        return [{"label": f"[error] {str(e)[:40]}", "type": "text", "detail": "", "info": ""}]


def _clean_expired(now):
    """清理过期缓存"""
    expired = [k for k, (ts, _) in _cache.items() if now - ts > _CACHE_TTL]
    for k in expired:
        del _cache[k]


def handle_completion(request_data: dict) -> dict:
    """
    FastAPI 处理器
    输入: {"code": "...", "line": 5, "col": 10}
    输出: {"completions": [{"label": "...", "type": "...", "detail": "...", "info": "..."}]}
    """
    code = request_data.get("code", "")
    line = request_data.get("line", 1)
    col = request_data.get("col", 0)

    t0 = time.time()
    completions = get_completions(code, line, col)
    elapsed = (time.time() - t0) * 1000

    return {
        "completions": completions,
        "elapsed_ms": round(elapsed, 1),
        "count": len(completions),
    }
