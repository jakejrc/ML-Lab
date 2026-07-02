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


def get_signatures(code, line, col):
    """获取函数参数签名（用于函数调用时的参数提示）"""
    try:
        script = jedi.Script(code)
        signatures = script.get_signatures(line, col)
        results = []
        for s in signatures:
            params = []
            for p in s.params:
                desc = p.description
                # 尝试获取默认值
                default = ""
                try:
                    if p.has_default():
                        default = str(p.default_value)
                except Exception:
                    pass
                if default:
                    params.append(f"{desc}={default}")
                else:
                    params.append(desc)
            results.append({
                "name": s.name,
                "params": params,
                "index": s.index if hasattr(s, "index") else 0,
                "docstring": (s.docstring() or "").split("\n")[0][:100],
            })
        return results
    except Exception as e:
        return []


def handle_signatures(request_data: dict) -> dict:
    """FastAPI 处理器：参数签名"""
    code = request_data.get("code", "")
    line = request_data.get("line", 1)
    col = request_data.get("col", 0)
    sigs = get_signatures(code, line, col)
    return {"signatures": sigs, "count": len(sigs)}


def _extract_imports(line):
    """从 import 语句中提取所有定义的名称"""
    names = set()
    stripped = line.strip()
    if stripped.startswith("import "):
        parts = stripped[7:].split(",")
        for p in parts:
            p = p.strip()
            if " as " in p:
                names.add(p.split(" as ")[-1].strip())
            else:
                names.add(p.split()[0])
    elif stripped.startswith("from "):
        # from X import Y, Z as W
        if " import " in stripped:
            after_import = stripped.split(" import ", 1)[1]
            for p in after_import.split(","):
                p = p.strip()
                if " as " in p:
                    names.add(p.split(" as ")[-1].strip())
                else:
                    names.add(p.split()[0])
    return names


def get_diagnostics(code):
    """代码诊断：语法错误 + 未定义变量警告"""
    issues = []
    lines = code.split("\n")

    # 1. 语法检查
    try:
        compile(code, "<sandbox>", "exec")
    except SyntaxError as e:
        issues.append({
            "line": e.lineno or 1,
            "col": (e.offset or 0) - 1,
            "end_col": (e.offset or 0) + 5,
            "message": f"语法错误: {e.msg}",
            "severity": "error",
        })
        # 语法错误时跳过后续检查
        return issues

    # 2. Jedi 静态分析：未定义变量、类型错误等
    try:
        script = jedi.Script(code)
        # 获取所有名称引用
        names = script.get_names(all_scopes=True, definitions=True)
        defined_names = set()
        used_names = set()

        # 通过 import 语句解析别名
        for line in code.split("\n"):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                defined_names.update(_extract_imports(line))

        for n in names:
            if n.type in ("class", "function", "statement", "import", "param"):
                defined_names.add(n.name)
            else:
                used_names.add(n.name)

        # 检查可能的问题
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # 检查常见的拼写错误
            for n in list(used_names):
                if n not in defined_names and not n.startswith("_"):
                    # 只在当前行中出现的未定义名
                    if n in line:
                        # 排除 Python 内置函数
                        builtins = {"print", "len", "range", "int", "float", "str", "list",
                                     "dict", "set", "tuple", "bool", "type", "isinstance",
                                     "hasattr", "getattr", "setattr", "open", "input",
                                     "super", "object", "property", "staticmethod",
                                     "classmethod", "enumerate", "zip", "map", "filter",
                                     "sorted", "reversed", "min", "max", "sum", "abs",
                                     "round", "pow", "divmod", "hex", "oct", "bin",
                                     "ord", "chr", "repr", "eval", "exec", "compile",
                                     "__import__", "id", "dir", "vars", "globals",
                                     "locals", "format", "any", "all", "iter", "next",
                                     "slice", "memoryview", "bytearray", "bytes",
                                     "frozenset", "NotImplemented", "Ellipsis",
                                     "Exception", "BaseException", "ValueError",
                                     "TypeError", "KeyError", "IndexError",
                                     "AttributeError", "ImportError", "ModuleNotFoundError",
                                     "FileNotFoundError", "ZeroDivisionError",
                                     "StopIteration", "KeyboardInterrupt",
                                     "SystemExit", "RuntimeError", "OSError", "IOError",
                                     "Warning", "UserWarning", "DeprecationWarning",
                                     "True", "False", "None", "self", "cls",
                                     "ValueError", "SystemError", "TabError",
                                     "IndentationError", "EOFError", "FloatingPointError",
                                     "OverflowError", "ArithmeticError", "LookupError",
                                     "AssertionError", "BufferError",
                                     "ReferenceError", "NameError", "UnboundLocalError",
                                     "NotImplementedError", "RecursionError",
                                     "PermissionError", "ProcessLookupError",
                                     "TimeoutError", "ConnectionError",
                                     "ConnectionRefusedError", "ConnectionResetError",
                                     "BrokenPipeError", "ChildProcessError",
                                     "InterruptedError", "IsADirectoryError",
                                     "NotADirectoryError", "FileExistsError",
                                     "BlockingIOError", "ResourceWarning",
                                     "ImportWarning", "PendingDeprecationWarning",
                                     "SyntaxWarning", "RuntimeWarning",
                                     "FutureWarning", "BytesWarning", "UnicodeWarning"}
                        if n not in builtins and n not in defined_names:
                            col = line.index(n)
                            issues.append({
                                "line": i,
                                "col": col,
                                "end_col": col + len(n),
                                "message": f"可能未定义的变量: {n}",
                                "severity": "warning",
                            })

    except Exception:
        pass

    return issues


def handle_diagnostics(request_data: dict) -> dict:
    """FastAPI 处理器：代码诊断"""
    code = request_data.get("code", "")
    issues = get_diagnostics(code)
    return {"issues": issues, "count": len(issues)}
