"""
ML-Lab 实验管理器
P0.1: 实验记录、历史查看、实验对比
P0.2: 数据导出（实验报告、代码）
P0.3: 模型持久化（保存/加载）
P0.4: 超参数网格搜索
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any


def record_experiment(_g: dict, exp_type: str, algo: str, report: str,
                       images: dict = None, params: dict = None, metrics: dict = None):
    """记录实验到 log"""
    record = {
        "type": exp_type,
        "algorithm": algo,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "report": report,
        "images": images or {},
        "params": params or {},
        "metrics": metrics or {},
        "dataset": _g.get("dataset_name", "未加载"),
        "model_name": _g.get("model_name", "未训练"),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if not _g.get("experiment_log"):
        _g["experiment_log"] = []
    _g["experiment_log"].append(record)
    return record


def get_experiment_history(_g: dict, exp_type: str = None, limit: int = 20) -> list:
    """获取实验历史"""
    log = _g.get("experiment_log", [])
    if exp_type:
        log = [e for e in log if e.get("type") == exp_type]
    return list(reversed(log))[:limit]


def get_all_experiment_history(_g: dict, limit: int = 50) -> list:
    """获取所有实验历史"""
    log = _g.get("experiment_log", [])
    return list(reversed(log))[:limit]


_history_style = """
<style>
.exp-view-btn:hover { background: rgba(96,165,250,0.3) !important; }
.exp-compare-table th { position: sticky; top: 0; background: #1e293b; z-index: 1; }
</style>
"""

def format_history_html(records: list, max_records: int = 15) -> str:
    """将实验记录渲染为 HTML 表格"""
    if not records:
        return '<div style="color:#94a3b8;padding:16px;text-align:center;font-size:13px;">暂无实验记录</div>'
    records = records[:max_records]
    html = _history_style
    html += '<div style="max-height:320px;overflow-y:auto;font-family:system-ui,sans-serif;">'
    html += '<table class="eval-table exp-compare-table" style="font-size:11px;width:100%;">'
    html += '<tr><th style="padding:5px 8px;text-align:left;">#</th>'
    html += '<th style="padding:5px 8px;text-align:left;">时间</th>'
    html += '<th style="padding:5px 8px;text-align:left;">类型</th>'
    html += '<th style="padding:5px 8px;text-align:left;">算法</th>'
    html += '<th style="padding:5px 8px;text-align:left;">数据集</th>'
    html += '<th style="padding:5px 8px;text-align:center;">操作</th></tr>'
    type_icons = {
        "classification": "C", "regression": "R",
        "clustering": "CL", "association": "A"
    }
    for idx, rec in enumerate(records):
        bg = 'background:rgba(255,255,255,0.02);' if idx % 2 == 0 else ''
        tn = rec.get("type", "?")
        icon = type_icons.get(tn, "?")
        html += f'<tr style="{bg}">'
        html += f'<td style="padding:4px 8px;color:#64748b;">{idx+1}</td>'
        html += f'<td style="padding:4px 8px;color:#94a3b8;">{rec.get("time", "")}</td>'
        html += f'<td style="padding:4px 8px;">{icon}</td>'
        html += f'<td style="padding:4px 8px;font-weight:600;color:#60a5fa;">{rec.get("algorithm", "")}</td>'
        html += f'<td style="padding:4px 8px;color:#94a3b8;">{rec.get("dataset", "")}</td>'
        html += f'<td style="padding:4px 8px;text-align:center;">'
        html += f'<button class="exp-view-btn" data-idx="{idx}" '
        html += f'style="background:rgba(96,165,250,0.15);color:#60a5fa;'
        html += f'border:1px solid rgba(96,165,250,0.3);border-radius:4px;'
        html += f'padding:2px 10px;font-size:11px;cursor:pointer;">查看</button>'
        html += '</td></tr>'
    html += '</table></div></div>'
    return html


def format_comparison_html(records: list) -> str:
    """多实验并排对比"""
    if not records:
        return '<div style="color:#94a3b8;padding:16px;text-align:center;">请先选择要对比的实验</div>'
    n = len(records)
    html = '<div style="font-family:system-ui,sans-serif;">'
    html += f'<h4 style="color:#e2e8f0;margin:0 0 12px;font-size:13px;">对比 {n} 个实验</h4>'
    html += '<table class="info-table" style="width:100%;">'
    html += '<tr><th style="padding:5px 10px;text-align:left;color:#94a3b8;">属性</th>'
    for rec in records:
        html += f'<th style="padding:5px 10px;text-align:center;color:#60a5fa;">{rec.get("algorithm", "?")}</th>'
    html += '</tr>'
    for label, key in [("类型","type"),("数据集","dataset"),("时间","time")]:
        html += f'<tr><td style="padding:4px 10px;color:#94a3b8;font-size:12px;">{label}</td>'
        for rec in records:
            html += f'<td style="padding:4px 10px;text-align:center;font-size:12px;">{rec.get(key, "-")}</td>'
        html += '</tr>'
    html += '</table></div>'
    return html


def export_data_csv(_g: dict, output_path: str = None) -> str:
    """将当前数据导出为 CSV"""
    X = _g.get("X")
    if X is None:
        return None
    fnames = _g.get("feature_names") or [f"F{i}" for i in range(X.shape[1])]
    y = _g.get("y")
    df = pd.DataFrame(X, columns=fnames)
    if y is not None:
        tname = _g.get("target_names", ["target"])[0] if _g.get("target_names") else "target"
        df[tname] = y
    if output_path is None:
        from datetime import datetime
        output_path = os.path.join(
            os.path.dirname(__file__), "..",
            f"export_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_path, index=False)
    return output_path


def export_experiment_report(_g: dict, exp_index: int = -1, output_path: str = None) -> str:
    """导出实验报告为 TXT"""
    log = _g.get("experiment_log", [])
    if not log:
        return None
    if exp_index < 0 or exp_index >= len(log):
        exp_index = len(log) - 1
    rec = log[exp_index]
    lines = []
    lines.append("=" * 55)
    lines.append("  ML-Lab 实验报告")
    lines.append("=" * 55)
    lines.append(f"  类型: {rec.get('type', 'N/A').upper()}")
    lines.append(f"  算法: {rec.get('algorithm', 'N/A')}")
    lines.append(f"  数据集: {rec.get('dataset', 'N/A')}")
    lines.append(f"  时间: {rec.get('time', 'N/A')}")
    lines.append("-" * 55)
    lines.append("评估结果:")
    lines.append(rec.get("report", "无"))
    lines.append("=" * 55)
    report_text = "\n".join(lines)
    if output_path is None:
        from datetime import datetime
        output_path = os.path.join(
            os.path.dirname(__file__), "..",
            f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    return output_path


def export_code_py(code_text: str, output_path: str = None) -> str:
    """导出代码为 .py 文件"""
    if not code_text or not code_text.strip():
        return None
    if output_path is None:
        from datetime import datetime
        output_path = os.path.join(
            os.path.dirname(__file__), "..",
            f"sandbox_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code_text)
    return output_path


def get_param_grid(algo: str) -> list:
    """获取算法可调优超参数网格"""
    grids = {
        "逻辑回归": [
            {"name": "C", "range": [0.01, 0.1, 1.0, 10.0, 100.0], "type": "float"},
            {"name": "max_iter", "range": [50, 100, 200, 500], "type": "int"},
        ],
        "K近邻": [{"name": "n_neighbors", "range": [3, 5, 7, 9, 11, 15], "type": "int"}],
        "决策树": [
            {"name": "max_depth", "range": [3, 5, 7, 10], "type": "int"},
            {"name": "min_samples_split", "range": [2, 5, 10], "type": "int"},
        ],
        "SVM": [
            {"name": "C", "range": [0.1, 1.0, 10, 100.0], "type": "float"},
            {"name": "gamma", "range": ["scale", "auto", 0.01, 0.1, 1.0], "type": "mixed"},
        ],
        "随机森林": [
            {"name": "n_estimators", "range": [50, 100, 200, 300], "type": "int"},
            {"name": "max_depth", "range": [3, 5, 7, 10], "type": "int"},
        ],
        "梯度提升 (GBDT)": [
            {"name": "n_estimators", "range": [50, 100, 200], "type": "int"},
            {"name": "learning_rate", "range": [0.01, 0.05, 0.1, 0.2], "type": "float"},
        ],
        "神经网络": [
            {"name": "hidden_layer_sizes", "range": ["32","64","128","64,32","128,64"], "type": "str"},
            {"name": "learning_rate_init", "range": [0.001, 0.01, 0.1], "type": "float"},
        ],
        "Ridge 回归": [{"name": "alpha", "range": [0.01, 0.1, 1.0, 10.0, 100.0], "type": "float"}],
        "Lasso 回归": [{"name": "alpha", "range": [0.001, 0.01, 0.1, 1.0, 10.0], "type": "float"}],
        "多项式回归": [{"name": "degree", "range": [2, 3, 4, 5], "type": "int"}],
        "SVR": [{"name": "C", "range": [0.1, 1.0, 10.0], "type": "float"}],
        "决策树回归": [{"name": "max_depth", "range": [3, 5, 7, 10], "type": "int"}],
        "K-Means": [{"name": "n_clusters", "range": [2, 3, 4, 5, 6, 8, 10], "type": "int"}],
        "DBSCAN": [
            {"name": "eps", "range": [0.1, 0.3,0.5, 0.8, 1.0], "type": "float"},
            {"name": "min_samples", "range": [2, 3, 5, 10], "type": "int"},
        ],
        "层次聚类": [{"name": "n_clusters", "range": [2, 3, 4, 5, 6], "type": "int"}],
    }
    return grids.get(algo, [])


def run_grid_search(_g: dict, algo: str, task_type: str, param_grid: dict,
                    scoring: str = None) -> dict:
    """网格搜索所有参数组合"""
    from ml_lab.algorithms import create_algorithm
    from sklearn.model_selection import cross_val_score
    import itertools, time

    if _g.get("X_train") is None:
        return {"error": "请先加载数据集"}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    results = []
    best_score = -float('inf')
    best_params = None

    for combo in combinations:
        params = dict(zip(keys, combo))
        try:
            start = time.time()
            model = create_algorithm(algo, **params)
            model.fit(_g["X_train"], _g["y_train"])
            elapsed = time.time() - start

            if task_type == "classification":
                score = model.score(_g["X_test"], _g["y_test"]) if hasattr(model, 'score') else 0
                if hasattr(model, 'model') and model.model:
                    cv = cross_val_score(model.model, _g["X_train"], _g["y_train"], cv=3, scoring=scoring or 'accuracy')
                    cv_mean = cv.mean()
                else:
                    cv_mean = score
            else:
                score = model.score(_g["X_test"], _g["y_test"]) if hasattr(model, 'score') else 0
                if hasattr(model, 'model') and model.model:
                    cv = cross_val_score(model.model, _g["X_train"], _g["y_train"], cv=3, scoring=scoring or 'r2')
                    cv_mean = cv.mean()
                else:
                    cv_mean = score

            results.append({
                "params": params, "test_score": round(score, 4),
                "cv_mean": round(cv_mean, 4), "elapsed": round(elapsed, 3),
            })
            if cv_mean > best_score:
                best_score = cv_mean
                best_params = params
        except Exception as e:
            results.append({
                "params": params, "test_score": 0, "cv_mean": 0, "error": str(e)[:50],
            })

    return {
        "algo": algo, "total": len(combinations),
        "best_params": best_params, "best_score": round(best_score, 4),
        "results": results, "task_type": task_type, "param_keys": keys,
    }


def format_grid_search_html(result: dict) -> str:
    """网格搜索结果格式化为 HTML"""
    if "error" in result:
        return f'<div style="color:#f87171;">{result["error"]}</div>'

    html = '<div style="font-family:system-ui,sans-serif;max-height:400px;overflow-y:auto;">'
    html += f'<h4 style="color:#e2e8f0;margin:0 0 8px;font-size:13px;">'
    html += f'网格搜索结果 - {result["algo"]}</h4>'
    html += f'<div style="font-size:12px;color:#94a3b8;margin-bottom:12px;">'
    html += f'共 {result["total"]} 个组合'
    if result["best_params"]:
        html += f' | 最佳参数: {result["best_params"]} | 得分: {result["best_score"]}'
    html += '</div>'

    html += '<table class="eval-table" style="font-size:11px;width:100%;">'
    keys = result.get("param_keys", [])
    html += '<tr style="background:rgba(255,255,255,0.05);">'
    for k in keys:
        html += f'<th style="padding:5px 8px;text-align:center;">{k}</th>'
    html += '<th style="padding:5px 8px;text-align:center;">测试得分</th>'
    html += '<th style="padding:5px 8px;text-align:center;">交叉验证</th></tr>'

    for r in result["results"]:
        is_best = r.get("params") == result["best_params"]
        bg = 'background:rgba(96,165,250,0.08);' if is_best else ''
        if not bg:
            idx_in = result["results"].index(r)
            bg = 'background:rgba(255,255,255,0.02);' if idx_in % 2 == 0 else ''
        html += f'<tr style="{bg}">'
        for k in keys:
            val = r.get("params", {}).get(k, "")
            html += f'<td style="padding:4px 8px;text-align:center;font-size:11px;">{val}</td>'
        s = r.get("test_score", 0)
        c = r.get("cv_mean", 0)
        sc = "#34d399" if s >= 0.85 else ('#fbbf24' if s >= 0.6 else '#f87171')
        cc = "#34d399" if c >= 0.85 else ('#fbbf24' if c >= 0.6 else '#f87171')
        html += f'<td style="padding:4px 8px;text-align:center;font-weight:600;color:{sc};">{s}</td>'
        html += f'<td style="padding:4px 8px;text-align:center;font-weight:600;color:{cc};">{c}</td>'
        html += '</tr>'

    html += '</table></div>'
    return html


def save_model(_g: dict, output_path: str = None) -> str:
    """保存模型为 joblib"""
    model = _g.get("model")
    if model is None:
        return None
    try:
        import joblib as jl
    except ImportError:
        import pickle as jl
    if output_path is None:
        from datetime import datetime
        model_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir,
            f"model_{_g.get('model_name','unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    jl.dump(model, output_path)
    return output_path


def list_saved_models() -> list:
    """列出已保存的模型"""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    if not os.path.exists(model_dir):
        return []
    models = []
    for f in sorted(os.listdir(model_dir), reverse=True):
        if f.endswith(('.joblib', '.pkl')):
            fpath = os.path.join(model_dir, f)
            sz = os.path.getsize(fpath)
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
            models.append({
                "name": f, "path": fpath,
                "size": f"{sz/1024:.1f} KB",
                "mtime": mt.strftime("%Y-%m-%d %H:%M"),
            })
    return models


def format_saved_models_html(models: list) -> str:
    """已保存模型列表渲染为 HTML"""
    if not models:
        return '<div style="color:#94a3b8;padding:16px;text-align:center;">暂无已保存的模型</div>'
    html = '<div style="max-height:250px;overflow-y:auto;">'
    html += '<table class="info-table" style="width:100%;font-size:12px;">'
    html += '<tr style="background:rgba(255,255,255,0.05);">'
    html += '<th style="padding:5px 10px;text-align:left;">模型文件</th>'
    html += '<th style="padding:5px 10px;text-align:right;">大小</th>'
    html += '<th style="padding:5px 10px;text-align:right;">保存时间</th></tr>'
    for m in models:
        html += f'<tr><td style="padding:4px 10px;color:#60a5fa;">{m["name"]}</td>'
        html += f'<td style="padding:4px 10px;text-align:right;color:#94a3b8;">{m["size"]}</td>'
        html += f'<td style="padding:4px 10px;text-align:right;color:#94a3b8;">{m["mtime"]}</td></tr>'
    html += '</table></div>'
    return html
