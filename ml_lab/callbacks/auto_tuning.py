# -*- coding: utf-8 -*-
"""超参数自动搜索模块"""
import json
import itertools
import time
import numpy as np
from ml_lab.state import _g
from sklearn.model_selection import cross_val_score


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
            {"name": "eps", "range": [0.1, 0.3, 0.5, 0.8, 1.0], "type": "float"},
            {"name": "min_samples", "range": [2, 3, 5, 10], "type": "int"},
        ],
        "层次聚类": [{"name": "n_clusters", "range": [2, 3, 4, 5, 6], "type": "int"}],
    }
    return grids.get(algo, [])


def format_param_grid_html(algo, grid):
    """将参数网格格式化为HTML显示"""
    if not grid:
        return '<div style="color:#94a3b8;">暂无可调优参数</div>'
    html = '<div style="font-size:12px;color:#e2e8f0;margin-bottom:8px;">'
    html += f'<b>{algo}</b> 可调优参数：</div><table style="font-size:11px;width:100%;">'
    html += '<tr><th style="color:#94a3b8;text-align:left;">参数</th><th style="color:#94a3b8;text-align:left;">搜索范围</th></tr>'
    for p in grid:
        r = p["range"]
        r_str = f'{r[0]} ~ {r[-1]} ({len(r)}个值)' if len(r) > 2 else str(r)
        html += f'<tr><td style="padding:2px 8px;">{p["name"]}</td><td style="padding:2px 8px;">{r_str}</td></tr>'
    html += '</table>'
    return html


def on_auto_tune(algo, task_type):
    """自动调参回调"""
    from ml_lab.algorithms import create_algorithm

    if _g.get("X_train") is None:
        return "请先在数据工作台加载数据集", ""

    grid = get_param_grid(algo)
    if not grid:
        return f"算法 [{algo}] 暂无预定义参数网格", ""

    # 构建参数网格字典
    param_grid = {}
    for p in grid:
        param_grid[p["name"]] = p["range"]

    # 执行网格搜索
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    scoring = "accuracy" if task_type == "classification" else ("r2" if task_type == "regression" else "silhouette")

    results = []
    best_score = -float("inf")
    best_params = None
    total = len(combinations)

    status_prefix = f"正在搜索 {total} 个参数组合..."

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        try:
            start_t = time.time()
            model = create_algorithm(algo, **params)
            model.fit(_g["X_train"], _g["y_train"])
            elapsed = time.time() - start_t

            # 评分
            score = 0
            if hasattr(model, "score"):
                try:
                    score = model.score(_g["X_test"], _g["y_test"])
                except Exception:
                    pass

            cv_mean = score
            if hasattr(model, "model") and model.model is not None:
                try:
                    cv_scores = cross_val_score(
                        model.model, _g["X_train"], _g["y_train"],
                        cv=3, scoring=scoring
                    )
                    cv_mean = cv_scores.mean()
                except Exception:
                    pass

            results.append({
                "params": params, "test_score": round(score, 4),
                "cv_mean": round(cv_mean, 4), "elapsed": round(elapsed, 3),
            })
            if cv_mean > best_score:
                best_score = cv_mean
                best_params = params

        except Exception as e:
            results.append({
                "params": params, "test_score": 0, "cv_mean": 0,
                "error": str(e)[:60],
            })

    # 生成结果HTML
    html = f'<div style="font-family:system-ui,sans-serif;">'
    html += f'<h4 style="color:#e2e8f0;margin:0 0 8px;">网格搜索结果 - {algo}</h4>'
    html += f'<div style="font-size:12px;color:#94a3b8;margin-bottom:8px;">'
    html += f'共 {total} 个组合 | 最佳参数: {best_params} | CV得分: {best_score:.4f}</div>'
    html += '<table class="eval-table" style="font-size:11px;width:100%;">'
    html += '<tr><th>#</th>'
    for k in keys:
        html += f'<th>{k}</th>'
    html += '<th>测试得分</th><th>CV得分</th></tr>'

    for i, r in enumerate(results):
        color = "#22c55e" if r.get("error") is None else "#f87171"
        html += f'<tr style="color:{color};">'
        html += f'<td>{i+1}</td>'
        for k in keys:
            v = r["params"].get(k, "")
            html += f'<td>{v}</td>'
        html += f'<td>{r["test_score"]}</td>'
        html += f'<td>{r["cv_mean"]}</td>'
        html += '</tr>'

    html += '</table></div>'

    return status_prefix + f" 完成！最佳: {best_params}", html
