# -*- coding: utf-8 -*-
"""ML-Lab 算法推荐向导。
根据数据特征自动推荐适合的机器学习算法，包含推荐理由和适用场景。
"""

import numpy as np


def recommend_algorithms(X, y, task_type="classification", dataset_name="", feature_names=None):
    """根据数据特征推荐算法"""
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y)) if y is not None else 0
    is_binary = n_classes == 2
    has_missing = np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y)))
    n_categories = sum(1 for i in range(min(n_features, 10))
                       if len(np.unique(X[:, i])) < 10) if n_features > 0 else 0
    high_dim = n_features > 50
    large_sample = n_samples > 10000
    small_sample = n_samples < 200
    sparse_ratio = n_categories / min(n_features, 10) if n_features > 0 else 0
    is_sparse = sparse_ratio > 0.6

    recommendations = []

    if task_type == "classification":
        _add_classification_recs(recommendations, n_samples, n_features, n_classes,
                                 is_binary, has_missing, high_dim, large_sample,
                                 small_sample, is_sparse, n_categories)
    elif task_type == "regression":
        _add_regression_recs(recommendations, n_samples, n_features,
                             has_missing, high_dim, large_sample, small_sample)
    elif task_type == "clustering":
        _add_clustering_recs(recommendations, n_samples, n_features, n_classes,
                             has_missing, large_sample, small_sample)

    recommendations.sort(key=lambda r: r["score"], reverse=True)

    context = {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "task_type": task_type,
        "is_binary": is_binary,
        "has_missing": has_missing,
        "high_dim": high_dim,
        "large_sample": large_sample,
        "small_sample": small_sample,
    }
    return recommendations, context


def _add_classification_recs(recs, n, f, nc, bin_cls, miss, hd, large, small, sparse, ncat):
    # 逻辑回归
    s, r = 85, []
    if f <= 50: s += 5; r.append("特征维度适中")
    else: s -= 5; r.append("高维特征建议使用正则化")
    if bin_cls: s += 5; r.append("二分类问题效果最佳")
    else: r.append("支持多分类 (One-vs-Rest)")
    if not miss: s += 3; r.append("数据完整")
    else: s -= 5; r.append("需先处理缺失值")
    if large: s += 3; r.append("大数据集训练快")
    r.append("可解释性强，系数即特征权重")
    recs.append({"algorithm": "逻辑回归", "reason": "；".join(r),
                 "difficulty": "★★☆", "score": min(s, 100), "page": "分类实验",
                 "params": {"算法": "逻辑回归", "学习率": 0.01, "迭代次数": 200}})

    # K近邻
    s, r = 80, []
    if small: s += 8; r.append("小样本场景表现稳定")
    elif large: s -= 10; r.append("大数据集预测变慢")
    if f <= 20: s += 5; r.append("低维特征距离度量有效")
    elif hd: s -= 10; r.append("高维下距离度量失效")
    if miss: s -= 3; r.append("对缺失值敏感")
    r.append("无需训练，惰性学习")
    recs.append({"algorithm": "K近邻 (KNN)", "reason": "；".join(r),
                 "difficulty": "★☆☆", "score": max(s, 30), "page": "分类实验",
                 "params": {"算法": "K近邻", "K值": 5}})

    # 决策树
    s, r = 85, []
    if miss: s += 5; r.append("天然处理缺失值")
    if sparse: s += 5; r.append("处理离散特征效果好")
    r.append("可解释性强，规则可视化")
    if small: s -= 5; r.append("小样本易过拟合")
    if f > 30: s -= 3; r.append("高维需控制深度")
    recs.append({"algorithm": "决策树", "reason": "；".join(r),
                 "difficulty": "★☆☆", "score": min(s, 100), "page": "分类实验",
                 "params": {"算法": "决策树", "最大深度": 5, "分裂标准": "gini"}})

    # SVM
    s, r = 75, []
    if small and f <= 20: s += 10; r.append("小样本高维表现出色")
    elif large: s -= 10; r.append("大数据集训练耗时")
    if bin_cls: s += 8; r.append("二分类最优算法之一")
    r.append("核技巧处理非线性边界")
    if f > n: s += 5; r.append("特征数>样本数仍有效")
    recs.append({"algorithm": "支持向量机 (SVM)", "reason": "；".join(r),
                 "difficulty": "★★★", "score": min(s, 100), "page": "分类实验",
                 "params": {"算法": "SVM", "C值": 1.0, "核函数": "rbf"}})

    # 朴素贝叶斯
    s, r = 70, []
    if sparse or ncat > f * 0.3: s += 10; r.append("离散/文本特征天然适配")
    if hd: s += 5; r.append("高维特征表现稳定")
    r.append("计算极快，适合在线学习")
    recs.append({"algorithm": "朴素贝叶斯", "reason": "；".join(r),
                 "difficulty": "★★☆", "score": min(s, 100), "page": "分类实验",
                 "params": {"算法": "朴素贝叶斯"}})

    # 随机森林
    s, r = 90, []
    if not small: s += 5; r.append("集成学习方差低泛化强")
    if miss: s += 5; r.append("对缺失值和异常值鲁棒")
    if hd: s += 5; r.append("高维自动选择重要特征")
    if large: s += 3; r.append("可并行训练")
    r.append("特征重要性可解释")
    recs.append({"algorithm": "随机森林", "reason": "；".join(r),
                 "difficulty": "★★☆", "score": min(s, 100), "page": "分类实验",
                 "params": {"算法": "随机森林", "树数量": 100}})

    # GBDT
    s, r = 88, []
    if not small: s += 5; r.append("Boosting精度高")
    if f <= 30: s += 3; r.append("低维特征提升显著")
    r.append("竞赛/工业界首选")
    if large: s -= 3; r.append("大数据需控制树数")
    recs.append({"algorithm": "梯度提升树 (GBDT)", "reason": "；".join(r),
                 "difficulty": "★★★", "score": min(s, 100), "page": "分类实验",
                 "params": {"算法": "GBDT", "学习率": 0.1, "树数量": 100}})

    # 神经网络
    s, r = 78, []
    if large: s += 10; r.append("大数据发挥深度学习优势")
    elif small: s -= 10; r.append("小样本易过拟合")
    if hd: s += 5; r.append("自动提取高阶特征")
    if f <= 5 and small: s -= 5; r.append("简单任务无需深度模型")
    r.append("需调参，计算资源要求高")
    recs.append({"algorithm": "神经网络 (MLP)", "reason": "；".join(r),
                 "difficulty": "★★★", "score": max(s, 30), "page": "分类实验",
                 "params": {"算法": "神经网络", "隐藏层": "64,32", "学习率": 0.001}})


def _add_regression_recs(recs, n, f, miss, hd, large, small):
    # 线性回归
    s, r = 85, ["最基础的回归模型，可解释性强"]
    if f <= 10: s += 5; r.append("低维特征线性可解释")
    if hd: s -= 5; r.append("高维需使用正则化")
    recs.append({"algorithm": "线性回归", "reason": "；".join(r),
                 "difficulty": "★☆☆", "score": min(s, 100), "page": "回归实验",
                 "params": {"算法": "线性回归"}})

    # 岭回归
    s, r = 82, ["L2正则化防止过拟合"]
    if hd: s += 8; r.append("高维特征下更稳定")
    r.append("特征共线性时仍有效")
    recs.append({"algorithm": "岭回归 (Ridge)", "reason": "；".join(r),
                 "difficulty": "★★☆", "score": min(s, 100), "page": "回归实验",
                 "params": {"算法": "岭回归", "正则化强度": 1.0}})

    # Lasso
    s, r = 80, ["L1正则化自动特征选择"]
    if hd: s += 8; r.append("高维稀疏场景表现优异")
    recs.append({"algorithm": "Lasso 回归", "reason": "；".join(r),
                 "difficulty": "★★☆", "score": min(s, 100), "page": "回归实验",
                 "params": {"算法": "Lasso 回归", "正则化强度": 1.0}})

    # 决策树回归
    s, r = 78, ["非线性关系建模"]
    if miss: s += 5; r.append("天然处理缺失值")
    r.append("可解释性中等")
    recs.append({"algorithm": "决策树回归", "reason": "；".join(r),
                 "difficulty": "★☆☆", "score": min(s, 100), "page": "回归实验",
                 "params": {"算法": "决策树回归", "最大深度": 5}})

    # 随机森林回归
    s, r = 90, ["集成学习高精度"]
    if not small: s += 5; r.append("降低方差泛化强")
    if miss: s += 3; r.append("对异常值鲁棒")
    recs.append({"algorithm": "随机森林回归", "reason": "；".join(r),
                 "difficulty": "★★☆", "score": min(s, 100), "page": "回归实验",
                 "params": {"算法": "随机森林回归", "树数量": 100}})

    # SVR
    s, r = 72, ["核技巧处理非线性回归"]
    if small: s += 8; r.append("小样本表现好")
    elif large: s -= 8; r.append("大数据集训练慢")
    recs.append({"algorithm": "支持向量回归 (SVR)", "reason": "；".join(r),
                 "difficulty": "★★★", "score": max(s, 30), "page": "回归实验",
                 "params": {"算法": "SVR", "C值": 1.0, "核函数": "rbf"}})


def _add_clustering_recs(recs, n, f, nc, miss, large, small):
    # K-Means
    s, r = 90, ["简单快速，最常用聚类算法"]
    if f <= 20: s += 5; r.append("低维数据聚类效果好")
    if large: s += 5; r.append("大数据集高效可扩展")
    if nc > 0 and nc <= 10: s += 3; r.append(f"类别数{nc}适合作为K值参考")
    recs.append({"algorithm": "K-Means", "reason": "；".join(r),
                 "difficulty": "★☆☆", "score": min(s, 100), "page": "聚类实验",
                 "params": {"算法": "K-Means", "聚类数": min(nc, 5) if nc > 0 else 3}})

    # DBSCAN
    s, r = 78, ["无需预设聚类数，识别噪声点"]
    if f <= 10: s += 5; r.append("低维密度聚类有效")
    if miss: s -= 5; r.append("对缺失值敏感")
    if nc > 5: s += 5; r.append("非球形聚类效果好")
    recs.append({"algorithm": "DBSCAN", "reason": "；".join(r),
                 "difficulty": "★★★", "score": min(s, 100), "page": "聚类实验",
                 "params": {"算法": "DBSCAN", "邻域半径": 0.5, "最小样本数": 5}})

    # 层次聚类
    s, r = 75, ["树状图可视化层次结构"]
    if small: s += 10; r.append("小样本层次结构清晰")
    elif large: s -= 10; r.append("大数据集计算开销大")
    recs.append({"algorithm": "层次聚类", "reason": "；".join(r),
                 "difficulty": "★★☆", "score": max(s, 30), "page": "聚类实验",
                 "params": {"算法": "层次聚类", "聚类数": min(nc, 5) if nc > 0 else 3,
                            "链接方式": "ward"}})


def format_recommendation_html(recommendations, context):
    """将推荐结果格式化为 HTML 卡片"""
    if not recommendations:
        return '<div class="card" style="padding:20px;text-align:center;color:#94a3b8;">暂无推荐算法</div>'

    ctx = context
    task_map = {"classification": "分类", "regression": "回归", "clustering": "聚类"}
    ctx_html = (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;font-size:12px;">'
        f'<span style="background:#eff6ff;color:#1e40af;padding:3px 10px;border-radius:12px;">'
        f'📊 {ctx["dataset_name"] or "当前数据集"}</span>'
        f'<span style="background:#f0fdf4;color:#166534;padding:3px 10px;border-radius:12px;">'
        f'任务: {task_map.get(ctx["task_type"], ctx["task_type"])}</span>'
        f'<span style="background:#fefce8;color:#854d0e;padding:3px 10px;border-radius:12px;">'
        f'样本: {ctx["n_samples"]}</span>'
        f'<span style="background:#fef2f2;color:#991b1b;padding:3px 10px;border-radius:12px;">'
        f'特征: {ctx["n_features"]}</span>'
        + (f'<span style="background:#f5f3ff;color:#5b21b6;padding:3px 10px;border-radius:12px;">'
           f'类别: {ctx["n_classes"]}</span>' if ctx["n_classes"] > 0 else "")
        + '</div>'
    )

    recs_html = ""
    for i, rec in enumerate(recommendations):
        score = rec["score"]
        if score >= 85:
            level, color, bg = "强烈推荐", "#059669", "#f0fdf4"
        elif score >= 70:
            level, color, bg = "推荐", "#2563eb", "#eff6ff"
        elif score >= 55:
            level, color, bg = "可选", "#d97706", "#fefce8"
        else:
            level, color, bg = "补充", "#6b7280", "#f9fafb"

        params_str = " | ".join(f"{k}={v}" for k, v in rec.get("params", {}).items())

        recs_html += (
            f'<div style="display:flex;align-items:center;padding:10px 14px;margin-bottom:6px;'
            f'background:{bg};border-radius:8px;border-left:3px solid {color};">'
            f'<div style="flex:1;">'
            f'<div style="font-weight:600;font-size:14px;color:#1e293b;">'
            f'{rec["algorithm"]}'
            f'<span style="font-size:11px;color:{color};margin-left:6px;">{level}</span>'
            f'<span style="font-size:11px;color:#94a3b8;margin-left:4px;">{rec["difficulty"]}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:2px;">{rec["reason"]}</div>'
            f'<div style="font-size:11px;color:#94a3b8;margin-top:1px;">参数建议: {params_str}</div>'
            f'</div>'
            f'<div style="font-size:16px;font-weight:700;color:{color};min-width:36px;text-align:center;">'
            f'{score}</div></div>'
        )

    html = (
        '<div class="card" style="padding:16px;">'
        '<div style="font-size:15px;font-weight:600;color:#1e293b;margin-bottom:10px;">'
        '🤖 算法推荐'
        '<span style="font-size:11px;font-weight:400;color:#94a3b8;margin-left:6px;">'
        '基于数据特征自动分析 · 评分越高越推荐</span>'
        '</div>'
        f'{ctx_html}{recs_html}'
        '<div style="font-size:10px;color:#94a3b8;margin-top:8px;text-align:center;">'
        '💡 点击左侧导航栏对应实验页面开始训练 · '
        '推荐仅供参考，建议多算法对比</div>'
        '</div>'
    )
    return html
