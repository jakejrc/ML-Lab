# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.4 — 特征工程回调函数
从 callbacks.py 拆分而来
"""
import sys, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import gradio as gr

from ml_lab.state import _g, _sync
from ml_lab.preprocessing import get_dataset_type
from ml_lab.feature_engineering import (
    scale_data, get_scaling_stats,
    select_features_univariate, select_features_rfe, select_features_model_based,
    apply_pca, construct_polynomial_features, construct_interaction_features,
    construct_statistical_features, discretize_features, run_feature_pipeline
)
from ml_lab.visualization import (
    plot_feature_scaling_comparison, plot_feature_scaling_distribution,
    plot_feature_selection_scores, plot_feature_importance_ranking,
    plot_pca_explained_variance, plot_pca_2d_scatter,
    plot_feature_correlation_heatmap, plot_feature_engineering_summary,
    plot_feature_distribution_after_construction,
    fig_to_image,
)
from ml_lab.ui_styles import APP_CSS, _card, ETHICS

def on_fe_scaling(method):

    """模块1: 特征缩放"""

    if _g["X_train"] is None:

        return None, None, "请先在「数据工作台」加载数据集"

    try:

        X_tr = _g["X_train"].copy()

        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]

        X_scaled, _, scaler = scale_data(X_tr, method=method)

        stats = get_scaling_stats(X_tr, X_scaled, fnames)

        img1 = fig_to_image(plot_feature_scaling_comparison(

            X_tr, X_scaled, feature_names=fnames,

            title=f"特征缩放对比 ({method})"))

        img2 = fig_to_image(plot_feature_scaling_distribution(

            X_tr, X_scaled, feature_idx=0, feature_name=fnames[0],

            title=f"特征分布对比: {fnames[0]}"))

        # 构建 HTML 表格

        scale_style = """

        <style>

            .scale-table { width:100%; border-collapse:collapse; margin:6px 0; font-size:13px; }

            .scale-table th { background:transparent; color:#2d3748; padding:8px 10px; text-align:center; font-weight:600; border-bottom:2px solid #e2e8f0; }

            .scale-table td { padding:7px 10px; border-bottom:1px solid #edf2f7; text-align:center; }

            .scale-table td:first-child { text-align:left; font-weight:500; color:#4a5568; }

            .scale-table tr:hover td { background:#f7fafc; }

            .scale-meta { font-size:12px; color:#718096; margin-bottom:8px; }

            .scale-meta span { margin-right:14px; }

        </style>"""

        rows = ""

        for s in stats:

            for s in stats:

                feat = s["特征"]

                bef_mean = s["缩放前_均值"]

                aft_mean = s["缩放后_均值"]

                bef_std = s["缩放前_标准差"]

                aft_std = s["缩放后_标准差"]

                rows += "<tr><td>" + feat + "</td><td>" + f"{bef_mean:.4f}" + "</td><td>" + f"{aft_mean:.4f}" + "</td><td>" + f"{bef_std:.4f}" + "</td><td>" + f"{aft_std:.4f}" + "</td></tr>\n"

        html = ('<div class="scale-meta">'

                '<span>' + "缩放方法" + ': <b>' + method + '</b></span>'

                '<span>' + "特征数" + ': <b>' + str(X_tr.shape[1]) + '</b></span>'

                '<span>' + "样本数" + ': <b>' + str(X_tr.shape[0]) + '</b></span></div>'

                '<table class="scale-table"><thead><tr>'

                '<th>' + "特征" + '</th><th>' + "缩放前均值" + '</th><th>' + "缩放后均值" + '</th>'

                '<th>' + "缩放前标准差" + '</th><th>' + "缩放后标准差" + '</th>'

                '</tr></thead><tbody>' + rows + '</tbody></table>' + scale_style)

        return img1, img2, html

        # 保存特征工程结果用于报告

        _g["fe_results"]["scaling"] = {"method": method, "n_features": X_tr.shape[1], "n_samples": X_tr.shape[0], "image1": img1, "image2": img2, "info": "\n".join(lines)}

        return img1, img2, "\n".join(lines)

    except Exception as e:

        return None, None, f"错误: {e}"

def on_fe_feature_selection(method, k):

    """模块2: 特征选择"""

    if _g["X_train"] is None:

        return None, "请先在「数据工作台」加载数据集"

    if _g["y_train"] is None:

        return None, "标签数据不可用"

    try:

        X_tr, y_tr = _g["X_train"], _g["y_train"]

        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]

        dtype = get_dataset_type(_g["dataset_name"])

        task = "classification" if dtype == "classification" else "regression"

        k = int(k)

        k = min(k, X_tr.shape[1])



        img = None

        sel_css = """

        <style>

            .sel-table { width:100%; border-collapse:collapse; margin:6px 0; font-size:13px; }

            .sel-table th { background:transparent; color:#2d3748; padding:8px 10px; text-align:center; font-weight:600; border-bottom:2px solid #e2e8f0; }

            .sel-table td { padding:7px 10px; border-bottom:1px solid #edf2f7; text-align:center; }

            .sel-table td:first-child { text-align:left; font-weight:500; color:#4a5568; }

            .sel-table tr:hover td { background:#f7fafc; }

            .sel-table .sel-yes { color:#38a169; font-weight:700; }

            .sel-meta { font-size:12px; color:#718096; margin-bottom:8px; }

            .sel-meta span { margin-right:14px; }

        </style>"""

        sel_rows = ""



        if method in ("f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression"):

            X_sel, indices, scores, pvals = select_features_univariate(

                X_tr, y_tr, method=method, k=k, task_type=task)

            score_name = {"f_classif": "F值", "f_regression": "F值",

                          "mutual_info_classif": "互信息", "mutual_info_regression": "互信息"}.get(method, "得分")

            img = fig_to_image(plot_feature_selection_scores(

                scores, feature_names=fnames, top_n=None,

                title=f"特征选择得分 ({method})", score_name=score_name))

            p_header = ""

            if pvals is not None:

                p_header = "<th>p值</th>"

            sel_rows += f"<tr><th>排名</th><th>特征</th><th>{score_name}</th><th>选中</th>{p_header}</tr>"

            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            for rank, i in enumerate(ranked[:k], 1):

                p_str = f"<td>{pvals[i]:.4f}</td>" if pvals is not None else ""

                sel_rows += (f'<tr><td>{rank}</td><td>{fnames[i]}</td>'

                             f'<td>{scores[i]:.4f}</td>'

                             f'<td class="sel-yes">✓</td>{p_str}</tr>\n')

            for rank, i in enumerate(ranked[k:], k+1):

                p_str = f"<td>{pvals[i]:.4f}</td>" if pvals is not None else ""

                sel_rows += (f'<tr><td>{rank}</td><td>{fnames[i]}</td>'

                             f'<td>{scores[i]:.4f}</td><td></td>{p_str}</tr>\n')

        elif method == "rfe":

            X_sel, indices, ranking = select_features_rfe(X_tr, y_tr, n_features=k, task_type=task)

            img = fig_to_image(plot_feature_importance_ranking(

                1.0 / np.array(ranking, dtype=float), feature_names=fnames,

                title="RFE 特征排名 (1/排名)"))

            sel_rows += "<tr><th>特征</th><th>排名</th><th>选中</th></tr>"

            for i, r in enumerate(ranking):

                mark = '<td class="sel-yes">✓</td>' if r == 1 else '<td></td>'

                sel_rows += f'<tr><td>{fnames[i]}</td><td>{r}</td>{mark}</tr>\n'

        elif method == "model_based":

            X_sel, indices, importances = select_features_model_based(X_tr, y_tr, task_type=task)

            img = fig_to_image(plot_feature_importance_ranking(

                importances, feature_names=fnames,

                title="RandomForest 特征重要性"))

            sel_rows += "<tr><th>排名</th><th>特征</th><th>重要性</th><th>选中</th></tr>"

            ranked = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

            for rank, i in enumerate(ranked, 1):

                mark = '<td class="sel-yes">✓</td>' if i in indices else '<td></td>'

                sel_rows += f'<tr><td>{rank}</td><td>{fnames[i]}</td><td>{importances[i]:.4f}</td>{mark}</tr>\n'



        selected_names = [fnames[i] for i in indices]

        info_html = (f'<div class="sel-meta">'

                     f'<span>方法: <b>{method}</b></span>'

                     f'<span>原始特征数: <b>{X_tr.shape[1]}</b></span>'

                     f'<span>选择特征数: <b>{k}</b></span></div>'

                     f'<table class="sel-table"><thead><tr>{sel_rows[:sel_rows.find("</tr>")+5]}</tr></thead>'

                     f'<tbody>{sel_rows[sel_rows.find("</tr>")+5:]}</tbody></table>'

                     f'<div style="margin-top:8px;font-size:12px;color:#4a5568;">'

                     f'选中特征: <b>{", ".join(selected_names)}</b></div>{sel_css}')

        return img, info_html

    except Exception as e:

        return None, f"错误: {e}"

def on_fe_pca(n):

    """模块3: PCA降维"""

    if _g["X_train"] is None:

        return None, None, "请先在「数据工作台」加载数据集"

    try:

        X_tr, y_tr = _g["X_train"], _g["y_train"]

        n = int(n)

        n = min(n, X_tr.shape[1])

        X_pca, pca_model, evr = apply_pca(X_tr, n_components=n)

        img1 = fig_to_image(plot_pca_explained_variance(evr, title="PCA 方差解释率"))

        X_2d = X_pca[:, :2] if X_pca.shape[1] >= 2 else X_pca

        img2 = fig_to_image(plot_pca_2d_scatter(X_2d, y_tr, title="PCA 2D 投影"))

        cumsum = np.cumsum(evr)

        rows = ""

        for i in range(len(evr)):

            rows += "<tr><td>PC" + str(i+1) + "</td><td>" + f"{evr[i]:.4f}" + "</td><td>" + f"{cumsum[i]:.4f}" + "</td></tr>\n"

        html = (

            '<table class="info-table"><thead><tr>'

            '<th>主成分</th><th>方差解释率</th><th>累计解释率</th>'

            '</tr></thead><tbody>'

            + rows

            + '</tbody></table>'

            '<div style="margin-top:8px;color:#888;font-size:0.9em;">'

            + '前' + str(n) + '个主成分累计解释方差: ' + f"{cumsum[-1]:.4f}" + ' (' + f"{cumsum[-1]*100:.1f}" + '%)'

            + '</div>'

        )

        return img1, img2, html

    except Exception as e:

        return None, None, f"错误: {e}"

def on_fe_construct(method, degree):

    """模块4: 特征构造"""

    if _g["X_train"] is None:

        return None, "请先在「数据工作台」加载数据集"

    try:

        X_tr = _g["X_train"].copy()

        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]

        degree = int(degree)

        orig_n = X_tr.shape[1]



        if method == "polynomial":

            X_new, _ = construct_polynomial_features(X_tr, degree=degree)

            new_names = [f"P{d}" for d in range(X_new.shape[1])]

            info = (

                '<table class="info-table"><tbody>'

                '<tr><td style="font-weight:600;width:120px;">方法</td><td>多项式特征 (degree=' + str(degree) + ')</td></tr>'

                '<tr><td style="font-weight:600;">原始特征数</td><td>' + str(orig_n) + '</td></tr>'

                '<tr><td style="font-weight:600;">构造后特征数</td><td>' + str(X_new.shape[1]) + '</td></tr>'

                '</tbody></table>'

            )

        elif method == "interaction":

            X_new, new_names = construct_interaction_features(X_tr)

            info = (

                '<table class="info-table"><tbody>'

                '<tr><td style="font-weight:600;width:120px;">方法</td><td>交互特征 (两两相乘)</td></tr>'

                '<tr><td style="font-weight:600;">原始特征数</td><td>' + str(orig_n) + '</td></tr>'

                '<tr><td style="font-weight:600;">新增交互特征</td><td>' + str(len(new_names)) + '</td></tr>'

                '<tr><td style="font-weight:600;">构造后特征数</td><td>' + str(X_new.shape[1]) + '</td></tr>'

                '</tbody></table>'

            )

        elif method == "statistical":

            X_new, new_names = construct_statistical_features(X_tr)

            info = (

                '<table class="info-table"><tbody>'

                '<tr><td style="font-weight:600;width:120px;">方法</td><td>统计特征 (行统计量)</td></tr>'

                '<tr><td style="font-weight:600;">新增特征</td><td>' + ', '.join(new_names) + '</td></tr>'

                '<tr><td style="font-weight:600;">原始特征数</td><td>' + str(orig_n) + '</td></tr>'

                '<tr><td style="font-weight:600;">构造后特征数</td><td>' + str(X_new.shape[1]) + '</td></tr>'

                '</tbody></table>'

            )

        else:

            return None, f"未知方法: {method}"



        img = fig_to_image(plot_feature_distribution_after_construction(

            X_new, original_n_features=orig_n, new_feature_names=new_names,

            title=f"特征构造后分布 ({method})"))

        return img, info

    except Exception as e:

        return None, f"错误: {e}"

def on_fe_discretize(n_bins, strategy):

    """模块5: 分箱离散化"""

    if _g["X_train"] is None:

        return None, "请先在「数据工作台」加载数据集"

    try:

        X_tr = _g["X_train"].copy()

        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]

        n_bins = int(n_bins)

        X_disc, enc = discretize_features(X_tr, n_bins=n_bins, strategy=strategy)



        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(X_tr[:, 0], bins=30, alpha=0.6, color='steelblue', edgecolor='k', label='原始分布')

        unique_bins = np.unique(X_disc[:, 0])

        colors = ['red','green','blue','orange','purple','cyan','pink','brown','gray','olive']

        for b in unique_bins:

            mask = X_disc[:, 0] == b

            if mask.any():

                lo, hi = X_tr[mask, 0].min(), X_tr[mask, 0].max()

                ax.axvspan(lo, hi, alpha=0.15, color=colors[int(b) % len(colors)])

        ax.set_title(f"分箱效果: {fnames[0]} ({strategy}, {n_bins}箱)")

        ax.set_xlabel(fnames[0])

        ax.set_ylabel("频数")

        ax.legend()

        ax.grid(True, alpha=0.3)

        img = fig_to_image(fig)



        bin_edges = enc.bin_edges_[0]

        edges_str = ", ".join([f"{e:.2f}" for e in bin_edges])

        info = (

            '<table class="info-table"><tbody>'

            '<tr><td style="font-weight:600;width:120px;">策略</td><td>' + strategy + '</td></tr>'

            '<tr><td style="font-weight:600;">分箱数</td><td>' + str(n_bins) + '</td></tr>'

            '<tr><td style="font-weight:600;">特征数</td><td>' + str(X_tr.shape[1]) + '</td></tr>'

            '<tr><td style="font-weight:600;">' + fnames[0] + ' 分箱边界</td><td>[' + edges_str + ']</td></tr>'

            '</tbody></table>'

        )

        return img, info

    except Exception as e:

        return None, f"错误: {e}"

def on_fe_correlation():

    """模块6: 相关性分析"""

    if _g["X_train"] is None:

        return None, "请先在「数据工作台」加载数据集"

    try:

        X_tr = _g["X_train"]

        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]

        corr = np.corrcoef(X_tr, rowvar=False)

        img = fig_to_image(plot_feature_correlation_heatmap(X_tr, feature_names=fnames))



        n = X_tr.shape[1]

        high_corr = []

        for i in range(n):

            for j in range(i + 1, n):

                r = abs(corr[i, j])

                if r > 0.7:

                    high_corr.append((fnames[i], fnames[j], corr[i, j]))

        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)



        html_parts = []

        html_parts.append(

            '<table class="info-table"><tbody>'

            '<tr><td style="font-weight:600;width:120px;">特征数</td><td>' + str(n) + '</td></tr>'

            '<tr><td style="font-weight:600;">高相关阈值</td><td>|r| &gt; 0.7</td></tr>'

            '</tbody></table>'

        )

        if high_corr:

            rows = ""

            for f1, f2, r in high_corr[:10]:

                rows += "<tr><td>" + f1 + "</td><td>" + f2 + "</td><td>" + f"{r:.4f}" + "</td></tr>\n"

            html_parts.append(

                '<div style="margin-top:8px;font-weight:600;">高相关性特征对 (|r| &gt; 0.7)</div>'

                '<table class="info-table"><thead><tr>'

                '<th>特征1</th><th>特征2</th><th>相关系数</th>'

                '</tr></thead><tbody>'

                + rows

                + '</tbody></table>'

            )

        else:

            html_parts.append(

                '<div style="margin-top:8px;color:#888;">无高相关性特征对 (所有特征对 |r| ≤ 0.7)</div>'

            )

        return img, "\n".join(html_parts)

    except Exception as e:

        return None, f"错误: {e}"
