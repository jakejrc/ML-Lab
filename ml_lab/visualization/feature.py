# -*- coding: utf-8 -*-
"""
ML-Lab 可视化模块 v3.0
v3.0 新增：聚类散点图、肘部法则图、层次聚类树状图、DBSCAN eps 分析、PCA 方差解释图、PCA 投影图。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---- 中文字体配置 ----
_CN_FONTS = [f.name for f in matplotlib.font_manager.fontManager.ttflist
             if any(kw in f.name for kw in ['WenQuanYi', 'Noto Sans CJK', 'SimHei', 'Microsoft YaHei'])]
if _CN_FONTS:
    matplotlib.rcParams['font.sans-serif'] = _CN_FONTS + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

try:
    from scipy.cluster.hierarchy import dendrogram as _dendrogram, linkage as _linkage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import platform, os, matplotlib.font_manager as fm

def plot_feature_scaling_comparison(X_before, X_after, feature_names=None, title="特征缩放对比"):
    """对比缩放前后特征分布 (箱线图)"""
    n_features = min(X_before.shape[1], 6)  # 最多展示6个特征
    feature_names = feature_names or [f"F{i}" for i in range(X_before.shape[1])]

    fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 7))
    if n_features == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_features):
        # 缩放前
        axes[0, i].boxplot(X_before[:, i].astype(float), vert=True, patch_artist=True,
                           boxprops=dict(facecolor='#93c5fd', color='#1e40af'),
                           medianprops=dict(color='#dc2626', linewidth=2))
        axes[0, i].set_title(f"{feature_names[i]}\n缩放前", fontsize=9)
        axes[0, i].grid(True, alpha=0.3, axis='y')

        # 缩放后
        axes[1, i].boxplot(X_after[:, i].astype(float), vert=True, patch_artist=True,
                           boxprops=dict(facecolor='#86efac', color='#166534'),
                           medianprops=dict(color='#dc2626', linewidth=2))
        axes[1, i].set_title(f"{feature_names[i]}\n缩放后", fontsize=9)
        axes[1, i].grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_feature_scaling_distribution(X_before, X_after, feature_idx=0,
                                       feature_name=None, title="特征分布对比"):
    """对比单个特征缩放前后的分布 (直方图)"""
    feature_name = feature_name or f"特征{feature_idx}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(X_before[:, feature_idx].astype(float), bins=30, color='#3b82f6',
                 alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[0].axvline(np.mean(X_before[:, feature_idx]), color='#dc2626', linestyle='--',
                    linewidth=1.5, label=f"均值={np.mean(X_before[:, feature_idx]):.2f}")
    axes[0].set_title(f"{feature_name} - 缩放前", fontsize=11)
    axes[0].set_xlabel("值")
    axes[0].set_ylabel("频数")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(X_after[:, feature_idx].astype(float), bins=30, color='#22c55e',
                 alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[1].axvline(np.mean(X_after[:, feature_idx]), color='#dc2626', linestyle='--',
                    linewidth=1.5, label=f"均值={np.mean(X_after[:, feature_idx]):.2f}")
    axes[1].set_title(f"{feature_name} - 缩放后", fontsize=11)
    axes[1].set_xlabel("值")
    axes[1].set_ylabel("频数")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_feature_selection_scores(scores, feature_names=None, top_n=None,
                                  title="特征选择得分", score_name="得分"):
    """特征选择得分条形图"""
    feature_names = feature_names or [f"F{i}" for i in range(len(scores))]
    n = top_n or len(scores)

    # 排序
    sorted_idx = np.argsort(scores)[::-1][:n]
    sorted_scores = scores[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # 颜色渐变
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n))

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(5, n * 0.4)))
    bars = ax.barh(range(n), sorted_scores, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel(score_name, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # 在条形上标注数值
    for bar, score in zip(bars, sorted_scores):
        ax.text(bar.get_width() + max(sorted_scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_feature_importance_ranking(importances, feature_names=None,
                                     title="基于模型的特征重要性排名"):
    """RandomForest 特征重要性排名"""
    feature_names = feature_names or [f"F{i}" for i in range(len(importances))]
    sorted_idx = np.argsort(importances)[::-1]
    sorted_imp = importances[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))

    fig, ax = plt.subplots(figsize=(max(8, len(importances) * 0.8), max(5, len(importances) * 0.4)))
    ax.barh(range(len(sorted_imp)), sorted_imp, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(sorted_imp)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("重要性", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    for i, (idx, imp) in enumerate(zip(sorted_idx, sorted_imp)):
        ax.text(imp + max(sorted_imp) * 0.01, i, f'{imp:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_pca_explained_variance(explained_variance_ratio, title="PCA 方差解释率"):
    """PCA 方差解释率 - 柱状图+累积线"""
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = range(1, n + 1)
    bars = ax1.bar(x, explained_variance_ratio, color='#3b82f6', alpha=0.7,
                   edgecolor='white', label='各主成分方差解释率')
    ax1.set_xlabel("主成分", fontsize=11)
    ax1.set_ylabel("方差解释率", fontsize=11, color='#3b82f6')
    ax1.tick_params(axis='y', labelcolor='#3b82f6')

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, 'ro-', linewidth=2, markersize=6, label='累计方差解释率')
    ax2.set_ylabel("累计方差解释率", fontsize=11, color='#dc2626')
    ax2.tick_params(axis='y', labelcolor='#dc2626')

    # 标注95%线
    ax2.axhline(y=0.95, color='#f59e0b', linestyle='--', alpha=0.7, label='95%阈值')
    ax2.text(n * 0.6, 0.96, '95%', fontsize=10, color='#f59e0b')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    ax1.set_xticks(list(x))
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_pca_2d_scatter(X_pca, y, title="PCA 2D 投影"):
    """PCA 2D 散点图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))

    for cls, color in zip(classes, colors):
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=str(cls),
                   alpha=0.6, edgecolors='white', linewidths=0.5, s=60)

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(title="类别", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_correlation_heatmap(X, feature_names=None, title="特征相关性热力图"):
    """特征相关性热力图"""
    feature_names = feature_names or [f"F{i}" for i in range(X.shape[1])]
    corr = np.corrcoef(X.T)

    fig, ax = plt.subplots(figsize=(max(8, X.shape[1] * 0.8), max(6, X.shape[1] * 0.6)))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(feature_names, fontsize=9)

    # 标注数值
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(corr[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8, label='相关系数')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_feature_engineering_summary(results, title="特征工程流水线摘要"):
    """特征工程流水线步骤摘要可视化"""
    steps = results.get("步骤结果", [])
    if not steps:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "无处理步骤", ha='center', va='center', fontsize=14, color='gray')
        return fig

    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
    if len(steps) == 1:
        axes = [axes]

    for i, step in enumerate(steps):
        dims = step.get("输出维度", step.get("当前维度", (0, 0)))
        ax = axes[i]

        # 步骤卡片
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.85, facecolor='#eff6ff',
                                   edgecolor='#3b82f6', linewidth=2, transform=ax.transAxes))

        step_label = step.get("步骤", "unknown")
        # 中文映射
        label_map = {
            "scaling": "特征缩放",
            "feature_selection": "特征选择",
            "pca": "PCA降维",
            "polynomial": "多项式特征",
            "interaction": "交互特征",
            "statistical": "统计特征",
            "discretize": "分箱离散化",
        }
        cn_label = label_map.get(step_label, step_label)

        ax.text(0.5, 0.82, cn_label, ha='center', va='center', fontsize=12,
                fontweight='bold', color='#1e40af', transform=ax.transAxes)
        ax.text(0.5, 0.55, f"{dims[0]} x {dims[1]}", ha='center', va='center',
                fontsize=16, fontweight='bold', color='#1e293b', transform=ax.transAxes)
        ax.text(0.5, 0.35, f"样本 x 特征", ha='center', va='center',
                fontsize=9, color='#64748b', transform=ax.transAxes)

        # 参数信息
        params = step.get("参数", {})
        param_text = ", ".join([f"{k}={v}" for k, v in params.items()])
        if param_text:
            ax.text(0.5, 0.2, param_text, ha='center', va='center',
                    fontsize=8, color='#94a3b8', transform=ax.transAxes)

        ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_feature_distribution_after_construction(X_new, original_n_features,
                                                 new_feature_names=None, title="构造后特征分布"):
    """展示新增构造特征的分布"""
    n_original = original_n_features
    n_new = X_new.shape[1] - n_original
    if n_new <= 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "无新增特征", ha='center', va='center', fontsize=14, color='gray')
        return fig

    new_feature_names = new_feature_names or [f"新特征{i}" for i in range(n_new)]
    n_show = min(n_new, 6)

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        col = n_original + i
        axes[i].hist(X_new[:, col], bins=25, color='#8b5cf6', alpha=0.7,
                     edgecolor='white', linewidth=0.5)
        axes[i].set_title(new_feature_names[i] if i < len(new_feature_names) else f"新特征{i}",
                          fontsize=10)
        axes[i].set_xlabel("值", fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
