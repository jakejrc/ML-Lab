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

def plot_clustering_scatter(X, labels, title="聚类结果", centers=None, feature_names=None):
    """聚类散点图，自动 PCA 降维到 2D"""
    fig, ax = plt.subplots(figsize=(8, 6))
    centers_2d = None
    if X.shape[1] > 2:
        pca = PCA(n_components=2); X_2d = pca.fit_transform(X)
        xlabel, ylabel = "主成分 1", "主成分 2"
        if centers is not None: centers_2d = pca.transform(centers)
    else:
        X_2d = X[:,:2]; xlabel = feature_names[0] if feature_names else "特征 0"; ylabel = feature_names[1] if feature_names else "特征 1"
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(X_2d[noise_mask,0], X_2d[noise_mask,1], c='lightgray', s=15, alpha=0.4, label='噪声点', marker='x')
    cluster_labels = sorted(set(labels) - {-1})
    cmap = plt.cm.Set1(np.linspace(0,1,max(len(cluster_labels),2)))
    for i, lb in enumerate(cluster_labels):
        mask = labels == lb
        ax.scatter(X_2d[mask,0], X_2d[mask,1], c=[cmap[i%len(cmap)]], s=30, alpha=0.7, edgecolors='k', linewidths=0.3, label=f'簇 {lb} ({mask.sum()}个)')
    if centers_2d is not None:
        ax.scatter(centers_2d[:,0], centers_2d[:,1], c='black', s=120, marker='*', zorder=5, edgecolors='white', linewidths=1.5, label='聚类中心')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig


def plot_elbow_curve(history, title="肘部法则"):
    fig, ax = plt.subplots(figsize=(8, 5))
    kr = history.get("k_range", []); inertias = history.get("inertia", [])
    if not kr or not inertias:
        ax.text(0.5,0.5,"无肘部法则数据",ha='center',va='center',transform=ax.transAxes,fontsize=14); ax.set_title(title); return fig
    ax.plot(kr, inertias, 'bo-', linewidth=2, markersize=8)
    for k, v in zip(kr, inertias): ax.annotate(f'{v:.0f}', xy=(k,v), xytext=(5,5), textcoords='offset points', fontsize=8)
    if len(inertias) >= 3:
        diffs = [inertias[i]-inertias[i+1] for i in range(len(inertias)-1)]
        sd = [diffs[i]-diffs[i+1] for i in range(len(diffs)-1)]
        ei = sd.index(max(sd)); ek = kr[ei+1]
        ax.annotate(f'推荐 K={ek}', xy=(ek, inertias[ei+1]), xytext=(ek+1.2, inertias[ei+1]*0.85), arrowprops=dict(arrowstyle='->',color='red',lw=2), fontsize=11, color='red', fontweight='bold')
    ax.set_xlabel("聚类数 K"); ax.set_ylabel("惯性 (SSE)"); ax.set_title(title); ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig


def plot_dendrogram(X, title="层次聚类树状图"):
    """层次聚类树状图"""
    if not HAS_SCIPY:
        fig, ax = plt.subplots(figsize=(8,5)); ax.text(0.5,0.5,"需要 scipy 库: pip install scipy",ha='center',va='center',transform=ax.transAxes,fontsize=14); ax.set_title(title); return fig
    fig, ax = plt.subplots(figsize=(12, 6))
    Xp = PCA(n_components=min(15, X.shape[1])).fit_transform(X) if X.shape[1] > 20 else X
    Z = _linkage(Xp, method='ward')
    _dendrogram(Z, ax=ax, truncate_mode='lastp', p=12, leaf_font_size=8, show_contracted=True)
    ax.set_xlabel("样本（或簇）"); ax.set_ylabel("距离"); ax.set_title(title); ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    return fig


def plot_dbscan_eps_analysis(history, title="DBSCAN eps 参数分析"):
    """DBSCAN 不同 eps 值的聚类数变化"""
    fig, ax = plt.subplots(figsize=(8, 5))
    er = history.get("eps_range", []); cc = history.get("cluster_counts", [])
    if not er or not cc:
        ax.text(0.5,0.5,"无 eps 分析数据",ha='center',va='center',transform=ax.transAxes,fontsize=14); ax.set_title(title); return fig
    ax.plot(er, cc, 'gs-', linewidth=2, markersize=8); ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='单簇基线')
    for e, c in zip(er, cc): ax.annotate(str(c), xy=(e,c), xytext=(0,8), textcoords='offset points', fontsize=8, ha='center')
    ax.set_xlabel("eps"); ax.set_ylabel("聚类数"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# PCA 降维可视化（v3.0 新增）
# ═══════════════════════════════════════════════════════════════

def plot_pca_variance(history, title="PCA 方差解释率"):
    """各主成分方差解释率 + 累计曲线"""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    vr = history.get("full_explained_variance_ratio", []); cv = history.get("full_cumulative_variance", [])
    if not vr:
        ax1.text(0.5,0.5,"无 PCA 方差数据",ha='center',va='center',transform=ax.transAxes,fontsize=14); ax1.set_title(title); return fig
    x = range(1, len(vr)+1)
    ax1.bar(x, vr, color='steelblue', alpha=0.7, label='单个方差解释率')
    ax1.set_xlabel("主成分编号"); ax1.set_ylabel("方差解释率", color='steelblue'); ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(x, cv, 'ro-', linewidth=2, markersize=6, label='累计方差解释率')
    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.6, label='95% 阈值')
    ax2.axhline(y=0.99, color='orange', linestyle='--', alpha=0.6, label='99% 阈值')
    ax2.set_ylabel("累计方差解释率", color='red'); ax2.tick_params(axis='y', labelcolor='red')
    ax1.set_title(title); ax1.set_xticks(x)
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=8, loc='center right'); plt.tight_layout()
    return fig


def plot_pca_projection(X, labels=None, title="PCA 投影图"):
    """PCA 降维 2D 投影"""
    nd = X.shape[1]
    if nd < 2:
        fig, ax = plt.subplots(figsize=(8,6)); ax.text(0.5,0.5,"PCA 投影至少需要 2 个主成分",ha='center',va='center',transform=ax.transAxes,fontsize=14); ax.set_title(title); return fig
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        ul = sorted(set(labels)); cmap = plt.cm.Set1(np.linspace(0,1,max(len(ul),2)))
        for i, lb in enumerate(ul):
            if lb == -1:
                mask = labels == lb; ax.scatter(X[mask,0], X[mask,1], c='lightgray', s=15, alpha=0.4, marker='x', label='噪声')
            else:
                mask = labels == lb; ax.scatter(X[mask,0], X[mask,1], c=[cmap[i%len(cmap)]], s=30, alpha=0.7, edgecolors='k', linewidths=0.3, label=f'类别 {lb}')
    else:
        ax.scatter(X[:,0], X[:,1], c='steelblue', s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
    ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2"); ax.set_title(title); ax.legend(fontsize=8); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════
