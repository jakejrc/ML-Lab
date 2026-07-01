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

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="混淆矩阵"):
    cm = confusion_matrix(y_true, y_pred); nc = cm.shape[0]
    if class_names is None: class_names = [f"类别{i}" for i in range(nc)]
    fig, ax = plt.subplots(figsize=(max(6,nc), max(5,nc)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues'); ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(nc), yticks=np.arange(nc), xticklabels=class_names, yticklabels=class_names, ylabel="真实标签", xlabel="预测标签", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max()/2.0
    for i in range(nc):
        for j in range(nc):
            ax.text(j, i, format(cm[i,j],'d'), ha="center", va="center", color="white" if cm[i,j]>thresh else "black", fontsize=12)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, class_names=None, title="ROC曲线"):
    nc = y_proba.shape[1]
    if class_names is None: class_names = [f"类别{i}" for i in range(nc)]
    fig, ax = plt.subplots(figsize=(8,6))
    if nc == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:,1]); auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={auc_val:.3f})')
    else:
        yb = label_binarize(y_true, classes=range(nc)); all_fpr = np.linspace(0,1,100); mean_tpr = np.zeros_like(all_fpr)
        for i in range(nc):
            fi, ti, _ = roc_curve(yb[:,i], y_proba[:,i]); mean_tpr += np.interp(all_fpr, fi, ti)
            ax.plot(fi, ti, lw=1, alpha=0.5, label=f'{class_names[i]} (AUC={auc(fi,ti):.2f})')
        mean_tpr /= nc; ax.plot(all_fpr, mean_tpr, 'k--', lw=2, label=f'宏平均 (AUC={auc(all_fpr,mean_tpr):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,label='随机基线'); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("假正率"); ax.set_ylabel("真正率"); ax.set_title(title); ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names=None, top_n=10, title="特征重要性"):
    if feature_names is None: feature_names = [f"特征{i}" for i in range(len(importances))]
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n*0.5)))
    bars = ax.barh(range(top_n), importances[indices], color='steelblue', edgecolor='k')
    ax.set_yticks(range(top_n)); ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("重要性"); ax.set_title(title); ax.invert_yaxis()
    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    return fig

