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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

try:
    from scipy.cluster.hierarchy import dendrogram as _dendrogram, linkage as _linkage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import platform, os, matplotlib.font_manager as fm

def _setup_chinese_font():
    """跨平台中文字体配置"""
    _system = platform.system()
    if _system == 'Windows':
        _fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi']
    elif _system == 'Darwin':
        _fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:
        # Linux / Docker — 优先查找已安装的 CJK 字体
        _fonts = []
        _candidates = [
            'Noto Sans CJK SC', 'Noto Sans SC', 'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'DejaVu Sans',
        ]
        _available = {f.name for f in fm.fontManager.ttflist}
        for _c in _candidates:
            if _c in _available:
                _fonts.append(_c)
        if not _fonts:
            _fonts = ['DejaVu Sans']
    plt.rcParams['font.sans-serif'] = _fonts
    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()


# ═══════════════════════════════════════════════════════════════
# 数据探索
# ═══════════════════════════════════════════════════════════════

def plot_data_distribution(X, y, title="数据分布", feature_names=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if X.shape[1] > 2:
        pca = PCA(n_components=2); X_2d = pca.fit_transform(X)
        xlabel, ylabel = "主成分 1", "主成分 2"
        title = f"{title} (PCA 降维)"
    else:
        X_2d = X[:, :2]
        xlabel = feature_names[0] if feature_names else "特征 0"
        ylabel = feature_names[1] if feature_names else "特征 1"
    classes = np.unique(y)
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(classes), 2)))
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap[i]], label=f'类别 {cls}', alpha=0.7, s=30, edgecolors='k', linewidths=0.3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig


def plot_preprocessing_comparison(X_before, X_after, title="预处理对比", feature_idx=0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(X_before[:, feature_idx].astype(float), bins=30, color='salmon', edgecolor='k', alpha=0.7)
    axes[0].set_title(f"预处理前 — 特征 {feature_idx}"); axes[0].set_xlabel("特征值"); axes[0].set_ylabel("频数")
    axes[0].axvline(np.nanmean(X_before[:, feature_idx]), color='red', linestyle='--', label=f'均值: {np.nanmean(X_before[:, feature_idx]):.2f}')
    axes[0].legend(fontsize=8)
    axes[1].hist(X_after[:, feature_idx], bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    axes[1].set_title(f"预处理后 — 特征 {feature_idx}"); axes[1].set_xlabel("特征值"); axes[1].set_ylabel("频数")
    axes[1].axvline(np.nanmean(X_after[:, feature_idx]), color='red', linestyle='--', label=f'均值: {np.nanmean(X_after[:, feature_idx]):.2f}')
    axes[1].legend(fontsize=8)
    plt.suptitle(title, fontsize=12, fontweight='bold'); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 训练过程
# ═══════════════════════════════════════════════════════════════

def plot_training_history(history, title="训练过程"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "loss" in history and len(history["loss"]) > 1:
        losses = history["loss"]; epochs = range(1, len(losses)+1)
        axes[0].plot(epochs, losses, 'b-', linewidth=2, label='训练损失')
        axes[0].set_xlabel("迭代次数"); axes[0].set_ylabel("损失值"); axes[0].set_title("损失变化曲线"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        min_idx = np.argmin(losses)
        axes[0].annotate(f"最小损失: {losses[min_idx]:.4f}", xy=(min_idx+1, losses[min_idx]), xytext=(min_idx+5, losses[min_idx]*1.1), arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    else:
        axes[0].text(0.5, 0.5, "无训练损失数据", ha='center', va='center', transform=axes[0].transAxes); axes[0].set_title("损失变化曲线")
    if "weights" in history and len(history["weights"]) > 1:
        weights = np.array(history["weights"])
        for i in range(min(weights.shape[1], 5)):
            axes[1].plot(range(len(weights)), weights[:, i], '-', label=f"w{i}", linewidth=1.5)
        axes[1].set_xlabel("迭代次数"); axes[1].set_ylabel("权重值"); axes[1].set_title("权重变化曲线"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "无权重变化数据", ha='center', va='center', transform=axes[1].transAxes); axes[1].set_title("权重变化曲线")
    plt.suptitle(title, fontsize=12, fontweight='bold'); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 分类/回归结果
# ═══════════════════════════════════════════════════════════════

def plot_decision_boundary(model, X, y, title="决策边界", resolution=0.02):
    pca = PCA(n_components=2); X_2d = pca.fit_transform(X)
    try:
        from sklearn.base import clone; model_2d = clone(model); model_2d.fit(X_2d, y)
    except Exception:
        model_2d = model
    x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
    y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF','#FFFFAA','#FFAAFF','#AAFFFF','#FFDAB9','#D8BFD8','#B0E0E6','#98FB98'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF','#FFD700','#FF00FF','#00FFFF','#FF6347','#9370DB','#4682B4','#3CB371'])
    n = len(np.unique(Z))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light[:n])
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
    for i, cls in enumerate(np.unique(y)):
        mask = y == cls; ax.scatter(X_2d[mask,0], X_2d[mask,1], c=[cmap_bold(i%10)], label=f'类别 {cls}', edgecolors='k', s=20, linewidths=0.3)
    ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2"); ax.set_title(title); ax.legend(fontsize=8); plt.tight_layout()
    return fig


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


def plot_regression_results(y_true, y_pred, title="回归结果"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn,mx],[mn,mx],'r--',lw=2,label='理想预测线'); axes[0].set_xlabel("真实值"); axes[0].set_ylabel("预测值"); axes[0].set_title("预测值 vs 真实值"); axes[0].legend()
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='零残差线'); axes[1].set_xlabel("残差"); axes[1].set_ylabel("频数"); axes[1].set_title("残差分布"); axes[1].legend()
    plt.suptitle(title, fontsize=12, fontweight='bold'); plt.tight_layout()
    return fig


def plot_gradient_descent_2d(X, y, learning_rate=0.01, n_iterations=50):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    Xn = (X-X.mean())/(X.std()+1e-8); yn = (y-y.mean())/(y.std()+1e-8)
    wr = np.linspace(-3,3,50); br = np.linspace(-3,3,50); W,B = np.meshgrid(wr,br)
    L = np.zeros_like(W)
    for i in range(len(wr)):
        for j in range(len(br)):
            L[j,i] = np.mean((W[j,i]*Xn+B[j,i]-yn)**2)
    im = axes[0].contourf(W,B,L,levels=20,cmap='RdYlBu_r'); fig.colorbar(im,ax=axes[0])
    axes[0].set_xlabel("权重 w"); axes[0].set_ylabel("偏置 b"); axes[0].set_title("损失曲面 + 梯度下降路径")
    w,b = 0.0,0.0; pw,pb = [w],[b]; n = len(yn)
    for _ in range(n_iterations):
        p = w*Xn+b; e = p-yn; w -= learning_rate*(2/n)*np.sum(e*Xn); b -= learning_rate*(2/n)*np.sum(e)
        pw.append(w); pb.append(b)
    axes[0].plot(pw,pb,'k.-',markersize=4,linewidth=1.5); axes[0].plot(pw[0],pb[0],'go',ms=10,label='起点'); axes[0].plot(pw[-1],pb[-1],'r*',ms=15,label='终点'); axes[0].legend()
    colors = plt.cm.viridis(np.linspace(0,1,n_iterations))
    axes[1].scatter(Xn,yn,c='gray',alpha=0.3,s=20,label='数据点')
    for i in range(0,n_iterations,max(1,n_iterations//8)):
        axes[1].plot(Xn,pw[i]*Xn+pb[i],color=colors[i],alpha=0.6,linewidth=1,label=f'第{i}次')
    axes[1].plot(Xn,pw[-1]*Xn+pb[-1],'r-',linewidth=2.5,label='最终拟合')
    axes[1].set_xlabel("X"); axes[1].set_ylabel("y"); axes[1].set_title("拟合过程"); axes[1].legend(fontsize=7,loc='upper left'); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 聚类可视化（v3.0 新增）
# ═══════════════════════════════════════════════════════════════

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

def fig_to_image(fig, max_pixels=600*450):
    """matplotlib Figure → RGB numpy 数组（保持宽高比，控制传输体积）

    Args:
        fig: matplotlib Figure 对象
        max_pixels: 最大像素数（宽×高），超过则等比缩小
    """
    from PIL import Image as _Image
    import io as _io
    fig.canvas.draw()
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi)
    plt.close(fig)
    buf.seek(0)
    img = _Image.open(buf).convert('RGB')
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), _Image.LANCZOS)
    return np.asarray(img)


def plot_learning_curve(train_sizes, train_scores_mean, train_scores_std,
                        val_scores_mean, val_scores_std, title="学习曲线"):
    """绘制学习曲线 - 诊断过拟合/欠拟合

    Args:
        train_sizes: 训练样本数量
        train_scores_mean: 训练集得分均值
        train_scores_std: 训练集得分标准差
        val_scores_mean: 验证集得分均值
        val_scores_std: 验证集得分标准差
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制训练集得分带
    ax.fill_between(train_sizes, 
                    np.array(train_scores_mean) - np.array(train_scores_std),
                    np.array(train_scores_mean) + np.array(train_scores_std),
                    alpha=0.15, color="#2563eb")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="#2563eb", 
            linewidth=2, markersize=6, label="训练集得分")

    # 绘制验证集得分带
    ax.fill_between(train_sizes,
                    np.array(val_scores_mean) - np.array(val_scores_std),
                    np.array(val_scores_mean) + np.array(val_scores_std),
                    alpha=0.15, color="#dc2626")
    ax.plot(train_sizes, val_scores_mean, 'o-', color="#dc2626",
            linewidth=2, markersize=6, label="验证集得分")

    # 添加差距区域标注
    gap = np.array(train_scores_mean) - np.array(val_scores_mean)
    if np.mean(gap) > 0.1:
        ax.annotate("过拟合区域\n(高方差)", 
                    xy=(train_sizes[-1], val_scores_mean[-1]),
                    xytext=(train_sizes[-1]*0.7, val_scores_mean[-1]-0.1),
                    fontsize=10, color="#dc2626",
                    arrowprops=dict(arrowstyle='->', color='#dc2626'))
    elif np.mean(train_scores_mean) < 0.7:
        ax.annotate("欠拟合区域\n(高偏差)",
                    xy=(train_sizes[len(train_sizes)//2], train_scores_mean[len(train_sizes)//2]),
                    xytext=(train_sizes[len(train_sizes)//2]*0.6, 0.5),
                    fontsize=10, color="#f59e0b",
                    arrowprops=dict(arrowstyle='->', color='#f59e0b'))

    ax.set_xlabel("训练样本数量", fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 添加诊断说明
    diagnosis = _diagnose_fit(gap, train_scores_mean, val_scores_mean)
    ax.text(0.02, 0.02, diagnosis, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='#f0f9ff', alpha=0.8))

    plt.tight_layout()
    return fig


def _diagnose_fit(gap, train_scores, val_scores):
    """诊断模型是过拟合还是欠拟合"""
    avg_gap = np.mean(gap)
    avg_val = np.mean(val_scores)
    avg_train = np.mean(train_scores)

    if avg_gap > 0.15 and avg_train > 0.85:
        return "📊 诊断: 过拟合 (高方差)\n   → 增加数据量 / 增加正则化 / 简化模型"
    elif avg_gap > 0.1 and avg_train > 0.8:
        return "📊 诊断: 轻微过拟合\n   → 可考虑增加正则化"
    elif avg_val < 0.6 and avg_train < 0.7:
        return "📊 诊断: 欠拟合 (高偏差)\n   → 增加模型复杂度 / 增加特征"
    elif avg_gap < 0.05 and avg_val > 0.8:
        return "📊 诊断: 拟合良好 ✓\n   → 模型泛化能力强"
    else:
        return "📊 诊断: 正常范围内\n   → 可继续调优"


def plot_cross_validation(scores, model_name="模型"):
    """绘制交叉验证得分柱状图

    Args:
        scores: 各折得分列表
        model_name: 模型名称
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    folds = range(1, len(scores) + 1)
    colors = ['#2563eb' if s >= np.mean(scores) else '#f59e0b' for s in scores]

    bars = ax.bar(folds, scores, color=colors, edgecolor='white', linewidth=1.5)

    # 添加数值标签
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加均值线
    mean_score = np.mean(scores)
    ax.axhline(y=mean_score, color='#dc2626', linestyle='--', linewidth=2,
               label=f'均值: {mean_score:.3f} ± {np.std(scores):.3f}')

    ax.set_xlabel("折数 (Fold)", fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.set_title(f"{model_name} - {len(scores)}折交叉验证", fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加稳定性评估
    cv_std = np.std(scores)
    if cv_std < 0.02:
        stability = "稳定性: 优秀 ✓"
    elif cv_std < 0.05:
        stability = "稳定性: 良好"
    else:
        stability = "稳定性: 一般 (考虑增加数据量)"

    ax.text(0.98, 0.02, stability, transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom', 
            bbox=dict(boxstyle='round', facecolor='#dcfce7', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_validation_curve(param_range, train_scores_mean, train_scores_std,
                          val_scores_mean, val_scores_std, param_name="参数",
                          title="验证曲线"):
    """绘制验证曲线 - 分析超参数影响

    Args:
        param_range: 参数取值范围
        train_scores_mean: 训练集得分均值
        train_scores_std: 训练集得分标准差
        val_scores_mean: 验证集得分均值
        val_scores_std: 验证集得分标准差
        param_name: 参数名称
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(param_range,
                    np.array(train_scores_mean) - np.array(train_scores_std),
                    np.array(train_scores_mean) + np.array(train_scores_std),
                    alpha=0.15, color="#2563eb")
    ax.plot(param_range, train_scores_mean, 'o-', color="#2563eb",
            linewidth=2, markersize=6, label="训练集得分")

    ax.fill_between(param_range,
                    np.array(val_scores_mean) - np.array(val_scores_std),
                    np.array(val_scores_mean) + np.array(val_scores_std),
                    alpha=0.15, color="#dc2626")
    ax.plot(param_range, val_scores_mean, 'o-', color="#dc2626",
            linewidth=2, markersize=6, label="验证集得分")

    # 标注最佳参数
    best_idx = np.argmax(val_scores_mean)
    best_param = param_range[best_idx]
    best_score = val_scores_mean[best_idx]
    ax.axvline(x=best_param, color='#16a34a', linestyle=':', linewidth=2)
    ax.annotate(f'最佳: {param_name}={best_param:.3f}\n得分={best_score:.3f}',
                xy=(best_param, best_score),
                xytext=(best_param + (param_range[-1] - param_range[0])*0.1, best_score - 0.1),
                fontsize=10, color='#16a34a',
                arrowprops=dict(arrowstyle='->', color='#16a34a'))

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_model_comparison(model_names, train_scores, test_scores, cv_scores=None):
    """绘制多模型对比图

    Args:
        model_names: 模型名称列表
        train_scores: 训练集得分列表
        test_scores: 测试集得分列表
        cv_scores: 交叉验证得分列表（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_names))
    width = 0.25

    if cv_scores is not None:
        bars1 = ax.bar(x - width, train_scores, width, label='训练集', color='#93c5fd', edgecolor='white')
        bars2 = ax.bar(x, test_scores, width, label='测试集', color='#2563eb', edgecolor='white')
        bars3 = ax.bar(x + width, cv_scores, width, label='交叉验证', color='#1e40af', edgecolor='white')
    else:
        bars1 = ax.bar(x - width/2, train_scores, width, label='训练集', color='#93c5fd', edgecolor='white')
        bars2 = ax.bar(x + width/2, test_scores, width, label='测试集', color='#2563eb', edgecolor='white')

    # 添加数值标签
    for bars in [bars1, bars2] + ([bars3] if cv_scores is not None else []):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("模型", fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.set_title("模型性能对比", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注最佳模型
    if cv_scores is not None:
        best_idx = np.argmax(cv_scores)
        ax.annotate(f'★ 最佳模型',
                    xy=(x[best_idx] + width, cv_scores[best_idx]),
                    xytext=(x[best_idx] + width + 0.3, cv_scores[best_idx] + 0.05),
                    fontsize=11, fontweight='bold', color='#16a34a',
                    arrowprops=dict(arrowstyle='->', color='#16a34a'))
    else:
        best_idx = np.argmax(test_scores)
        ax.annotate(f'★ 最佳模型',
                    xy=(x[best_idx] + width/2, test_scores[best_idx]),
                    xytext=(x[best_idx] + 0.3, test_scores[best_idx] + 0.05),
                    fontsize=11, fontweight='bold', color='#16a34a',
                    arrowprops=dict(arrowstyle='->', color='#16a34a'))

    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════
# 回归扩展可视化（v3.1 新增）
# ═══════════════════════════════════════════════════════════════

def plot_regularization_comparison(X_train, y_train, X_test, y_test,
                                   alphas=None, title="正则化强度对比"):
    """对比不同正则化强度下的系数变化（Ridge / Lasso 通用）

    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        alphas: 正则化强度列表
        title: 图表标题
    """
    from sklearn.linear_model import Ridge, Lasso, LinearRegression

    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10, 50, 100]

    n_features = min(X_train.shape[1], 10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Ridge 系数路径
    ridge_coefs = []
    for a in alphas:
        m = Ridge(alpha=a).fit(X_train, y_train)
        ridge_coefs.append(m.coef_[:n_features])
    ridge_coefs = np.array(ridge_coefs).T

    for i in range(n_features):
        axes[0].plot(alphas, ridge_coefs[i], 'o-', markersize=4,
                     linewidth=1.5, label=f"w{i}")
    axes[0].set_xscale('log')
    axes[0].set_xlabel("alpha (L2 正则化强度)")
    axes[0].set_ylabel("系数值")
    axes[0].set_title("Ridge 回归系数路径")
    axes[0].legend(fontsize=7, ncol=2, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linewidth=0.5)

    # 2. Lasso 系数路径
    lasso_coefs = []
    for a in alphas:
        m = Lasso(alpha=a, max_iter=5000).fit(X_train, y_train)
        lasso_coefs.append(m.coef_[:n_features])
    lasso_coefs = np.array(lasso_coefs).T

    for i in range(n_features):
        axes[1].plot(alphas, lasso_coefs[i], 'o-', markersize=4,
                     linewidth=1.5, label=f"w{i}")
    axes[1].set_xscale('log')
    axes[1].set_xlabel("alpha (L1 正则化强度)")
    axes[1].set_ylabel("系数值")
    axes[1].set_title("Lasso 回归系数路径")
    axes[1].legend(fontsize=7, ncol=2, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linewidth=0.5)

    # 3. R² 分数对比
    ridge_scores = [Ridge(alpha=a).fit(X_train, y_train).score(X_test, y_test) for a in alphas]
    lasso_scores = [Lasso(alpha=a, max_iter=5000).fit(X_train, y_train).score(X_test, y_test) for a in alphas]

    axes[2].semilogx(alphas, ridge_scores, 'bo-', linewidth=2, markersize=5, label='Ridge')
    axes[2].semilogx(alphas, lasso_scores, 'rs-', linewidth=2, markersize=5, label='Lasso')
    lr_score = LinearRegression().fit(X_train, y_train).score(X_test, y_test)
    axes[2].axhline(lr_score, color='gray', linestyle='--', label=f'线性回归 (R²={lr_score:.3f})')
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("R² 分数")
    axes[2].set_title("测试集 R² 对比")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_polynomial_comparison(X_train, y_train, X_test, y_test,
                               max_degree=5, title="多项式回归阶数对比"):
    """对比不同阶数的多项式回归效果

    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        max_degree: 最大多项式阶数
        title: 图表标题
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

    # 只使用第一个特征进行可视化
    x_train_1d = X_train[:, 0:1]
    x_test_1d = X_test[:, 0:1]

    degrees = list(range(1, max_degree + 1))
    train_scores = []
    test_scores = []

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    x_sort = np.sort(x_train_1d, axis=0)
    x_curve = np.linspace(x_sort.min(), x_sort.max(), 200).reshape(-1, 1)

    for i, d in enumerate(degrees):
        ax = axes[i]
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('lr', LinearRegression())
        ])
        pipe.fit(x_train_1d, y_train)
        y_curve = pipe.predict(x_curve)

        train_r2 = pipe.score(x_train_1d, y_train)
        test_r2 = pipe.score(x_test_1d, y_test)
        train_scores.append(train_r2)
        test_scores.append(test_r2)

        ax.scatter(x_train_1d, y_train, alpha=0.4, s=20, c='steelblue', edgecolors='k', linewidths=0.3)
        ax.plot(x_curve, y_curve, 'r-', linewidth=2)
        ax.set_title(f"degree={d} (训练R²={train_r2:.3f}, 测试R²={test_r2:.3f})", fontsize=10)
        ax.set_xlabel("特征1")
        ax.set_ylabel("目标值")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(len(degrees), len(axes)):
        axes[j].set_visible(False)

    # R² 汇总
    summary_ax = fig.add_axes([0.08, 0.02, 0.84, 0.12])
    summary_ax.bar([f"d={d}" for d in degrees], test_scores, color='steelblue',
                   alpha=0.7, edgecolor='k', label='测试集 R²')
    summary_ax.plot([f"d={d}" for d in degrees], train_scores, 'rs-',
                    markersize=8, label='训练集 R²')
    summary_ax.set_ylabel("R² 分数")
    summary_ax.legend(loc='upper right')
    summary_ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    return fig


def plot_regression_comparison(X_train, y_train, X_test, y_test,
                               algorithm_results, title="回归模型对比"):
    """对比多个回归模型的预测效果

    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        algorithm_results: list of {"name": str, "model": object}
        title: 图表标题
    """
    n = len(algorithm_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, result in enumerate(algorithm_results):
        ax = axes[i]
        y_pred = result["model"].predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='理想预测线')

        # 计算指标
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax.set_title(f"{result['name']}\nR²={r2:.4f}, RMSE={rmse:.4f}", fontsize=11)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_search_results(search_result, param_name, title="超参数搜索结果"):
    """可视化超参数搜索结果（热力图/折线图）

    Args:
        search_result: GridSearchCV 或 RandomizedSearchCV 的拟合结果
        param_name: 要可视化的参数名
        title: 图表标题
    """
    results = search_result.cv_results_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 参数 vs 得分
    try:
        param_values = [r[param_name] for r in results['params']]
        # 如果参数值是数值类型
        numeric = all(isinstance(v, (int, float)) for v in param_values)
        if numeric:
            axes[0].plot(param_values, results['mean_test_score'], 'bo-', markersize=6)
            axes[0].fill_between(param_values,
                                results['mean_test_score'] - results['std_test_score'],
                                results['mean_test_score'] + results['std_test_score'],
                                alpha=0.15, color='blue')
            axes[0].set_xlabel(param_name)
        else:
            # 分类参数用柱状图
            unique_vals = sorted(set(str(v) for v in param_values))
            means = []
            for v in unique_vals:
                idx = [i for i, p in enumerate(param_values) if str(p) == v]
                means.append(np.mean([results['mean_test_score'][i] for i in idx]))
            axes[0].bar(range(len(unique_vals)), means, color='steelblue', edgecolor='k')
            axes[0].set_xticks(range(len(unique_vals)))
            axes[0].set_xticklabels(unique_vals, rotation=30, ha='right')
            axes[0].set_xlabel(param_name)

        axes[0].set_ylabel('平均交叉验证得分')
        axes[0].set_title(f'{param_name} 对模型性能的影响')
        axes[0].grid(True, alpha=0.3)

        # 标注最优参数
        best_idx = search_result.best_index_
        best_score = results['mean_test_score'][best_idx]
        best_param = results['params'][best_idx].get(param_name, 'N/A')
        axes[0].axhline(best_score, color='red', linestyle='--', alpha=0.5,
                        label=f'最优: {best_param} (score={best_score:.4f})')
        axes[0].legend(fontsize=9)

    except Exception:
        axes[0].text(0.5, 0.5, "无法生成参数-得分图", ha='center', va='center',
                     transform=axes[0].transAxes)
        axes[0].set_title(f'{param_name} 搜索结果')

    # 2. 搜索耗时 vs 排名
    axes[1].scatter(results['rank_test_score'],
                     results['mean_fit_time'],
                     c=results['mean_test_score'],
                     cmap='RdYlGn', s=50, edgecolors='k', linewidths=0.5)
    axes[1].set_xlabel('排名')
    axes[1].set_ylabel('平均拟合时间 (s)')
    axes[1].set_title('排名 vs 耗时 (颜色=得分)')
    fig.colorbar(axes[1].collections[0], ax=axes[1], label='CV 得分')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig




# ═══════════════════════════════════════════════════════════════
# 特征工程可视化
# ═══════════════════════════════════════════════════════════════

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
    plt.tight_layout()
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
    plt.tight_layout()
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
    plt.tight_layout()
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
    plt.tight_layout()
    return fig
