# -*- coding: utf-8 -*-
"""
ML-Lab 可视化模块
提供数据分布、决策边界、训练过程、模型评估等交互式可视化功能。
所有图表使用 matplotlib 生成，适配 Gradio 界面输出。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize


def plot_data_distribution(X, y, title="数据分布", feature_names=None):
    """绘制数据分布图（PCA降维到2D）

    Args:
        X (np.ndarray): 特征数据
        y (np.ndarray): 标签
        title (str): 图表标题
        feature_names (list): 特征名称

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA 2D
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        xlabel, ylabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", \
                         f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
    else:
        X_2d = X[:, :2]
        fn = feature_names or ["特征1", "特征2"]
        xlabel, ylabel = fn[0], fn[1]

    classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    for cls, color in zip(classes, colors):
        mask = y == cls
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color],
                        label=f"类别{cls}", alpha=0.7, s=40, edgecolors='k',
                        linewidths=0.5)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"{title} (PCA降维)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 类别分布柱状图
    unique, counts = np.unique(y, return_counts=True)
    axes[1].bar([f"类别{u}" for u in unique], counts, color=colors, edgecolor='k')
    axes[1].set_xlabel("类别")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("各类别样本数量")
    for i, v in enumerate(counts):
        axes[1].text(i, v + 0.5, str(v), ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_preprocessing_comparison(X_raw, X_processed, method_name,
                                   feature_idx=0):
    """绘制预处理前后对比图

    Args:
        X_raw (np.ndarray): 原始数据
        X_processed (np.ndarray): 处理后数据
        method_name (str): 处理方法名称
        feature_idx (int): 展示的特征索引

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 原始数据分布
    axes[0].hist(X_raw[:, feature_idx], bins=30, color='steelblue',
                 edgecolor='k', alpha=0.7)
    axes[0].set_title(f"原始数据 (特征{feature_idx})")
    axes[0].set_xlabel("值")
    axes[0].set_ylabel("频数")
    axes[0].axvline(np.nanmean(X_raw[:, feature_idx]), color='red',
                    linestyle='--', label=f'均值={np.nanmean(X_raw[:, feature_idx]):.2f}')
    axes[0].legend()

    # 处理后数据分布
    axes[1].hist(X_processed[:, feature_idx], bins=30, color='coral',
                 edgecolor='k', alpha=0.7)
    axes[1].set_title(f"{method_name}后 (特征{feature_idx})")
    axes[1].set_xlabel("值")
    axes[1].set_ylabel("频数")
    axes[1].axvline(np.nanmean(X_processed[:, feature_idx]), color='blue',
                    linestyle='--',
                    label=f'均值={np.nanmean(X_processed[:, feature_idx]):.2f}')
    axes[1].legend()

    # 对比箱线图
    data_to_plot = [
        X_raw[:, feature_idx:feature_idx + 1].flatten(),
        X_processed[:, feature_idx:feature_idx + 1].flatten()
    ]
    bp = axes[2].boxplot(data_to_plot, labels=["原始", method_name],
                         patch_artist=True)
    bp["boxes"][0].set_facecolor('steelblue')
    bp["boxes"][1].set_facecolor('coral')
    axes[2].set_title("分布对比 (箱线图)")

    plt.tight_layout()
    return fig


def plot_decision_boundary(model, X, y, title="决策边界",
                            feature_indices=(0, 1)):
    """绘制决策边界可视化（2D特征空间）

    Args:
        model: 已训练的分类模型（需有 predict 方法）
        X (np.ndarray): 特征数据
        y (np.ndarray): 标签
        title (str): 图表标题
        feature_indices (tuple): 用于绘制的两个特征索引

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 如果特征超过2维，先用PCA降到2维
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        # 训练一个对应的模型
        from sklearn.base import clone
        model_2d = clone(model)
        model_2d.fit(X_2d, y)
        xlabel, ylabel = "PC1", "PC2"
    else:
        X_2d = X[:, feature_indices[0]:feature_indices[1] + 1]
        model_2d = model
        xlabel, ylabel = f"特征{feature_indices[0]}", f"特征{feature_indices[1]}"

    # 创建网格
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    # 预测
    try:
        Z = model_2d.predict(mesh)
    except Exception:
        return fig

    Z = Z.reshape(xx.shape)

    # 绘制决策区域
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes <= 10:
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_classes, 2)))
        contour = ax.contourf(xx, yy, Z, alpha=0.3, levels=n_classes - 1,
                              colors=colors[:n_classes])
    else:
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab20')

    # 绘制数据点
    for cls in classes:
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"类别{cls}",
                   edgecolors='k', s=40, linewidths=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_history(history, title="训练过程"):
    """绘制训练过程曲线

    Args:
        history (dict): 包含 'loss' 键的字典
        title (str): 图表标题

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    if "loss" in history and len(history["loss"]) > 1:
        losses = history["loss"]
        epochs = range(1, len(losses) + 1)
        axes[0].plot(epochs, losses, 'b-', linewidth=2, label='训练损失')
        axes[0].set_xlabel("迭代次数")
        axes[0].set_ylabel("损失值")
        axes[0].set_title("损失变化曲线")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 标注最小损失
        min_idx = np.argmin(losses)
        axes[0].annotate(f"最小损失: {losses[min_idx]:.4f}",
                         xy=(min_idx + 1, losses[min_idx]),
                         xytext=(min_idx + 5, losses[min_idx] * 1.1),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=9, color='red')
    else:
        axes[0].text(0.5, 0.5, "无训练损失数据",
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("损失变化曲线")

    # 权重变化（如果有）
    if "weights" in history and len(history["weights"]) > 1:
        weights = np.array(history["weights"])
        for i in range(min(weights.shape[1], 5)):
            axes[1].plot(range(len(weights)), weights[:, i], '-',
                         label=f"w{i}", linewidth=1.5)
        axes[1].set_xlabel("迭代次数")
        axes[1].set_ylabel("权重值")
        axes[1].set_title("权重变化曲线")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "无权重变化数据",
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("权重变化曲线")

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                           title="混淆矩阵"):
    """绘制混淆矩阵热力图

    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        class_names (list): 类别名称
        title (str): 图表标题

    Returns:
        matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"类别{i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="真实标签", xlabel="预测标签",
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # 在格子中标注数字
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, class_names=None, title="ROC曲线"):
    """绘制ROC曲线

    Args:
        y_true (np.ndarray): 真实标签
        y_proba (np.ndarray): 预测概率 (n_samples, n_classes)
        class_names (list): 类别名称
        title (str): 图表标题

    Returns:
        matplotlib.figure.Figure
    """
    n_classes = y_proba.shape[1]

    if class_names is None:
        class_names = [f"类别{i}" for i in range(n_classes)]

    # 二分类时只画一条曲线
    if n_classes == 2:
        y_true_bin = y_true
        y_score = y_proba[:, 1]
    else:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        y_score = y_proba

    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    else:
        # 多分类：宏平均
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            auc_i = auc(fpr_i, tpr_i)
            ax.plot(fpr_i, tpr_i, lw=1, alpha=0.5,
                    label=f'{class_names[i]} (AUC={auc_i:.2f})')
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, 'k--', lw=2,
                label=f'宏平均 (AUC = {macro_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机基线')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("假正率 (FPR)")
    ax.set_ylabel("真正率 (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names=None, top_n=10,
                             title="特征重要性"):
    """绘制特征重要性条形图

    Args:
        importances (np.ndarray): 特征重要性
        feature_names (list): 特征名称
        top_n (int): 显示前N个特征
        title (str): 图表标题

    Returns:
        matplotlib.figure.Figure
    """
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(len(importances))]

    # 排序
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.5)))
    bars = ax.barh(range(len(top_names)), top_importances, color='steelblue',
                   edgecolor='k')
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("重要性")
    ax.set_title(title)
    ax.invert_yaxis()

    # 标注数值
    for bar, val in zip(bars, top_importances):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    return fig


def plot_regression_results(y_true, y_pred, title="回归结果"):
    """绘制回归预测结果对比图

    Args:
        y_true (np.ndarray): 真实值
        y_pred (np.ndarray): 预测值
        title (str): 图表标题

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 预测值 vs 真实值散点图
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k',
                    linewidths=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
                 label='理想预测线')
    axes[0].set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    axes[0].set_title("预测值 vs 真实值")
    axes[0].legend()

    # 残差分布
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='k',
                 alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='零残差线')
    axes[1].set_xlabel("残差 (真实值 - 预测值)")
    axes[1].set_ylabel("频数")
    axes[1].set_title("残差分布")
    axes[1].legend()

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_gradient_descent_2d(X, y, learning_rate=0.01, n_iterations=50):
    """绘制2D梯度下降可视化动画帧

    Args:
        X (np.ndarray): 一维特征
        y (np.ndarray): 目标值
        learning_rate (float): 学习率
        n_iterations (int): 迭代次数

    Returns:
        matplotlib.figure.Figure: 包含损失曲面和梯度下降路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 标准化
    X_norm = (X - X.mean()) / (X.std() + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)

    # 左图：损失曲面
    w_range = np.linspace(-3, 3, 50)
    b_range = np.linspace(-3, 3, 50)
    W, B = np.meshgrid(w_range, b_range)
    Loss = np.zeros_like(W)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            pred = W[j, i] * X_norm + B[j, i]
            Loss[j, i] = np.mean((pred - y_norm) ** 2)

    im = axes[0].contourf(W, B, Loss, levels=20, cmap='RdYlBu_r')
    fig.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("权重 w")
    axes[0].set_ylabel("偏置 b")
    axes[0].set_title("损失函数曲面 + 梯度下降路径")

    # 梯度下降路径
    w, b = 0.0, 0.0
    path_w, path_b = [w], [b]
    n = len(y_norm)
    for _ in range(n_iterations):
        pred = w * X_norm + b
        error = pred - y_norm
        dw = (2 / n) * np.sum(error * X_norm)
        db = (2 / n) * np.sum(error)
        w -= learning_rate * dw
        b -= learning_rate * db
        path_w.append(w)
        path_b.append(b)

    axes[0].plot(path_w, path_b, 'k.-', markersize=4, linewidth=1.5)
    axes[0].plot(path_w[0], path_b[0], 'go', markersize=10, label='起点')
    axes[0].plot(path_w[-1], path_b[-1], 'r*', markersize=15, label='终点')
    axes[0].legend()

    # 右图：拟合过程
    colors = plt.cm.viridis(np.linspace(0, 1, n_iterations))
    axes[1].scatter(X_norm, y_norm, c='gray', alpha=0.3, s=20, label='数据点')
    for i in range(0, n_iterations, max(1, n_iterations // 8)):
        pred_line = path_w[i] * X_norm + path_b[i]
        axes[1].plot(X_norm, pred_line, color=colors[i], alpha=0.6,
                     linewidth=1, label=f'第{i}次')
    axes[1].plot(X_norm, path_w[-1] * X_norm + path_b[-1], 'r-',
                 linewidth=2.5, label='最终拟合')
    axes[1].set_xlabel("特征 X (标准化)")
    axes[1].set_ylabel("目标 y (标准化)")
    axes[1].set_title("线性拟合过程")
    axes[1].legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    return fig


def fig_to_image(fig):
    """将 matplotlib Figure 转换为 RGB numpy 数组

    Args:
        fig (matplotlib.figure.Figure): 图表对象

    Returns:
        np.ndarray: RGB图像数组
    """
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf).copy()
    plt.close(fig)
    return img[:, :, :3]  # 去掉alpha通道