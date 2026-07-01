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



def plot_multi_metric_comparison(model_names, accuracies, precisions, recalls, f1_scores):
    """绘制多指标详细对比图 (Acc/Precision/Recall/F1)

    Args:
        model_names: 模型名称列表
        accuracies: 准确率列表
        precisions: 精确率列表
        recalls: 召回率列表
        f1_scores: F1分数列表
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.18

    bars1 = ax.bar(x - 1.5 * width, accuracies, width, label='Accuracy',
                   color='#93c5fd', edgecolor='white')
    bars2 = ax.bar(x - 0.5 * width, precisions, width, label='Precision',
                   color='#34d399', edgecolor='white')
    bars3 = ax.bar(x + 0.5 * width, recalls, width, label='Recall',
                   color='#fbbf24', edgecolor='white')
    bars4 = ax.bar(x + 1.5 * width, f1_scores, width, label='F1-Score',
                   color='#f87171', edgecolor='white')

    # 添加数值标签
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel("模型", fontsize=12)
    ax.set_ylabel("得分", fontsize=12)
    ax.set_title("多指标详细对比 (Acc/Precision/Recall/F1)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(loc='lower right', fontsize=10, ncol=4)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    # 用灰色虚线标记 0.5/0.8 参考线
    for threshold in [0.5, 0.8]:
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

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
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig




# ═══════════════════════════════════════════════════════════════
# 特征工程可视化
# ═══════════════════════════════════════════════════════════════
