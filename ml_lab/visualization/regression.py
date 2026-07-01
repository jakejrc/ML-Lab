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

def plot_regression_results(y_true, y_pred, title="回归结果"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn,mx],[mn,mx],'r--',lw=2,label='理想预测线'); axes[0].set_xlabel("真实值"); axes[0].set_ylabel("预测值"); axes[0].set_title("预测值 vs 真实值"); axes[0].legend()
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='零残差线'); axes[1].set_xlabel("残差"); axes[1].set_ylabel("频数"); axes[1].set_title("残差分布"); axes[1].legend()
    plt.suptitle(title, fontsize=12, fontweight='bold'); plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

