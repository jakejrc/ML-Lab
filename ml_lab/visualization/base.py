# -*- coding: utf-8 -*-
"""
ML-Lab 可视化模块 v3.0
v3.0 新增：聚类散点图、肘部法则图、层次聚类树状图、DBSCAN eps 分析、PCA 方差解释图、PCA 投影图。
"""

import numpy as np
from functools import lru_cache
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

def _register_cjk_font():
    """在 Linux/Docker 中注册 CJK 中文字体。

    优先级：
    1. 项目自带的 SimHei.ttf（fonts/SimHei.ttf）
    2. Docker 预提取的 NotoSansCJKSC OTF（/usr/share/fonts/opentype/noto/）
    3. 系统安装的 WenQuanYi Micro Hei（树莓派/Debian）
    4. fontManager 中已有的 CJK 字体
    """
    import inspect
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    # 1. 项目自带的 SimHei.ttf
    _simhei_path = os.path.join(_project_root, 'fonts', 'SimHei.ttf')
    if os.path.isfile(_simhei_path):
        try:
            fm.fontManager.addfont(_simhei_path)
            return 'SimHei'
        except Exception:
            pass
    # 2. Docker 预提取的 NotoSansCJKSC OTF
    for _noto in ['/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf',
                  '/usr/share/fonts/opentype/noto/NotoSansCJKSC-Bold.otf']:
        if os.path.isfile(_noto):
            try:
                fm.fontManager.addfont(_noto)
                return 'Noto Sans CJK SC'
            except Exception:
                pass
    # 3. WenQuanYi 文泉驿（树莓派/Debian）
    for _wqy in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                 '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
        if os.path.isfile(_wqy):
            try:
                fm.fontManager.addfont(_wqy)
                return 'WenQuanYi Micro Hei' if 'microhei' in _wqy else 'WenQuanYi Zen Hei'
            except Exception:
                pass
    return None


def _setup_chinese_font():
    """跨平台中文字体配置（兼容 Docker 中 TTC 字体注册不全的问题）"""
    _system = platform.system()
    if _system == 'Windows':
        _fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi']
    elif _system == 'Darwin':
        _fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:
        # Linux / Docker — 注册预提取的 CJK OTF 字体
        _registered = _register_cjk_font()
        if _registered:
            _fonts = [_registered, 'DejaVu Sans']
        else:
            # Fallback: 查找 fontManager 中已有的 CJK 字体
            _fonts = []
            _candidates = [
                'Noto Sans CJK SC', 'Noto Sans CJK', 'Noto Sans SC',
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                'Droid Sans Fallback',
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

    # 智能判断：连续值（回归）vs 离散值（分类）
    classes = np.unique(y)
    n_classes = len(classes)
    is_continuous = False
    try:
        y_float = y.astype(float)
        # 唯一值过多 或 本身就是浮点型 → 视为连续（回归）
        if n_classes > 20 or (np.issubdtype(y.dtype, np.floating) and n_classes > 10):
            is_continuous = True
    except (ValueError, TypeError):
        pass

    if is_continuous:
        # 回归/连续数据：用颜色映射 + colorbar，避免图例过长
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_float, cmap='viridis',
                        alpha=0.7, s=30, edgecolors='k', linewidths=0.3)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('目标值', fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3); plt.tight_layout()
    else:
        # 分类数据：分类图例（类别较多时分多列）
        cmap = plt.cm.Set1(np.linspace(0, 1, max(n_classes, 2)))
        for i, cls in enumerate(classes):
            mask = y == cls
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap[i % len(cmap)]],
                       label=f'类别 {cls}', alpha=0.7, s=30, edgecolors='k', linewidths=0.3)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        # 类别多时图例分多列，避免遮挡
        ncol = 1 if n_classes <= 10 else (2 if n_classes <= 20 else 3)
        ax.legend(fontsize=8, loc='best', ncol=ncol, framealpha=0.9, markerscale=0.8)
        ax.grid(True, alpha=0.3); plt.tight_layout()
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
    plt.suptitle(title, fontsize=12, fontweight='bold'); plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════
# 训练过程
# ═══════════════════════════════════════════════════════════════

def plot_training_history(history, title="训练过程"):
    has_loss = "loss" in history and len(history["loss"]) > 1
    has_weights = "weights" in history and len(history["weights"]) > 1
    if has_loss and has_weights:
        # 有损失 + 有权重：双面板
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # --- 损失曲线 ---
        losses = history["loss"]; epochs = list(range(1, len(losses) + 1))
        axes[0].plot(epochs, losses, 'b-', linewidth=2, label='训练损失')
        axes[0].set_xlabel("迭代次数"); axes[0].set_ylabel("损失值")
        axes[0].set_title("损失变化曲线"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        min_idx = np.argmin(losses)
        # 标注位置在轴内（避免越界）
        y_min, y_max = axes[0].get_ylim()
        annot_y = losses[min_idx] - (y_max - y_min) * 0.08
        if annot_y < y_min:
            annot_y = losses[min_idx] + (y_max - y_min) * 0.08
        axes[0].annotate(f"最小损失: {losses[min_idx]:.4f}",
                         xy=(min_idx + 1, losses[min_idx]),
                         xytext=(min_idx + 1, annot_y),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=9, color='red', ha='center')
        # --- 权重曲线 ---
        weights = np.array(history["weights"])
        for i in range(min(weights.shape[1], 5)):
            axes[1].plot(range(len(weights)), weights[:, i], '-', label=f"w{i}", linewidth=1.5)
        axes[1].set_xlabel("迭代次数"); axes[1].set_ylabel("权重值")
        axes[1].set_title("权重变化曲线"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    elif has_loss:
        # 只有损失：单宽面板
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        losses = history["loss"]; epochs = list(range(1, len(losses) + 1))
        ax.plot(epochs, losses, 'b-', linewidth=2, label='训练损失')
        ax.set_xlabel("迭代次数"); ax.set_ylabel("损失值")
        ax.set_title("损失变化曲线"); ax.legend(); ax.grid(True, alpha=0.3)
        min_idx = np.argmin(losses)
        y_min, y_max = ax.get_ylim()
        annot_y = losses[min_idx] - (y_max - y_min) * 0.08
        if annot_y < y_min:
            annot_y = losses[min_idx] + (y_max - y_min) * 0.08
        ax.annotate(f"最小损失: {losses[min_idx]:.4f}",
                    xy=(min_idx + 1, losses[min_idx]),
                    xytext=(min_idx + 1, annot_y),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red', ha='center')
        # 在右侧添加模型说明（无权重信息）
        ax.text(0.97, 0.95, "注：该模型无可训练权重参数\n故不显示权重变化曲线",
                transform=ax.transAxes, fontsize=8, color='gray',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.text(0.5, 0.5, "无训练过程数据", ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("训练过程")
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════
# 分类/回归结果
# ═══════════════════════════════════════════════════════════════

def plot_decision_boundary(model, X, y, title="决策边界", resolution=0.02):
    # 提取底层 sklearn 模型（兼容 ML-Lab 包装类）
    inner = getattr(model, 'model', model)
    pca = PCA(n_components=2); X_2d = pca.fit_transform(X)
    model_2d = None
    # 尝试 clone + re-fit 底层模型
    try:
        from sklearn.base import clone
        model_2d = clone(inner)
        model_2d.fit(X_2d, y)
    except Exception:
        model_2d = None
    # 如果 clone 失败，用 KNN 在 2D 数据上快速训练用于可视化
    if model_2d is None:
        try:
            from sklearn.neighbors import KNeighborsClassifier
            model_2d = KNeighborsClassifier(n_neighbors=min(15, len(np.unique(y))*3))
            model_2d.fit(X_2d, y)
        except Exception:
            return None
    x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
    y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF','#FFFFAA','#FFAAFF','#AAFFFF','#FFDAB9','#D8BFD8','#B0E0E6','#98FB98'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF','#FFD700','#FF00FF','#00FFFF','#FF6347','#9370DB','#4682B4','#3CB371'])
    n = len(np.unique(Z))
    ax.contourf(xx, yy, Z, alpha=0.3, colors=cmap_light.colors[:n])
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
    for i, cls in enumerate(np.unique(y)):
        mask = y == cls; ax.scatter(X_2d[mask,0], X_2d[mask,1], color=cmap_bold.colors[i%10], label=f'类别 {cls}', edgecolors='k', s=20, linewidths=0.3)
    ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2"); ax.set_title(title); ax.legend(fontsize=8); plt.tight_layout()
    return fig


def fig_to_image(fig, max_pixels=600*450):
    """matplotlib Figure → numpy RGB 数组（Gradio Image 兼容）

    Args:
        fig: matplotlib Figure 对象
        max_pixels: 最大像素数（宽×高），超过则等比缩小
    Returns:
        np.ndarray: RGB 数组，可直接被 Gradio Image 组件渲染
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

