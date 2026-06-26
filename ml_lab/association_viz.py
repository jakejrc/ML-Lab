# -*- coding: utf-8 -*-
"""
ML-Lab 关联规则可视化模块 (v3.5 新增)

提供关联规则挖掘结果的可视化图表:
- Top-K 频繁项集支持度柱状图
- 规则置信度/提升度散点图
- 项集长度分布饼图
- 规则热力图 (support vs confidence)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from collections import defaultdict


def _setup_chinese_font():
    """配置中文字体（跨平台：Windows / Linux Docker）
    优先使用项目自带 SimHei 字体（fonts/SimHei.ttf），
    其次尝试 Docker 预提取的 Noto Sans CJK SC OTF，
    最后回退到 fontManager 中已安装的中文字体。
    """
    import inspect
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    _simhei_path = os.path.join(_project_root, 'fonts', 'SimHei.ttf')
    _font_name = None
    # 1. 项目自带 SimHei 字体
    if os.path.isfile(_simhei_path):
        try:
            fm.fontManager.addfont(_simhei_path)
            _font_name = 'SimHei'
        except Exception:
            pass
    # 2. Docker/Linux 环境下的 Noto Sans CJK SC OTF
    if not _font_name:
        _cjk_otf = '/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf'
        if os.path.isfile(_cjk_otf):
            try:
                fm.fontManager.addfont(_cjk_otf)
                _font_name = 'Noto Sans CJK SC'
            except Exception:
                pass
    # 3. 回退：通过 fontManager 搜索已安装的中文字体
    if not _font_name:
        for _candidate in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                           'SimHei', 'Microsoft YaHei']:
            _matches = [f.name for f in fm.fontManager.ttflist if f.name == _candidate]
            if _matches:
                _font_name = _candidate
                break
    if _font_name:
        plt.rcParams['font.sans-serif'] = [_font_name] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def plot_top_frequent_items(model, top_k=15, title="Top-K 频繁项集"):
    """
    绘制 Top-K 频繁项集支持度柱状图。

    Args:
        model: AprioriModel 或 FPGrowthModel 实例
        top_k: int, 显示前 K 个
        title: str, 图表标题

    Returns:
        matplotlib.figure.Figure
    """
    _setup_chinese_font()

    freq = sorted(model.frequent_items, key=lambda x: -x[1])[:top_k]
    if not freq:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "未找到频繁项集\n请调整 min_support 参数",
                ha='center', va='center', fontsize=14, color='#94a3b8')
        ax.set_axis_off()
        return fig

    # 项集名称 (缩短显示)
    labels = []
    supports = []
    for itemset, sup in freq:
        items = sorted(itemset)
        # 缩短项名
        short_items = []
        for item in items:
            # 取 = 号前的特征名和 = 号后的值
            parts = item.split("=")
            if len(parts) == 2:
                short_items.append(f"{parts[0][-8:]}={parts[1]}")
            else:
                short_items.append(item[-10:])
        label = " + ".join(short_items)
        if len(label) > 30:
            label = label[:27] + "..."
        labels.append(label)
        supports.append(sup * 100)

    fig, ax = plt.subplots(figsize=(12, max(5, top_k * 0.4)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(labels)))[::-1]
    bars = ax.barh(range(len(labels)), supports, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("支持度 (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, supports):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va='center', fontsize=8, color='#475569')

    plt.tight_layout()
    return fig


def plot_rules_scatter(model, top_k=50, title="关联规则: 置信度 vs 提升度"):
    """
    绘制规则散点图 (置信度 x 轴, 提升度 y 轴, 气泡大小=支持度)。

    Args:
        model: AprioriModel 或 FPGrowthModel 实例
        top_k: int, 显示前 K 条规则
        title: str

    Returns:
        matplotlib.figure.Figure
    """
    _setup_chinese_font()

    rules = model.rules[:top_k]
    if not rules:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "未找到满足条件的关联规则\n请调整 min_confidence 或 min_lift",
                ha='center', va='center', fontsize=14, color='#94a3b8')
        ax.set_axis_off()
        return fig

    confidences = [r["confidence"] * 100 for r in rules]
    lifts = [r["lift"] for r in rules]
    supports = [r["support"] * 100 for r in rules]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(confidences, lifts, s=[max(20, s * 30) for s in supports],
                         c=supports, cmap='YlOrRd', alpha=0.7, edgecolors='white', linewidth=0.5)

    ax.set_xlabel("置信度 (%)", fontsize=11)
    ax.set_ylabel("提升度 (Lift)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.axhline(y=1.0, color='#e2e8f0', linestyle='--', linewidth=1, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("支持度 (%)", fontsize=10)

    # 标注 top-3 规则
    top3 = sorted(rules, key=lambda r: -r["lift"])[:3]
    for r in top3:
        ant = " + ".join(sorted(r["antecedent"]))[:20]
        cons = " + ".join(sorted(r["consequent"]))[:15]
        ax.annotate(f"{ant} -> {cons}",
                    xy=(r["confidence"] * 100, r["lift"]),
                    fontsize=7, color='#334155',
                    xytext=(5, 5), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=0.5))

    plt.tight_layout()
    return fig


def plot_item_length_distribution(model, title="频繁项集长度分布"):
    """
    绘制频繁项集长度分布饼图。

    Args:
        model: AprioriModel 或 FPGrowthModel 实例
        title: str

    Returns:
        matplotlib.figure.Figure
    """
    _setup_chinese_font()

    length_dist = model.history.get("item_length_dist", {})
    if not length_dist:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "无频繁项集数据", ha='center', va='center',
                fontsize=14, color='#94a3b8')
        ax.set_axis_off()
        return fig

    lengths = sorted(length_dist.keys())
    counts = [length_dist[l] for l in lengths]
    labels = [f"{l}-项集" for l in lengths]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color('#334155')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    total = sum(counts)
    legend_labels = [f"{lab}: {cnt} 项 ({cnt/total*100:.1f}%)" for lab, cnt in zip(labels, counts)]
    ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(labels), 3), fontsize=9)

    plt.tight_layout()
    return fig


def plot_rules_heatmap(model, top_k=20, title="关联规则热力图"):
    """
    绘制关联规则热力图 (支持度 vs 置信度)。

    Args:
        model: AprioriModel 或 FPGrowthModel 实例
        top_k: int
        title: str

    Returns:
        matplotlib.figure.Figure
    """
    _setup_chinese_font()

    rules = model.rules[:top_k]
    if not rules:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无关联规则数据", ha='center', va='center',
                fontsize=14, color='#94a3b8')
        ax.set_axis_off()
        return fig

    n_rules = len(rules)
    confidences = np.array([r["confidence"] for r in rules]).reshape(-1, 1)
    lifts = np.array([r["lift"] for r in rules]).reshape(-1, 1)
    supports = np.array([r["support"] for r in rules])

    # 构建矩阵: 行=规则, 列=[置信度, 提升度, 支持度]
    matrix = np.hstack([confidences, lifts, supports.reshape(-1, 1)])

    # 规则标签
    labels = []
    for r in rules:
        ant = " + ".join(sorted(r["antecedent"]))[:18]
        cons = " + ".join(sorted(r["consequent"]))[:14]
        labels.append(f"{ant}\n-> {cons}")

    fig, ax = plt.subplots(figsize=(8, max(6, n_rules * 0.35)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(3))
    ax.set_xticklabels(["置信度", "提升度", "支持度"], fontsize=10)
    ax.set_yticks(range(n_rules))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # 添加数值标注
    for i in range(n_rules):
        for j in range(3):
            val = matrix[i, j]
            text_color = "white" if val > matrix.max() * 0.6 else "#334155"
            ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                    fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.6, label="数值")
    plt.tight_layout()
    return fig
