# -*- coding: utf-8 -*-
"""
ML-Lab FP 树可视化 v4 — 出版级风格重制
采用 Nature/NMI 配色，渐变连线，现代卡片式 Header Table。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


# ── 配色方案 ──
COLORS = {
    'bg':           '#FAFBFE',
    'title':        '#1B2838',
    'subtitle':     '#5A6C7D',
    'tree_line':    '#B0BEC5',
    'tree_line_hi': '#607D8B',
    'null_node':    '#546E7A',
    'null_text':    '#ECEFF1',
    'node_border':  '#263238',
    'node_text':    '#1B2838',
    'count_text':   '#D32F2F',
    'header_bg':    '#FFFFFF',
    'header_title': '#1B2838',
    'header_item':  '#37474F',
    'header_val':   '#D32F2F',
    'header_num':   '#546E7A',
    'bar_bg':       '#ECEFF1',
    'bar_fill':     '#66BB6A',
    'bar_fill2':    '#43A047',
    'sep':          '#CFD8DC',
    'footer':       '#90A4AE',
    'legend_bg':    '#F5F7FA',
    'accent':       '#1E88E5',
    'depth_colors': ['#1565C0', '#1E88E5', '#42A5F5', '#90CAF9',
                     '#BBDEFB', '#E3F2FD'],
}


# ── 节点主色渐变 ──
NODE_CMAP = LinearSegmentedColormap.from_list(
    'fp_node', ['#E8F5E9', '#C8E6C9', '#A5D6A7', '#81C784',
                '#66BB6A', '#4CAF50', '#43A047', '#388E3C', '#2E7D32', '#1B5E20'],
    N=256
)


def _setup_font():
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
    plt.rcParams['font.family'] = 'sans-serif'


# ═══════════════════════════════════════════════════════════════
# 树布局引擎
# ═══════════════════════════════════════════════════════════════

def _build_node_list(root):
    """深度优先遍历，生成扁平节点列表与边列表。"""
    nodes = []
    edges = []
    _counter = [0]

    def walk(node, parent_id):
        nid = _counter[0]; _counter[0] += 1
        nodes.append(dict(id=nid, name=node.name, count=node.count,
                          children=[], parent=parent_id, depth=0))
        if parent_id is not None:
            edges.append((parent_id, nid))
            nodes[parent_id]['children'].append(nid)
        for child in sorted(node.children.values(), key=lambda c: c.name):
            walk(child, nid)

    walk(root, None)

    # 计算深度
    def set_depth(nid, d):
        nodes[nid]['depth'] = d
        for c in nodes[nid]['children']:
            set_depth(c, d + 1)
    if nodes:
        set_depth(0, 0)

    return nodes, edges


def _layout_tree(nodes, edges, h_spacing=2.6, v_spacing=2.2):
    """
    改进的树布局：
    - 子树宽度按叶子数 + 节点数加权
    - 同层节点间自动分配间距
    """
    if not nodes:
        return {}, 0

    node_map = {n['id']: n for n in nodes}

    # 子树宽度
    def subtree_w(nid):
        ch = node_map[nid]['children']
        if not ch:
            return 1.0
        return sum(subtree_w(c) for c in ch) + 0.3 * (len(ch) - 1)

    widths = {nid: subtree_w(nid) for nid in node_map}
    total_w = widths[0]

    pos = {}

    def assign(nid, left, right):
        pos[nid] = {'x': (left + right) / 2}
        ch = node_map[nid]['children']
        if ch:
            scale = (right - left) / widths[nid]
            cur = left
            for c in ch:
                w = widths[c]
                assign(c, cur, cur + w * scale)
                cur += w * scale

    assign(0, 0, total_w * h_spacing)

    max_depth = max(n['depth'] for n in nodes)
    for n in nodes:
        pos[n['id']]['y'] = (max_depth - n['depth']) * v_spacing

    return pos, max_depth


# ═══════════════════════════════════════════════════════════════
# 绘制函数
# ═══════════════════════════════════════════════════════════════

def _draw_edge(ax, x1, y1, x2, y2, depth=0):
    """绘制贝塞尔曲线连线，越深颜色越浅。"""
    mid_y = (y1 + y2) / 2
    t = np.linspace(0, 1, 60)
    bx = (1 - t)**2 * x1 + 2 * (1 - t) * t * ((x1 + x2) / 2) + t**2 * x2
    by = (1 - t)**2 * y1 + 2 * (1 - t) * t * mid_y + t**2 * y2

    alpha = max(0.25, 0.7 - depth * 0.1)
    lw = max(1.0, 2.2 - depth * 0.25)
    color = COLORS['tree_line']

    ax.plot(bx, by, color=color, lw=lw, alpha=alpha, zorder=1,
            solid_capstyle='round')


def _draw_null_node(ax, x, y):
    """绘制 NULL 根节点 — 圆形深色。"""
    r = 0.65
    # 阴影
    shadow = plt.Circle((x + 0.04, y - 0.04), r, color='#B0BEC5',
                         alpha=0.35, zorder=4)
    ax.add_patch(shadow)
    # 主圆
    circle = plt.Circle((x, y), r, color=COLORS['null_node'],
                         ec=COLORS['node_border'], lw=2.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y + 0.08, 'NULL', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['null_text'], zorder=6)
    ax.text(x, y - 0.18, 'root', ha='center', va='center',
            fontsize=7, color='#B0BEC5', style='italic', zorder=6)


def _draw_tree_node(ax, x, y, name, count, max_count, depth=0):
    """绘制普通 FP 树节点 — 现代卡片风格。"""
    display_name = name if len(name) <= 7 else name[:6] + '..'

    # 节点尺寸
    bw, bh = 1.9, 1.05

    # 颜色强度
    intensity = min(1.0, count / max(max_count, 1))
    fill_color = NODE_CMAP(0.25 + 0.65 * intensity)
    edge_color = NODE_CMAP(0.55 + 0.35 * intensity)

    # 投影阴影
    shadow = FancyBboxPatch(
        (x - bw/2 + 0.06, y - bh/2 - 0.06), bw, bh,
        boxstyle="round,pad=0.18", facecolor='#90A4AE',
        alpha=0.22, edgecolor='none', zorder=3
    )
    ax.add_patch(shadow)

    # 主体
    box = FancyBboxPatch(
        (x - bw/2, y - bh/2), bw, bh,
        boxstyle="round,pad=0.18",
        facecolor=fill_color, edgecolor=edge_color, lw=2, zorder=5
    )
    ax.add_patch(box)

    # 顶部色条（按深度着色）
    dci = min(depth, len(COLORS['depth_colors']) - 1)
    bar_color = COLORS['depth_colors'][dci]
    bar = FancyBboxPatch(
        (x - bw/2 + 0.08, y + bh/2 - 0.22), bw - 0.16, 0.14,
        boxstyle="round,pad=0.04",
        facecolor=bar_color, edgecolor='none', alpha=0.7, zorder=6
    )
    ax.add_patch(bar)

    # 文字
    ax.text(x, y + 0.14, display_name, ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['node_text'],
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2, foreground=fill_color)])

    # 计数徽章
    badge_x = x + bw/2 - 0.2
    badge_y = y - 0.3
    badge = plt.Circle((badge_x, badge_y), 0.28,
                        color=COLORS['count_text'], ec='white', lw=1.5, zorder=8)
    ax.add_patch(badge)
    ax.text(badge_x, badge_y, str(count), ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=9)


def _draw_header_table(ax, header_items, y_top):
    """绘制现代卡片式 Header Table。"""
    n = len(header_items)
    if n == 0:
        return

    col_x = [1.0, 6.0, 9.0]     # 项名 / 总支持度 / 链节点
    col_w = [4.5, 2.5, 2.0]
    row_h = 0.72
    pad_top = 0.2

    # 整体背景卡片
    card_h = (n + 1.5) * row_h + 1.0
    card = FancyBboxPatch(
        (0.1, y_top - card_h), 11.0, card_h,
        boxstyle="round,pad=0.3",
        facecolor=COLORS['header_bg'],
        edgecolor=COLORS['sep'], lw=1.5, zorder=2,
        alpha=0.95
    )
    ax.add_patch(card)

    # 标题栏
    title_bar = FancyBboxPatch(
        (0.1, y_top - row_h - 0.3), 11.0, row_h + 0.1,
        boxstyle="round,pad=0.25",
        facecolor='#E8F5E9', edgecolor='#A5D6A7', lw=1.2, zorder=3
    )
    ax.add_patch(title_bar)

    # 标题文字
    ax.text(5.6, y_top - row_h/2 - 0.25, 'Header Table',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=COLORS['header_title'], zorder=4)
    ax.text(5.6, y_top - row_h/2 - 0.62,
            f'{n} items sorted by support',
            ha='center', va='center', fontsize=8,
            color=COLORS['subtitle'], style='italic', zorder=4)

    # 表头行
    hy = y_top - row_h - 0.9
    headers = ['Item', 'Support', 'Nodes']
    for cx, hdr, cw in zip(col_x, headers, col_w):
        ax.text(cx, hy, hdr, ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLORS['subtitle'],
                zorder=4)

    # 分隔线
    ax.plot([0.5, 10.7], [hy - row_h/2 + 0.05, hy - row_h/2 + 0.05],
            color=COLORS['sep'], lw=1.0, zorder=3)

    max_sup = header_items[0][1] if header_items else 1

    # 数据行
    for i, (item_name, total, n_links) in enumerate(header_items):
        ry = hy - (i + 0.8) * row_h

        # 交替行背景
        if i % 2 == 0:
            row_bg = FancyBboxPatch(
                (0.3, ry - row_h/2 + 0.05), 10.6, row_h - 0.1,
                boxstyle="round,pad=0.08",
                facecolor='#F5F7FA', edgecolor='none', alpha=0.6, zorder=2
            )
            ax.add_patch(row_bg)

        # 项名 + 支持度条
        display = item_name if len(item_name) <= 8 else item_name[:7] + '..'
        ax.text(col_x[0], ry + 0.08, display, ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLORS['header_item'],
                zorder=4)

        # 迷你条形图
        bar_max_w = 3.5
        bar_w = max(0.4, bar_max_w * total / max(max_sup, 1))
        bar_bg_patch = FancyBboxPatch(
            (col_x[0] - bar_max_w/2, ry - 0.28), bar_max_w, 0.16,
            boxstyle="round,pad=0.04",
            facecolor=COLORS['bar_bg'], edgecolor='none', zorder=3
        )
        ax.add_patch(bar_bg_patch)
        bar_fill = FancyBboxPatch(
            (col_x[0] - bar_max_w/2, ry - 0.28), bar_w, 0.16,
            boxstyle="round,pad=0.04",
            facecolor=COLORS['bar_fill'], edgecolor='none', alpha=0.85, zorder=3
        )
        ax.add_patch(bar_fill)

        # 总支持度（数字徽章）
        badge = FancyBboxPatch(
            (col_x[1] - 0.55, ry - 0.22), 1.1, 0.44,
            boxstyle="round,pad=0.1",
            facecolor='#FFEBEE', edgecolor='#EF9A9A', lw=0.8, zorder=3
        )
        ax.add_patch(badge)
        ax.text(col_x[1], ry, str(total), ha='center', va='center',
                fontsize=11, fontweight='bold', color=COLORS['header_val'],
                zorder=4)

        # 链节点数
        badge2 = FancyBboxPatch(
            (col_x[2] - 0.45, ry - 0.2), 0.9, 0.4,
            boxstyle="round,pad=0.08",
            facecolor='#ECEFF1', edgecolor='#B0BEC5', lw=0.6, zorder=3
        )
        ax.add_patch(badge2)
        ax.text(col_x[2], ry, str(n_links), ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLORS['header_num'],
                zorder=4)

    return card_h


def plot_fptree(model, title="FP-Growth Tree"):
    """
    FP 树出版级可视化。
    参数 model: 训练好的 FPGrowthModel 实例
    """
    _setup_font()

    tree = getattr(model, '_tree', None)
    if tree is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "FP Tree not built\nRun FP-Growth first",
                ha='center', va='center', fontsize=15, color='#90A4AE',
                fontweight='bold')
        ax.set_axis_off()
        return fig

    nodes, edges = _build_node_list(tree.root)
    if len(nodes) <= 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "FP Tree is empty\nAdjust min_support",
                ha='center', va='center', fontsize=15, color='#90A4AE',
                fontweight='bold')
        ax.set_axis_off()
        return fig

    pos, max_depth = _layout_tree(nodes, edges)

    # ── Header Table 数据 ──
    header_items = []
    for item_name, (head, _) in tree.header_table.items():
        total, n_links = 0, 0
        nd = head
        while nd is not None:
            total += nd.count
            n_links += 1
            nd = nd.next
        header_items.append((item_name, total, n_links))
    header_items.sort(key=lambda x: -x[1])

    # ── 画布尺寸 ──
    has_header = len(header_items) > 0
    n_display_nodes = len(nodes)
    fig_w = max(15, n_display_nodes * 0.7 + 6)
    fig_h = max(9, (max_depth + 1) * 2.6 + 1.5)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COLORS['bg'])

    tree_w_ratio = 0.58 if has_header else 0.94
    tree_ax = fig.add_axes([0.01, 0.07, tree_w_ratio, 0.86])
    header_ax = fig.add_axes([0.62, 0.07, 0.36, 0.86]) if has_header else None

    # ═════════════ 绘制树 ═════════════
    ax = tree_ax
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')

    # 标题
    ax.text(0.5, 1.02, 'FP-Growth Tree Structure',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=16, fontweight='bold', color=COLORS['title'],
            path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['bg'])])

    # 坐标范围
    all_x = [pos[n['id']]['x'] for n in nodes]
    all_y = [pos[n['id']]['y'] for n in nodes]
    xpad = 2.0
    ypad = 1.5
    ax.set_xlim(min(all_x) - xpad, max(all_x) + xpad)
    ax.set_ylim(min(all_y) - ypad, max(all_y) + ypad + 0.5)

    max_count = max((n['count'] for n in nodes if n['name'] != 'NULL'), default=1)

    # 连线（先画，在节点下方）
    for pid, cid in edges:
        p = pos[pid]; c = pos[cid]
        pnode = next(n for n in nodes if n['id'] == pid)
        _draw_edge(ax, p['x'], p['y'], c['x'], c['y'], depth=pnode['depth'])

    # 节点
    for n in nodes:
        x = pos[n['id']]['x']
        y = pos[n['id']]['y']
        if n['name'] == 'NULL':
            _draw_null_node(ax, x, y)
        else:
            _draw_tree_node(ax, x, y, n['name'], n['count'], max_count, n['depth'])

    # ═════════════ Header Table ═════════════
    if header_ax:
        hax = header_ax
        hax.set_facecolor(COLORS['bg'])
        hax.set_xlim(-0.5, 12)
        hax.set_ylim(-1, len(header_items) * 0.72 + 4)
        hax.axis('off')
        _draw_header_table(hax, header_items, len(header_items) * 0.72 + 3.5)

    # ═════════════ 图例 ═════════════
    legend_text = (
        "Node color intensity = support count  |  "
        "Red badge = transaction count  |  "
        "Top color bar = tree depth"
    )
    fig.text(0.5, 0.012, legend_text, ha='center', va='bottom',
             fontsize=8.5, color=COLORS['footer'], style='italic')

    # 节点数统计
    n_inner = sum(1 for n in nodes if n['name'] != 'NULL')
    fig.text(0.02, 0.012,
             f"Nodes: {n_inner + 1} (1 root + {n_inner} items)  |  "
             f"Depth: {max_depth + 1} levels  |  "
             f"Header items: {len(header_items)}",
             ha='left', va='bottom', fontsize=8, color=COLORS['footer'])

    return fig
