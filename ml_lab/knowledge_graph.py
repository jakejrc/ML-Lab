# -*- coding: utf-8 -*-
"""
ML-Lab 知识图谱模块
定义机器学习核心概念及其关联关系，支持交互式可视化。
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os, json, tempfile

# ── 颜色方案 ──
COLOR_MAP = {
    "基础":      "#4F46E5",
    "数据":      "#10B981",
    "预处理":    "#059669",
    "特征工程":  "#8B5CF6",
    "监督学习":  "#F59E0B",
    "无监督学习": "#EF4444",
    "评估":      "#EC4899",
    "模型优化":  "#06B6D4",
    "深度学习":  "#F97316",
    "部署":      "#6366F1",
}

RELATION_COLORS = {
    "前置知识":   "#94A3B8",
    "包含":       "#64748B",
    "相关":       "#CBD5E1",
    "实现方式":   "#A78BFA",
    "评估方法":   "#F472B6",
    "优化方法":   "#22D3EE",
}

NODES = [
    ("math_basics", "数学基础", "基础", "线性代数、概率统计、微积分"),
    ("linear_algebra", "线性代数", "基础", "矩阵运算、特征值分解、SVD"),
    ("probability", "概率统计", "基础", "概率分布、贝叶斯定理、假设检验"),
    ("calculus", "微积分", "基础", "导数、梯度、偏导数、链式法则"),
    ("optimization", "最优化理论", "基础", "凸优化、梯度下降、拉格朗日乘数法"),
    ("data_collection", "数据采集", "数据", "数据获取、API调用、爬虫"),
    ("dataset", "数据集", "数据", "训练集/测试集/验证集划分"),
    ("data_exploration", "数据探索", "数据", "EDA、可视化分析、统计描述"),
    ("data_quality", "数据质量", "数据", "缺失值、异常值、重复值检测"),
    ("missing_value", "缺失值处理", "预处理", "均值填充、中位数填充、删除"),
    ("outlier", "异常值处理", "预处理", "IQR法、Z-Score法、DBSCAN检测"),
    ("encoding", "特征编码", "预处理", "Label Encoding、One-Hot Encoding"),
    ("data_split", "数据集划分", "预处理", "训练/测试/验证集、分层抽样"),
    ("scaling", "特征缩放", "特征工程", "标准化、归一化、鲁棒缩放"),
    ("standardization", "标准化", "特征工程", "StandardScaler: 均值0方差1"),
    ("normalization", "归一化", "特征工程", "MinMaxScaler: 缩放到[0,1]"),
    ("feature_selection", "特征选择", "特征工程", "过滤法/包裹法/嵌入法"),
    ("pca", "PCA降维", "特征工程", "主成分分析、方差解释率"),
    ("feature_construction", "特征构造", "特征工程", "多项式、交互特征、统计特征"),
    ("discretization", "分箱离散化", "特征工程", "等宽分箱、等频分箱、KMeans分箱"),
    ("correlation", "相关性分析", "特征工程", "皮尔逊相关系数、热力图"),
    ("classification", "分类算法", "监督学习", "预测离散类别标签"),
    ("logistic_regression", "逻辑回归", "监督学习", "线性分类器、Sigmoid函数"),
    ("knn", "K近邻", "监督学习", "基于距离的非参数分类器"),
    ("decision_tree", "决策树", "监督学习", "树形结构、信息增益、基尼系数"),
    ("svm", "支持向量机", "监督学习", "最大间隔分类器、核技巧"),
    ("naive_bayes", "朴素贝叶斯", "监督学习", "基于贝叶斯定理、特征独立性假设"),
    ("random_forest", "随机森林", "监督学习", "Bagging集成、多棵决策树"),
    ("gbdt", "梯度提升树", "监督学习", "Boosting集成、GBDT/XGBoost/LightGBM"),
    ("neural_network", "神经网络", "监督学习", "多层感知机、反向传播"),
    ("regression", "回归算法", "监督学习", "预测连续数值"),
    ("linear_regression", "线性回归", "监督学习", "最小二乘法、OLS"),
    ("ridge", "Ridge回归", "监督学习", "L2正则化、防止过拟合"),
    ("lasso", "Lasso回归", "监督学习", "L1正则化、特征选择"),
    ("polynomial_regression", "多项式回归", "监督学习", "多项式特征+线性回归"),
    ("svr", "SVR", "监督学习", "支持向量回归、ε-不敏感损失"),
    ("decision_tree_reg", "决策树回归", "监督学习", "CART回归树"),
    ("clustering", "聚类算法", "无监督学习", "无标签数据分组"),
    ("kmeans", "K-Means", "无监督学习", "基于距离的划分聚类"),
    ("dbscan", "DBSCAN", "无监督学习", "基于密度的聚类、可发现任意形状"),
    ("hierarchical", "层次聚类", "无监督学习", "树状图、凝聚/分裂策略"),
    ("association", "关联规则", "无监督学习", "发现频繁项集和关联关系"),
    ("apriori", "Apriori", "无监督学习", "逐层搜索候选集、连接+剪枝"),
    ("fp_growth", "FP-Growth", "无监督学习", "FP树压缩、无需候选集"),
    ("eval_metrics", "评估指标", "评估", "模型性能量化评估"),
    ("accuracy", "准确率", "评估", "正确预测比例"),
    ("precision_recall", "精确率与召回率", "评估", "Precision、Recall、F1-Score"),
    ("confusion_matrix", "混淆矩阵", "评估", "TP/FP/FN/TN矩阵"),
    ("roc_auc", "ROC与AUC", "评估", "ROC曲线、AUC面积"),
    ("mse_rmse", "MSE/RMSE", "评估", "均方误差、均方根误差"),
    ("mae", "MAE", "评估", "平均绝对误差"),
    ("r2_score", "R²决定系数", "评估", "模型解释方差比例"),
    ("silhouette", "轮廓系数", "评估", "聚类效果评估"),
    ("elbow_method", "肘部法则", "评估", "K-Means最优K值选择"),
    ("model_optimization", "模型优化", "模型优化", "提升模型性能的技术"),
    ("cross_validation", "交叉验证", "模型优化", "K折交叉验证、留一法"),
    ("grid_search", "网格搜索", "模型优化", "超参数网格搜索"),
    ("regularization", "正则化", "模型优化", "L1/L2正则化、防止过拟合"),
    ("learning_curve", "学习曲线", "模型优化", "过拟合/欠拟合诊断"),
    ("bias_variance", "偏差-方差权衡", "模型优化", "模型复杂度与泛化能力"),
    ("ensemble", "集成学习", "模型优化", "Bagging/Boosting/Stacking"),
    ("feature_importance", "特征重要性", "模型优化", "模型特征贡献度分析"),
]

EDGES = [
    ("math_basics", "linear_algebra", "包含"),
    ("math_basics", "probability", "包含"),
    ("math_basics", "calculus", "包含"),
    ("math_basics", "optimization", "包含"),
    ("calculus", "optimization", "前置知识"),
    ("data_collection", "dataset", "包含"),
    ("dataset", "data_split", "包含"),
    ("data_exploration", "data_quality", "相关"),
    ("data_quality", "missing_value", "包含"),
    ("data_quality", "outlier", "包含"),
    ("dataset", "data_exploration", "相关"),
    ("missing_value", "scaling", "前置知识"),
    ("encoding", "scaling", "前置知识"),
    ("data_split", "scaling", "前置知识"),
    ("scaling", "standardization", "包含"),
    ("scaling", "normalization", "包含"),
    ("feature_selection", "pca", "相关"),
    ("feature_selection", "feature_construction", "相关"),
    ("feature_construction", "discretization", "相关"),
    ("feature_construction", "correlation", "相关"),
    ("linear_algebra", "linear_regression", "前置知识"),
    ("probability", "naive_bayes", "前置知识"),
    ("optimization", "logistic_regression", "前置知识"),
    ("optimization", "neural_network", "前置知识"),
    ("calculus", "neural_network", "前置知识"),
    ("scaling", "svm", "前置知识"),
    ("scaling", "knn", "前置知识"),
    ("scaling", "neural_network", "前置知识"),
    ("feature_selection", "classification", "前置知识"),
    ("pca", "classification", "前置知识"),
    ("classification", "logistic_regression", "实现方式"),
    ("classification", "knn", "实现方式"),
    ("classification", "decision_tree", "实现方式"),
    ("classification", "svm", "实现方式"),
    ("classification", "naive_bayes", "实现方式"),
    ("classification", "random_forest", "实现方式"),
    ("classification", "gbdt", "实现方式"),
    ("classification", "neural_network", "实现方式"),
    ("decision_tree", "random_forest", "相关"),
    ("decision_tree", "gbdt", "相关"),
    ("regression", "linear_regression", "实现方式"),
    ("regression", "ridge", "实现方式"),
    ("regression", "lasso", "实现方式"),
    ("regression", "polynomial_regression", "实现方式"),
    ("regression", "svr", "实现方式"),
    ("regression", "decision_tree_reg", "实现方式"),
    ("linear_regression", "ridge", "相关"),
    ("linear_regression", "lasso", "相关"),
    ("linear_regression", "polynomial_regression", "相关"),
    ("clustering", "kmeans", "实现方式"),
    ("clustering", "dbscan", "实现方式"),
    ("clustering", "hierarchical", "实现方式"),
    ("association", "apriori", "实现方式"),
    ("association", "fp_growth", "实现方式"),
    ("eval_metrics", "accuracy", "包含"),
    ("eval_metrics", "precision_recall", "包含"),
    ("eval_metrics", "confusion_matrix", "包含"),
    ("eval_metrics", "roc_auc", "包含"),
    ("eval_metrics", "mse_rmse", "包含"),
    ("eval_metrics", "mae", "包含"),
    ("eval_metrics", "r2_score", "包含"),
    ("eval_metrics", "silhouette", "包含"),
    ("eval_metrics", "elbow_method", "包含"),
    ("classification", "accuracy", "评估方法"),
    ("classification", "precision_recall", "评估方法"),
    ("classification", "confusion_matrix", "评估方法"),
    ("classification", "roc_auc", "评估方法"),
    ("regression", "mse_rmse", "评估方法"),
    ("regression", "mae", "评估方法"),
    ("regression", "r2_score", "评估方法"),
    ("clustering", "silhouette", "评估方法"),
    ("clustering", "elbow_method", "评估方法"),
    ("model_optimization", "cross_validation", "包含"),
    ("model_optimization", "grid_search", "包含"),
    ("model_optimization", "regularization", "包含"),
    ("model_optimization", "learning_curve", "包含"),
    ("model_optimization", "bias_variance", "包含"),
    ("model_optimization", "ensemble", "包含"),
    ("model_optimization", "feature_importance", "包含"),
    ("classification", "cross_validation", "优化方法"),
    ("classification", "grid_search", "优化方法"),
    ("regression", "regularization", "优化方法"),
    ("regression", "cross_validation", "优化方法"),
    ("neural_network", "regularization", "优化方法"),
    ("ridge", "regularization", "相关"),
    ("lasso", "regularization", "相关"),
    ("random_forest", "ensemble", "相关"),
    ("gbdt", "ensemble", "相关"),
    ("decision_tree", "bias_variance", "相关"),
    ("classification", "regression", "相关"),
    ("knn", "kmeans", "相关"),
    ("svm", "svr", "相关"),
    ("decision_tree", "decision_tree_reg", "相关"),
    ("logistic_regression", "neural_network", "相关"),
    ("pca", "feature_selection", "相关"),
    ("data_exploration", "correlation", "相关"),
    ("data_exploration", "eval_metrics", "相关"),
]


def build_graph():
    G = nx.Graph()
    for node_id, label, category, desc in NODES:
        G.add_node(node_id, label=label, category=category, description=desc)
    for src, dst, rel_type in EDGES:
        G.add_edge(src, dst, relation=rel_type)
    return G


def get_category_stats(G):
    stats = {}
    for node_id, data in G.nodes(data=True):
        cat = data.get("category", "其他")
        stats[cat] = stats.get(cat, 0) + 1
    return stats


def get_relation_stats(G):
    stats = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "其他")
        stats[rel] = stats.get(rel, 0) + 1
    return stats


def find_shortest_path(G, source, target):
    try:
        path = nx.shortest_path(G, source=source, target=target)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def find_related_concepts(G, node_id, max_distance=1):
    if node_id not in G:
        return []
    related = {}
    for n in G.neighbors(node_id):
        edge_data = G.get_edge_data(node_id, n)
        related[n] = {
            "label": G.nodes[n].get("label", n),
            "category": G.nodes[n].get("category", ""),
            "relation": edge_data.get("relation", "相关") if edge_data else "相关",
            "distance": 1,
        }
    if max_distance >= 2:
        for n1 in G.neighbors(node_id):
            for n2 in G.neighbors(n1):
                if n2 != node_id and n2 not in related:
                    edge_data = G.get_edge_data(n1, n2)
                    related[n2] = {
                        "label": G.nodes[n2].get("label", n2),
                        "category": G.nodes[n2].get("category", ""),
                        "relation": f"通过 {G.nodes[n1].get('label', n1)} 关联",
                        "distance": 2,
                    }
    return dict(sorted(related.items(), key=lambda x: x[1]["distance"]))


def search_concepts(G, keyword):
    keyword = keyword.lower()
    results = []
    for node_id, data in G.nodes(data=True):
        label = data.get("label", "").lower()
        desc = data.get("description", "").lower()
        if keyword in label or keyword in desc or keyword in node_id:
            results.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "category": data.get("category", ""),
                "description": data.get("description", ""),
            })
    return results


def plot_knowledge_graph(G, figsize=(28, 20), layout_seed=42):
    """渲染知识图谱静态图 — 白字+深色描边，任何颜色节点上清晰可读"""
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    pos = nx.spring_layout(G, k=5.5, iterations=200, seed=layout_seed)

    categories = {}
    for node_id, data in G.nodes(data=True):
        cat = data.get("category", "其他")
        categories.setdefault(cat, []).append(node_id)

    # 边
    edge_colors, edge_widths = [], []
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "相关")
        edge_colors.append(RELATION_COLORS.get(rel, "#CBD5E1"))
        edge_widths.append(0.6 if rel == "相关" else 1.0)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.25, style='solid')

    # 节点 — 大尺寸
    for cat, nodes in categories.items():
        color = COLOR_MAP.get(cat, "#94A3B8")
        nodelist = [n for n in nodes if n in G]
        sizes = [3000 + 500 * len(list(G.neighbors(n))) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodelist,
                               node_color=color, node_size=sizes, alpha=0.92,
                               edgecolors='white', linewidths=2.5)

    # 标签 — 白色文字 + 深色描边（path_effects），任何颜色节点上都清晰
    for node_id, (x, y) in pos.items():
        label = G.nodes[node_id].get("label", node_id)
        deg = len(list(G.neighbors(node_id)))
        fs = 10 + deg * 0.6  # 字号随连接数增大
        ax.text(x, y, label, fontsize=fs, ha='center', va='center',
                fontweight='bold', color='white',
                path_effects=[pe.withStroke(linewidth=3.5, foreground='#1e293b')])

    # 图例
    legend_patches = []
    for cat, color in COLOR_MAP.items():
        if cat in categories:
            legend_patches.append(mpatches.Patch(color=color, label=f"{cat} ({len(categories[cat])})"))
    legend1 = ax.legend(handles=legend_patches, title="概念类别", loc='upper left',
                        fontsize=10, title_fontsize=11, framealpha=0.9, edgecolor='#e2e8f0')
    ax.add_artist(legend1)

    rel_patches = [mpatches.Patch(color=c, label=r) for r, c in RELATION_COLORS.items()]
    legend2 = ax.legend(handles=rel_patches, title="关系类型", loc='upper right',
                        fontsize=10, title_fontsize=11, framealpha=0.9, edgecolor='#e2e8f0')
    ax.add_artist(legend2)

    ax.set_title("ML-Lab 机器学习知识图谱", fontsize=20, fontweight='bold', pad=24, color='#1e293b')
    ax.axis('off')
    plt.tight_layout(pad=1.5)
    return fig


def generate_interactive_html(G, width="100%", height="700px"):
    try:
        from pyvis.network import Network
    except ImportError:
        return _generate_fallback_html(G)
    net = Network(height=height, width=width, directed=False, bgcolor="#FAFAFA", font_color="#1e293b")
    for node_id, data in G.nodes(data=True):
        cat = data.get("category", "其他")
        color = COLOR_MAP.get(cat, "#94A3B8")
        label = data.get("label", node_id)
        desc = data.get("description", "")
        title = f"<b>{label}</b><br><span style='color:{color};'>{cat}</span><br><small>{desc}</small>"
        size = 20 + 5 * len(list(G.neighbors(node_id)))
        net.add_node(node_id, label=label, title=title, color=color, size=size, borderWidth=2, borderColor="#ffffff")
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "相关")
        color = RELATION_COLORS.get(rel, "#CBD5E1")
        width_val = 1 if rel == "相关" else 2
        net.add_edge(u, v, title=rel, color=color, width=width_val, smooth={"type": "continuous"})
    net.set_options("""{
      "physics": {
        "stabilization": {"iterations": 200},
        "barnesHut": {"gravitationalConstant": -3000, "centralGravity": 0.3, "springLength": 200, "springConstant": 0.04, "damping": 0.5}
      },
      "interaction": {"hover": true, "tooltipDelay": 200, "zoomView": true, "dragView": true, "navigationButtons": true, "keyboard": true},
      "edges": {"smooth": {"type": "continuous"}, "font": {"size": 10, "color": "#64748b"}}
    }""")
    return net.generate_html()


def _generate_fallback_html(G):
    nodes_json = []
    for node_id, data in G.nodes(data=True):
        nodes_json.append({"id": node_id, "label": data.get("label", node_id), "category": data.get("category", ""), "description": data.get("description", "")})
    edges_json = []
    for u, v, data in G.edges(data=True):
        edges_json.append({"source": u, "target": v, "relation": data.get("relation", "相关")})

    import json as _json
    color_map_json = _json.dumps(COLOR_MAP, ensure_ascii=False)
    rel_colors_json = _json.dumps(RELATION_COLORS, ensure_ascii=False)
    nodes_json_str = _json.dumps(nodes_json, ensure_ascii=False)
    edges_json_str = _json.dumps(edges_json, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML-Lab 知识图谱</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: system-ui, sans-serif; background: #fafafa; }}
  #graph {{ width: 100%; height: 700px; position: relative; }}
  .tooltip {{
    position: absolute; padding: 10px 14px; background: rgba(30,41,59,0.95);
    color: #e2e8f0; border-radius: 8px; font-size: 13px; pointer-events: none;
    max-width: 280px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    display: none; line-height: 1.5; z-index: 100;
  }}
  .tooltip .cat {{ font-size: 11px; opacity: 0.7; }}
  .tooltip .desc {{ font-size: 12px; margin-top: 4px; color: #94a3b8; }}
  .legend {{
    position: absolute; top: 16px; left: 16px; background: white;
    padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    font-size: 12px; max-height: 70vh; overflow-y: auto; z-index: 10;
  }}
  .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
  .legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }}
  .legend-title {{ font-weight: 600; margin-bottom: 6px; color: #1e293b; }}
  .controls {{
    position: absolute; bottom: 16px; right: 16px; display: flex; gap: 6px; z-index: 10;
  }}
  .controls button {{
    padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px;
    background: white; cursor: pointer; font-size: 12px; color: #475569;
  }}
  .controls button:hover {{ background: #f1f5f9; }}
</style>
</head>
<body>
<div id="graph">
  <div class="tooltip" id="tooltip"></div>
  <div class="legend" id="legend"></div>
  <div class="controls">
    <button onclick="zoomIn()">+</button>
    <button onclick="zoomOut()">-</button>
    <button onclick="resetView()">重置</button>
  </div>
</div>
<script>
const nodesData = {nodes_json_str};
const edgesData = {edges_json_str};
const colorMap = {color_map_json};
const width = document.getElementById('graph').clientWidth;
const height = 700;
const svg = d3.select('#graph').insert('svg', ':first-child').attr('width', width).attr('height', height);
const g = svg.append('g');
const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (event) => g.attr('transform', event.transform));
svg.call(zoom);
const simulation = d3.forceSimulation(nodesData)
    .force('link', d3.forceLink(edgesData).id(d => d.id).distance(120))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(30));
const link = g.append('g').selectAll('line').data(edgesData).join('line')
    .attr('stroke', d => {{ const rc = {rel_colors_json}; return rc[d.relation] || '#CBD5E1'; }})
    .attr('stroke-width', d => d.relation === '相关' ? 0.8 : 1.5).attr('stroke-opacity', 0.4);
const node = g.append('g').selectAll('circle').data(nodesData).join('circle')
    .attr('r', d => 15 + Math.random() * 12).attr('fill', d => colorMap[d.category] || '#94A3B8')
    .attr('stroke', '#fff').attr('stroke-width', 1.5).attr('cursor', 'pointer')
    .call(d3.drag().on('start', (event, d) => {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
        .on('drag', (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
        .on('end', (event, d) => {{ if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));
const label = g.append('g').selectAll('text').data(nodesData).join('text')
    .text(d => d.label).attr('font-size', '12px').attr('font-weight', 'bold')
    .attr('fill', '#fff').attr('stroke', '#1e293b').attr('stroke-width', '3.5px').attr('paint-order', 'stroke')
    .attr('text-anchor', 'middle').attr('dy', '0.35em').attr('pointer-events', 'none');
const tooltip = document.getElementById('tooltip');
node.on('mouseover', (event, d) => {{ tooltip.style.display = 'block'; tooltip.innerHTML = '<b>' + d.label + '</b><br><span class=\"cat\">' + d.category + '</span><div class=\"desc\">' + (d.description || '') + '</div>'; }})
    .on('mousemove', (event) => {{ tooltip.style.left = (event.offsetX + 12) + 'px'; tooltip.style.top = (event.offsetY - 10) + 'px'; }})
    .on('mouseout', () => {{ tooltip.style.display = 'none'; }});
const legendEl = document.getElementById('legend');
let legendHtml = '<div class="legend-title">概念类别</div>';
for (const [cat, color] of Object.entries(colorMap)) {{
    if (nodesData.some(n => n.category === cat)) {{
        const count = nodesData.filter(n => n.category === cat).length;
        legendHtml += '<div class=\"legend-item\"><span class=\"legend-color\" style=\"background:' + color + '\"></span>' + cat + ' (' + count + ')</div>';
    }}
}}
legendEl.innerHTML = legendHtml;
simulation.on('tick', () => {{
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    label.attr('x', d => d.x).attr('y', d => d.y);
}});
function zoomIn() {{ svg.transition().call(zoom.scaleBy, 1.3); }}
function zoomOut() {{ svg.transition().call(zoom.scaleBy, 0.7); }}
function resetView() {{ svg.transition().call(zoom.transform, d3.zoomIdentity); }}
</script>
</body>
</html>"""
    return html


def save_interactive_html(G, output_path=None):
    html = generate_interactive_html(G)
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, encoding='utf-8')
        output_path = tmp.name
        tmp.close()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path


KG = build_graph()

if __name__ == "__main__":
    G = build_graph()
    print(f"知识图谱: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    for cat, count in sorted(get_category_stats(G).items()):
        print(f"  {cat}: {count}")
    html_path = save_interactive_html(G)
    print(f"交互式 HTML 已保存: {html_path}")
