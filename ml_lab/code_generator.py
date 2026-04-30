# -*- coding: utf-8 -*-
"""
ML-Lab v3.2 — 一键复制代码生成器

根据实验参数生成可直接运行的 sklearn Python 代码。
"""

from datetime import datetime


# 算法名 → (sklearn 模块, sklearn 类名) 的映射
SKLEARN_IMPORT_MAP = {
    "逻辑回归":       ("sklearn.linear_model",         "LogisticRegression"),
    "K近邻":          ("sklearn.neighbors",            "KNeighborsClassifier"),
    "决策树":          ("sklearn.tree",                 "DecisionTreeClassifier"),
    "SVM":            ("sklearn.svm",                  "SVC"),
    "朴素贝叶斯":     ("sklearn.naive_bayes",          "GaussianNB"),
    "随机森林":       ("sklearn.ensemble",             "RandomForestClassifier"),
    "梯度提升 (GBDT)": ("sklearn.ensemble",             "GradientBoostingClassifier"),
    "神经网络":       ("sklearn.neural_network",       "MLPClassifier"),
    "线性回归":       ("sklearn.linear_model",         "LinearRegression"),
    "Ridge 回归":     ("sklearn.linear_model",         "Ridge"),
    "Lasso 回归":     ("sklearn.linear_model",         "Lasso"),
    "多项式回归":     ("sklearn.preprocessing",        None),
    "SVR 回归":       ("sklearn.svm",                  "SVR"),
    "决策树回归":     ("sklearn.tree",                 "DecisionTreeRegressor"),
    "K-Means":        ("sklearn.cluster",              "KMeans"),
    "DBSCAN":         ("sklearn.cluster",              "DBSCAN"),
    "层次聚类":       ("sklearn.cluster",              "AgglomerativeClustering"),
    "PCA":            ("sklearn.decomposition",        "PCA"),
}

# ML-Lab 参数名 → sklearn 参数名
PARAM_ALIAS = {
    "C": "C", "max_depth": "max_depth", "max_iter": "max_iter",
    "hidden_layer_sizes": "hidden_layer_sizes", "learning_rate_init": "learning_rate_init",
    "alpha": "alpha", "learning_rate": "learning_rate",
    "degree": "degree", "epsilon": "eps", "kernel": "kernel",
    "n_clusters": "n_clusters", "n_components": "n_components",
    "min_samples": "min_samples", "linkage": "linkage",
    "affinity": "affinity", "init": "init",
}

# 数据集加载代码片段
DATASET_LOAD_SNIPPETS = {
    "鸢尾花 (Iris)": """\
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
feature_names = list(data.feature_names)
target_names = list(data.target_names)""",

    "葡萄酒 (Wine)": """\
from sklearn.datasets import load_wine
data = load_wine()
X, y = data.data, data.target
feature_names = list(data.feature_names)
target_names = list(data.target_names)""",

    "乳腺癌 (Breast Cancer)": """\
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)
target_names = list(data.target_names)""",

    "手写数字 (Digits)": """\
from sklearn.datasets import load_digits
data = load_digits()
X, y = data.data, data.target
feature_names = [f"pixel_{i}" for i in range(64)]
target_names = [str(i) for i in range(10)]""",

    "波士顿房价 (Boston)": """\
# 波士顿房价数据 — 使用 California Housing 替代 (sklearn 已移除 Boston)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)
target_names = ["房价(MEDV)"]""",

    "加州房价 (California)": """\
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)
target_names = ["房价(MedianHV)"]""",

    "线性回归": """\
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=5, noise=5.0, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
target_names = ["target"]""",

    "合成分类数据": """\
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
target_names = [f"class_0", "class_1"]""",

    "聚类球形 (Blobs)": """\
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=3, cluster_std=1.0, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
target_names = [f"cluster_{i}" for i in range(len(set(y)))]""",

    "聚类环形 (Circles)": """\
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
feature_names = ["feature_0", "feature_1"]
target_names = ["外环", "内环"]""",
}


def _build_header(task_type, algo, dataset_name):
    task_label = {"classification": "分类", "regression": "回归", "clustering": "聚类"}[task_type]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        f"# ══════════════════════════════════════════════════════════════",
        f"# ML-Lab v3.2  一键复制代码",
        f"# 实验类型: {task_label} | 算法: {algo} | 数据集: {dataset_name}",
        f"# 生成时间: {ts}",
        f"# 说明: 此代码由 ML-Lab 根据当前实验参数自动生成，可直接复制运行",
        f"# ══════════════════════════════════════════════════════════════",
        "",
    ]


def _build_imports(algo, task_type):
    lines = [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import (",
        "    accuracy_score, classification_report, confusion_matrix,",
        "    mean_squared_error, mean_absolute_error, r2_score,",
        "    silhouette_score, adjusted_rand_score",
        ")",
    ]
    if algo in SKLEARN_IMPORT_MAP:
        module, cls_name = SKLEARN_IMPORT_MAP[algo]
        if cls_name:
            lines.append(f"from {module} import {cls_name}")
    if algo == "多项式回归":
        lines.append("from sklearn.preprocessing import PolynomialFeatures")
        lines.append("from sklearn.linear_model import LinearRegression")
        lines.append("from sklearn.pipeline import make_pipeline")
    if task_type == "clustering":
        lines.append("from sklearn.preprocessing import StandardScaler")
    lines.append("")
    return lines


def _build_dataset_section(dataset_name):
    lines = [
        "# ══════════════════════════════════════════════════════════════",
        "# 1. 加载数据集",
        "# ══════════════════════════════════════════════════════════════",
    ]
    snippet = DATASET_LOAD_SNIPPETS.get(dataset_name)
    if snippet:
        lines.extend(snippet.split("\n"))
    else:
        lines.append(f"# 数据集: {dataset_name}")
        lines.append("X, y = np.array([...]), np.array([...])  # 请替换为实际数据")
    lines.extend([
        "",
        'print(f"数据形状: {X.shape}, 标签形状: {y.shape}")',
        "",
    ])
    return lines


def _build_split_section(task_type, test_ratio):
    lines = []
    if task_type in ("classification", "regression"):
        stratify = "stratify=y" if task_type == "classification" else ""
        lines.extend([
            "# ══════════════════════════════════════════════════════════════",
            "# 2. 划分训练集与测试集",
            "# ══════════════════════════════════════════════════════════════",
            "X_train, X_test, y_train, y_test = train_test_split(",
            f"    X, y, test_size={test_ratio}, random_state=42, {stratify})",
            'print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")',
            "",
        ])
    elif task_type == "clustering":
        lines.extend([
            "# ══════════════════════════════════════════════════════════════",
            "# 2. 数据标准化 (聚类前推荐)",
            "# ══════════════════════════════════════════════════════════════",
            "scaler = StandardScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
        ])
    return lines


def _build_model_section(algo, params):
    lines = [
        "# ══════════════════════════════════════════════════════════════",
        f"# 3. 构建 {algo} 模型",
        "# ══════════════════════════════════════════════════════════════",
    ]
    if algo == "多项式回归":
        deg = params.get("degree", 2)
        lines.append("# 多项式回归: 使用 Pipeline 组合多项式特征与线性回归")
        lines.append(f"model = make_pipeline(PolynomialFeatures(degree={deg}), LinearRegression())")
    elif algo in SKLEARN_IMPORT_MAP:
        _, cls_name = SKLEARN_IMPORT_MAP[algo]
        if cls_name:
            parts = []
            for k, v in params.items():
                sk_key = PARAM_ALIAS.get(k, k)
                if isinstance(v, str):
                    parts.append(f'{sk_key}="{v}"')
                else:
                    parts.append(f"{sk_key}={v}")
            param_str = ", ".join(parts)
            lines.append(f"model = {cls_name}({param_str})" if param_str else f"model = {cls_name}()")
    else:
        lines.append(f"model = ... # 请检查算法: {algo}")
    lines.extend([
        'print(f"模型: {model}")',
        "",
    ])
    return lines


def _build_train_section(task_type):
    lines = [
        "# ══════════════════════════════════════════════════════════════",
        "# 4. 训练模型",
        "# ══════════════════════════════════════════════════════════════",
    ]
    if task_type == "clustering":
        lines.extend([
            "labels = model.fit_predict(X_scaled)",
            'print("聚类完成!")',
            "",
        ])
    else:
        lines.extend([
            "model.fit(X_train, y_train)",
            'print("训练完成!")',
            "",
        ])
    return lines


def _build_eval_section(task_type, algo):
    lines = [
        "# ══════════════════════════════════════════════════════════════",
        "# 5. 模型评估与可视化",
        "# ══════════════════════════════════════════════════════════════",
    ]
    if task_type == "classification":
        lines.extend([
            "y_pred = model.predict(X_test)",
            'acc = accuracy_score(y_test, y_pred)',
            'print(f"准确率: {acc:.4f}")',
            'print("\\n分类报告:")',
            'print(classification_report(y_test, y_pred, target_names=target_names))',
            "",
            "# 混淆矩阵可视化",
            "fig, ax = plt.subplots(figsize=(6, 5))",
            "cm = confusion_matrix(y_test, y_pred)",
            'im = ax.imshow(cm, cmap="Blues")',
            "ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))",
            "ax.set_xticklabels(target_names); ax.set_yticklabels(target_names)",
            'ax.set_xlabel("预测值"); ax.set_ylabel("真实值")',
            f'ax.set_title("{algo} 混淆矩阵")',
            "for i in range(cm.shape[0]):",
            "    for j in range(cm.shape[1]):",
            '        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)',
            "plt.colorbar(im)",
            "plt.tight_layout()",
            'plt.savefig("confusion_matrix.png", dpi=150)',
            "plt.show()",
        ])
    elif task_type == "regression":
        lines.extend([
            "y_pred = model.predict(X_test)",
            'print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")',
            'print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")',
            'print(f"R²:   {r2_score(y_test, y_pred):.4f}")',
            "",
            "# 预测 vs 真实值散点图",
            "fig, ax = plt.subplots(figsize=(6, 6))",
            "ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')",
            'ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)',
            'ax.set_xlabel("真实值"); ax.set_ylabel("预测值")',
            f'ax.set_title("{algo} 预测 vs 真实")',
            "plt.tight_layout()",
            'plt.savefig("regression_result.png", dpi=150)',
            "plt.show()",
        ])
    elif task_type == "clustering":
        lines.extend([
            'print(f"轮廓系数: {silhouette_score(X_scaled, labels):.4f}")',
            "try:",
            '    print(f"调整兰德指数: {adjusted_rand_score(y, labels):.4f}")',
            "except Exception:",
            "    pass",
            "",
            "# 聚类散点图",
            "fig, ax = plt.subplots(figsize=(7, 6))",
            "scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', alpha=0.7)",
            'fn0 = feature_names[0] if len(feature_names) > 0 else "Feature 0"',
            'fn1 = feature_names[1] if len(feature_names) > 1 else "Feature 1"',
            "ax.set_xlabel(fn0); ax.set_ylabel(fn1)",
            f'ax.set_title("{algo} 聚类结果")',
            'plt.colorbar(scatter, label="簇标签")',
            "plt.tight_layout()",
            'plt.savefig("clustering_result.png", dpi=150)',
            "plt.show()",
        ])
    lines.append("")
    return lines


def generate_sklearn_code(task_type, algo, params, dataset_name, test_ratio=0.3):
    """根据实验参数生成可直接运行的 sklearn 代码。

    Args:
        task_type: "classification" / "regression" / "clustering"
        algo: 算法名称（中文）
        params: dict, 当前实验的有效参数键值对
        dataset_name: 数据集名称
        test_ratio: 测试集比例

    Returns:
        str: 完整可运行的 Python 代码字符串
    """
    all_lines = []
    all_lines.extend(_build_header(task_type, algo, dataset_name))
    all_lines.extend(_build_imports(algo, task_type))
    all_lines.extend(_build_dataset_section(dataset_name))
    all_lines.extend(_build_split_section(task_type, test_ratio))
    all_lines.extend(_build_model_section(algo, params))
    all_lines.extend(_build_train_section(task_type))
    all_lines.extend(_build_eval_section(task_type, algo))
    return "\n".join(all_lines)
