# -*- coding: utf-8 -*-
"""代码沙箱 - 上下文感知代码片段"""

from ml_lab.state import _g

# 分类代码片段
SNIPPETS_CLASSIFICATION = {
    "📊 快速训练分类模型": """# --- 快速训练分类模型 ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))
""",

    "📈 混淆矩阵可视化": """# --- 混淆矩阵 ---
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test,
                                       display_labels=target_names,
                                       cmap='Blues', xticks_rotation=45)
plt.title("混淆矩阵")
plt.tight_layout()
plt.show()
""",

    "🔍 特征重要性分析": """# --- 特征重要性 ---
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
ax.barh(importance["feature"], importance["importance"], color="#4F46E5")
ax.set_xlabel("重要性")
ax.set_title("特征重要性排序")
plt.tight_layout()
plt.show()
""",
}

SNIPPETS_REGRESSION = {
    "📊 快速训练回归模型": """# --- 快速训练回归模型 ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
""",

    "📈 预测值 vs 真实值": """# --- 预测值 vs 真实值 ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.6, color="#4F46E5", edgecolors="white")
min_v = min(y_test.min(), y_pred.min())
max_v = max(y_test.max(), y_pred.max())
ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="理想预测")
ax.set_xlabel("真实值")
ax.set_ylabel("预测值")
ax.set_title(f"预测效果 (R²={r2_score(y_test, y_pred):.3f})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
}

SNIPPETS_CLUSTERING = {
    "🔵 K-Means 聚类": """# --- K-Means 聚类 ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print(f"轮廓系数: {silhouette_score(X, labels):.4f}")
print(f"聚类中心:\n{kmeans.cluster_centers_}")
""",

    "📊 聚类散点图": """# --- 聚类结果可视化 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 原始聚类
scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30, edgecolors="white")
ax1.set_title("K-Means 聚类结果")
ax1.set_xlabel(feature_names[0] if feature_names else "特征1")
ax1.set_ylabel(feature_names[1] if feature_names else "特征2")
plt.colorbar(scatter, ax=ax1)

# 肘部法则
inertias = []
Ks = range(2, 11)
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
ax2.plot(Ks, inertias, "o-", color="#4F46E5")
ax2.set_xlabel("K 值")
ax2.set_ylabel("惯量 (Inertia)")
ax2.set_title("肘部法则")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
}

# 通用代码片段（与数据上下文无关）
SNIPPETS_GENERAL = {
    "📋 数据概览": """# --- 数据概览 ---
import pandas as pd
import numpy as np

df = pd.DataFrame(X, columns=feature_names) if feature_names is not None else pd.DataFrame(X)
df["target"] = y

print("形状:", X.shape)
print("\n前5行:")
print(df.head())
print("\n统计摘要:")
print(df.describe())
print("\n目标分布:")
print(pd.Series(y).value_counts())
""",

    "📊 散点图矩阵": """# --- 散点图矩阵 ---
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(X, columns=feature_names)
pd.plotting.scatter_matrix(df, figsize=(10, 10), alpha=0.5, diagonal="hist")
plt.suptitle("散点图矩阵", fontsize=14)
plt.tight_layout()
plt.show()
""",

    "📈 相关性热力图": """# --- 相关性热力图 ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame(X, columns=feature_names)
corr = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(val) > 0.5 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("特征相关性热力图")
plt.tight_layout()
plt.show()
""",

    "📦 数据预处理": """# --- 数据预处理 ---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 重新划分（如果不需要原有划分）
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"训练集: {X_tr.shape}, 测试集: {X_te.shape}")
print(f"标准化后均值: {X_scaled.mean(axis=0).round(4)}")
print(f"标准化后标准差: {X_scaled.std(axis=0).round(4)}")
""",

    "🧪 PCA 降维可视化": """# --- PCA 降维可视化 ---
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis",
                      s=30, alpha=0.7, edgecolors="white")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
ax.set_title("PCA 降维可视化")
plt.colorbar(scatter, ax=ax)
plt.tight_layout()
plt.show()
""",

    "📝 自定义分析": """# --- 在此编写你的自定义分析 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 你的代码从这里开始...
print("数据已就绪，开始分析!")
""",
}


def get_data_context():
    """获取当前数据上下文信息"""
    ctx = {
        "loaded": _g["X"] is not None,
        "n_samples": _g["X"].shape[0] if _g["X"] is not None else 0,
        "n_features": _g["X"].shape[1] if _g["X"] is not None else 0,
        "feature_names": _g["feature_names"],
        "target_names": _g["target_names"],
        "dataset_name": _g.get("dataset_name", "未加载"),
    }
    if _g["y"] is not None:
        unique = len(set(_g["y"].flatten())) if hasattr(_g["y"], "flatten") else len(set(_g["y"]))
        ctx["task_type"] = "classification" if unique <= 20 else "regression"
        ctx["n_classes"] = unique
    else:
        ctx["task_type"] = "unsupervised"
        ctx["n_classes"] = 0
    return ctx


def get_recommended_snippets():
    """根据数据上下文推荐代码片段"""
    ctx = get_data_context()
    recommended = []

    if not ctx["loaded"]:
        return [("📋 数据概览", SNIPPETS_GENERAL["📋 数据概览"])]

    # 总是推荐数据概览
    recommended.append(("📋 数据概览", SNIPPETS_GENERAL["📋 数据概览"]))

    # 根据任务类型推荐
    if ctx["task_type"] == "classification":
        recommended.append(("📊 训练分类模型", SNIPPETS_CLASSIFICATION["📊 快速训练分类模型"]))
        recommended.append(("📈 混淆矩阵", SNIPPETS_CLASSIFICATION["📈 混淆矩阵可视化"]))
        if ctx["n_features"] > 2:
            recommended.append(("🔍 特征重要性", SNIPPETS_CLASSIFICATION["🔍 特征重要性分析"]))
    elif ctx["task_type"] == "regression":
        recommended.append(("📊 训练回归模型", SNIPPETS_REGRESSION["📊 快速训练回归模型"]))
        recommended.append(("📈 预测效果图", SNIPPETS_REGRESSION["📈 预测值 vs 真实值"]))
    else:
        recommended.append(("🔵 K-Means 聚类", SNIPPETS_CLUSTERING["🔵 K-Means 聚类"]))
        recommended.append(("📊 聚类可视化", SNIPPETS_CLUSTERING["📊 聚类散点图"]))

    # 通用推荐
    recommended.append(("📈 相关性热力图", SNIPPETS_GENERAL["📈 相关性热力图"]))
    recommended.append(("🧪 PCA 降维", SNIPPETS_GENERAL["🧪 PCA 降维可视化"]))

    return recommended


def on_insert_snippet(current_code, snippet_key):
    """插入代码片段到编辑器（追加到末尾）"""
    all_snippets = {}
    for cat in [SNIPPETS_CLASSIFICATION, SNIPPETS_REGRESSION,
                SNIPPETS_CLUSTERING, SNIPPETS_GENERAL]:
        all_snippets.update(cat)

    snippet = all_snippets.get(snippet_key, "")
    if not snippet:
        return current_code

    # 在现有代码后追加空行 + 片段
    if current_code and current_code.strip():
        new_code = current_code.rstrip() + "\n\n" + snippet
    else:
        new_code = snippet

    return new_code


def on_get_context_snippets():
    """获取上下文感知的推荐片段HTML（用于状态显示）"""
    ctx = get_data_context()
    html = f'<div style="font-size:11px;color:#94a3b8;">'
    if ctx["loaded"]:
        html += f'📊 {ctx["dataset_name"]} | '
        html += f'样本: {ctx["n_samples"]} | 特征: {ctx["n_features"]} | '
        html += f'类型: {ctx["task_type"]}'
    else:
        html += "💡 上传数据后获取智能代码推荐"
    html += "</div>"
    return html
