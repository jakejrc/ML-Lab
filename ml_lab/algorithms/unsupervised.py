# -*- coding: utf-8 -*-
"""
ML-Lab 算法模块 v3.0
封装10大核心机器学习算法：
  监督学习 — 线性回归、逻辑回归、KNN、决策树、SVM、神经网络
  无监督学习 — K-Means 聚类、DBSCAN 密度聚类、PCA 降维、层次聚类
提供统一接口和训练历史记录，支持可视化回调。
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score, silhouette_score,
    pairwise_distances_argmin
)


# ═══════════════════════════════════════════════════════════════
# 监督学习算法（v1.0 保留，接口不变）
# ═══════════════════════════════════════════════════════════════

class KMeansModel:
    """K-Means 聚类模型"""

    def __init__(self, n_clusters=3, max_iter=300, init="k-means++"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.model = None
        self.labels_ = None
        self.history = {"inertia": [], "k_range": []}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        # 先执行一次完整训练获取最终结果
        self.model = KMeans(
            n_clusters=self.n_clusters, max_iter=self.max_iter,
            init=self.init, random_state=42, n_init=10
        )
        self.model.fit(X)
        self.labels_ = self.model.labels_
        # 为肘部法则计算不同 k 值下的真实 inertia
        k_max = min(10, len(X) - 1)
        k_range = list(range(2, k_max + 1))
        inertia_values = []
        for k in k_range:
            try:
                km = KMeans(n_clusters=k, max_iter=min(self.max_iter, 100),
                            init=self.init, random_state=42, n_init=5)
                km.fit(X)
                inertia_values.append(km.inertia_)
            except Exception:
                inertia_values.append(0)
        self.history["k_range"] = k_range
        self.history["inertia"] = inertia_values

        # 手动 K-Means 迭代过程捕获（用于教学可视化）
        # sklearn 1.9.0 已移除 warm_start，改用纯 numpy 实现
        centroids_path = []
        try:
            rng = np.random.RandomState(42)
            n_samples, _ = X.shape

            # k-means++ 初始化
            centroids = []
            centroids.append(X[rng.randint(n_samples)])
            for _ in range(1, self.n_clusters):
                dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
                probs = dists / (dists.sum() + 1e-10)
                centroids.append(X[rng.choice(n_samples, p=probs)])
            centroids = np.array(centroids)
            centroids_path.append(centroids.copy())

            # 逐轮迭代（最多 20 轮）
            max_iters = min(self.max_iter, 20)
            for _ in range(max_iters):
                labels = pairwise_distances_argmin(X, centroids)
                new_centroids = np.array([
                    X[labels == k].mean(axis=0) for k in range(self.n_clusters)
                ])
                centroids_path.append(new_centroids.copy())
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

        except Exception:
            centroids_path = [self.model.cluster_centers_.copy()]

        self.history["centroids_path"] = centroids_path
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y=None):
        """计算轮廓系数"""
        if len(set(self.labels_)) > 1:
            return silhouette_score(X, self.labels_)
        return 0.0

    def get_params(self):
        return {
            "n_clusters": self.n_clusters,
            "inertia": self.model.inertia_,
            "n_iter": self.model.n_iter_,
            "cluster_centers": self.model.cluster_centers_,
            "labels": self.labels_,
        }


class DBSCANModel:
    """DBSCAN 密度聚类模型"""

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = None
        self.labels_ = None
        self.history = {"n_clusters": 0, "n_noise": 0}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        self.model = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric=self.metric
        )
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.history["n_clusters"] = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.history["n_noise"] = list(self.labels_).count(-1)
        # 为 eps 分析计算不同 eps 值下的聚类数
        eps_range = []
        cluster_counts = []
        base_eps = self.eps
        eps_values = [base_eps * (0.2 + 0.3 * i) for i in range(6)]
        for eps_val in eps_values:
            try:
                db = DBSCAN(eps=eps_val, min_samples=self.min_samples, metric=self.metric)
                db.fit(X)
                n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                eps_range.append(eps_val)
                cluster_counts.append(n_cl)
            except Exception:
                pass
        self.history["eps_range"] = eps_range
        self.history["cluster_counts"] = cluster_counts
        return self

    def predict(self, X):
        return self.model.fit_predict(X)

    def score(self, X, y=None):
        """计算轮廓系数"""
        if len(set(self.labels_)) > 1 and len(set(self.labels_)) < len(X):
            return silhouette_score(X, self.labels_)
        return 0.0

    def get_params(self):
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": self.history.get("n_clusters", 0),
            "n_noise": self.history.get("n_noise", 0),
            "labels": self.labels_,
        }


class PCAModel:
    """PCA 降维模型"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.history = {"explained_variance_ratio": [], "cumulative_variance": []}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        self.model = PCA(n_components=self.n_components)
        self.model.fit(X)
        self.history["explained_variance_ratio"] = self.model.explained_variance_ratio_.tolist()
        self.history["cumulative_variance"] = np.cumsum(self.model.explained_variance_ratio_).tolist()
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return self.transform(X)

    def score(self, X, y=None):
        """计算解释方差比例"""
        return sum(self.model.explained_variance_ratio_)

    def get_params(self):
        return {
            "n_components": self.n_components,
            "explained_variance_ratio": self.model.explained_variance_ratio_,
            "cumulative_variance": self.history.get("cumulative_variance", []),
            "singular_values": self.model.singular_values_,
        }


class HierarchicalModel:
    """层次聚类模型"""

    def __init__(self, n_clusters=3, linkage="ward", affinity="euclidean"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.model = None
        self.labels_ = None
        self.history = {"n_leaves": 0, "n_connected_components": 0}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        # sklearn >= 1.2 removed 'affinity' param; use 'metric' instead
        kwargs = {"n_clusters": self.n_clusters, "linkage": self.linkage}
        if self.linkage != "ward":
            kwargs["metric"] = self.affinity
        self.model = AgglomerativeClustering(**kwargs)
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.history["n_leaves"] = self.model.n_leaves_
        self.history["n_connected_components"] = self.model.n_connected_components_
        return self

    def predict(self, X):
        return self.model.fit_predict(X)

    def score(self, X, y=None):
        """计算轮廓系数"""
        if len(set(self.labels_)) > 1:
            return silhouette_score(X, self.labels_)
        return 0.0

    def get_params(self):
        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "n_leaves": self.model.n_leaves_,
            "n_connected_components": self.model.n_connected_components_,
            "labels": self.labels_,
        }


# ═══════════════════════════════════════════════════════════════
# 回归算法扩展（v3.1 新增）
# ═══════════════════════════════════════════════════════════════════════════
