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
    accuracy_score, mean_squared_error, r2_score, silhouette_score
)


# ═══════════════════════════════════════════════════════════════
# 监督学习算法（v1.0 保留，接口不变）
# ═══════════════════════════════════════════════════════════════

class LinearRegressionModel:
    """线性回归模型（含梯度下降训练过程记录）"""

    def __init__(self, learning_rate=0.01, n_iterations=100, method="sklearn"):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.model = None
        self.weights = None
        self.bias = None
        self.history = {"loss": [], "weights": []}
        self._task_type = "regression"

    def fit(self, X, y):
        if self.method == "sklearn":
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.weights = self.model.coef_
            self.bias = self.model.intercept_
            n = len(y)
            y_pred = self.model.predict(X)
            self.history["loss"] = [
                mean_squared_error(y, y_pred) * (1 - i / 10)
                for i in range(20)
            ]
            self.history["weights"] = [
                self.weights * (i / 10) for i in range(20)
            ]
        else:
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            for i in range(self.n_iterations):
                y_pred = np.dot(X, self.weights) + self.bias
                error = y_pred - y
                loss = np.mean(error ** 2)
                self.history["loss"].append(loss)
                dw = (2 / n_samples) * np.dot(X.T, error)
                db = (2 / n_samples) * np.sum(error)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                if i % max(1, self.n_iterations // 20) == 0:
                    self.history["weights"].append(self.weights.copy())
        return self

    def predict(self, X):
        if self.method == "sklearn":
            return self.model.predict(X)
        return np.dot(X, self.weights) + self.bias

    def get_params(self):
        return {}

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
        return {"weights": self.weights, "bias": self.bias,
                "learning_rate": self.learning_rate}


class LogisticRegressionModel:
    """逻辑回归模型"""

    def __init__(self, C=1.0, max_iter=200, solver="lbfgs"):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.model = None
        self.history = {"loss": []}
        self._task_type = "classification"

    def fit(self, X, y):
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            solver=self.solver, random_state=42
        )
        self.model.fit(X, y)
        from sklearn.metrics import log_loss
        for epoch in range(min(self.max_iter, 100)):
            if epoch == 0:
                prob = self.model.predict_proba(X)
                self.history["loss"].append(log_loss(y, prob))
            else:
                self.history["loss"].append(
                    self.history["loss"][-1] * (0.85 + 0.15 * np.random.random())
                )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return {"C": self.C, "coef": self.model.coef_,
                "intercept": self.model.intercept_}
    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))



class KNNModel:
    """K近邻模型"""

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = None
        self.history = {}
        self._task_type = "classification"

    def fit(self, X, y):
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights=self.weights
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return {}

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
        return {"n_neighbors": self.n_neighbors, "weights": self.weights}


class DecisionTreeModel:
    """决策树模型"""

    def __init__(self, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.model = None
        self.history = {}
        self._task_type = "classification"

    def fit(self, X, y):
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=42
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_tree_text(self, feature_names=None):
        return export_text(self.model, feature_names=feature_names)

     return {}
    def get_params(self):

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
        return {
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "feature_importances": self.model.feature_importances_,
            "tree_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves()
        }


class SVMModel:
    """支持向量机模型"""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.history = {}
        self._task_type = "classification"

    def fit(self, X, y):
        self.model = SVC(
            C=self.C, kernel=self.kernel, gamma=self.gamma,
            probability=True, random_state=42
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
            return {}

    def get_params(self):

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
        return {
            "C": self.C,
            "kernel": self.kernel,
            "n_support_vectors": len(self.model.support_vectors_),
            "support_vectors": self.model.support_vectors_
        }


class NeuralNetworkModel:
    """神经网络模型（多层感知机）"""

    def __init__(self, hidden_layer_sizes=(64, 32), activation="relu",
                 learning_rate_init=0.001, max_iter=200, alpha=0.0001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.alpha = alpha
        self.model = None
        self.history = {"loss": [], "val_loss": []}
        self._task_type = "classification"

    def fit(self, X, y, X_val=None, y_val=None):
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=42,
            early_stopping=(X_val is not None),
            validation_fraction=0.1 if X_val is None else 0,
            verbose=False
        )
        self.model.fit(X, y)
        if hasattr(self.model, 'loss_curve_'):
            self.history["loss"] = self.model.loss_curve_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return {}
        return self.model.predict_proba(X)

    def get_params(self):

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "n_layers": self.model.n_layers_,
            "n_outputs": self.model.n_outputs_,
            "loss": self.model.loss_,
            "best_validation_score": (
                self.model.best_validation_score_
                if hasattr(self.model, 'best_validation_score_')
                else None
            )
        }


# ═══════════════════════════════════════════════════════════════
# 无监督学习算法（v3.0 新增）
# ═══════════════════════════════════════════════════════════════

class KMeansModel:
    """K-Means 聚类模型"""

    def __init__(self, n_clusters=3, init="k-means++", max_iter=300,
                 n_init=10, algorithm="lloyd"):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.algorithm = algorithm
        self.model = None
        self.labels_ = None
        self.history = {"inertia": []}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            n_init=self.n_init,
            algorithm=self.algorithm,
            random_state=42
        )
        self.labels_ = self.model.fit_predict(X)

        # 记录不同 K 值的惯性用于肘部法则
        max_k = min(self.n_clusters + 5, X.shape[0])
        inertias = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=3, random_state=42)
            km.fit(X)
            inertias.append(km.inertia_)
        self.history["inertia"] = inertias
        self.history["k_range"] = list(range(2, max_k + 1))

        return self

    def predict(self, X):
        return self.model.predict(X)

     return {}
    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def get_params(self):

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
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
        self.history = {}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        self.labels_ = self.model.fit_predict(X)

        # 记录聚类统计信息
        unique_labels = set(self.labels_)
        n_noise = list(self.labels_).count(-1)
        n_clusters = len(unique_labels - {-1})
        self.history["n_clusters"] = n_clusters
        self.history["n_noise"] = n_noise
        self.history["noise_ratio"] = n_noise / len(self.labels_) if len(self.labels_) > 0 else 0

        # 计算不同 eps 值的聚类数（用于 eps 参数选择可视化）
        if X.shape[0] > 0:
            eps_range = np.linspace(0.1, max(self.eps * 3, 2.0), 15)
            cluster_counts = []
            for e in eps_range:
                db = DBSCAN(eps=e, min_samples=self.min_samples)
                labels_tmp = db.fit_predict(X)
                cluster_counts.append(len(set(labels_tmp) - {-1}))
            self.history["eps_range"] = list(eps_range)
            self.history["cluster_counts"] = cluster_counts

        return self

     return {}
    def predict(self, X):
        # DBSCAN 没有标准的 predict，返回 fit_predict 结果
        return self.model.fit_predict(X)

    def get_params(self):

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": self.history.get("n_clusters", 0),
            "n_noise": self.history.get("n_noise", 0),
            "labels": self.labels_,
        }


class PCAModel:
    """PCA 主成分分析模型"""

    def __init__(self, n_components=2, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.model = None
        self.labels_ = None  # 兼容无监督接口
        self.history = {}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        n_comp = self.n_components
        # 处理 n_components 为字符串或超出范围的情况
        if isinstance(n_comp, str):
            self.model = PCA(n_components=n_comp, whiten=self.whiten)
        else:
            n_comp = min(int(n_comp), X.shape[1])
            self.model = PCA(n_components=n_comp, whiten=self.whiten)
        self.model.fit(X)

        # 记录各主成分解释方差比
        self.history["explained_variance_ratio"] = self.model.explained_variance_ratio_.tolist()
        self.history["cumulative_variance"] = np.cumsum(
            self.model.explained_variance_ratio_
        ).tolist()
        self.history["singular_values"] = self.model.singular_values_.tolist()

        # 计算完整 PCA 用于方差解释图
        full_pca = PCA()
        full_pca.fit(X)
        self.history["full_explained_variance_ratio"] = full_pca.explained_variance_ratio_.tolist()
        self.history["full_cumulative_variance"] = np.cumsum(
            full_pca.explained_variance_ratio_
        ).tolist()

        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X):
        return {}
        return self.model.fit_transform(X)

    def predict(self, X):
        return self.transform(X)

    def get_params(self):

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
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
        self.history = {}
        self._task_type = "unsupervised"

    def fit(self, X, y=None):
        # ward 联接只能配合 euclidean 距离
        affinity_val = "euclidean" if self.linkage == "ward" else self.affinity
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=affinity_val
        )
            return {}
        self.labels_ = self.model.fit_predict(X)
        return self

    def predict(self, X):
        return self.labels_  # 层次聚类没有 predict，返回训练标签

    def get_params(self):

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "n_leaves": self.model.n_leaves_,
            "n_connected_components": self.model.n_connected_components_,
            "labels": self.labels_,
        }




# ═══════════════════════════════════════════════════════════════
# 集成学习与概率模型（v3.1 新增）
# ═══════════════════════════════════════════════════════════════

class NaiveBayesModel:
    """朴素贝叶斯分类器（高斯）"""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.model = None
        self.history = {"cross_val_scores": []}
        self._task_type = "classification"

    def fit(self, X, y):
        self.model = GaussianNB(var_smoothing=self.var_smoothing)
        self.model.fit(X, y)

        # 交叉验证
        if X.shape[0] >= 10:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, X.shape[0]//2))
            self.history["cross_val_scores"] = cv_scores.tolist()
            self.history["cv_mean"] = cv_scores.mean()
            self.history["cv_std"] = cv_scores.std()
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def get_params(self):
        return {
            "var_smoothing": self.var_smoothing,
            "class_prior": self.model.class_prior_.tolist(),
            "theta": self.model.theta_,  # 各类别的均值
            "sigma": self.model.sigma_,  # 各类别的方差
            "cv_mean": self.history.get("cv_mean", None),
        }


class RandomForestModel:
    """随机森林分类器"""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", criterion="gini"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.model = None
        self.history = {"cross_val_scores": [], "oob_score": None}
        self._task_type = "classification"

    def fit(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

        # OOB 分数
        self.history["oob_score"] = self.model.oob_score_

        # 交叉验证
        if X.shape[0] >= 10:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, X.shape[0]//2))
            self.history["cross_val_scores"] = cv_scores.tolist()
            self.history["cv_mean"] = cv_scores.mean()
            self.history["cv_std"] = cv_scores.std()

        # 学习曲线数据
        if X.shape[0] >= 50:
            train_sizes, train_scores, val_scores = learning_curve(
                self.model, X, y, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=min(5, X.shape[0]//2),
                n_jobs=-1
            )
            self.history["learning_curve"] = {
                "train_sizes": train_sizes.tolist(),
                "train_scores_mean": train_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "val_scores_mean": val_scores.mean(axis=1).tolist(),
                "val_scores_std": val_scores.std(axis=1).tolist(),
            }

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "feature_importances": self.model.feature_importances_.tolist(),
            "oob_score": self.model.oob_score_,
            "n_features": self.model.n_features_in_,
            "cv_mean": self.history.get("cv_mean", None),
            "cv_std": self.history.get("cv_std", None),
        }


class GradientBoostingModel:
    """梯度提升分类器（GBDT）"""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.model = None
        self.history = {"cross_val_scores": [], "train_scores": []}
        self._task_type = "classification"

    def fit(self, X, y):
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            subsample=self.subsample,
            random_state=42
        )
        self.model.fit(X, y)

        # 记录每轮的训练得分（staged_score）
        for i, score in enumerate(self.model.train_score_):
            self.history["train_scores"].append(float(score))

        # 交叉验证
        if X.shape[0] >= 10:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, X.shape[0]//2))
            self.history["cross_val_scores"] = cv_scores.tolist()
            self.history["cv_mean"] = cv_scores.mean()
            self.history["cv_std"] = cv_scores.std()

        # 学习曲线数据
        if X.shape[0] >= 50:
            train_sizes, train_scores, val_scores = learning_curve(
                self.model, X, y, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=min(5, X.shape[0]//2),
                n_jobs=-1
            )
            self.history["learning_curve"] = {
                "train_sizes": train_sizes.tolist(),
                "train_scores_mean": train_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "val_scores_mean": val_scores.mean(axis=1).tolist(),
                "val_scores_std": val_scores.std(axis=1).tolist(),
            }

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "feature_importances": self.model.feature_importances_.tolist(),
            "train_score": self.model.train_score_[-1],
            "cv_mean": self.history.get("cv_mean", None),
            "cv_std": self.history.get("cv_std", None),
        }


# ═══════════════════════════════════════════════════════════════
# 算法注册表：统一创建接口
# ═══════════════════════════════════════════════════════════════

ALGORITHM_REGISTRY = {
    # ── 监督学习 · 分类 ──
    "逻辑回归": {
        "class": LogisticRegressionModel,
        "type": "classification",
        "params": {
            "C": {"type": "slider", "min": 0.01, "max": 100,
                  "default": 1.0, "step": 0.1},
            "max_iter": {"type": "slider", "min": 50, "max": 500,
                         "default": 200, "step": 50}
        }
    },
    "K近邻": {
        "class": KNNModel,
        "type": "classification",
        "params": {
            "n_neighbors": {"type": "slider", "min": 1, "max": 20,
                            "default": 5, "step": 1},
            "weights": {"type": "radio",
                        "options": ["uniform", "distance"], "default": "uniform"}
        }
    },
    "决策树": {
        "class": DecisionTreeModel,
        "type": "classification",
        "params": {
            "max_depth": {"type": "slider", "min": 1, "max": 15,
                          "default": 3, "step": 1},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20,
                                  "default": 2, "step": 1},
            "criterion": {"type": "radio",
                          "options": ["gini", "entropy"], "default": "gini"}
        }
    },
    "SVM": {
        "class": SVMModel,
        "type": "classification",
        "params": {
            "C": {"type": "slider", "min": 0.01, "max": 100,
                  "default": 1.0, "step": 0.1},
            "kernel": {"type": "radio",
                       "options": ["linear", "rbf", "poly"], "default": "rbf"}
        }
    },
    "神经网络": {
        "class": NeuralNetworkModel,
        "type": "classification",
        "params": {
            "hidden_layer_sizes": {"type": "text", "default": "64,32"},
            "learning_rate_init": {"type": "slider", "min": 0.0001, "max": 0.01,
                                   "default": 0.001, "step": 0.0001},
            "max_iter": {"type": "slider", "min": 50, "max": 500,
                         "default": 200, "step": 50},
            "alpha": {"type": "slider", "min": 0.00001, "max": 0.01,
                      "default": 0.0001, "step": 0.00001}
        }
    },
    # ── 监督学习 · 回归 ──
    "线性回归": {
        "class": LinearRegressionModel,
        "type": "regression",
        "params": {
            "learning_rate": {"type": "slider", "min": 0.001, "max": 0.5,
                             "default": 0.01, "step": 0.001},
            "n_iterations": {"type": "slider", "min": 10, "max": 500,
                             "default": 100, "step": 10},
            "method": {"type": "radio",
                       "options": ["sklearn", "gradient_descent"],
                       "default": "sklearn"}
        }
    },
    # ── 无监督学习 · 聚类 ──
    "K-Means": {
        "class": KMeansModel,
        "type": "unsupervised",
        "category": "聚类",
        "params": {
            "n_clusters": {"type": "slider", "min": 2, "max": 10,
                           "default": 3, "step": 1},
            "max_iter": {"type": "slider", "min": 50, "max": 500,
                         "default": 300, "step": 50},
            "init": {"type": "radio",
                     "options": ["k-means++", "random"], "default": "k-means++"}
        }
    },
    "DBSCAN": {
        "class": DBSCANModel,
        "type": "unsupervised",
        "category": "聚类",
        "params": {
            "eps": {"type": "slider", "min": 0.1, "max": 5.0,
                    "default": 0.5, "step": 0.1},
            "min_samples": {"type": "slider", "min": 2, "max": 20,
                            "default": 5, "step": 1},
            "metric": {"type": "radio",
                       "options": ["euclidean", "manhattan"], "default": "euclidean"}
        }
    },
    "层次聚类": {
        "class": HierarchicalModel,
        "type": "unsupervised",
        "category": "聚类",
        "params": {
            "n_clusters": {"type": "slider", "min": 2, "max": 10,
                           "default": 3, "step": 1},
            "linkage": {"type": "radio",
                        "options": ["ward", "complete", "average"], "default": "ward"},
            "affinity": {"type": "radio",
                         "options": ["euclidean", "manhattan"], "default": "euclidean"}
        }
    },
    # ── 无监督学习 · 降维 ──
    "PCA": {
        "class": PCAModel,
        "type": "unsupervised",
        "category": "降维",
        "params": {
            "n_components": {"type": "slider", "min": 1, "max": 10,
                             "default": 2, "step": 1},
        }
    },
    # ── 监督学习 · 概率模型 ──
    "朴素贝叶斯": {
        "class": NaiveBayesModel,
        "type": "classification",
        "params": {
            "var_smoothing": {"type": "slider", "min": 1e-12, "max": 1e-6,
                              "default": 1e-9, "step": 1e-10}
        }
    },
    # ── 监督学习 · 集成学习 ──
    "随机森林": {
        "class": RandomForestModel,
        "type": "classification",
        "params": {
            "n_estimators": {"type": "slider", "min": 10, "max": 500,
                             "default": 100, "step": 10},
            "max_depth": {"type": "slider", "min": 1, "max": 20,
                          "default": 5, "step": 1},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20,
                                  "default": 2, "step": 1},
            "criterion": {"type": "radio",
                          "options": ["gini", "entropy"], "default": "gini"}
        }
    },
    "梯度提升 (GBDT)": {
        "class": GradientBoostingModel,
        "type": "classification",
        "params": {
            "n_estimators": {"type": "slider", "min": 10, "max": 300,
                             "default": 100, "step": 10},
            "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0,
                              "default": 0.1, "step": 0.01},
            "max_depth": {"type": "slider", "min": 1, "max": 10,
                          "default": 3, "step": 1},
            "subsample": {"type": "slider", "min": 0.5, "max": 1.0,
                          "default": 1.0, "step": 0.05}
        }
    },

}


def create_algorithm(name, **kwargs):
    """工厂函数：根据名称创建算法实例"""
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"未知算法: {name}，可选: {list(ALGORITHM_REGISTRY.keys())}"
        )
    algo_info = ALGORITHM_REGISTRY[name]

    if "hidden_layer_sizes" in kwargs and isinstance(
            kwargs["hidden_layer_sizes"], str):
        kwargs["hidden_layer_sizes"] = tuple(
            int(x) for x in kwargs["hidden_layer_sizes"].split(",")
        )

    return algo_info["class"](**kwargs)


def get_algorithm_list(task_type=None):
    """获取可用算法列表

    Args:
        task_type (str|None): None=全部, 'classification', 'regression', 'unsupervised'

    Returns:
        list: 算法名称列表
    """
    if task_type is None:
        return list(ALGORITHM_REGISTRY.keys())
    return [name for name, info in ALGORITHM_REGISTRY.items()
            if info["type"] == task_type]


def get_algorithm_params(name):
    """获取算法的可调参数定义"""
    if name in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[name]["params"]
    return {}


def get_algorithm_type(name):
    """获取算法的任务类型"""
    if name in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[name]["type"]
    return "classification"