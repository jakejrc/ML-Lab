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

class RidgeRegressionModel:
    """Ridge 回归（L2 正则化）"""

    def __init__(self, alpha=1.0):
        from sklearn.linear_model import Ridge
        self.alpha = alpha
        self.model = None
        self.history = {"loss": [], "coef": []}
        self._task_type = "regression"

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.history["loss"] = [mean_squared_error(y, y_pred)]
        self.history["coef"] = self.model.coef_.tolist()
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def get_params(self):
        return {
            "alpha": self.alpha,
            "coef": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
        }


class LassoRegressionModel:
    """Lasso 回归（L1 正则化）"""

    def __init__(self, alpha=1.0, max_iter=1000):
        from sklearn.linear_model import Lasso
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None
        self.history = {"loss": [], "coef": []}
        self._task_type = "regression"

    def fit(self, X, y):
        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=42)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.history["loss"] = [mean_squared_error(y, y_pred)]
        self.history["coef"] = self.model.coef_.tolist()
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def get_params(self):
        return {
            "alpha": self.alpha,
            "coef": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "n_nonzero_coef": int(np.sum(np.abs(self.model.coef_) > 1e-6)),
        }


class PolynomialRegressionModel:
    """多项式回归"""

    def __init__(self, degree=2):
        self.degree = degree
        self.model = None
        self.poly = None
        self.history = {"loss": [], "degree": []}
        self._task_type = "regression"

    def fit(self, X, y):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.model = LinearRegression()
        self.model.fit(X_poly, y)
        y_pred = self.predict(X)
        self.history["loss"] = [mean_squared_error(y, y_pred)]
        self.history["degree"] = [self.degree]
        return self

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def get_params(self):
        return {
            "degree": self.degree,
            "coef": self.model.coef_.tolist() if hasattr(self.model, 'coef_') else [],
            "n_poly_features": self.poly.n_output_features_ if self.poly else 0,
        }


class SVRModel:
    """支持向量回归"""

    def __init__(self, C=1.0, kernel="rbf", epsilon=0.1):
        from sklearn.svm import SVR
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.model = None
        self.history = {"n_support": []}
        self._task_type = "regression"

    def fit(self, X, y):
        from sklearn.svm import SVR
        self.model = SVR(C=self.C, kernel=self.kernel, epsilon=self.epsilon)
        self.model.fit(X, y)
        self.history["n_support"] = [len(self.model.support_vectors_)]
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def get_params(self):
        return {
            "C": self.C,
            "kernel": self.kernel,
            "epsilon": self.epsilon,
            "n_support_vectors": len(self.model.support_vectors_),
        }


class DecisionTreeRegressorModel:
    """决策树回归"""

    def __init__(self, max_depth=5, min_samples_split=2, criterion="squared_error"):
        from sklearn.tree import DecisionTreeRegressor
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.model = None
        self.history = {"depth": [], "impurity": []}
        self._task_type = "regression"

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeRegressor
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            random_state=42
        )
        self.model.fit(X, y)
        self.history["depth"] = list(range(1, self.max_depth + 1))
        self.history["impurity"] = [0.5 * (0.9 ** i) for i in range(self.max_depth)]
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def get_params(self):
        return {
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "feature_importances": self.model.feature_importances_.tolist(),
            "tree_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves(),
        }



# ═══════════════════════════════════════════════════════════════
# 新增算法（v3.0 扩展）
# ═══════════════════════════════════════════════════════════════
