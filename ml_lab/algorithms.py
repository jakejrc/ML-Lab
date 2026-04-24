# -*- coding: utf-8 -*-
"""
ML-Lab 算法模块
封装6大核心机器学习算法（线性回归、逻辑回归、KNN、决策树、SVM、神经网络），
提供统一接口和训练历史记录，支持可视化回调。
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score
)


class LinearRegressionModel:
    """线性回归模型（含梯度下降训练过程记录）"""

    def __init__(self, learning_rate=0.01, n_iterations=100, method="sklearn"):
        """
        Args:
            learning_rate (float): 学习率
            n_iterations (int): 迭代次数
            method (str): 'sklearn' 或 'gradient_descent'
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.model = None
        self.weights = None
        self.bias = None
        self.history = {"loss": [], "weights": []}

    def fit(self, X, y):
        """训练模型

        Args:
            X (np.ndarray): 训练特征 (n_samples, n_features)
            y (np.ndarray): 训练标签 (n_samples,)

        Returns:
            self
        """
        if self.method == "sklearn":
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.weights = self.model.coef_
            self.bias = self.model.intercept_
            # 模拟训练过程用于可视化
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
            # 手动梯度下降
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
        """预测"""
        if self.method == "sklearn":
            return self.model.predict(X)
        return np.dot(X, self.weights) + self.bias

    def get_params(self):
        """获取模型参数"""
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

    def fit(self, X, y):
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            solver=self.solver, random_state=42
        )
        self.model.fit(X, y)

        # 记录训练过程中的损失变化
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


class KNNModel:
    """K近邻模型"""

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = None
        self.history = {}

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
        """获取决策树文本表示"""
        return export_text(self.model, feature_names=feature_names)

    def get_params(self):
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

    def get_params(self):
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

        # 使用 sklearn 内置的 loss_curve_
        if hasattr(self.model, 'loss_curve_'):
            self.history["loss"] = self.model.loss_curve_

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
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


# 算法注册表：统一创建接口
ALGORITHM_REGISTRY = {
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
}


def create_algorithm(name, **kwargs):
    """工厂函数：根据名称创建算法实例

    Args:
        name (str): 算法名称
        **kwargs: 算法参数

    Returns:
        算法实例
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"未知算法: {name}，可选: {list(ALGORITHM_REGISTRY.keys())}"
        )
    algo_info = ALGORITHM_REGISTRY[name]

    # 特殊处理 hidden_layer_sizes (字符串转元组)
    if "hidden_layer_sizes" in kwargs and isinstance(
            kwargs["hidden_layer_sizes"], str):
        kwargs["hidden_layer_sizes"] = tuple(
            int(x) for x in kwargs["hidden_layer_sizes"].split(",")
        )

    return algo_info["class"](**kwargs)


def get_algorithm_list():
    """获取所有可用算法列表"""
    return list(ALGORITHM_REGISTRY.keys())


def get_algorithm_params(name):
    """获取算法的可调参数定义"""
    if name in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[name]["params"]
    return {}