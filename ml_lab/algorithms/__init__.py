# -*- coding: utf-8 -*-
"""
ML-Lab 算法模块 v3.0 (Package)
拆分自 algorithms.py，按任务类型分模块：
  - classification.py: 分类模型
  - regression.py: 回归模型
  - unsupervised.py: 无监督模型（聚类/降维）
"""

from ml_lab.algorithms.classification import (
    LinearRegressionModel, LogisticRegressionModel, KNNModel,
    DecisionTreeModel, SVMModel, NeuralNetworkModel,
    NaiveBayesModel, RandomForestModel, GradientBoostingModel,
)
from ml_lab.algorithms.regression import (
    RidgeRegressionModel, LassoRegressionModel, PolynomialRegressionModel,
    SVRModel, DecisionTreeRegressorModel,
)
from ml_lab.algorithms.unsupervised import (
    KMeansModel, DBSCANModel, PCAModel, HierarchicalModel,
)
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
    "Ridge 回归": {
        "class": RidgeRegressionModel,
        "type": "regression",
        "params": {
            "alpha": {"type": "slider", "min": 0.01, "max": 100,
                      "default": 1.0, "step": 0.1}
        }
    },
    "Lasso 回归": {
        "class": LassoRegressionModel,
        "type": "regression",
        "params": {
            "alpha": {"type": "slider", "min": 0.01, "max": 100,
                      "default": 1.0, "step": 0.1},
            "max_iter": {"type": "slider", "min": 100, "max": 5000,
                         "default": 1000, "step": 100}
        }
    },
    "多项式回归": {
        "class": PolynomialRegressionModel,
        "type": "regression",
        "params": {
            "degree": {"type": "slider", "min": 2, "max": 6,
                       "default": 2, "step": 1}
        }
    },
    "SVR 回归": {
        "class": SVRModel,
        "type": "regression",
        "params": {
            "C": {"type": "slider", "min": 0.01, "max": 100,
                  "default": 1.0, "step": 0.1},
            "kernel": {"type": "radio",
                       "options": ["linear", "rbf", "poly"], "default": "rbf"},
            "epsilon": {"type": "slider", "min": 0.01, "max": 1.0,
                        "default": 0.1, "step": 0.01}
        }
    },
    "决策树回归": {
        "class": DecisionTreeRegressorModel,
        "type": "regression",
        "params": {
            "max_depth": {"type": "slider", "min": 1, "max": 15,
                          "default": 5, "step": 1},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20,
                                  "default": 2, "step": 1},
            "criterion": {"type": "radio",
                          "options": ["squared_error", "absolute_error"],
                          "default": "squared_error"}
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
