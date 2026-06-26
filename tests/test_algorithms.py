# -*- coding: utf-8 -*-
"""测试算法模块"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.algorithms import (
    create_algorithm, get_algorithm_list, get_algorithm_params,
    get_algorithm_type, ALGORITHM_REGISTRY,
)


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    return X, y


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def clustering_data():
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(30, 2) + [0, 0],
        np.random.randn(30, 2) + [5, 5],
        np.random.randn(30, 2) + [10, 10],
    ])
    return X


class TestGetAlgorithmList:
    def test_all_algorithms(self):
        algos = get_algorithm_list()
        assert len(algos) >= 10

    def test_classification_algorithms(self):
        algos = get_algorithm_list(task_type="classification")
        assert len(algos) > 0
        assert "决策树" in algos or "逻辑回归" in algos

    def test_regression_algorithms(self):
        algos = get_algorithm_list(task_type="regression")
        assert len(algos) > 0

    def test_unsupervised_algorithms(self):
        algos = get_algorithm_list(task_type="unsupervised")
        assert len(algos) > 0


class TestGetAlgorithmParams:
    def test_known_algorithm(self):
        params = get_algorithm_params("决策树")
        assert isinstance(params, dict)

    def test_unknown_algorithm(self):
        params = get_algorithm_params("不存在的算法")
        assert params == {}


class TestGetAlgorithmType:
    def test_known_algorithm(self):
        t = get_algorithm_type("决策树")
        assert t in ("classification", "regression", "unsupervised")

    def test_unknown_algorithm(self):
        t = get_algorithm_type("不存在的算法")
        assert t == "classification"


class TestCreateAlgorithm:
    def test_decision_tree(self, classification_data):
        X, y = classification_data
        algo = create_algorithm("决策树", max_depth=3)
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_logistic_regression(self, classification_data):
        X, y = classification_data
        algo = create_algorithm("逻辑回归")
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_knn(self, classification_data):
        X, y = classification_data
        algo = create_algorithm("K近邻", n_neighbors=5)
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_svm(self, classification_data):
        X, y = classification_data
        algo = create_algorithm("SVM")
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_naive_bayes(self, classification_data):
        X, y = classification_data
        algo = create_algorithm("朴素贝叶斯")
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_random_forest(self, classification_data):
        X, y = classification_data
        algo = create_algorithm("随机森林", n_estimators=10)
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_gbdt(self, classification_data):
        X, y = classification_data
        algo = create_algorithm("梯度提升 (GBDT)", n_estimators=10)
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_linear_regression(self, regression_data):
        X, y = regression_data
        algo = create_algorithm("线性回归")
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_ridge(self, regression_data):
        X, y = regression_data
        algo = create_algorithm("Ridge 回归")
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_lasso(self, regression_data):
        X, y = regression_data
        algo = create_algorithm("Lasso 回归")
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_decision_tree_regression(self, regression_data):
        X, y = regression_data
        algo = create_algorithm("决策树回归")
        algo.fit(X, y)
        preds = algo.predict(X)
        assert len(preds) == len(y)

    def test_kmeans(self, clustering_data):
        X = clustering_data
        algo = create_algorithm("K-Means", n_clusters=3)
        algo.fit(X)
        labels = algo.labels_
        assert len(labels) == len(X)

    def test_dbscan(self, clustering_data):
        X = clustering_data
        algo = create_algorithm("DBSCAN", eps=1.0, min_samples=3)
        algo.fit(X)
        labels = algo.labels_
        assert len(labels) == len(X)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="未知算法"):
            create_algorithm("不存在的算法")

    def test_hidden_layer_sizes_string(self):
        algo = create_algorithm("神经网络", hidden_layer_sizes="64,32")
        assert algo.hidden_layer_sizes == (64, 32)

    def test_hidden_layer_sizes_tuple(self):
        algo = create_algorithm("神经网络", hidden_layer_sizes=(64, 32))
        assert algo.hidden_layer_sizes == (64, 32)


class TestAlgorithmRegistry:
    def test_all_entries_have_required_keys(self):
        for name, info in ALGORITHM_REGISTRY.items():
            assert "class" in info, f"{name} missing 'class'"
            assert "type" in info, f"{name} missing 'type'"
            assert "params" in info, f"{name} missing 'params'"

    def test_all_types_valid(self):
        valid_types = {"classification", "regression", "unsupervised"}
        for name, info in ALGORITHM_REGISTRY.items():
            assert info["type"] in valid_types, f"{name} has invalid type: {info['type']}"
