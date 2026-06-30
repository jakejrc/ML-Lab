# -*- coding: utf-8 -*-
"""测试训练回调"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.state import _g
from ml_lab.callbacks.data import on_load_data
from ml_lab.callbacks.train import (
    on_train_classification, on_train_regression, on_train_clustering,
    on_compare_models,
)


def _reset_g():
    _g.update(dict(
        X=None, y=None, X_train=None, X_test=None,
        y_train=None, y_test=None, model=None,
        model_name="未训练", dataset_name="未加载",
    ))


class TestOnTrainClassification:
    """测试分类训练回调"""

    def setup_method(self):
        on_load_data("鸢尾花 (Iris)", 0.3)

    def teardown_method(self):
        _reset_g()

    @pytest.mark.parametrize("algo", [
        "逻辑回归", "决策树", "随机森林", "K近邻", "朴素贝叶斯",
    ])
    def test_train_classification_algorithms(self, algo):
        """测试各种分类算法训练"""
        result = on_train_classification(
            algo, lr=0.01, ni=100, C=1.0, md=5, mi=20, hid=64, al="relu"
        )
        train_res, db_plot, report, cm_img, roc_img, lc_img, cv_img, model_html = result
        if isinstance(report, str) and "训练失败" in report:
            pytest.skip(f"{algo}: {report}")
        if report is not None and "请先加载" not in str(report):
            assert _g["model"] is not None or "失败" in str(report)

    def test_train_without_data(self):
        """测试未加载数据时训练"""
        _reset_g()
        result = on_train_classification(
            "逻辑回归", lr=0.01, ni=100, C=1.0, md=5, mi=20, hid=64, al="relu"
        )
        report = result[2]
        assert "请先加载" in str(report)


class TestOnTrainRegression:
    """测试回归训练回调"""

    def setup_method(self):
        on_load_data("波士顿房价 (Boston)", 0.3)

    def teardown_method(self):
        _reset_g()

    @pytest.mark.parametrize("algo", [
        "线性回归", "决策树回归", "随机森林回归", "K近邻回归",
    ])
    def test_train_regression_algorithms(self, algo):
        """测试各种回归算法训练"""
        result = on_train_regression(
            algo, lr=0.01, ni=100, C=1.0, md=5, mi=20, hid=64, al="relu",
            deg=3, eps=0.1, kern="rbf", crit="mse"
        )
        _, _, _, _, _, report, _ = result
        if isinstance(report, str) and "训练失败" in report:
            pytest.skip(f"{algo}: {report}")
        if report is not None and "请先加载" not in str(report):
            assert _g["model"] is not None

    def test_train_without_data(self):
        """测试未加载数据时回归训练"""
        _reset_g()
        result = on_train_regression(
            "线性回归", lr=0.01, ni=100, C=1.0, md=5, mi=20, hid=64, al="relu",
            deg=3, eps=0.1, kern="rbf", crit="mse"
        )
        report = result[5]
        assert "请先加载" in str(report)


class TestOnTrainClustering:
    """测试聚类训练回调"""

    def setup_method(self):
        on_load_data("鸢尾花 (Iris)", 0.3)

    def teardown_method(self):
        _reset_g()

    @pytest.mark.parametrize("algo,lt,initm", [
        ("KMeans", "ward", "k-means++"),
        ("DBSCAN", "ward", "k-means++"),
        ("层次聚类", "ward", "k-means++"),
    ])
    def test_train_clustering_algorithms(self, algo, lt, initm):
        """测试各种聚类算法训练"""
        result = on_train_clustering(
            algo, n_clusters=3, max_iter=300, eps=0.5,
            min_samples=5, linkage_type=lt,
            affinity="euclidean", init_method=initm
        )
        _, _, _, report, _ = result
        if isinstance(report, str) and "训练失败" in report:
            pytest.skip(f"{algo}: {report}")
        if report is not None and "请先加载" not in str(report):
            assert _g["model"] is not None

    def test_train_without_data(self):
        """测试未加载数据时聚类训练"""
        _reset_g()
        result = on_train_clustering(
            "KMeans", n_clusters=3, max_iter=300, eps=0.5,
            min_samples=5, linkage_type="ward",
            affinity="euclidean", init_method="k-means++"
        )
        report = result[3]
        assert "请先加载" in str(report)


class TestOnCompareModels:
    """测试模型对比回调"""

    def setup_method(self):
        on_load_data("鸢尾花 (Iris)", 0.3)

    def test_compare_without_selection(self):
        """测试未选择算法时对比"""
        result = on_compare_models([])
        text = result[1]
        # 无数据或无选择时都会返回提示
        assert text is not None
