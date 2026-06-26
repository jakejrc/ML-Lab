# -*- coding: utf-8 -*-
"""测试特征工程模块"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.feature_engineering import (
    scale_data, get_scaling_stats, encode_labels, one_hot_features,
    select_features_univariate, select_features_rfe, select_features_model_based,
    apply_pca, construct_polynomial_features, construct_interaction_features,
    construct_statistical_features, discretize_features, run_feature_pipeline,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    return X, y


class TestScaleData:
    def test_standard(self, sample_data):
        X, _ = sample_data
        Xs, _, scaler = scale_data(X, method="standard")
        assert Xs.shape == X.shape
        assert np.abs(Xs.mean(axis=0)).max() < 1e-10

    def test_minmax(self, sample_data):
        X, _ = sample_data
        Xs, _, _ = scale_data(X, method="minmax")
        assert Xs.min() >= -0.01 and Xs.max() <= 1.01

    def test_robust(self, sample_data):
        X, _ = sample_data
        Xs, _, _ = scale_data(X, method="robust")
        assert Xs.shape == X.shape

    def test_normalizer(self, sample_data):
        X, _ = sample_data
        Xs, _, _ = scale_data(X, method="normalizer")
        norms = np.linalg.norm(Xs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_unknown_method_raises(self, sample_data):
        X, _ = sample_data
        with pytest.raises(ValueError, match="未知缩放方法"):
            scale_data(X, method="unknown")

    def test_with_test_data(self, sample_data):
        X, _ = sample_data
        X_test = np.random.randn(20, 5)
        Xs_train, Xs_test, _ = scale_data(X, X_test, method="standard")
        assert Xs_test is not None
        assert Xs_test.shape == X_test.shape


class TestGetScalingStats:
    def test_returns_list_of_dicts(self, sample_data):
        X, _ = sample_data
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        stats = get_scaling_stats(X, Xs, ["f1", "f2", "f3", "f4", "f5"])
        assert len(stats) == 5
        assert "特征" in stats[0]
        assert "缩放前_均值" in stats[0]


class TestEncodeLabels:
    def test_label_encoding(self):
        y = np.array(["cat", "dog", "bird", "cat"])
        y_enc, enc = encode_labels(y, method="label")
        assert len(np.unique(y_enc)) == 3

    def test_unknown_method_raises(self):
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="未知编码方法"):
            encode_labels(y, method="unknown")


class TestOneHotFeatures:
    def test_all_columns(self):
        X = np.array([[0, 1], [1, 0], [2, 1]])
        X_enc, enc, cols = one_hot_features(X)
        assert X_enc.shape[1] > X.shape[1]

    def test_specific_columns(self):
        X = np.array([[0, 1, 5], [1, 0, 6], [2, 1, 7]], dtype=float)
        X_enc, enc, cols = one_hot_features(X, columns=[0])
        assert X_enc.shape[0] == 3


class TestSelectFeaturesUnivariate:
    def test_f_classif(self, sample_data):
        X, y = sample_data
        X_sel, indices, scores, pvals = select_features_univariate(X, y, method="f_classif", k=3)
        assert X_sel.shape[1] == 3
        assert len(indices) == 3
        assert len(scores) == 5

    def test_f_regression(self, sample_data):
        X, _ = sample_data
        y = np.random.randn(100)
        X_sel, indices, scores, pvals = select_features_univariate(X, y, method="f_regression", k=3)
        assert X_sel.shape[1] == 3

    def test_mutual_info(self, sample_data):
        X, y = sample_data
        X_sel, indices, scores, pvals = select_features_univariate(
            X, y, method="mutual_info_classif", k=2)
        assert X_sel.shape[1] == 2

    def test_unknown_method_raises(self, sample_data):
        X, y = sample_data
        with pytest.raises(ValueError, match="未知方法"):
            select_features_univariate(X, y, method="unknown")

    def test_k_exceeds_features(self, sample_data):
        X, y = sample_data
        X_sel, indices, scores, pvals = select_features_univariate(X, y, k=100)
        assert X_sel.shape[1] == 5  # capped to n_features


class TestSelectFeaturesRFE:
    def test_classification(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)  # binary classification for liblinear
        X_sel, indices, ranking = select_features_rfe(X, y, n_features=3, task_type="classification")
        assert X_sel.shape[1] == 3

    def test_regression(self, sample_data):
        X, _ = sample_data
        y = np.random.randn(100)
        X_sel, indices, ranking = select_features_rfe(X, y, n_features=3, task_type="regression")
        assert X_sel.shape[1] == 3


class TestSelectFeaturesModelBased:
    def test_classification(self, sample_data):
        X, y = sample_data
        X_sel, indices, importances = select_features_model_based(X, y, task_type="classification")
        assert len(importances) == 5

    def test_regression(self, sample_data):
        X, _ = sample_data
        y = np.random.randn(100)
        X_sel, indices, importances = select_features_model_based(X, y, task_type="regression")
        assert len(importances) == 5


class TestApplyPCA:
    def test_basic(self, sample_data):
        X, _ = sample_data
        X_pca, pca, evr = apply_pca(X, n_components=2)
        assert X_pca.shape == (100, 2)
        assert len(evr) == 2
        assert sum(evr) <= 1.0 + 1e-10

    def test_n_exceeds_features(self, sample_data):
        X, _ = sample_data
        X_pca, pca, evr = apply_pca(X, n_components=100)
        assert X_pca.shape[1] == 5  # capped


class TestConstructPolynomialFeatures:
    def test_degree_2(self):
        X = np.random.randn(10, 3)
        X_poly, poly = construct_polynomial_features(X, degree=2)
        assert X_poly.shape[0] == 10
        assert X_poly.shape[1] > X.shape[1]

    def test_degree_1(self):
        X = np.array([[1, 2], [3, 4]])
        X_poly, poly = construct_polynomial_features(X, degree=1)
        assert X_poly.shape[1] == X.shape[1]


class TestConstructInteractionFeatures:
    def test_basic(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_inter, names = construct_interaction_features(X)
        assert X_inter.shape[0] == 2
        assert len(names) == 3  # C(3,2) = 3

    def test_single_feature(self):
        X = np.array([[1], [2], [3]])
        X_inter, names = construct_interaction_features(X)
        assert X_inter.shape[0] == 3


class TestConstructStatisticalFeatures:
    def test_basic(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_enhanced, names = construct_statistical_features(X)
        assert X_enhanced.shape == (2, 9)  # 3 original + 6 new
        assert len(names) == 6


class TestDiscretizeFeatures:
    def test_quantile(self, sample_data):
        X, _ = sample_data
        X_disc, enc = discretize_features(X, n_bins=5, strategy="quantile")
        assert X_disc.shape == X.shape
        assert X_disc.max() <= 4

    def test_uniform(self, sample_data):
        X, _ = sample_data
        X_disc, enc = discretize_features(X, n_bins=5, strategy="uniform")
        assert X_disc.shape == X.shape

    def test_specific_columns(self, sample_data):
        X, _ = sample_data
        X_disc, enc = discretize_features(X, n_bins=3, columns=[0, 1])
        assert X_disc.shape == X.shape


class TestRunFeaturePipeline:
    def test_scaling_then_pca(self, sample_data):
        X, y = sample_data
        steps = [
            ("scaling", {"method": "standard"}),
            ("pca", {"n_components": 2}),
        ]
        X_out, results = run_feature_pipeline(X, y, steps)
        assert X_out.shape == (100, 2)
        assert len(results["步骤结果"]) == 2

    def test_feature_selection(self, sample_data):
        X, y = sample_data
        steps = [
            ("feature_selection", {"method": "f_classif", "k": 3}),
        ]
        X_out, results = run_feature_pipeline(X, y, steps)
        assert X_out.shape[1] == 3

    def test_polynomial(self, sample_data):
        X, y = sample_data
        steps = [
            ("polynomial", {"degree": 2}),
        ]
        X_out, results = run_feature_pipeline(X, y, steps)
        assert X_out.shape[1] > 5

    def test_discretize(self, sample_data):
        X, y = sample_data
        steps = [
            ("discretize", {"n_bins": 5, "strategy": "quantile"}),
        ]
        X_out, results = run_feature_pipeline(X, y, steps)
        assert X_out.shape == X.shape
