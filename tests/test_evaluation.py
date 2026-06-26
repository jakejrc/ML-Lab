# -*- coding: utf-8 -*-
"""测试模型评估模块"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.evaluation import (
    evaluate_classification, evaluate_regression, evaluate_clustering,
    format_evaluation_report, format_evaluation_table,
    get_evaluation_dataframe,
)


class TestEvaluateClassification:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        results = evaluate_classification(y_true, y_pred)
        assert results["准确率 (Accuracy)"] == 1.0
        assert results["F1分数 (F1, macro)"] == 1.0

    def test_with_probabilities(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])
        results = evaluate_classification(y_true, y_pred, y_proba)
        assert "准确率 (Accuracy)" in results

    def test_imperfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        results = evaluate_classification(y_true, y_pred)
        assert 0 <= results["准确率 (Accuracy)"] <= 1

    def test_per_class_metrics(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        results = evaluate_classification(y_true, y_pred)
        assert "各类别指标" in results
        assert len(results["各类别指标"]) > 0


class TestEvaluateRegression:
    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = evaluate_regression(y_true, y_pred)
        assert results["R²分数"] == pytest.approx(1.0)
        assert results["均方误差 (MSE)"] == pytest.approx(0.0)

    def test_imperfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        results = evaluate_regression(y_true, y_pred)
        assert results["R²分数"] > 0.9
        assert results["均方根误差 (RMSE)"] > 0

    def test_all_keys_present(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.5])
        results = evaluate_regression(y_true, y_pred)
        expected_keys = ["均方误差 (MSE)", "均方根误差 (RMSE)", "平均绝对误差 (MAE)", "R²分数", "解释方差分"]
        for k in expected_keys:
            assert k in results


class TestEvaluateClustering:
    def test_basic_clustering(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [5, 5],
        ])
        labels = np.array([0]*20 + [1]*20)
        results = evaluate_clustering(X, labels)
        assert results["聚类数"] == 2
        assert results["噪声点数"] == 0
        assert "轮廓系数 (Silhouette)" in results

    def test_with_noise(self):
        X = np.random.randn(30, 2)
        labels = np.array([0]*10 + [1]*10 + [-1]*10)
        results = evaluate_clustering(X, labels)
        assert results["噪声点数"] == 10

    def test_with_true_labels(self):
        X = np.random.randn(40, 2)
        labels = np.array([0]*20 + [1]*20)
        y_true = np.array([0]*20 + [1]*20)
        results = evaluate_clustering(X, labels, y_true=y_true)
        assert "同质化分数 (Homogeneity)" in results

    def test_single_cluster(self):
        X = np.random.randn(10, 2)
        labels = np.zeros(10, dtype=int)
        results = evaluate_clustering(X, labels)
        assert results["聚类数"] == 1
        assert results["轮廓系数 (Silhouette)"] is None


class TestFormatEvaluationReport:
    def test_classification_report(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        results = evaluate_classification(y_true, y_pred)
        report = format_evaluation_report(results, "classification")
        assert "模型评估报告" in report
        assert "准确率" in report

    def test_regression_report(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.5])
        results = evaluate_regression(y_true, y_pred)
        report = format_evaluation_report(results, "regression")
        assert "模型评估报告" in report

    def test_clustering_report(self):
        X = np.random.randn(30, 2)
        labels = np.array([0]*15 + [1]*15)
        results = evaluate_clustering(X, labels)
        report = format_evaluation_report(results, "unsupervised")
        assert "模型评估报告" in report
        assert "聚类数" in report


class TestFormatEvaluationTable:
    def test_classification_table(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        results = evaluate_classification(y_true, y_pred)
        html = format_evaluation_table(results, "classification")
        assert "eval-table" in html
        assert "eval-container" in html

    def test_regression_table(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.5])
        results = evaluate_regression(y_true, y_pred)
        html = format_evaluation_table(results, "regression")
        assert "eval-table" in html

    def test_clustering_table(self):
        X = np.random.randn(30, 2)
        labels = np.array([0]*15 + [1]*15)
        results = evaluate_clustering(X, labels)
        html = format_evaluation_table(results, "unsupervised")
        assert "eval-table" in html


class TestGetEvaluationDataframe:
    def test_classification_df(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        results = evaluate_classification(y_true, y_pred)
        df = get_evaluation_dataframe(results, "classification")
        assert len(df) > 0

    def test_regression_df(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.5])
        results = evaluate_regression(y_true, y_pred)
        df = get_evaluation_dataframe(results, "regression")
        assert len(df) > 0
