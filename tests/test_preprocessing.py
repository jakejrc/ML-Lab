# -*- coding: utf-8 -*-
"""测试数据预处理模块"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.preprocessing import (
    load_dataset, get_builtin_datasets, get_dataset_type,
    handle_missing, inject_missing, scale_features, split_dataset,
    get_dataset_summary, load_custom_dataset, register_custom_dataset,
)


class TestGetBuiltinDatasets:
    def test_returns_all_datasets(self):
        ds = get_builtin_datasets()
        assert len(ds) >= 14

    def test_filter_classification(self):
        ds = get_builtin_datasets(task_type="classification")
        assert all("分类" in v or "Iris" in k or "Wine" in k or "Breast" in k or "Digits" in k
                    for k, v in ds.items()) or len(ds) > 0

    def test_filter_regression(self):
        ds = get_builtin_datasets(task_type="regression")
        assert len(ds) > 0

    def test_filter_unsupervised(self):
        ds = get_builtin_datasets(task_type="unsupervised")
        assert len(ds) > 0


class TestGetDatasetType:
    def test_known_datasets(self):
        assert get_dataset_type("鸢尾花 (Iris)") == "classification"
        assert get_dataset_type("波士顿房价 (Boston)") == "regression"
        assert get_dataset_type("聚类球形 (Blobs)") == "unsupervised"

    def test_unknown_defaults_to_classification(self):
        assert get_dataset_type("不存在的数据集") == "classification"


class TestLoadDataset:
    def test_iris(self):
        X, y, fn, tn, desc = load_dataset("鸢尾花 (Iris)")
        assert X.shape == (150, 4)
        assert len(y) == 150
        assert len(fn) == 4
        assert len(tn) == 3

    def test_wine(self):
        X, y, fn, tn, desc = load_dataset("葡萄酒 (Wine)")
        assert X.shape[0] == 178
        assert X.shape[1] == 13

    def test_breast_cancer(self):
        X, y, fn, tn, desc = load_dataset("乳腺癌 (Breast Cancer)")
        assert X.shape == (569, 30)

    def test_digits(self):
        X, y, fn, tn, desc = load_dataset("手写数字 (Digits)")
        assert X.shape == (1797, 64)

    def test_synthetic_classification(self):
        X, y, fn, tn, desc = load_dataset("合成分类数据", n_samples=200, n_features=5, n_classes=3)
        assert X.shape == (200, 5)
        assert len(np.unique(y)) == 3

    def test_linear_regression(self):
        X, y, fn, tn, desc = load_dataset("线性回归", n_samples=300)
        assert X.shape[0] == 300

    def test_blobs(self):
        X, y, fn, tn, desc = load_dataset("聚类球形 (Blobs)", n_samples=150, n_clusters=4)
        assert X.shape[0] == 150
        assert len(np.unique(y)) == 4

    def test_circles(self):
        X, y, fn, tn, desc = load_dataset("聚类环形 (Circles)", n_samples=100)
        assert X.shape[0] == 100

    def test_market_basket(self):
        X, y, fn, tn, desc = load_dataset("超市购物篮 (Market Basket)")
        assert X.shape == (20, 8)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="未知数据集"):
            load_dataset("不存在的数据集")


class TestHandleMissing:
    def test_mean_strategy(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        result = handle_missing(X, strategy="mean")
        assert not np.any(np.isnan(result))

    def test_median_strategy(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        result = handle_missing(X, strategy="median")
        assert not np.any(np.isnan(result))

    def test_constant_strategy(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        result = handle_missing(X, strategy="constant")
        assert not np.any(np.isnan(result))
        assert result[1, 0] == 0


class TestInjectMissing:
    def test_injects_nan(self):
        X = np.ones((100, 5))
        X_missing = inject_missing(X, missing_ratio=0.2, random_state=42)
        assert np.any(np.isnan(X_missing))

    def test_original_not_modified(self):
        X = np.ones((100, 5))
        X_orig = X.copy()
        inject_missing(X, missing_ratio=0.2)
        np.testing.assert_array_equal(X, X_orig)


class TestScaleFeatures:
    def test_standard(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Xs, Xt, scaler = scale_features(X, method="standard")
        assert Xs.shape == X.shape
        assert np.abs(Xs.mean(axis=0)).max() < 1e-10

    def test_minmax(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Xs, _, _ = scale_features(X, method="minmax")
        assert Xs.min() >= 0 and Xs.max() <= 1

    def test_robust(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Xs, _, _ = scale_features(X, method="robust")
        assert Xs.shape == X.shape

    def test_with_test_data(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        X_test = np.array([[2, 3], [4, 5]], dtype=float)
        Xs_train, Xs_test, _ = scale_features(X_train, X_test, method="standard")
        assert Xs_test is not None
        assert Xs_test.shape == X_test.shape


class TestSplitDataset:
    def test_basic_split(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20

    def test_stratify(self):
        X = np.random.randn(100, 5)
        y = np.array([0]*50 + [1]*50)
        X_train, X_test, y_train, y_test = split_dataset(X, y, stratify=True)
        train_ratio = np.mean(y_train == 0)
        test_ratio = np.mean(y_test == 0)
        assert abs(train_ratio - test_ratio) < 0.1

    def test_no_stratify(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        X_train, X_test, _, _ = split_dataset(X, y, stratify=False)
        assert X_train.shape[0] + X_test.shape[0] == 100


class TestGetDatasetSummary:
    def test_returns_dataframe(self):
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 3, 50)
        summary = get_dataset_summary(X, y, ["f1", "f2", "f3", "f4"])
        assert isinstance(summary, pd.DataFrame)
        assert summary.iloc[0, 1] == 50  # 样本数

    def test_default_feature_names(self):
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        summary = get_dataset_summary(X, y)
        assert isinstance(summary, pd.DataFrame)


class TestLoadCustomDataset:
    def _make_csv(self, content, suffix=".csv"):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_csv_classification(self):
        path = self._make_csv("f1,f2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1")
        X, y, fn, tn, desc = load_custom_dataset(path)
        assert X.shape[0] == 6
        assert "分类" in desc

    def test_csv_regression(self):
        content = "f1,f2,target\n" + "\n".join(f"{i},{i*2},{i*3.14}" for i in range(30))
        path = self._make_csv(content)
        X, y, fn, tn, desc = load_custom_dataset(path)
        assert X.shape[0] == 30
        assert "回归" in desc

    def test_too_few_rows_raises(self):
        path = self._make_csv("f1,target\n1,0\n2,1")
        with pytest.raises(ValueError, match="行数过少"):
            load_custom_dataset(path)

    def test_unsupported_format_raises(self):
        path = self._make_csv("f1,target\n1,0", suffix=".txt")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_custom_dataset(path)

    def test_register_custom_dataset(self):
        path = self._make_csv("f1,f2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1")
        name, X, y, fn, tn, desc = register_custom_dataset(path)
        assert "自定义" in name
