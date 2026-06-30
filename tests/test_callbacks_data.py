# -*- coding: utf-8 -*-
"""测试数据加载与预处理回调"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.state import _g
from ml_lab.callbacks.data import on_load_data, on_preprocess, on_export_report
from ml_lab.preprocessing import get_builtin_datasets


def _reset_g():
    _g.update(dict(
        X=None, y=None, X_train=None, X_test=None,
        y_train=None, y_test=None,
        feature_names=None, target_names=None,
        dataset_name="未加载",
    ))


class TestOnLoadData:
    """测试 on_load_data 回调"""

    def teardown_method(self):
        _reset_g()

    @pytest.mark.parametrize("dataset_name", [
        "鸢尾花 (Iris)",
        "葡萄酒 (Wine)",
        "乳腺癌 (Breast Cancer)",
        "手写数字 (Digits)",
        "波士顿房价 (Boston)",
        "加州房价 (California)",
        "糖尿病 (Diabetes)",
    ])
    def test_load_builtin_datasets(self, dataset_name):
        """测试加载内置数据集"""
        img, info, summary, ds_html = on_load_data(dataset_name, 0.3)
        assert _g["X"] is not None, f"{dataset_name}: X 未加载"
        assert _g["y"] is not None
        assert _g["X_train"] is not None
        assert _g["X_test"] is not None
        assert _g["feature_names"] is not None
        assert len(_g["X"]) > 0
        assert len(_g["y"]) == len(_g["X"])
        assert img is not None

    def test_load_association_datasets(self):
        """测试加载关联规则数据集"""
        img, info, summary, ds_html = on_load_data("超市购物篮 (Market Basket)", 0.3)
        assert _g["X"] is not None
        assert _g["dataset_name"] == "超市购物篮 (Market Basket)"

    def test_load_all_builtin_datasets(self):
        """验证所有内置数据集均可加载"""
        all_datasets = get_builtin_datasets()
        assert len(all_datasets) >= 14
        for name in all_datasets:
            img, info, summary, ds_html = on_load_data(name, 0.3)
            assert _g["X"] is not None, f"{name}: X 加载失败"
            _reset_g()

    def test_unknown_dataset(self):
        """测试未知数据集返回错误"""
        img, info, summary, ds_html = on_load_data("不存在的数据集", 0.3)
        assert "加载失败" in str(info)


class TestOnPreprocess:
    """测试数据预处理回调"""

    def setup_method(self):
        on_load_data("鸢尾花 (Iris)", 0.3)

    def teardown_method(self):
        _reset_g()

    def test_preprocess_no_data(self):
        """测试未加载数据时预处理"""
        _reset_g()
        img, html_info = on_preprocess("标准化 (StandardScaler)", 0)
        assert "请先加载数据集" in html_info


class TestOnExportReport:
    """测试报告导出回调"""

    def test_export_with_html(self):
        result = on_export_report("<html><body><h1>Test Report</h1></body></html>")
        assert result is not None

    def test_export_without_html(self):
        _g["last_eval_report"] = "<html><body><p>Saved Report</p></body></html>"
        result = on_export_report()
        assert result is not None
