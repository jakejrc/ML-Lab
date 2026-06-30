# -*- coding: utf-8 -*-
"""测试 MLState 状态管理器"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.state import MLState, _g, _sync


class TestMLStateBasics:
    """测试 MLState 基本功能"""

    def setup_method(self):
        self.s = MLState()

    def test_default_values(self):
        """测试默认值"""
        assert self.s["X"] is None
        assert self.s["X_train"] is None
        assert self.s["dataset_name"] == "未加载"
        assert self.s["model_name"] == "未训练"
        assert self.s["report_images"] == {}
        assert self.s["experiment_log"] == []

    def test_dict_access(self):
        """测试 dict 式访问（向后兼容）"""
        self.s["X"] = np.array([1, 2, 3])
        assert np.array_equal(self.s["X"], [1, 2, 3])

    def test_dot_access(self):
        """测试属性式访问"""
        self.s.X = np.array([1, 2, 3])
        assert np.array_equal(self.s.X, [1, 2, 3])
        assert np.array_equal(self.s["X"], [1, 2, 3])

    def test_reset(self):
        """测试重置"""
        self.s.X = np.array([1, 2, 3])
        self.s.dataset_name = "测试数据"
        self.s.reset()
        assert self.s.X is None
        assert self.s.dataset_name == "未加载"

    def test_has_data(self):
        """测试 has_data 属性"""
        assert not self.s.has_data
        self.s.X_train = np.array([1, 2, 3])
        assert self.s.has_data

    def test_has_model(self):
        """测试 has_model 属性"""
        assert not self.s.has_model
        self.s.model = "test_model"
        assert self.s.has_model

    def test_task_type_default(self):
        """测试默认 task_type"""
        assert self.s.task_type == "unknown"
        self.s.last_task_type = "classification"
        assert self.s.task_type == "classification"

    def test_sync(self):
        """测试 sync 方法"""
        X_train = np.array([[1, 2], [3, 4]])
        X_test = np.array([[5, 6]])
        y_train = np.array([0, 1])
        y_test = np.array([0])
        fnames = ["f1", "f2"]
        tnames = ["c0", "c1"]

        self.s.sync(X_train, X_test, y_train, y_test, fnames, tnames, X=X_train, y=y_train)

        assert np.array_equal(self.s.X_train, X_train)
        assert np.array_equal(self.s.X_test, X_test)
        assert np.array_equal(self.s.y_train, y_train)
        assert np.array_equal(self.s.y_test, y_test)
        assert self.s.feature_names == fnames
        assert self.s.target_names == tnames
        assert np.array_equal(self.s.X, X_train)
        assert np.array_equal(self.s.y, y_train)

    def test_sync_without_xy(self):
        """测试 sync 不传 X, y"""
        self.s.sync([1], [2], [3], [4], ["f"], ["t"])
        assert self.s.X is None  # 不覆盖已有值

    def test_update_model(self):
        """测试 update_model"""
        model = "test_model_object"
        self.s.update_model(model, "决策树")
        assert self.s.model == model
        assert self.s.model_name == "决策树"

    def test_log_experiment(self):
        """测试实验日志"""
        entry = {"algo": "决策树", "accuracy": 0.95}
        self.s.log_experiment(entry)
        assert len(self.s.experiment_log) == 1
        assert self.s.experiment_log[0] == entry
        self.s.log_experiment({"algo": "SVM", "accuracy": 0.93})
        assert len(self.s.experiment_log) == 2

    def test_to_dict(self):
        """测试导出可序列化快照"""
        self.s.X = np.array([1, 2, 3])
        self.s.model = "some_model"
        self.s.report_images = {"img1": b"data"}
        d = self.s.to_dict()
        assert "model" not in d
        assert "report_images" not in d
        assert np.array_equal(d["X"], [1, 2, 3])

    def test_dict_contains(self):
        """测试 in 操作符"""
        assert "X" in self.s
        assert "nonexistent" not in self.s

    def test_dict_len(self):
        """测试 len()"""
        assert len(self.s) >= 15


class TestGlobalSingleton:
    """测试全局单例 _g"""

    def test_singleton_type(self):
        assert isinstance(_g, MLState)

    def test_sync_function(self):
        """测试 _sync 向后兼容"""
        _g.reset()
        _sync([1], [2], [3], [4], ["f"], ["t"], X=[0], y=[5])
        assert _g.X_train == [1]
        assert _g.X == [0]
        assert _g.y == [5]

    def test_import_compatibility(self):
        """验证 _g 的 dict 式访问仍然可用"""
        _g["dataset_name"] = "测试"
        assert _g["dataset_name"] == "测试"
        assert _g.dataset_name == "测试"

    def test_reset_via_singleton(self):
        _g.reset()
        assert _g.dataset_name == "未加载"
