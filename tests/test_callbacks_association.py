# -*- coding: utf-8 -*-
"""测试关联规则回调"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.state import _g
from ml_lab.callbacks.data import on_load_data
from ml_lab.callbacks.association import on_run_association, on_copy_association_code


class TestOnRunAssociation:
    """测试关联规则挖掘回调"""

    def setup_method(self):
        on_load_data("超市购物篮 (Market Basket)", 0.3)

    def teardown_method(self):
        _g.update(dict(X=None, y=None, model=None, dataset_name="未加载"))

    @pytest.mark.parametrize("algo_name", ["Apriori", "FP-Growth"])
    def test_run_association(self, algo_name):
        """测试关联规则挖掘"""
        result = on_run_association(
            algo_name, min_support=0.1, min_confidence=0.5, min_lift=1.0,
            max_length=5, disc_method="label", n_bins=5, top_k=10
        )
        if isinstance(result, tuple) and len(result) == 6:
            img1, img2, img3, img4, report_html, model_html = result
            assert report_html is not None

    def test_run_without_data(self):
        """测试未加载数据时运行"""
        _reset_g()
        result = on_run_association(
            "Apriori", min_support=0.1, min_confidence=0.5, min_lift=1.0,
            max_length=5, disc_method="label", n_bins=5, top_k=10
        )
        assert len(result) == 6
        # 错误消息中包含"请先"字样
        err_msg = str(result[4])
        assert "请先" in err_msg


def _reset_g():
    _g.update(dict(X=None, y=None, model=None, dataset_name="未加载",
                   last_assoc_model=None))


class TestOnCopyAssociationCode:
    """测试关联规则代码复制回调"""

    def test_copy_without_training(self):
        """测试未训练时复制代码"""
        _reset_g()
        result = on_copy_association_code(
            "Apriori", min_support=0.1, min_confidence=0.5, min_lift=1.0,
            max_length=5, disc_method="label", n_bins=5
        )
        assert len(result) == 3
        assert "先执行" in str(result[0])
