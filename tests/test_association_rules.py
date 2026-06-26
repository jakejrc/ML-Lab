# -*- coding: utf-8 -*-
"""测试关联规则挖掘模块"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.association_rules import (
    _to_boolean_matrix, apriori_frequent_itemsets, generate_rules,
    fpgrowth_frequent_itemsets, AprioriModel, FPGrowthModel,
)


@pytest.fixture
def market_basket_data():
    items = ["牛奶", "面包", "尿布", "黄油", "啤酒"]
    X = np.array([
        [1,1,1,0,0],
        [0,1,1,1,1],
        [1,1,0,1,0],
        [1,1,1,0,0],
        [0,1,1,1,1],
        [1,0,1,0,1],
        [0,1,0,0,1],
        [1,1,0,0,0],
    ], dtype=float)
    return X, items


class TestToBooleanMatrix:
    def test_boolean_data(self, market_basket_data):
        X, items = market_basket_data
        df_bool, item_map = _to_boolean_matrix(X, items)
        assert isinstance(df_bool, pd.DataFrame)
        assert df_bool.shape == (8, 5)
        assert all(df_bool.dtypes == bool)

    def test_continuous_data(self):
        X = np.random.randn(50, 3)
        names = ["温度", "湿度", "风速"]
        df_bool, item_map = _to_boolean_matrix(X, names, n_bins=3)
        assert df_bool.shape[0] == 50
        assert df_bool.shape[1] > 3  # one-hot expanded

    def test_with_target(self, market_basket_data):
        X, items = market_basket_data
        y = np.array([0,1,0,1,0,1,0,1])
        df_bool, item_map = _to_boolean_matrix(X, items, y=y, target_names=["购买"])
        assert any("购买" in col for col in df_bool.columns)


class TestAprioriFrequentItemsets:
    def test_basic(self, market_basket_data):
        X, items = market_basket_data
        df_bool, _ = _to_boolean_matrix(X, items)
        frequent = apriori_frequent_itemsets(df_bool, min_support=0.2)
        assert len(frequent) > 0
        for itemset, support in frequent:
            assert 0 < support <= 1

    def test_high_min_support(self, market_basket_data):
        X, items = market_basket_data
        df_bool, _ = _to_boolean_matrix(X, items)
        frequent = apriori_frequent_itemsets(df_bool, min_support=0.9)
        assert len(frequent) == 0 or all(s >= 0.9 for _, s in frequent)


class TestGenerateRules:
    def test_basic(self, market_basket_data):
        X, items = market_basket_data
        df_bool, _ = _to_boolean_matrix(X, items)
        frequent = apriori_frequent_itemsets(df_bool, min_support=0.2)
        rules = generate_rules(frequent, min_confidence=0.5, min_lift=1.0)
        assert isinstance(rules, list)
        for rule in rules:
            assert "antecedent" in rule
            assert "consequent" in rule
            assert "support" in rule
            assert "confidence" in rule
            assert "lift" in rule


class TestFPGrowthFrequentItemsets:
    def test_basic(self, market_basket_data):
        X, items = market_basket_data
        df_bool, _ = _to_boolean_matrix(X, items)
        frequent = fpgrowth_frequent_itemsets(df_bool, min_support=0.2)
        assert len(frequent) > 0

    def test_high_min_support(self, market_basket_data):
        X, items = market_basket_data
        df_bool, _ = _to_boolean_matrix(X, items)
        frequent = fpgrowth_frequent_itemsets(df_bool, min_support=0.9)
        assert len(frequent) == 0 or all(s >= 0.9 for _, s in frequent)

    def test_matches_apriori(self, market_basket_data):
        X, items = market_basket_data
        df_bool, _ = _to_boolean_matrix(X, items)
        apriori_fi = apriori_frequent_itemsets(df_bool, min_support=0.3)
        fp_fi = fpgrowth_frequent_itemsets(df_bool, min_support=0.3)
        apriori_sets = {itemset for itemset, _ in apriori_fi}
        fp_sets = {itemset for itemset, _ in fp_fi}
        assert apriori_sets == fp_sets


class TestAprioriModel:
    def test_fit_and_predict(self, market_basket_data):
        X, items = market_basket_data
        model = AprioriModel(min_support=0.2, min_confidence=0.5)
        model.fit(X, feature_names=items)
        assert len(model.frequent_items) > 0
        assert model.history["n_transactions"] == 8

    def test_get_params(self, market_basket_data):
        X, items = market_basket_data
        model = AprioriModel(min_support=0.3)
        model.fit(X, feature_names=items)
        params = model.get_params()
        assert "min_support" in params
        assert "n_rules" in params


class TestFPGrowthModel:
    def test_fit(self, market_basket_data):
        X, items = market_basket_data
        model = FPGrowthModel(min_support=0.2, min_confidence=0.5)
        model.fit(X, feature_names=items)
        assert len(model.frequent_items) > 0

    def test_get_params(self, market_basket_data):
        X, items = market_basket_data
        model = FPGrowthModel(min_support=0.3)
        model.fit(X, feature_names=items)
        params = model.get_params()
        assert "min_support" in params
