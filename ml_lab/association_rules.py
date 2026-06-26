# -*- coding: utf-8 -*-
"""
ML-Lab 关联规则挖掘模块 (v3.5 新增)

支持 Apriori 和 FP-Growth 两种经典关联规则挖掘算法。
提供统一接口，支持可视化回调和频繁项集/规则提取。
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# 数据预处理: 将任意数据集转换为事务型布尔矩阵
# ═══════════════════════════════════════════════════════════════

def _to_boolean_matrix(X, feature_names=None, y=None, target_names=None,
                        discretize_method="quantile", n_bins=3):
    """
    将数据集转换为布尔事务矩阵。

    策略:
    - 布尔数据(值仅含0和1): 直接作为事务矩阵，每列即为一个项
    - 连续特征: 按分位数离散化后 one-hot
    - 分类特征(整数, 值域 <= 10): 直接 one-hot
    - 目标变量 y: 同样 one-hot 编码
    """
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # 快速检测：如果所有数据只有0和1，直接作为布尔矩阵使用（购物篮场景）
    unique_all = np.unique(X)
    is_already_boolean = (set(unique_all) <= {0.0, 1.0})

    if is_already_boolean:
        item_map = {}
        bool_cols = {}
        for j in range(n_features):
            col_name = feature_names[j]
            item_map[col_name] = col_name
            bool_cols[col_name] = X[:, j].astype(bool)

        if y is not None:
            y = np.array(y)
            unique_y = np.unique(y)
            y_label = target_names[0] if target_names else "Target"
            for val in unique_y:
                if target_names and len(target_names) > 1:
                    val_label = target_names[int(val)] if int(val) < len(target_names) else str(int(val))
                else:
                    val_label = str(int(val)) if np.all(y == np.floor(y)) else f"{val:.2f}"
                col_name = f"{y_label}={val_label}"
                bool_cols[col_name] = (y == val)
                item_map[col_name] = f"{y_label} = {val_label}"

        df_bool = pd.DataFrame(bool_cols, dtype=bool)
        return df_bool, item_map

    # 非布尔数据：走离散化逻辑
    item_map = {}
    bool_cols = {}

    for j in range(n_features):
        col_data = X[:, j]
        unique_vals = np.unique(col_data)
        is_discrete = (np.all(col_data == np.floor(col_data)) and len(unique_vals) <= 10)

        if is_discrete:
            for val in unique_vals:
                col_name = f"{feature_names[j]}={int(val)}"
                bool_cols[col_name] = (col_data == val)
                item_map[col_name] = f"{feature_names[j]} 取值 {int(val)}"
        else:
            if discretize_method == "quantile":
                bins = pd.qcut(col_data, q=n_bins, labels=False, duplicates='drop')
            else:
                bins = pd.cut(col_data, bins=n_bins, labels=False, include_lowest=True)
            unique_bins = np.unique(bins[~np.isnan(bins)])
            level_names = ["低", "中", "高"] if len(unique_bins) <= 3 else [f"L{i}" for i in range(len(unique_bins))]
            for idx, b in enumerate(sorted(unique_bins)):
                b_int = int(b)
                label = level_names[idx] if idx < len(level_names) else f"Level{idx}"
                col_name = f"{feature_names[j]}={label}"
                bool_cols[col_name] = (bins == b)
                item_map[col_name] = f"{feature_names[j]} {label}值"

    if y is not None:
        y = np.array(y)
        unique_y = np.unique(y)
        y_label = target_names[0] if target_names else "Target"
        for val in unique_y:
            if target_names and len(target_names) > 1:
                val_label = target_names[int(val)] if int(val) < len(target_names) else str(int(val))
            else:
                val_label = str(int(val)) if np.all(y == np.floor(y)) else f"{val:.2f}"
            col_name = f"{y_label}={val_label}"
            bool_cols[col_name] = (y == val)
            item_map[col_name] = f"{y_label} = {val_label}"

    df_bool = pd.DataFrame(bool_cols, dtype=bool)
    return df_bool, item_map


# ═══════════════════════════════════════════════════════════════
# Apriori 算法 (纯 Python 实现)
# ═══════════════════════════════════════════════════════════════

def _apriori_gen(Lk, k):
    """由 L(k-1) 生成候选 C(k)"""
    candidates = set()
    Lk_list = list(Lk)
    for i in range(len(Lk_list)):
        for j in range(i + 1, len(Lk_list)):
            l1 = list(Lk_list[i])
            l2 = list(Lk_list[j])
            l1.sort()
            l2.sort()
            if l1[:-1] == l2[:-1]:
                candidate = frozenset(sorted(set(l1) | set(l2)))
                if len(candidate) == k:
                    candidates.add(candidate)
    return candidates


def apriori_frequent_itemsets(df_bool, min_support=0.1, max_length=4):
    """Apriori 算法挖掘频繁项集。"""
    n_trans = len(df_bool)
    min_count = max(1, int(np.ceil(min_support * n_trans)))

    transactions = []
    for _, row in df_bool.iterrows():
        items = frozenset(df_bool.columns[row.values].tolist())
        transactions.append(items)

    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[frozenset([item])] += 1

    Lk = {itemset for itemset, count in item_counts.items() if count >= min_count}
    frequent_items = [(itemset, count / n_trans) for itemset, count in item_counts.items() if count >= min_count]

    k = 2
    while Lk and k <= max_length:
        Ck = _apriori_gen(Lk, k)
        ck_counts = defaultdict(int)
        for t in transactions:
            for candidate in Ck:
                if candidate.issubset(t):
                    ck_counts[candidate] += 1

        Lk = {itemset for itemset, count in ck_counts.items() if count >= min_count}
        for itemset, count in ck_counts.items():
            if count >= min_count:
                frequent_items.append((itemset, count / n_trans))
        k += 1

    return frequent_items


def generate_rules(frequent_items, min_confidence=0.5, min_lift=1.0):
    """从频繁项集生成关联规则。"""
    support_map = {}
    for itemset, sup in frequent_items:
        support_map[itemset] = sup

    rules = []
    for itemset, sup in frequent_items:
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for i in range(1, len(items)):
            for antecedent in combinations(items, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if not consequent:
                    continue
                ant_sup = support_map.get(antecedent, 0)
                if ant_sup == 0:
                    continue
                confidence = sup / ant_sup
                cons_sup = support_map.get(consequent, 0)
                if cons_sup == 0:
                    continue
                lift = confidence / cons_sup
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": sup,
                        "confidence": confidence,
                        "lift": lift,
                    })

    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules


# ═══════════════════════════════════════════════════════════════
# FP-Growth 算法 (纯 Python 实现)
# ═══════════════════════════════════════════════════════════════

class _FPNode:
    """FP 树节点"""
    __slots__ = ('name', 'count', 'parent', 'children', 'next')

    def __init__(self, name, count=1, parent=None):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None


class _FPTree:
    """FP-Growth 树"""

    def __init__(self):
        self.root = _FPNode("NULL")
        self.header_table = {}

    def _insert(self, transaction, count=1):
        current = self.root
        for item in transaction:
            if item in current.children:
                current.children[item].count += count
            else:
                new_node = _FPNode(item, count, current)
                current.children[item] = new_node
                if item not in self.header_table:
                    self.header_table[item] = [new_node, new_node]
                else:
                    self.header_table[item][1].next = new_node
                    self.header_table[item][1] = new_node
            current = current.children[item]

    def build(self, transactions, min_support_count):
        item_counts = defaultdict(int)
        for items, count in transactions:
            for item in items:
                item_counts[item] += count

        frequent_items = {item: cnt for item, cnt in item_counts.items()
                          if cnt >= min_support_count}
        if not frequent_items:
            return False

        self._item_order = sorted(frequent_items.keys(),
                                  key=lambda x: (-frequent_items[x], x))

        for items, count in transactions:
            filtered = [item for item in items if item in frequent_items]
            filtered.sort(key=lambda x: (-frequent_items[x], x))
            if filtered:
                self._insert(filtered, count)

        return True

    def _get_prefix_paths(self, item):
        paths = []
        node = self.header_table.get(item, [None, None])[0]
        while node is not None:
            path = []
            count = node.count
            current = node.parent
            while current and current.name != "NULL":
                path.append(current.name)
                current = current.parent
            if path:
                paths.append((path, count))
            node = node.next
        return paths

    def mine_patterns(self, min_support_count, prefix=frozenset()):
        patterns = []
        sorted_items = sorted(self.header_table.keys(),
                              key=lambda x: (self._count_nodes(x), str(x)))

        for item in sorted_items:
            item_support = self._count_nodes(item)
            if item_support >= min_support_count:
                new_pattern = prefix | frozenset([item])
                patterns.append((new_pattern, item_support))

                prefix_paths = self._get_prefix_paths(item)
                if prefix_paths:
                    cond_tree = _FPTree()
                    path_counts = defaultdict(int)
                    for path, count in prefix_paths:
                        path_key = tuple(path)
                        path_counts[path_key] += count

                    cond_transactions = [(list(pk), cnt) for pk, cnt in path_counts.items()]
                    if cond_tree.build(cond_transactions, min_support_count):
                        sub_patterns = cond_tree.mine_patterns(min_support_count, new_pattern)
                        patterns.extend(sub_patterns)

        return patterns

    def _count_nodes(self, item):
        total = 0
        node = self.header_table.get(item, [None, None])[0]
        while node is not None:
            total += node.count
            node = node.next
        return total


def fpgrowth_frequent_itemsets(df_bool, min_support=0.1, max_length=4):
    """FP-Growth 算法挖掘频繁项集。"""
    n_trans = len(df_bool)
    min_count = max(1, int(np.ceil(min_support * n_trans)))

    transactions = []
    for _, row in df_bool.iterrows():
        items = [col for col in df_bool.columns if row[col]]
        transactions.append((items, 1))

    tree = _FPTree()
    if not tree.build(transactions, min_count):
        return []

    patterns = tree.mine_patterns(min_count)
    frequent_items = []
    for pattern, count in patterns:
        support = count / n_trans
        if len(pattern) <= max_length:
            frequent_items.append((frozenset(pattern), support))

    return frequent_items


# ═══════════════════════════════════════════════════════════════
# 统一模型接口
# ═══════════════════════════════════════════════════════════════

class AprioriModel:
    """Apriori 关联规则挖掘模型"""

    def __init__(self, min_support=0.1, min_confidence=0.5, min_lift=1.0,
                 max_length=4, discretize_method="quantile", n_bins=3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_length = max_length
        self.discretize_method = discretize_method
        self.n_bins = n_bins
        self.frequent_items = []
        self.rules = []
        self.history = {}
        self._task_type = "association"
        self._item_map = {}
        self._df_bool = None

    def fit(self, X, y=None, feature_names=None, target_names=None):
        start_time = datetime.now()
        self._df_bool, self._item_map = _to_boolean_matrix(
            X, feature_names, y, target_names,
            self.discretize_method, self.n_bins
        )
        self.frequent_items = apriori_frequent_itemsets(
            self._df_bool, self.min_support, self.max_length
        )
        self.rules = generate_rules(
            self.frequent_items, self.min_confidence, self.min_lift
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        self.history = {
            "n_transactions": len(self._df_bool),
            "n_items": len(self._df_bool.columns),
            "n_frequent_items": len(self.frequent_items),
            "n_rules": len(self.rules),
            "elapsed_time": elapsed,
            "support_counts": [item[1] for item in self.frequent_items],
            "rule_confidences": [r["confidence"] for r in self.rules],
            "rule_lifts": [r["lift"] for r in self.rules],
            "item_length_dist": defaultdict(int),
        }
        for itemset, _ in self.frequent_items:
            self.history["item_length_dist"][len(itemset)] += 1
        return self

    def get_params(self):
        return {
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_lift": self.min_lift,
            "max_length": self.max_length,
            "n_frequent_items": len(self.frequent_items),
            "n_rules": len(self.rules),
        }


class FPGrowthModel:
    """FP-Growth 关联规则挖掘模型"""

    def __init__(self, min_support=0.1, min_confidence=0.5, min_lift=1.0,
                 max_length=4, discretize_method="quantile", n_bins=3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_length = max_length
        self.discretize_method = discretize_method
        self.n_bins = n_bins
        self.frequent_items = []
        self.rules = []
        self.history = {}
        self._task_type = "association"
        self._item_map = {}
        self._df_bool = None
        self._tree = None          # FP 树对象，用于可视化

    def fit(self, X, y=None, feature_names=None, target_names=None):
        start_time = datetime.now()
        self._df_bool, self._item_map = _to_boolean_matrix(
            X, feature_names, y, target_names,
            self.discretize_method, self.n_bins
        )

        # 构建 FP 树并挖掘频繁项集
        n_trans = len(self._df_bool)
        min_count = max(1, int(np.ceil(self.min_support * n_trans)))
        transactions = []
        for _, row in self._df_bool.iterrows():
            items = [col for col in self._df_bool.columns if row[col]]
            transactions.append((items, 1))
        tree = _FPTree()
        tree.build(transactions, min_count)
        self._tree = tree

        self.frequent_items = fpgrowth_frequent_itemsets(
            self._df_bool, self.min_support, self.max_length
        )
        self.rules = generate_rules(
            self.frequent_items, self.min_confidence, self.min_lift
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        self.history = {
            "n_transactions": len(self._df_bool),
            "n_items": len(self._df_bool.columns),
            "item_length_dist": defaultdict(int),
            "n_frequent_items": len(self.frequent_items),
            "n_rules": len(self.rules),
            "elapsed_time": elapsed,
            "support_counts": [item[1] for item in self.frequent_items],
            "rule_confidences": [r["confidence"] for r in self.rules],
            "rule_lifts": [r["lift"] for r in self.rules],
        }
        for itemset, _ in self.frequent_items:
            self.history["item_length_dist"][len(itemset)] += 1
        return self

    def get_params(self):
        return {
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_lift": self.min_lift,
            "max_length": self.max_length,
            "n_frequent_items": len(self.frequent_items),
            "n_rules": len(self.rules),
        }


# ═══════════════════════════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_association(model):
    """评估关联规则挖掘结果。"""
    hist = model.history
    rules = model.rules
    freq = model.frequent_items

    result = {
        "task_type": "association",
        "algorithm": model.__class__.__name__,
        "n_transactions": hist.get("n_transactions", 0),
        "n_items": hist.get("n_items", 0),
        "n_frequent_items": len(freq),
        "n_rules": len(rules),
        "elapsed_time": f"{hist.get('elapsed_time', 0):.3f}s",
        "item_length_distribution": dict(hist.get("item_length_dist", {})),
    }

    if rules:
        confidences = [r["confidence"] for r in rules]
        lifts = [r["lift"] for r in rules]
        supports = [r["support"] for r in rules]
        result.update({
            "avg_confidence": float(np.mean(confidences)),
            "max_confidence": float(np.max(confidences)),
            "avg_lift": float(np.mean(lifts)),
            "max_lift": float(np.max(lifts)),
            "avg_support": float(np.mean(supports)),
            "max_support": float(np.max(supports)),
        })

    if freq:
        freq_supports = [item[1] for item in freq]
        result["max_frequent_support"] = float(np.max(freq_supports))
        result["avg_frequent_support"] = float(np.mean(freq_supports))

    return result
