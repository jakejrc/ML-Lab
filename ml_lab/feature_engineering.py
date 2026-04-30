<<<<<<< Updated upstream
# -*- coding: utf-8 -*-
"""
ML-Lab 特征工程模块
提供特征缩放、编码、特征选择、特征构造等功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    PolynomialFeatures, KBinsDiscretizer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# 特征缩放
# ═══════════════════════════════════════════════════════════════

def scale_data(X_train, X_test=None, method="standard"):
    """对特征进行缩放处理

    Args:
        X_train: 训练数据 (2D array)
        X_test: 测试数据 (可选)
        method: 缩放方法 (standard/minmax/robust/normalizer/maxabs)

    Returns:
        (X_train_scaled, X_test_scaled, scaler)
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "normalizer": Normalizer(),
        "maxabs": MaxAbsScaler() if hasattr(__import__('sklearn.preprocessing'), 'MaxAbsScaler') else None,
    }

    scaler = scalers.get(method)
    if scaler is None:
        raise ValueError(f"未知缩放方法: {method}")

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if X_test is not None else None

    return X_train_s, X_test_s, scaler


def get_scaling_stats(X_before, X_after, feature_names=None):
    """获取缩放前后的统计信息"""
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(X_before.shape[1])]

    stats = []
    for i, name in enumerate(feature_names):
        stats.append({
            "特征": name,
            "缩放前_均值": round(np.nanmean(X_before[:, i]), 4),
            "缩放前_标准差": round(np.nanstd(X_before[:, i]), 4),
            "缩放前_最小值": round(np.nanmin(X_before[:, i]), 4),
            "缩放前_最大值": round(np.nanmax(X_before[:, i]), 4),
            "缩放后_均值": round(np.nanmean(X_after[:, i]), 4),
            "缩放后_标准差": round(np.nanstd(X_after[:, i]), 4),
            "缩放后_最小值": round(np.nanmin(X_after[:, i]), 4),
            "缩放后_最大值": round(np.nanmax(X_after[:, i]), 4),
        })
    return stats


# ═══════════════════════════════════════════════════════════════
# 特征编码
# ═══════════════════════════════════════════════════════════════

def encode_labels(y, method="label"):
    """对标签进行编码

    Args:
        y: 标签数据 (1D array)
        method: 编码方法 (label/ordinal)

    Returns:
        (y_encoded, encoder)
    """
    if method == "label":
        enc = LabelEncoder()
    elif method == "ordinal":
        enc = OrdinalEncoder()
    else:
        raise ValueError(f"未知编码方法: {method}")

    y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
    y_encoded = enc.fit_transform(y_reshaped)
    if len(y.shape) == 1:
        y_encoded = y_encoded.ravel()
    return y_encoded, enc


def one_hot_features(X, columns=None):
    """对特征进行 One-Hot 编码

    Args:
        X: 特征矩阵 (2D array)
        columns: 需要编码的列索引列表 (None=所有列)

    Returns:
        (X_encoded, encoder)
    """
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if columns is not None:
        X_subset = X[:, columns]
        X_encoded = enc.fit_transform(X_subset)
        # 合并编码后的特征和非编码特征
        mask = np.ones(X.shape[1], dtype=bool)
        mask[columns] = False
        X_other = X[:, mask]
        return np.hstack([X_other, X_encoded]), enc, columns
    else:
        X_encoded = enc.fit_transform(X)
        return X_encoded, enc, None


# ═══════════════════════════════════════════════════════════════
# 特征选择
# ═══════════════════════════════════════════════════════════════

def select_features_univariate(X, y, method="f_classif", k=5, task_type="classification"):
    """单变量特征选择

    Args:
        X: 特征矩阵
        y: 标签
        method: 选择方法 (f_classif/f_regression/mutual_info_classif/mutual_info_regression)
        k: 选择的特征数
        task_type: 任务类型

    Returns:
        (X_selected, selected_indices, scores)
    """
    score_funcs = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression,
    }

    func = score_funcs.get(method)
    if func is None:
        raise ValueError(f"未知方法: {method}")

    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=func, k=k)
    X_selected = selector.fit_transform(X, y)

    scores = selector.scores_
    pvalues = selector.pvalues_ if hasattr(selector, 'pvalues_') else None
    indices = selector.get_support(indices=True)

    return X_selected, indices, scores, pvalues


def select_features_rfe(X, y, n_features=5, task_type="classification"):
    """递归特征消除 (RFE)

    Args:
        X: 特征矩阵
        y: 标签
        n_features: 保留特征数
        task_type: 分类/回归

    Returns:
        (X_selected, selected_indices, ranking)
    """
    if task_type == "classification":
        estimator = LogisticRegression(max_iter=500, solver='liblinear')
    else:
        estimator = Ridge(alpha=1.0)

    n_features = min(n_features, X.shape[1])
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    X_selected = selector.fit_transform(X, y)

    return X_selected, selector.get_support(indices=True), selector.ranking_


def select_features_model_based(X, y, threshold="median", task_type="classification"):
    """基于模型的特征选择 (RandomForest 特征重要性)

    Args:
        X: 特征矩阵
        y: 标签
        threshold: 特征重要性阈值
        task_type: 分类/回归

    Returns:
        (X_selected, selected_indices, importances)
    """
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X, y)
    importances = model.feature_importances_

    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    X_selected = selector.fit_transform(X, y)
    indices = selector.get_support(indices=True)

    return X_selected, indices, importances


# ═══════════════════════════════════════════════════════════════
# PCA 降维
# ═══════════════════════════════════════════════════════════════

def apply_pca(X, n_components=2):
    """PCA 降维

    Args:
        X: 特征矩阵
        n_components: 降维目标维度

    Returns:
        (X_pca, pca_model, explained_variance_ratio)
    """
    n_comp = min(n_components, X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    return X_pca, pca, pca.explained_variance_ratio_


# ═══════════════════════════════════════════════════════════════
# 特征构造
# ═══════════════════════════════════════════════════════════════

def construct_polynomial_features(X, degree=2):
    """多项式特征构造

    Args:
        X: 特征矩阵
        degree: 多项式阶数

    Returns:
        (X_poly, poly_model)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


def construct_interaction_features(X):
    """构造交互特征 (两两相乘)

    Args:
        X: 特征矩阵

    Returns:
        (X_inter, interaction_names)
    """
    n_features = X.shape[1]
    interactions = []
    names = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
            names.append(f"f{i}_x_f{j}")
    if interactions:
        X_inter = np.hstack(interactions)
    else:
        X_inter = X.copy()
        names = [f"f{i}" for i in range(n_features)]
    return X_inter, names


def construct_statistical_features(X):
    """构造统计特征 (每行统计量)

    Args:
        X: 特征矩阵

    Returns:
        (X_enhanced, new_feature_names)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    minimum = np.min(X, axis=1, keepdims=True)
    maximum = np.max(X, axis=1, keepdims=True)
    median = np.median(X, axis=1, keepdims=True)
    range_val = maximum - minimum

    new_features = np.hstack([mean, std, minimum, maximum, median, range_val])
    new_names = ["行均值", "行标准差", "行最小值", "行最大值", "行中位数", "行极差"]
    X_enhanced = np.hstack([X, new_features])

    return X_enhanced, new_names


# ═══════════════════════════════════════════════════════════════
# 分箱离散化
# ═══════════════════════════════════════════════════════════════

def discretize_features(X, n_bins=5, strategy="quantile", columns=None):
    """特征分箱离散化

    Args:
        X: 特征矩阵
        n_bins: 分箱数
        strategy: 分箱策略 (uniform/quantile/kmeans)
        columns: 要分箱的列索引 (None=所有列)

    Returns:
        (X_discrete, encoder)
    """
    enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)

    if columns is not None:
        X_out = X.copy()
        X_out[:, columns] = enc.fit_transform(X[:, columns])
        return X_out, enc
    else:
        X_discrete = enc.fit_transform(X)
        return X_discrete, enc


# ═══════════════════════════════════════════════════════════════
# 综合流水线
# ═══════════════════════════════════════════════════════════════

def run_feature_pipeline(X, y, steps, task_type="classification", **kwargs):
    """执行特征工程流水线

    Args:
        X: 原始特征矩阵
        y: 标签
        steps: 处理步骤列表, 每步为 (step_name, params_dict)
        task_type: 任务类型
        **kwargs: 额外参数

    Returns:
        (X_transformed, results_dict)  results_dict 包含每步的详细信息
    """
    X_curr = X.copy()
    results = {"原始维度": X.shape, "步骤结果": []}

    for step_name, params in steps:
        step_result = {"步骤": step_name, "参数": params}

        if step_name == "scaling":
            X_test = kwargs.get("X_test")
            X_new, X_test_new, scaler = scale_data(X_curr, X_test, params.get("method", "standard"))
            step_result["输出维度"] = X_new.shape
            step_result["方法"] = params.get("method", "standard")
            X_curr = X_new
            results["X_test_transformed"] = X_test_new

        elif step_name == "feature_selection":
            method = params.get("method", "f_classif")
            k = params.get("k", 5)
            if method in ("f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression"):
                X_new, indices, scores, pvals = select_features_univariate(
                    X_curr, y, method=method, k=k, task_type=task_type)
                step_result["选中特征索引"] = indices.tolist()
                step_result["得分"] = scores.tolist()
                if pvals is not None:
                    step_result["p值"] = [round(float(p), 6) for p in pvals]
            elif method == "rfe":
                X_new, indices, ranking = select_features_rfe(
                    X_curr, y, n_features=k, task_type=task_type)
                step_result["选中特征索引"] = indices.tolist()
                step_result["排名"] = ranking.tolist()
            elif method == "model_based":
                X_new, indices, importances = select_features_model_based(
                    X_curr, y, threshold=params.get("threshold", "median"), task_type=task_type)
                step_result["选中特征索引"] = indices.tolist()
                step_result["特征重要性"] = importances.tolist()
            step_result["输出维度"] = X_new.shape
            X_curr = X_new

        elif step_name == "pca":
            n_comp = params.get("n_components", 2)
            X_new, pca_model, evr = apply_pca(X_curr, n_components=n_comp)
            step_result["输出维度"] = X_new.shape
            step_result["方差解释率"] = [round(float(v), 4) for v in evr]
            step_result["累计方差解释率"] = round(float(np.cumsum(evr)[-1]), 4)
            X_curr = X_new

        elif step_name == "polynomial":
            degree = params.get("degree", 2)
            X_new, poly_model = construct_polynomial_features(X_curr, degree=degree)
            step_result["输出维度"] = X_new.shape
            step_result["阶数"] = degree
            X_curr = X_new

        elif step_name == "interaction":
            X_new, names = construct_interaction_features(X_curr)
            step_result["输出维度"] = X_new.shape
            step_result["交互特征数"] = len(names)
            X_curr = X_new

        elif step_name == "statistical":
            X_new, names = construct_statistical_features(X_curr)
            step_result["输出维度"] = X_new.shape
            step_result["新增特征"] = names
            X_curr = X_new

        elif step_name == "discretize":
            n_bins = params.get("n_bins", 5)
            strategy = params.get("strategy", "quantile")
            X_new, enc = discretize_features(X_curr, n_bins=n_bins, strategy=strategy)
            step_result["输出维度"] = X_new.shape
            X_curr = X_new

        step_result["当前维度"] = X_curr.shape
        results["步骤结果"].append(step_result)

    results["最终维度"] = X_curr.shape
    return X_curr, results
=======
# -*- coding: utf-8 -*-
"""
ML-Lab 特征工程模块
提供特征缩放、编码、特征选择、特征构造等功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    PolynomialFeatures, KBinsDiscretizer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# 特征缩放
# ═══════════════════════════════════════════════════════════════

def scale_data(X_train, X_test=None, method="standard"):
    """对特征进行缩放处理

    Args:
        X_train: 训练数据 (2D array)
        X_test: 测试数据 (可选)
        method: 缩放方法 (standard/minmax/robust/normalizer/maxabs)

    Returns:
        (X_train_scaled, X_test_scaled, scaler)
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "normalizer": Normalizer(),
        "maxabs": MaxAbsScaler() if hasattr(__import__('sklearn.preprocessing'), 'MaxAbsScaler') else None,
    }

    scaler = scalers.get(method)
    if scaler is None:
        raise ValueError(f"未知缩放方法: {method}")

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if X_test is not None else None

    return X_train_s, X_test_s, scaler


def get_scaling_stats(X_before, X_after, feature_names=None):
    """获取缩放前后的统计信息"""
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(X_before.shape[1])]

    stats = []
    for i, name in enumerate(feature_names):
        stats.append({
            "特征": name,
            "缩放前_均值": round(np.nanmean(X_before[:, i]), 4),
            "缩放前_标准差": round(np.nanstd(X_before[:, i]), 4),
            "缩放前_最小值": round(np.nanmin(X_before[:, i]), 4),
            "缩放前_最大值": round(np.nanmax(X_before[:, i]), 4),
            "缩放后_均值": round(np.nanmean(X_after[:, i]), 4),
            "缩放后_标准差": round(np.nanstd(X_after[:, i]), 4),
            "缩放后_最小值": round(np.nanmin(X_after[:, i]), 4),
            "缩放后_最大值": round(np.nanmax(X_after[:, i]), 4),
        })
    return stats


# ═══════════════════════════════════════════════════════════════
# 特征编码
# ═══════════════════════════════════════════════════════════════

def encode_labels(y, method="label"):
    """对标签进行编码

    Args:
        y: 标签数据 (1D array)
        method: 编码方法 (label/ordinal)

    Returns:
        (y_encoded, encoder)
    """
    if method == "label":
        enc = LabelEncoder()
    elif method == "ordinal":
        enc = OrdinalEncoder()
    else:
        raise ValueError(f"未知编码方法: {method}")

    y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
    y_encoded = enc.fit_transform(y_reshaped)
    if len(y.shape) == 1:
        y_encoded = y_encoded.ravel()
    return y_encoded, enc


def one_hot_features(X, columns=None):
    """对特征进行 One-Hot 编码

    Args:
        X: 特征矩阵 (2D array)
        columns: 需要编码的列索引列表 (None=所有列)

    Returns:
        (X_encoded, encoder)
    """
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if columns is not None:
        X_subset = X[:, columns]
        X_encoded = enc.fit_transform(X_subset)
        # 合并编码后的特征和非编码特征
        mask = np.ones(X.shape[1], dtype=bool)
        mask[columns] = False
        X_other = X[:, mask]
        return np.hstack([X_other, X_encoded]), enc, columns
    else:
        X_encoded = enc.fit_transform(X)
        return X_encoded, enc, None


# ═══════════════════════════════════════════════════════════════
# 特征选择
# ═══════════════════════════════════════════════════════════════

def select_features_univariate(X, y, method="f_classif", k=5, task_type="classification"):
    """单变量特征选择

    Args:
        X: 特征矩阵
        y: 标签
        method: 选择方法 (f_classif/f_regression/mutual_info_classif/mutual_info_regression)
        k: 选择的特征数
        task_type: 任务类型

    Returns:
        (X_selected, selected_indices, scores)
    """
    score_funcs = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression,
    }

    func = score_funcs.get(method)
    if func is None:
        raise ValueError(f"未知方法: {method}")

    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=func, k=k)
    X_selected = selector.fit_transform(X, y)

    scores = selector.scores_
    pvalues = selector.pvalues_ if hasattr(selector, 'pvalues_') else None
    indices = selector.get_support(indices=True)

    return X_selected, indices, scores, pvalues


def select_features_rfe(X, y, n_features=5, task_type="classification"):
    """递归特征消除 (RFE)

    Args:
        X: 特征矩阵
        y: 标签
        n_features: 保留特征数
        task_type: 分类/回归

    Returns:
        (X_selected, selected_indices, ranking)
    """
    if task_type == "classification":
        estimator = LogisticRegression(max_iter=500, solver='liblinear')
    else:
        estimator = Ridge(alpha=1.0)

    n_features = min(n_features, X.shape[1])
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    X_selected = selector.fit_transform(X, y)

    return X_selected, selector.get_support(indices=True), selector.ranking_


def select_features_model_based(X, y, threshold="median", task_type="classification"):
    """基于模型的特征选择 (RandomForest 特征重要性)

    Args:
        X: 特征矩阵
        y: 标签
        threshold: 特征重要性阈值
        task_type: 分类/回归

    Returns:
        (X_selected, selected_indices, importances)
    """
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X, y)
    importances = model.feature_importances_

    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    X_selected = selector.fit_transform(X, y)
    indices = selector.get_support(indices=True)

    return X_selected, indices, importances


# ═══════════════════════════════════════════════════════════════
# PCA 降维
# ═══════════════════════════════════════════════════════════════

def apply_pca(X, n_components=2):
    """PCA 降维

    Args:
        X: 特征矩阵
        n_components: 降维目标维度

    Returns:
        (X_pca, pca_model, explained_variance_ratio)
    """
    n_comp = min(n_components, X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    return X_pca, pca, pca.explained_variance_ratio_


# ═══════════════════════════════════════════════════════════════
# 特征构造
# ═══════════════════════════════════════════════════════════════

def construct_polynomial_features(X, degree=2):
    """多项式特征构造

    Args:
        X: 特征矩阵
        degree: 多项式阶数

    Returns:
        (X_poly, poly_model)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


def construct_interaction_features(X):
    """构造交互特征 (两两相乘)

    Args:
        X: 特征矩阵

    Returns:
        (X_inter, interaction_names)
    """
    n_features = X.shape[1]
    interactions = []
    names = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
            names.append(f"f{i}_x_f{j}")
    if interactions:
        X_inter = np.hstack(interactions)
    else:
        X_inter = X.copy()
        names = [f"f{i}" for i in range(n_features)]
    return X_inter, names


def construct_statistical_features(X):
    """构造统计特征 (每行统计量)

    Args:
        X: 特征矩阵

    Returns:
        (X_enhanced, new_feature_names)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    minimum = np.min(X, axis=1, keepdims=True)
    maximum = np.max(X, axis=1, keepdims=True)
    median = np.median(X, axis=1, keepdims=True)
    range_val = maximum - minimum

    new_features = np.hstack([mean, std, minimum, maximum, median, range_val])
    new_names = ["行均值", "行标准差", "行最小值", "行最大值", "行中位数", "行极差"]
    X_enhanced = np.hstack([X, new_features])

    return X_enhanced, new_names


# ═══════════════════════════════════════════════════════════════
# 分箱离散化
# ═══════════════════════════════════════════════════════════════

def discretize_features(X, n_bins=5, strategy="quantile", columns=None):
    """特征分箱离散化

    Args:
        X: 特征矩阵
        n_bins: 分箱数
        strategy: 分箱策略 (uniform/quantile/kmeans)
        columns: 要分箱的列索引 (None=所有列)

    Returns:
        (X_discrete, encoder)
    """
    enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)

    if columns is not None:
        X_out = X.copy()
        X_out[:, columns] = enc.fit_transform(X[:, columns])
        return X_out, enc
    else:
        X_discrete = enc.fit_transform(X)
        return X_discrete, enc


# ═══════════════════════════════════════════════════════════════
# 综合流水线
# ═══════════════════════════════════════════════════════════════

def run_feature_pipeline(X, y, steps, task_type="classification", **kwargs):
    """执行特征工程流水线

    Args:
        X: 原始特征矩阵
        y: 标签
        steps: 处理步骤列表, 每步为 (step_name, params_dict)
        task_type: 任务类型
        **kwargs: 额外参数

    Returns:
        (X_transformed, results_dict)  results_dict 包含每步的详细信息
    """
    X_curr = X.copy()
    results = {"原始维度": X.shape, "步骤结果": []}

    for step_name, params in steps:
        step_result = {"步骤": step_name, "参数": params}

        if step_name == "scaling":
            X_test = kwargs.get("X_test")
            X_new, X_test_new, scaler = scale_data(X_curr, X_test, params.get("method", "standard"))
            step_result["输出维度"] = X_new.shape
            step_result["方法"] = params.get("method", "standard")
            X_curr = X_new
            results["X_test_transformed"] = X_test_new

        elif step_name == "feature_selection":
            method = params.get("method", "f_classif")
            k = params.get("k", 5)
            if method in ("f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression"):
                X_new, indices, scores, pvals = select_features_univariate(
                    X_curr, y, method=method, k=k, task_type=task_type)
                step_result["选中特征索引"] = indices.tolist()
                step_result["得分"] = scores.tolist()
                if pvals is not None:
                    step_result["p值"] = [round(float(p), 6) for p in pvals]
            elif method == "rfe":
                X_new, indices, ranking = select_features_rfe(
                    X_curr, y, n_features=k, task_type=task_type)
                step_result["选中特征索引"] = indices.tolist()
                step_result["排名"] = ranking.tolist()
            elif method == "model_based":
                X_new, indices, importances = select_features_model_based(
                    X_curr, y, threshold=params.get("threshold", "median"), task_type=task_type)
                step_result["选中特征索引"] = indices.tolist()
                step_result["特征重要性"] = importances.tolist()
            step_result["输出维度"] = X_new.shape
            X_curr = X_new

        elif step_name == "pca":
            n_comp = params.get("n_components", 2)
            X_new, pca_model, evr = apply_pca(X_curr, n_components=n_comp)
            step_result["输出维度"] = X_new.shape
            step_result["方差解释率"] = [round(float(v), 4) for v in evr]
            step_result["累计方差解释率"] = round(float(np.cumsum(evr)[-1]), 4)
            X_curr = X_new

        elif step_name == "polynomial":
            degree = params.get("degree", 2)
            X_new, poly_model = construct_polynomial_features(X_curr, degree=degree)
            step_result["输出维度"] = X_new.shape
            step_result["阶数"] = degree
            X_curr = X_new

        elif step_name == "interaction":
            X_new, names = construct_interaction_features(X_curr)
            step_result["输出维度"] = X_new.shape
            step_result["交互特征数"] = len(names)
            X_curr = X_new

        elif step_name == "statistical":
            X_new, names = construct_statistical_features(X_curr)
            step_result["输出维度"] = X_new.shape
            step_result["新增特征"] = names
            X_curr = X_new

        elif step_name == "discretize":
            n_bins = params.get("n_bins", 5)
            strategy = params.get("strategy", "quantile")
            X_new, enc = discretize_features(X_curr, n_bins=n_bins, strategy=strategy)
            step_result["输出维度"] = X_new.shape
            X_curr = X_new

        step_result["当前维度"] = X_curr.shape
        results["步骤结果"].append(step_result)

    results["最终维度"] = X_curr.shape
    return X_curr, results
>>>>>>> Stashed changes
