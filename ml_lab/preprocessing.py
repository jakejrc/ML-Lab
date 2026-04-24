# -*- coding: utf-8 -*-
"""
ML-Lab 数据预处理模块
提供数据加载、缺失值处理、特征缩放、数据集划分等功能，
并通过可视化展示预处理前后对比效果。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    make_classification, make_regression,
    load_digits
)


def get_builtin_datasets():
    """获取内置数据集列表

    Returns:
        dict: {数据集名称: 描述}
    """
    return {
        "鸢尾花 (Iris)": "150个样本, 4个特征, 3分类 — 经典入门数据集",
        "葡萄酒 (Wine)": "178个样本, 13个特征, 3分类 — 化学成分分类",
        "乳腺癌 (Breast Cancer)": "569个样本, 30个特征, 2分类 — 医学诊断",
        "手写数字 (Digits)": "1797个样本, 64个特征, 10分类 — 图像识别",
        "合成分类数据": "1000个样本, 可调特征数和类别数 — 随机生成",
        "合成回归数据": "500个样本, 5个特征 — 线性回归练习",
    }


def load_dataset(name, n_samples=1000, n_features=10, n_classes=2,
                 noise=0.1, random_state=42):
    """加载指定数据集

    Args:
        name (str): 数据集名称
        n_samples (int): 合成数据样本数
        n_features (int): 合成数据特征数
        n_classes (int): 合成数据类别数
        noise (float): 合成数据噪声比例
        random_state (int): 随机种子

    Returns:
        tuple: (X, y, feature_names, target_names, description)
    """
    if name == "鸢尾花 (Iris)":
        data = load_iris()
        return (data.data, data.target, list(data.feature_names),
                list(data.target_names), data.DESCR[:200])
    elif name == "葡萄酒 (Wine)":
        data = load_wine()
        return (data.data, data.target, list(data.feature_names),
                list(data.target_names), data.DESCR[:200])
    elif name == "乳腺癌 (Breast Cancer)":
        data = load_breast_cancer()
        return (data.data, data.target, list(data.feature_names),
                list(data.target_names), data.DESCR[:200])
    elif name == "手写数字 (Digits)":
        data = load_digits()
        return (data.data, data.target,
                [f"pixel_{i}" for i in range(64)],
                [str(i) for i in range(10)], data.DESCR[:200])
    elif name == "合成分类数据":
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=min(n_features, n_classes * 2),
            n_redundant=0, n_classes=n_classes,
            n_clusters_per_class=1, random_state=random_state,
            flip_y=noise
        )
        return (X, y,
                [f"feature_{i}" for i in range(n_features)],
                [f"class_{i}" for i in range(n_classes)],
                f"合成数据: {n_samples}样本, {n_features}特征, {n_classes}类")
    elif name == "合成回归数据":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            noise=noise * 50, random_state=random_state
        )
        return (X, y,
                [f"feature_{i}" for i in range(n_features)],
                ["target"],
                f"合成回归数据: {n_samples}样本, {n_features}特征")
    else:
        raise ValueError(f"未知数据集: {name}")


def handle_missing(X, strategy="mean"):
    """处理缺失值

    Args:
        X (np.ndarray): 含缺失值的数据 (NaN)
        strategy (str): 填充策略 ('mean', 'median', 'most_frequent', 'constant')

    Returns:
        np.ndarray: 填充后的数据
    """
    fill_value = 0 if strategy == "constant" else None
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    return imputer.fit_transform(X)


def inject_missing(X, missing_ratio=0.1, random_state=42):
    """向数据中注入缺失值（用于演示缺失值处理）

    Args:
        X (np.ndarray): 原始数据
        missing_ratio (float): 缺失值比例 (0~1)
        random_state (int): 随机种子

    Returns:
        np.ndarray: 含缺失值的数据
    """
    rng = np.random.RandomState(random_state)
    X_missing = X.copy().astype(float)
    mask = rng.random(X_missing.shape) < missing_ratio
    X_missing[mask] = np.nan
    return X_missing


def scale_features(X_train, X_test=None, method="standard"):
    """特征缩放

    Args:
        X_train (np.ndarray): 训练集特征
        X_test (np.ndarray): 测试集特征 (可选)
        method (str): 缩放方法 ('standard', 'minmax', 'robust')

    Returns:
        tuple: (scaled_train, scaled_test, scaler)
            scaler 对象, 可用于后续数据的变换
    """
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler = scalers[method]()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def split_dataset(X, y, test_size=0.3, random_state=42, stratify=True):
    """划分训练集和测试集

    Args:
        X (np.ndarray): 特征数据
        y (np.ndarray): 标签
        test_size (float): 测试集比例
        random_state (int): 随机种子
        stratify (bool): 是否分层抽样

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )


def get_dataset_summary(X, y, feature_names=None):
    """获取数据集统计摘要

    Args:
        X (np.ndarray): 特征数据
        y (np.ndarray): 标签
        feature_names (list): 特征名称列表

    Returns:
        pd.DataFrame: 统计摘要
    """
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(X.shape[1])]

    stats = {
        "样本数": X.shape[0],
        "特征数": X.shape[1],
        "缺失值数": int(np.isnan(X).sum()),
        "类别数": len(np.unique(y)),
    }

    for i, name in enumerate(feature_names[:6]):
        stats[f"{name}(均值)"] = f"{np.nanmean(X[:, i]):.3f}"
        stats[f"{name}(标准差)"] = f"{np.nanstd(X[:, i]):.3f}"

    return pd.DataFrame(list(stats.items()), columns=["指标", "值"])