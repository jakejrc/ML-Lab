<<<<<<< Updated upstream
# -*- coding: utf-8 -*-
"""
ML-Lab 数据预处理模块 v3.1
提供数据加载、缺失值处理、特征缩放、数据集划分等功能。
v3.0: 新增聚类合成数据集、数据集任务类型查询。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.datasets import (    load_iris, load_wine, load_breast_cancer,
    make_classification, make_regression, make_blobs, make_circles,
    load_digits
)

# 数据集元信息：{名称: (task_type, description)}
_DATASET_META = {
    "鸢尾花 (Iris)": ("classification", "150样本, 4特征, 3分类 — 经典入门数据集"),
    "葡萄酒 (Wine)": ("classification", "178样本, 13特征, 3分类 — 化学成分分类"),
    "乳腺癌 (Breast Cancer)": ("classification", "569样本, 30特征, 2分类 — 医学诊断"),
    "手写数字 (Digits)": ("classification", "1797样本, 64特征, 10分类 — 图像识别"),
    "合成分类数据": ("classification", "可调参数随机分类数据集"),
    "线性回归": ("regression", "500样本, 5特征 — 线性回归练习"),
    "波士顿房价 (Boston)": ("regression", "506样本, 13特征 — 房价预测经典数据集"),
    "加州房价 (California)": ("regression", "20640样本, 13特征 — 加州房价预测(California Housing)"),
    "聚类球形 (Blobs)": ("unsupervised", "可调簇数的球形聚类数据集"),
    "聚类环形 (Circles)": ("unsupervised", "含噪声的同心环聚类数据集"),
}


def get_builtin_datasets(task_type=None):
    """获取内置数据集列表

    Args:
        task_type (str|None): None=全部, 'classification', 'regression', 'unsupervised'

    Returns:
        dict: {数据集名称: 描述}
    """
    if task_type is None:
        return {k: v[1] for k, v in _DATASET_META.items()}
    return {k: v[1] for k, v in _DATASET_META.items() if v[0] == task_type}


def get_dataset_type(name):
    """获取数据集的任务类型

    Returns:
        str: 'classification' / 'regression' / 'unsupervised'
    """
    if name in _DATASET_META:
        return _DATASET_META[name][0]
    return "classification"


def load_dataset(name, n_samples=1000, n_features=10, n_classes=2,
                 noise=0.1, random_state=42, n_clusters=3):
    """加载指定数据集

    Args:
        name (str): 数据集名称
        n_samples (int): 样本数
        n_features (int): 特征数
        n_classes (int): 类别数
        noise (float): 噪声比例
        random_state (int): 随机种子
        n_clusters (int): 聚类数 (仅聚类数据集使用)

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
                f"合成分类: {n_samples}样本, {n_features}特征, {n_classes}类")
    elif name == "线性回归":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            noise=noise * 50, random_state=random_state
        )
        return (X, y,
                [f"feature_{i}" for i in range(n_features)],
                ["target"],
                f"回归数据: {n_samples}样本, {n_features}特征")
    elif name == "波士顿房价 (Boston)":
        import os as _os
        _csv = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "data", "boston_housing.csv")
        if not _os.path.exists(_csv):
            raise FileNotFoundError(f"波士顿房价数据文件未找到: {_csv}")
        _df = pd.read_csv(_csv)
        _feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                          "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
        _X = _df.drop(columns=["MEDV"]).values.astype(np.float64)
        _y = _df["MEDV"].values.astype(np.float64)
        return (_X, _y, _feature_names, ["房价(MEDV)"],
                "波士顿房价: 506样本, 13特征 — 波士顿地区房价预测")
    elif name == "加州房价 (California)":
        import os as _os
        _csv = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "data", "housing.csv")
        if not _os.path.exists(_csv):
            raise FileNotFoundError(f"加州房价数据文件未找到: {_csv}")
        _df = pd.read_csv(_csv)
        # 填充缺失值
        _df["total_bedrooms"] = _df["total_bedrooms"].fillna(_df["total_bedrooms"].median())
        # 类别特征编码
        _ocean = pd.get_dummies(_df["ocean_proximity"], prefix="ocean", drop_first=True)
        _num_cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
                     "total_bedrooms", "population", "households", "median_income"]
        _feature_names = _num_cols + [c for c in _ocean.columns]
        _X = pd.concat([_df[_num_cols], _ocean], axis=1).values.astype(np.float64)
        _y = _df["median_house_value"].values.astype(np.float64)
        return (_X, _y, _feature_names, ["房价(MedianHV)"],
                "加州房价: 20640样本, 13特征 — 加州地区房价预测")
    elif name == "聚类球形 (Blobs)":
        X, y = make_blobs(
            n_samples=n_samples, n_features=min(n_features, 10),
            centers=n_clusters, cluster_std=noise * 5 + 0.5,
            random_state=random_state
        )
        return (X, y,
                [f"feature_{i}" for i in range(X.shape[1])],
                [f"cluster_{i}" for i in range(n_clusters)],
                f"聚类球形: {n_samples}样本, {X.shape[1]}特征, {n_clusters}簇")
    elif name == "聚类环形 (Circles)":
        X, y = make_circles(
            n_samples=n_samples, noise=noise * 0.3 + 0.05,
            factor=0.5, random_state=random_state
        )
        return (X, y,
                [f"feature_{i}" for i in range(2)],
                ["外环", "内环"],
                f"聚类环形: {n_samples}样本, 2特征, 2簇")
    else:
        raise ValueError(f"未知数据集: {name}")


def handle_missing(X, strategy="mean"):
    """处理缺失值"""
    fill_value = 0 if strategy == "constant" else None
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    return imputer.fit_transform(X)


def inject_missing(X, missing_ratio=0.1, random_state=42):
    """向数据中注入缺失值（用于演示缺失值处理）"""
    rng = np.random.RandomState(random_state)
    X_missing = X.copy().astype(float)
    mask = rng.random(X_missing.shape) < missing_ratio
    X_missing[mask] = np.nan
    return X_missing


def scale_features(X_train, X_test=None, method="standard"):
    """特征缩放"""
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
    """划分训练集和测试集"""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )


def get_dataset_summary(X, y, feature_names=None):
    """获取数据集统计摘要"""
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(X.shape[1])]

    stats = {
        "样本数": X.shape[0],
        "特征数": X.shape[1],
        "缺失值数": int(np.isnan(X).sum()),
        "类别/簇数": len(np.unique(y)),
    }

    for i, name in enumerate(feature_names[:6]):
        stats[f"{name}(均值)"] = f"{np.nanmean(X[:, i]):.3f}"
        stats[f"{name}(标准差)"] = f"{np.nanstd(X[:, i]):.3f}"

=======
# -*- coding: utf-8 -*-
"""
ML-Lab 数据预处理模块 v3.1
提供数据加载、缺失值处理、特征缩放、数据集划分等功能。
v3.0: 新增聚类合成数据集、数据集任务类型查询。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.datasets import (    load_iris, load_wine, load_breast_cancer,
    make_classification, make_regression, make_blobs, make_circles,
    load_digits
)

# 数据集元信息：{名称: (task_type, description)}
_DATASET_META = {
    "鸢尾花 (Iris)": ("classification", "150样本, 4特征, 3分类 — 经典入门数据集"),
    "葡萄酒 (Wine)": ("classification", "178样本, 13特征, 3分类 — 化学成分分类"),
    "乳腺癌 (Breast Cancer)": ("classification", "569样本, 30特征, 2分类 — 医学诊断"),
    "手写数字 (Digits)": ("classification", "1797样本, 64特征, 10分类 — 图像识别"),
    "合成分类数据": ("classification", "可调参数随机分类数据集"),
    "线性回归": ("regression", "500样本, 5特征 — 线性回归练习"),
    "波士顿房价 (Boston)": ("regression", "506样本, 13特征 — 房价预测经典数据集"),
    "加州房价 (California)": ("regression", "20640样本, 13特征 — 加州房价预测(California Housing)"),
    "聚类球形 (Blobs)": ("unsupervised", "可调簇数的球形聚类数据集"),
    "聚类环形 (Circles)": ("unsupervised", "含噪声的同心环聚类数据集"),
}


def get_builtin_datasets(task_type=None):
    """获取内置数据集列表

    Args:
        task_type (str|None): None=全部, 'classification', 'regression', 'unsupervised'

    Returns:
        dict: {数据集名称: 描述}
    """
    if task_type is None:
        return {k: v[1] for k, v in _DATASET_META.items()}
    return {k: v[1] for k, v in _DATASET_META.items() if v[0] == task_type}


def get_dataset_type(name):
    """获取数据集的任务类型

    Returns:
        str: 'classification' / 'regression' / 'unsupervised'
    """
    if name in _DATASET_META:
        return _DATASET_META[name][0]
    return "classification"


def load_dataset(name, n_samples=1000, n_features=10, n_classes=2,
                 noise=0.1, random_state=42, n_clusters=3):
    """加载指定数据集

    Args:
        name (str): 数据集名称
        n_samples (int): 样本数
        n_features (int): 特征数
        n_classes (int): 类别数
        noise (float): 噪声比例
        random_state (int): 随机种子
        n_clusters (int): 聚类数 (仅聚类数据集使用)

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
                f"合成分类: {n_samples}样本, {n_features}特征, {n_classes}类")
    elif name == "线性回归":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            noise=noise * 50, random_state=random_state
        )
        return (X, y,
                [f"feature_{i}" for i in range(n_features)],
                ["target"],
                f"回归数据: {n_samples}样本, {n_features}特征")
    elif name == "波士顿房价 (Boston)":
        import os as _os
        _csv = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "data", "boston_housing.csv")
        if not _os.path.exists(_csv):
            raise FileNotFoundError(f"波士顿房价数据文件未找到: {_csv}")
        _df = pd.read_csv(_csv)
        _feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                          "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
        _X = _df.drop(columns=["MEDV"]).values.astype(np.float64)
        _y = _df["MEDV"].values.astype(np.float64)
        return (_X, _y, _feature_names, ["房价(MEDV)"],
                "波士顿房价: 506样本, 13特征 — 波士顿地区房价预测")
    elif name == "加州房价 (California)":
        import os as _os
        _csv = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "data", "housing.csv")
        if not _os.path.exists(_csv):
            raise FileNotFoundError(f"加州房价数据文件未找到: {_csv}")
        _df = pd.read_csv(_csv)
        # 填充缺失值
        _df["total_bedrooms"] = _df["total_bedrooms"].fillna(_df["total_bedrooms"].median())
        # 类别特征编码
        _ocean = pd.get_dummies(_df["ocean_proximity"], prefix="ocean", drop_first=True)
        _num_cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
                     "total_bedrooms", "population", "households", "median_income"]
        _feature_names = _num_cols + [c for c in _ocean.columns]
        _X = pd.concat([_df[_num_cols], _ocean], axis=1).values.astype(np.float64)
        _y = _df["median_house_value"].values.astype(np.float64)
        return (_X, _y, _feature_names, ["房价(MedianHV)"],
                "加州房价: 20640样本, 13特征 — 加州地区房价预测")
    elif name == "聚类球形 (Blobs)":
        X, y = make_blobs(
            n_samples=n_samples, n_features=min(n_features, 10),
            centers=n_clusters, cluster_std=noise * 5 + 0.5,
            random_state=random_state
        )
        return (X, y,
                [f"feature_{i}" for i in range(X.shape[1])],
                [f"cluster_{i}" for i in range(n_clusters)],
                f"聚类球形: {n_samples}样本, {X.shape[1]}特征, {n_clusters}簇")
    elif name == "聚类环形 (Circles)":
        X, y = make_circles(
            n_samples=n_samples, noise=noise * 0.3 + 0.05,
            factor=0.5, random_state=random_state
        )
        return (X, y,
                [f"feature_{i}" for i in range(2)],
                ["外环", "内环"],
                f"聚类环形: {n_samples}样本, 2特征, 2簇")
    else:
        raise ValueError(f"未知数据集: {name}")


def handle_missing(X, strategy="mean"):
    """处理缺失值"""
    fill_value = 0 if strategy == "constant" else None
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    return imputer.fit_transform(X)


def inject_missing(X, missing_ratio=0.1, random_state=42):
    """向数据中注入缺失值（用于演示缺失值处理）"""
    rng = np.random.RandomState(random_state)
    X_missing = X.copy().astype(float)
    mask = rng.random(X_missing.shape) < missing_ratio
    X_missing[mask] = np.nan
    return X_missing


def scale_features(X_train, X_test=None, method="standard"):
    """特征缩放"""
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
    """划分训练集和测试集"""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )


def get_dataset_summary(X, y, feature_names=None):
    """获取数据集统计摘要"""
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(X.shape[1])]

    stats = {
        "样本数": X.shape[0],
        "特征数": X.shape[1],
        "缺失值数": int(np.isnan(X).sum()),
        "类别/簇数": len(np.unique(y)),
    }

    for i, name in enumerate(feature_names[:6]):
        stats[f"{name}(均值)"] = f"{np.nanmean(X[:, i]):.3f}"
        stats[f"{name}(标准差)"] = f"{np.nanstd(X[:, i]):.3f}"

>>>>>>> Stashed changes
    return pd.DataFrame(list(stats.items()), columns=["指标", "值"])