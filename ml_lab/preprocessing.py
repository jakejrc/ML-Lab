# -*- coding: utf-8 -*-
"""
ML-Lab 数据预处理模块 v3.2
提供数据加载、缺失值处理、特征缩放、数据集划分等功能。
v3.0: 新增聚类合成数据集、数据集任务类型查询。
"""

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
    make_classification, make_regression, make_blobs, make_circles,
    load_digits, load_diabetes, fetch_olivetti_faces)

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
    "超市购物篮 (Market Basket)": ("association", "20笔交易, 8种商品 — 购物篮关联规则分析"),
    "糖尿病 (Diabetes)": ("regression", "442样本, 10特征 — 糖尿病进展预测"),
    "Olivetti人脸 (Faces)": ("unsupervised", "400样本, 4096特征(64×64) — 人脸降维与聚类"),
    "农作物分类 (Crops)": ("classification", "合成数据, 5特征 — 农业遥感模拟分类"),
}


def get_builtin_datasets(task_type=None):
    """获取内置数据集列表

    Args:
        task_type (str|None): None=全部, 'classification', 'regression', 'unsupervised', 'association'

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
    elif name == "超市购物篮 (Market Basket)":
        # 经典超市购物篮事务型数据（布尔矩阵，20笔交易 x 8种商品）
        _items = ["牛奶", "面包", "尿布", "黄油", "啤酒", "可乐", "鸡蛋", "薯片"]
        # 预定义购物模式 — 包含典型的"啤酒尿布"等关联
        _X = np.array([
            [1,1,1,0,0,0,1,0],
            [0,1,1,1,1,0,0,0],
            [1,1,0,1,0,1,1,0],
            [1,1,1,0,0,0,0,0],
            [0,1,1,1,1,0,0,1],
            [1,0,1,0,1,0,1,0],
            [0,1,0,0,1,1,0,1],
            [1,1,0,0,0,1,1,0],
            [0,1,1,1,0,0,0,1],
            [1,1,1,0,1,0,1,0],
            [0,0,1,0,1,0,0,0],
            [1,1,0,0,0,0,1,0],
            [0,1,1,0,1,1,0,0],
            [1,0,1,1,0,0,1,0],
            [0,1,1,1,1,0,0,0],
            [1,1,0,0,0,1,0,1],
            [0,0,1,0,1,0,1,0],
            [1,1,1,1,0,0,0,0],
            [0,1,0,0,1,1,0,1],
            [1,1,1,0,0,0,1,0],
        ], dtype=np.float64)
        _y = np.zeros(_X.shape[0], dtype=np.int64)
        return (_X, _y, _items, ["购物篮"],
                f"超市购物篮: {_X.shape[0]}笔交易, {_X.shape[1]}种商品 — 关联规则分析")
    elif name == "糖尿病 (Diabetes)":
        import os as _os
        import json as _json
        _path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "diabetes.npz")
        _d = np.load(_path, allow_pickle=True)
        # feature_names 可能存为 JSON 字符串或 object 数组
        if "feature_names_json" in _d:
            fn = _json.loads(str(_d["feature_names_json"]))
        elif "feature_names" in _d:
            fn = list(_d["feature_names"])
        else:
            fn = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        return (_d["data"].astype(np.float64), _d["target"].astype(np.float64),
                fn, ["疾病进展"],
                "糖尿病: 442样本, 10特征 — 糖尿病进展量化指标预测")
    elif name == "Olivetti人脸 (Faces)":
        import os as _os
        _path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "olivetti_faces.npz")
        _d = np.load(_path)
        X, y = _d["data"].astype(np.float64), _d["target"].astype(np.int64)
        fn = [f"pixel_{i}" for i in range(X.shape[1])]
        return (X, y, fn,
                [f"人物{i}" for i in range(40)],
                "Olivetti人脸: 400样本, 4096特征(64×64) — 人脸图像降维与聚类")
    elif name == "农作物分类 (Crops)":
        import os as _os
        _path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "crops.npz")
        _d = np.load(_path)
        X, y = _d["data"].astype(np.float64), _d["target"].astype(np.int64)
        fn = ["NDVI(植被指数)", "EVI(增强植被)", "NDWI(水体指数)",
              "SAVI(土壤调节)", "纹理方差"][:X.shape[1]]
        tn = ["小麦", "玉米", "大豆"][:3]
        return (X, y, fn, tn,
                f"农作物: {X.shape[0]}样本, {X.shape[1]}特征, 3类 — 遥感模拟分类")
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


def get_dataset_summary(X, y, feature_names=None, task_type=None):
    """获取数据集统计摘要

    Args:
        task_type (str|None): 'classification'/'regression'/'unsupervised'/'association'
    """
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(X.shape[1])]

    stats = {
        "样本数": X.shape[0],
        "特征数": X.shape[1],
        "缺失值数": int(np.isnan(X).sum()),
    }

    # 根据任务类型决定类别/簇数显示方式
    if task_type == "regression":
        stats["目标值范围"] = f"{y.min():.2f} ~ {y.max():.2f}"
        stats["目标值均值"] = f"{y.mean():.2f}"
    elif task_type == "association":
        pass  # 关联规则无类别概念
    else:
        stats["类别/簇数"] = len(np.unique(y))

    for i, name in enumerate(feature_names[:6]):
        stats[f"{name}(均值)"] = f"{np.nanmean(X[:, i]):.3f}"
        stats[f"{name}(标准差)"] = f"{np.nanstd(X[:, i]):.3f}"
    return pd.DataFrame(list(stats.items()), columns=["指标", "值"])

# ═══════════════════════════════════════════════════════════════
# v3.6: 自定义数据集加载
# ═══════════════════════════════════════════════════════════════

def load_custom_dataset(file_path, target_col=-1, task_type="auto"):
    """加载用户上传的自定义数据集（CSV / Excel）

    Args:
        file_path (str): 上传文件路径
        target_col (int|str): 目标列位置或名称，-1 表示最后一列
        task_type (str): 'classification' / 'regression' / 'auto'
            auto 时根据目标列类型自动判断（连续→回归，离散→分类）

    Returns:
        tuple: (X, y, feature_names, target_names, description)

    Raises:
        ValueError: 数据集行数不足或格式不支持
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请上传 CSV 或 Excel 文件")

    if df.shape[0] < 5:
        raise ValueError(f"数据集行数过少（{df.shape[0]}行），至少需要 5 行")

    # 确定目标列
    if isinstance(target_col, str) and target_col in df.columns:
        y_col = target_col
    else:
        idx = int(target_col) if target_col != -1 else -1
        y_col = df.columns[idx]

    feature_cols = [c for c in df.columns if c != y_col]

    # 数值化：非数值列尝试 LabelEncoder
    from sklearn.preprocessing import LabelEncoder as _LE
    _encoders = {}
    for col in feature_cols + [y_col]:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            le = _LE()
            df[col] = le.fit_transform(df[col].astype(str))
            _encoders[col] = le

    # 分离 X, y
    X = df[feature_cols].values.astype(np.float64)
    y = df[y_col].values

    # 确保特征列全是数值
    _num_mask = np.all(~np.isnan(X), axis=0)
    feature_cols = [c for c, m in zip(feature_cols, _num_mask) if m]
    X = df[feature_cols].values.astype(np.float64)

    if X.shape[1] == 0:
        raise ValueError("没有可用的数值特征列，请确保数据中包含数值型特征")

    feature_names = list(feature_cols)
    target_names = [y_col]

    # 自动判断任务类型
    if task_type == "auto":
        _y_valid = y[~np.isnan(y.astype(float))] if np.issubdtype(y.dtype, np.floating) else y
        _unique = len(np.unique(_y_valid))
        task_type = "classification" if _unique <= 20 else "regression"

    # 分类任务：y 需要是整数标签
    if task_type == "classification":
        y = y.astype(np.int64)

    description = (f"自定义数据集: {X.shape[0]}样本, {X.shape[1]}特征 — "
                   f"{'分类' if task_type == 'classification' else '回归'}任务, "
                   f"目标列='{y_col}'")

    # 注册到元信息
    _ds_name = f"自定义: {os.path.basename(file_path)}"
    _DATASET_META[_ds_name] = (task_type, description)

    return X, y, feature_names, target_names, description


def register_custom_dataset(file_path, target_col=-1, task_type="auto"):
    """加载并注册自定义数据集，返回数据集名称

    Returns:
        tuple: (dataset_name, X, y, feature_names, target_names, description)
    """
    X, y, fn, tn, desc = load_custom_dataset(file_path, target_col, task_type)
    ds_name = f"自定义: {os.path.basename(file_path)}"
    return ds_name, X, y, fn, tn, desc
