<<<<<<< Updated upstream
# -*- coding: utf-8 -*-
"""
ML-Lab 模型评估模块 v3.0
提供分类、回归和无监督聚类任务的全面评估指标计算与展示。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score
)


# ═══════════════════════════════════════════════════════════════
# 分类评估
# ═══════════════════════════════════════════════════════════════

def evaluate_classification(y_true, y_pred, y_proba=None):
    """全面评估分类模型"""
    results = {
        "准确率 (Accuracy)": accuracy_score(y_true, y_pred),
        "精确率 (Precision, macro)": precision_score(y_true, y_pred, average='macro'),
        "召回率 (Recall, macro)": recall_score(y_true, y_pred, average='macro'),
        "F1分数 (F1, macro)": f1_score(y_true, y_pred, average='macro'),
        "F1分数 (F1, weighted)": f1_score(y_true, y_pred, average='weighted'),
    }
    report = classification_report(y_true, y_pred, output_dict=True)
    results["分类报告"] = report

    classes = sorted(set(y_true.tolist() + y_pred.tolist()))
    per_class = {}
    for cls in classes:
        cls_key = str(cls)
        if cls_key in report and "support" in report[cls_key]:
            per_class[f"类别{cls}"] = {
                "precision": report[cls_key]["precision"],
                "recall": report[cls_key]["recall"],
                "f1-score": report[cls_key]["f1-score"],
                "support": int(report[cls_key]["support"]),
            }
    results["各类别指标"] = per_class
    return results


# ═══════════════════════════════════════════════════════════════
# 回归评估
# ═══════════════════════════════════════════════════════════════

def evaluate_regression(y_true, y_pred):
    """全面评估回归模型"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "均方误差 (MSE)": mean_squared_error(y_true, y_pred),
        "均方根误差 (RMSE)": rmse,
        "平均绝对误差 (MAE)": mean_absolute_error(y_true, y_pred),
        "R²分数": r2_score(y_true, y_pred),
        "解释方差分": 1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8),
    }


# ═══════════════════════════════════════════════════════════════
# 聚类评估（v3.0 新增）
# ═══════════════════════════════════════════════════════════════

def evaluate_clustering(X, labels, y_true=None):
    """全面评估聚类模型

    内部指标（无需真实标签）：
        - 轮廓系数 (Silhouette): [-1, 1]，越接近 1 越好
        - CH 指数 (Calinski-Harabasz): 越大越好
        - DB 指数 (Davies-Bouldin): 越小越好

    外部指标（需要真实标签，可选）：
        - 同质化分数 (Homogeneity): [0, 1]
        - 完整性分数 (Completeness): [0, 1]
        - V-measure: 同质化与完整性的调和均值

    Args:
        X (np.ndarray): 特征数据
        labels (np.ndarray): 聚类标签 (-1 为噪声)
        y_true (np.ndarray): 真实标签 (可选)

    Returns:
        dict: 评估指标字典
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int((labels == -1).sum()) if -1 in unique_labels else 0

    results = {
        "聚类数": n_clusters,
        "噪声点数": n_noise,
        "样本总数": len(labels),
    }

    valid_mask = labels != -1
    n_valid = valid_mask.sum()
    labels_valid = labels[valid_mask] if n_valid > 0 else labels

    if n_clusters >= 2 and n_valid >= 2:
        X_valid = X[valid_mask]
        try:
            results["轮廓系数 (Silhouette)"] = silhouette_score(X_valid, labels_valid)
        except Exception:
            results["轮廓系数 (Silhouette)"] = None
        try:
            results["CH 指数 (Calinski-Harabasz)"] = calinski_harabasz_score(X_valid, labels_valid)
        except Exception:
            results["CH 指数 (Calinski-Harabasz)"] = None
        try:
            results["DB 指数 (Davies-Bouldin)"] = davies_bouldin_score(X_valid, labels_valid)
        except Exception:
            results["DB 指数 (Davies-Bouldin)"] = None
    else:
        for k in ["轮廓系数 (Silhouette)", "CH 指数 (Calinski-Harabasz)", "DB 指数 (Davies-Bouldin)"]:
            results[k] = None

    if y_true is not None and n_clusters >= 2:
        try:
            results["同质化分数 (Homogeneity)"] = homogeneity_score(y_true[valid_mask], labels_valid)
            results["完整性分数 (Completeness)"] = completeness_score(y_true[valid_mask], labels_valid)
            results["V-measure"] = v_measure_score(y_true[valid_mask], labels_valid)
        except Exception:
            pass

    return results


# ═══════════════════════════════════════════════════════════════
# 格式化输出
# ═══════════════════════════════════════════════════════════════

def format_evaluation_report(results, task_type="classification"):
    """格式化评估报告为可读文本"""
    lines = []
    lines.append("=" * 50)
    lines.append("模型评估报告")
    lines.append("=" * 50)

    if task_type == "classification":
        lines.append("\n--- 总体指标 ---")
        for key in ["准确率 (Accuracy)", "精确率 (Precision, macro)",
                     "召回率 (Recall, macro)", "F1分数 (F1, macro)",
                     "F1分数 (F1, weighted)"]:
            if key in results:
                lines.append(f"  {key}: {results[key]:.4f}")
        if "各类别指标" in results:
            lines.append("\n--- 各类别指标 ---")
            lines.append(f"  {'类别':<10} {'精确率':>8} {'召回率':>8} {'F1分数':>8} {'样本数':>8}")
            lines.append("  " + "-" * 46)
            for cn, m in results["各类别指标"].items():
                lines.append(f"  {cn:<10} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1-score']:>8.4f} {int(m['support']):>8d}")

    elif task_type == "regression":
        lines.append("\n--- 回归指标 ---")
        for key, val in results.items():
            if isinstance(val, (int, float)):
                lines.append(f"  {key}: {val:.4f}")

    elif task_type == "unsupervised":
        lines.append("\n--- 聚类概况 ---")
        for key in ["聚类数", "噪声点数", "样本总数"]:
            if key in results:
                lines.append(f"  {key}: {results[key]}")
        lines.append("\n--- 内部评估指标（无需真实标签） ---")
        desc_map = {"轮廓系数 (Silhouette)": "越接近1越好", "CH 指数 (Calinski-Harabasz)": "越大越好", "DB 指数 (Davies-Bouldin)": "越小越好"}
        for key, desc in desc_map.items():
            val = results.get(key)
            lines.append(f"  {key}: {val:.4f}  ({desc})" if val is not None else f"  {key}: 无法计算")
        if "同质化分数 (Homogeneity)" in results:
            lines.append("\n--- 外部评估指标（对比真实标签） ---")
            for key in ["同质化分数 (Homogeneity)", "完整性分数 (Completeness)", "V-measure"]:
                val = results.get(key)
                if val is not None:
                    lines.append(f"  {key}: {val:.4f}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def get_evaluation_dataframe(results, task_type="classification"):
    """将评估结果转换为 DataFrame"""
    import pandas as pd
    rows = []
    if task_type == "classification":
        for k, v in results.items():
            if isinstance(v, (int, float)):
                rows.append({"指标": k, "值": f"{v:.4f}"})
    elif task_type == "regression":
        for k, v in results.items():
            if isinstance(v, (int, float)):
                rows.append({"指标": k, "值": f"{v:.4f}"})
    elif task_type == "unsupervised":
        for k, v in results.items():
            if isinstance(v, (int, float)):
                rows.append({"指标": k, "值": f"{v:.4f}"})
            elif isinstance(v, str):
                rows.append({"指标": k, "值": v})
=======
# -*- coding: utf-8 -*-
"""
ML-Lab 模型评估模块 v3.0
提供分类、回归和无监督聚类任务的全面评估指标计算与展示。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score
)


# ═══════════════════════════════════════════════════════════════
# 分类评估
# ═══════════════════════════════════════════════════════════════

def evaluate_classification(y_true, y_pred, y_proba=None):
    """全面评估分类模型"""
    results = {
        "准确率 (Accuracy)": accuracy_score(y_true, y_pred),
        "精确率 (Precision, macro)": precision_score(y_true, y_pred, average='macro'),
        "召回率 (Recall, macro)": recall_score(y_true, y_pred, average='macro'),
        "F1分数 (F1, macro)": f1_score(y_true, y_pred, average='macro'),
        "F1分数 (F1, weighted)": f1_score(y_true, y_pred, average='weighted'),
    }
    report = classification_report(y_true, y_pred, output_dict=True)
    results["分类报告"] = report

    classes = sorted(set(y_true.tolist() + y_pred.tolist()))
    per_class = {}
    for cls in classes:
        cls_key = str(cls)
        if cls_key in report and "support" in report[cls_key]:
            per_class[f"类别{cls}"] = {
                "precision": report[cls_key]["precision"],
                "recall": report[cls_key]["recall"],
                "f1-score": report[cls_key]["f1-score"],
                "support": int(report[cls_key]["support"]),
            }
    results["各类别指标"] = per_class
    return results


# ═══════════════════════════════════════════════════════════════
# 回归评估
# ═══════════════════════════════════════════════════════════════

def evaluate_regression(y_true, y_pred):
    """全面评估回归模型"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "均方误差 (MSE)": mean_squared_error(y_true, y_pred),
        "均方根误差 (RMSE)": rmse,
        "平均绝对误差 (MAE)": mean_absolute_error(y_true, y_pred),
        "R²分数": r2_score(y_true, y_pred),
        "解释方差分": 1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8),
    }


# ═══════════════════════════════════════════════════════════════
# 聚类评估（v3.0 新增）
# ═══════════════════════════════════════════════════════════════

def evaluate_clustering(X, labels, y_true=None):
    """全面评估聚类模型

    内部指标（无需真实标签）：
        - 轮廓系数 (Silhouette): [-1, 1]，越接近 1 越好
        - CH 指数 (Calinski-Harabasz): 越大越好
        - DB 指数 (Davies-Bouldin): 越小越好

    外部指标（需要真实标签，可选）：
        - 同质化分数 (Homogeneity): [0, 1]
        - 完整性分数 (Completeness): [0, 1]
        - V-measure: 同质化与完整性的调和均值

    Args:
        X (np.ndarray): 特征数据
        labels (np.ndarray): 聚类标签 (-1 为噪声)
        y_true (np.ndarray): 真实标签 (可选)

    Returns:
        dict: 评估指标字典
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int((labels == -1).sum()) if -1 in unique_labels else 0

    results = {
        "聚类数": n_clusters,
        "噪声点数": n_noise,
        "样本总数": len(labels),
    }

    valid_mask = labels != -1
    n_valid = valid_mask.sum()
    labels_valid = labels[valid_mask] if n_valid > 0 else labels

    if n_clusters >= 2 and n_valid >= 2:
        X_valid = X[valid_mask]
        try:
            results["轮廓系数 (Silhouette)"] = silhouette_score(X_valid, labels_valid)
        except Exception:
            results["轮廓系数 (Silhouette)"] = None
        try:
            results["CH 指数 (Calinski-Harabasz)"] = calinski_harabasz_score(X_valid, labels_valid)
        except Exception:
            results["CH 指数 (Calinski-Harabasz)"] = None
        try:
            results["DB 指数 (Davies-Bouldin)"] = davies_bouldin_score(X_valid, labels_valid)
        except Exception:
            results["DB 指数 (Davies-Bouldin)"] = None
    else:
        for k in ["轮廓系数 (Silhouette)", "CH 指数 (Calinski-Harabasz)", "DB 指数 (Davies-Bouldin)"]:
            results[k] = None

    if y_true is not None and n_clusters >= 2:
        try:
            results["同质化分数 (Homogeneity)"] = homogeneity_score(y_true[valid_mask], labels_valid)
            results["完整性分数 (Completeness)"] = completeness_score(y_true[valid_mask], labels_valid)
            results["V-measure"] = v_measure_score(y_true[valid_mask], labels_valid)
        except Exception:
            pass

    return results


# ═══════════════════════════════════════════════════════════════
# 格式化输出
# ═══════════════════════════════════════════════════════════════

def format_evaluation_report(results, task_type="classification"):
    """格式化评估报告为可读文本"""
    lines = []
    lines.append("=" * 50)
    lines.append("模型评估报告")
    lines.append("=" * 50)

    if task_type == "classification":
        lines.append("\n--- 总体指标 ---")
        for key in ["准确率 (Accuracy)", "精确率 (Precision, macro)",
                     "召回率 (Recall, macro)", "F1分数 (F1, macro)",
                     "F1分数 (F1, weighted)"]:
            if key in results:
                lines.append(f"  {key}: {results[key]:.4f}")
        if "各类别指标" in results:
            lines.append("\n--- 各类别指标 ---")
            lines.append(f"  {'类别':<10} {'精确率':>8} {'召回率':>8} {'F1分数':>8} {'样本数':>8}")
            lines.append("  " + "-" * 46)
            for cn, m in results["各类别指标"].items():
                lines.append(f"  {cn:<10} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1-score']:>8.4f} {int(m['support']):>8d}")

    elif task_type == "regression":
        lines.append("\n--- 回归指标 ---")
        for key, val in results.items():
            if isinstance(val, (int, float)):
                lines.append(f"  {key}: {val:.4f}")

    elif task_type == "unsupervised":
        lines.append("\n--- 聚类概况 ---")
        for key in ["聚类数", "噪声点数", "样本总数"]:
            if key in results:
                lines.append(f"  {key}: {results[key]}")
        lines.append("\n--- 内部评估指标（无需真实标签） ---")
        desc_map = {"轮廓系数 (Silhouette)": "越接近1越好", "CH 指数 (Calinski-Harabasz)": "越大越好", "DB 指数 (Davies-Bouldin)": "越小越好"}
        for key, desc in desc_map.items():
            val = results.get(key)
            lines.append(f"  {key}: {val:.4f}  ({desc})" if val is not None else f"  {key}: 无法计算")
        if "同质化分数 (Homogeneity)" in results:
            lines.append("\n--- 外部评估指标（对比真实标签） ---")
            for key in ["同质化分数 (Homogeneity)", "完整性分数 (Completeness)", "V-measure"]:
                val = results.get(key)
                if val is not None:
                    lines.append(f"  {key}: {val:.4f}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def get_evaluation_dataframe(results, task_type="classification"):
    """将评估结果转换为 DataFrame"""
    import pandas as pd
    rows = []
    if task_type == "classification":
        for k, v in results.items():
            if isinstance(v, (int, float)):
                rows.append({"指标": k, "值": f"{v:.4f}"})
    elif task_type == "regression":
        for k, v in results.items():
            if isinstance(v, (int, float)):
                rows.append({"指标": k, "值": f"{v:.4f}"})
    elif task_type == "unsupervised":
        for k, v in results.items():
            if isinstance(v, (int, float)):
                rows.append({"指标": k, "值": f"{v:.4f}"})
            elif isinstance(v, str):
                rows.append({"指标": k, "值": v})
>>>>>>> Stashed changes
    return pd.DataFrame(rows) if rows else pd.DataFrame()