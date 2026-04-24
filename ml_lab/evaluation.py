# -*- coding: utf-8 -*-
"""
ML-Lab 模型评估模块
提供分类和回归任务的全面评估指标计算与展示。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)


def evaluate_classification(y_true, y_pred, y_proba=None):
    """全面评估分类模型

    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        y_proba (np.ndarray): 预测概率 (可选)

    Returns:
        dict: 评估指标字典
    """
    results = {
        "准确率 (Accuracy)": accuracy_score(y_true, y_pred),
        "精确率 (Precision, macro)": precision_score(y_true, y_pred,
                                                      average='macro'),
        "召回率 (Recall, macro)": recall_score(y_true, y_pred,
                                                average='macro'),
        "F1分数 (F1, macro)": f1_score(y_true, y_pred, average='macro'),
        "F1分数 (F1, weighted)": f1_score(y_true, y_pred,
                                            average='weighted'),
    }

    # 详细分类报告
    report = classification_report(y_true, y_pred, output_dict=True)
    results["分类报告"] = report

    # 每个类别的指标
    # 只取实际类别，排除 macro avg / weighted avg
    classes = sorted(set(y_true.tolist() + y_pred.tolist()))
    per_class = {}
    for cls in classes:
        cls_key = str(cls)
        # 确保匹配到具体类别而非汇总行
        if cls_key in report and "support" in report[cls_key]:
            per_class[f"类别{cls}"] = {
                "precision": report[cls_key]["precision"],
                "recall": report[cls_key]["recall"],
                "f1-score": report[cls_key]["f1-score"],
                "support": int(report[cls_key]["support"]),
            }
    results["各类别指标"] = per_class

    return results


def evaluate_regression(y_true, y_pred):
    """全面评估回归模型

    Args:
        y_true (np.ndarray): 真实值
        y_pred (np.ndarray): 预测值

    Returns:
        dict: 评估指标字典
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "均方误差 (MSE)": mean_squared_error(y_true, y_pred),
        "均方根误差 (RMSE)": rmse,
        "平均绝对误差 (MAE)": mean_absolute_error(y_true, y_pred),
        "R²分数": r2_score(y_true, y_pred),
        "解释方差分": 1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8),
    }


def format_evaluation_report(results, task_type="classification"):
    """格式化评估报告为可读文本

    Args:
        results (dict): evaluate_classification 或 evaluate_regression 的结果
        task_type (str): 'classification' 或 'regression'

    Returns:
        str: 格式化报告
    """
    lines = []
    lines.append("=" * 50)
    lines.append("模型评估报告")
    lines.append("=" * 50)

    if task_type == "classification":
        # 总体指标
        lines.append("\n--- 总体指标 ---")
        for key in ["准确率 (Accuracy)", "精确率 (Precision, macro)",
                     "召回率 (Recall, macro)", "F1分数 (F1, macro)",
                     "F1分数 (F1, weighted)"]:
            if key in results:
                lines.append(f"  {key}: {results[key]:.4f}")

        # 每个类别
        if "各类别指标" in results:
            lines.append("\n--- 各类别指标 ---")
            lines.append(f"  {'类别':<10} {'精确率':>8} {'召回率':>8} "
                         f"{'F1分数':>8} {'样本数':>8}")
            lines.append("  " + "-" * 46)
            for cls_name, metrics in results["各类别指标"].items():
                lines.append(
                    f"  {cls_name:<10} {metrics['precision']:>8.4f} "
                    f"{metrics['recall']:>8.4f} {metrics['f1-score']:>8.4f} "
                    f"{int(metrics['support']):>8d}"
                )

    elif task_type == "regression":
        lines.append("\n--- 回归指标 ---")
        for key, val in results.items():
            lines.append(f"  {key}: {val:.4f}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def get_evaluation_dataframe(results, task_type="classification"):
    """将评估结果转换为 DataFrame

    Args:
        results (dict): 评估结果
        task_type (str): 'classification' 或 'regression'

    Returns:
        pd.DataFrame: 评估结果表格
    """
    import pandas as pd

    if task_type == "classification":
        metrics = {k: v for k, v in results.items()
                   if isinstance(v, (int, float))}
        return pd.DataFrame(list(metrics.items()), columns=["指标", "值"])
    elif task_type == "regression":
        return pd.DataFrame(list(results.items()), columns=["指标", "值"])
    return pd.DataFrame()