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



def format_evaluation_table(results, task_type="classification"):
    """将评估结果格式化为 HTML 表格（供 Gradio HTML 组件显示）"""
    style = """
    <style>
        .eval-table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }
        .eval-table th {
            background: transparent;
            color: white; padding: 10px 14px; text-align: left; font-weight: 600; border: none;
        }
        .eval-table td { padding: 8px 14px; border-bottom: 1px solid #e2e8f0; }
        .eval-table tr:hover td { background: #f7fafc; }
        .eval-table .metric-name { font-weight: 500; color: #2d3748; }
        .eval-table .metric-value { font-family: 'Courier New', monospace; font-weight: 600; color: #4a5568; text-align: center; }
        .eval-table .metric-desc { color: #718096; font-size: 12px; }
        .eval-table .section-header td {
            background: #edf2f7; font-weight: 700; color: #4a5568;
            padding: 8px 14px; border-bottom: 2px solid #cbd5e0;
        }
        .eval-badge {
            display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 600; margin-left: 6px;
        }
        .eval-badge-good { background: #c6f6d5; color: #276749; }
        .eval-badge-info { background: #bee3f8; color: #2a4365; }
        .eval-badge-warn { background: #fefcbf; color: #744210; }
        .eval-container { padding: 4px 0; }
        .eval-title { font-size: 14px; font-weight: 700; color: #2d3748; margin-bottom: 6px; }
    </style>
    """
    rows_html = ""

    if task_type == "classification":
        metrics = [
            ("\u51c6\u786e\u7387 (Accuracy)", "\u603b\u4f53\u9884\u6d4b\u6b63\u786e\u7387"),
            ("\u7cbe\u786e\u7387 (Precision, macro)", "\u5404\u7c7b\u522b\u7cbe\u786e\u7387\u7684\u7b97\u672f\u5e73\u5747"),
            ("\u53ec\u56de\u7387 (Recall, macro)", "\u5404\u7c7b\u522b\u53ec\u56de\u7387\u7684\u7b97\u672f\u5e73\u5747"),
            ("F1\u5206\u6570 (F1, macro)", "\u7cbe\u786e\u7387\u4e0e\u53ec\u56de\u7387\u7684\u8c03\u548c\u5747\u503c(macro)"),
            ("F1\u5206\u6570 (F1, weighted)", "\u7cbe\u786e\u7387\u4e0e\u53ec\u56de\u7387\u7684\u8c03\u548c\u5747\u503c(weighted)"),
        ]
        rows_html += '<tr class="section-header"><td colspan="3">\u603b\u4f53\u6307\u6807</td></tr>\n'
        for key, desc in metrics:
            val = results.get(key)
            if val is not None:
                badge_class = "eval-badge-good" if val >= 0.8 else "eval-badge-info" if val >= 0.6 else "eval-badge-warn"
                badge_text = "\u4f18\u79c0" if val >= 0.8 else "\u826f\u597d" if val >= 0.6 else "\u5f85\u6539\u8fdb"
                rows_html += (
                    '<tr><td class="metric-name">' + key +
                    '<span class="eval-badge ' + badge_class + '">' + badge_text + '</span></td>'
                    '<td class="metric-value">' + '{:.4f}'.format(val) + '</td>'
                    '<td class="metric-desc">' + desc + '</td></tr>\n'
                )
        per_class = results.get("\u5404\u7c7b\u522b\u6307\u6807", {})
        if per_class:
            rows_html += '<tr class="section-header"><td colspan="3">\u5404\u7c7b\u522b\u6307\u6807</td></tr>\n'
            rows_html += (
                '<tr><td class="metric-name">\u7c7b\u522b</td>'
                '<td class="metric-value">\u7cbe\u786e\u7387</td><td class="metric-value">\u53ec\u56de\u7387</td></tr>\n'
            )
            for cn, m in per_class.items():
                rows_html += (
                    '<tr><td class="metric-name">' + str(cn) + '</td>'
                    '<td class="metric-value">' + '{:.4f}'.format(m["precision"]) + '</td>'
                    '<td class="metric-value">' + '{:.4f}'.format(m["recall"]) + '</td></tr>\n'
                )

    elif task_type == "regression":
        desc_map = {
            "\u5747\u65b9\u8bef\u5dee (MSE)": "\u8d8a\u5c0f\u8d8a\u597d\uff0c\u5bf9\u5f02\u5e38\u503c\u654f\u611f",
            "\u5747\u65b9\u6839\u8bef\u5dee (RMSE)": "\u8d8a\u5c0f\u8d8a\u597d\uff0c\u4e0e\u76ee\u6807\u503c\u540c\u91cf\u7eb2",
            "\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee (MAE)": "\u8d8a\u5c0f\u8d8a\u597d\uff0c\u9c81\u68d2\u6027\u8f83\u5f3a",
            "R\u00b2\u5206\u6570": "\u8d8a\u63a5\u8fd11\u8d8a\u597d\uff0c1\u4e3a\u5b8c\u7f8e\u62df\u5408",
            "\u89e3\u91ca\u65b9\u5dee\u5206": "\u8d8a\u63a5\u8fd11\u8d8a\u597d",
        }
        rows_html += '<tr class="section-header"><td colspan="3">\u56de\u5f52\u6307\u6807</td></tr>\n'
        for key, desc in desc_map.items():
            val = results.get(key)
            if val is not None:
                if "R\u00b2" in key or "\u89e3\u91ca" in key:
                    badge_class = "eval-badge-good" if val >= 0.8 else "eval-badge-info" if val >= 0.5 else "eval-badge-warn"
                    badge_text = "\u4f18\u79c0" if val >= 0.8 else "\u826f\u597d" if val >= 0.5 else "\u5f85\u6539\u8fdb"
                else:
                    badge_class = "eval-badge-info"
                    badge_text = key.split("(")[0].strip() if "(" in key else key[:6]
                rows_html += (
                    '<tr><td class="metric-name">' + key +
                    '<span class="eval-badge ' + badge_class + '">' + badge_text + '</span></td>'
                    '<td class="metric-value">' + '{:.4f}'.format(val) + '</td>'
                    '<td class="metric-desc">' + desc + '</td></tr>\n'
                )

    elif task_type == "unsupervised":
        rows_html += '<tr class="section-header"><td colspan="3">\u805a\u7c7b\u6982\u51b5</td></tr>\n'
        for label in ["\u805a\u7c7b\u6570", "\u566a\u58f0\u70b9\u6570", "\u6837\u672c\u603b\u6570"]:
            val = results.get(label)
            if val is not None:
                rows_html += (
                    '<tr><td class="metric-name">' + label + '</td>'
                    '<td class="metric-value">' + str(val) + '</td>'
                    '<td class="metric-desc"></td></tr>\n'
                )
        rows_html += '<tr class="section-header"><td colspan="3">\u5185\u90e8\u8bc4\u4f30\u6307\u6807\uff08\u65e0\u9700\u771f\u5b9e\u6807\u7b7e\uff09</td></tr>\n'
        internal_metrics = [
            ("\u8f6e\u5ed3\u7cfb\u6570 (Silhouette)", "\u8d8a\u63a5\u8fd11\u8d8a\u597d"),
            ("CH \u6307\u6570 (Calinski-Harabasz)", "\u8d8a\u5927\u8d8a\u597d"),
            ("DB \u6307\u6570 (Davies-Bouldin)", "\u8d8a\u5c0f\u8d8a\u597d"),
        ]
        for key, desc in internal_metrics:
            val = results.get(key)
            if val is not None:
                rows_html += (
                    '<tr><td class="metric-name">' + key + '</td>'
                    '<td class="metric-value">' + '{:.4f}'.format(val) + '</td>'
                    '<td class="metric-desc">' + desc + '</td></tr>\n'
                )
            else:
                rows_html += (
                    '<tr><td class="metric-name">' + key + '</td>'
                    '<td class="metric-value" style="color:#a0aec0;">N/A</td>'
                    '<td class="metric-desc">\u65e0\u6cd5\u8ba1\u7b97</td></tr>\n'
                )
        if "\u540c\u8d28\u5316\u5206\u6570 (Homogeneity)" in results:
            rows_html += '<tr class="section-header"><td colspan="3">\u5916\u90e8\u8bc4\u4f30\u6307\u6807\uff08\u5bf9\u6bd4\u771f\u5b9e\u6807\u7b7e\uff09</td></tr>\n'
            ext_metrics = [
                ("\u540c\u8d28\u5316\u5206\u6570 (Homogeneity)", "[0,1] \u8d8a\u5927\u8d8a\u597d"),
                ("\u5b8c\u6574\u6027\u5206\u6570 (Completeness)", "[0,1] \u8d8a\u5927\u8d8a\u597d"),
                ("V-measure", "\u540c\u8d28\u5316\u4e0e\u5b8c\u6574\u6027\u7684\u8c03\u548c\u5747\u503c"),
            ]
            for key, desc in ext_metrics:
                val = results.get(key)
                if val is not None:
                    rows_html += (
                        '<tr><td class="metric-name">' + key + '</td>'
                        '<td class="metric-value">' + '{:.4f}'.format(val) + '</td>'
                        '<td class="metric-desc">' + desc + '</td></tr>\n'
                    )

    html = (
        '<div class="eval-container">'
        '<div class="eval-title">\u6a21\u578b\u8bc4\u4f30\u62a5\u544a</div>'
        '<table class="eval-table">'
        '<thead><tr><th>\u6307\u6807</th><th>\u6570\u503c</th><th>\u8bf4\u660e</th></tr></thead>'
        '<tbody>' + rows_html + '</tbody>'
        '</table></div>' + style
    )
    return html

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
    return pd.DataFrame(rows) if rows else pd.DataFrame()