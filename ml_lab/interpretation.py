# -*- coding: utf-8 -*-
"""ML-Lab Model Interpretation Module"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Chinese font config ----
import matplotlib.font_manager as _fm
_cn_fonts = [f.name for f in _fm.fontManager.ttflist
             if any(kw in f.name for kw in ['WenQuanYi', 'Noto Sans CJK', 'SimHei', 'Microsoft YaHei'])]
if _cn_fonts:
    matplotlib.rcParams['font.sans-serif'] = _cn_fonts + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
del _fm, _cn_fonts
from sklearn.inspection import permutation_importance


def compute_feature_importance(model, feature_names=None):
    try:
        inner = getattr(model, 'model', model)
        if not hasattr(inner, 'feature_importances_'):
            return {"error": "no_feature_importances"}
        importances = inner.feature_importances_
        n_features = len(importances)
        if feature_names is None:
            feature_names = ["F%d" % i for i in range(n_features)]
        else:
            feature_names = list(feature_names)[:n_features]
            while len(feature_names) < n_features:
                feature_names.append("F%d" % len(feature_names))
        indices = np.argsort(importances)[::-1]
        return {
            "importances": importances.tolist(),
            "feature_names": feature_names,
            "sorted_indices": indices.tolist(),
            "sorted_importances": [float(importances[i]) for i in indices],
            "sorted_features": [feature_names[i] for i in indices],
        }
    except Exception as e:
        return {"error": str(e)}


def compute_permutation_importance(estimator, X, y, feature_names=None,
                                    n_repeats=10, scoring=None):
    try:
        inner = getattr(estimator, 'model', estimator)
        result = permutation_importance(
            inner, X, y, n_repeats=n_repeats,
            scoring=scoring, n_jobs=-1, random_state=42
        )
        n_features = len(result.importances_mean)
        if feature_names is None:
            feature_names = ["F%d" % i for i in range(n_features)]
        else:
            feature_names = list(feature_names)[:n_features]
            while len(feature_names) < n_features:
                feature_names.append("F%d" % len(feature_names))
        indices = np.argsort(result.importances_mean)[::-1]
        return {
            "importances_mean": result.importances_mean.tolist(),
            "importances_std": result.importances_std.tolist(),
            "feature_names": feature_names,
            "sorted_indices": indices.tolist(),
            "sorted_importances": [float(result.importances_mean[i]) for i in indices],
            "sorted_std": [float(result.importances_std[i]) for i in indices],
            "sorted_features": [feature_names[i] for i in indices],
        }
    except Exception as e:
        return {"error": str(e)}


def compute_cluster_feature_contribution(X, labels, feature_names=None):
    try:
        from sklearn.feature_selection import f_classif
        f_scores, p_values = f_classif(X, labels)
        n_features = len(f_scores)
        if feature_names is None:
            feature_names = ["F%d" % i for i in range(n_features)]
        else:
            feature_names = list(feature_names)[:n_features]
        indices = np.argsort(f_scores)[::-1]
        return {
            "f_scores": f_scores.tolist(),
            "p_values": p_values.tolist(),
            "feature_names": feature_names,
            "sorted_indices": indices.tolist(),
            "sorted_scores": [float(f_scores[i]) for i in indices],
            "sorted_features": [feature_names[i] for i in indices],
        }
    except Exception as e:
        return {"error": str(e)}


def plot_feature_importance_detailed(result, top_n=15, title="Feature Importance"):
    if "error" in result:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, result["error"],
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title(title)
        return fig

    sorted_features = result["sorted_features"][:top_n]
    sorted_importances = result["sorted_importances"][:top_n]
    has_std = "sorted_std" in result
    sorted_std = result.get("sorted_std", [])[:top_n] if has_std else None

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.4)))

    # Left: bar chart
    ax = axes[0]
    y_pos = range(len(sorted_features))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_features)))
    bars = ax.barh(y_pos, sorted_importances, color=colors[::-1], edgecolor='navy', linewidth=0.5)
    if sorted_std is not None:
        ax.errorbar(sorted_importances, y_pos, xerr=sorted_std,
                    fmt='none', ecolor='gray', capsize=3, alpha=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance Ranking", fontsize=13, fontweight='bold')
    for bar, val in zip(bars, sorted_importances):
        ax.text(bar.get_width() + max(sorted_importances) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                "%.4f" % val, va='center', fontsize=8, color='navy')
    ax.grid(True, alpha=0.3, axis='x')

    # Right: cumulative contribution
    ax = axes[1]
    cumulative = np.cumsum(sorted_importances) / sum(sorted_importances)
    ax.plot(range(1, len(cumulative) + 1), cumulative, 'o-',
            color='#dc2626', linewidth=2, markersize=6)
    ax.axhline(y=0.8, color='#16a34a', linestyle='--', alpha=0.7, label="80% threshold")
    idx_80 = np.argmax(cumulative >= 0.8) + 1 if np.any(cumulative >= 0.8) else len(cumulative)
    ax.axvline(x=idx_80, color='#16a34a', linestyle=':', alpha=0.7)
    ax.annotate("Top-%d reaches 80%%" % idx_80,
                xy=(idx_80, 0.8), xytext=(idx_80 + 1, 0.6),
                fontsize=10, color='#16a34a',
                arrowprops=dict(arrowstyle='->', color='#16a34a'))
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("Cumulative Contribution", fontsize=12)
    ax.set_title("Cumulative Feature Importance", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(cumulative) + 1)
    ax.set_ylim(0, 1.05)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_interpretation_summary(model, X, y, feature_names=None,
                                  top_n=15, title="Model Interpretation Report"):
    result = compute_feature_importance(model, feature_names)
    if "error" not in result:
        return plot_feature_importance_detailed(result, top_n=top_n, title=title)

    if hasattr(model, 'model') and model.model is not None:
        result = compute_permutation_importance(
            model, X, y, feature_names, n_repeats=5
        )
        if "error" not in result:
            return plot_feature_importance_detailed(result, top_n=top_n, title=title)

    labels = getattr(model, 'labels_', None)
    if labels is not None:
        result = compute_cluster_feature_contribution(X, labels, feature_names)
        if "error" not in result:
            return plot_feature_importance_detailed(
                result, top_n=top_n, title="Cluster Feature Contribution"
            )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "Feature importance not available for this model",
            ha='center', va='center', fontsize=14, color='gray')
    ax.set_title(title)
    return fig
