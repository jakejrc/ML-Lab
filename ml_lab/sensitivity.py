# -*- coding: utf-8 -*-
"""
ML-Lab v3.9 - Parameter Sensitivity Analysis Module
"""

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
from sklearn.model_selection import validation_curve


def compute_validation_curve(estimator, X, y, param_name, param_range,
                             cv=5, scoring=None):
    try:
        train_scores, val_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        return {
            "param_range": list(param_range),
            "param_name": param_name,
            "train_scores_mean": train_scores.mean(axis=1).tolist(),
            "train_scores_std": train_scores.std(axis=1).tolist(),
            "val_scores_mean": val_scores.mean(axis=1).tolist(),
            "val_scores_std": val_scores.std(axis=1).tolist(),
        }
    except Exception as e:
        return {"error": str(e), "param_name": param_name}


def compute_multi_parameter_sensitivity(estimator, X, y, param_grid,
                                         cv=5, scoring=None):
    results = {}
    for param_name, param_range in param_grid.items():
        result = compute_validation_curve(
            estimator, X, y, param_name, param_range, cv=cv, scoring=scoring
        )
        results[param_name] = result
    return results


def plot_parameter_sensitivity(results, title="\u53c2\u6570\u654f\u611f\u6027\u5206\u6790"):
    valid_results = {k: v for k, v in results.items()
                     if "error" not in v}
    if not valid_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "\u53c2\u6570\u654f\u611f\u6027\u5206\u6790\u5931\u8d25\n\u6570\u636e\u96c6\u592a\u5c0f\u6216\u6a21\u578b\u4e0d\u652f\u6301",
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title(title)
        return fig

    n_params = len(valid_results)
    fig, axes = plt.subplots(1, n_params, figsize=(7 * n_params, 5.5))
    if n_params == 1:
        axes = [axes]

    for ax, (param_name, data) in zip(axes, valid_results.items()):
        param_range = data["param_range"]
        train_mean = np.array(data["train_scores_mean"])
        train_std = np.array(data["train_scores_std"])
        val_mean = np.array(data["val_scores_mean"])
        val_std = np.array(data["val_scores_std"])

        is_numeric = all(isinstance(v, (int, float)) for v in param_range)

        ax.fill_between(range(len(param_range)),
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.15, color="#2563eb")
        ax.plot(range(len(param_range)), train_mean, 'o-',
                color="#2563eb", linewidth=2, markersize=6,
                label="\u8bad\u7ec3\u96c6\u5f97\u5206")

        ax.fill_between(range(len(param_range)),
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.15, color="#dc2626")
        ax.plot(range(len(param_range)), val_mean, 's-',
                color="#dc2626", linewidth=2, markersize=6,
                label="\u9a8c\u8bc1\u96c6\u5f97\u5206")

        best_idx = np.argmax(val_mean)
        best_val = param_range[best_idx]
        best_score = val_mean[best_idx]
        ax.axvline(x=best_idx, color='#16a34a', linestyle=':', linewidth=2)

        if is_numeric:
            label_text = "\u6700\u4f73 %s=%s\n\u5f97\u5206=%.4f" % (param_name, str(best_val), best_score)
        else:
            label_text = "\u6700\u4f73=%s\n\u5f97\u5206=%.4f" % (str(best_val), best_score)

        ax.annotate(label_text,
                    xy=(best_idx, best_score),
                    xytext=(best_idx + n_params * 0.8, best_score - 0.08),
                    fontsize=9, color='#16a34a',
                    arrowprops=dict(arrowstyle='->', color='#16a34a'),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4",
                              ec="#16a34a", alpha=0.8))

        if is_numeric:
            step = max(1, len(param_range) // 8)
            tick_indices = list(range(0, len(param_range), step))
            if tick_indices[-1] != len(param_range) - 1:
                tick_indices.append(len(param_range) - 1)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([str(param_range[i]) for i in tick_indices],
                               rotation=30, ha='right', fontsize=9)
        else:
            ax.set_xticks(range(len(param_range)))
            ax.set_xticklabels([str(v) for v in param_range],
                               rotation=30, ha='right', fontsize=9)

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel("\u5f97\u5206", fontsize=12)
        ax.set_title("\u53c2\u6570 %s \u654f\u611f\u6027\u5206\u6790" % param_name,
                     fontsize=13, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_parameter_heatmap(estimator, X, y, param1_name, param1_range,
                            param2_name, param2_range, cv=5, scoring=None):
    try:
        from sklearn.model_selection import GridSearchCV
        param_grid = {param1_name: param1_range, param2_name: param2_range}
        grid = GridSearchCV(estimator, param_grid, cv=cv,
                            scoring=scoring, n_jobs=-1)
        grid.fit(X, y)

        scores = grid.cv_results_["mean_test_score"].reshape(
            len(param2_range), len(param1_range)
        )

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(scores, interpolation='nearest', cmap='RdYlGn',
                       aspect='auto', vmin=0, vmax=1)

        for i in range(len(param2_range)):
            for j in range(len(param1_range)):
                text_color = "white" if scores[i, j] < 0.5 else "black"
                ax.text(j, i, "%.3f" % scores[i, j],
                        ha="center", va="center", fontsize=8,
                        color=text_color)

        best_idx = np.unravel_index(scores.argmax(), scores.shape)
        ax.scatter(best_idx[1], best_idx[0],
                   marker='*', s=200, color='gold',
                   edgecolors='black', linewidths=1, zorder=5)

        ax.set_xticks(range(len(param1_range)))
        ax.set_xticklabels([str(v) for v in param1_range],
                           rotation=30, ha='right')
        ax.set_yticks(range(len(param2_range)))
        ax.set_yticklabels([str(v) for v in param2_range])
        ax.set_xlabel(param1_name, fontsize=12)
        ax.set_ylabel(param2_name, fontsize=12)

        bp = grid.best_params_
        fig.colorbar(im, ax=ax, label="CV avg score", shrink=0.8)
        ax.set_title("Heatmap: %s vs %s\nBest: %s=%s, %s=%s" % (
            param1_name, param2_name,
            param1_name, str(bp[param1_name]),
            param2_name, str(bp[param2_name])),
            fontsize=13, fontweight='bold')

        plt.tight_layout()
        return fig
    except Exception:
        return None


PARAMETER_SWEEP_CONFIG = {
    "K\u8fd1\u90bb": {
        "n_neighbors": list(range(1, 22, 2)),
    },
    "\u51b3\u7b56\u6811": {
        "max_depth": list(range(1, 16, 1)),
    },
    "\u968f\u673a\u68ee\u6797": {
        "n_estimators": [10, 30, 50, 80, 100, 150, 200],
        "max_depth": list(range(1, 12, 1)),
    },
    "\u68af\u5ea6\u63d0\u5347 (GBDT)": {
        "n_estimators": [10, 30, 50, 80, 100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    },
    "\u903b\u8f91\u56de\u5f52": {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    },
    "SVM": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "\u795e\u7ecf\u7f51\u7edc": {
        "max_iter": [50, 100, 150, 200, 300, 400, 500],
        "learning_rate_init": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    },
    "K-Means": {
        "n_clusters": list(range(2, 11)),
    },
}


def get_sweep_params(algo_name):
    return PARAMETER_SWEEP_CONFIG.get(algo_name, {})
