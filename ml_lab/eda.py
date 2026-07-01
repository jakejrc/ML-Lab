
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_cf = [f.name for f in _fm.fontManager.ttflist if 'WenQuanYi' in f.name]
if _cf:
    matplotlib.rcParams['font.sans-serif'] = _cf + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
import pandas as pd


def compute_data_profile(X, y=None, feature_names=None, target_names=None):
    profile = {}
    n_samples, n_features = X.shape
    profile["n_samples"] = n_samples
    profile["n_features"] = n_features

    if feature_names is None:
        feature_names = ["F%d" % i for i in range(n_features)]

    df = pd.DataFrame(X, columns=feature_names[:n_features])
    stats_df = df.describe().T
    stats_df["missing"] = df.isnull().sum().values
    stats_df["missing_pct"] = (df.isnull().sum().values / n_samples * 100)
    stats_df["skewness"] = df.skew().values
    stats_df["kurtosis"] = df.kurtosis().values
    stats_df["range"] = stats_df["max"] - stats_df["min"]
    stats_df["cv"] = stats_df["std"] / (stats_df["mean"].abs() + 1e-10)
    profile["feature_stats"] = stats_df
    profile["feature_names"] = feature_names[:n_features]

    if y is not None:
        profile["has_target"] = True
        if len(np.unique(y)) <= 20:
            classes, counts = np.unique(y, return_counts=True)
            profile["target_type"] = "classification"
            profile["n_classes"] = len(classes)
            profile["class_counts"] = dict(zip(
                [target_names[i] if target_names and i < len(target_names) else str(i) for i in classes],
                counts
            ))
            profile["class_balance"] = float(counts.min() / counts.max()) if counts.max() > 0 else 0.0
        else:
            profile["target_type"] = "regression"
            profile["target_stats"] = {
                "mean": float(np.mean(y)), "std": float(np.std(y)),
                "min": float(np.min(y)), "max": float(np.max(y)),
            }
    else:
        profile["has_target"] = False

    profile["total_missing"] = int(np.isnan(X).sum())
    profile["missing_features"] = [feature_names[i] for i in range(n_features) if np.isnan(X[:, i]).any()]

    if n_features > 1:
        corr = np.corrcoef(X.T)
        profile["correlation_matrix"] = corr
    else:
        profile["correlation_matrix"] = np.array([[1.0]])

    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
    profile["outlier_count"] = int(outlier_mask.sum())
    profile["outlier_pct"] = float(outlier_mask.sum() / (n_samples * n_features) * 100)

    return profile


def plot_eda_summary(profile, top_n=8):
    fn = profile["feature_names"]
    n_features = len(fn)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Data Exploration Summary", fontsize=14, fontweight='bold', y=1.01)

    # 1. Feature stats table
    ax = axes[0, 0]
    ax.axis('off')
    stats = profile["feature_stats"]
    display_cols = ["mean", "std", "min", "25%", "50%", "75%", "max", "range"]
    available = [c for c in display_cols if c in stats.columns]
    table_data = stats[available].head(min(top_n, n_features)).round(3)
    if len(table_data) > 0:
        tbl = ax.table(cellText=table_data.values, rowLabels=table_data.index,
                       colLabels=table_data.columns, cellLoc='center', loc='center', fontsize=8)
        tbl.auto_set_column_width(col=list(range(len(available))))
        for key, cell in tbl.get_celld().items():
            cell.set_fontsize(7)
            if key[0] == 0:
                cell.set_text_props(fontweight='bold')
    ax.set_title("Feature Statistics (top-%d)" % min(top_n, n_features), fontsize=11, fontweight='bold')

    # 2. Correlation heatmap
    ax = axes[0, 1]
    corr = profile["correlation_matrix"]
    if corr.shape[0] <= 30:
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        n_show = min(corr.shape[0], top_n)
        tick_labels = fn[:n_show]
        ax.set_xticks(range(n_show))
        ax.set_yticks(range(n_show))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Correlation")
    else:
        ax.text(0.5, 0.5, "Too many features", ha='center', va='center', fontsize=12)
    ax.set_title("Feature Correlation", fontsize=11, fontweight='bold')

    # 3. Class distribution
    ax = axes[1, 0]
    if profile.get("target_type") == "classification":
        cc = profile["class_counts"]
        classes = list(cc.keys())
        counts = list(cc.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
        bars = ax.bar(range(len(classes)), counts, color=colors, edgecolor='gray')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Class Distribution (balance=%.3f)" % profile["class_balance"], fontsize=11, fontweight='bold')
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                    str(cnt), ha='center', va='bottom', fontsize=9)
    elif profile.get("target_type") == "regression":
        ts = profile["target_stats"]
        ax.text(0.5, 0.5, "Target Stats\nmean=%.3f, std=%.3f\n[%.3f, %.3f]" %
                (ts["mean"], ts["std"], ts["min"], ts["max"]),
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("Target Variable", fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No target (Unsupervised)", ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("Target Analysis", fontsize=11, fontweight='bold')

    # 4. Data quality
    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        "Data Quality Summary", "",
        "Samples: %d" % profile["n_samples"],
        "Features: %d" % profile["n_features"],
        "",
        "Missing: %d (%.2f%%)" % (profile["total_missing"],
          profile["total_missing"] / (profile["n_samples"] * profile["n_features"]) * 100),
        "Outliers: %d (%.2f%%)" % (profile["outlier_count"], profile["outlier_pct"]),
    ]
    if profile.get("target_type") == "classification":
        lines.append("Classes: %d" % profile["n_classes"])
        lines.append("Balance: %.3f" % profile["class_balance"])
    if profile["missing_features"]:
        lines.append("")
        lines.append("Missing features:")
        for mf in profile["missing_features"][:5]:
            lines.append("  - " + mf)
    ax.text(0.1, 0.95, "\n".join(lines), va='top', fontsize=10,
            fontfamily='monospace', transform=ax.transAxes)
    ax.set_title("Data Quality", fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_feature_distributions(X, feature_names=None, max_features=9):
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = ["F%d" % i for i in range(n_features)]
    n_plot = min(n_features, max_features)
    cols = min(3, n_plot)
    rows = int(np.ceil(n_plot / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i in range(n_plot):
        ax = axes[i]
        data = X[:, i]
        ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(data), color='red', linestyle='--', lw=1.5, label='mean')
        ax.axvline(np.median(data), color='green', linestyle=':', lw=1.5, label='median')
        ax.set_xlabel(feature_names[i], fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(n_plot, len(axes)):
        axes[j].axis('off')
    fig.suptitle("Feature Distributions", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
