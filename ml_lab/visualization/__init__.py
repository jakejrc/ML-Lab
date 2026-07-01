# -*- coding: utf-8 -*-
"""ML-Lab 可视化模块 v3.0 (Package)"""

from ml_lab.visualization.base import (
    _register_cjk_font, _setup_chinese_font,
    plot_data_distribution, plot_preprocessing_comparison,
    plot_training_history, plot_decision_boundary,
    fig_to_image, _diagnose_fit,
)
from ml_lab.visualization.classification import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
)
from ml_lab.visualization.regression import (
    plot_regression_results, plot_gradient_descent_2d,
    plot_regularization_comparison, plot_polynomial_comparison,
    plot_regression_comparison,
)
from ml_lab.visualization.clustering import (
    plot_clustering_scatter, plot_elbow_curve, plot_dendrogram,
    plot_dbscan_eps_analysis, plot_pca_variance, plot_pca_projection,
)
from ml_lab.visualization.evaluation import (
    plot_learning_curve, plot_cross_validation, plot_validation_curve,
    plot_model_comparison, plot_multi_metric_comparison,
    plot_search_results,
)
from ml_lab.visualization.feature import (
    plot_feature_scaling_comparison, plot_feature_scaling_distribution,
    plot_feature_selection_scores, plot_feature_importance_ranking,
    plot_pca_explained_variance, plot_pca_2d_scatter,
    plot_feature_correlation_heatmap, plot_feature_engineering_summary,
    plot_feature_distribution_after_construction,
)
