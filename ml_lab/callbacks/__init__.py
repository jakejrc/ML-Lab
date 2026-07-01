# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.4 — 回调函数包
从 callbacks.py 拆分而来，保持向后兼容
"""

from ml_lab.callbacks.data import (
    on_load_data, on_preprocess,
    on_export_report, on_load_custom_data, on_load_template,
)
from ml_lab.callbacks.train import (
    on_train_classification, on_compare_models,
    on_train_regression, on_train_clustering,
    on_copy_classification_code, on_copy_regression_code,
    on_copy_clustering_code,
    _extract_params,
)
from ml_lab.callbacks.association import (
    on_run_association,
    _format_assoc_table, on_copy_association_code,
)
from ml_lab.callbacks.sandbox import (
    on_download_code, on_sandbox_load_data, on_run_code,
)
from ml_lab.callbacks.feature import (
    on_fe_scaling, on_fe_feature_selection,
    on_fe_pca, on_fe_construct, on_fe_discretize, on_fe_correlation,
)
from ml_lab.callbacks.auto_tuning import on_auto_tune

from ml_lab.callbacks.assistant import (
    on_kg_render, on_chat, on_preset, on_learning_path_click,
    on_ai_test_connection, on_ai_save_config, on_ai_context_chat,
)
