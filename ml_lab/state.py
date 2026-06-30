# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.1 — 全局状态管理
"""

# 全局状态字典
_g = dict(
    X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None,
    feature_names=None, target_names=None, model=None,
    dataset_name="未加载", model_name="未训练",
    report_images={},
    last_eval_report="",
    last_cluster_report="",
    last_algo_name="",
    last_task_type="",
    # 特征工程实验记录
    fe_results={},
    # 多次实验记录（用于报告汇总）
    experiment_log=[],
)


def _sync(X_train, X_test, y_train, y_test, fnames, tnames, X=None, y=None):
    """同步全局状态"""
    _g['X_train'] = X_train
    _g['X_test'] = X_test
    _g['y_train'] = y_train
    _g['y_test'] = y_test
    _g['feature_names'] = fnames
    _g['target_names'] = tnames
    if X is not None:
        _g['X'] = X
    if y is not None:
        _g['y'] = y
