# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.4 — 回调函数
所有数据加载、训练、评估、可视化、导出等回调函数
"""

import sys, os, io, traceback
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gradio as gr

from ml_lab.state import _g, _sync
from ml_lab.preprocessing import (
    load_dataset, get_builtin_datasets, get_dataset_type,
    handle_missing, inject_missing, scale_features, split_dataset, get_dataset_summary,
    load_custom_dataset, register_custom_dataset
)
from ml_lab.algorithms import (
    create_algorithm, get_algorithm_list, get_algorithm_params,
    get_algorithm_type, ALGORITHM_REGISTRY
)
from ml_lab.visualization import (
    plot_data_distribution, plot_preprocessing_comparison,
    plot_decision_boundary, plot_training_history, plot_learning_curve,
    plot_cross_validation,
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_regression_results, plot_gradient_descent_2d, fig_to_image,
    plot_clustering_scatter, plot_elbow_curve, plot_dendrogram,
    plot_model_comparison,
    plot_dbscan_eps_analysis, plot_pca_variance, plot_pca_projection,
    plot_regularization_comparison, plot_polynomial_comparison,
    plot_regression_comparison, plot_search_results,
    plot_feature_scaling_comparison, plot_feature_scaling_distribution,
    plot_feature_selection_scores, plot_feature_importance_ranking,
    plot_pca_explained_variance, plot_pca_2d_scatter,
    plot_feature_correlation_heatmap, plot_feature_engineering_summary,
    plot_feature_distribution_after_construction,
)
from ml_lab.evaluation import (
    evaluate_classification, evaluate_regression, evaluate_clustering,
    format_evaluation_report, format_evaluation_table
)
from ml_lab.feature_engineering import (
    scale_data, get_scaling_stats,
    select_features_univariate, select_features_rfe, select_features_model_based,
    apply_pca, construct_polynomial_features, construct_interaction_features,
    construct_statistical_features, discretize_features, run_feature_pipeline
)
from ml_lab.llm_assistant import ask, reset_conversation, PRESET_QUESTIONS
from ml_lab.code_generator import generate_sklearn_code
from ml_lab.association_rules import (
    AprioriModel, FPGrowthModel, evaluate_association as eval_assoc
)
from ml_lab.association_viz import (
    plot_top_frequent_items, plot_rules_scatter,
    plot_item_length_distribution, plot_rules_heatmap
)
from ml_lab.fptree_viz import plot_fptree
from ml_lab.version import VERSION, FULL_NAME
from ml_lab.report_generator import generate_html_report
from ml_lab.experiment_history import log_experiment, get_experiments, get_statistics
from ml_lab.learning_progress import (record_activity, mark_stage_completed, get_progress_summary, STAGE_MODULE_MAP)

from ml_lab.sandbox_templates import SANDBOX_TEMPLATES
from ml_lab.ui_styles import APP_CSS, _card, ETHICS

def on_load_data(dataset_name, test_ratio):



    try:



        # 强制重新设置中文字体（线程/上下文安全）
        import matplotlib as _mpl
        import os as _os
        _mpl.rcParams['axes.unicode_minus'] = False
        import matplotlib.font_manager as _fm
        _font_registered = False
        # 1. 项目自带 SimHei（动态获取项目根目录，兼容 Docker 和树莓派）
        _project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        _simhei = _os.path.join(_project_root, 'fonts', 'SimHei.ttf')
        if _os.path.isfile(_simhei):
            try:
                _fm.fontManager.addfont(_simhei)
                _mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                _font_registered = True
            except Exception:
                pass
        # 2. Docker Noto CJK SC
        if not _font_registered:
            for _noto in ['/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf', '/usr/share/fonts/opentype/noto/NotoSansCJKSC-Bold.otf']:
                if _os.path.isfile(_noto):
                    try:
                        _fm.fontManager.addfont(_noto)
                        _mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
                        _font_registered = True
                        break
                    except Exception:
                        pass
        # 3. WenQuanYi（树莓派）
        if not _font_registered:
            _wqy = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
            if _os.path.isfile(_wqy):
                try:
                    _fm.fontManager.addfont(_wqy)
                    _mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
                    _font_registered = True
                except Exception:
                    pass
        # 4. 兜底
        if not _font_registered:
            _mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']







        X, y, fn, tn, desc = load_dataset(dataset_name)



        summary = get_dataset_summary(X, y, fn)



        fig = plot_data_distribution(X, y, title=dataset_name, feature_names=fn)



        img = fig_to_image(fig)



        dtype = get_dataset_type(dataset_name)



        if dtype in ("classification", "regression"):



            strat = dtype == "classification"



            Xtr, Xte, ytr, yte = split_dataset(X, y, test_size=float(test_ratio), stratify=strat)



            _sync(Xtr, Xte, ytr, yte, fn, tn, X, y)



            info = ('<table class="info-table"><tbody>'

                    f'<tr><td class="info-label">数据集</td><td>{dataset_name}</td></tr>'

                    f'<tr><td class="info-label">任务类型</td><td>{"分类" if dtype=="classification" else "回归"}</td></tr>'

                    f'<tr><td class="info-label">样本数</td><td>{X.shape[0]}</td></tr>'

                    f'<tr><td class="info-label">特征数</td><td>{X.shape[1]}</td></tr>'

                    f'<tr><td class="info-label">类别数</td><td>{len(np.unique(y))}</td></tr>'

                    f'<tr><td class="info-label">训练集</td><td>{Xtr.shape[0]}</td></tr>'

                    f'<tr><td class="info-label">测试集</td><td>{Xte.shape[0]}</td></tr>'

                    '</tbody></table>')



        else:



            _sync(X, None, y, None, fn, tn, X, y)



            task_label = "关联规则" if dtype == "association" else "聚类 (无监督)"

            info = ('<table class="info-table"><tbody>'

                    f'<tr><td class="info-label">数据集</td><td>{dataset_name}</td></tr>'

                    f'<tr><td class="info-label">任务类型</td><td>{task_label}</td></tr>'

                    f'<tr><td class="info-label">样本数</td><td>{X.shape[0]}</td></tr>'

                    f'<tr><td class="info-label">特征数</td><td>{X.shape[1]}</td></tr>'

                    '</tbody></table>')



        _g["dataset_name"] = dataset_name



        ds_html = (f'<div class="status-card"><div style="display:flex;justify-content:space-between;'



                   f'align-items:center;"><span class="status-label">数据集</span>'



                   f'<span class="status-value" id="sv-ds">{dataset_name}</span></div>'



                   f'<div style="margin-top:4px;"><span class="status-label">任务</span>'



                   f'<span style="font-size:11px;color:rgba(255,255,255,0.7);margin-left:8px;">'



                   f'{"分类" if dtype=="classification" else "回归" if dtype=="regression" else "聚类"}</span></div></div>')



        return img, info, summary, ds_html



    except Exception as e:



        return None, f"加载失败: {e}", None, gr.HTML('<div class="status-card"><span class="status-value">加载失败</span></div>')

def on_preprocess(method, feat_idx):

    if _g["X_train"] is None: return None, "请先加载数据集"

    # 支持 feat_idx 为单个索引或列表
    if isinstance(feat_idx, (list, tuple)):
        idx = int(feat_idx[0]) if feat_idx else 0
    else:
        idx = int(feat_idx)
    idx = min(idx, _g["X_train"].shape[1] - 1)

    Xm = inject_missing(_g["X_train"], 0.1) if not np.isnan(_g["X_train"]).any() else _g["X_train"].copy()

    # 固定使用均值策略填充缺失值（预处理演示目的）
    Xc = handle_missing(Xm, strategy="mean")

    # 将显示名称映射为缩放方法参数
    _scale_map = {
        "标准化 (StandardScaler)": "standard",
        "归一化 (MinMaxScaler)": "minmax",
        "最大绝对值缩放 (MaxAbsScaler)": "maxabs",
        "鲁棒缩放 (RobustScaler)": "robust",
    }
    _scale_method = _scale_map.get(method, "standard")
    Xs, _, _ = scale_features(Xc, method=_scale_method)

    fig = plot_preprocessing_comparison(Xm, Xs, f"缩放方法: {method}", feature_idx=idx)


def on_export_report(eval_html=None):

    """导出当前实验为 HTML 报告"""

    import os, traceback



    try:

        if isinstance(eval_html, dict):

            eval_html = eval_html.get("value", "") or str(eval_html)

        report_text = ""

        if eval_html and str(eval_html).strip():

            report_text = str(eval_html)

        if not report_text:

            report_text = _g.get("last_eval_report", "") or _g.get("last_cluster_report", "")

        if not report_text or str(report_text).strip() in ("", "未训练"):

            logs = _g.get("experiment_log", [])

            if logs:

                report_text = str(logs[-1].get("report", ""))

        if not report_text or not str(report_text).strip():

            return None

        report_text = str(report_text)

        images = []

        for key, img_path in _g.get("report_images", {}).items():

            if img_path is not None:

                images.append((key, img_path))

        algo = _g.get("last_algo_name") or "实验"

        report_dir = "/tmp/gradio"

        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir,

            f"ML-Lab-{algo}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

        info = {

            "title": f"ML-Lab 实验报告 — {algo}",

            "task_type": _g.get("last_task_type", ""),

            "algorithm": algo,

            "dataset_name": _g.get("dataset_name", ""),

            "parameters": {},

            "evaluation_html": report_text,

            "images": images,

            "code": _g.get("last_generated_code", "# 选择算法并训练后生成代码"),

        }

        return generate_html_report(info, save_path=report_path)

    except Exception:
        return None

def on_load_custom_data(file_obj, file_path_text, target_col, task_type, test_ratio):

    """加载用户上传的自定义数据集"""

    import os as _os

    _fp = None

    if file_obj is not None:

        _fp = file_obj if isinstance(file_obj, str) else file_obj.name

    elif file_path_text and os.path.exists(file_path_text):

        _fp = file_path_text

    if _fp is None:

        return None, "请先选择 CSV 或 Excel 文件，或输入本地文件路径", None, gr.HTML('<div class="status-card"><span class="status-value">未加载</span></div>')



    try:

        file_path = _fp

        X, y, fn, tn, desc = load_custom_dataset(file_path, target_col, task_type)



        summary = get_dataset_summary(X, y, fn)

        fig = plot_data_distribution(X, y, title="自定义数据集", feature_names=fn)

        img = fig_to_image(fig)



        ds_name = f"自定义: {os.path.basename(file_path)}"



        _y_valid = y[~np.isnan(y.astype(float))] if np.issubdtype(y.dtype, np.floating) else y

        _unique = len(np.unique(_y_valid))

        dtype = "classification" if _unique <= 20 else "regression"



        if dtype in ("classification", "regression"):

            strat = dtype == "classification"

            Xtr, Xte, ytr, yte = split_dataset(X, y, test_size=float(test_ratio), stratify=strat)

            _sync(Xtr, Xte, ytr, yte, fn, tn, X, y)



            info = ('<table class="info-table"><tbody>'

                    f'<tr><td class="info-label">数据集</td><td>{ds_name}</td></tr>'

                    f'<tr><td class="info-label">任务类型</td><td>{"分类" if dtype=="classification" else "回归"}</td></tr>'

                    f'<tr><td class="info-label">样本数</td><td>{X.shape[0]}</td></tr>'

                    f'<tr><td class="info-label">特征数</td><td>{X.shape[1]}</td></tr>'

                    f'<tr><td class="info-label">类别数</td><td>{len(np.unique(y))}</td></tr>'

                    f'<tr><td class="info-label">训练集</td><td>{Xtr.shape[0]}</td></tr>'

                    f'<tr><td class="info-label">测试集</td><td>{Xte.shape[0]}</td></tr>'

                    '</tbody></table>')

        else:

            _sync(X, None, y, None, fn, tn, X, y)

            info = ('<table class="info-table"><tbody>'

                    f'<tr><td class="info-label">数据集</td><td>{ds_name}</td></tr>'

                    f'<tr><td class="info-label">任务类型</td><td>{dtype}</td></tr>'

                    f'<tr><td class="info-label">样本数</td><td>{X.shape[0]}</td></tr>'

                    f'<tr><td class="info-label">特征数</td><td>{X.shape[1]}</td></tr>'

                    '</tbody></table>')



        _g["dataset_name"] = ds_name



        ds_html = (f'<div class="status-card"><div style="display:flex;justify-content:space-between;'

                   f'align-items:center;"><span class="status-label">数据集</span>'

                   f'<span class="status-value" id="sv-ds">{ds_name}</span></div>'

                   f'<div style="margin-top:4px;"><span class="status-label">任务</span>'

                   f'<span style="font-size:11px;color:rgba(255,255,255,0.7);margin-left:8px;">'

                   f'{"分类" if dtype=="classification" else "回归"}</span></div></div>')



        return img, info, summary, ds_html



    except Exception as e:

        import traceback as _tb

        _tb.print_exc()

        return None, f"加载失败: {e}", None, gr.HTML(f'<div class="status-card"><span class="status-value">加载失败: {e}</span></div>')

def on_load_template(template_name):

    """根据模板名称加载代码沙箱模板"""

    return SANDBOX_TEMPLATES.get(template_name, SANDBOX_TEMPLATES["默认（查看数据）"])
