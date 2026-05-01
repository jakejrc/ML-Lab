# -*- coding: utf-8 -*-

"""

ML-Lab v3.4 — 机器学习可视化实验平台

在 v3.1 基础上增强学习路径（进度追踪+核心知识点）。

页面：学习路径 · 数据工作台 · 特征工程 · 分类实验 · 回归实验 · 聚类实验 · 代码沙箱 · AI助教

"""


import sys, os, argparse, io, traceback, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from ml_lab.preprocessing import (

    load_dataset, get_builtin_datasets, get_dataset_type,

    handle_missing, inject_missing, scale_features, split_dataset, get_dataset_summary

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
    plot_learning_curve,

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
from ml_lab.llm_assistant import (

    ask, reset_conversation, get_api_key, PRESET_QUESTIONS

)

from ml_lab.code_generator import generate_sklearn_code


import gradio as gr

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


# ═══════════════════════════════════════════════════════════════

# 全局状态

# ═══════════════════════════════════════════════════════════════

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

    _g.update(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,

              feature_names=fnames, target_names=tnames,

              X=X if X is not None else X_train, y=y if y is not None else y_train)

    # 初始化特征工程结果字典
    if not _g.get("fe_results"):

        _g["fe_results"] = {}

    # 初始化实验日志
    if not _g.get("experiment_log"):

        _g["experiment_log"] = []


# ═══════════════════════════════════════════════════════════════

# 回调 — 数据加载（分类/回归/聚类 统一入口）

# ═══════════════════════════════════════════════════════════════


def on_load_data(dataset_name, test_ratio):

    try:

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

            info = ('<table class="info-table"><tbody>'
                    f'<tr><td class="info-label">数据集</td><td>{dataset_name}</td></tr>'
                    f'<tr><td class="info-label">任务类型</td><td>聚类 (无监督)</td></tr>'
                    f'<tr><td class="info-label">样本数</td><td>{X.shape[0]}</td></tr>'
                    f'<tr><td class="info-label">特征数</td><td>{X.shape[1]}</td></tr>'
                    f'<tr><td class="info-label">簇数</td><td>{len(np.unique(y))}</td></tr>'
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

    idx = min(int(feat_idx), _g["X_train"].shape[1]-1)

    Xm = inject_missing(_g["X_train"], 0.1) if not np.isnan(_g["X_train"]).any() else _g["X_train"].copy()

    Xc = handle_missing(Xm, strategy=method)

    Xs, _, _ = scale_features(Xc, method="standard")

    fig = plot_preprocessing_comparison(Xm, Xs, f"缺失值填充({method})+标准化", feature_idx=idx)

    pp_html = ('<table class="info-table"><tbody>'
                f'<tr><td class="info-label">原始缺失值</td><td>{int(np.isnan(Xm).sum())}</td></tr>'
                f'<tr><td class="info-label">处理后缺失值</td><td>{int(np.isnan(Xs).sum())}</td></tr>'
                f'<tr><td class="info-label">特征{idx} 均值</td><td>{np.nanmean(Xs[:,idx]):.3f}</td></tr>'
                f'<tr><td class="info-label">特征{idx} 标准差</td><td>{np.nanstd(Xs[:,idx]):.3f}</td></tr>'
                '</tbody></table>')
    return fig_to_image(fig), pp_html


# ═══════════════════════════════════════════════════════════════

# 回调 — 分类训练

# ═══════════════════════════════════════════════════════════════


# 需要整数类型的参数名集合（Gradio Slider 返回 float）

INT_PARAMS = {

    'n_iterations', 'max_iter', 'max_depth', 'min_samples_split', 'min_samples_leaf',

    'n_estimators', 'n_neighbors', 'n_clusters', 'hidden_layer_sizes',

    'random_state', 'min_samples', 'max_leaf_nodes', 'degree',

}


def on_train_classification(algo, lr, ni, C, md, mi, hid, al):

    if _g["X_train"] is None: return None, None, "请先加载数据集", None, None, None, None, gr.HTML('')

    try:

        pm = {"learning_rate":lr,"n_iterations":ni,"C":C,"max_depth":md,

              "max_iter":mi,"hidden_layer_sizes":hid,"learning_rate_init":lr,"alpha":al}

        ap = get_algorithm_params(algo); cp = {}

        if ap:

            for p in ap:

                if p in pm and pm[p] is not None:

                    v = pm[p]

                    cp[p] = int(v) if isinstance(v,(int,float)) and p in INT_PARAMS else (float(v) if isinstance(v,(int,float)) else v)

        model = create_algorithm(algo, **cp)

        model.fit(_g["X_train"], _g["y_train"])

        _g["model"], _g["model_name"] = model, algo

        ypred = model.predict(_g["X_test"])

        yprob = None

        if hasattr(model,'predict_proba'):

            try: yprob = model.predict_proba(_g["X_test"])

            except: pass

        res = evaluate_classification(_g["y_test"], ypred, yprob)

        rpt = format_evaluation_table(res, "classification")

        cm = fig_to_image(plot_confusion_matrix(_g["y_test"],ypred,class_names=_g["target_names"]))

        roc = None

        if yprob is not None and yprob.shape[1]>=2:

            roc = fig_to_image(plot_roc_curve(_g["y_test"],yprob,class_names=_g["target_names"]))

        try: db = fig_to_image(plot_decision_boundary(model,_g["X_train"],_g["y_train"],title=f"{algo} 决策边界"))

        except: db = None

        tr = None

        if hasattr(model,'history') and model.history:

            tr = fig_to_image(plot_training_history(model.history,title=f"{algo} 训练过程"))

        # 新增: 学习曲线可视化
        lc_img = None
        cv_img = None

        # 从模型历史中获取学习曲线数据
        if hasattr(model, 'history') and model.history:
            lc_data = model.history.get("learning_curve")
            if lc_data:
                lc_img = fig_to_image(plot_learning_curve(
                    lc_data["train_sizes"],
                    lc_data["train_scores_mean"],
                    lc_data["train_scores_std"],
                    lc_data["val_scores_mean"],
                    lc_data["val_scores_std"],
                    title=f"{algo} 学习曲线"
                ))

            # 交叉验证可视化
            cv_scores = model.history.get("cross_val_scores")
            if cv_scores and len(cv_scores) > 0:
                cv_img = fig_to_image(plot_cross_validation(cv_scores, algo))

        # 如果没有内置学习曲线，手动计算（用底层 sklearn 模型避免 clone 兼容问题）
        if lc_img is None and _g["X_train"].shape[0] >= 50 and hasattr(model, 'model'):
            try:
                from sklearn.model_selection import learning_curve
                train_sizes, train_scores, val_scores = learning_curve(
                    model.model, _g["X_train"], _g["y_train"],
                    train_sizes=np.linspace(0.1, 1.0, 8),
                    cv=min(5, _g["X_train"].shape[0]//2),
                    n_jobs=-1, scoring='accuracy'
                )
                lc_img = fig_to_image(plot_learning_curve(
                    train_sizes.tolist(),
                    train_scores.mean(axis=1).tolist(),
                    train_scores.std(axis=1).tolist(),
                    val_scores.mean(axis=1).tolist(),
                    val_scores.std(axis=1).tolist(),
                    title=f"{algo} 学习曲线"
                ))
            except Exception:
                pass

        # 收集报告数据
        _g["last_eval_report"] = rpt
        _g["last_algo_name"] = algo
        _g["last_task_type"] = "classification"
        _g["report_images"].update({
            "confusion_matrix": cm, "roc_curve": roc,
            "decision_boundary": db, "learning_curve": lc_img,
        })

        # 记录实验日志
        _g["experiment_log"].append({
            "type": "classification", "algorithm": algo, "time": datetime.now().strftime("%H:%M:%S"),
            "report": rpt, "images": dict(_g["report_images"]),
        })

        md_html = (f'<div class="status-card" style="margin-top:6px;"><div style="display:flex;'

                   f'justify-content:space-between;align-items:center;"><span class="status-label">模型</span>'

                   f'<span class="status-value" id="sv-md">{algo}</span></div></div>')

        return tr, db, rpt, cm, roc, lc_img, cv_img, md_html

    except Exception as e:

        return None,None,f"训练失败: {e}\n{traceback.format_exc()}",None,None,None,None,gr.HTML('')


# ═══════════════════════════════════════════════════════════════
# 回调 — 模型对比
# ═══════════════════════════════════════════════════════════════

def on_compare_models(selected_algos):
    """多模型对比训练"""
    if _g["X_train"] is None:
        return None, "请先加载数据集"
    if not selected_algos:
        return None, "请至少选择一个算法"

    try:
        from sklearn.model_selection import cross_val_score
        import time

        results = []
        train_scores = []
        test_scores = []
        cv_scores_list = []
        model_names = []

        # 默认参数配置
        default_params = {
            "逻辑回归": {"C": 1.0, "max_iter": 200},
            "K近邻": {"n_neighbors": 5},
            "决策树": {"max_depth": 5, "criterion": "gini"},
            "SVM": {"C": 1.0, "kernel": "rbf"},
            "朴素贝叶斯": {},
            "随机森林": {"n_estimators": 100, "max_depth": 5},
            "梯度提升 (GBDT)": {"n_estimators": 100, "learning_rate": 0.1},
            "神经网络": {"hidden_layer_sizes": "64,32", "max_iter": 200}
        }

        for algo in selected_algos:
            try:
                start_time = time.time()
                params = default_params.get(algo, {})
                model = create_algorithm(algo, **params)
                model.fit(_g["X_train"], _g["y_train"])

                train_score = model.score(_g["X_train"], _g["y_train"]) if hasattr(model, 'score') else 0
                test_score = model.score(_g["X_test"], _g["y_test"]) if hasattr(model, 'score') else 0

                # 交叉验证
                if hasattr(model, 'model') and model.model is not None:
                    cv_scores = cross_val_score(model.model, _g["X_train"], _g["y_train"], cv=5)
                    cv_mean = cv_scores.mean()
                else:
                    cv_mean = test_score

                elapsed = time.time() - start_time

                results.append({
                    "算法": algo,
                    "训练集得分": f"{train_score:.4f}",
                    "测试集得分": f"{test_score:.4f}",
                    "交叉验证": f"{cv_mean:.4f}",
                    "耗时(s)": f"{elapsed:.2f}"
                })

                train_scores.append(train_score)
                test_scores.append(test_score)
                cv_scores_list.append(cv_mean)
                model_names.append(algo)

            except Exception as e:
                results.append({
                    "算法": algo,
                    "训练集得分": "错误",
                    "测试集得分": "错误",
                    "交叉验证": "错误",
                    "耗时(s)": str(e)[:20]
                })

        # 生成对比图表
        if model_names:
            compare_img = fig_to_image(plot_model_comparison(
                model_names, train_scores, test_scores, cv_scores_list
            ))
        else:
            compare_img = None

        # 格式化结果文本
        result_text = "📊 模型对比结果\n" + "="*50 + "\n"
        result_text += f"{'算法':<15} {'训练集':<10} {'测试集':<10} {'交叉验证':<10} {'耗时'}\n"
        result_text += "-"*50 + "\n"
        for r in results:
            result_text += f"{r['算法']:<15} {r['训练集得分']:<10} {r['测试集得分']:<10} {r['交叉验证']:<10} {r['耗时(s)']}s\n"

        # 找出最佳模型
        if cv_scores_list:
            best_idx = max(range(len(cv_scores_list)), key=lambda i: cv_scores_list[i] if isinstance(cv_scores_list[i], (int, float)) else 0)
            result_text += "\n" + "="*50
            result_text += f"\n🏆 最佳模型: {model_names[best_idx]} (交叉验证得分: {cv_scores_list[best_idx]:.4f})"

        return compare_img, result_text

    except Exception as e:
        return None, f"对比失败: {e}\n{traceback.format_exc()}"


# ═══════════════════════════════════════════════════════════════

# 回调 — 回归训练

# ═══════════════════════════════════════════════════════════════


def on_train_regression(algo, lr, ni, C, md, mi, hid, al, deg, eps, kern, crit):

    if _g["X_train"] is None: return None, None, None, None, None, "请先加载数据集", gr.HTML('')

    try:

        pm = {"learning_rate":lr,"n_iterations":ni,"C":C,"max_depth":md,

              "max_iter":mi,"hidden_layer_sizes":hid,"learning_rate_init":lr,"alpha":al,

              "degree":deg,"epsilon":eps,"kernel":kern,"criterion":crit}

        ap = get_algorithm_params(algo); cp = {}

        if ap:

            for p in ap:

                if p in pm and pm[p] is not None:

                    v = pm[p]; cp[p] = int(v) if isinstance(v,(int,float)) and p in INT_PARAMS else (float(v) if isinstance(v,(int,float)) else v)

        model = create_algorithm(algo, **cp)

        model.fit(_g["X_train"], _g["y_train"])

        _g["model"], _g["model_name"] = model, algo

        ypred = model.predict(_g["X_test"])

        res = evaluate_regression(_g["y_test"],ypred)

        rpt = format_evaluation_table(res, "regression")

        rg = fig_to_image(plot_regression_results(_g["y_test"],ypred,title=f"{algo} 回归结果"))

        gd = None

        if _g["X_train"].shape[1]>=1 and hasattr(model,'history'):

            gd = fig_to_image(plot_gradient_descent_2d(_g["X_train"][:,0],_g["y_train"],

                learning_rate=cp.get('learning_rate',0.01)))

        # ── 新增：正则化对比可视化 ──

        reg_cmp = None

        if algo in ("Ridge 回归", "Lasso 回归"):

            try:

                reg_cmp = fig_to_image(plot_regularization_comparison(

                    _g["X_train"], _g["y_train"], _g["X_test"], _g["y_test"],

                    alphas=[0.01, 0.1, 1.0, 10.0, 50.0, 100.0],

                    title=f"{algo} 正则化强度对比"))

            except Exception: pass

        # ── 新增：多项式阶数对比可视化 ──

        poly_cmp = None

        if algo == "多项式回归":

            try:

                poly_cmp = fig_to_image(plot_polynomial_comparison(

                    _g["X_train"], _g["y_train"], _g["X_test"], _g["y_test"],

                    max_degree=int(cp.get('degree', 2)) + 2,

                    title="多项式回归阶数对比"))

            except Exception: pass

        # ── 新增：回归模型全面对比 ──

        model_cmp = None

        if algo in ("线性回归", "Ridge 回归", "Lasso 回归"):

            try:

                from ml_lab.algorithms import (LinearRegressionModel, RidgeRegressionModel,

                                         LassoRegressionModel, SVRModel,

                                         DecisionTreeRegressorModel)

                cmp_algos = []

                cmp_algos.append({"name": "线性回归", "model": LinearRegressionModel()})

                cmp_algos.append({"name": "Ridge 回归", "model": RidgeRegressionModel(alpha=1.0)})

                cmp_algos.append({"name": "Lasso 回归", "model": LassoRegressionModel(alpha=1.0)})

                for ca in cmp_algos:

                    try: ca["model"].fit(_g["X_train"], _g["y_train"])

                    except: pass

                model_cmp = fig_to_image(plot_regression_comparison(

                    _g["X_train"], _g["y_train"], _g["X_test"], _g["y_test"],

                    cmp_algos, title="线性/Ridge/Lasso 回归对比"))

            except Exception: pass

        # 收集报告数据
        _g["last_eval_report"] = rpt
        _g["last_algo_name"] = algo
        _g["last_task_type"] = "regression"
        _g["report_images"].update({
            "regression_results": rg, "gradient_descent": gd,
            "regularization": reg_cmp, "polynomial": poly_cmp,
        })

        # 记录实验日志
        _g["experiment_log"].append({
            "type": "regression", "algorithm": algo, "time": datetime.now().strftime("%H:%M:%S"),
            "report": rpt, "images": dict(_g["report_images"]),
        })

        md_html = (f'<div class="status-card" style="margin-top:6px;"><div style="display:flex;'

                   f'justify-content:space-between;align-items:center;"><span class="status-label">模型</span>'

                   f'<span class="status-value" id="sv-md">{algo}</span></div></div>')

        return gd, rg, reg_cmp, poly_cmp, model_cmp, rpt, md_html

    except Exception as e:

        return None,None,None,None,None,f"训练失败: {e}\n{traceback.format_exc()}",gr.HTML('')


# ═══════════════════════════════════════════════════════════════

# 回调 — 聚类训练（v3.0 新增）

# ═══════════════════════════════════════════════════════════════


def on_train_clustering(algo, n_clusters, max_iter, eps, min_samples,

                         linkage_type, affinity, init_method):

    if _g["X"] is None: return None, None, None, "请先加载数据集", gr.HTML('')

    try:

        pm = {}

        if algo == "K-Means":

            pm = {"n_clusters": int(n_clusters), "max_iter": int(max_iter), "init": init_method}

        elif algo == "DBSCAN":

            pm = {"eps": float(eps), "min_samples": int(min_samples)}

        elif algo == "层次聚类":

            pm = {"n_clusters": int(n_clusters), "linkage": linkage_type, "affinity": affinity}

        elif algo == "PCA":

            pm = {"n_components": int(n_clusters)}


        model = create_algorithm(algo, **pm)

        model.fit(_g["X"], _g["y"])

        _g["model"], _g["model_name"] = model, algo

        _labels = getattr(model, 'labels_', None)

        labels = _labels if _labels is not None else (model.predict(_g["X"]) if hasattr(model,'predict') else _g["y"])


        # 算法专属图

        img2 = None

        if algo == "PCA":
            # PCA: labels is 2D projection, use dedicated projection plot
            X_pca = model.transform(_g["X"])
            labels_for_pca = _g["y"]
            fig_pca = plot_pca_projection(X_pca, labels=labels_for_pca, title="PCA 投影图")
            img1 = fig_to_image(fig_pca)
            evr = model.history.get("explained_variance_ratio", [])
            cv = model.history.get("cumulative_variance", [])
            if evr:
                pca_hist = {"full_explained_variance_ratio": evr, "full_cumulative_variance": cv}
                img2 = fig_to_image(plot_pca_variance(pca_hist, title="PCA 方差解释率"))
        else:
            # 通用聚类散点图（K-Means / DBSCAN / 层次聚类）
            centers = getattr(model, 'model', None)
            centers_coords = centers.cluster_centers_ if hasattr(centers, 'cluster_centers_') else None
            fig_scatter = plot_clustering_scatter(_g["X"], labels, title=f"{algo} 聚类结果",
                                                  centers=centers_coords, feature_names=_g["feature_names"])
            img1 = fig_to_image(fig_scatter)

            # 算法专属第二图
            if algo == "K-Means" and model.history.get("inertia"):
                img2 = fig_to_image(plot_elbow_curve(model.history, title="K-Means 肘部法则"))
            elif algo == "DBSCAN" and model.history.get("eps_range"):
                img2 = fig_to_image(plot_dbscan_eps_analysis(model.history, title="DBSCAN eps 参数分析"))
            elif algo == "层次聚类":
                img2 = fig_to_image(plot_dendrogram(_g["X"], title="层次聚类树状图"))


        # 评估

        from ml_lab.evaluation import evaluate_clustering, format_evaluation_report

        eval_labels = _g["y"] if algo == "PCA" else labels
        eval_res = evaluate_clustering(_g["X"], eval_labels, y_true=_g["y"])

        rpt = format_evaluation_table(eval_res, "unsupervised")


        # 收集报告数据
        _g["last_cluster_report"] = rpt
        _g["last_algo_name"] = algo
        _g["last_task_type"] = "clustering"
        _g["report_images"].update({
            "cluster_scatter": img1, "cluster_analysis": img2,
        })

        md_html = (f'<div class="status-card" style="margin-top:6px;"><div style="display:flex;'

                   f'justify-content:space-between;align-items:center;"><span class="status-label">模型</span>'

                   f'<span class="status-value" id="sv-md">{algo}</span></div></div>')

        return img1, img2, None, rpt, md_html

    except Exception as e:

        return None,None,None,f"训练失败: {e}\n{traceback.format_exc()}",gr.HTML('')


# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# 回调 — 一键复制代码（v3.4 新增）
# ═══════════════════════════════════════════════════════════════

def _extract_params(task_type, **kwargs):
    """从 Gradio 输入中提取当前实验的有效参数"""
    if task_type == "classification":
        algo = kwargs.get("algo")
        pm = {"learning_rate": kwargs.get("lr"), "n_iterations": kwargs.get("ni"),
              "C": kwargs.get("C"), "max_depth": kwargs.get("md"),
              "max_iter": kwargs.get("mi"), "hidden_layer_sizes": kwargs.get("hid"),
              "learning_rate_init": kwargs.get("lr"), "alpha": kwargs.get("al")}
        ap = get_algorithm_params(algo) if algo else {}
        cp = {}
        if ap:
            for p in ap:
                if p in pm and pm[p] is not None:
                    v = pm[p]
                    if isinstance(v, (int, float)) and p in INT_PARAMS:
                        cp[p] = int(v)
                    elif isinstance(v, (int, float)):
                        cp[p] = float(v)
                    else:
                        cp[p] = v
        return cp
    elif task_type == "regression":
        algo = kwargs.get("algo")
        pm = {"learning_rate": kwargs.get("lr"), "n_iterations": kwargs.get("ni"),
              "C": kwargs.get("C"), "max_depth": kwargs.get("md"),
              "max_iter": kwargs.get("mi"), "hidden_layer_sizes": kwargs.get("hid"),
              "learning_rate_init": kwargs.get("lr"), "alpha": kwargs.get("al"),
              "degree": kwargs.get("deg"), "epsilon": kwargs.get("eps"),
              "kernel": kwargs.get("kern"), "criterion": kwargs.get("crit")}
        ap = get_algorithm_params(algo) if algo else {}
        cp = {}
        if ap:
            for p in ap:
                if p in pm and pm[p] is not None:
                    v = pm[p]
                    if isinstance(v, (int, float)) and p in INT_PARAMS:
                        cp[p] = int(v)
                    elif isinstance(v, (int, float)):
                        cp[p] = float(v)
                    else:
                        cp[p] = v
        return cp
    elif task_type == "clustering":
        algo = kwargs.get("algo")
        pm = {"n_clusters": kwargs.get("n_clusters"), "max_iter": kwargs.get("max_iter"),
              "eps": kwargs.get("eps"), "min_samples": kwargs.get("min_samples"),
              "linkage": kwargs.get("linkage"), "affinity": kwargs.get("affinity"),
              "init": kwargs.get("init")}
        cp = {}
        if algo == "K-Means":
            for k in ["n_clusters", "max_iter", "init"]:
                if pm.get(k) is not None:
                    cp[k] = int(pm[k]) if isinstance(pm[k], (int, float)) else pm[k]
        elif algo == "DBSCAN":
            for k in ["eps", "min_samples"]:
                if pm.get(k) is not None:
                    cp[k] = float(pm[k]) if k == "eps" else int(pm[k])
        elif algo == "层次聚类":
            for k in ["n_clusters", "linkage", "affinity"]:
                if pm.get(k) is not None:
                    cp[k] = int(pm[k]) if k == "n_clusters" else pm[k]
        elif algo == "PCA":
            n = pm.get("n_clusters")
            if n is not None:
                cp["n_components"] = int(n)
        return cp
    return {}


def on_copy_classification_code(algo, lr, ni, C, md, mi, hid, al):
    """生成分类实验的可复制代码"""
    try:
        params = _extract_params("classification",
            algo=algo, lr=lr, ni=ni, C=C, md=md, mi=mi, hid=hid, al=al)
        code = generate_sklearn_code("classification", algo, params,
                                     _g.get("dataset_name", "未加载"), test_ratio=0.3)
        _g["last_generated_code"] = code
        return code
    except Exception as e:
        return f"代码生成失败: {e}"


def on_copy_regression_code(algo, lr, ni, C, md, mi, hid, al, deg, eps, kern, crit):
    """生成回归实验的可复制代码"""
    try:
        params = _extract_params("regression",
            algo=algo, lr=lr, ni=ni, C=C, md=md, mi=mi, hid=hid, al=al,
            deg=deg, eps=eps, kern=kern, crit=crit)
        code = generate_sklearn_code("regression", algo, params,
                                     _g.get("dataset_name", "未加载"), test_ratio=0.3)
        _g["last_generated_code"] = code
        return code
    except Exception as e:
        return f"代码生成失败: {e}"


def on_copy_clustering_code(algo, n_clusters, max_iter, eps, min_samples,
                            linkage, affinity, init_method):
    """生成聚类实验的可复制代码"""
    try:
        params = _extract_params("clustering",
            algo=algo, n_clusters=n_clusters, max_iter=max_iter,
            eps=eps, min_samples=min_samples, linkage=linkage,
            affinity=affinity, init=init_method)
        code = generate_sklearn_code("clustering", algo, params,
                                     _g.get("dataset_name", "未加载"), test_ratio=0.3)
        _g["last_generated_code"] = code
        return code
    except Exception as e:
        return f"代码生成失败: {e}"

# 回调 — 代码沙箱（v3.0 新增）

# ═══════════════════════════════════════════════════════════════


SANDBOX_TEMPLATE = """# ML-Lab 代码沙箱 — 在此编写和运行 Python 代码

# 已预导入: numpy (as np), pandas (as pd), sklearn, matplotlib

# 已加载数据集变量: X, y, feature_names, target_names


import numpy as np

import pandas as pd


# 示例: 查看数据形状

print("数据形状:", X.shape)

print("前5行:\\n", pd.DataFrame(X[:5], columns=feature_names))

"""


def on_run_code(code):

    """安全执行用户代码，捕获输出"""

    if not code.strip():

        return "请输入代码"

    old_stdout = sys.stdout

    old_stderr = sys.stderr

    sys.stdout = io.StringIO()

    sys.stderr = io.StringIO()

    # 数据未加载时注入提示变量，避免 AttributeError

    if _g["X"] is None:

        return "⚠️ 请先前往【数据工作台】加载数据集，再使用代码沙箱。\n已预导入的库: numpy, pandas, sklearn, matplotlib 可直接使用。"

    local_ns = {"np": np, "pd": __import__("pandas"),

                "X": _g["X"], "y": _g["y"],

                "feature_names": _g["feature_names"],

                "target_names": _g["target_names"]}

    local_ns["matplotlib"] = __import__("matplotlib")

    local_ns["plt"] = local_ns["matplotlib"].pyplot

    local_ns["sklearn"] = __import__("sklearn")

    local_ns["X_train"] = _g["X_train"]

    local_ns["y_train"] = _g["y_train"]

    local_ns["X_test"] = _g["X_test"]

    local_ns["y_test"] = _g["y_test"]


    try:

        exec(code, {"__builtins__": __builtins__}, local_ns)

        output = sys.stdout.getvalue()

        errors = sys.stderr.getvalue()

        result = output

        if errors.strip():

            result += "\n[stderr]\n" + errors

        return result if result.strip() else "(代码执行成功，无输出)"

    except Exception as e:

        return f"错误: {type(e).__name__}: {e}"

    finally:

        sys.stdout = old_stdout

        sys.stderr = old_stderr


# ═══════════════════════════════════════════════════
# 回调 — 特征工程工作台
# ═══════════════════════════════════════════════════

def on_fe_scaling(method):
    """模块1: 特征缩放"""
    if _g["X_train"] is None:
        return None, None, "请先在「数据工作台」加载数据集"
    try:
        X_tr = _g["X_train"].copy()
        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]
        X_scaled, _, scaler = scale_data(X_tr, method=method)
        stats = get_scaling_stats(X_tr, X_scaled, fnames)
        img1 = fig_to_image(plot_feature_scaling_comparison(
            X_tr, X_scaled, feature_names=fnames,
            title=f"特征缩放对比 ({method})"))
        img2 = fig_to_image(plot_feature_scaling_distribution(
            X_tr, X_scaled, feature_idx=0, feature_name=fnames[0],
            title=f"特征分布对比: {fnames[0]}"))
        # 构建 HTML 表格
        scale_style = """
        <style>
            .scale-table { width:100%; border-collapse:collapse; margin:6px 0; font-size:13px; }
            .scale-table th { background:transparent; color:#2d3748; padding:8px 10px; text-align:center; font-weight:600; border-bottom:2px solid #e2e8f0; }
            .scale-table td { padding:7px 10px; border-bottom:1px solid #edf2f7; text-align:center; }
            .scale-table td:first-child { text-align:left; font-weight:500; color:#4a5568; }
            .scale-table tr:hover td { background:#f7fafc; }
            .scale-meta { font-size:12px; color:#718096; margin-bottom:8px; }
            .scale-meta span { margin-right:14px; }
        </style>"""
        rows = ""
        for s in stats:
            rows += (f'<tr><td>{s["\u7279\u5f81"]}</td>'
                     f'<td>{s["\u7f29\u653e\u524d_\u5747\u503c"]:.4f}</td>'
                     f'<td>{s["\u7f29\u653e\u540e_\u5747\u503c"]:.4f}</td>'
                     f'<td>{s["\u7f29\u653e\u524d_\u6807\u51c6\u5dee"]:.4f}</td>'
                     f'<td>{s["\u7f29\u653e\u540e_\u6807\u51c6\u5dee"]:.4f}</td></tr>\n')
        html = (f'<div class="scale-meta">'
                f'<span>\u7f29\u653e\u65b9\u6cd5: <b>{method}</b></span>'
                f'<span>\u7279\u5f81\u6570: <b>{X_tr.shape[1]}</b></span>'
                f'<span>\u6837\u672c\u6570: <b>{X_tr.shape[0]}</b></span></div>'
                f'<table class="scale-table"><thead><tr>'
                f'<th>\u7279\u5f81</th><th>\u7f29\u653e\u524d\u5747\u503c</th><th>\u7f29\u653e\u540e\u5747\u503c</th>'
                f'<th>\u7f29\u653e\u524d\u6807\u51c6\u5dee</th><th>\u7f29\u653e\u540e\u6807\u51c6\u5dee</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>{scale_style}')
        return img1, img2, html
        # 保存特征工程结果用于报告
        _g["fe_results"]["scaling"] = {"method": method, "n_features": X_tr.shape[1], "n_samples": X_tr.shape[0], "image1": img1, "image2": img2, "info": "\n".join(lines)}
        return img1, img2, "\n".join(lines)
    except Exception as e:
        return None, None, f"错误: {e}"


def on_fe_feature_selection(method, k):
    """模块2: 特征选择"""
    if _g["X_train"] is None:
        return None, "请先在「数据工作台」加载数据集"
    if _g["y_train"] is None:
        return None, "标签数据不可用"
    try:
        X_tr, y_tr = _g["X_train"], _g["y_train"]
        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]
        dtype = get_dataset_type(_g["dataset_name"])
        task = "classification" if dtype == "classification" else "regression"
        k = int(k)
        k = min(k, X_tr.shape[1])

        img = None
        sel_css = """
        <style>
            .sel-table { width:100%; border-collapse:collapse; margin:6px 0; font-size:13px; }
            .sel-table th { background:transparent; color:#2d3748; padding:8px 10px; text-align:center; font-weight:600; border-bottom:2px solid #e2e8f0; }
            .sel-table td { padding:7px 10px; border-bottom:1px solid #edf2f7; text-align:center; }
            .sel-table td:first-child { text-align:left; font-weight:500; color:#4a5568; }
            .sel-table tr:hover td { background:#f7fafc; }
            .sel-table .sel-yes { color:#38a169; font-weight:700; }
            .sel-meta { font-size:12px; color:#718096; margin-bottom:8px; }
            .sel-meta span { margin-right:14px; }
        </style>"""
        sel_rows = ""

        if method in ("f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression"):
            X_sel, indices, scores, pvals = select_features_univariate(
                X_tr, y_tr, method=method, k=k, task_type=task)
            score_name = {"f_classif": "F值", "f_regression": "F值",
                          "mutual_info_classif": "互信息", "mutual_info_regression": "互信息"}.get(method, "得分")
            img = fig_to_image(plot_feature_selection_scores(
                scores, feature_names=fnames, top_n=None,
                title=f"特征选择得分 ({method})", score_name=score_name))
            p_header = ""
            if pvals is not None:
                p_header = "<th>p值</th>"
            sel_rows += f"<tr><th>排名</th><th>特征</th><th>{score_name}</th><th>选中</th>{p_header}</tr>"
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(ranked[:k], 1):
                p_str = f"<td>{pvals[i]:.4f}</td>" if pvals is not None else ""
                sel_rows += (f'<tr><td>{rank}</td><td>{fnames[i]}</td>'
                             f'<td>{scores[i]:.4f}</td>'
                             f'<td class="sel-yes">✓</td>{p_str}</tr>\n')
            for rank, i in enumerate(ranked[k:], k+1):
                p_str = f"<td>{pvals[i]:.4f}</td>" if pvals is not None else ""
                sel_rows += (f'<tr><td>{rank}</td><td>{fnames[i]}</td>'
                             f'<td>{scores[i]:.4f}</td><td></td>{p_str}</tr>\n')
        elif method == "rfe":
            X_sel, indices, ranking = select_features_rfe(X_tr, y_tr, n_features=k, task_type=task)
            img = fig_to_image(plot_feature_importance_ranking(
                1.0 / np.array(ranking, dtype=float), feature_names=fnames,
                title="RFE 特征排名 (1/排名)"))
            sel_rows += "<tr><th>特征</th><th>排名</th><th>选中</th></tr>"
            for i, r in enumerate(ranking):
                mark = '<td class="sel-yes">✓</td>' if r == 1 else '<td></td>'
                sel_rows += f'<tr><td>{fnames[i]}</td><td>{r}</td>{mark}</tr>\n'
        elif method == "model_based":
            X_sel, indices, importances = select_features_model_based(X_tr, y_tr, task_type=task)
            img = fig_to_image(plot_feature_importance_ranking(
                importances, feature_names=fnames,
                title="RandomForest 特征重要性"))
            sel_rows += "<tr><th>排名</th><th>特征</th><th>重要性</th><th>选中</th></tr>"
            ranked = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
            for rank, i in enumerate(ranked, 1):
                mark = '<td class="sel-yes">✓</td>' if i in indices else '<td></td>'
                sel_rows += f'<tr><td>{rank}</td><td>{fnames[i]}</td><td>{importances[i]:.4f}</td>{mark}</tr>\n'

        selected_names = [fnames[i] for i in indices]
        info_html = (f'<div class="sel-meta">'
                     f'<span>方法: <b>{method}</b></span>'
                     f'<span>原始特征数: <b>{X_tr.shape[1]}</b></span>'
                     f'<span>选择特征数: <b>{k}</b></span></div>'
                     f'<table class="sel-table"><thead><tr>{sel_rows[:sel_rows.find("</tr>")+5]}</tr></thead>'
                     f'<tbody>{sel_rows[sel_rows.find("</tr>")+5:]}</tbody></table>'
                     f'<div style="margin-top:8px;font-size:12px;color:#4a5568;">'
                     f'选中特征: <b>{", ".join(selected_names)}</b></div>{sel_css}')
        return img, info_html
    except Exception as e:
        return None, f"错误: {e}"


def on_fe_pca(n):
    """模块3: PCA降维"""
    if _g["X_train"] is None:
        return None, None, "请先在「数据工作台」加载数据集"
    try:
        X_tr, y_tr = _g["X_train"], _g["y_train"]
        n = int(n)
        n = min(n, X_tr.shape[1])
        X_pca, pca_model, evr = apply_pca(X_tr, n_components=n)
        img1 = fig_to_image(plot_pca_explained_variance(evr, title="PCA 方差解释率"))
        X_2d = X_pca[:, :2] if X_pca.shape[1] >= 2 else X_pca
        img2 = fig_to_image(plot_pca_2d_scatter(X_2d, y_tr, title="PCA 2D 投影"))
        cumsum = np.cumsum(evr)
        rows = ""
        for i in range(len(evr)):
            rows += "<tr><td>PC" + str(i+1) + "</td><td>" + f"{evr[i]:.4f}" + "</td><td>" + f"{cumsum[i]:.4f}" + "</td></tr>\n"
        html = (
            '<table class="info-table"><thead><tr>'
            '<th>主成分</th><th>方差解释率</th><th>累计解释率</th>'
            '</tr></thead><tbody>'
            + rows
            + '</tbody></table>'
            '<div style="margin-top:8px;color:#888;font-size:0.9em;">'
            + '前' + str(n) + '个主成分累计解释方差: ' + f"{cumsum[-1]:.4f}" + ' (' + f"{cumsum[-1]*100:.1f}" + '%)'
            + '</div>'
        )
        return img1, img2, html
    except Exception as e:
        return None, None, f"错误: {e}"


def on_fe_construct(method, degree):
    """模块4: 特征构造"""
    if _g["X_train"] is None:
        return None, "请先在「数据工作台」加载数据集"
    try:
        X_tr = _g["X_train"].copy()
        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]
        degree = int(degree)
        orig_n = X_tr.shape[1]

        if method == "polynomial":
            X_new, _ = construct_polynomial_features(X_tr, degree=degree)
            new_names = [f"P{d}" for d in range(X_new.shape[1])]
            info = (
                '<table class="info-table"><tbody>'
                '<tr><td style="font-weight:600;width:120px;">方法</td><td>多项式特征 (degree=' + str(degree) + ')</td></tr>'
                '<tr><td style="font-weight:600;">原始特征数</td><td>' + str(orig_n) + '</td></tr>'
                '<tr><td style="font-weight:600;">构造后特征数</td><td>' + str(X_new.shape[1]) + '</td></tr>'
                '</tbody></table>'
            )
        elif method == "interaction":
            X_new, new_names = construct_interaction_features(X_tr)
            info = (
                '<table class="info-table"><tbody>'
                '<tr><td style="font-weight:600;width:120px;">方法</td><td>交互特征 (两两相乘)</td></tr>'
                '<tr><td style="font-weight:600;">原始特征数</td><td>' + str(orig_n) + '</td></tr>'
                '<tr><td style="font-weight:600;">新增交互特征</td><td>' + str(len(new_names)) + '</td></tr>'
                '<tr><td style="font-weight:600;">构造后特征数</td><td>' + str(X_new.shape[1]) + '</td></tr>'
                '</tbody></table>'
            )
        elif method == "statistical":
            X_new, new_names = construct_statistical_features(X_tr)
            info = (
                '<table class="info-table"><tbody>'
                '<tr><td style="font-weight:600;width:120px;">方法</td><td>统计特征 (行统计量)</td></tr>'
                '<tr><td style="font-weight:600;">新增特征</td><td>' + ', '.join(new_names) + '</td></tr>'
                '<tr><td style="font-weight:600;">原始特征数</td><td>' + str(orig_n) + '</td></tr>'
                '<tr><td style="font-weight:600;">构造后特征数</td><td>' + str(X_new.shape[1]) + '</td></tr>'
                '</tbody></table>'
            )
        else:
            return None, f"未知方法: {method}"

        img = fig_to_image(plot_feature_distribution_after_construction(
            X_new, original_n_features=orig_n, new_feature_names=new_names,
            title=f"特征构造后分布 ({method})"))
        return img, info
    except Exception as e:
        return None, f"错误: {e}"


def on_fe_discretize(n_bins, strategy):
    """模块5: 分箱离散化"""
    if _g["X_train"] is None:
        return None, "请先在「数据工作台」加载数据集"
    try:
        X_tr = _g["X_train"].copy()
        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]
        n_bins = int(n_bins)
        X_disc, enc = discretize_features(X_tr, n_bins=n_bins, strategy=strategy)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(X_tr[:, 0], bins=30, alpha=0.6, color='steelblue', edgecolor='k', label='原始分布')
        unique_bins = np.unique(X_disc[:, 0])
        colors = ['red','green','blue','orange','purple','cyan','pink','brown','gray','olive']
        for b in unique_bins:
            mask = X_disc[:, 0] == b
            if mask.any():
                lo, hi = X_tr[mask, 0].min(), X_tr[mask, 0].max()
                ax.axvspan(lo, hi, alpha=0.15, color=colors[int(b) % len(colors)])
        ax.set_title(f"分箱效果: {fnames[0]} ({strategy}, {n_bins}箱)")
        ax.set_xlabel(fnames[0])
        ax.set_ylabel("频数")
        ax.legend()
        ax.grid(True, alpha=0.3)
        img = fig_to_image(fig)

        bin_edges = enc.bin_edges_[0]
        edges_str = ", ".join([f"{e:.2f}" for e in bin_edges])
        info = (
            '<table class="info-table"><tbody>'
            '<tr><td style="font-weight:600;width:120px;">策略</td><td>' + strategy + '</td></tr>'
            '<tr><td style="font-weight:600;">分箱数</td><td>' + str(n_bins) + '</td></tr>'
            '<tr><td style="font-weight:600;">特征数</td><td>' + str(X_tr.shape[1]) + '</td></tr>'
            '<tr><td style="font-weight:600;">' + fnames[0] + ' 分箱边界</td><td>[' + edges_str + ']</td></tr>'
            '</tbody></table>'
        )
        return img, info
    except Exception as e:
        return None, f"错误: {e}"


def on_fe_correlation():
    """模块6: 相关性分析"""
    if _g["X_train"] is None:
        return None, "请先在「数据工作台」加载数据集"
    try:
        X_tr = _g["X_train"]
        fnames = _g["feature_names"] or [f"F{i}" for i in range(X_tr.shape[1])]
        corr = np.corrcoef(X_tr, rowvar=False)
        img = fig_to_image(plot_feature_correlation_heatmap(X_tr, feature_names=fnames))

        n = X_tr.shape[1]
        high_corr = []
        for i in range(n):
            for j in range(i + 1, n):
                r = abs(corr[i, j])
                if r > 0.7:
                    high_corr.append((fnames[i], fnames[j], corr[i, j]))
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

        html_parts = []
        html_parts.append(
            '<table class="info-table"><tbody>'
            '<tr><td style="font-weight:600;width:120px;">特征数</td><td>' + str(n) + '</td></tr>'
            '<tr><td style="font-weight:600;">高相关阈值</td><td>|r| &gt; 0.7</td></tr>'
            '</tbody></table>'
        )
        if high_corr:
            rows = ""
            for f1, f2, r in high_corr[:10]:
                rows += "<tr><td>" + f1 + "</td><td>" + f2 + "</td><td>" + f"{r:.4f}" + "</td></tr>\n"
            html_parts.append(
                '<div style="margin-top:8px;font-weight:600;">高相关性特征对 (|r| &gt; 0.7)</div>'
                '<table class="info-table"><thead><tr>'
                '<th>特征1</th><th>特征2</th><th>相关系数</th>'
                '</tr></thead><tbody>'
                + rows
                + '</tbody></table>'
            )
        else:
            html_parts.append(
                '<div style="margin-top:8px;color:#888;">无高相关性特征对 (所有特征对 |r| ≤ 0.7)</div>'
            )
        return img, "\n".join(html_parts)
    except Exception as e:
        return None, f"错误: {e}"


def on_load_template():

    """加载代码沙箱模板"""

    return SANDBOX_TEMPLATE


# ═══════════════════════════════════════════════════════════════

# 回调 — AI 助教

# ═══════════════════════════════════════════════════════════════


def on_chat(msg, history):

    if not msg.strip(): return "", history

    resp = ask(msg)

    history = history or []

    history += [{"role":"user","content":msg},{"role":"assistant","content":resp}]

    return "", history


def on_preset(q):

    if not q: return "", []

    return "", [{"role":"user","content":q},{"role":"assistant","content":ask(q)}]


# ═══════════════════════════════════════════════════════════════

# CSS

# ═══════════════════════════════════════════════════════════════


APP_CSS = """

footer { display: none !important; }

.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }

body { background: #f1f5f9 !important; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif !important; }

.top-nav { background: linear-gradient(135deg,#1e40af 0%,#2563eb 60%,#3b82f6 100%); box-shadow: 0 4px 16px rgba(30,64,175,0.25); }

.top-nav-inner { width:100%; margin:0 auto; display:flex; align-items:center; justify-content:space-between; padding:20px 28px; box-sizing:border-box; }

.brand { display:flex; align-items:center; gap:14px; }

.brand-icon img { width:100%; height:100%; object-fit:contain; }

.brand-icon { width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center; overflow:hidden; flex-shrink:0; }

.brand-title { color:#fff!important; font-size:17px; font-weight:700; }

.brand-sub { color:rgba(255,255,255,0.8)!important; font-size:11px; margin-top:2px; }

.nav-tags { display:flex; gap:5px; }

.nav-tag { font-size:11px; color:rgba(255,255,255,0.9)!important; font-weight:500; padding:5px 12px; border-radius:18px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.15); }

.sidebar-panel { background: linear-gradient(180deg,#1e3a8a 0%,#1e40af 40%,#2563eb 100%)!important; min-width:230px!important; max-width:230px!important; padding:0!important; box-shadow:4px 0 20px rgba(30,58,138,0.18)!important; border:none!important; }

.sidebar-header { padding:18px 14px 12px; border-bottom:1px solid rgba(255,255,255,0.12); margin:0 10px 10px; }

.sidebar-title { font-size:10px!important; font-weight:700!important; color:rgba(255,255,255,0.55)!important; letter-spacing:1.5px!important; text-transform:uppercase; }

.sidebar-nav .wrap { background:transparent!important; border:none!important; padding:0!important; gap:0!important; }

.sidebar-nav label { display:flex!important; align-items:center!important; width:calc(100% - 22px)!important; margin:3px 11px!important; padding:11px 14px!important; border-radius:10px!important; font-size:13px!important; font-weight:500!important; color:rgba(255,255,255,0.82)!important; background:rgba(255,255,255,0.07)!important; border:1px solid rgba(255,255,255,0.08)!important; cursor:pointer!important; transition:all 0.15s!important; }

.sidebar-nav label:hover { background:rgba(255,255,255,0.15)!important; color:#fff!important; }

.sidebar-nav input[type="radio"] { accent-color:#fff!important; }

.sidebar-nav label.selected { background:rgba(255,255,255,0.2)!important; border-color:rgba(255,255,255,0.3)!important; color:#fff!important; font-weight:600!important; box-shadow:0 2px 8px rgba(0,0,0,0.15)!important; }

.status-card { margin:0 11px; padding:10px 12px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.1); border-radius:9px; }

.status-label { font-size:9px; font-weight:600; color:rgba(255,255,255,0.5); letter-spacing:0.8px; text-transform:uppercase; }

.status-value { font-size:12px; color:#fff; font-weight:500; }

.page-panel { background:#f1f5f9!important; padding:20px 24px!important; min-height:calc(100vh - 62px); border-radius:0!important; }

.page-panel[data-init-hidden] { display:none!important; }

.content-card { background:#fff; border-radius:14px; box-shadow:0 2px 8px rgba(0,0,0,0.04); border:1px solid #e2e8f0; overflow:hidden; margin-bottom:14px; }

.card-header { background:linear-gradient(135deg,#1e40af 0%,#3b82f6 100%); padding:16px 20px; }

.card-header-inner { display:flex; align-items:center; gap:12px; }

.card-icon { width:42px; height:42px; background:rgba(255,255,255,0.18); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:22px; border:1px solid rgba(255,255,255,0.2); }

.card-title { color:#fff!important; font-size:16px; font-weight:700; margin:0; }

.card-desc { color:rgba(255,255,255,0.85)!important; font-size:12px; margin-top:3px; }

.card-body { padding:20px; }

.ethics-bar { background:#fffbeb; border-left:4px solid #f59e0b; border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:18px; font-size:12px; color:#92400e; line-height:1.7; }

.step-title { font-size:13px; font-weight:700; color:#1e40af; margin:16px 0 10px; padding-left:12px; border-left:3px solid #3b82f6; }

button.primary { background:linear-gradient(135deg,#2563eb,#1d4ed8)!important; border:none!important; border-radius:8px!important; font-weight:600!important; }

.footer-bar { text-align:center; padding:12px 20px; color:#94a3b8; font-size:10px; border-top:1px solid #e2e8f0; background:#fff; }

.gr-row { gap: 0!important; }

.code-editor { font-family: 'Consolas', 'Monaco', 'Courier New', monospace!important; font-size: 13px!important; line-height: 1.6!important; }

.code-editor textarea { min-height: 380px!important; }

.code-output { font-family: 'Consolas', 'Monaco', monospace!important; font-size: 12px!important; background: #1e293b!important; color: #e2e8f0!important; border-radius: 8px!important; padding: 16px!important; max-height: 400px!important; overflow-y: auto!important; }

.tab-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; margin-right:6px; }

.tab-badge-supervised { background:#dbeafe; color:#1d4ed8; }

.tab-badge-unsupervised { background:#dcfce7; color:#166534; }

.tab-badge-tool { background:#fef3c7; color:#92400e; }


#page-code.maximized { position: fixed!important; top: 0!important; left: 0!important; right: 0!important; bottom: 0!important; z-index: 99999!important; background: #f1f5f9!important; padding: 20px!important; overflow-y: auto!important; display: flex!important; flex-direction: column!important; }

.maximized .code-editor { min-height: 400px!important; flex: 1 1 auto!important; }

.maximized .code-output { min-height: 250px!important; flex: 0 0 auto!important; }

#page-code.maximized .code-editor textarea { min-height: calc(100vh - 260px)!important; height: calc(100vh - 260px)!important; font-size: 14px!important; line-height: 1.7!important; }

#page-code.maximized .code-output textarea { min-height: calc(100vh - 420px)!important; height: calc(100vh - 420px)!important; }

.maximize-btn { position: absolute; top: 8px; right: 8px; z-index: 10; background: #1e40af!important; color: #fff!important; border: none!important; border-radius: 6px!important; padding: 8px 16px!important; font-size: 13px!important; font-weight: 600!important; cursor: pointer!important; transition: background 0.2s!important; }

.maximize-btn:hover { background: #1d4ed8!important; }

.maximized .maximize-btn { position: fixed!important; top: 20px!important; right: 30px!important; z-index: 100001!important; }

.maximize-btn:hover { background: #1d4ed8!important; }

.snippet-btn { background: #fff!important; border: 1px solid #cbd5e1!important; border-radius: 6px!important; padding: 5px 12px!important; font-size: 11px!important; color: #334155!important; cursor: pointer!important; font-weight: 500!important; transition: all 0.15s ease!important; }

.snippet-btn:hover { background: #eff6ff!important; border-color: #3b82f6!important; color: #1e40af!important; }

.code-editor textarea { min-height: 340px!important; }

.code-output { min-height: 220px!important; }


/* 学习路径样式 */
.learning-path { position:relative; }
.lp-timeline { position:relative; padding-left:28px; }
.lp-timeline::before { content:''; position:absolute; left:11px; top:0; bottom:0; width:3px; background:linear-gradient(180deg,#3b82f6,#8b5cf6); border-radius:2px; }
.lp-step { position:relative; margin-bottom:20px; padding:16px 20px; background:#fff; border:1px solid #e2e8f0; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.04); transition:all 0.2s; cursor:pointer; }
.lp-step:hover { border-color:#93c5fd; box-shadow:0 2px 12px rgba(59,130,246,0.12); transform:translateY(-1px); }
.lp-step.lp-active { border-color:#3b82f6; box-shadow:0 2px 12px rgba(59,130,246,0.15); background:#eff6ff; }
.lp-step.lp-done { border-color:#86efac; background:#f0fdf4; }
.lp-step::before { content:''; position:absolute; left:-22px; top:20px; width:14px; height:14px; background:#fff; border:3px solid #3b82f6; border-radius:50%; z-index:1; }
.lp-step.lp-done::before { background:#22c55e; border-color:#22c55e; }
.lp-step.lp-active::before { background:#3b82f6; border-color:#3b82f6; box-shadow:0 0 0 4px rgba(59,130,246,0.2); }
.lp-step-num { position:absolute; left:-21px; top:19px; width:12px; height:12px; display:flex; align-items:center; justify-content:center; font-size:8px; font-weight:700; color:#fff; z-index:2; }
.lp-step-title { font-size:15px; font-weight:700; color:#1e293b; margin-bottom:4px; display:flex; align-items:center; gap:8px; }
.lp-step-badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:10px; font-weight:600; }
.lp-badge-data { background:#dbeafe; color:#1d4ed8; }
.lp-badge-sup { background:#dcfce7; color:#166534; }
.lp-badge-unsup { background:#fef3c7; color:#92400e; }
.lp-badge-tool { background:#f3e8ff; color:#7c3aed; }
.lp-badge-report { background:#fce7f3; color:#be185d; }
.lp-step-desc { font-size:12px; color:#64748b; line-height:1.6; margin-bottom:8px; }
.lp-step-tasks { font-size:11px; color:#94a3b8; }
.lp-step-tasks b { color:#64748b; }


/* 学习路径知识点 */
.lp-knowledge { background: #fefce8; border-left: 3px solid #eab308; padding: 10px 14px; margin: 8px 0; border-radius: 0 8px 8px 0; font-size: 12px; color: #713f12; line-height: 1.6; }

/* 学习路径完成标记 */
.lp-check { display: none; font-size: 16px; margin-left: 8px; }
.lp-step.completed .lp-check { display: inline; }
.lp-step.completed { opacity: 0.85; }
.lp-step.completed .lp-step-title { color: #16a34a; }

/* 报告信息表格 */
.report-info-table { width: 100%; border-collapse: collapse; margin: 12px auto; }
.report-info-table td { padding: 6px 12px; border: 1px solid #e2e8f0; font-size: 13px; }
.report-info-table td:first-child { background: #f1f5f9; font-weight: 600; width: 60px; text-align: center; }

/* 报告概览 */
.report-summary { background: #eff6ff; border-radius: 8px; padding: 12px 16px; margin: 12px 0; border-left: 3px solid #3b82f6; font-size: 13px; }

/* 报告结论区域 */
.report-conclusion { background: #fefce8; border-radius: 8px; padding: 16px; margin: 12px 0; border-left: 3px solid #eab308; }
.report-conclusion p { color: #713f12; font-style: italic; }

.report-config { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin-bottom:16px; }
.report-section-item { display:flex; align-items:center; gap:10px; padding:10px 14px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; margin-bottom:6px; font-size:13px; }
.report-section-item:hover { background:#f1f5f9; }
.report-section-icon { font-size:16px; }
.report-section-title { font-weight:600; color:#334155; flex:1; }
.report-preview { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; max-height:600px; overflow-y:auto; }
.report-preview h1 { font-size:20px; font-weight:700; color:#1e293b; text-align:center; margin-bottom:4px; }
.report-preview h2 { font-size:16px; font-weight:700; color:#1e40af; margin:18px 0 8px; padding-left:10px; border-left:3px solid #3b82f6; }
.report-preview h3 { font-size:14px; font-weight:600; color:#334155; margin:12px 0 6px; }
.report-preview p { font-size:13px; color:#475569; line-height:1.7; margin:6px 0; }
.report-preview table { width:100%; border-collapse:collapse; margin:10px 0; font-size:12px; }
.report-preview th { background:#eff6ff; color:#1e40af; padding:8px 12px; text-align:left; border:1px solid #dbeafe; font-weight:600; }
.report-preview td { padding:8px 12px; border:1px solid #e2e8f0; color:#334155; }
.report-preview .report-meta { text-align:center; color:#94a3b8; font-size:11px; margin-bottom:16px; }
.report-preview .img-placeholder { background:#f1f5f9; border:2px dashed #cbd5e1; border-radius:8px; padding:40px; text-align:center; color:#94a3b8; font-size:13px; margin:10px 0; }

"""


# ═══════════════════════════════════════════════════════════════

# HTML 片段

# ═══════════════════════════════════════════════════════════════


TOP_HTML = """<div class="top-nav"><div class="top-nav-inner">

  <div class="brand"><div class="brand-icon"><img src="data:image/png;base64,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" alt="logo"></div><div>

    <div class="brand-title">ML-Lab v3.4</div>

    <div class="brand-sub">机器学习可视化实验平台 · 监督学习 + 无监督学习 + 代码沙箱</div>

  </div></div>

  <div class="nav-tags">

    <span class="nav-tag" data-nav="学习路径">学习路径</span>
    <span class="nav-tag" data-nav="数据工作台">数据工作台</span>

    <span class="nav-tag" data-nav="特征工程">特征工程</span>

    <span class="nav-tag" data-nav="分类实验">分类实验</span><span class="nav-tag" data-nav="回归实验">回归实验</span>

    <span class="nav-tag" data-nav="聚类实验">聚类实验</span><span class="nav-tag" data-nav="代码沙箱">代码沙箱</span><span class="nav-tag" data-nav="AI助教">AI助学</span>

  </div>

</div></div>

<script>

document.addEventListener('DOMContentLoaded', function(){

  document.querySelectorAll('.nav-tag[data-nav]').forEach(function(tag){

    tag.style.cursor = 'pointer';

    tag.addEventListener('click', function(){

      var target = this.getAttribute('data-nav');

      // 通过data-nav找到对应的data-nav-id
      var stepEl = document.querySelector('.lp-step[data-nav="' + target + '"]');

      var targetId = stepEl ? stepEl.getAttribute('data-nav-id') : null;

      if (!targetId) return;

      var allPages = ["page-learning","page-data","page-fe","page-classify","page-regress","page-cluster","page-code","page-ai"];

      var css = "<style>" +

        allPages.filter(function(p){return p!==targetId;}).map(function(p){return "#"+p+"{display:none!important;}";}).join("") +

        "#"+targetId+"{display:flex!important;}" +

        "
        .info-table { width:100%; border-collapse:collapse; margin:6px 0; font-size:13px; }
        .info-table td { padding:7px 12px; border-bottom:1px solid #edf2f7; }
        .info-table .info-label { font-weight:500; color:#718096; width:35%; }
        .info-table tr:last-child td { border-bottom:none; }
        .info-table tr:hover td { background:#f7fafc; }
        
    </style>";

      var styleEl = document.getElementById('ml-lab-page-switch');
      if (!styleEl) { styleEl = document.createElement('style'); styleEl.id = 'ml-lab-page-switch'; document.head.appendChild(styleEl); }
      styleEl.textContent = css.slice(7, -8);

      window.scrollTo({top: 0, behavior: "smooth"});

    });

  });

});


</script>"""


LEARNING_PATH_HTML = """<div class="learning-path">
<div class="lp-timeline">
    <div class="lp-step" data-nav="📊 数据工作台" data-nav-id="page-data" data-step="data">
        <div class="lp-step-title">
            📊 数据探索与理解
            <span class="lp-step-badge lp-badge-data">数据准备</span>
            <span class="lp-check" id="lp-check-data">✅</span>
        </div>
        <div class="lp-step-desc">选择内置数据集（鸢尾花/波士顿房价/加州房价等），查看数据分布、统计信息，理解特征含义与目标变量。</div>
        <div class="lp-knowledge">
            <b>📚 核心知识点：</b><br>
            • 数据集划分：训练集/测试集，stratify 分层抽样<br>
            • 数据类型：数值型（连续/离散）、类别型<br>
            • 数据质量：缺失值、异常值、重复值检测<br>
            • 描述性统计：均值、中位数、标准差、四分位数
        </div>
        <div class="lp-step-tasks"><b>实验任务：</b>加载鸢尾花数据集 → 查看分布图 → 查看统计摘要 → 调整测试集比例观察变化</div>
        <div class="lp-step-tasks" style="margin-top:6px;"><b>思考问题：</b><br>• 数据集有多少条记录？多少个特征？<br>• 各特征的分布是否均衡？是否存在偏态？<br>• 特征之间是否存在相关性？<br>• 目标变量的类别分布如何？</div>
    </div>
    <div class="lp-step" data-nav="⚙️ 特征工程" data-nav-id="page-fe" data-step="fe">
        <div class="lp-step-title">
            ⚙️ 特征工程实践
            <span class="lp-step-badge lp-badge-tool">特征工程</span>
            <span class="lp-check" id="lp-check-fe">✅</span>
        </div>
        <div class="lp-step-desc">学习特征缩放、特征选择、PCA降维、特征构造等核心技术，理解其对模型性能的影响。</div>
        <div class="lp-knowledge">
            <b>📚 核心知识点：</b><br>
            • 标准化 (StandardScaler)：均值0，方差1，适合SVM/神经网络<br>
            • 归一化 (MinMaxScaler)：缩放到[0,1]，适合KNN/距离度量<br>
            • PCA：通过方差最大化投影降维，需注意解释率<br>
            • 特征选择：过滤法/包裹法/嵌入法
        </div>
        <div class="lp-step-tasks"><b>实验任务：</b>特征缩放对比 → 特征选择 → PCA降维 → 相关性分析</div>
        <div class="lp-step-tasks" style="margin-top:6px;"><b>思考问题：</b><br>• 标准化和归一化有何区别？何时使用哪种？<br>• PCA 降维后保留了多少方差解释率？<br>• 哪些特征对目标变量的贡献最大？<br>• 特征工程对模型精度提升有多大帮助？</div>
    </div>
    <div class="lp-step" data-nav="🏷️ 分类实验" data-nav-id="page-classify" data-step="cls">
        <div class="lp-step-title">
            🏷️ 分类算法实验
            <span class="lp-step-badge lp-badge-sup">监督学习</span>
            <span class="lp-check" id="lp-check-cls">✅</span>
        </div>
        <div class="lp-step-desc">使用逻辑回归、KNN、决策树、SVM、朴素贝叶斯、随机森林、GBDT、神经网络等算法进行分类训练。</div>
        <div class="lp-knowledge">
            <b>📚 核心知识点：</b><br>
            • 评估指标：准确率、精确率、召回率、F1分数<br>
            • 混淆矩阵：TP/FP/FN/TN，理解各类错误<br>
            • ROC曲线与AUC：阈值选择与模型比较<br>
            • 过拟合/欠拟合：学习曲线诊断
        </div>
        <div class="lp-step-tasks"><b>实验任务：</b>选择算法 → 训练模型 → 查看混淆矩阵/ROC/学习曲线 → 至少完成3种算法对比</div>
        <div class="lp-step-tasks" style="margin-top:6px;"><b>思考问题：</b><br>• 哪个算法在该数据集上表现最好？为什么？<br>• 模型是否存在过拟合或欠拟合？<br>• 不同算法的混淆矩阵差异说明了什么？</div>
    </div>
    <div class="lp-step" data-nav="📈 回归实验" data-nav-id="page-regress" data-step="reg">
        <div class="lp-step-title">
            📈 回归算法实验
            <span class="lp-step-badge lp-badge-sup">监督学习</span>
            <span class="lp-check" id="lp-check-reg">✅</span>
        </div>
        <div class="lp-step-desc">使用线性回归、Ridge、Lasso、多项式回归、SVR、随机森林回归等算法进行回归分析。</div>
        <div class="lp-knowledge">
            <b>📚 核心知识点：</b><br>
            • 评估指标：MSE、RMSE、MAE、R²<br>
            • 正则化：L1 (Lasso 特征选择)、L2 (Ridge 收缩)<br>
            • 偏置-方差权衡：模型复杂度与泛化能力<br>
            • 残差分析：检验回归假设
        </div>
        <div class="lp-step-tasks"><b>实验任务：</b>选择回归数据集 → 线性回归 → 对比Ridge/Lasso正则化效果 → 多项式回归对比</div>
        <div class="lp-step-tasks" style="margin-top:6px;"><b>思考问题：</b><br>• Lasso 回归如何实现特征选择？<br>• 正则化强度如何影响模型复杂度？<br>• 多项式阶数过高会导致什么问题？</div>
    </div>
    <div class="lp-step" data-nav="🔵 聚类实验" data-nav-id="page-cluster" data-step="clu">
        <div class="lp-step-title">
            🔵 聚类算法实验
            <span class="lp-step-badge lp-badge-unsup">无监督学习</span>
            <span class="lp-check" id="lp-check-clu">✅</span>
        </div>
        <div class="lp-step-desc">使用 K-Means、DBSCAN、层次聚类进行无监督学习，理解聚类评估指标。</div>
        <div class="lp-knowledge">
            <b>📚 核心知识点：</b><br>
            • K-Means：基于距离的划分，需预设K值<br>
            • DBSCAN：基于密度的聚类，可发现任意形状<br>
            • 层次聚类：树状图 (Dendrogram) 展示聚类层次<br>
            • 评估：轮廓系数、肘部法则
        </div>
        <div class="lp-step-tasks"><b>实验任务：</b>K-Means → 观察肘部图 → DBSCAN对比 → 层次聚类树状图</div>
        <div class="lp-step-tasks" style="margin-top:6px;"><b>思考问题：</b><br>• K-Means 的最优 K 值如何确定？<br>• DBSCAN 与 K-Means 的区别？各适合什么场景？<br>• 层次聚类的树状图如何帮助确定簇数？</div>
    </div>
    <div class="lp-step" data-nav="💻 代码沙箱" data-nav-id="page-code" data-step="code">
        <div class="lp-step-title">
            💻 代码实践与拓展
            <span class="lp-step-badge lp-badge-tool">工具</span>
            <span class="lp-check" id="lp-check-code">✅</span>
        </div>
        <div class="lp-step-desc">在代码沙箱中自由编写 Python 代码，调用 scikit-learn API 进行个性化实验。</div>
        <div class="lp-knowledge">
            <b>📚 核心知识点：</b><br>
            • Pipeline：构建完整的 ML 流水线<br>
            • GridSearchCV：超参数网格搜索<br>
            • cross_val_score：交叉验证<br>
            • 自定义评估指标和可视化
        </div>
        <div class="lp-step-tasks"><b>实验任务：</b>用 Pipeline 构建分类流水线 → 使用 GridSearchCV 调参 → 交叉验证评估</div>
        <div class="lp-step-tasks" style="margin-top:6px;"><b>思考问题：</b><br>• 能否手动实现一个简单的 KNN 分类器？<br>• 如何用交叉验证评估模型的稳定性？<br>• Pipeline 的好处是什么？</div>
    </div>
</div></div>"""
def _card(icon, title, desc, body=""):

    return (f'<div class="content-card"><div class="card-header"><div class="card-header-inner">'

            f'<div class="card-icon">{icon}</div>'

            f'<div><div class="card-title">{title}</div><div class="card-desc">{desc}</div></div>'

            f'</div></div><div class="card-body">{body}</div></div>')

ETHICS = ('<div class="ethics-bar"><strong>🔒 数据伦理提醒：</strong>'

    '实验数据均为公开学术数据集，已脱敏处理。在实际应用中，务必遵守'

    '<strong>数据隐私保护法规</strong>（如《个人信息保护法》），'

    '尊重数据主权，不滥用数据，维护信息安全。</div>')

# ═══════════════════════════════════════════════════════════════

# 参数面板构建辅助

# ═══════════════════════════════════════════════════════════════

def _make_supervised_param_components():

    """创建监督学习通用参数滑块（分类/回归共享）"""

    with gr.Column():

        lr_sl = gr.Slider(0.001, 0.5, 0.01, 0.001, label="learning_rate", visible=False)

        ni_sl = gr.Slider(10, 500, 100, 10, label="n_iterations", visible=False)

        C_sl  = gr.Slider(0.01, 100, 1.0, 0.1, label="C")

        md_sl = gr.Slider(1, 15, 3, 1, label="max_depth")

        mi_sl = gr.Slider(50, 500, 200, 50, label="max_iter")

        hid   = gr.Textbox("64,32", label="hidden_layer_sizes", visible=False)

        al_sl = gr.Slider(0.01, 100, 1.0, 0.1, label="alpha (正则化/惩罚系数)", visible=False)

        deg_sl = gr.Slider(2, 6, 2, 1, label="degree (多项式阶数)", visible=False)

        eps_sl = gr.Slider(0.01, 1.0, 0.1, 0.01, label="epsilon (SVR容差)", visible=False)

        kern_sl = gr.Dropdown(choices=["linear", "rbf", "poly"], value="rbf",
                              label="kernel", visible=False)

        crit_sl = gr.Dropdown(choices=["squared_error", "absolute_error"],
                              value="squared_error", label="criterion", visible=False)

    return lr_sl, ni_sl, C_sl, md_sl, mi_sl, hid, al_sl, deg_sl, eps_sl, kern_sl, crit_sl


def _bind_param_visibility(algo_dd, lr_sl, ni_sl, C_sl, md_sl, mi_sl, hid, al_sl,
                              deg_sl=None, eps_sl=None, kern_sl=None, crit_sl=None):

    """绑定算法选择→参数显隐"""

    def _show(a):

        p = get_algorithm_params(a)

        keys = ["lr","ni","C","md","mi","hid","al","deg","eps","kern","crit"]

        v = {k: False for k in keys}

        m = {"learning_rate":"lr","n_iterations":"ni","C":"C","max_depth":"md",

             "max_iter":"mi","hidden_layer_sizes":"hid","alpha":"al",

             "degree":"deg","epsilon":"eps","kernel":"kern","criterion":"crit"}

        if p:

            for x in p:

                if x in m: v[m[x]] = True

        return [gr.update(visible=v[k]) for k in keys]

    _outs = [lr_sl,ni_sl,C_sl,md_sl,mi_sl,hid,al_sl,deg_sl,eps_sl,kern_sl,crit_sl]

    _outs = [o for o in _outs if o is not None]

    _keys = ["lr","ni","C","md","mi","hid","al","deg","eps","kern","crit"][:len(_outs)]

    def _show_filtered(a):

        full = _show(a)

        return [full[["lr","ni","C","md","mi","hid","al","deg","eps","kern","crit"].index(k)] for k in _keys]

    algo_dd.change(fn=_show_filtered, inputs=[algo_dd], outputs=_outs)


# ═══════════════════════════════════════════════════════════════

# 界面构建

# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# 回调 — 学习路径导航
# ═══════════════════════════════════════════════════════════════

def on_learning_path_click(evt: gr.EventData):
    """学习路径步骤点击时跳转到对应页面"""
    return gr.skip()  # 由 JS 处理跳转

def create_app():

    with gr.Blocks(title="ML-Lab v3.4") as app:

        gr.HTML(TOP_HTML)


        with gr.Row(equal_height=False):


            # ═══ 左侧边栏 ═══

            with gr.Column(scale=0, min_width=230, elem_classes="sidebar-panel"):

                gr.HTML('<div class="sidebar-header"><div class="sidebar-title">功能导航</div></div>')

                nav = gr.Radio(

                    choices=[

                        "📖 学习路径",

                        "📊 数据工作台",

                        "⚙️ 特征工程",

                        "🏷️ 分类实验",

                        "📈 回归实验",

                        "🔵 聚类实验",

                        "💻 代码沙箱",

                        "🤖 AI助教",


                    ],

                    value="📖 学习路径", label="", container=False,

                    elem_classes="sidebar-nav",

                )

                gr.HTML('<div style="padding:14px 0 4px;"><div class="sidebar-title" style="margin:0 11px 8px;">当前状态</div></div>')

                status_ds = gr.HTML('<div class="status-card"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">数据集</span><span class="status-value" id="sv-ds">未加载</span></div></div>')

                status_md = gr.HTML('<div class="status-card" style="margin-top:6px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">模型</span><span class="status-value" id="sv-md">未训练</span></div></div>')


            # ═══ 页面0: 学习路径 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-learning", visible=True) as page_learning:

                gr.HTML(_card("📖","学习路径","引导式机器学习实验流程，从数据探索到报告撰写的完整学习路线",

                    '<span class="tab-badge tab-badge-tool">教学引导</span>'

                    '<span style="color:#64748b;font-size:12px;">7个实验阶段 · 含思考题与操作指引</span>' ))

                gr.HTML(LEARNING_PATH_HTML)


            # ═══ 页面1: 数据工作台 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-data", visible=True) as page_data:

                gr.HTML(_card("📊","数据工作台","选择数据集、探索数据分布、执行预处理",

                    ETHICS+'<div class="step-title">步骤 1 · 加载数据集</div>'))

                with gr.Row():

                    all_ds = list(get_builtin_datasets().keys())

                    ds_dd = gr.Dropdown(choices=all_ds, value="鸢尾花 (Iris)", label="数据集")

                    ts_sl = gr.Slider(0.1, 0.5, 0.3, 0.05, label="测试集比例")

                    load_btn = gr.Button("加载数据集", variant="primary")

                data_img = gr.Image(label="数据分布（PCA 降维可视化）", height=360)

                data_info = gr.HTML(label="数据信息")

                data_summary = gr.Dataframe(label="数据摘要")

                gr.HTML('<div class="step-title">步骤 2 · 数据预处理</div>')

                with gr.Row():

                    method_dd = gr.Dropdown(choices=["mean","median","most_frequent","constant"],

                        value="mean", label="缺失值填充策略")

                    feat_sl = gr.Slider(0, 50, 0, 1, label="展示特征索引")

                    pp_btn = gr.Button("执行预处理并对比", variant="secondary")

                pp_img = gr.Image(label="预处理前后对比", height=320)

                pp_info = gr.HTML(label="预处理结果")


            # ═══ 页面2: 分类实验 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-classify", visible=True) as page_classify:

                gr.HTML(_card("🏷️","分类实验","监督学习 · 分类算法训练与评估",

                    '<span class="tab-badge tab-badge-supervised">监督学习</span>'

                    '<span style="color:#64748b;font-size:12px;">逻辑回归 · KNN · 决策树 · SVM · 朴素贝叶斯 · 随机森林 · GBDT · 神经网络</span>'

                    +'<div class="step-title">选择算法与参数</div>'))

                with gr.Row():

                    cls_algos = get_algorithm_list("classification")

                    algo_dd_cls = gr.Dropdown(choices=cls_algos, value="决策树", label="算法")

                    lr_c, ni_c, C_c, md_c, mi_c, hid_c, al_c, *_ = _make_supervised_param_components()

                _bind_param_visibility(algo_dd_cls, lr_c, ni_c, C_c, md_c, mi_c, hid_c, al_c)

                train_btn_cls = gr.Button("开始训练", variant="primary", size="lg")

                gr.HTML('<div class="step-title">训练结果</div>')

                with gr.Row():

                    train_img_cls = gr.Image(label="训练过程", height=340)

                    db_img_cls = gr.Image(label="决策边界", height=340)

                result_txt_cls = gr.HTML(label="评估结果", value='<div style="color:#a0aec0;padding:20px;text-align:center;">训练模型后显示评估结果表格</div>')

                gr.HTML('<div class="step-title">代码复现</div>')

                with gr.Row():
                    copy_code_btn_cls = gr.Button("📋 生成并复制代码", variant="secondary")
                code_display_cls = gr.Code(label="Python 代码 (可直接复制运行)", language="python", lines=28, interactive=True)

                with gr.Row():

                    cm_img_cls = gr.Image(label="混淆矩阵", height=340)

                    roc_img_cls = gr.Image(label="ROC曲线", height=340)

                # 新增: 学习曲线和交叉验证
                gr.HTML('<div class="step-title">模型诊断</div>')

                with gr.Row():

                    lc_img_cls = gr.Image(label="学习曲线 (过拟合/欠拟合诊断)", height=340)

                    cv_img_cls = gr.Image(label="交叉验证得分", height=340)

                # 新增: 多模型对比
                gr.HTML('<div class="step-title">模型对比</div>')

                with gr.Row():
                    compare_algos = gr.CheckboxGroup(
                        choices=cls_algos,
                        value=["决策树", "随机森林"],
                        label="选择要对比的算法 (可多选)"
                    )
                    compare_btn = gr.Button("开始对比", variant="secondary")

                compare_img = gr.Image(label="模型性能对比", height=400)
                compare_result = gr.Textbox(label="对比结果", lines=8)


            # ═══ 页面3: 回归实验 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-regress", visible=True) as page_regress:

                gr.HTML(_card("📈","回归实验","监督学习 · 回归算法训练与评估",

                    '<span class="tab-badge tab-badge-supervised">监督学习</span>'

                    '<span style="color:#64748b;font-size:12px;">线性回归 · Ridge · Lasso · 多项式回归 · SVR · 决策树回归</span>'

                    +'<div class="step-title">选择算法与参数</div>'))

                with gr.Row():

                    reg_algos = get_algorithm_list("regression")

                    algo_dd_reg = gr.Dropdown(choices=reg_algos, value="线性回归", label="算法")

                    lr_r, ni_r, C_r, md_r, mi_r, hid_r, al_r, deg_r, eps_r, kern_r, crit_r = _make_supervised_param_components()

                _bind_param_visibility(algo_dd_reg, lr_r, ni_r, C_r, md_r, mi_r, hid_r, al_r,
                                       deg_sl=deg_r, eps_sl=eps_r, kern_sl=kern_r, crit_sl=crit_r)

                train_btn_reg = gr.Button("开始训练", variant="primary", size="lg")

                gr.HTML('<div class="step-title">训练结果</div>')

                with gr.Row():

                    gd_img = gr.Image(label="梯度下降可视化", height=340)

                    rg_img = gr.Image(label="回归拟合结果", height=340)

                with gr.Row():

                    reg_cmp_img = gr.Image(label="正则化强度对比", height=340, visible=False)

                    poly_cmp_img = gr.Image(label="多项式阶数对比", height=340, visible=False)

                with gr.Row():

                    model_cmp_img = gr.Image(label="回归模型对比", height=340, visible=False)

                result_txt_reg = gr.HTML(label="评估结果", value='<div style="color:#a0aec0;padding:20px;text-align:center;">训练模型后显示评估结果表格</div>')

                gr.HTML('<div class="step-title">代码复现</div>')

                with gr.Row():
                    copy_code_btn_reg = gr.Button("📋 生成并复制代码", variant="secondary")
                code_display_reg = gr.Code(label="Python 代码 (可直接复制运行)", language="python", lines=28, interactive=True)


            # ═══ 页面3.5: 特征工程工作台 ═══
            with gr.Column(elem_classes="page-panel", elem_id="page-fe", visible=True) as page_fe:
                gr.HTML(_card("⚙️","特征工程工作台","系统化特征处理与变换流水线",
                    '<span class="tab-badge" style="background:rgba(139,92,246,0.9);">数据处理</span>'
                    '<span style="color:#64748b;font-size:12px;">特征缩放 · 特征选择 · PCA降维 · 特征构造 · 分箱离散化 · 相关性分析</span>'))

                gr.HTML('<div class="step-title">模块 1 · 特征缩放</div>')
                with gr.Row():
                    fe_scale_method = gr.Dropdown(
                        choices=["standard", "minmax", "robust", "normalizer"],
                        value="standard", label="缩放方法",
                        info="StandardScaler / MinMaxScaler / RobustScaler / Normalizer")
                    fe_scale_btn = gr.Button("执行缩放", variant="secondary")
                with gr.Row():
                    fe_scale_img1 = gr.Image(label="箱线图对比 (缩放前后)", height=320)
                    fe_scale_img2 = gr.Image(label="单特征分布对比", height=320)
                fe_scale_info = gr.HTML(label="缩放统计信息", value='<div style="color:#a0aec0;padding:20px;text-align:center;">执行缩放后显示统计信息</div>')

                gr.HTML('<div class="step-title">模块 2 · 特征选择</div>')
                with gr.Row():
                    fe_sel_method = gr.Dropdown(
                        choices=["f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression", "rfe", "model_based"],
                        value="f_classif", label="选择方法",
                        info="F值 / 互信息 / RFE / RandomForest重要性")
                    fe_sel_k = gr.Slider(1, 20, 5, 1, label="选择特征数 (k)")
                    fe_sel_btn = gr.Button("执行特征选择", variant="secondary")
                with gr.Row():
                    fe_sel_img = gr.Image(label="特征得分/重要性排名", height=360)
                    fe_sel_info = gr.HTML(label="特征选择结果")

                gr.HTML('<div class="step-title">模块 3 · PCA 降维</div>')
                with gr.Row():
                    fe_pca_n = gr.Slider(1, 10, 2, 1, label="目标维度 (n_components)")
                    fe_pca_btn = gr.Button("执行 PCA", variant="secondary")
                with gr.Row():
                    fe_pca_img1 = gr.Image(label="方差解释率", height=320)
                    fe_pca_img2 = gr.Image(label="PCA 2D 投影", height=320)
                fe_pca_info = gr.HTML(label="PCA 分析结果")

                gr.HTML('<div class="step-title">模块 4 · 特征构造</div>')
                with gr.Row():
                    fe_const_method = gr.Dropdown(
                        choices=["polynomial", "interaction", "statistical"],
                        value="polynomial", label="构造方法",
                        info="多项式特征 / 交互特征 / 统计特征")
                    fe_const_degree = gr.Slider(2, 4, 2, 1, label="多项式阶数")
                    fe_const_btn = gr.Button("执行特征构造", variant="secondary")
                with gr.Row():
                    fe_const_img = gr.Image(label="构造后特征分布", height=320)
                    fe_const_info = gr.HTML(label="特征构造结果")

                gr.HTML('<div class="step-title">模块 5 · 分箱离散化</div>')
                with gr.Row():
                    fe_disc_bins = gr.Slider(3, 10, 5, 1, label="分箱数")
                    fe_disc_strategy = gr.Dropdown(
                        choices=["uniform", "quantile", "kmeans"],
                        value="quantile", label="分箱策略")
                    fe_disc_btn = gr.Button("执行分箱", variant="secondary")
                with gr.Row():
                    fe_disc_img = gr.Image(label="分箱效果", height=320)
                    fe_disc_info = gr.HTML(label="分箱结果")

                gr.HTML('<div class="step-title">模块 6 · 相关性分析</div>')
                with gr.Row():
                    fe_corr_btn = gr.Button("生成相关性热力图", variant="secondary")
                with gr.Row():
                    fe_corr_img = gr.Image(label="特征相关性热力图", height=360)
                    fe_corr_info = gr.HTML(label="相关性分析结果")



            # ═══ 页面4: 聚类实验 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-cluster", visible=True) as page_cluster:

                gr.HTML(_card("🔵","聚类实验","无监督学习 · 聚类与降维算法",

                    '<span class="tab-badge tab-badge-unsupervised">无监督学习</span>'

                    '<span style="color:#64748b;font-size:12px;">K-Means · DBSCAN · 层次聚类 · PCA 降维</span>'

                    +'<div class="step-title">选择算法与参数</div>'))

                unsup_algos = get_algorithm_list("unsupervised")

                with gr.Row():
                    with gr.Column(scale=1):
                        algo_dd_uns = gr.Dropdown(choices=unsup_algos, value="K-Means", label="算法")
                    with gr.Column(scale=1):
                        n_cl = gr.Slider(2, 10, 3, 1, label="n_clusters / n_components", visible=True)
                        eps_sl = gr.Slider(0.1, 5.0, 0.5, 0.1, label="eps (DBSCAN)", visible=False)
                        ms_sl = gr.Slider(2, 20, 5, 1, label="min_samples (DBSCAN)", visible=False)
                        max_iter_cl = gr.Slider(50, 500, 300, 50, label="max_iter (K-Means)", visible=True)
                        init_dd = gr.Dropdown(choices=["k-means++","random"], value="k-means++", label="init (K-Means)", visible=True)
                        linkage_dd = gr.Dropdown(choices=["ward","complete","average"], value="ward", label="linkage (层次聚类)", visible=False)
                        affinity_dd = gr.Dropdown(choices=["euclidean","manhattan"], value="euclidean", label="affinity (层次聚类)", visible=False)

                train_btn_uns = gr.Button("开始聚类", variant="primary", size="lg")

                gr.HTML('<div class="step-title">聚类结果</div>')

                with gr.Row():

                    cluster_img1 = gr.Image(label="聚类散点图 / PCA 投影", height=340)

                    cluster_img2 = gr.Image(label="肘部法则 / 树状图 / 方差解释", height=340)


                cluster_img3 = gr.Image(label="补充可视化", height=340, visible=False)

                result_txt_uns = gr.HTML(label="评估结果", value='<div style="color:#a0aec0;padding:20px;text-align:center;">训练模型后显示评估结果表格</div>')

                gr.HTML('<div class="step-title">代码复现</div>')

                with gr.Row():
                    copy_code_btn_uns = gr.Button("📋 生成并复制代码", variant="secondary")
                code_display_uns = gr.Code(label="Python 代码 (可直接复制运行)", language="python", lines=28, interactive=True)


            # ═══ 页面5: 代码沙箱 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-code", visible=True) as page_code:

                gr.HTML(_card("💻","代码沙箱","在线编写和运行 Python 代码",

                    '<span class="tab-badge tab-badge-tool">开发工具</span>'

                    '<span style="color:#64748b;font-size:12px;">已预导入 numpy, pandas, sklearn, matplotlib 和当前数据集</span>'))

                with gr.Row():

                    template_btn = gr.Button("加载模板", size="sm")

                    clear_code_btn = gr.Button("清空代码", size="sm")

                gr.HTML('''<div id="code-hint-bar" style="display:flex;flex-wrap:wrap;gap:5px;margin:6px 0 10px;align-items:center;">

  <span style="font-size:11px;color:#64748b;font-weight:600;margin-right:2px;">常用代码片段：</span>

</div>''')

                code_editor = gr.Code(label="Python 代码", language="python", lines=30,

                                       value=SANDBOX_TEMPLATE, elem_classes="code-editor",

                                       elem_id="sandbox-code-editor")

                with gr.Row():

                    run_btn = gr.Button("▶ 运行代码", variant="primary", size="lg")

                    max_btn = gr.HTML('''<div id="sandbox-maximize-btn" class="maximize-btn" title="最大化/还原编辑器">⛶ 全屏编辑</div>''')

                code_output = gr.Textbox(label="运行输出", lines=20, elem_classes="code-output",

                                          interactive=False)

                
                gr.HTML('<div class="ethics-bar"><strong>💡 使用提示：</strong>'

                    '变量 <code>X</code>, <code>y</code>, <code>X_train</code>, <code>y_train</code>, '

                    '<code>X_test</code>, <code>y_test</code>, <code>feature_names</code>, '

                    '<code>target_names</code> 已自动注入。'

                    '支持 matplotlib 绑图（需在代码末尾加 <code>plt.show()</code>）。'

                    '按 <code>Tab</code> 键可触发代码补全提示。</div>')


            # ═══ 页面6: AI助教 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-ai", visible=True) as page_ai:

                gr.HTML(_card("🤖","AI助教","基于通义千问大模型，随时解答机器学习疑问",

                    '<span class="tab-badge tab-badge-tool">AI 辅助</span>'

                    '<p style="color:#64748b;font-size:13px;">支持算法原理讲解、代码解释、错误排查等</p>'))

                chatbot = gr.Chatbot(label="对话区域", height=380)

                with gr.Row():

                    msg_in  = gr.Textbox(label="输入问题", placeholder="例如：K-Means 的原理是什么？", scale=5)

                    send_b  = gr.Button("发送", variant="primary", scale=1)

                    clear_b = gr.Button("清空", scale=0)

                gr.HTML('<div class="step-title">常见问题</div>')

                with gr.Row():

                    preset_btns = [gr.Button(q, size="sm") for q in PRESET_QUESTIONS[:6]]

        PAGE_MAP = {
            "📖 学习路径":  ("page-learning", 0),
            "📊 数据工作台": ("page-data",     1),
            "⚙️ 特征工程":  ("page-fe",       2),
            "🏷️ 分类实验":  ("page-classify", 3),
            "📈 回归实验":  ("page-regress", 4),
            "🔵 聚类实验":  ("page-cluster", 5),
            "💻 代码沙箱":  ("page-code",    6),
            "🤖 AI助教":    ("page-ai",      7),
        }

        PAGE_ORDER = ["page-learning", "page-data", "page-fe", "page-classify", "page-regress", "page-cluster", "page-code", "page-ai"]


        def on_nav(page_name):

            target_id, _ = PAGE_MAP.get(page_name, ("page-data", 0))

            hidden = [p for p in PAGE_ORDER if p != target_id]

            shown = [target_id]

            css = ("<style>" +

                   ",".join(['#'+p for p in hidden]) + "{display:none!important;}" +

                   ",".join(['#'+p for p in shown]) + "{display:flex!important;}" +

                   "</style>")

            return css


        dynamic_css = gr.HTML(value="<style>#page-data,#page-fe,#page-classify,#page-regress,#page-cluster,#page-code,#page-ai{display:none!important;}#page-learning{display:flex!important;}</style>", elem_id="dynamic-nav-css")

        nav.change(fn=on_nav, inputs=[nav], outputs=[dynamic_css])

        gr.HTML('<div class="footer-bar">ML-Lab v3.4 · Python · Scikit-learn · Gradio · 通义千问 · 江苏工程职业技术学院 · MIT License</div>')

        # 学习路径步骤点击跳转
        

        # ── 事件绑定 ──
        # 数据工作台
        load_btn.click(fn=on_load_data, inputs=[ds_dd, ts_sl], outputs=[data_img, data_info, data_summary, status_ds])
        pp_btn.click(fn=on_preprocess, inputs=[method_dd, feat_sl], outputs=[pp_img, pp_info])

        # 特征工程工作台
        fe_scale_btn.click(fn=on_fe_scaling, inputs=[fe_scale_method],
            outputs=[fe_scale_img1, fe_scale_img2, fe_scale_info])
        fe_sel_btn.click(fn=on_fe_feature_selection, inputs=[fe_sel_method, fe_sel_k],
            outputs=[fe_sel_img, fe_sel_info])
        fe_pca_btn.click(fn=on_fe_pca, inputs=[fe_pca_n],
            outputs=[fe_pca_img1, fe_pca_img2, fe_pca_info])
        fe_const_btn.click(fn=on_fe_construct, inputs=[fe_const_method, fe_const_degree],
            outputs=[fe_const_img, fe_const_info])
        fe_disc_btn.click(fn=on_fe_discretize, inputs=[fe_disc_bins, fe_disc_strategy],
            outputs=[fe_disc_img, fe_disc_info])
        fe_corr_btn.click(fn=on_fe_correlation, inputs=[],
            outputs=[fe_corr_img, fe_corr_info])

        # 分类实验
        train_btn_cls.click(fn=on_train_classification,
            inputs=[algo_dd_cls, lr_c, ni_c, C_c, md_c, mi_c, hid_c, al_c],
            outputs=[train_img_cls, db_img_cls, result_txt_cls, cm_img_cls, roc_img_cls, lc_img_cls, cv_img_cls, status_md])

        # 模型对比按钮
        compare_btn.click(fn=on_compare_models,
            inputs=[compare_algos],
            outputs=[compare_img, compare_result])

        # 回归实验
        train_btn_reg.click(fn=on_train_regression,
            inputs=[algo_dd_reg, lr_r, ni_r, C_r, md_r, mi_r, hid_r, al_r, deg_r, eps_r, kern_r, crit_r],
            outputs=[gd_img, rg_img, reg_cmp_img, poly_cmp_img, model_cmp_img, result_txt_reg, status_md])

        # 聚类实验 - 算法切换时动态显示参数
        def on_cluster_algo_change(algo_name):
            """根据聚类算法显示/隐藏对应参数"""
            algo = algo_name if algo_name else "K-Means"
            if algo == "K-Means":
                return [
                    gr.update(visible=True),   # n_cl
                    gr.update(visible=False),  # eps
                    gr.update(visible=False),  # ms
                    gr.update(visible=True),   # max_iter
                    gr.update(visible=True),   # init
                    gr.update(visible=False),  # linkage
                    gr.update(visible=False),  # affinity
                ]
            elif algo == "DBSCAN":
                return [
                    gr.update(visible=False),  # n_cl
                    gr.update(visible=True),   # eps
                    gr.update(visible=True),   # ms
                    gr.update(visible=False),  # max_iter
                    gr.update(visible=False),  # init
                    gr.update(visible=False),  # linkage
                    gr.update(visible=False),  # affinity
                ]
            elif algo == "层次聚类":
                return [
                    gr.update(visible=True),   # n_cl
                    gr.update(visible=False),  # eps
                    gr.update(visible=False),  # ms
                    gr.update(visible=False),  # max_iter
                    gr.update(visible=False),  # init
                    gr.update(visible=True),   # linkage
                    gr.update(visible=True),   # affinity
                ]
            elif algo == "PCA":
                return [
                    gr.update(visible=True),   # n_cl (n_components)
                    gr.update(visible=False),  # eps
                    gr.update(visible=False),  # ms
                    gr.update(visible=False),  # max_iter
                    gr.update(visible=False),  # init
                    gr.update(visible=False),  # linkage
                    gr.update(visible=False),  # affinity
                ]
            else:
                return [
                    gr.update(visible=True),   # n_cl
                    gr.update(visible=False),  # eps
                    gr.update(visible=False),  # ms
                    gr.update(visible=False),  # max_iter
                    gr.update(visible=False),  # init
                    gr.update(visible=False),  # linkage
                    gr.update(visible=False),  # affinity
                ]

        algo_dd_uns.change(
            fn=on_cluster_algo_change,
            inputs=[algo_dd_uns],
            outputs=[n_cl, eps_sl, ms_sl, max_iter_cl, init_dd, linkage_dd, affinity_dd]
        )

        # 聚类实验
        train_btn_uns.click(fn=on_train_clustering,
            inputs=[algo_dd_uns, n_cl, max_iter_cl, eps_sl, ms_sl, linkage_dd, affinity_dd, init_dd],
            outputs=[cluster_img1, cluster_img2, cluster_img3, result_txt_uns, status_md])

        # 代码沙箱
        run_btn.click(fn=on_run_code, inputs=[code_editor], outputs=[code_output])
        template_btn.click(fn=on_load_template, outputs=[code_editor])
        clear_code_btn.click(fn=lambda: ("",), outputs=[code_editor])

        # AI助教
        msg_in.submit(fn=on_chat, inputs=[msg_in, chatbot], outputs=[msg_in, chatbot])
        send_b.click(fn=on_chat, inputs=[msg_in, chatbot], outputs=[msg_in, chatbot])
        clear_b.click(fn=lambda: ("", []), outputs=[msg_in, chatbot])
        for pb in preset_btns:
            pb.click(fn=on_preset, inputs=[pb], outputs=[msg_in, chatbot])

        # 一键复制代码 - 分类实验
        copy_code_btn_cls.click(fn=on_copy_classification_code,
            inputs=[algo_dd_cls, lr_c, ni_c, C_c, md_c, mi_c, hid_c, al_c],
            outputs=[code_display_cls])

        # 一键复制代码 - 回归实验
        copy_code_btn_reg.click(fn=on_copy_regression_code,
            inputs=[algo_dd_reg, lr_r, ni_r, C_r, md_r, mi_r, hid_r, al_r, deg_r, eps_r, kern_r, crit_r],
            outputs=[code_display_reg])

        # 一键复制代码 - 聚类实验
        copy_code_btn_uns.click(fn=on_copy_clustering_code,
            inputs=[algo_dd_uns, n_cl, max_iter_cl, eps_sl, ms_sl, linkage_dd, affinity_dd, init_dd],
            outputs=[code_display_uns])

    return app


if __name__ == "__main__":
    print("=" * 55)
    print("  ML-Lab v3.4: 机器学习可视化实验平台")
    print("  监督学习 + 无监督学习 + 代码沙箱")
    print("  地址: http://127.0.0.1:7860")
    print("=" * 55)
    app = create_app()
    app.queue(max_size=20).launch(css=APP_CSS, server_name="0.0.0.0", server_port=7860, js="(function(){\n  var attempts = 0;\n  var maxAttempts = 40;\n  var timer = setInterval(function(){\n    attempts++;\n    var nav = document.querySelector('.top-nav');\n    if (nav && nav.parentElement && nav.parentElement.tagName !== 'BODY') {\n      document.body.prepend(nav);\n    }\n    if (attempts >= maxAttempts) clearInterval(timer);\n  }, 500);\n})();")
