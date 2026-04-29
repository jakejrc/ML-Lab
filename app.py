# -*- coding: utf-8 -*-

"""

ML-Lab v3.2 — 机器学习可视化实验平台

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

    format_evaluation_report

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

            info = (f"数据集: {dataset_name}\n任务类型: {'分类' if dtype=='classification' else '回归'}\n"

                    f"样本: {X.shape[0]}, 特征: {X.shape[1]}, 类别: {len(np.unique(y))}\n"

                    f"训练集: {Xtr.shape[0]}, 测试集: {Xte.shape[0]}")

        else:

            _sync(X, None, y, None, fn, tn, X, y)

            info = (f"数据集: {dataset_name}\n任务类型: 聚类 (无监督)\n"

                    f"样本: {X.shape[0]}, 特征: {X.shape[1]}, 簇数: {len(np.unique(y))}")

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

    return fig_to_image(fig), (f"原始缺失: {int(np.isnan(Xm).sum())}\n处理后缺失: {int(np.isnan(Xs).sum())}\n"

                                f"特征{idx} 均值: {np.nanmean(Xs[:,idx]):.3f}\n标准差: {np.nanstd(Xs[:,idx]):.3f}")


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

        rpt = format_evaluation_report(res, "classification")

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

        rpt = format_evaluation_report(res,"regression")

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

        rpt = format_evaluation_report(eval_res, "unsupervised")


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
        lines = [f"缩放方法: {method}", f"特征数: {X_tr.shape[1]}", f"样本数: {X_tr.shape[0]}", ""]
        lines.append(f"{'特征':<12} {'缩放前均值':>10} {'缩放后均值':>10} {'缩放前标准差':>12} {'缩放后标准差':>12}")
        lines.append("-" * 62)
        for s in stats:
            lines.append(f"{s['特征']:<12} {s['缩放前_均值']:>10.4f} {s['缩放后_均值']:>10.4f} "
                         f"{s['缩放前_标准差']:>12.4f} {s['缩放后_标准差']:>12.4f}")
        return img1, img2, "\n".join(lines)
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
        info_lines = [f"方法: {method}", f"原始特征数: {X_tr.shape[1]}", f"选择特征数: {k}", ""]

        if method in ("f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression"):
            X_sel, indices, scores, pvals = select_features_univariate(
                X_tr, y_tr, method=method, k=k, task_type=task)
            score_name = {"f_classif": "F值", "f_regression": "F值",
                          "mutual_info_classif": "互信息", "mutual_info_regression": "互信息"}.get(method, "得分")
            img = fig_to_image(plot_feature_selection_scores(
                scores, feature_names=fnames, top_n=None,
                title=f"特征选择得分 ({method})", score_name=score_name))
            info_lines.append(f"{'排名':<4} {'特征':<12} {score_name:<12} {'选中'}")
            info_lines.append("-" * 36)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(ranked[:k], 1):
                p_str = f"{pvals[i]:.4f}" if pvals is not None else ""
                info_lines.append(f"{rank:<4} {fnames[i]:<12} {scores[i]:>10.4f}   {'✓':>3}  {p_str}")
        elif method == "rfe":
            X_sel, indices, ranking = select_features_rfe(X_tr, y_tr, n_features=k, task_type=task)
            img = fig_to_image(plot_feature_importance_ranking(
                1.0 / np.array(ranking, dtype=float), feature_names=fnames,
                title="RFE 特征排名 (1/排名)"))
            info_lines.append(f"{'特征':<12} {'排名':<8} {'选中'}")
            info_lines.append("-" * 28)
            for i, r in enumerate(ranking):
                info_lines.append(f"{fnames[i]:<12} {r:<8} {'✓' if r == 1 else ''}")
        elif method == "model_based":
            X_sel, indices, importances = select_features_model_based(X_tr, y_tr, task_type=task)
            img = fig_to_image(plot_feature_importance_ranking(
                importances, feature_names=fnames,
                title="RandomForest 特征重要性"))
            info_lines.append(f"{'排名':<4} {'特征':<12} {'重要性':<10} {'选中'}")
            info_lines.append("-" * 32)
            ranked = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
            for rank, i in enumerate(ranked, 1):
                info_lines.append(f"{rank:<4} {fnames[i]:<12} {importances[i]:>10.4f}   {'✓' if i in indices else ''}")

        info_lines.append(f"\n选中特征: {[fnames[i] for i in indices]}")
        return img, "\n".join(info_lines)
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
        lines = [f"原始维度: {X_tr.shape[1]}", f"降维后: {n}", ""]
        lines.append(f"{'主成分':<8} {'方差解释率':>12} {'累计解释率':>12}")
        lines.append("-" * 36)
        cumsum = np.cumsum(evr)
        for i in range(len(evr)):
            lines.append(f"PC{i+1:<7} {evr[i]:>12.4f} {cumsum[i]:>12.4f}")
        lines.append(f"\n前{n}个主成分累计解释方差: {cumsum[-1]:.4f} ({cumsum[-1]*100:.1f}%)")
        return img1, img2, "\n".join(lines)
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
            info = f"方法: 多项式特征 (degree={degree})\n原始特征数: {orig_n}\n构造后: {X_new.shape[1]}"
        elif method == "interaction":
            X_new, new_names = construct_interaction_features(X_tr)
            info = f"方法: 交互特征 (两两相乘)\n原始特征数: {orig_n}\n新增交互特征: {len(new_names)}\n构造后: {X_new.shape[1]}"
        elif method == "statistical":
            X_new, new_names = construct_statistical_features(X_tr)
            info = f"方法: 统计特征 (行统计量)\n新增特征: {', '.join(new_names)}\n原始特征数: {orig_n}\n构造后: {X_new.shape[1]}"
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
        info = f"策略: {strategy}\n分箱数: {n_bins}\n特征数: {X_tr.shape[1]}\n\n{fnames[0]} 分箱边界: [{edges_str}]"
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

        lines = [f"特征数: {n}", f"\n高相关性特征对 (|r| > 0.7):"]
        if high_corr:
            lines.append(f"{'特征1':<12} {'特征2':<12} {'相关系数':>10}")
            lines.append("-" * 36)
            for f1, f2, r in high_corr[:10]:
                lines.append(f"{f1:<12} {f2:<12} {r:>10.4f}")
        else:
            lines.append("无 (所有特征对 |r| ≤ 0.7)")
        return img, "\n".join(lines)
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

.top-nav-inner { max-width:1600px; margin:0 auto; display:flex; align-items:center; justify-content:space-between; padding:14px 28px; }

.brand { display:flex; align-items:center; gap:12px; }

.brand-icon { width:40px; height:40px; background:rgba(255,255,255,0.18); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; border:1px solid rgba(255,255,255,0.25); }

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

  <div class="brand"><div class="brand-icon">🧪</div><div>

    <div class="brand-title">ML-Lab v3.2</div>

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

        "</style>";

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

    with gr.Blocks(title="ML-Lab v3.2") as app:

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

                data_info = gr.Textbox(label="数据信息", lines=4)

                data_summary = gr.Dataframe(label="数据摘要")

                gr.HTML('<div class="step-title">步骤 2 · 数据预处理</div>')

                with gr.Row():

                    method_dd = gr.Dropdown(choices=["mean","median","most_frequent","constant"],

                        value="mean", label="缺失值填充策略")

                    feat_sl = gr.Slider(0, 50, 0, 1, label="展示特征索引")

                    pp_btn = gr.Button("执行预处理并对比", variant="secondary")

                pp_img = gr.Image(label="预处理前后对比", height=320)

                pp_info = gr.Textbox(label="预处理结果", lines=4)


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

                result_txt_cls = gr.Textbox(label="评估结果", lines=10)

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

                result_txt_reg = gr.Textbox(label="评估结果", lines=10)


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
                fe_scale_info = gr.Textbox(label="缩放统计信息", lines=8)

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
                    fe_sel_info = gr.Textbox(label="特征选择结果", lines=10)

                gr.HTML('<div class="step-title">模块 3 · PCA 降维</div>')
                with gr.Row():
                    fe_pca_n = gr.Slider(1, 10, 2, 1, label="目标维度 (n_components)")
                    fe_pca_btn = gr.Button("执行 PCA", variant="secondary")
                with gr.Row():
                    fe_pca_img1 = gr.Image(label="方差解释率", height=320)
                    fe_pca_img2 = gr.Image(label="PCA 2D 投影", height=320)
                fe_pca_info = gr.Textbox(label="PCA 分析结果", lines=8)

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
                    fe_const_info = gr.Textbox(label="特征构造结果", lines=6)

                gr.HTML('<div class="step-title">模块 5 · 分箱离散化</div>')
                with gr.Row():
                    fe_disc_bins = gr.Slider(3, 10, 5, 1, label="分箱数")
                    fe_disc_strategy = gr.Dropdown(
                        choices=["uniform", "quantile", "kmeans"],
                        value="quantile", label="分箱策略")
                    fe_disc_btn = gr.Button("执行分箱", variant="secondary")
                with gr.Row():
                    fe_disc_img = gr.Image(label="分箱效果", height=320)
                    fe_disc_info = gr.Textbox(label="分箱结果", lines=5)

                gr.HTML('<div class="step-title">模块 6 · 相关性分析</div>')
                with gr.Row():
                    fe_corr_btn = gr.Button("生成相关性热力图", variant="secondary")
                with gr.Row():
                    fe_corr_img = gr.Image(label="特征相关性热力图", height=360)
                    fe_corr_info = gr.Textbox(label="相关性分析结果", lines=8)


            # ═══ 页面4: 聚类实验 ═══

            with gr.Column(elem_classes="page-panel", elem_id="page-cluster", visible=True) as page_cluster:

                gr.HTML(_card("🔵","聚类实验","无监督学习 · 聚类与降维算法",

                    '<span class="tab-badge tab-badge-unsupervised">无监督学习</span>'

                    '<span style="color:#64748b;font-size:12px;">K-Means · DBSCAN · 层次聚类 · PCA 降维</span>'

                    +'<div class="step-title">选择算法与参数</div>'))

                unsup_algos = get_algorithm_list("unsupervised")

                with gr.Row():

                    algo_dd_uns = gr.Dropdown(choices=unsup_algos, value="K-Means", label="算法")

                    with gr.Column():

                        n_cl = gr.Slider(2, 10, 3, 1, label="n_clusters / n_components")

                        eps_sl = gr.Slider(0.1, 5.0, 0.5, 0.1, label="eps (DBSCAN)")

                        ms_sl = gr.Slider(2, 20, 5, 1, label="min_samples (DBSCAN)")

                with gr.Row():

                    max_iter_cl = gr.Slider(50, 500, 300, 50, label="max_iter (K-Means)")

                    linkage_dd = gr.Dropdown(choices=["ward","complete","average"], value="ward", label="linkage (层次聚类)")

                    affinity_dd = gr.Dropdown(choices=["euclidean","manhattan"], value="euclidean", label="affinity (层次聚类)")

                    init_dd = gr.Dropdown(choices=["k-means++","random"], value="k-means++", label="init (K-Means)")

                train_btn_uns = gr.Button("开始聚类", variant="primary", size="lg")

                gr.HTML('<div class="step-title">聚类结果</div>')

                with gr.Row():

                    cluster_img1 = gr.Image(label="聚类散点图 / PCA 投影", height=340)

                    cluster_img2 = gr.Image(label="肘部法则 / 树状图 / 方差解释", height=340)

                cluster_img3 = gr.Image(label="补充可视化", height=340, visible=False)

                result_txt_uns = gr.Textbox(label="评估结果", lines=12)


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

        gr.HTML('<div class="footer-bar">ML-Lab v3.2 · Python · Scikit-learn · Gradio · 通义千问 · 江苏工程职业技术学院 · MIT License</div>')

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


    return app


if __name__ == "__main__":
    print("=" * 55)
    print("  ML-Lab v3.2: 机器学习可视化实验平台")
    print("  监督学习 + 无监督学习 + 代码沙箱")
    print("  地址: http://127.0.0.1:7860")
    print("=" * 55)
    app = create_app()
    app.queue(max_size=20).launch(css=APP_CSS, server_name="0.0.0.0", server_port=7860)
