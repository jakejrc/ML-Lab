# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.1 — 回调函数
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
        _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
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








def on_run_association(algo_name, min_support, min_confidence, min_lift,

                       max_length, disc_method, n_bins, top_k):

    """执行关联规则挖掘"""

    if _g["X"] is None:

        return (None, None, None, None,

                '<div style="color:#f87171;padding:20px;text-align:center;">请先在数据工作台加载数据集</div>',

                gr.HTML(''))



    try:

        algo_name = algo_name if algo_name else "Apriori"



        # 创建模型

        if algo_name == "Apriori":

            model = AprioriModel(

                min_support=float(min_support),

                min_confidence=float(min_confidence),

                min_lift=float(min_lift),

                max_length=int(max_length),

                discretize_method=disc_method,

                n_bins=int(n_bins),

            )

        else:  # FP-Growth

            model = FPGrowthModel(

                min_support=float(min_support),

                min_confidence=float(min_confidence),

                min_lift=float(min_lift),

                max_length=int(max_length),

                discretize_method=disc_method,

                n_bins=int(n_bins),

            )



        model.fit(_g["X"], y=_g["y"],

                  feature_names=_g["feature_names"],

                  target_names=_g["target_names"])



        _g["model"] = model

        _g["model_name"] = algo_name

        _g["last_assoc_rules"] = model.rules

        _g["last_assoc_model"] = model



        # 生成图表

        top_k_val = int(top_k)

        img1 = fig_to_image(plot_top_frequent_items(model, top_k=top_k_val,

                           title=f"{algo_name} Top-{top_k_val} 频繁项集"))

        img2 = fig_to_image(plot_rules_scatter(model, top_k=min(top_k_val, 50),

                           title=f"{algo_name} 关联规则 (置信度 vs 提升度)"))

        img3 = fig_to_image(plot_item_length_distribution(model,

                           title=f"{algo_name} 频繁项集长度分布"))

        if algo_name == "FP-Growth":

            img4 = fig_to_image(plot_fptree(model, title="FP-Growth 树结构可视化"))

        else:

            img4 = fig_to_image(plot_rules_heatmap(model, top_k=min(top_k_val, 20),

                               title=f"{algo_name} 关联规则热力图"))



        # 评估结果表格

        eval_res = eval_assoc(model)

        rpt = _format_assoc_table(eval_res, model)



        # 状态更新

        md_html = (f'<div class="status-card" style="margin-top:6px;"><div style="display:flex;'

                   f'justify-content:space-between;align-items:center;"><span class="status-label">模型</span>'

                   f'<span class="status-value" id="sv-md">{algo_name}</span></div></div>')



        return img1, img2, img3, img4, rpt, md_html



    except Exception as e:

        import traceback

        return (None, None, None, None,

                f"挖掘失败: {e}\n{traceback.format_exc()}",

                gr.HTML(''))






def _format_assoc_table(eval_res, model):

    """格式化关联规则评估结果为 HTML 表格"""

    rules = model.rules

    n_rules_shown = min(len(rules), 15)



    html = '<div style="font-family:system-ui,sans-serif;">'



    # 统计信息表

    html += '<div style="margin-bottom:16px;">'

    html += '<h4 style="margin:0 0 8px;color:#e2e8f0;font-size:13px;">挖掘统计</h4>'

    html += '<table class="info-table">'

    stats = [

        ("事务数", eval_res.get("n_transactions", 0)),

        ("布尔项数", eval_res.get("n_items", 0)),

        ("频繁项集数", eval_res.get("n_frequent_items", 0)),

        ("关联规则数", eval_res.get("n_rules", 0)),

        ("运行耗时", eval_res.get("elapsed_time", "N/A")),

    ]

    if rules:

        stats.extend([

            ("平均置信度", f"{eval_res.get('avg_confidence', 0):.4f}"),

            ("最大置信度", f"{eval_res.get('max_confidence', 0):.4f}"),

            ("平均提升度", f"{eval_res.get('avg_lift', 0):.4f}"),

            ("最大提升度", f"{eval_res.get('max_lift', 0):.4f}"),

        ])

    for label, val in stats:

        html += f'<tr><td style="color:#94a3b8;padding:4px 10px;font-size:12px;">{label}</td>'

        html += f'<td style="padding:4px 10px;font-size:12px;font-weight:600;">{val}</td></tr>'

    html += '</table></div>'



    # 关联规则表

    if rules:

        html += '<h4 style="margin:0 0 8px;color:#e2e8f0;font-size:13px;">'

        html += f'关联规则 (Top {n_rules_shown}/{len(rules)})</h4>'

        html += '<div style="overflow-x:auto;max-height:400px;overflow-y:auto;">'

        html += '<table class="eval-table" style="font-size:11px;">'

        html += '<tr style="background:rgba(255,255,255,0.05);">'

        html += '<th style="padding:6px 8px;text-align:left;">#</th>'

        html += '<th style="padding:6px 8px;text-align:left;">前件 (Antecedent)</th>'

        html += '<th style="padding:6px 8px;text-align:left;">后件 (Consequent)</th>'

        html += '<th style="padding:6px 8px;text-align:right;">支持度</th>'

        html += '<th style="padding:6px 8px;text-align:right;">置信度</th>'

        html += '<th style="padding:6px 8px;text-align:right;">提升度</th>'

        html += '</tr>'

        for idx, r in enumerate(rules[:n_rules_shown]):

            bg = 'background:rgba(255,255,255,0.02);' if idx % 2 == 0 else ''

            ant = " + ".join(sorted(r["antecedent"]))

            cons = " + ".join(sorted(r["consequent"]))

            lift_val = r["lift"]

            lift_color = "#34d399" if lift_val >= 1.5 else ("#fbbf24" if lift_val >= 1.0 else "#f87171")

            html += f'<tr style="{bg}">'

            html += f'<td style="padding:4px 8px;color:#64748b;">{idx+1}</td>'

            html += f'<td style="padding:4px 8px;">{ant}</td>'

            html += f'<td style="padding:4px 8px;font-weight:600;">{cons}</td>'

            html += f'<td style="padding:4px 8px;text-align:right;">{r["support"]:.4f}</td>'

            html += f'<td style="padding:4px 8px;text-align:right;">{r["confidence"]:.4f}</td>'

            html += f'<td style="padding:4px 8px;text-align:right;color:{lift_color};font-weight:700;">{lift_val:.4f}</td>'

            html += '</tr>'

        html += '</table></div>'

        if len(rules) > n_rules_shown:

            html += f'<div style="color:#64748b;font-size:11px;margin-top:6px;">'

            html += f'共 {len(rules)} 条规则，仅显示前 {n_rules_shown} 条 (按提升度降序)</div>'

    else:

        html += '<div style="color:#fbbf24;padding:16px;text-align:center;font-size:13px;">'

        html += '未找到满足条件的关联规则。<br>建议降低 min_confidence 或 min_lift 阈值。</div>'



    html += '</div>'

    return html






def on_copy_association_code(algo_name, min_support, min_confidence, min_lift,

                              max_length, disc_method, n_bins):

    """生成关联规则实验的可复制代码"""

    if _g["X"] is None:

        return "请先加载数据集"



    algo = algo_name if algo_name else "Apriori"

    fn = "AprioriModel" if algo == "Apriori" else "FPGrowthModel"

    ds_name = _g.get("dataset_name", "未加载")



    code = f"""# ML-Lab 关联规则挖掘代码 — {algo}

# 数据集: {ds_name}



import numpy as np

import pandas as pd

from ml_lab.association_rules import {fn}, _to_boolean_matrix, generate_rules



# 1. 加载数据 (已由数据工作台加载)

X = np.array({_g["X"].tolist()})

y = np.array({_g["y"].tolist()})

feature_names = {_g["feature_names"]}

target_names = {_g["target_names"]}



# 2. 转换为布尔事务矩阵

df_bool, item_map = _to_boolean_matrix(

    X, feature_names, y, target_names,

    discretize_method="{disc_method}", n_bins={n_bins}

)

print(f"事务矩阵: {{df_bool.shape[0]}} 条事务, {{df_bool.shape[1]}} 个项")



# 3. 创建并训练模型

model = {fn}(

    min_support={float(min_support)},

    min_confidence={float(min_confidence)},

    min_lift={float(min_lift)},

    max_length={int(max_length)},

    discretize_method="{disc_method}",

    n_bins={int(n_bins)},

)

model.fit(X, y=y, feature_names=feature_names, target_names=target_names)



# 4. 输出结果

print(f"\n频繁项集数: {{len(model.frequent_items)}}")

print(f"关联规则数: {{len(model.rules)}}")



# 5. Top-10 关联规则

print("\nTop-10 关联规则 (按提升度排序):")

print("-" * 80)

for i, r in enumerate(model.rules[:10]):

    ant = " + ".join(sorted(r["antecedent"]))

    cons = " + ".join(sorted(r["consequent"]))

    print(f"{{i+1}}. {{ant}} -> {{cons}}")

    print(f"   支持度={{r['support']:.4f}}, 置信度={{r['confidence']:.4f}}, 提升度={{r['lift']:.4f}}")

"""

    _g["last_generated_code"] = code

    return code





# 回调 — 代码沙箱（v3.0 新增）



# ═══════════════════════════════════════════════════════════════




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

            if img_path and os.path.isfile(img_path):

                images.append((key, img_path))

        algo = _g.get("last_algo_name") or "实验"

        report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")

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

ANDBOX_TEMPLATES = {

    "默认（查看数据）": """# ML-Lab 代码沙箱 — 在此编写和运行 Python 代码



# 已预导入: numpy (as np), pandas (as pd), sklearn, matplotlib, os, json

# 数据变量: X, y, X_train, y_train, X_test, y_test, feature_names, target_names

# 提示: 可直接在下方上传 CSV/Excel 文件，无需前往数据工作台



import numpy as np

import pandas as pd



# 查看数据形状

print("数据形状:", X.shape)

print("前5行:\\n", pd.DataFrame(X[:5], columns=feature_names))

""",



    "基础折线图": """import matplotlib.pyplot as plt

import numpy as np



# 生成示例数据

x = np.linspace(0, 2 * np.pi, 100)

y1 = np.sin(x)

y2 = np.cos(x)



# 绘制折线图

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, y1, label='sin(x)', linewidth=2, color='#4F46E5')

ax.plot(x, y2, label='cos(x)', linewidth=2, color='#10B981', linestyle='--')

ax.set_title('三角函数图像', fontsize=16)

ax.set_xlabel('x', fontsize=13)

ax.set_ylabel('y', fontsize=13)

ax.legend(fontsize=12)

ax.grid(True, alpha=0.3)

ax.set_facecolor('#FAFAFA')

plt.tight_layout()

plt.show()

""",



    "柱状图": """import matplotlib.pyplot as plt

import numpy as np



# 示例数据

categories = ['语文', '数学', '英语', '物理', '化学']

scores_A = [85, 92, 78, 88, 76]

scores_B = [90, 80, 85, 72, 82]



fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(categories))

width = 0.35



bars1 = ax.bar(x - width/2, scores_A, width, label='班级A', color='#4F46E5', edgecolor='white')

bars2 = ax.bar(x + width/2, scores_B, width, label='班级B', color='#F59E0B', edgecolor='white')



ax.set_title('各科成绩对比', fontsize=16)

ax.set_xlabel('科目', fontsize=13)

ax.set_ylabel('分数', fontsize=13)

ax.set_xticks(x)

ax.set_xticklabels(categories, fontsize=12)

ax.legend(fontsize=12)

ax.grid(axis='y', alpha=0.3)

ax.set_facecolor('#FAFAFA')



# 数值标签

for bar in bars1:

    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,

            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)

for bar in bars2:

    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,

            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)



plt.tight_layout()

plt.show()

""",



    "散点图": """import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs



# 生成聚类数据

X_blob, y_blob = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444']



fig, ax = plt.subplots(figsize=(8, 6))

for i, c in enumerate(colors):

    mask = y_blob == i

    ax.scatter(X_blob[mask, 0], X_blob[mask, 1], c=c, s=30, alpha=0.7,

               label=f'类别 {i}', edgecolors='white', linewidth=0.5)



ax.set_title('散点图 — 聚类数据可视化', fontsize=16)

ax.set_xlabel('特征 1', fontsize=13)

ax.set_ylabel('特征 2', fontsize=13)

ax.legend(fontsize=11, markerscale=1.5)

ax.grid(True, alpha=0.3)

ax.set_facecolor('#FAFAFA')

plt.tight_layout()

plt.show()

""",



    "饼图": """import matplotlib.pyplot as plt



# 数据

labels = ['Python', 'JavaScript', 'Java', 'C/C++', 'Go', '其他']

sizes = [28, 22, 16, 12, 8, 14]

colors = ['#4F46E5', '#F59E0B', '#EF4444', '#10B981', '#8B5CF6', '#94A3B8']

explode = (0.05, 0, 0, 0, 0, 0)  # 突出Python



fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(

    sizes, explode=explode, labels=labels, colors=colors,

    autopct='%1.1f%%', startangle=140, pctdistance=0.75,

    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)

)



for text in texts:

    text.set_fontsize(12)

for autotext in autotexts:

    autotext.set_fontsize(10)

    autotext.set_color('white')

    autotext.set_fontweight('bold')



ax.set_title('编程语言使用占比（环形图）', fontsize=16, pad=20)

plt.tight_layout()

plt.show()

""",



    "直方图 + 箱线图": """import matplotlib.pyplot as plt

import numpy as np



# 生成正态分布数据

np.random.seed(42)

data = np.random.normal(loc=100, scale=15, size=500)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))



# 直方图

ax1.hist(data, bins=30, color='#4F46E5', edgecolor='white', alpha=0.85)

ax1.axvline(data.mean(), color='#EF4444', linestyle='--', linewidth=2, label=f'均值={data.mean():.1f}')

ax1.axvline(np.median(data), color='#10B981', linestyle='--', linewidth=2, label=f'中位数={np.median(data):.1f}')

ax1.set_title('成绩分布直方图', fontsize=14)

ax1.set_xlabel('分数', fontsize=12)

ax1.set_ylabel('人数', fontsize=12)

ax1.legend(fontsize=11)

ax1.grid(axis='y', alpha=0.3)

ax1.set_facecolor('#FAFAFA')



# 箱线图

bp = ax2.boxplot(data, patch_artist=True, widths=0.4,

                  boxprops=dict(facecolor='#DBEAFE', edgecolor='#4F46E5', linewidth=1.5),

                  medianprops=dict(color='#EF4444', linewidth=2),

                  whiskerprops=dict(color='#4F46E5'),

                  capprops=dict(color='#4F46E5'),

                  flierprops=dict(marker='o', markerfacecolor='#F59E0B', markersize=5))

ax2.set_title('成绩箱线图', fontsize=14)

ax2.set_ylabel('分数', fontsize=12)

ax2.set_xticklabels(['成绩'])

ax2.grid(axis='y', alpha=0.3)

ax2.set_facecolor('#FAFAFA')



plt.tight_layout()

plt.show()

""",



    "热力图（相关系数矩阵）": """import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# 使用已加载的数据集

df = pd.DataFrame(X, columns=feature_names)

if hasattr(df.iloc[0, 0], '__float__') or np.issubdtype(df.iloc[0, 0].dtype, np.number):

    corr = df.corr()



    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')



    ax.set_xticks(range(len(corr.columns)))

    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)

    ax.set_yticklabels(corr.columns, fontsize=10)



    # 数值标注

    for i in range(len(corr)):

        for j in range(len(corr)):

            val = corr.iloc[i, j]

            color = 'white' if abs(val) > 0.5 else 'black'

            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)



    plt.colorbar(im, ax=ax, shrink=0.8, label='相关系数')

    ax.set_title('特征相关系数热力图', fontsize=16)

    plt.tight_layout()

    plt.show()

else:

    print("当前数据集不适合绘制相关系数矩阵（非数值型特征）")

""",



    "3D 散点图": """import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll

from mpl_toolkits.mplot3d import Axes3D



# 生成瑞士卷数据

X_sr, t = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)



fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection='3d')



scatter = ax.scatter(X_sr[:, 0], X_sr[:, 1], X_sr[:, 2],

                     c=t, cmap='viridis', s=10, alpha=0.8, edgecolors='none')

ax.set_title('3D 散点图 — 瑞士卷 (Swiss Roll)', fontsize=14)

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('Z')

fig.colorbar(scatter, ax=ax, shrink=0.6, label='t 参数')

ax.view_init(elev=20, azim=45)

plt.tight_layout()

plt.show()

""",



    "子图组合（多图排列）": """import matplotlib.pyplot as plt

import numpy as np



np.random.seed(42)

x = np.linspace(0, 10, 50)



fig, axes = plt.subplots(2, 2, figsize=(10, 8))

fig.suptitle('子图组合示例', fontsize=16, fontweight='bold')



# 左上：折线图

axes[0, 0].plot(x, np.sin(x), color='#4F46E5', linewidth=2)

axes[0, 0].set_title('折线图')

axes[0, 0].grid(True, alpha=0.3)



# 右上：柱状图

axes[0, 1].bar(['A', 'B', 'C', 'D'], [15, 30, 45, 20], color=['#4F46E5', '#10B981', '#F59E0B', '#EF4444'])

axes[0, 1].set_title('柱状图')



# 左下：散点图

axes[1, 0].scatter(np.random.randn(50), np.random.randn(50), c='#8B5CF6', s=40, alpha=0.7)

axes[1, 0].set_title('散点图')

axes[1, 0].grid(True, alpha=0.3)



# 右下：饼图

axes[1, 1].pie([35, 25, 20, 20], labels=['Q1', 'Q2', 'Q3', 'Q4'],

               colors=['#4F46E5', '#10B981', '#F59E0B', '#EF4444'],

               autopct='%1.0f%%', startangle=90,

               wedgeprops=dict(edgecolor='white', linewidth=2))

axes[1, 1].set_title('饼图')



plt.tight_layout()

plt.show()

""",



    "机器学习流程（分类）": """# 使用已加载数据集完成分类流程

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns



# 训练模型

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)



# 预测

y_pred = model.predict(X_test)



# 评估

print("=== 分类报告 ===")

print(classification_report(y_test, y_pred, target_names=target_names))



print(f"训练集准确率: {model.score(X_train, y_train):.4f}")

print(f"测试集准确率: {model.score(X_test, y_test):.4f}")



# 特征重要性

if hasattr(model, 'feature_importances_'):

    import numpy as np

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))



    # 混淆矩阵

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,

                xticklabels=target_names, yticklabels=target_names)

    ax1.set_title('混淆矩阵', fontsize=14)

    ax1.set_xlabel('预测值')

    ax1.set_ylabel('真实值')



    # 特征重要性

    importance.plot(kind='barh', color='#4F46E5', ax=ax2)

    ax2.set_title('特征重要性 Top 10', fontsize=14)

    ax2.set_xlabel('重要性')

    ax2.grid(axis='x', alpha=0.3)

    ax2.set_facecolor('#FAFAFA')



    plt.tight_layout()

    plt.show()

""",



    "机器学习流程（回归）": """# 使用已加载数据集完成回归流程

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import numpy as np



# 训练两个模型

models = {

    '线性回归': LinearRegression(),

    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42)

}



fig, axes = plt.subplots(1, 2, figsize=(12, 5))



for idx, (name, model) in enumerate(models.items()):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    r2 = r2_score(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'{name}: R²={r2:.4f}, RMSE={rmse:.4f}')



    # 预测值 vs 真实值

    ax = axes[idx]

    ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#4F46E5', edgecolors='white')



    min_val = min(y_test.min(), y_pred.min())

    max_val = max(y_test.max(), y_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')

    ax.set_title(f'{name}\nR²={r2:.4f}', fontsize=13)

    ax.set_xlabel('真实值')

    ax.set_ylabel('预测值')

    ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3)

    ax.set_facecolor('#FAFAFA')

    ax.set_aspect('equal', adjustable='box')



plt.suptitle('回归模型: 预测值 vs 真实值', fontsize=15)

plt.tight_layout()

plt.show()

""",

}



# 默认模板（兼容旧引用）

# SANDBOX_TEMPLATE removed - use SANDBOX_TEMPLATES directly






def on_download_code(code):

    """将代码编辑器内容保存为 .py 文件供下载"""

    import tempfile, os

    if not code or not code.strip():

        return None

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')

    # 添加文件头注释

    header = "# -*- coding: utf-8 -*-\n# ML-Lab 代码沙箱导出\n# 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"

    tmp.write(header + code)

    tmp.close()

    return tmp.name












def on_sandbox_load_data(file_obj):

    """代码沙箱 - 加载用户上传的数据集，注入全局状态"""

    import os as _os

    if file_obj is None:

        return "⚠️ 请先选择 CSV 或 Excel 文件", None



    fp = file_obj if isinstance(file_obj, str) else file_obj.name

    if not fp or not os.path.exists(fp):

        return f"文件不存在: {fp}", None



    try:

        from ml_lab.preprocessing import load_custom_dataset, get_dataset_summary, split_dataset

        from ml_lab.visualization import plot_data_distribution, fig_to_image



        X, y, fn, tn, desc = load_custom_dataset(fp, "-1", "auto")



        # 判断任务类型

        _y_valid = y[~np.isnan(y.astype(float))] if np.issubdtype(y.dtype, np.floating) else y

        _unique = len(np.unique(_y_valid))

        dtype = "classification" if _unique <= 20 else "regression"



        # 分割数据集（stratify 失败时自动降级为随机分割）

        if dtype in ("classification", "regression"):

            strat = dtype == "classification"

            try:

                Xtr, Xte, ytr, yte = split_dataset(X, y, test_size=0.2, stratify=strat)

            except ValueError:

                # 类别样本不足，回退到非分层分割

                Xtr, Xte, ytr, yte = split_dataset(X, y, test_size=0.2, stratify=False)

            _sync(Xtr, Xte, ytr, yte, fn, tn, X, y)

        else:

            _sync(X, None, y, None, fn, tn, X, y)



        ds_name = os.path.basename(fp)

        _g["dataset_name"] = f"沙箱上传: {ds_name}"



        info = (f"✅ 数据加载成功!\n"

                f"文件: {ds_name}\n"

                f"任务类型: {'分类' if dtype == 'classification' else '回归'}\n"

                f"样本数: {X.shape[0]}\n"

                f"特征数: {X.shape[1]}\n"

                f"特征列: {fn}\n"

                f"目标列: {tn if tn else '(索引-1)'}\n"

                f"训练集: {Xtr.shape[0]} / 测试集: {Xte.shape[0]}\n\n"

                f"现在可以使用变量 X, y, X_train, y_train, X_test, y_test 运行代码了!")

        return info, None



    except Exception as e:

        return f"❌ 加载失败: {type(e).__name__}: {e}", None




def on_run_code(code):



    """安全执行用户代码，捕获输出和 matplotlib 图形"""



    import matplotlib.pyplot as _plt



    if not code.strip():



        return "请输入代码", None



    old_stdout = sys.stdout



    old_stderr = sys.stderr



    sys.stdout = io.StringIO()



    sys.stderr = io.StringIO()



    # 数据未加载时注入提示变量，避免 AttributeError



    # 数据未加载时，仍允许运行纯计算代码（仅提示数据变量不可用）

    data_loaded = _g["X"] is not None



    local_ns = {"np": np, "pd": __import__("pandas"),

                "os": __import__("os"),

                "json": __import__("json")}

    if data_loaded:

        local_ns.update({

                "X": _g["X"], "y": _g["y"],

                "feature_names": _g["feature_names"],

                "target_names": _g["target_names"]})



    local_ns["matplotlib"] = __import__("matplotlib")



    local_ns["plt"] = local_ns["matplotlib"].pyplot



    local_ns["sklearn"] = __import__("sklearn")



    if data_loaded:

        local_ns["X_train"] = _g["X_train"]

        local_ns["y_train"] = _g["y_train"]

        local_ns["X_test"] = _g["X_test"]

        local_ns["y_test"] = _g["y_test"]

    else:

        local_ns["X"] = local_ns["y"] = None

        local_ns["X_train"] = local_ns["y_train"] = None

        local_ns["X_test"] = local_ns["y_test"] = None

        local_ns["feature_names"] = local_ns["target_names"] = None



    plot_path = None



    try:



        # 记录 exec 前已有的 figure 编号



        fig_ids_before = set(_plt.get_fignums())



        exec(code, {"__builtins__": __builtins__}, local_ns)



        output = sys.stdout.getvalue()



        errors = sys.stderr.getvalue()



        result = output



        if errors.strip():



            result += "\n[stderr]\n" + errors



        if not result.strip():



            result = "(代码执行成功，无输出)"



        # 捕获 exec 后新增的 figure



        fig_ids_after = set(_plt.get_fignums())



        new_fig_ids = fig_ids_after - fig_ids_before



        if new_fig_ids:



            import tempfile



            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)



            tmp.close()



            # 取最新一个 figure 保存



            latest_fig = _plt.figure(max(new_fig_ids))



            latest_fig.savefig(tmp.name, dpi=120, bbox_inches="tight")



            plot_path = tmp.name



            # 关闭用户新建的 figure，避免内存泄漏



            for fid in sorted(new_fig_ids):



                _plt.close(fid)



        return result, plot_path



    except Exception as e:



        return f"错误: {type(e).__name__}: {e}", None



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

            for s in stats:

                feat = s["特征"]

                bef_mean = s["缩放前_均值"]

                aft_mean = s["缩放后_均值"]

                bef_std = s["缩放前_标准差"]

                aft_std = s["缩放后_标准差"]

                rows += "<tr><td>" + feat + "</td><td>" + f"{bef_mean:.4f}" + "</td><td>" + f"{aft_mean:.4f}" + "</td><td>" + f"{bef_std:.4f}" + "</td><td>" + f"{aft_std:.4f}" + "</td></tr>\n"

        html = ('<div class="scale-meta">'

                '<span>' + "缩放方法" + ': <b>' + method + '</b></span>'

                '<span>' + "特征数" + ': <b>' + str(X_tr.shape[1]) + '</b></span>'

                '<span>' + "样本数" + ': <b>' + str(X_tr.shape[0]) + '</b></span></div>'

                '<table class="scale-table"><thead><tr>'

                '<th>' + "特征" + '</th><th>' + "缩放前均值" + '</th><th>' + "缩放后均值" + '</th>'

                '<th>' + "缩放前标准差" + '</th><th>' + "缩放后标准差" + '</th>'

                '</tr></thead><tbody>' + rows + '</tbody></table>' + scale_style)

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






def on_load_template(template_name):

    """根据模板名称加载代码沙箱模板"""

    return SANDBOX_TEMPLATES.get(template_name, SANDBOX_TEMPLATES["默认（查看数据）"])





# ═══════════════════════════════════════════════════════════════



# 回调 — AI 助教



# ═══════════════════════════════════════════════════════════════
# 回调 — 知识图谱

def on_kg_render():
    """渲染知识图谱3D交互式可视化"""
    try:
        kg_html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kg_3d.html")
        if not os.path.exists(kg_html_path):
            return '<div style="color:#f87171;padding:20px;">kg_3d.html 文件不存在</div>'
        with open(kg_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        import base64
        b64 = base64.b64encode(html_content.encode('utf-8')).decode('ascii')
        return f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:85vh;border:none;border-radius:12px;"></iframe>'
    except Exception as e:
        return f'<div style="color:#f87171;padding:20px;">渲染失败: {e}</div>'




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











# ═══════════════════════════════════════════════════════════════



# HTML 片段



# ═══════════════════════════════════════════════════════════════











LEARNING_PATH_HTML = """<div class="learning-path">

<div class="lp-timeline">

    <div class="lp-step" data-nav="数据工作台" data-nav-id="page-data" data-step="data">

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

    <div class="lp-step" data-nav="特征工程" data-nav-id="page-fe" data-step="fe">

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

    <div class="lp-step" data-nav="分类实验" data-nav-id="page-classify" data-step="cls">

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

    <div class="lp-step" data-nav="回归实验" data-nav-id="page-regress" data-step="reg">

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

    <div class="lp-step" data-nav="聚类实验" data-nav-id="page-cluster" data-step="clu">

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

    <div class="lp-step" data-nav="关联规则" data-nav-id="page-assoc" data-step="assoc">

        <div class="lp-step-title">

            🔗 关联规则挖掘

            <span class="lp-step-badge lp-badge-unsup">关联规则</span>

            <span class="lp-check" id="lp-check-assoc">✅</span>

        </div>

        <div class="lp-step-desc">使用 Apriori、FP-Growth 算法进行关联规则挖掘，发现购物篮中商品之间的频繁模式和关联关系。</div>

        <div class="lp-knowledge">

            <b>📚 核心知识点：</b><br>

            • 支持度：项集在所有事务中出现的频率<br>

            • 置信度：包含前件的事务中同时包含后件的概率<br>

            • 提升度：置信度与后件支持度的比值，衡量规则的实际价值<br>

            • Apriori：逐层搜索的候选集生成策略<br>

            • FP-Growth：无需候选集，基于FP树的高效挖掘算法

        </div>

        <div class="lp-step-tasks"><b>实验任务：</b>加载购物篮数据集 → Apriori挖掘 → FP-Growth对比 → 调整支持度/置信度参数</div>

        <div class="lp-step-tasks" style="margin-top:6px;"><b>思考问题：</b><br>• 支持度和置信度分别代表什么含义？<br>• 提升度大于1说明什么？<br>• Apriori 和 FP-Growth 的效率差异在哪里？<br>• 经典的"啤酒与尿布"关联在实际业务中有何意义？</div>

    </div>



    <div class="lp-step" data-nav="代码沙箱" data-nav-id="page-code" data-step="code">

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









# ═══════════════════════════════════════════════════════════════



# 参数面板构建辅助



# ═══════════════════════════════════════════════════════════════




def on_learning_path_click(evt: gr.EventData):

    """学习路径步骤点击时跳转到对应页面"""

    return gr.skip()  # 由 JS 处理跳转



