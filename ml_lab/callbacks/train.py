# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.4 — 训练回调函数
从 callbacks.py 拆分而来
"""
import sys, traceback
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gradio as gr

from ml_lab.state import _g, _sync
from ml_lab.preprocessing import load_dataset, get_builtin_datasets, get_dataset_type
from ml_lab.algorithms import create_algorithm, get_algorithm_list, get_algorithm_params, get_algorithm_type, ALGORITHM_REGISTRY
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
)
from ml_lab.evaluation import (
    evaluate_classification, evaluate_regression, evaluate_clustering,
    format_evaluation_report, format_evaluation_table
)
from ml_lab.experiment_history import log_experiment, get_experiments, get_statistics
from ml_lab.learning_progress import record_activity, mark_stage_completed, get_progress_summary, STAGE_MODULE_MAP
from ml_lab.ui_styles import APP_CSS, _card, ETHICS
from ml_lab.code_generator import generate_sklearn_code
from ml_lab.code_generator import generate_sklearn_code

# ── 整数参数集合（用于参数类型转换） ──
INT_PARAMS = {
    'n_iterations', 'max_iter', 'max_depth', 'min_samples_split', 'min_samples_leaf',
    'n_estimators', 'n_neighbors', 'n_clusters', 'hidden_layer_sizes',
    'random_state', 'min_samples', 'max_leaf_nodes', 'degree',
}



from ml_lab.fptree_viz import plot_fptree
from ml_lab.ui_styles import APP_CSS, _card, ETHICS

def on_copy_classification_code(algo, lr, ni, C, md, mi, hid, al):
    """生成分类实验的可复制代码"""
    if _g.get("model") is None:
        return "⚠️ 请先训练模型，再复制代码", "📋 生成并复制代码", ""
    try:
        params = _extract_params("classification",
            algo=algo, lr=lr, ni=ni, C=C, md=md, mi=mi, hid=hid, al=al)
        code = generate_sklearn_code("classification", algo, params,
                                     _g.get("dataset_name", "未加载"), test_ratio=0.3)
        _g["last_generated_code"] = code
        return code, "✅ 已复制剪贴板", code
    except Exception as e:
        return f"代码生成失败: {e}", "📋 生成并复制代码"

def on_copy_regression_code(algo, lr, ni, C, md, mi, hid, al, deg, eps, kern, crit):

    """生成回归实验的可复制代码"""

    try:

        if _g.get("model") is None:
            return "⚠️ 请先训练模型，再复制代码", "📋 生成并复制代码", ""

        params = _extract_params("regression",

            algo=algo, lr=lr, ni=ni, C=C, md=md, mi=mi, hid=hid, al=al,

            deg=deg, eps=eps, kern=kern, crit=crit)

        code = generate_sklearn_code("regression", algo, params,

                                     _g.get("dataset_name", "未加载"), test_ratio=0.3)

        _g["last_generated_code"] = code

        return code, "✅ 已复制剪贴板", code

    except Exception as e:

        return f"代码生成失败: {e}", "📋 生成并复制代码", ""

def on_copy_clustering_code(algo, n_clusters, max_iter, eps, min_samples,

                            linkage, affinity, init_method):

    """生成聚类实验的可复制代码"""

    try:

        if _g.get("model") is None:
            return "⚠️ 请先训练模型，再复制代码", "📋 生成并复制代码", ""

        params = _extract_params("clustering",

            algo=algo, n_clusters=n_clusters, max_iter=max_iter,

            eps=eps, min_samples=min_samples, linkage=linkage,

            affinity=affinity, init=init_method)

        code = generate_sklearn_code("clustering", algo, params,

                                     _g.get("dataset_name", "未加载"), test_ratio=0.3)

        _g["last_generated_code"] = code

        return code, "✅ 已复制剪贴板", code

    except Exception as e:

        return f"代码生成失败: {e}", "📋 生成并复制代码", ""



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



            except Exception: pass



        res = evaluate_classification(_g["y_test"], ypred, yprob)



        rpt = format_evaluation_table(res, "classification")



        cm = fig_to_image(plot_confusion_matrix(_g["y_test"],ypred,class_names=_g["target_names"]))



        roc = None



        if yprob is not None and yprob.shape[1]>=2:



            roc = fig_to_image(plot_roc_curve(_g["y_test"],yprob,class_names=_g["target_names"]))



        try: db = fig_to_image(plot_decision_boundary(model,_g["X_train"],_g["y_train"],title=f"{algo} 决策边界"))



        except Exception: db = None



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



                    except Exception: pass



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
