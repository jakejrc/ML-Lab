# -*- coding: utf-8 -*-
"""
ML-Lab: 机器学习可视化实验平台
左右布局 · 蓝白主题 · Gradio 6
页面切换：纯 CSS display + JS DOM 操作（绕开 Gradio 6 Column visible bug）
"""

import sys, os, argparse, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_lab.preprocessing import (
    load_dataset, get_builtin_datasets, handle_missing,
    inject_missing, scale_features, split_dataset, get_dataset_summary
)
from ml_lab.algorithms import (
    create_algorithm, get_algorithm_list, get_algorithm_params, ALGORITHM_REGISTRY
)
from ml_lab.visualization import (
    plot_data_distribution, plot_preprocessing_comparison,
    plot_decision_boundary, plot_training_history,
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_regression_results, plot_gradient_descent_2d, fig_to_image
)
from ml_lab.evaluation import (
    evaluate_classification, evaluate_regression,
    format_evaluation_report, get_evaluation_dataframe
)
from ml_lab.llm_assistant import (
    ask, reset_conversation, get_api_key, PRESET_QUESTIONS, SYSTEM_PROMPT
)

import gradio as gr

# ═══════════════════════════════════════════════════════════════
# 全局状态
# ═══════════════════════════════════════════════════════════════
_g = dict(
    X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None,
    feature_names=None, target_names=None, model=None,
    dataset_name="未加载", model_name="未训练",
)

def _sync(X_train, X_test, y_train, y_test, fnames, tnames, X=None, y=None):
    _g.update(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
              feature_names=fnames, target_names=tnames,
              X=X if X is not None else X_train, y=y if y is not None else y_train)

# ═══════════════════════════════════════════════════════════════
# 回调函数
# ═══════════════════════════════════════════════════════════════

def on_load_data(dataset_name, test_ratio):
    try:
        X, y, fn, tn, desc = load_dataset(dataset_name)
        summary = get_dataset_summary(X, y, fn)
        fig = plot_data_distribution(X, y, title=dataset_name, feature_names=fn)
        img = fig_to_image(fig)
        Xtr, Xte, ytr, yte = split_dataset(X, y, test_size=float(test_ratio))
        _sync(Xtr, Xte, ytr, yte, fn, tn, X, y)
        _g["dataset_name"] = dataset_name
        info = (f"数据集: {dataset_name}\n样本: {X.shape[0]}, 特征: {X.shape[1]}, "
                f"类别: {len(np.unique(y))}\n训练集: {Xtr.shape[0]}, 测试集: {Xte.shape[0]}")
        ds_html = f'<div class="status-card"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">数据集</span><span class="status-value" id="sv-ds">{dataset_name}</span></div></div>'
        return img, info, summary, ds_html
    except Exception as e:
        return None, f"加载失败: {e}", None, gr.HTML('<div class="status-card"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">数据集</span><span class="status-value" id="sv-ds">加载失败</span></div></div>')

def on_preprocess(method, feat_idx):
    if _g["X_train"] is None: return None, "请先加载数据集"
    idx = min(int(feat_idx), _g["X_train"].shape[1] - 1)
    Xm = inject_missing(_g["X_train"], 0.1) if not np.isnan(_g["X_train"]).any() else _g["X_train"].copy()
    Xc = handle_missing(Xm, strategy=method)
    Xs, _, _ = scale_features(Xc, method="standard")
    fig = plot_preprocessing_comparison(Xm, Xs, f"缺失值填充({method})+标准化", feature_idx=idx)
    return fig_to_image(fig), (f"原始缺失: {int(np.isnan(Xm).sum())}\n处理后缺失: {int(np.isnan(Xs).sum())}\n"
                                f"特征{idx} 均值: {np.nanmean(Xs[:,idx]):.3f}\n标准差: {np.nanstd(Xs[:,idx]):.3f}")

def on_train(algo, lr, n_iter, C, max_depth, max_iter, hidden, alpha):
    if _g["X_train"] is None: return None, None, "请先加载数据集", None, None
    try:
        ap = get_algorithm_params(algo)
        pm = {"learning_rate":lr,"n_iterations":n_iter,"n_estimators":n_iter,
              "C":C,"max_depth":max_depth,"max_iter":max_iter,
              "hidden_layer_sizes":hidden,"learning_rate_init":lr,"alpha":alpha}
        cp = {}
        if ap:
            for p in ap:
                if p in pm and pm[p] is not None:
                    v = pm[p]
                    if isinstance(v, (int,float)): cp[p] = v
                    else:
                        try: cp[p] = float(v) if '.' in str(v) else int(v)
                        except: cp[p] = v
        model = create_algorithm(algo, **cp)
        model.fit(_g["X_train"], _g["y_train"])
        _g["model"], _g["model_name"] = model, algo
        ypred = model.predict(_g["X_test"])
        yprob = None
        if hasattr(model,'predict_proba'):
            try: yprob = model.predict_proba(_g["X_test"])
            except: pass
        atype = ALGORITHM_REGISTRY[algo]["type"]
        if atype == "classification":
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
            md_html = f'<div class="status-card" style="margin-top:6px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">模型</span><span class="status-value" id="sv-md">{algo}</span></div></div>'
            return tr, db, rpt, cm, roc, md_html
        else:
            res = evaluate_regression(_g["y_test"],ypred)
            rpt = format_evaluation_report(res,"regression")
            rg = fig_to_image(plot_regression_results(_g["y_test"],ypred,title=f"{algo} 回归结果"))
            gd = None
            if _g["X_train"].shape[1]>=1 and hasattr(model,'history'):
                gd = fig_to_image(plot_gradient_descent_2d(_g["X_train"][:,0],_g["y_train"],
                    learning_rate=cp.get('learning_rate',0.01)))
            md_html = f'<div class="status-card" style="margin-top:6px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">模型</span><span class="status-value" id="sv-md">{algo}</span></div></div>'
            return gd, rg, rpt, None, None, md_html
    except Exception as e:
        import traceback
        md_html = '<div class="status-card" style="margin-top:6px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">模型</span><span class="status-value" id="sv-md">训练失败</span></div></div>'
        return None,None,f"训练失败: {e}\n{traceback.format_exc()}",None,None,md_html

def on_evaluate():
    if _g["model"] is None: return "请先训练模型", None, None
    ypred = _g["model"].predict(_g["X_test"])
    aname = next((n for n,info in ALGORITHM_REGISTRY.items() if isinstance(_g["model"],info["class"])),None)
    atype = ALGORITHM_REGISTRY.get(aname,{}).get("type","classification")
    if atype == "classification":
        yprob = None
        if hasattr(_g["model"],'predict_proba'):
            try: yprob = _g["model"].predict_proba(_g["X_test"])
            except: pass
        res = evaluate_classification(_g["y_test"],ypred,yprob)
        rpt = format_evaluation_report(res,"classification")
        cm = fig_to_image(plot_confusion_matrix(_g["y_test"],ypred,class_names=_g["target_names"],title="测试集混淆矩阵"))
        roc = None
        if yprob is not None and yprob.shape[1]>=2:
            roc = fig_to_image(plot_roc_curve(_g["y_test"],yprob,class_names=_g["target_names"],title="测试集ROC曲线"))
        return rpt, cm, roc
    else:
        res = evaluate_regression(_g["y_test"],ypred)
        rpt = format_evaluation_report(res,"regression")
        rg = fig_to_image(plot_regression_results(_g["y_test"],ypred,title="测试集回归结果"))
        return rpt, rg, None

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
# CSS — 关键：用 .page-panel 来控制页面显示/隐藏
# ═══════════════════════════════════════════════════════════════

APP_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
body { background: #f1f5f9 !important; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif !important; }

/* 顶部导航 */
.top-nav { background: linear-gradient(135deg,#1e40af 0%,#2563eb 60%,#3b82f6 100%); box-shadow: 0 4px 16px rgba(30,64,175,0.25); }
.top-nav-inner { max-width:1600px; margin:0 auto; display:flex; align-items:center; justify-content:space-between; padding:14px 28px; }
.brand { display:flex; align-items:center; gap:12px; }
.brand-icon { width:40px; height:40px; background:rgba(255,255,255,0.18); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; border:1px solid rgba(255,255,255,0.25); }
.brand-title { color:#fff!important; font-size:17px; font-weight:700; }
.brand-sub { color:rgba(255,255,255,0.8)!important; font-size:11px; margin-top:2px; }
.nav-tags { display:flex; gap:5px; }
.nav-tag { font-size:11px; color:rgba(255,255,255,0.9)!important; font-weight:500; padding:5px 12px; border-radius:18px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.15); }

/* 左侧边栏 */
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

/* ★ 关键：页面面板显示/隐藏 — 通过 data-page 属性选择 */
.page-panel { background:#f1f5f9!important; padding:20px 24px!important; min-height:calc(100vh - 62px); border-radius:0!important; }
.page-panel[data-init-hidden] { display:none!important; }

/* 内容卡片 */
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
"""

# ═══════════════════════════════════════════════════════════════
# JS — 纯 DOM 操作切换页面面板
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# HTML 片段
# ═══════════════════════════════════════════════════════════════

TOP_HTML = """<div class="top-nav"><div class="top-nav-inner">
  <div class="brand"><div class="brand-icon">🧪</div><div>
    <div class="brand-title">ML-Lab</div>
    <div class="brand-sub">机器学习可视化实验平台 · 江苏工程职业技术学院</div>
  </div></div>
  <div class="nav-tags">
    <span class="nav-tag">看见算法</span><span class="nav-tag">动手调参</span>
    <span class="nav-tag">AI助学</span><span class="nav-tag">实证评估</span>
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
# 界面
# ═══════════════════════════════════════════════════════════════

def create_app():
    with gr.Blocks(title="ML-Lab") as app:

        gr.HTML(TOP_HTML)

        with gr.Row(equal_height=False):

            # ═══ 左侧边栏 ═══
            with gr.Column(scale=0, min_width=230, elem_classes="sidebar-panel"):
                gr.HTML('<div class="sidebar-header"><div class="sidebar-title">功能导航</div></div>')
                nav = gr.Radio(
                    choices=["📊 数据工作台", "🔬 算法实验室", "📈 模型评估", "🤖 AI助教"],
                    value="📊 数据工作台", label="", container=False,
                    elem_classes="sidebar-nav",
                )
                gr.HTML('<div style="padding:14px 0 4px;"><div class="sidebar-title" style="margin:0 11px 8px;">当前状态</div></div>')
                status_ds = gr.HTML('<div class="status-card"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">数据集</span><span class="status-value" id="sv-ds">未加载</span></div></div>')
                status_md = gr.HTML('<div class="status-card" style="margin-top:6px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">模型</span><span class="status-value" id="sv-md">未训练</span></div></div>')

            # ═══ 页面1: 数据工作台（始终渲染，通过 data-page 控制） ═══
            with gr.Column(elem_classes="page-panel", elem_id="page-data", visible=True) as page_data:
                gr.HTML(_card("📊","数据工作台","选择数据集、探索数据分布、执行预处理",
                    ETHICS+'<div class="step-title">步骤 1 · 加载数据集</div>'))
                with gr.Row():
                    ds_dd = gr.Dropdown(choices=list(get_builtin_datasets().keys()),
                                       value="鸢尾花 (Iris)", label="数据集")
                    ts_sl = gr.Slider(0.1,0.5,0.3,0.05,label="测试集比例")
                    load_btn = gr.Button("加载数据集", variant="primary")
                data_img = gr.Image(label="数据分布（PCA降维可视化）", height=360)
                data_info = gr.Textbox(label="数据信息", lines=3)
                data_summary = gr.Dataframe(label="数据摘要")
                gr.HTML('<div class="step-title">步骤 2 · 数据预处理</div>')
                with gr.Row():
                    method_dd = gr.Dropdown(choices=["mean","median","most_frequent","constant"],
                        value="mean", label="缺失值填充策略")
                    feat_sl = gr.Slider(0,50,0,1,label="展示特征索引")
                    pp_btn = gr.Button("执行预处理并对比", variant="secondary")
                pp_img = gr.Image(label="预处理前后对比", height=320)
                pp_info = gr.Textbox(label="预处理结果", lines=4)

            # ═══ 页面2: 算法实验室 ═══
            with gr.Column(elem_classes="page-panel", elem_id="page-algo", visible=True) as page_algo:
                gr.HTML(_card("🔬","算法实验室","选择算法、调整超参数、观察训练过程",
                    '<div class="step-title">选择算法与参数</div>'))
                with gr.Row():
                    algo_dd = gr.Dropdown(choices=get_algorithm_list(),value="决策树",label="算法")
                    with gr.Column():
                        lr_sl = gr.Slider(0.001,0.5,0.01,0.001,label="learning_rate",visible=False)
                        ni_sl = gr.Slider(10,500,100,10,label="n_iterations",visible=False)
                        C_sl  = gr.Slider(0.01,100,1.0,0.1,label="C")
                        md_sl = gr.Slider(1,15,3,1,label="max_depth")
                        mi_sl = gr.Slider(50,500,200,50,label="max_iter")
                        hid   = gr.Textbox("64,32",label="hidden_layer_sizes",visible=False)
                        al_sl = gr.Slider(1e-5,0.01,0.0001,1e-5,label="alpha",visible=False)

                def _show_params(a):
                    p = get_algorithm_params(a)
                    v = {k:False for k in ["lr","ni","C","md","mi","hid","al"]}
                    m = {"learning_rate":"lr","n_iterations":"ni","C":"C","max_depth":"md",
                         "max_iter":"mi","hidden_layer_sizes":"hid","alpha":"al"}
                    if p:
                        for x in p:
                            if x in m: v[m[x]]=True
                    return [gr.update(visible=v[k]) for k in ["lr","ni","C","md","mi","hid","al"]]

                algo_dd.change(fn=_show_params,inputs=[algo_dd],outputs=[lr_sl,ni_sl,C_sl,md_sl,mi_sl,hid,al_sl])
                train_btn = gr.Button("开始训练", variant="primary", size="lg")
                gr.HTML('<div class="step-title">训练结果</div>')
                with gr.Row():
                    train_img = gr.Image(label="训练过程", height=340)
                    db_img    = gr.Image(label="决策边界", height=340)
                result_txt = gr.Textbox(label="评估结果", lines=10)
                with gr.Row():
                    cm_img  = gr.Image(label="混淆矩阵", height=340)
                    roc_img = gr.Image(label="ROC曲线", height=340)

            # ═══ 页面3: 模型评估 ═══
            with gr.Column(elem_classes="page-panel", elem_id="page-eval", visible=True) as page_eval:
                gr.HTML(_card("📈","模型评估","对已训练模型进行全面的测试集评估",""))
                eval_btn = gr.Button("开始评估", variant="primary")
                eval_rpt = gr.Textbox(label="评估报告", lines=14)
                with gr.Row():
                    eval_cm  = gr.Image(label="混淆矩阵 / 回归结果", height=360)
                    eval_roc = gr.Image(label="ROC曲线", height=360)

            # ═══ 页面4: AI助教 ═══
            with gr.Column(elem_classes="page-panel", elem_id="page-ai", visible=True) as page_ai:
                gr.HTML(_card("🤖","AI助教","基于通义千问大模型，随时解答机器学习疑问",
                    '<p style="color:#64748b;font-size:13px;">支持算法原理讲解、代码解释、错误排查等</p>'))
                chatbot = gr.Chatbot(label="对话区域", height=380)
                with gr.Row():
                    msg_in  = gr.Textbox(label="输入问题", placeholder="例如：什么是梯度下降？", scale=5)
                    send_b  = gr.Button("发送", variant="primary", scale=1)
                    clear_b = gr.Button("清空", scale=0)
                gr.HTML('<div class="step-title">常见问题</div>')
                with gr.Row():
                    preset_btns = [gr.Button(q, size="sm") for q in PRESET_QUESTIONS[:6]]



        # 动态CSS控制面板显示
        current_page = gr.State(0)
        dynamic_css = gr.HTML(value="<style>#page-algo,#page-eval,#page-ai{display:none!important;}</style>", every=0.5)

        def on_nav_change(page_name):
            m = {"📊 数据工作台":0, "🔬 算法实验室":1, "📈 模型评估":2, "🤖 AI助教":3}
            idx = m.get(page_name, 0)
            pages = ['page-data', 'page-algo', 'page-eval', 'page-ai']
            hidden = [p for i, p in enumerate(pages) if i != idx]
            css = "<style>" + ",".join(['#' + p for p in hidden]) + "{display:none!important;}" + "#" + pages[idx] + "{display:flex!important;}</style>"
            return idx, css

        nav.change(fn=on_nav_change, inputs=[nav], outputs=[current_page, dynamic_css])

        gr.HTML('<div class="footer-bar">ML-Lab v1.0 · Python · Scikit-learn · Gradio · 通义千问 · 江苏工程职业技术学院 · MIT License</div>')

        # ── 事件绑定（功能回调） ──
        load_btn.click(fn=on_load_data, inputs=[ds_dd,ts_sl], outputs=[data_img,data_info,data_summary,status_ds])
        pp_btn.click(fn=on_preprocess, inputs=[method_dd,feat_sl], outputs=[pp_img,pp_info])
        train_btn.click(fn=on_train, inputs=[algo_dd,lr_sl,ni_sl,C_sl,md_sl,mi_sl,hid,al_sl],
                        outputs=[train_img,db_img,result_txt,cm_img,roc_img,status_md])
        eval_btn.click(fn=on_evaluate, outputs=[eval_rpt,eval_cm,eval_roc])
        send_b.click(fn=on_chat, inputs=[msg_in,chatbot], outputs=[msg_in,chatbot])
        msg_in.submit(fn=on_chat, inputs=[msg_in,chatbot], outputs=[msg_in,chatbot])
        clear_b.click(fn=lambda: ("", reset_conversation(), []), outputs=[msg_in, chatbot, chatbot])
        for btn, q in zip(preset_btns, PRESET_QUESTIONS[:6]):
            btn.click(fn=on_preset, inputs=[gr.Textbox(value=q, visible=False)], outputs=[msg_in, chatbot])

    return app

# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()
    if args.api_key:
        from ml_lab.llm_assistant import set_api_key; set_api_key(args.api_key)
    app = create_app()
    print("="*55+"\n  ML-Lab: 机器学习可视化实验平台\n"+f"  地址: http://{args.host}:{args.port}\n"+"="*55)
    app.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True, css=APP_CSS)

if __name__ == "__main__":
    main()