"""ML-Lab 页面 UI 构建模块
在活跃 Gradio Blocks 上下文中创建所有 UI 组件
"""
import gradio as gr
from ml_lab.ui_styles import _card, ETHICS
from ml_lab.html_templates import LEARNING_PATH_HTML
from ml_lab.sandbox_templates import SANDBOX_TEMPLATES
SANDBOX_TEMPLATE = SANDBOX_TEMPLATES["默认（查看数据）"]
from ml_lab.algorithms import get_algorithm_list
from ml_lab.preprocessing import get_builtin_datasets
from ml_lab.llm_assistant import PRESET_QUESTIONS
from ml_lab.ui_helpers import _make_supervised_param_components, _bind_param_visibility


def build_pages():
    """构建所有页面 UI 组件，返回组件引用字典
    通过 locals() 捕获所有局部变量作为组件引用
    """

    # Hide all non-default page panels via CSS (prevents flash during Gradio re-render)
    gr.HTML("""<style>
    #page-learning, #page-data, #page-fe, #page-classify, #page-regress,
    #page-cluster, #page-assoc, #page-code, #page-ai { display: none !important; }
    #page-kg { display: flex !important; }
    </style>""")

    # ── Float AI Chat: Hidden Components ──

    float_input = gr.Textbox(visible=False, elem_id="float-chat-gradio-input")

    float_output = gr.HTML('<div id="float-chat-output"></div>', visible=False, elem_id="float-chat-output-wrap")

    float_trigger = gr.Button(visible=False, elem_id="float-chat-trigger")

    float_clear_trigger = gr.Button(visible=False, elem_id="float-chat-clear-trigger")





    with gr.Row(equal_height=False):





        # ═══ 左侧边栏 ═══



        with gr.Column(scale=0, min_width=230, elem_classes="sidebar-panel"):

            gr.HTML('<div style="padding:14px 0 4px;"><div class="sidebar-title" style="margin:0 11px 8px;">当前状态</div></div>')

            status_ds = gr.HTML('<div class="status-card"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">数据集</span><span class="status-value" id="sv-ds">未加载</span></div></div>')

            status_md = gr.HTML('<div class="status-card" style="margin-top:6px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="status-label">模型</span><span class="status-value" id="sv-md">未训练</span></div></div>')

            gr.HTML('<div class="sidebar-header"><div class="sidebar-title">功能导航</div></div>')

            nav = gr.Radio(

                choices=[
                    "🧠 知识图谱",
                    "📖 学习路径",
                    "📊 数据工作台",
                    "⚙️ 特征工程",
                    "🏷️ 分类实验",
                    "📈 回归实验",
                    "🔵 聚类实验",
                    "🔗 关联规则",
                    "💻 代码沙箱",
                    "🤖 AI助教"
                ],

                value="🧠 知识图谱", label="", container=False,

                elem_classes="sidebar-nav",

            )

# ═══ 页面0: 学习路径 ═══



        # ═══ 页面1: 知识图谱 ═══
        with gr.Column(elem_classes="page-panel", elem_id="page-kg", visible=True) as page_kg:
            gr.HTML(_card("🧠","知识图谱","机器学习核心概念关系网络 · 交互式探索与学习",
                '<span class="tab-badge tab-badge-tool">62个概念 · 96条关系 · 9大类别</span>'))
            gr.HTML('<div class="step-title">交互式知识图谱</div>')
            gr.HTML('<div style="color:#94a3b8;font-size:12px;margin:0 0 10px 0;">'
                    '拖拽节点浏览 · 滚轮缩放 · 悬停查看概念详情 · 点击拖拽可调整布局</div>')
            kg_html = gr.HTML(label="知识图谱可视化")
            gr.HTML('<div class="ethics-bar" style="margin-top:12px;">'
                    '<strong>💡 使用说明：</strong>'
                    '知识图谱涵盖机器学习核心概念，按类别着色。'
                    '节点大小表示关联度（关联越多越大）。'
                    '悬停查看概念描述，点击拖拽调整布局。</div>')

        # ═══ 页面2: 学习路径 ═══
        with gr.Column(elem_classes="page-panel", elem_id="page-learning", visible=True) as page_learning:



            gr.HTML(_card("📖","学习路径","引导式机器学习实验流程，从数据探索到报告撰写的完整学习路线",



                '<span class="tab-badge tab-badge-tool">教学引导</span>'



                '<span style="color:#64748b;font-size:12px;">7个实验阶段 · 含思考题与操作指引</span>' ))



            gr.HTML(LEARNING_PATH_HTML)





        # ═══ 页面1: 数据工作台 ═══



        with gr.Column(elem_classes="page-panel", elem_id="page-data", visible=True) as page_data:



            gr.HTML(_card("📊","数据工作台","选择数据集、探索数据分布、执行预处理",



                ETHICS+'<div class="step-title">步骤 1 · 加载数据集</div>'))



            with gr.Tabs():

                with gr.Tab("📚 内置数据集", id="builtin") as tab_builtin:

                    with gr.Row():

                        all_ds = list(get_builtin_datasets().keys())

                        ds_dd = gr.Dropdown(choices=all_ds, value="鸢尾花 (Iris)", label="数据集")

                        ts_sl = gr.Slider(0.1, 0.5, 0.3, 0.05, label="测试集比例")

                        load_btn = gr.Button("加载数据集", variant="primary")



                with gr.Tab("📤 自定义上传", id="custom") as tab_custom:

                    gr.HTML('<div style="font-size:12px;color:#94a3b8;margin:0 0 8px 0;">'

                            '上传 CSV 或 Excel 文件，自动识别数值特征和目标列</div>')

                    upload_file = gr.File(label="上传数据文件", file_types=[".csv", ".xls", ".xlsx"],

                                         type="filepath")

                    with gr.Row():

                        upload_path = gr.Textbox(label="或输入本地文件路径",

                                                 placeholder="D:\\path\\to\\data.csv")

                        custom_target_col = gr.Textbox(value="-1",

                                                       label="目标列（列名或索引，-1=最后一列）",

                                                       placeholder="例: -1 或 target")

                    with gr.Row():

                        custom_task_type = gr.Dropdown(

                            choices=["auto", "classification", "regression"],

                            value="auto", label="任务类型（auto=自动检测）")

                        custom_ts_sl = gr.Slider(0.1, 0.5, 0.3, 0.05, label="测试集比例")

                    custom_load_btn = gr.Button("加载自定义数据集", variant="primary")



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



                +'<div class="step-title">选择算法与参数</div>'

                +'</div>'))



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
            cls_code_clipboard = gr.Textbox(visible=True, elem_id="cls-code-clipboard", label="", show_label=False)



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

            compare_multi_img = gr.Image(label="多指标详细对比 (Acc/Precision/Recall/F1)", height=340)

            compare_result = gr.Textbox(label="对比结果", lines=8)



            gr.HTML('<div class="step-title">实验报告</div>')

            with gr.Row():

                export_btn_cls = gr.Button("📄 导出 HTML 报告", variant="secondary")

                export_file_cls = gr.File(label="下载报告", visible=True)



        # ═══ 页面3: 回归实验 ═══



        with gr.Column(elem_classes="page-panel", elem_id="page-regress", visible=True) as page_regress:



            gr.HTML(_card("📈","回归实验","监督学习 · 回归算法训练与评估",



                '<span class="tab-badge tab-badge-supervised">监督学习</span>'



                '<span style="color:#64748b;font-size:12px;">线性回归 · Ridge · Lasso · 多项式回归 · SVR · 决策树回归</span>'



                +'<div class="step-title">选择算法与参数</div>'

                +'</div>'))



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
            reg_code_clipboard = gr.Textbox(visible=True, elem_id="reg-code-clipboard", label="", show_label=False)



            gr.HTML('<div class="step-title">实验报告</div>')

            with gr.Row():

                export_btn_reg = gr.Button("📄 导出 HTML 报告", variant="secondary")

                export_file_reg = gr.File(label="下载报告", visible=True)



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



                +'<div class="step-title">选择算法与参数</div>'

                +'</div>'))



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
            uns_code_clipboard = gr.Textbox(visible=True, elem_id="uns-code-clipboard", label="", show_label=False)



            gr.HTML('<div class="step-title">实验报告</div>')

            with gr.Row():

                export_btn_uns = gr.Button("📄 导出 HTML 报告", variant="secondary")

                export_file_uns = gr.File(label="下载报告", visible=True)



        # ═══ 页面5: 关联规则挖掘 ═══



        with gr.Column(elem_classes="page-panel", elem_id="page-assoc", visible=True) as page_assoc:



            gr.HTML(_card("🔗","关联规则挖掘","无监督学习 · Apriori / FP-Growth 关联规则算法",



                '<span class="tab-badge tab-badge-unsupervised">关联规则</span>'



                '<span style="color:#64748b;font-size:12px;">Apriori · FP-Growth · 频繁项集 · 关联规则</span>'



                +'<div class="step-title">选择算法与参数</div>'

                +'<div style="color:#ef4444;font-size:12px;font-weight:600;margin:6px 0 10px 0;padding:6px 10px;background:rgba(239,68,68,0.08);border-radius:6px;border-left:3px solid #ef4444;">'

                +'⚠️ 请先在数据工作台加载 <b>超市购物篮 (Market Basket)</b> 数据集，再执行挖掘！</div>'))



            with gr.Row():

                with gr.Column(scale=1):

                    assoc_algo_dd = gr.Dropdown(

                        choices=["Apriori", "FP-Growth"],

                        value="Apriori",

                        label="算法"

                    )

                with gr.Column(scale=2):

                    assoc_min_sup = gr.Slider(0.01, 0.5, 0.1, 0.01, label="最小支持度 (min_support)")

                    assoc_min_conf = gr.Slider(0.1, 1.0, 0.5, 0.05, label="最小置信度 (min_confidence)")

                    assoc_min_lift = gr.Slider(0.5, 5.0, 1.0, 0.1, label="最小提升度 (min_lift)")



            with gr.Row():

                with gr.Column(scale=1):

                    assoc_max_len = gr.Slider(2, 6, 4, 1, label="最大项集长度 (max_length)")

                with gr.Column(scale=1):

                    assoc_top_k = gr.Slider(5, 30, 15, 1, label="Top-K 显示数量")



            # Apriori 独有说明

            assoc_apriori_desc = gr.HTML(

                value='<div style="color:#3b82f6;font-size:11px;padding:6px 10px;background:rgba(59,130,246,0.06);border-radius:6px;margin:4px 0;">'

                      '📊 <b>Apriori</b>：逐层搜索候选集，通过连接和剪枝生成频繁项集。适合中小规模数据集，支持度阈值越低候选集越多。</div>',

                visible=True

            )

            # FP-Growth 独有说明

            assoc_fpg_desc = gr.HTML(

                value='<div style="color:#10b981;font-size:11px;padding:6px 10px;background:rgba(16,185,129,0.06);border-radius:6px;margin:4px 0;">'

                      '🌳 <b>FP-Growth</b>：构建FP树压缩事务数据，无需生成候选集。适合大规模数据集，通常比 Apriori 快 2-10 倍。</div>',

                visible=False

            )

            # 离散化参数（仅购物篮以外数据时显示，默认隐藏）

            with gr.Row(visible=False) as assoc_disc_row:

                with gr.Column(scale=1):

                    assoc_n_bins = gr.Slider(2, 5, 3, 1, label="离散化分箱数 (n_bins)")

                with gr.Column(scale=1):

                    assoc_disc_method = gr.Dropdown(

                        choices=["quantile", "equal_width"],

                        value="quantile",

                        label="离散化方法"

                    )



            assoc_run_btn = gr.Button("开始挖掘", variant="primary", size="lg")



            gr.HTML('<div class="step-title">挖掘结果</div>')



            with gr.Row():

                assoc_img1 = gr.Image(label="频繁项集支持度", height=340)

                assoc_img2 = gr.Image(label="规则散点图 (置信度 vs 提升度)", height=340)



            with gr.Row():

                assoc_img3 = gr.Image(label="项集长度分布", height=300)

                assoc_img4 = gr.Image(label="FP-Growth 树结构 / 规则热力图", height=340)



            assoc_result = gr.HTML(label="评估结果",

                value='<div style="color:#a0aec0;padding:20px;text-align:center;">挖掘后显示频繁项集和关联规则表格</div>')



            gr.HTML('<div class="step-title">代码复现</div>')



            with gr.Row():

                assoc_copy_btn = gr.Button("📋 生成并复制代码", variant="secondary")

            assoc_code_display = gr.Code(label="Python 代码", language="python", lines=28, interactive=True)
            assoc_code_clipboard = gr.Textbox(visible=True, elem_id="assoc-code-clipboard", label="", show_label=False)



            gr.HTML('<div class="step-title">实验报告</div>')

            with gr.Row():

                export_btn_assoc = gr.Button("📄 导出 HTML 报告", variant="secondary")

                export_file_assoc = gr.File(label="下载报告", visible=True)



        # ═══ 页面6: 代码沙箱 ═══



        with gr.Column(elem_classes="page-panel", elem_id="page-code", visible=True) as page_code:



            gr.HTML(_card("💻","代码沙箱","在线编写和运行 Python 代码",



                '<span class="tab-badge tab-badge-tool">开发工具</span>'



                '<span style="color:#64748b;font-size:12px;">已预导入 numpy, pandas, sklearn, matplotlib · 支持直接上传数据文件</span>'))



            with gr.Row():

                sandbox_upload = gr.File(label="📁 上传数据文件 (CSV/Excel)", file_types=[".csv", ".xls", ".xlsx"],

                                         type="filepath", scale=3)

                sandbox_load_btn = gr.Button("加载数据", variant="secondary", size="sm", scale=1)

                clear_code_btn = gr.Button("清空代码", size="sm")

                download_code_btn = gr.Button("💾 下载 .py", size="sm")



            gr.HTML('''<div id="code-hint-bar" style="display:flex;flex-wrap:wrap;gap:5px;margin:6px 0 10px;align-items:center;">



<span style="font-size:11px;color:#64748b;font-weight:600;margin-right:2px;">常用代码片段：</span>



</div>''')



            code_editor = gr.Code(label="Python 代码", language="python", lines=30,



                                   value=SANDBOX_TEMPLATE, elem_classes="code-editor",



                                   elem_id="sandbox-code-editor")



            with gr.Row():



                template_dd = gr.Dropdown(choices=list(SANDBOX_TEMPLATES.keys()),

                                         value=list(SANDBOX_TEMPLATES.keys())[0],

                                         label="代码模板", show_label=True, scale=3)



                run_btn = gr.Button("▶ 运行代码", variant="primary", size="lg", scale=1)



                max_btn = gr.HTML('''<div id="sandbox-maximize-btn" class="maximize-btn" title="最大化/还原编辑器">⛶ 全屏编辑</div>''')



            code_output = gr.Textbox(label="运行输出", lines=10, elem_classes="code-output",

                                              interactive=False)

            code_plot = gr.Image(label="图形输出", elem_classes="code-plot")

            code_download = gr.File(label="下载文件", interactive=False, visible=True)





            gr.HTML('<div class="ethics-bar"><strong>💡 使用提示：</strong>'



                '可直接在上方上传 CSV/Excel 文件，点击「加载数据」后即可使用 <code>X</code>, <code>y</code>, <code>X_train</code>, <code>y_train</code>, '



                '<code>X_test</code>, <code>y_test</code>, <code>feature_names</code>, '



                '<code>target_names</code> 变量。目标列默认为最后一列。'



                '支持 matplotlib 画图，图形将自动显示在右侧输出区。</div>')





        # ═══ 页面7: AI助教 ═══



        with gr.Column(elem_classes="page-panel", elem_id="page-ai", visible=True) as page_ai:



            gr.HTML(_card("🤖","AI助教","基于大模型API，随时解答机器学习疑问",



                '<span class="tab-badge tab-badge-tool">AI 辅助</span>'



                '<p style="color:#64748b;font-size:13px;">模型配置请编辑项目根目录 .env 文件</p>'))



            chatbot = gr.Chatbot(label="对话区域", height=380)



            with gr.Row():



                msg_in  = gr.Textbox(label="输入问题", placeholder="例如：K-Means 的原理是什么？", scale=5)



                send_b  = gr.Button("发送", variant="primary", scale=1)



                clear_b = gr.Button("清空", scale=0)



            gr.HTML('<div class="step-title">常见问题</div>')



            with gr.Row():



                preset_btns = [gr.Button(q, size="sm") for q in PRESET_QUESTIONS[:6]]



    PAGE_MAP = {
        "🧠 知识图谱":  ("page-kg",      0),
        "📖 学习路径":  ("page-learning", 1),
        "📊 数据工作台": ("page-data",     2),
        "⚙️ 特征工程":  ("page-fe",       3),
        "🏷️ 分类实验":  ("page-classify", 4),
        "📈 回归实验":  ("page-regress", 5),
        "🔵 聚类实验":  ("page-cluster", 6),
        "🔗 关联规则":  ("page-assoc",   7),
        "💻 代码沙箱":  ("page-code",    8),
        "🤖 AI助教":    ("page-ai",      9),
    }



    PAGE_ORDER = ["page-kg", "page-learning", "page-data", "page-fe", "page-classify", "page-regress", "page-cluster", "page-assoc", "page-code", "page-ai"]





    def on_nav(page_name):
        target_id, _ = PAGE_MAP.get(page_name, ("page-kg", 0))
        print(f"[NAV] click='{page_name}' -> target='{target_id}'")
        # 切页完全交给客户端 JS，on_nav 仅作为 Radio 事件占位（避免 Radio 被 disabled）
        return [gr.skip() for _ in PAGE_ORDER]





    dynamic_css = gr.HTML(value="", visible=False)# 页面切换改用客户端 JS（launch js= 中已注入）



    gr.HTML('<div class="footer-bar">ML-Lab v3.8.1 · Python · Scikit-learn · Gradio · 江苏工程职业技术学院 南通市人工智能新质技术重点实验室 · MIT License</div>')



    # 学习路径步骤点击跳转






    # 捕获所有局部变量作为组件引用
    _comps = {}
    for _k, _v in list(locals().items()):
        if not _k.startswith("_") and _k not in ("gr",):
            _comps[_k] = _v
    return _comps
