"""ML-Lab 事件绑定模块"""
import gradio as gr
from ml_lab.callbacks import (
    on_load_data, on_preprocess,
    on_train_classification, on_compare_models, on_train_regression,
    on_train_clustering,
    on_copy_classification_code, on_copy_regression_code,
    on_copy_clustering_code, on_copy_association_code,
    on_run_association, on_export_report,
    on_load_custom_data, on_download_code,
    on_sandbox_load_data, on_run_code, on_load_template,
    on_fe_scaling, on_fe_feature_selection, on_fe_pca,
    on_fe_construct, on_fe_discretize, on_fe_correlation,
    on_kg_render, on_chat, on_preset,
    on_learning_path_click,
)


def bind_events(comps):
    """绑定所有事件处理函数
    通过 globals().update(comps) 注入组件引用
    """
    globals().update(comps)

    # ── 事件绑定 ──

    # 数据工作台

    load_btn.click(fn=on_load_data, inputs=[ds_dd, ts_sl], outputs=[data_img, data_info, data_summary, status_ds])


    # nav.change 仅作为 Radio 事件占位（避免 Radio 被 disabled）
    # 实际切页由客户端 JS 完成，不绑定 outputs 避免面板闪烁
    nav.change(fn=lambda x: None, inputs=[nav], outputs=[])
    custom_load_btn.click(fn=on_load_custom_data,

        inputs=[upload_file, upload_path, custom_target_col, custom_task_type, custom_ts_sl],

        outputs=[data_img, data_info, data_summary, status_ds])

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

    # 导出报告

    export_btn_cls.click(fn=on_export_report, inputs=[result_txt_cls], outputs=[export_file_cls])


    compare_btn.click(fn=on_compare_models,

        inputs=[compare_algos],

        outputs=[compare_img, compare_result])


    # 回归实验

    export_btn_reg.click(fn=on_export_report, inputs=[result_txt_reg], outputs=[export_file_reg])


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

    export_btn_uns.click(fn=on_export_report, inputs=[result_txt_uns], outputs=[export_file_uns])


    train_btn_uns.click(fn=on_train_clustering,

        inputs=[algo_dd_uns, n_cl, max_iter_cl, eps_sl, ms_sl, linkage_dd, affinity_dd, init_dd],

        outputs=[cluster_img1, cluster_img2, cluster_img3, result_txt_uns, status_md])


    # 关联规则算法切换回调

    def on_assoc_algo_change(algo_name):

        algo = algo_name if algo_name else "Apriori"

        if algo == "Apriori":

            return [

                gr.update(visible=True),   # assoc_apriori_desc

                gr.update(visible=False),  # assoc_fpg_desc

            ]

        else:  # FP-Growth

            return [

                gr.update(visible=False),  # assoc_apriori_desc

                gr.update(visible=True),   # assoc_fpg_desc

            ]


    # 关联规则算法切换

    assoc_algo_dd.change(

        fn=on_assoc_algo_change,

        inputs=[assoc_algo_dd],

        outputs=[assoc_apriori_desc, assoc_fpg_desc]

    )


    # 关联规则挖掘

    export_btn_assoc.click(fn=on_export_report, inputs=[assoc_result], outputs=[export_file_assoc])


    assoc_run_btn.click(fn=on_run_association,

        inputs=[assoc_algo_dd, assoc_min_sup, assoc_min_conf, assoc_min_lift,

                assoc_max_len, assoc_disc_method, assoc_n_bins, assoc_top_k],

        outputs=[assoc_img1, assoc_img2, assoc_img3, assoc_img4, assoc_result, status_md])


    # 知识图谱渲染
    kg_html.value = on_kg_render()

    # 代码沙箱

    sandbox_load_btn.click(fn=on_sandbox_load_data, inputs=[sandbox_upload], outputs=[code_output, code_plot])

    run_btn.click(fn=on_run_code, inputs=[code_editor], outputs=[code_output, code_plot])

    template_dd.change(fn=on_load_template, inputs=[template_dd], outputs=[code_editor])

    clear_code_btn.click(fn=lambda: ("",), outputs=[code_editor])

    download_code_btn.click(fn=on_download_code, inputs=[code_editor], outputs=[code_download])


    # AI助教

    msg_in.submit(fn=on_chat, inputs=[msg_in, chatbot], outputs=[msg_in, chatbot])

    send_b.click(fn=on_chat, inputs=[msg_in, chatbot], outputs=[msg_in, chatbot])

    clear_b.click(fn=lambda: ("", []), outputs=[msg_in, chatbot])

    for pb in preset_btns:

        pb.click(fn=on_preset, inputs=[pb], outputs=[msg_in, chatbot])

    # 一键复制代码 - 分类实验

    copy_code_btn_cls.click(fn=on_copy_classification_code,

        inputs=[algo_dd_cls, lr_c, ni_c, C_c, md_c, mi_c, hid_c, al_c],

        outputs=[code_display_cls, copy_code_btn_cls, cls_code_clipboard],
    )










    copy_code_btn_reg.click(fn=on_copy_regression_code,

        inputs=[algo_dd_reg, lr_r, ni_r, C_r, md_r, mi_r, hid_r, al_r, deg_r, eps_r, kern_r, crit_r],

        outputs=[code_display_reg, copy_code_btn_reg, reg_code_clipboard],
    )

    # 一键复制代码 - 聚类实验

    copy_code_btn_uns.click(fn=on_copy_clustering_code,

        inputs=[algo_dd_uns, n_cl, max_iter_cl, eps_sl, ms_sl, linkage_dd, affinity_dd, init_dd],

        outputs=[code_display_uns, copy_code_btn_uns, uns_code_clipboard],
    )

    # 一键复制代码 - 关联规则挖掘

    assoc_copy_btn.click(fn=on_copy_association_code,

        inputs=[assoc_algo_dd, assoc_min_sup, assoc_min_conf, assoc_min_lift,

                assoc_max_len, assoc_disc_method, assoc_n_bins],

        outputs=[assoc_code_display, assoc_copy_btn, assoc_code_clipboard],
    )

    # ── Float AI Chat: Callbacks ──

    _float_messages = []

    def on_float_chat(message):

        import requests as _req

        # 过滤空消息

        msg = (message or '').strip()

        if not msg:

            return "[错误：消息内容不能为空]"

        _float_messages.append(("user", msg))

        headers = {"Authorization": "Bearer fef2988ca571a100603a5ded229e1c96.TwiAhzLCUKsEDJdm", "Content-Type": "application/json"}

        api_msgs = [{"role": "system", "content": '你是ML-Lab机器学习实验平台的AI助教，用中文简洁回答。**重要限制**：你只能解答与机器学习、数据分析、Python编程、数学基础（线性代数/概率统计/微积分）相关的问题。如果用户问的是与课程无关的内容（如其他学科知识、时事新闻、个人建议等），请礼貌拒绝直接回复：抱歉，我只能解答本课程相关的机器学习、数据分析或编程问题。后面不要跟任何无关内容。'}]

        for role, text in _float_messages:

            t = (text or '').strip()

            if not t:

                continue

            api_role = role if role != "bot" else "assistant"

            api_msgs.append({"role": api_role, "content": t})

        try:

            resp = _req.post("https://open.bigmodel.cn/api/paas/v4/chat/completions", headers=headers, json={"model": "glm-4-flash", "messages": api_msgs, "max_tokens": 512}, timeout=30)

            resp.raise_for_status()

            reply = resp.json()["choices"][0]["message"]["content"]

        except Exception as e:

            # 尝试解析 API 返回的详细错误信息

            err_detail = str(e)[:80]

            try:

                err_body = e.response.json() if hasattr(e, 'response') and e.response is not None else None

                if err_body and 'error' in err_body:

                    err_detail = f"{err_body['error'].get('code','')}: {err_body['error'].get('message','')}"

            except:

                pass

            reply = f"[请求失败: {err_detail}]"

        # 后端 Markdown → HTML 转换（前端 textContent 不渲染 Markdown）

        import re as _md_re

        _md_reply = reply

        # 先转义 HTML 特殊字符，再替换 Markdown 语法

        _md_reply = _md_reply.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # 粗体 **text** → <strong>text</strong>

        _md_reply = _md_re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', _md_reply)

        _md_reply = _md_re.sub(r'__(.+?)__', r'<strong>\1</strong>', _md_reply)

        # 行内代码 `code` → <code>code</code>

        _md_reply = _md_re.sub(r'`([^`]+)`', r'<code style="background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:0.9em">\1</code>', _md_reply)

        # 换行 -> <br>

        _md_reply = _md_reply.replace('\n', '<br>')

        _float_messages.append(("bot", _md_reply))

        return _md_reply

    def on_float_clear():

        _float_messages.clear()

        return ""

    float_trigger.click(fn=on_float_chat, inputs=[float_input], outputs=[float_output])

    float_clear_trigger.click(fn=on_float_clear, inputs=[], outputs=[float_output])
