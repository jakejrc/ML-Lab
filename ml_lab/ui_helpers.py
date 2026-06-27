# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.1 — UI 辅助函数
"""

import gradio as gr
from ml_lab.algorithms import get_algorithm_params

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





