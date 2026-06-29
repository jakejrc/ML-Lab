# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.2 — 关联规则回调函数
从 callbacks.py 拆分而来
"""
import sys, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import gradio as gr

from ml_lab.state import _g, _sync
from ml_lab.association_rules import (
    AprioriModel, FPGrowthModel, evaluate_association as eval_assoc
)
from ml_lab.association_viz import (
    plot_top_frequent_items, plot_rules_scatter,
    plot_item_length_distribution, plot_rules_heatmap
)
from ml_lab.fptree_viz import plot_fptree
from ml_lab.visualization import fig_to_image
from ml_lab.ui_styles import APP_CSS, _card, ETHICS

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

    if _g.get("last_assoc_model") is None:

        return "⚠️ 请先执行关联规则挖掘，再复制代码", "📋 生成并复制代码", ""



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

    return code, "✅ 已复制剪贴板", code
