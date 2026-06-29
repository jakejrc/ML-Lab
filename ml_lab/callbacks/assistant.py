# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.2 — AI助教/知识图谱回调函数
从 callbacks.py 拆分而来
"""
import sys, os
from datetime import datetime
import gradio as gr

from ml_lab.state import _g, _sync
from ml_lab.llm_assistant import ask, reset_conversation, PRESET_QUESTIONS
from ml_lab.learning_progress import record_activity, mark_stage_completed, get_progress_summary, STAGE_MODULE_MAP
from ml_lab.ui_styles import APP_CSS, _card, ETHICS

def on_kg_render():
    """渲染知识图谱3D交互式可视化"""
    try:
        kg_html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "kg_3d.html")
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

def on_learning_path_click(evt: gr.EventData):

    """学习路径步骤点击时跳转到对应页面"""

    return gr.skip()  # 由 JS 处理跳转
