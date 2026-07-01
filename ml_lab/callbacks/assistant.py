# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.4 — AI助教/知识图谱回调函数
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




def on_ai_test_connection(base_url, model, api_key):
    """测试 LLM 连接（使用UI输入值）"""
    try:
        import requests
        base_url = base_url.strip().rstrip("/")
        api_key = api_key.strip() if api_key else ""
        if not base_url or not api_key:
            return '<div style="font-size:11px;color:#ef4444;">&#10060; 请先填写 API Base URL 和 API Key</div>'
        if not model:
            model = "qwen-turbo"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model.strip(),
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5
        }
        r = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            return '<div style="font-size:11px;color:#059669;">&#9989; 连接成功！模型响应正常</div>'
        else:
            return f'<div style="font-size:11px;color:#ef4444;">&#10060; 连接失败: HTTP {r.status_code}</div>'
    except requests.exceptions.Timeout:
        return '<div style="font-size:11px;color:#ef4444;">&#10060; 连接超时（15s），请检查 Base URL 是否正确</div>'
    except requests.exceptions.ConnectionError:
        return '<div style="font-size:11px;color:#ef4444;">&#10060; 无法连接服务器，请检查 Base URL 和网络</div>'
    except Exception as e:
        return f'<div style="font-size:11px;color:#ef4444;">&#10060; 连接失败: {str(e)[:80]}</div>'



def on_ai_save_config(base_url, model, api_key):
    """保存 LLM 配置到 .env 文件"""
    try:
        from ml_lab.llm_assistant import _config, _CONFIG_PATHS
        _config["base_url"] = base_url
        _config["model"] = model
        if api_key:
            _config["api_key"] = api_key
        # 写入 .env 文件
        env_path = _CONFIG_PATHS[0]
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"LLM_BASE_URL={base_url}\n")
            f.write(f"LLM_MODEL={model}\n")
            f.write(f"LLM_API_KEY={api_key}\n")
        # 设置环境变量
        os.environ["LLM_BASE_URL"] = base_url
        os.environ["LLM_MODEL"] = model
        if api_key:
            os.environ["LLM_API_KEY"] = api_key
        return '<div style="font-size:11px;color:#059669;">\u2705 配置已保存至 .env 文件</div>'
    except Exception as e:
        return f'<div style="font-size:11px;color:#ef4444;">\u274c 保存失败: {str(e)[:80]}</div>'


def on_ai_context_chat(msg, history):
    """上下文增强的对话"""
    if not msg.strip():
        return "", history

    # 构建包含上下文的系统提示
    ctx_parts = []
    ds = _g.get("dataset_name", "未加载")
    ctx_parts.append(f"当前数据集: {ds}")

    algo = _g.get("last_algo_name", None)
    if algo:
        task = _g.get("last_task_type", "未知")
        ctx_parts.append(f"最近训练的模型: {algo} ({task})")

    n = _g.get("X_train", None)
    if n is not None:
        ctx_parts.append(f"训练样本数: {_g['X_train'].shape[0]}")
        ctx_parts.append(f"特征数: {_g['X_train'].shape[1]}")

    context_str = " | ".join(ctx_parts) if ctx_parts else ""

    # 调用 LLM
    from ml_lab.llm_assistant import ask
    try:
        # 将上下文拼接到消息前
        enhanced_msg = f"[实验上下文: {context_str}]\n\n{msg}" if context_str else msg
        resp = ask(enhanced_msg)
    except Exception as e:
        resp = f"抱歉，AI 助教暂时无法回答。错误: {str(e)[:100]}\n\n请检查「模型设置」中的 API 配置是否正确。"

    history = history or []
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": resp})
    return "", history


