# -*- coding: utf-8 -*-
"""ML-Lab UI 样式与辅助函数。

包含 Gradio 前端 CSS 样式表、页面卡片组件和伦理提醒条。
"""

# ── 全局 CSS 样式表 ──
APP_CSS = """

footer { display: none !important; }

.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }

body { background: #f1f5f9 !important; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif !important; }

.top-nav { background: linear-gradient(135deg,#1e40af 0%,#2563eb 60%,#3b82f6 100%); box-shadow: 0 2px 8px rgba(0,0,0,0.15); position: relative; margin-bottom:0; padding-bottom:0; }
.top-nav-inner { width:100%; margin:0 auto; display:flex; align-items:center; justify-content:space-between; padding:20px 28px; box-sizing:border-box; }
.html-container { padding: 0 !important; }
.gradio-container .main { max-width: 100% !important; padding: 0 !important; }
.gradio-container .contain { max-width: 100% !important; }

.top-nav-inner { width:100%; margin:0 auto; display:flex; align-items:center; justify-content:space-between; padding:20px 28px; box-sizing:border-box; }

.brand { display:flex; align-items:center; gap:14px; }

.brand-icon img { width:100%; height:100%; object-fit:contain; }

.brand-icon { width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center; overflow:hidden; flex-shrink:0; }

.brand-title { color:#fff!important; font-size:24px; font-weight:700; text-shadow:none!important; -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale; }

.brand-sub { color:rgba(255,255,255,0.8)!important; font-size:11px; margin-top:2px; }

.nav-tags { display:flex; gap:5px; }

.nav-tag { font-size:11px; color:rgba(255,255,255,0.9)!important; font-weight:500; padding:5px 12px; border-radius:18px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.15); }

.sidebar-panel { background: linear-gradient(180deg,#1e3a8a 0%,#1e40af 40%,#2563eb 100%)!important; min-width:230px!important; max-width:230px!important; padding:0!important; box-shadow:4px 0 20px rgba(30,58,138,0.18)!important; border:none!important; overflow-y:auto!important; }

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

.page-panel { display:none!important; background:#f1f5f9!important; padding:20px 24px!important; min-height:calc(100vh - 62px); border-radius:0!important; }

.page-panel.ml-visible { display:flex!important; }

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

.footer-bar { text-align:center; padding:14px 20px; color:#475569; font-size:16px; border-top:1px solid #e2e8f0; background:#fff; font-weight:500; }

.gr-row { gap: 0!important; }

.code-editor { font-family: 'Consolas', 'Monaco', 'Courier New', monospace!important; font-size: 13px!important; line-height: 1.6!important; }

.code-editor textarea { min-height: 380px!important; }

.code-plot {
    border: 1px solid var(--border-color);
    border-radius: 12px;
    background: var(--card-bg);
    overflow: hidden;
}
.code-plot img {
    max-height: 520px;
    object-fit: contain;
}
.code-output { font-family: 'Consolas', 'Monaco', monospace!important; font-size: 12px!important; background: #1e293b!important; color: #e2e8f0!important; border-radius: 8px!important; padding: 16px!important; max-height: 400px!important; overflow-y: auto!important; }

.tab-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; margin-right:6px; }

.tab-badge-supervised { background:#dbeafe; color:#1d4ed8; }

.tab-badge-unsupervised { background:#dcfce7; color:#166534; }

.tab-badge-tool { background:#fef3c7; color:#92400e; }


#page-code.maximized { position: fixed!important; top: 0!important; left: 0!important; right: 0!important; bottom: 0!important; z-index: 99999!important; background: #f1f5f9!important; padding: 20px!important; overflow-y: auto!important; display: flex!important; flex-direction: column!important; }

.maximized .code-editor { min-height: 400px!important; flex: 1 1 auto!important; }

.maximized .code-plot {
    border: 1px solid var(--border-color);
    border-radius: 12px;
    background: var(--card-bg);
    overflow: hidden;
}
.code-plot img {
    max-height: 520px;
    object-fit: contain;
}
.code-output { min-height: 250px!important; flex: 0 0 auto!important; }

#page-code.maximized .code-editor textarea { min-height: calc(100vh - 260px)!important; height: calc(100vh - 260px)!important; font-size: 14px!important; line-height: 1.7!important; }

#page-code.maximized .code-output textarea { min-height: calc(100vh - 420px)!important; height: calc(100vh - 420px)!important; }

.maximize-btn { position: absolute; top: 8px; right: 8px; z-index: 10; background: #1e40af!important; color: #fff!important; border: none!important; border-radius: 6px!important; padding: 8px 16px!important; font-size: 13px!important; font-weight: 600!important; cursor: pointer!important; transition: background 0.2s!important; }

.maximize-btn:hover { background: #1d4ed8!important; }

.maximized .maximize-btn { position: fixed!important; top: 20px!important; right: 30px!important; z-index: 100001!important; }

.maximize-btn:hover { background: #1d4ed8!important; }

.snippet-btn { background: #fff!important; border: 1px solid #cbd5e1!important; border-radius: 6px!important; padding: 5px 12px!important; font-size: 11px!important; color: #334155!important; cursor: pointer!important; font-weight: 500!important; transition: all 0.15s ease!important; }

.snippet-btn:hover { background: #eff6ff!important; border-color: #3b82f6!important; color: #1e40af!important; }

.code-editor textarea { min-height: 340px!important; }

.code-plot {
    border: 1px solid var(--border-color);
    border-radius: 12px;
    background: var(--card-bg);
    overflow: hidden;
}
.code-plot img {
    max-height: 520px;
    object-fit: contain;
}
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

/* 数据信息表格（数据工作台） */
.info-table { width: 100%; border-collapse: collapse; margin: 0; }
.info-table td { padding: 7px 10px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }
.info-table td:first-child { color: #64748b; font-weight: 600; width: 50px; white-space: nowrap; }
.info-table td:last-child { color: #1e293b; font-weight: 500; text-align: left; }
.info-table tr:last-child td { border-bottom: none; }

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


/* ── Float Chat AI Assistant ── */
#float-chat-fab{position:fixed;bottom:28px;right:28px;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#7c3aed,#6d28d9);display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:99999;box-shadow:0 4px 16px rgba(124,58,237,0.4);transition:transform .3s ease,box-shadow .3s ease;user-select:none}
#float-chat-fab:hover{transform:scale(1.1);box-shadow:0 6px 20px rgba(124,58,237,0.5)}
#float-chat-fab.panel-open{opacity:0;pointer-events:none}
#float-chat-panel{position:fixed;bottom:28px;right:28px;width:380px;height:520px;background:#fff;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,.15);z-index:99998;display:none;flex-direction:column;overflow:hidden;font-family:sans-serif}
.float-chat-header{background:linear-gradient(135deg,#7c3aed,#6d28d9);color:white;padding:14px 18px;display:flex;align-items:center;justify-content:space-between}
.float-chat-title{font-size:15px;font-weight:600}
.float-chat-messages{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
.float-msg{padding:8px 12px;border-radius:12px;max-width:85%;font-size:13px;line-height:1.5;word-wrap:break-word}
.float-msg.user{background:#ede9fe;align-self:flex-end;color:#1e1b4b}
.float-msg.bot{background:#f3f4f6;align-self:flex-start;color:#111827}
#float-typing{color:#9ca3af;font-style:italic}
.float-chat-input-area{padding:10px 12px;border-top:1px solid #e5e7eb;display:flex;gap:8px;background:#fafafa}
#float-chat-input{flex:1;padding:8px 12px;border:1px solid #d1d5db;border-radius:20px;font-size:13px;outline:none}
#float-chat-input:focus{border-color:#7c3aed}
#float-chat-send{padding:8px 16px;background:linear-gradient(135deg,#7c3aed,#6d28d9);color:white;border:none;border-radius:20px;font-size:13px;cursor:pointer;white-space:nowrap}
# 隐藏剪贴板桥接Textbox（DOM中存在但不可见，用于JS轮询复制）
#cls-code-clipboard { position: absolute !important; left: -9999px !important; width: 1px !important; height: 1px !important; opacity: 0 !important; overflow: hidden !important; pointer-events: none !important; }
#cls-code-clipboard label { display: none !important; }
#reg-code-clipboard { position: absolute !important; left: -9999px !important; width: 1px !important; height: 1px !important; opacity: 0 !important; overflow: hidden !important; pointer-events: none !important; }
#reg-code-clipboard label { display: none !important; }
#uns-code-clipboard { position: absolute !important; left: -9999px !important; width: 1px !important; height: 1px !important; opacity: 0 !important; overflow: hidden !important; pointer-events: none !important; }
#uns-code-clipboard label { display: none !important; }
#assoc-code-clipboard { position: absolute !important; left: -9999px !important; width: 1px !important; height: 1px !important; opacity: 0 !important; overflow: hidden !important; pointer-events: none !important; }
#assoc-code-clipboard label { display: none !important; }
#float-chat-send:hover{opacity:.9}
"""




# ── 页面卡片组件 ──
def _card(icon, title, desc, body=""):
    """生成 Gradio 页面顶部信息卡片 HTML
/* ── 移动端响应式适配 ── */
/* ════════════════════════════════════════════
   Step4: UI/UX 优化 - 页面过渡动画
   ════════════════════════════════════════════ */

/* 页面切换动画 */
#page-kg, #page-learning, #page-data, #page-fe,
#page-classify, #page-regress, #page-cluster,
#page-assoc, #page-code, #page-ai {
    transition: opacity 0.25s ease, transform 0.25s ease;
}
.ml-visible {
    animation: fadeIn 0.25s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* 卡片悬浮效果 */
.page-panel {
    transition: box-shadow 0.2s ease;
}
.page-panel:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

/* 加载状态指示 */
button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

/* 错误信息美化 */
.gr-box.error, [class*="error"] {
    border-color: #f87171 !important;
    background: rgba(248,113,113,0.08) !important;
}

/* 表格行悬停 */
.eval-table tr:hover {
    background: rgba(59,130,246,0.08);
}

/* 按钮点击反馈 */
button:active:not(:disabled) {
    transform: scale(0.97);
}

/* 滚动条美化 */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(148,163,184,0.5);
}

@media (max-width: 768px) {
    /* Banner 适配 */
    .top-nav-inner {
        flex-direction: column;
        padding: 14px 12px;
        gap: 10px;
    }
    .brand-title { font-size: 18px !important; }
    .brand-sub { font-size: 10px !important; }
    .brand-icon { width: 36px; height: 36px; }

    /* 侧边栏隐藏/折叠 */
    .sidebar-panel {
        min-width: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    .sidebar-nav label {
        font-size: 12px !important;
        padding: 8px 10px !important;
    }

    /* 页面面板 */
    .page-panel { padding: 12px !important; }

    /* 卡片 */
    .content-card { border-radius: 10px; }
    .card-header { padding: 12px 14px; }
    .card-title { font-size: 14px !important; }
    .card-body { padding: 14px; }

    /* 行布局改为列 */
    .gr-row { flex-direction: column !important; gap: 8px !important; }
    .gr-row > * { max-width: 100% !important; }

    /* 代码编辑器 */
    .code-editor textarea { min-height: 240px !important; font-size: 12px !important; }

    /* 步骤标题 */
    .step-title { font-size: 12px !important; }

    /* 标签导航 */
    .nav-tags { flex-wrap: wrap; justify-content: center; }
    .nav-tag { font-size: 10px; padding: 4px 8px; }

    /* Float Chat */
    #float-chat-panel {
        width: calc(100vw - 20px);
        height: calc(100vh - 60px);
        bottom: 10px;
        right: 10px;
        border-radius: 12px;
    }
    #float-chat-fab {
        width: 48px;
        height: 48px;
        bottom: 16px;
        right: 16px;
    }

    /* 全屏代码 */
    #page-code.maximized { padding: 10px !important; }
    #page-code.maximized .code-editor textarea {
        min-height: calc(100vh - 220px) !important;
    }
}

@media (max-width: 480px) {
    .brand-title { font-size: 15px !important; }
    .nav-tag { font-size: 9px; padding: 3px 6px; border-radius: 12px; }
    .card-title { font-size: 13px !important; }
    .card-icon { width: 32px; height: 32px; font-size: 18px; }

    /* 表格小屏横向滚动 */
    .info-table, .eval-table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;
    }
}

/* 数据工作台表格左对齐 */
#page-data table td,
#page-data table th {
    text-align: left !important;
}

"""
    return (f'<div class="content-card"><div class="card-header"><div class="card-header-inner">'
            f'<div class="card-icon">{icon}</div>'
            f'<div><div class="card-title">{title}</div><div class="card-desc">{desc}</div></div>'
            f'</div></div><div class="card-body">{body}</div></div>')


# ── 数据伦理提醒条 ──
ETHICS = ('<div class="ethics-bar"><strong>🔒 数据伦理提醒：</strong>'
    '实验数据均为公开学术数据集，已脱敏处理。在实际应用中，务必遵守'
    '<strong>数据隐私保护法规</strong>（如《个人信息保护法》），'
    '尊重数据主权，不滥用数据，维护信息安全。</div>')
