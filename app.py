# -*- coding: utf-8 -*-

"""
ML-Lab v3.8.4 — 机器学习可视化实验平台
入口文件：导入 + 布局 + 事件 + 启动
"""

import sys, os, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gradio as gr

# ── 中文字体配置 ──
# Docker Noto CJK SC（Dockerfile 预提取的 OTF）
for _fp in ['/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf',
            '/usr/share/fonts/opentype/noto/NotoSansCJKSC-Bold.otf',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
    if os.path.exists(_fp):
        try: fm.fontManager.addfont(_fp)
        except: pass

# 项目自带的 SimHei
_simhei_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'SimHei.ttf')
if os.path.exists(_simhei_path):
    try: fm.fontManager.addfont(_simhei_path)
    except: pass

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f'[FONT] font.sans-serif={plt.rcParams["font.sans-serif"]}')

from ml_lab.sandbox_templates import SANDBOX_TEMPLATES
SANDBOX_TEMPLATE = SANDBOX_TEMPLATES["默认（查看数据）"]
from ml_lab.html_templates import TOP_HTML, LEARNING_PATH_HTML
from ml_lab.pages import build_pages
from ml_lab.events import bind_events
from ml_lab.version import VERSION, FULL_NAME
from ml_lab.ui_styles import APP_CSS, ETHICS
from ml_lab.jedi_backend import handle_completion, handle_signatures, handle_diagnostics


def create_app():

    with gr.Blocks(title=f"ML-Lab {FULL_NAME}") as app:

        gr.HTML(TOP_HTML)

        # 构建所有页面 UI 组件
        comps = build_pages()

        # 绑定所有事件
        bind_events(comps)

    return app






if __name__ == "__main__":

    import os as _os

    print("=" * 55)
    print(f"  ML-Lab {FULL_NAME}: 机器学习可视化实验平台")
    print("  监督学习 + 无监督学习 + 代码沙箱")
    print("  地址: http://0.0.0.0:7860")
    print("=" * 55)

    # ── 自动检测 root_path（反向代理兼容）──
    # 优先级：环境变量 > GRADIO_ROOT_PATH > 默认空
    _root_path = (
        _os.environ.get("GRADIO_ROOT_PATH", "").strip()
        or _os.environ.get("ROOT_PATH", "").strip()
    )
    if _root_path:
        _root_path = _root_path.rstrip("/")  # 去掉末尾斜杠
        print(f"  [INFO] root_path = '{_root_path}' (反向代理模式)")
    else:
        print("  [INFO] root_path 未设置 (直连模式)")

    app = create_app()
    queued_app = app.queue(max_size=20)
    from fastapi import Request
    @queued_app.app.post("/api/jedi_complete")
    async def jedi_complete_endpoint(request: Request):
        data = await request.json()
        return handle_completion(data)
    @queued_app.app.post("/api/jedi_signatures")
    async def jedi_signatures_endpoint(request: Request):
        data = await request.json()
        return handle_signatures(data)
    @queued_app.app.post("/api/jedi_diagnostics")
    async def jedi_diagnostics_endpoint(request: Request):
        data = await request.json()
        return handle_diagnostics(data)
    queued_app.launch(_app=queued_app.app,
        css=APP_CSS,
        server_name="0.0.0.0",
        server_port=7860,
        root_path=_root_path,
        js="""
/* ============================================
   ML-Lab Page Switcher v3 (Client-side JS)
   策略：CSS 类名 ml-visible 控制显隐（不受 Gradio 重渲染影响）
   ============================================ */
(function() {
  var PAGE_IDS = ['page-kg','page-learning','page-data','page-fe','page-classify','page-regress','page-cluster','page-assoc','page-code','page-ai'];
  var PAGE_MAP = {
    '🧠 知识图谱': 'page-kg', '📖 学习路径': 'page-learning', '📊 数据工作台': 'page-data',
    '⚙️ 特征工程': 'page-fe', '🏷️ 分类实验': 'page-classify', '📈 回归实验': 'page-regress',
    '🔵 聚类实验': 'page-cluster', '🔗 关联规则': 'page-assoc', '💻 代码沙箱': 'page-code', '🤖 AI助教': 'page-ai'
  };

  function applyVisibility(targetId) {
    PAGE_IDS.forEach(function(id) {
      var el = document.getElementById(id);
      if (!el) return;
      if (id === targetId) {
        el.classList.add('ml-visible');
        el.style.setProperty('display', 'flex', 'important');
      } else {
        el.classList.remove('ml-visible');
        el.style.setProperty('display', 'none', 'important');
      }
    });
  }

  function switchPage(targetId) {
    if (window.__currentPage === targetId) return;
    window.__currentPage = targetId;
    applyVisibility(targetId);
  }

  function bindNav() {
    var labels = document.querySelectorAll('.sidebar-nav label');
    if (labels.length === 0) return false;
    if (window.__mlLabNavInited && window.__mlLabNavBound) return true;
    window.__mlLabNavInited = true;

    // 使用事件委托：监听 document.body（永不被 Gradio 重渲染替换）
    if (!window.__mlLabNavBound) {
      document.body.addEventListener('click', function(e) {
        var label = e.target.closest('.sidebar-nav label');
        if (!label) return;
        var text = label.textContent.trim();
        var targetId = PAGE_MAP[text];
        if (targetId) switchPage(targetId);
      });
      window.__mlLabNavBound = true;
    }

    switchPage('page-kg');
    return true;
  }

  // 策略1: 全局 MutationObserver 监听 body
  var domObserver = new MutationObserver(function() {
    bindNav();
    if (window.__currentPage) applyVisibility(window.__currentPage);
  });
  domObserver.observe(document.body || document.documentElement, { childList: true, subtree: true });

  // 策略2: 定时轮询兜底
  var pollCount = 0;
  var pollTimer = setInterval(function() {
    pollCount++;
    bindNav();
    if (window.__currentPage) applyVisibility(window.__currentPage);
    if (pollCount > 60) {
      clearInterval(pollTimer);
      setInterval(function() {
        bindNav();
        if (window.__currentPage) applyVisibility(window.__currentPage);
      }, 2000);
    }
  }, 500);
})();
/* ============================================ */

(function() {\n  'use strict';\n\n  /* ============================================\n     ML-Lab Float Chat AI Assistant\n     ============================================ */\n\n  var isDragging = false;\n  var hasDragged = false;\n  var startX, startY;\n\n  function createFAB() {\n    var fab = document.createElement('div');\n    fab.id = 'float-chat-fab';\n    fab.innerHTML = '<span style=\"font-size:28px;line-height:1\">&#129302;</span>';\n\n    fab.addEventListener('mousedown', function(e) {\n      isDragging = true;\n      hasDragged = false;\n      startX = e.clientX;\n      startY = e.clientY;\n      fab.style.transition = 'none';\n      e.preventDefault();\n    });\n\n    document.addEventListener('mousemove', function(e) {\n      if (!isDragging) return;\n      var dx = e.clientX - startX;\n      var dy = e.clientY - startY;\n      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) hasDragged = true;\n      if (hasDragged) fab.style.transform = 'translate(' + dx + 'px,' + dy + 'px)';\n    });\n\n    document.addEventListener('mouseup', function() {\n      if (!isDragging) return;\n      isDragging = false;\n      fab.style.transition = '';\n      if (!hasDragged) togglePanel();\n    });\n\n    fab.addEventListener('click', function(e) {\n      if (hasDragged) { e.stopPropagation(); hasDragged = false; }\n    });\n\n    document.body.appendChild(fab);\n  }\n\n  function createPanel() {\n    var panel = document.createElement('div');\n    panel.id = 'float-chat-panel';\n    panel.style.display = 'none';\n    panel.innerHTML =\n      '<div class=\"float-chat-header\">' +\n        '<span class=\"float-chat-title\">AI \\u7b54\\u7591\\u52a9\\u624b</span>' +\n        '<button id=\"float-chat-close\" style=\"background:none;border:none;color:white;font-size:18px;cursor:pointer;padding:0 4px\">&times;</button>' +\n      '</div>' +\n      '<div id=\"float-chat-messages\" class=\"float-chat-messages\"></div>' +\n      '<div class=\"float-chat-input-area\">' +\n        '<input type=\"text\" id=\"float-chat-input\" placeholder=\"\\u8f93\\u5165\\u60a8\\u7684\\u95ee\\u9898...\" autocomplete=\"off\">' +\n        '<button id=\"float-chat-send\">\\u53d1\\u9001</button>' +\n      '</div>';\n    document.body.appendChild(panel);\n\n    var input = panel.querySelector('#float-chat-input');\n    var sendBtn = panel.querySelector('#float-chat-send');\n    var closeBtn = panel.querySelector('#float-chat-close');\n\n    closeBtn.addEventListener('click', function(e) { e.stopPropagation(); togglePanel(); });\n    sendBtn.addEventListener('click', function() { sendMessage(input.value); });\n    input.addEventListener('keydown', function(e) {\n      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input.value); }\n    });\n  }\n\n  function togglePanel() {\n    var panel = document.getElementById('float-chat-panel');\n    var fab = document.getElementById('float-chat-fab');\n    if (!panel || !fab) return;\n    var isOpen = panel.style.display === 'flex';\n    panel.style.display = isOpen ? 'none' : 'flex';\n    var span = fab.querySelector('span');\n    if (isOpen) {\n      fab.classList.remove('panel-open');\n      span.innerHTML = '&#129302;';\n    } else {\n      fab.classList.add('panel-open');\n      span.innerHTML = '&times;';\n      fab.style.transform = '';\n      fab.style.opacity = '';\n      fab.style.cursor = 'pointer';\n    }\n  }\n\nfunction addMessage(text, sender) {\n    var c = document.getElementById('float-chat-messages');\n    if (!c) return;\n    var d = document.createElement('div');\n    d.className = 'float-msg ' + sender;\n    d.innerHTML = text;\n    c.appendChild(d);\n    c.scrollTop = c.scrollHeight;\n  }\n\n  function showTyping() {\n    var c = document.getElementById('float-chat-messages');\n    if (!c) return;\n    var d = document.createElement('div');\n    d.className = 'float-msg bot';\n    d.id = 'float-typing';\n    d.textContent = '\\u6b63\\u5728\\u601d\\u8003...';\n    c.appendChild(d);\n    c.scrollTop = c.scrollHeight;\n  }\n\n  function removeTyping() {\n    var el = document.getElementById('float-typing');\n    if (el) el.remove();\n  }\n\n  function parseSSE(text) {\n    var lines = text.split('\\n');\n    for (var i = 0; i < lines.length; i++) {\n      var line = lines[i].trim();\n      if (line.indexOf('data: ') === 0) {\n        try { return JSON.parse(line.substring(6)); } catch(e) { continue; }\n      }\n    }\n    return null;\n  }\n\n  function sendMessage(text) {\n    text = (text || '').trim();\n    if (!text) return;\n    var input = document.getElementById('float-chat-input');\n    if (input) input.value = '';\n    addMessage(text, 'user');\n    showTyping();\n\n    var xhr = new XMLHttpRequest();\n    xhr.open('POST', '/gradio_api/call/on_float_chat', true);\n    xhr.setRequestHeader('Content-Type', 'application/json');\n    xhr.onload = function() {\n      if (xhr.status !== 200) {\n        removeTyping();\n        addMessage('\\u8bf7\\u6c42\\u5931\\u8d25 (HTTP ' + xhr.status + ')', 'bot');\n        return;\n      }\n      try {\n        var result = JSON.parse(xhr.responseText);\n        var eventId = result.event_id;\n        if (!eventId) {\n          removeTyping();\n          addMessage('\\u65e0\\u6cd5\\u83b7\\u53d6 event_id', 'bot');\n          return;\n        }\n        pollResult(eventId, 0);\n      } catch(e) {\n        removeTyping();\n        addMessage('\\u89e3\\u6790\\u54cd\\u5e94\\u5931\\u8d25', 'bot');\n      }\n    };\n    xhr.onerror = function() {\n      removeTyping();\n      addMessage('\\u7f51\\u7edc\\u9519\\u8bef\\uff0c\\u8bf7\\u68c0\\u67e5\\u8fde\\u63a5', 'bot');\n    };\n    xhr.send(JSON.stringify({data: [text]}));\n  }\n\n  function pollResult(eventId, attempts) {\n    if (attempts > 60) {\n      removeTyping();\n      addMessage('\\u8bf7\\u6c42\\u8d85\\u65f6', 'bot');\n      return;\n    }\n    var xhr = new XMLHttpRequest();\n    xhr.open('GET', '/gradio_api/call/on_float_chat/' + eventId, true);\n    xhr.onload = function() {\n      if (xhr.status !== 200) {\n        setTimeout(function() { pollResult(eventId, attempts + 1); }, 500);\n        return;\n      }\n      var parsed = parseSSE(xhr.responseText);\n      if (!parsed) {\n        setTimeout(function() { pollResult(eventId, attempts + 1); }, 500);\n        return;\n      }\n      if (Array.isArray(parsed) && parsed.length > 0 && parsed[0] !== null && parsed[0] !== undefined && String(parsed[0]).trim() !== '') {\n        removeTyping();\n        addMessage(String(parsed[0]).trim(), 'bot');\n      } else {\n        setTimeout(function() { pollResult(eventId, attempts + 1); }, 500);\n      }\n    };\n    xhr.onerror = function() {\n      setTimeout(function() { pollResult(eventId, attempts + 1); }, 1000);\n    };\n    xhr.send();\n  }\n\n  /* ---- Delay init for Gradio SPA ---- */\n  var initTimer = setInterval(function() {\n    if (document.body && document.body.children.length > 5) {\n      clearInterval(initTimer);\n      createFAB();\n      createPanel();\n    }\n  }, 200);\n  setTimeout(function() {\n    if (!document.getElementById('float-chat-fab')) { createFAB(); createPanel(); }\n  }, 5000);\n})();\n(function(){var attempts=0;var maxAttempts=40;var timer=setInterval(function(){attempts++;var nav=document.querySelector('.top-nav');if(nav&&nav.parentElement&&nav.parentElement.tagName!=='BODY'){document.body.prepend(nav);}if(attempts>=maxAttempts)clearInterval(timer);},500);})();\n(function(){\n      var fixCount=0;\n      function fixImageUrls(){\n        var imgs=document.querySelectorAll('img[src*=\"127.0.0.1\"],img[src*=\"localhost\"]');\n        imgs.forEach(function(img){\n          var src=img.getAttribute('src');\n          if(src){\n            var newSrc=src.replace(/https?:\\/\\/127\\.0\\.0\\.1:\\d+/,window.location.origin);\n            newSrc=newSrc.replace(/https?:\\/\\/localhost:\\d+/,window.location.origin);\n            if(newSrc!==src){img.setAttribute('src',newSrc);fixCount++;}\n          }\n        });\n        var elems=document.querySelectorAll('[style*=\"127.0.0.1\"],[style*=\"localhost\"]');\n        elems.forEach(function(el){\n          var style=el.getAttribute('style');\n          var newStyle=style.replace(/https?:\\/\\/127\\.0\\.0\\.1:\\d+/g,window.location.origin);\n          newStyle=newStyle.replace(/https?:\\/\\/localhost:\\d+/g,window.location.origin);\n          if(newStyle!==style){el.setAttribute('style',newStyle);}\n        });\n      }\n      fixImageUrls();\n      var observer=new MutationObserver(function(mutations){\n        if(mutations.some(function(m){return m.addedNodes.length>0;})){setTimeout(fixImageUrls,100);}\n      });\n      observer.observe(document.body,{childList:true,subtree:true});\n    })();\n\n(function(){\n  var navDebounce = null;\n  function setupClientNav() {\n    var radios = document.querySelectorAll('.sidebar-nav input[type=\"radio\"]');\n    if (radios.length === 0) return false;\n    radios.forEach(function(radio) {\n      radio.addEventListener('change', function() {\n        if (navDebounce) clearTimeout(navDebounce);\n        navDebounce = setTimeout(function() { clientNavSwitch(radio); }, 30);\n      });\n    });\n    return true;\n  }\n  function clientNavSwitch(radio) {\n    var pageIds = ['page-learning','page-data','page-fe','page-classify','page-regress','page-cluster','page-assoc','page-code','page-ai'];\n    var idx = Array.prototype.indexOf.call(document.querySelectorAll('.sidebar-nav input[type=\"radio\"]'), radio);\n    if (idx < 0 || idx >= pageIds.length) return;\n    var targetId = pageIds[idx];\n    for (var i = 0; i < pageIds.length; i++) {\n      var el = document.getElementById(pageIds[i]);\n      if (el) {\n        if (pageIds[i] === targetId) {
        el.classList.add('ml-visible');
        el.style.setProperty('display', 'flex', 'important');
      } else {
        el.classList.remove('ml-visible');
        el.style.setProperty('display', 'none', 'important');
      }\n      }\n    }\n  }\n  var navTimer = setInterval(function() {\n    if (setupClientNav()) clearInterval(navTimer);\n  }, 300);\n  setTimeout(function() { clearInterval(navTimer); }, 10000);\n})();  // ===== 自动下载: 监控gr.File文件链接并自动触发浏览器下载 =====\n  (function() {\n    var fileContainer = null;\n    var lastHref = '';\n    var observer = new MutationObserver(function(mutations) {\n      mutations.forEach(function(m) {\n        var added = m.addedNodes;\n        for (var i = 0; i < added.length; i++) {\n          if (added[i].nodeType === 1) {\n            var link = added[i].querySelector && added[i].querySelector('a[href*=\"file\"]');\n            if (!link && added[i].tagName === 'A' && added[i].href && added[i].href.indexOf('file') >= 0) {\n              link = added[i];\n            }\n            if (link && link.href !== lastHref) {\n              lastHref = link.href;\n              setTimeout(function() { link.click(); }, 200);\n            }\n          }\n        }\n      });\n    });\n    var containerTimer = setInterval(function() {\n      // 找到gr.File组件的容器（包含\"下载文件\"标签的父div）\n      var allDivs = document.querySelectorAll('.block');\n      for (var i = 0; i < allDivs.length; i++) {\n        if (allDivs[i].textContent.indexOf('下载文件') >= 0 && allDivs[i].querySelector('a[href*=\"file\"]')) {\n          fileContainer = allDivs[i];\n          clearInterval(containerTimer);\n          observer.observe(fileContainer, {childList: true, subtree: true});\n          // 如果当前已有链接，先记录\n          var existing = fileContainer.querySelector('a[href*=\"file\"]');\n          if (existing) lastHref = existing.href;\n          break;\n        }\n      }\n    }, 500);\n    setTimeout(function() { clearInterval(containerTimer); }, 15000);\n  })();
  // ===== Clipboard auto-copy: poll hidden Textboxes (execCommand fallback for HTTP) =====
  (function() {
    var lastVals = {'#cls-code-clipboard': '', '#reg-code-clipboard': '', '#uns-code-clipboard': '', '#assoc-code-clipboard': ''};
    var ids = Object.keys(lastVals);
    setInterval(function() {
      for (var i = 0; i < ids.length; i++) {
        var ta = document.querySelector(ids[i] + ' textarea');
        if (!ta) continue;
        var v = ta.value;
        if (v && v !== lastVals[ids[i]] && (v.indexOf('import') >= 0 || v.indexOf('sklearn') >= 0 || v.indexOf('from sklearn') >= 0 || v.indexOf('mlxtend') >= 0)) {
          lastVals[ids[i]] = v;
          var tmp = document.createElement('textarea');
          tmp.value = v;
          tmp.style.position = 'fixed';
          tmp.style.left = '-9999px';
          tmp.style.top = '0';
          document.body.appendChild(tmp);
          tmp.select();
          tmp.setSelectionRange(0, 99999);
          try {
            var ok = document.execCommand('copy');
            console.log('[ML-Lab] ' + ids[i] + ' execCommand copy: ' + (ok ? 'OK' : 'FAIL') + ' (' + v.length + ' chars)');
          } catch(e) {
            console.warn('[ML-Lab] execCommand copy error:', e);
          }
          document.body.removeChild(tmp);
        }
      }
    }, 300);
  })();

// ===== Jedi Code Completion =====
(function() {
  'use strict';
  var API_URL = window.location.origin + '/api/jedi_complete';
  var SIG_URL = window.location.origin + '/api/jedi_signatures';
  var DEBOUNCE = 200;
  var CACHE = {};
  var pending = null;
  var selectedIndex = -1;

  function getEditor() { return document.querySelector('.cm-editor'); }
  function getCode() { var c = document.querySelector('.cm-content'); return c ? c.textContent : ''; }

  function getCursorLineCol() {
    var sel = window.getSelection();
    if (!sel.rangeCount) return {line:1, col:0};
    var text = getCode();
    if (!text) return {line:1, col:0};
    var lines = text.split('\n');
    var offset = sel.getRangeAt(0).startOffset;
    var pos = 0;
    for (var i = 0; i < lines.length; i++) {
      if (offset <= pos + lines[i].length) return {line: i+1, col: offset - pos};
      pos += lines[i].length + 1;
    }
    return {line: lines.length, col: lines[lines.length-1].length};
  }

  function fetchCompletions(code, line, col, cb) {
    var key = line + ':' + col + ':' + code.length;
    if (CACHE[key] && Date.now() - CACHE[key].ts < 10000) { cb(CACHE[key].data); return; }
    fetch(API_URL, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({code:code, line:line, col:col})})
    .then(function(r){return r.json()})
    .then(function(d){CACHE[key]={ts:Date.now(),data:d}; cb(d)})
    .catch(function(){cb(null)});
  }

  function fetchSignatures(code, line, col, cb) {
    fetch(SIG_URL, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({code:code, line:line, col:col})})
    .then(function(r){return r.json()})
    .then(function(d){cb(d)})
    .catch(function(){cb(null)});
  }

  function showSignatureHint(sigs) {
    var old = document.getElementById('jedi-signature'); if(old) old.remove();
    if (!sigs || !sigs.length) return;
    var p = document.createElement('div');
    p.id = 'jedi-signature';
    p.style.cssText = 'position:fixed;left:50%;transform:translateX(-50%);bottom:40px;z-index:99999;background:#0f172a;border:1px solid #475569;border-radius:8px;padding:8px 16px;font-size:12px;font-family:monospace;color:#e2e8f0;max-width:600px;box-shadow:0 -4px 24px rgba(0,0,0,0.5);text-align:center;';
    for (var i = 0; i < Math.min(sigs.length, 3); i++) {
      var s = sigs[i];
      var paramsHtml = '';
      for (var j = 0; j < s.params.length; j++) {
        var isActive = (j === s.index);
        paramsHtml += (isActive ? '<b style="color:#facc15;">' : '<span style="color:#94a3b8;">') + s.params[j] + (isActive ? '</b>' : '</span>');
        if (j < s.params.length - 1) paramsHtml += ', ';
      }
      p.innerHTML = '<span style="color:#60a5fa;">' + s.name + '</span>( ' + paramsHtml + ' )';
      if (s.docstring) p.innerHTML += '<div style="font-size:11px;color:#64748b;margin-top:2px;">' + s.docstring + '</div>';
    }
    document.body.appendChild(p);
  }

  function showPanel(items, x, y) {
    var old = document.getElementById('jedi-completions'); if(old) old.remove();
    if (!items || !items.length) return;
    selectedIndex = 0;
    var p = document.createElement('div');
    p.id = 'jedi-completions';
    p.style.cssText = 'position:fixed;left:'+Math.max(10,x)+'px;top:'+Math.max(10,y+2)+'px;z-index:99999;background:#1e293b;border:1px solid #475569;border-radius:6px;max-height:300px;overflow-y:auto;min-width:280px;font-size:12px;font-family:monospace;box-shadow:0 8px 32px rgba(0,0,0,0.4);';
    var ul = document.createElement('ul');
    ul.style.cssText = 'list-style:none;margin:0;padding:4px 0;';
    for (var i = 0; i < Math.min(items.length, 20); i++) {
      var it = items[i];
      var li = document.createElement('li');
      li.style.cssText = 'padding:4px 12px;cursor:pointer;display:flex;align-items:center;gap:6px;color:#e2e8f0;';
      li.onmouseover = function(){selectItem(this);};
      li.onclick = function(){insertCompletion(this.dataset.label);};
      li.dataset.index = i;
      li.dataset.label = it.label;
      li.innerHTML = '<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:#334155;color:#94a3b8;min-width:40px;text-align:center;">'+(it.type||'txt')+'</span>'
        + '<span>'+it.label+'</span>'
        + '<span style="color:#64748b;font-size:11px;margin-left:auto;overflow:hidden;text-overflow:ellipsis;max-width:120px;">'+(it.detail||'')+'</span>';
      ul.appendChild(li);
    }
    p.appendChild(ul);
    document.body.appendChild(p);
    selectItem(ul.children[0]);
  }

  function selectItem(li) {
    if (!li) return;
    var items = document.querySelectorAll('#jedi-completions li');
    items.forEach(function(el) { el.style.background = 'transparent'; });
    li.style.background = '#334155';
    selectedIndex = parseInt(li.dataset.index);
    li.scrollIntoView({block: 'nearest'});
  }

  function selectRelative(delta) {
    var items = document.querySelectorAll('#jedi-completions li');
    if (!items.length) return;
    var newIdx = Math.max(0, Math.min(items.length - 1, selectedIndex + delta));
    selectItem(items[newIdx]);
  }

  function insertCompletion(label) {
    var p = document.getElementById('jedi-completions'); if(p) p.remove();
    var sig = document.getElementById('jedi-signature'); if(sig) sig.remove();
    var content = document.querySelector('.cm-content');
    if (!content) return;
    var sel = window.getSelection();
    if (!sel.rangeCount) return;
    var r = sel.getRangeAt(0);
    var node = r.startContainer;
    var text = node.textContent || '';
    var pos = r.startOffset;
    var start = pos;
    while (start > 0 && /[\w.]/.test(text[start-1])) start--;
    node.textContent = text.substring(0, start) + label + text.substring(pos);
    var nr = document.createRange();
    nr.setStart(node, start + label.length);
    nr.collapse(true);
    sel.removeAllRanges(); sel.addRange(nr);
  }

  function checkSignature(code, line, col) {
    var text = getCode();
    if (!text) return;
    var lines = text.split('\n');
    var currentLine = lines[line-1] || '';
    // 检查行中是否包含 '(' 且光标在 '(' 之后
    var parenPos = currentLine.lastIndexOf('(', col);
    if (parenPos >= 0 && col > parenPos) {
      fetchSignatures(code, line, col, function(d) {
        var old = document.getElementById('jedi-signature'); if(old) old.remove();
        if (d && d.signatures && d.signatures.length) {
          showSignatureHint(d.signatures);
        }
      });
    } else {
      var old = document.getElementById('jedi-signature'); if(old) old.remove();
    }
  }

  function onKeyUp(e) {
    var editor = getEditor();
    if (!editor || !editor.contains(e.target)) return;
    var tr = ['.', '_'];
    if ((e.key.length===1 && /[a-zA-Z0-9_]/.test(e.key)) || tr.indexOf(e.key)>=0) {
      if (pending) clearTimeout(pending);
      pending = setTimeout(function() {
        var code = getCode();
        var pos = getCursorLineCol();
        if (!code) return;
        var sel = window.getSelection();
        var rect = sel.rangeCount ? sel.getRangeAt(0).getBoundingClientRect() : null;
        fetchCompletions(code, pos.line, pos.col, function(d) {
          var o = document.getElementById('jedi-completions'); if(o) o.remove();
          if (d && d.completions && d.completions.length) {
            showPanel(d.completions, rect ? rect.left : 0, rect ? rect.bottom : 0);
          }
        });
        // 同时检查签名提示
        checkSignature(code, pos.line, pos.col);
      }, DEBOUNCE);
    } else if (e.key === '(') {
      // 输入 ( 时立即检查签名
      var code = getCode();
      var pos = getCursorLineCol();
      if (code) checkSignature(code, pos.line, pos.col);
    } else {
      // 非触发键关闭面板
      var panel = document.getElementById('jedi-completions');
      if (panel) { panel.remove(); }
    }
  }

  function onKeyDown(e) {
    var panel = document.getElementById('jedi-completions');
    if (panel) {
      if (e.key === 'ArrowDown') { selectRelative(1); e.preventDefault(); return; }
      if (e.key === 'ArrowUp') { selectRelative(-1); e.preventDefault(); return; }
      if (e.key === 'Enter' || e.key === 'Tab') {
        var items = panel.querySelectorAll('li');
        for (var i = 0; i < items.length; i++) {
          if (parseInt(items[i].dataset.index) === selectedIndex) {
            insertCompletion(items[i].dataset.label);
            e.preventDefault();
            return;
          }
        }
      }
    }
    if (e.key === 'Escape') {
      var o = document.getElementById('jedi-completions'); if(o) {o.remove();}
      var s = document.getElementById('jedi-signature'); if(s) {s.remove();}
      if (o || s) e.preventDefault();
    }
  }

  function onClick(e) {
    var p = document.getElementById('jedi-completions');
    if(p && !p.contains(e.target)) p.remove();
  }

  document.addEventListener('keyup', onKeyUp);
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('click', onClick);
  // ── 代码实时诊断 ──
  var DIAG_URL = window.location.origin + '/api/jedi_diagnostics';
  var diagTimer = null;
  var lastDiagCode = '';

  function fetchDiagnostics(code) {
    fetch(DIAG_URL, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({code:code})})
    .then(function(r){return r.json()})
    .then(function(d){showDiagnostics(d.issues)})
    .catch(function(){});
  }

  function showDiagnostics(issues) {
    var old = document.getElementById('jedi-diagnostics'); if(old) old.remove();
    if (!issues || !issues.length) return;

    var container = document.createElement('div');
    container.id = 'jedi-diagnostics';
    container.style.cssText = 'margin:4px 0;font-size:11px;font-family:monospace;';

    for (var i = 0; i < Math.min(issues.length, 5); i++) {
      var it = issues[i];
      var color = it.severity === 'error' ? '#f87171' : '#facc15';
      var icon = it.severity === 'error' ? '\u2716' : '\u26a0';
      var badge = document.createElement('div');
      badge.style.cssText = 'padding:2px 8px;border-left:3px solid ' + color + ';margin:2px 0;background:rgba(0,0,0,0.2);border-radius:2px;color:' + color + ';';
      badge.textContent = icon + ' L' + it.line + ': ' + it.message;
      container.appendChild(badge);
    }

    // 插入到代码编辑器上方
    var editor = getEditor();
    if (editor && editor.parentNode) {
      editor.parentNode.insertBefore(container, editor);
    }
  }

  function scheduleDiagnostics() {
    if (diagTimer) clearTimeout(diagTimer);
    diagTimer = setTimeout(function() {
      var code = getCode();
      if (!code || code === lastDiagCode) return;
      lastDiagCode = code;
      // 仅在代码变更后1秒无输入时检查
      if (code.trim()) fetchDiagnostics(code);
    }, 1000);
  }

  // 重写 onKeyUp 在现有逻辑后添加诊断调度
  var _origKeyUp = onKeyUp;
  onKeyUp = function(e) {
    _origKeyUp(e);
    scheduleDiagnostics();
  };

  console.log('[Jedi] Enhanced completion with keyboard nav + signatures + diagnostics');
})();

""",
    )

