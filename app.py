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

from ml_lab.logger import logger
logger.info(f'[FONT] font.sans-serif={plt.rcParams["font.sans-serif"]}')

from ml_lab.sandbox_templates import SANDBOX_TEMPLATES
SANDBOX_TEMPLATE = SANDBOX_TEMPLATES["默认（查看数据）"]
from ml_lab.html_templates import TOP_HTML, LEARNING_PATH_HTML
from ml_lab.pages import build_pages
from ml_lab.events import bind_events
from ml_lab.version import VERSION, FULL_NAME

from ml_lab.ui_styles import APP_CSS, ETHICS

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

    logger.info("=" * 55)
    logger.info(f"  ML-Lab {FULL_NAME}: 机器学习可视化实验平台")
    logger.info("  监督学习 + 无监督学习 + 代码沙箱")
    logger.info("  地址: http://0.0.0.0:7860")
    logger.info("=" * 55)

    # ── 自动检测 root_path（反向代理兼容）──
    # 优先级：环境变量 > GRADIO_ROOT_PATH > 默认空
    _root_path = (
        _os.environ.get("GRADIO_ROOT_PATH", "").strip()
        or _os.environ.get("ROOT_PATH", "").strip()
    )
    if _root_path:
        _root_path = _root_path.rstrip("/")  # 去掉末尾斜杠
        logger.info(f"root_path = '{_root_path}' (反向代理模式)")
    else:
        logger.info("root_path 未设置 (直连模式)")

    app = create_app()
    app.queue(max_size=20).launch(
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
""",
    )

