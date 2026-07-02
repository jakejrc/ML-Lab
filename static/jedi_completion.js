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
  console.log('[Jedi] Enhanced completion with keyboard nav + signatures');
})();
