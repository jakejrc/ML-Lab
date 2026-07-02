(function() {
  'use strict';
  var API_URL = window.location.origin + '/api/jedi_complete';
  var DEBOUNCE = 200;
  var CACHE = {};
  var pending = null;

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

  function showPanel(items, x, y) {
    var old = document.getElementById('jedi-completions'); if(old) old.remove();
    if (!items || !items.length) return;
    var p = document.createElement('div');
    p.id = 'jedi-completions';
    p.style.cssText = 'position:fixed;left:'+x+'px;top:'+(y+2)+'px;z-index:99999;background:#1e293b;border:1px solid #475569;border-radius:6px;max-height:300px;overflow-y:auto;min-width:280px;font-size:12px;font-family:monospace;box-shadow:0 8px 32px rgba(0,0,0,0.4);';
    var ul = document.createElement('ul');
    ul.style.cssText = 'list-style:none;margin:0;padding:4px 0;';
    for (var i = 0; i < Math.min(items.length, 20); i++) {
      var it = items[i];
      var li = document.createElement('li');
      li.style.cssText = 'padding:4px 12px;cursor:pointer;display:flex;align-items:center;gap:6px;color:#e2e8f0;';
      li.onmouseover = function(){this.style.background='#334155'};
      li.onmouseout = function(){this.style.background='transparent'};
      li.onclick = function(){insertCompletion(this.dataset.label)};
      li.innerHTML = '<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:#334155;color:#94a3b8;min-width:40px;text-align:center;">'+(it.type||'txt')+'</span>'
        + '<span>'+it.label+'</span>'
        + '<span style="color:#64748b;font-size:11px;margin-left:auto;overflow:hidden;text-overflow:ellipsis;max-width:120px;">'+(it.detail||'')+'</span>';
      li.dataset.label = it.label;
      ul.appendChild(li);
    }
    p.appendChild(ul);
    document.body.appendChild(p);
  }

  function insertCompletion(label) {
    var p = document.getElementById('jedi-completions'); if(p) p.remove();
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
      }, DEBOUNCE);
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Escape') { var o = document.getElementById('jedi-completions'); if(o) {o.remove(); e.preventDefault();} }
  }

  function onClick(e) { var o = document.getElementById('jedi-completions'); if(o && !o.contains(e.target)) o.remove(); }

  document.addEventListener('keyup', onKeyUp);
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('click', onClick);
  console.log('[Jedi] Completion injected');
})();
