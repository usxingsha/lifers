/* Lifers 自研 GUI：/api/bridge + Monaco（优先 static/vendor 离线，否则 jsDelivr） */

const MONACO_CDN = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs';
const MONACO_LOCAL = '/static/vendor/monaco/min/vs';

async function resolveMonacoVs() {
  try {
    const r = await fetch(`${MONACO_LOCAL}/loader.js`, { method: 'HEAD', cache: 'no-store' });
    if (r.ok) {
      return MONACO_LOCAL;
    }
  } catch (_) {
    /* offline or blocked */
  }
  return MONACO_CDN;
}

function el(id) {
  return document.getElementById(id);
}

function appendBubble(role, text) {
  const log = el('log');
  const div = document.createElement('div');
  div.className = `bubble ${role}`;
  const r = document.createElement('div');
  r.className = 'role';
  r.textContent = role === 'user' ? 'You' : 'Lifers';
  div.appendChild(r);
  const t = document.createElement('div');
  t.textContent = text;
  div.appendChild(t);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function loadTheme() {
  try {
    const r = await fetch('/api/editor-settings');
    const j = await r.json();
    if (!j.ok || !j.theme) return;
    const th = j.theme;
    document.documentElement.style.setProperty('--surface', th.surface || '#1e1e1e');
    document.documentElement.style.setProperty('--text', th.text || '#d4d4d4');
    document.documentElement.style.setProperty('--font-size', `${th.fontSizePx || 14}px`);
    document.documentElement.style.setProperty('--font-mono', th.fontFamily || 'Consolas, monospace');
    el('meta-line').textContent = `${th.sourceFile || ''} · LIFERS_ROOT=${j.lifersRoot || ''}`;
    const fb = el('editor-buf');
    fb.style.fontSize = `${th.fontSizePx || 14}px`;
    fb.style.fontFamily = th.fontFamily || 'Consolas, monospace';
  } catch {
    el('meta-line').textContent = 'theme load failed';
  }
}

async function sendMessage() {
  const ta = el('input');
  const text = (ta.value || '').trim();
  if (!text) return;
  const btn = el('send');
  btn.disabled = true;
  appendBubble('user', text);
  ta.value = '';
  try {
    const r = await fetch('/api/bridge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
      body: JSON.stringify({ text, contextFiles: [] }),
    });
    const j = await r.json();
    if (j.ok && (j.text || '').trim()) {
      appendBubble('assistant', j.text);
    } else {
      appendBubble('assistant', `（错误）${j.error || r.status}`);
    }
  } catch (e) {
    appendBubble('assistant', `（网络）${e}`);
  } finally {
    btn.disabled = false;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error('script ' + src));
    document.head.appendChild(s);
  });
}

let monacoEditor = null;

async function bootMonaco() {
  const st = el('monaco-status');
  const btn = el('btn-monaco');
  if (monacoEditor) {
    st.textContent = 'Monaco 已加载';
    return;
  }
  btn.disabled = true;
  st.textContent = '正在加载 Monaco…';
  try {
    const vs = await resolveMonacoVs();
    await loadScript(`${vs}/loader.js`);
    const req = window.require;
    req.config({ paths: { vs } });
    await new Promise((resolve, reject) => {
      req(['vs/editor/editor.main'], () => resolve(), (err) => reject(err));
    });
    el('monaco-host').classList.remove('hidden');
    el('editor-buf').classList.add('hidden');
    const th = getComputedStyle(document.documentElement);
    const fs = parseInt(th.getPropertyValue('--font-size'), 10) || 14;
    const ff = th.getPropertyValue('--font-mono').trim() || 'Consolas, monospace';
    monacoEditor = window.monaco.editor.create(el('monaco-host'), {
      value: el('editor-buf').value || '# Lifers buffer\n',
      language: 'markdown',
      theme: 'vs-dark',
      fontSize: fs,
      fontFamily: ff,
      automaticLayout: true,
      minimap: { enabled: true },
      wordWrap: 'on',
    });
    st.textContent = vs.startsWith(MONACO_LOCAL) ? 'Monaco 就绪（离线 vendor）' : 'Monaco 就绪（CDN）';
  } catch (e) {
    st.textContent = `Monaco 失败：${e}`;
    el('monaco-host').classList.add('hidden');
    el('editor-buf').classList.remove('hidden');
  } finally {
    btn.disabled = false;
  }
}

el('send').addEventListener('click', sendMessage);
el('input').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) {
    ev.preventDefault();
    sendMessage();
  }
});
el('btn-monaco').addEventListener('click', () => {
  bootMonaco();
});

loadTheme();
