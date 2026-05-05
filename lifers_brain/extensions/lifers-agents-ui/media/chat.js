(function () {
  const vscode = acquireVsCodeApi();

  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const btnSend = document.getElementById('btn-send');
  const ctxBar = document.getElementById('context-bar');
  const btnDocs = document.getElementById('btn-open-docs');
  const btnMarket = document.getElementById('btn-market');
  const ctxBadge = document.getElementById('ctx-badge');
  const btnCtxFold = document.getElementById('btn-ctx-fold');
  const ctxChev = document.getElementById('ctx-chev');
  const ctxExpanded = document.getElementById('ctx-expanded');
  const btnClearCtx = document.getElementById('btn-clear-all-ctx');
  const btnReviewReadme = document.getElementById('btn-review-readme');
  const selSandbox = document.getElementById('sel-sandbox');
  const btnAttachFile = document.getElementById('btn-attach-file');
  const btnAttachFolder = document.getElementById('btn-attach-folder');
  const sendSpinner = document.getElementById('send-spinner');
  const slashPop = document.getElementById('slash-pop');
  const atPop = document.getElementById('at-pop');
  const bridgeProgressEl = document.getElementById('bridge-progress');
  const bridgeProgressText = document.getElementById('bridge-progress-text');

  const brandIcon = typeof window.__LIFERS_BRAND_ICON__ === 'string' ? window.__LIFERS_BRAND_ICON__ : '';

  function currentBrandSrc() {
    const b = window.__LIFERS_BRAND__;
    if (!b || !b.kitten) return brandIcon;
    if (b.mode === 'kitten') return b.kitten;
    if (b.mode === 'phoenix') return b.phoenix || b.kitten;
    var ms = Math.max(1000, Number(b.intervalMs) || 4000);
    return Math.floor(Date.now() / ms) % 2 === 0 ? b.kitten : (b.phoenix || b.kitten);
  }

  /** @type {{ sessions: any[]; activeId: string | null; contextFiles: string[] }} */
  let state = {
    sessions: [],
    activeId: null,
    contextFiles: [],
  };

  let sending = false;

  /** Slash commands: whole line must be `/prefix` (optional filter after /) */
  const SLASH_CMDS = [
    { prefix: 'add-file', label: '添加上下文文件', hint: '打开多选文件对话框', type: 'pickContextFiles' },
    { prefix: 'add-folder', label: '添加目录', hint: '递归采集目录内文件', type: 'pickContextFolder' },
    { prefix: 'new', label: '新建会话', hint: 'New Agent', type: 'newSession' },
    { prefix: 'clear', label: '清空上下文', hint: '移除当前会话的路径列表', type: 'clearContext' },
    { prefix: 'sessions', label: '显示会话建立', hint: '左侧 Lifers 栏原生树', type: 'focusSessions' },
    { prefix: 'settings', label: 'Lifers 设置', hint: 'lifers.*', type: 'openSettings' },
    { prefix: 'readme', label: '打开 README', hint: '预览 Markdown', type: 'openReadme' },
    { prefix: 'chat', label: '前台显示 Chat', hint: '确保编辑器区面板', type: 'openChat' },
  ];

  let slashOpen = false;
  let slashFilter = '';
  let slashSel = 0;
  let slashItems = [];

  let atOpen = false;
  let atSel = 0;
  /** @type {string[]} */
  let atItems = [];
  let atQuery = '';
  let atReqSeq = 0;
  let lastAtReq = -1;
  let atDebounce = null;

  vscode.postMessage({ type: 'ready', surface: 'chat' });

  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (msg.type === 'bootstrap') {
      const st = msg.state || {};
      state.sessions = st.sessions || [];
      state.activeId = st.activeId || null;
      state.contextFiles = st.contextFiles || [];
      applyLifersConfig(msg.lifersConfig);
      renderAll();
    }
    if (msg.type === 'lifersConfig') {
      applyLifersConfig(msg);
    }
    if (msg.type === 'lifersBrand') {
      window.__LIFERS_BRAND__ = Object.assign({}, window.__LIFERS_BRAND__ || {}, msg.brand || {});
      window.__LIFERS_BRAND_ICON__ = window.__LIFERS_BRAND__.kitten || window.__LIFERS_BRAND_ICON__;
      if (typeof window.applyLifersBrandIcons === 'function') {
        window.applyLifersBrandIcons();
      }
      renderChat();
    }
    if (msg.type === 'reply') {
      /* 扩展先 post bootstrap 再 post reply；此处只结束「发送中」状态 */
      sending = false;
      setSendingUi(false);
    }
    if (msg.type === 'bridgeProgressClear') {
      clearBridgeProgress();
    }
    if (msg.type === 'bridgeProgress') {
      appendBridgeProgress(typeof msg.line === 'string' ? msg.line : '');
    }
    if (msg.type === 'context') {
      state.contextFiles = msg.files || [];
      renderContext();
    }
    if (msg.type === 'fileCompletionResult') {
      if (msg.requestId !== lastAtReq) return;
      atItems = Array.isArray(msg.files) ? msg.files : [];
      atSel = 0;
      renderAtMenu();
      layoutPops();
    }
  });

  window.addEventListener('resize', layoutPops);

  /**
   * @param {{ model?: string; sandbox?: boolean } | undefined} cfg
   */
  function applyLifersConfig(cfg) {
    if (!cfg) return;
    if (selSandbox && typeof cfg.sandbox === 'boolean') {
      selSandbox.value = cfg.sandbox ? 'safe' : 'unsafe';
    }
  }

  function setSendingUi(on) {
    if (sendSpinner) sendSpinner.classList.toggle('hidden', !on);
    if (btnSend) btnSend.disabled = !!on;
  }

  function clearBridgeProgress() {
    if (bridgeProgressText) bridgeProgressText.textContent = '';
    if (bridgeProgressEl) bridgeProgressEl.classList.add('hidden');
  }

  function appendBridgeProgress(line) {
    if (!line || !bridgeProgressText || !bridgeProgressEl) return;
    bridgeProgressEl.classList.remove('hidden');
    bridgeProgressText.textContent += (bridgeProgressText.textContent ? '\n' : '') + line;
    bridgeProgressText.scrollTop = bridgeProgressText.scrollHeight;
  }

  function layoutPops() {
    if (!inputEl) return;
    const card = inputEl.closest('.composer-input-card');
    if (!card || !slashPop || !atPop) return;
    const ta = inputEl.getBoundingClientRect();
    const cr = card.getBoundingClientRect();
    const top = ta.bottom - cr.top + 4;
    const left = ta.left - cr.left;
    const w = ta.width;
    [slashPop, atPop].forEach((el) => {
      if (!el.classList.contains('hidden')) {
        el.style.top = top + 'px';
        el.style.left = left + 'px';
        el.style.width = Math.min(Math.max(w, 280), 560) + 'px';
      }
    });
  }

  function closeMenus() {
    slashOpen = false;
    atOpen = false;
    if (slashPop) slashPop.classList.add('hidden');
    if (atPop) atPop.classList.add('hidden');
  }

  function detectTriggers() {
    if (!inputEl) return;
    const v = inputEl.value;
    const c = inputEl.selectionStart || 0;
    const before = v.slice(0, c);

    const lineStart = before.lastIndexOf('\n') + 1;
    const line = before.slice(lineStart);

    const slashMatch = line.match(/^\s*\/([\w-]*)$/);
    if (slashMatch) {
      atOpen = false;
      if (atPop) atPop.classList.add('hidden');
      slashOpen = true;
      slashFilter = (slashMatch[1] || '').toLowerCase();
      buildSlashList();
      renderSlashMenu();
      if (slashPop) slashPop.classList.remove('hidden');
      layoutPops();
      return;
    }
    slashOpen = false;
    if (slashPop) slashPop.classList.add('hidden');

    const atPos = before.lastIndexOf('@');
    if (atPos < 0) {
      atOpen = false;
      if (atPop) atPop.classList.add('hidden');
      return;
    }
    const afterAt = before.slice(atPos + 1);
    if (/[\s\n]/.test(afterAt)) {
      atOpen = false;
      if (atPop) atPop.classList.add('hidden');
      return;
    }

    slashOpen = false;
    atOpen = true;
    atQuery = afterAt;
    atReqSeq += 1;
    const myId = atReqSeq;
    lastAtReq = myId;
    if (atDebounce) clearTimeout(atDebounce);
    if (atPop) {
      atPop.classList.remove('hidden');
      atPop.innerHTML = '<div class="pop-empty">扫描工作区路径…</div>';
      layoutPops();
    }
    atDebounce = setTimeout(() => {
      vscode.postMessage({ type: 'fileCompletion', query: atQuery, requestId: myId });
    }, 160);
  }

  function buildSlashList() {
    slashItems = SLASH_CMDS.filter((cmd) => {
      if (!slashFilter) return true;
      const hay = (cmd.prefix + ' ' + cmd.label + ' ' + cmd.hint).toLowerCase();
      return hay.includes(slashFilter);
    });
    if (slashSel >= slashItems.length) slashSel = Math.max(0, slashItems.length - 1);
  }

  function renderSlashMenu() {
    if (!slashPop) return;
    slashPop.innerHTML = '';
    if (!slashItems.length) {
      slashPop.innerHTML = '<div class="pop-empty">无匹配命令</div>';
      return;
    }
    slashItems.forEach((cmd, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pop-item' + (i === slashSel ? ' active' : '');
      btn.setAttribute('role', 'option');
      btn.innerHTML =
        '<span class="pop-title">/' +
        escapeHtml(cmd.prefix) +
        '</span>' +
        '<span class="pop-hint">' +
        escapeHtml(cmd.label + ' — ' + cmd.hint) +
        '</span>';
      btn.addEventListener('mousedown', (e) => {
        e.preventDefault();
        slashSel = i;
        execSlashSelected();
      });
      slashPop.appendChild(btn);
    });
  }

  function renderAtMenu() {
    if (!atPop) return;
    atPop.innerHTML = '';
    if (!atItems.length) {
      atPop.innerHTML = '<div class="pop-empty">无匹配文件（调整 @ 后的关键字）</div>';
      return;
    }
    atItems.forEach((p, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pop-item' + (i === atSel ? ' active' : '');
      btn.setAttribute('role', 'option');
      btn.innerHTML = '<span class="pop-path">' + escapeHtml(p) + '</span>';
      btn.addEventListener('mousedown', (e) => {
        e.preventDefault();
        atSel = i;
        pickAtPath();
      });
      atPop.appendChild(btn);
    });
  }

  function execSlashSelected() {
    const cmd = slashItems[slashSel];
    closeMenus();
    removeSlashLine();
    if (!cmd) return;
    vscode.postMessage({ type: cmd.type });
  }

  /** Remove current line that contains only `/...` */
  function removeSlashLine() {
    if (!inputEl) return;
    const v = inputEl.value;
    const c = inputEl.selectionStart || 0;
    const before = v.slice(0, c);
    const lineStart = before.lastIndexOf('\n') + 1;
    const after = v.slice(c);
    const nextNl = after.indexOf('\n');
    const rest = nextNl >= 0 ? after.slice(nextNl) : '';
    inputEl.value = v.slice(0, lineStart) + rest;
    inputEl.selectionStart = inputEl.selectionEnd = lineStart;
  }

  /** Remove `@query` fragment before cursor */
  function removeAtFragment() {
    if (!inputEl) return;
    const v = inputEl.value;
    const c = inputEl.selectionStart || 0;
    const before = v.slice(0, c);
    const atPos = before.lastIndexOf('@');
    if (atPos < 0) return;
    inputEl.value = v.slice(0, atPos) + v.slice(c);
    inputEl.selectionStart = inputEl.selectionEnd = atPos;
  }

  function pickAtPath() {
    const p = atItems[atSel];
    closeMenus();
    removeAtFragment();
    if (p) {
      vscode.postMessage({ type: 'addContextRelPaths', paths: [p] });
    }
  }

  function renderContext() {
    const n = state.contextFiles.length;
    if (ctxBadge) ctxBadge.textContent = String(n);

    if (!ctxBar) return;

    if (!state.contextFiles.length) {
      ctxBar.innerHTML =
        '<span class="muted">暂无路径。<strong>@</strong> 搜索文件或下方 <strong>📄📁</strong>。</span>';
      return;
    }
    ctxBar.innerHTML = state.contextFiles
      .map((f) => '<span class="context-tag">' + escapeHtml(f) + '</span>')
      .join('');
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderAll() {
    renderContext();
    renderChat();
  }

  function renderChat() {
    const s = state.sessions.find((x) => x.id === state.activeId);
    messagesEl.innerHTML = '';
    if (!s) {
      messagesEl.innerHTML =
        '<div class="msg-bot">在<strong>左侧 Lifers →「会话建立」</strong>或<strong>资源管理器 →「会话建立」</strong>中选会话；或用 <strong>/new</strong>、命令「新建会话」。会话列表不在本 Chat 内。</div>';
      return;
    }
    (s.messages || []).forEach((m) => {
      appendMessage(m.role, m.content, true);
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendMessage(role, text, noScroll) {
    const div = document.createElement('div');
    const isUser = role === 'user';
    div.className = isUser ? 'msg-user' : 'msg-bot';
    if (brandIcon || (window.__LIFERS_BRAND__ && window.__LIFERS_BRAND__.kitten)) {
      var src = currentBrandSrc() || brandIcon;
      div.innerHTML =
        '<span class="msg-head"><img src="' +
        src +
        '" width="16" height="16" alt="" class="lifers-brand-img"/> ' +
        (isUser ? 'You' : 'Agent') +
        '</span><br/>' +
        escapeHtml(text).replace(/\n/g, '<br/>');
    } else {
      div.textContent = (isUser ? 'You\n' : 'Agent\n') + text;
    }
    messagesEl.appendChild(div);
    if (!noScroll) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  if (btnDocs) {
    btnDocs.addEventListener('click', () => {
      vscode.postMessage({ type: 'openReadme' });
    });
  }

  if (btnMarket) {
    btnMarket.addEventListener('click', () => {
      vscode.postMessage({
        type: 'openExternal',
        url: 'https://marketplace.visualstudio.com/search?term=lifers&target=VSCode',
      });
    });
  }

  if (btnCtxFold && ctxExpanded && ctxChev) {
    btnCtxFold.addEventListener('click', () => {
      ctxExpanded.classList.toggle('hidden');
      const collapsed = ctxExpanded.classList.contains('hidden');
      btnCtxFold.setAttribute('aria-expanded', (!collapsed).toString());
      ctxChev.textContent = collapsed ? '▸' : '▾';
    });
  }

  if (btnClearCtx) {
    btnClearCtx.addEventListener('click', () => {
      vscode.postMessage({ type: 'clearContext' });
    });
  }

  if (btnReviewReadme) {
    btnReviewReadme.addEventListener('click', () => {
      vscode.postMessage({ type: 'openReadme' });
    });
  }

  if (selSandbox) {
    selSandbox.addEventListener('change', () => {
      vscode.postMessage({ type: 'setLifersSandbox', sandbox: selSandbox.value === 'safe' });
    });
  }

  if (btnAttachFile) {
    btnAttachFile.addEventListener('click', () => {
      vscode.postMessage({ type: 'pickContextFiles' });
    });
  }

  if (btnAttachFolder) {
    btnAttachFolder.addEventListener('click', () => {
      vscode.postMessage({ type: 'pickContextFolder' });
    });
  }

  if (btnSend) btnSend.addEventListener('click', send);

  if (inputEl) {
    inputEl.addEventListener('input', () => {
      detectTriggers();
    });
    inputEl.addEventListener('click', () => {
      detectTriggers();
    });
    inputEl.addEventListener('keyup', () => {
      detectTriggers();
    });

    inputEl.addEventListener('keydown', (e) => {
      if (slashOpen && slashItems.length) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          slashSel = (slashSel + 1) % slashItems.length;
          renderSlashMenu();
          return;
        }
        if (e.key === 'ArrowUp') {
          e.preventDefault();
          slashSel = (slashSel - 1 + slashItems.length) % slashItems.length;
          renderSlashMenu();
          return;
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          closeMenus();
          return;
        }
      }

      if (atOpen && atItems.length) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          atSel = (atSel + 1) % atItems.length;
          renderAtMenu();
          return;
        }
        if (e.key === 'ArrowUp') {
          e.preventDefault();
          atSel = (atSel - 1 + atItems.length) % atItems.length;
          renderAtMenu();
          return;
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          closeMenus();
          return;
        }
      }

      if (e.key === 'Escape') {
        closeMenus();
      }

      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (slashOpen) {
          if (slashItems.length) execSlashSelected();
          return;
        }
        if (atOpen) {
          if (atItems.length) pickAtPath();
          return;
        }
        send();
      }

      if (e.key === 'Tab' && (slashOpen || atOpen)) {
        e.preventDefault();
        if (slashOpen && slashItems.length) execSlashSelected();
        else if (atOpen && atItems.length) pickAtPath();
      }
    });
  }

  function send() {
    if (sending) return;
    if (slashOpen || atOpen) return;
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    closeMenus();
    clearBridgeProgress();
    appendMessage('user', text);
    sending = true;
    setSendingUi(true);
    vscode.postMessage({
      type: 'send',
      text: text,
      sessionId: state.activeId,
      contextFiles: state.contextFiles,
    });
  }
})();
