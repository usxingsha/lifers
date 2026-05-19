/* Lifers Web UI v2 - 文件上传 + 图片 + 麦克风 + 终端 + 文件浏览 */

// ===== 状态管理 =====
const state = {
  gateUrl: localStorage.getItem('lifers_gate_url') || 'http://127.0.0.1:55555',
  maxChars: parseInt(localStorage.getItem('lifers_max_chars') || '2000'),
  temperature: parseFloat(localStorage.getItem('lifers_temperature') || '0.8'),
  stream: localStorage.getItem('lifers_stream') !== 'false',
  autoScroll: localStorage.getItem('lifers_auto_scroll') !== 'false',
  model: localStorage.getItem('lifers_model') || 'lifers',
  messages: [],
  isProcessing: false,
  startTime: 0,
  pendingFiles: [],
  pendingImages: [],
  isRecording: false,
  recognition: null,
};

function $(id) { return document.getElementById(id); }

const dom = {
  messages: $('messages'),
  welcome: $('welcome'),
  input: $('user-input'),
  sendBtn: $('btn-send'),
  connectionStatus: $('connection-status'),
  statusText: $('status-text'),
  charCount: $('char-count'),
  responseTime: $('response-time'),
  filePreviewBar: $('file-preview-bar'),
  dropOverlay: $('drop-overlay'),
  uploadStatus: $('upload-status'),
  termOutput: $('term-output'),
  termInput: $('term-input'),
  fileTree: $('file-tree'),
};

function init() {
  loadSettings();
  loadSession();
  setupEventListeners();
  setupDragDrop();
  setupFileInputs();
  checkConnection();
  refreshMonitor();
  document.getElementById('monitor-toggle').addEventListener('click', toggleMonitor);
  setInterval(checkConnection, 30000);
  setInterval(refreshMonitor, 30000);
}

function loadSettings() {
  $('cfg-gate-url').value = state.gateUrl;
  $('cfg-max-chars').value = state.maxChars;
  $('cfg-temperature').value = state.temperature;
  $('temp-value').textContent = state.temperature;
  $('cfg-stream').checked = state.stream;
  $('cfg-auto-scroll').checked = state.autoScroll;
}

function setupEventListeners() {
  dom.sendBtn.addEventListener('click', sendMessage);
  dom.input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  dom.input.addEventListener('input', function() {
    dom.charCount.textContent = dom.input.value.length + ' / 4000';
    dom.charCount.style.color = dom.input.value.length > 3800 ? 'var(--danger)' : 'var(--muted)';
    dom.input.style.height = 'auto';
    dom.input.style.height = Math.min(dom.input.scrollHeight, 150) + 'px';
  });

  $('btn-attach-file').addEventListener('click', function() { $('hidden-file-input').click(); });
  $('btn-attach-image').addEventListener('click', function() { $('hidden-image-input').click(); });
  $('btn-mic').addEventListener('click', toggleMicrophone);

  $('btn-clear').addEventListener('click', clearChat);
  $('btn-toggle-files').addEventListener('click', toggleFileBrowser);
  $('btn-toggle-terminal').addEventListener('click', toggleTerminal);
  $('btn-settings').addEventListener('click', toggleSettings);

  $('btn-refresh-files').addEventListener('click', loadFileTree);
  $('btn-upload-file').addEventListener('click', function() { $('hidden-file-input').click(); });

  $('btn-term-exec').addEventListener('click', execTerminal);
  dom.termInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); execTerminal(); }
  });
  $('btn-term-clear').addEventListener('click', function() { dom.termOutput.innerHTML = ''; });

  $('btn-save-settings').addEventListener('click', saveSettings);
  $('btn-close-settings').addEventListener('click', function() { $('settings-panel').classList.add('hidden'); });
  $('settings-overlay').addEventListener('click', function() { $('settings-panel').classList.add('hidden'); });
  $('cfg-temperature').addEventListener('input', function(e) {
    $('temp-value').textContent = parseFloat(e.target.value).toFixed(1);
  });

  var tips = document.querySelectorAll('.tip-card');
  for (var i = 0; i < tips.length; i++) {
    tips[i].addEventListener('click', function() {
      dom.input.value = this.dataset.msg;
      sendMessage();
    });
  }

  document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'k') { e.preventDefault(); clearChat(); }
    if (e.ctrlKey && e.key === 'm') { e.preventDefault(); toggleMonitor(); }
    if (e.ctrlKey && e.key === 'b') { e.preventDefault(); toggleFileBrowser(); }
    if (e.ctrlKey && e.key === '`') { e.preventDefault(); toggleTerminal(); }
  });
}

// ===== 拖拽上传 =====
function setupDragDrop() {
  var dropTarget = document.querySelector('.chat-main');
  var dragCounter = 0;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(function(evt) {
    dropTarget.addEventListener(evt, function(e) { e.preventDefault(); e.stopPropagation(); });
    document.body.addEventListener(evt, function(e) { e.preventDefault(); e.stopPropagation(); });
  });

  dropTarget.addEventListener('dragenter', function() {
    dragCounter++;
    dom.dropOverlay.classList.remove('hidden');
  });
  dropTarget.addEventListener('dragleave', function() {
    dragCounter--;
    if (dragCounter <= 0) { dragCounter = 0; dom.dropOverlay.classList.add('hidden'); }
  });
  dropTarget.addEventListener('drop', function(e) {
    dom.dropOverlay.classList.add('hidden');
    dragCounter = 0;
    processDroppedFiles(Array.from(e.dataTransfer.files || []));
  });
}

function processDroppedFiles(files) {
  var pending = [];
  for (var i = 0; i < files.length; i++) {
    pending.push(readFileAsync(files[i]));
  }
  Promise.all(pending).then(function(results) {
    for (var j = 0; j < results.length; j++) {
      var r = results[j];
      if (!r) continue;
      if (r.mime && r.mime.startsWith('image/')) {
        state.pendingImages.push(r);
      } else {
        state.pendingFiles.push(r);
      }
    }
    updateFilePreview();
  });
}

function readFileAsync(file) {
  if (file.size > 10 * 1024 * 1024) {
    dom.uploadStatus.textContent = file.name + ' 超过10MB限制';
    return Promise.resolve(null);
  }
  return new Promise(function(resolve) {
    var reader = new FileReader();
    reader.onload = function() {
      var dataUrl = reader.result;
      var base64 = dataUrl.split(',')[1] || '';
      var result = { name: file.name, data_base64: base64, mime: file.type, size: file.size };
      if (file.type.startsWith('image/')) {
        var img = new Image();
        img.onload = function() { result.width = img.width; result.height = img.height; resolve(result); };
        img.onerror = function() { resolve(result); };
        img.src = dataUrl;
      } else {
        resolve(result);
      }
    };
    reader.onerror = function() { resolve(null); };
    reader.readAsDataURL(file);
  });
}

function setupFileInputs() {
  $('hidden-file-input').addEventListener('change', function() {
    processDroppedFiles(Array.from(this.files));
    this.value = '';
  });
  $('hidden-image-input').addEventListener('change', function() {
    processDroppedFiles(Array.from(this.files));
    this.value = '';
  });
}

function updateFilePreview() {
  var all = state.pendingFiles.concat(state.pendingImages);
  if (all.length === 0) {
    dom.filePreviewBar.classList.add('hidden');
    dom.filePreviewBar.innerHTML = '';
    return;
  }
  dom.filePreviewBar.classList.remove('hidden');
  var html = '';
  var totalItems = all.length;
  var fileCount = state.pendingFiles.length;
  for (var i = 0; i < totalItems; i++) {
    var f = all[i];
    var isImg = f.mime && f.mime.startsWith('image/');
    var cls = isImg ? 'image-chip' : '';
    var icon = isImg ? '🖼' : '📄';
    var idx = i < fileCount ? i : i - fileCount;
    html += '<div class="file-preview-chip ' + cls + '"><span>' + icon + '</span><span class="chip-name">' + escHtml(f.name) + '</span><span class="chip-remove" data-idx="' + i + '">×</span></div>';
  }
  dom.filePreviewBar.innerHTML = html;
  var removes = dom.filePreviewBar.querySelectorAll('.chip-remove');
  for (var r = 0; r < removes.length; r++) {
    removes[r].addEventListener('click', function() {
      var idx = parseInt(this.dataset.idx);
      var fileCount2 = state.pendingFiles.length;
      if (idx >= fileCount2) {
        state.pendingImages.splice(idx - fileCount2, 1);
      } else {
        state.pendingFiles.splice(idx, 1);
      }
      updateFilePreview();
    });
  }
  dom.uploadStatus.textContent = all.length + ' 个文件';
}

// ===== 麦克风 =====
function toggleMicrophone() {
  if (state.isRecording) { stopRecording(); return; }
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { dom.uploadStatus.textContent = '浏览器不支持语音'; return; }
  state.recognition = new SR();
  state.recognition.lang = 'zh-CN';
  state.recognition.interimResults = true;
  state.recognition.continuous = true;
  state.recognition.onresult = function(e) {
    var t = '';
    for (var i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
    dom.input.value = t;
    dom.input.dispatchEvent(new Event('input'));
  };
  state.recognition.onerror = stopRecording;
  state.recognition.onend = stopRecording;
  state.recognition.start();
  state.isRecording = true;
  $('btn-mic').classList.add('recording');
  dom.uploadStatus.textContent = '🎤 录音中...';
}

function stopRecording() {
  if (state.recognition) { state.recognition.stop(); state.recognition = null; }
  state.isRecording = false;
  $('btn-mic').classList.remove('recording');
  dom.uploadStatus.textContent = '';
}

// ===== 连接检测 =====
function checkConnection() {
  fetch(state.gateUrl + '/health', { signal: AbortSignal.timeout(3000) })
    .then(function(r) { return r.json(); })
    .then(function(j) {
      if (j.ok) { dom.connectionStatus.className = 'status-dot online'; dom.statusText.textContent = '已连接'; }
    })
    .catch(function() {
      fetch('/health', { signal: AbortSignal.timeout(2000) })
        .then(function(r) {
          if (r.ok) { dom.connectionStatus.className = 'status-dot online'; dom.statusText.textContent = '已连接'; }
        })
        .catch(function() {
          dom.connectionStatus.className = 'status-dot offline';
          dom.statusText.textContent = '未连接';
        });
    });
}

// ===== 消息发送 =====
function sendMessage() {
  var text = dom.input.value.trim();
  var hasFiles = state.pendingFiles.length > 0 || state.pendingImages.length > 0;
  if ((!text && !hasFiles) || state.isProcessing) return;

  state.isProcessing = true;
  dom.sendBtn.disabled = true;

  var userText = text || '(发送了文件)';
  dom.input.value = '';
  dom.charCount.textContent = '0 / 4000';
  dom.input.style.height = 'auto';
  dom.welcome.classList.add('hidden');

  var attachHtml = '';
  for (var i = 0; i < state.pendingImages.length; i++) {
    var img = state.pendingImages[i];
    attachHtml += '<img class="msg-attach-img" src="data:' + img.mime + ';base64,' + img.data_base64 + '" alt="' + escHtml(img.name) + '" onclick="window.open(this.src)"/>';
  }
  for (var j = 0; j < state.pendingFiles.length; j++) {
    var f = state.pendingFiles[j];
    attachHtml += '<div class="msg-attach-file">📄 ' + escHtml(f.name) + ' (' + formatSize(f.size) + ')</div>';
  }
  var userContent = attachHtml ? '<div class="msg-attachments">' + attachHtml + '</div>' + escHtml(userText) : escHtml(userText);
  addMessage('user', userContent, false, true);

  state.messages.push({ role: 'user', content: userText });

  var pendingFiles = state.pendingFiles.slice();
  var pendingImages = state.pendingImages.slice();
  state.pendingFiles = [];
  state.pendingImages = [];
  updateFilePreview();

  var allUploads = pendingFiles.concat(pendingImages);
  var uploadPromise = Promise.resolve([]);
  if (allUploads.length > 0) {
    uploadPromise = fetch('/api/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: allUploads }),
      signal: AbortSignal.timeout(15000),
    }).then(function(r) { return r.json(); })
      .then(function(j) { return (j.ok && j.paths) ? j.paths : []; })
      .catch(function() { return []; });
  }

  var loadingId = addMessage('assistant', '...', true, false);
  state.startTime = performance.now();

  uploadPromise.then(function(contextFiles) {
    var body = JSON.stringify({
      text: userText,
      contextFiles: contextFiles,
      temperature: state.temperature,
      max_chars: state.maxChars,
    });

    if (state.stream) {
      return sendStreaming(body, loadingId);
    } else {
      return sendOnce(body, loadingId);
    }
  }).catch(function(e) {
    updateMessage(loadingId, '错误: ' + (e.message || '网络请求失败'));
  }).finally(function() {
    state.isProcessing = false;
    dom.sendBtn.disabled = false;
    dom.input.focus();
    saveSession();
  });
}

function sendOnce(body, loadingId) {
  return fetch(state.gateUrl + '/v1/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: body,
  }).then(function(r) { return r.json(); })
    .then(function(j) {
      var elapsed = Math.round(performance.now() - state.startTime);
      dom.responseTime.textContent = elapsed + 'ms';
      if (j.ok && j.text) {
        updateMessage(loadingId, j.text);
        state.messages.push({ role: 'assistant', content: j.text });
      } else {
        updateMessage(loadingId, '错误: ' + (j.error || '未返回内容'));
      }
    });
}

function sendStreaming(body, loadingId) {
  return fetch(state.gateUrl + '/v1/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: body,
  }).then(function(r) {
    if (!r.ok) return sendOnce(body, loadingId);
    var reader = r.body.getReader();
    var decoder = new TextDecoder('utf-8');
    var fullText = '';
    updateMessage(loadingId, '');
    function read() {
      return reader.read().then(function(result) {
        if (result.done) {
          var elapsed = Math.round(performance.now() - state.startTime);
          dom.responseTime.textContent = elapsed + 'ms';
          state.messages.push({ role: 'assistant', content: fullText });
          return;
        }
        fullText += decoder.decode(result.value, { stream: true });
        updateMessage(loadingId, fullText);
        return read();
      });
    }
    return read();
  });
}

	// ===== 简单 Markdown 渲染 =====
	function renderMarkdown(text) {
	  if (!text) return '';
	  var html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
	  // code blocks
	  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, function(m, lang, code) {
	    return '<pre><code class="language-' + (lang || 'plaintext') + '">' + code.trim() + '</code></pre>';
	  });
	  // inline code
	  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
	  // bold, italic
	  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
	  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
	  // links
	  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
	  // lists
	  html = html.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
	  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
	  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
	  // headings
	  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
	  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
	  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
	  // hr
	  html = html.replace(/^---+$/gm, '<hr>');
	  // paragraphs
	  html = html.replace(/\n\n/g, '</p><p>');
	  html = '<p>' + html + '</p>';
	  html = html.replace(/<p>\s*<\/p>/g, '');
	  return html;
	}

	// ===== 消息渲染 =====
	function addMessage(role, content, isLoading, isHtml) {
	  var msgDiv = document.createElement('div');
	  msgDiv.className = 'message ' + role + (isLoading ? ' loading' : '');
	  var id = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5);
	  msgDiv.id = id;

	  var avatar = document.createElement('div');
	  avatar.className = 'msg-avatar';
	  avatar.textContent = role === 'user' ? '👤' : '🤖';

	  var body = document.createElement('div');
	  body.className = 'msg-body';

	  var header = document.createElement('div');
	  header.className = 'msg-header';
	  var roleName = role === 'user' ? '你' : 'Lifers';
	  var time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
	  header.innerHTML = '<span class="msg-role">' + roleName + '</span><span class="msg-time">' + time + '</span>';

	  // copy button for assistant messages
	  if (role === 'assistant' && !isLoading) {
	    var actions = document.createElement('span');
	    actions.className = 'msg-actions';
	    actions.innerHTML = '<button class="msg-action-btn" title="复制" onclick="copyMessage(\'' + id + '\')">📋</button>';
	    header.appendChild(actions);
	  }

	  var contentDiv = document.createElement('div');
	  contentDiv.className = 'msg-content';
	  if (isHtml) {
	    contentDiv.innerHTML = content;
	  } else if (role === 'assistant') {
	    contentDiv.innerHTML = renderMarkdown(content);
	  } else {
	    contentDiv.textContent = content;
	  }

	  body.appendChild(header);
	  body.appendChild(contentDiv);
	  msgDiv.appendChild(avatar);
	  msgDiv.appendChild(body);
	  dom.messages.appendChild(msgDiv);
	  if (state.autoScroll) msgDiv.scrollIntoView({ behavior: 'smooth' });
	  return id;
	}

	function updateMessage(id, content) {
	  var msg = document.getElementById(id);
	  if (!msg) return;
	  var cd = msg.querySelector('.msg-content');
	  if (cd) cd.innerHTML = renderMarkdown(content);
	  if (state.autoScroll) msg.scrollIntoView({ behavior: 'smooth' });
	}

	// 复制消息
	function copyMessage(id) {
	  var msg = document.getElementById(id);
	  if (!msg) return;
	  var cd = msg.querySelector('.msg-content');
	  if (!cd) return;
	  navigator.clipboard.writeText(cd.textContent).then(function() {}).catch(function() {});
	}

	// ===== 会话持久化 =====
	function saveSession() {
	  try {
	    var toSave = state.messages.slice(-50);
	    localStorage.setItem('lifers_session', JSON.stringify(toSave));
	  } catch(e) {}
	}

	function loadSession() {
	  try {
	    var saved = localStorage.getItem('lifers_session');
	    if (saved) {
	      var msgs = JSON.parse(saved);
	      for (var i = 0; i < msgs.length; i++) {
	        var m = msgs[i];
	        addMessage(m.role, m.content, false, m.role === 'user');
	        state.messages.push(m);
	      }
	    }
	  } catch(e) {}
	}

	// ===== 停止生成 =====
	function stopGeneration() {
	  state.isProcessing = false;
	  dom.sendBtn.disabled = false;
	}

// ===== 文件浏览器 =====
var fileBrowserVisible = false;
function toggleFileBrowser() {
  fileBrowserVisible = !fileBrowserVisible;
  var fb = $('file-browser');
  if (fileBrowserVisible) {
    fb.classList.remove('hidden');
    fb.style.width = '280px';
    fb.style.minWidth = '200px';
    loadFileTree();
  } else {
    fb.classList.add('hidden');
    fb.style.width = '0';
    fb.style.minWidth = '0';
  }
}

function loadFileTree() {
  dom.fileTree.innerHTML = '<div class="file-tree-loading">加载中...</div>';
  fetch('/api/files', { signal: AbortSignal.timeout(5000) })
    .then(function(r) { return r.json(); })
    .then(function(j) {
      if (j.ok && j.tree) {
        renderFileTree(j.tree, dom.fileTree, 0);
      } else {
        dom.fileTree.innerHTML = '<div class="file-tree-loading">无法加载</div>';
      }
    })
    .catch(function() {
      dom.fileTree.innerHTML = '<div class="file-tree-loading">服务不可用</div>';
    });
}

function renderFileTree(items, container, depth) {
  container.innerHTML = '';
  for (var i = 0; i < items.length; i++) {
    var item = items[i];
    var div = document.createElement('div');
    div.className = 'file-tree-item ' + item.type;
    var indent = '';
    for (var d = 0; d < depth; d++) indent += '  ';
    var icon = item.type === 'dir' ? '📁' : '📄';
    div.innerHTML = '<span class="tree-icon">' + icon + '</span>' + indent + escHtml(item.name);
    if (item.type === 'file') {
      div.addEventListener('click', function(path) {
        return function() {
          fetch('/api/files?path=' + encodeURIComponent(path), { signal: AbortSignal.timeout(5000) })
            .then(function(r) { return r.json(); })
            .then(function(j) {
              if (j.ok && j.content) {
                var preview = j.content.substring(0, 2000);
                dom.input.value = '文件: ' + path + '\n```\n' + preview + '\n```\n\n' + dom.input.value;
                dom.input.dispatchEvent(new Event('input'));
                dom.input.focus();
              }
            });
        };
      }(item.path));
    }
    container.appendChild(div);
    if (item.children && item.type === 'dir') {
      renderFileTree(item.children, container, depth + 1);
    }
  }
}

// ===== 终端 =====
var terminalVisible = false;
function toggleTerminal() {
  terminalVisible = !terminalVisible;
  var tp = $('terminal-panel');
  if (terminalVisible) {
    tp.classList.remove('hidden');
    tp.style.width = '360px';
    tp.style.minWidth = '280px';
    setTimeout(function() { dom.termInput.focus(); }, 100);
  } else {
    tp.classList.add('hidden');
    tp.style.width = '0';
    tp.style.minWidth = '0';
  }
}

function execTerminal() {
  var cmd = dom.termInput.value.trim();
  if (!cmd) return;
  var shell = $('term-shell').value;
  dom.termOutput.innerHTML += '<div class="term-line"><span class="term-cmd">$ ' + escHtml(cmd) + '</span></div>';
  dom.termInput.value = '';
  dom.termOutput.scrollTop = dom.termOutput.scrollHeight;

  fetch('/api/exec', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cmd: cmd, shell: shell }),
    signal: AbortSignal.timeout(30000),
  }).then(function(r) { return r.json(); })
    .then(function(j) {
      if (j.stdout) dom.termOutput.innerHTML += '<div class="term-line"><span class="term-out">' + escHtml(j.stdout) + '</span></div>';
      if (j.stderr) dom.termOutput.innerHTML += '<div class="term-line"><span class="term-err">' + escHtml(j.stderr) + '</span></div>';
      if (j.exit_code !== 0) dom.termOutput.innerHTML += '<div class="term-line"><span class="term-err">退出码: ' + j.exit_code + '</span></div>';
    })
    .catch(function(e) {
      dom.termOutput.innerHTML += '<div class="term-line"><span class="term-err">错误: ' + escHtml(e.message) + '</span></div>';
    })
    .finally(function() {
      dom.termOutput.scrollTop = dom.termOutput.scrollHeight;
    });
}

// ===== 设置 =====
function toggleSettings() { $('settings-panel').classList.toggle('hidden'); }

function saveSettings() {
  state.gateUrl = $('cfg-gate-url').value;
  state.maxChars = parseInt($('cfg-max-chars').value) || 2000;
  state.temperature = parseFloat($('cfg-temperature').value) || 0.8;
  state.stream = $('cfg-stream').checked;
  state.autoScroll = $('cfg-auto-scroll').checked;
  localStorage.setItem('lifers_gate_url', state.gateUrl);
  localStorage.setItem('lifers_max_chars', state.maxChars);
  localStorage.setItem('lifers_temperature', state.temperature);
  localStorage.setItem('lifers_stream', state.stream);
  localStorage.setItem('lifers_auto_scroll', state.autoScroll);
  $('settings-panel').classList.add('hidden');
  checkConnection();
}

// ===== 聊天管理 =====
function clearChat() {
  dom.messages.innerHTML = '';
  state.messages = [];
  dom.welcome.classList.remove('hidden');
  dom.responseTime.textContent = '';
  state.pendingFiles = [];
  state.pendingImages = [];
  updateFilePreview();
}

// ===== 训练监控 =====
function toggleMonitor() {
  var panel = $('training-monitor');
  if (panel.classList.contains('hidden')) {
    panel.classList.remove('hidden');
    panel.classList.remove('collapsed');
    refreshMonitor();
  } else if (panel.classList.contains('collapsed')) {
    panel.classList.remove('collapsed');
  } else {
    panel.classList.add('collapsed');
    setTimeout(function() {
      if (panel.classList.contains('collapsed')) panel.classList.add('hidden');
    }, 3000);
  }
}

function refreshMonitor() {
  fetch('/api/monitor', { signal: AbortSignal.timeout(5000) })
    .then(function(r) { return r.json(); })
    .then(function(j) {
      if (!j.ok || !j.monitor) return;
      var m = j.monitor;
      var de = m.deep_escalate || {};
      var se = $('monitor-summary');
      if (de.active) {
        se.textContent = 'D=' + de.d_model + ' T' + de.ramp_iter + '/' + de.ramp_max + ' S' + de.sgd_step + '/' + de.sgd_total;
        se.className = 'monitor-summary';
      } else {
        se.textContent = '等待训练';
        se.className = 'monitor-summary warning';
      }
      var deEl = $('monitor-deep');
      if (de.active) {
        var pct = de.sgd_total ? Math.round((de.sgd_step / de.sgd_total) * 100) : '?';
        deEl.innerHTML = '<div class="monitor-row"><span class="key">D=' + de.d_model + ' V=' + de.vocab + '</span><span class="val good">' + pct + '%</span></div><div class="monitor-row"><span class="key">总进度</span><span class="val">' + de.overall_pct + '%</span></div>';
      }
    });
}

// ===== 工具函数 =====
function escHtml(s) {
  var div = document.createElement('div');
  div.textContent = s || '';
  return div.innerHTML;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + 'B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'KB';
  return (bytes / (1024 * 1024)).toFixed(1) + 'MB';
}

document.addEventListener('DOMContentLoaded', init);
