// @ts-check
const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const cp = require('child_process');
const crypto = require('crypto');
const { LifersSessionTreeProvider } = require('./session_tree.js');

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  const controller = new LifersAgentsController(context.extensionUri, context);

  const sessionTreeProvider = new LifersSessionTreeProvider(controller, getBrainRoot);
  controller.setSessionTreeProvider(sessionTreeProvider);
  context.subscriptions.push(vscode.window.registerTreeDataProvider('lifers.sessionTree', sessionTreeProvider));
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider('lifers.sessionTreeExplorer', sessionTreeProvider)
  );
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider('lifers.sessionTreeBottom', sessionTreeProvider)
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(LifersLauncherViewProvider.viewType, new LifersLauncherViewProvider(controller), {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(LifersControlPanelViewProvider.viewType, new LifersControlPanelViewProvider(controller), {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.activateSession', (sessionId) => {
      controller.activateSessionById(sessionId);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.newSession', () => controller.newSession())
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.addContext', () => controller.addContextFiles())
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.addContextFolder', () => controller.addContextFolder())
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.openSettings', () =>
      vscode.commands.executeCommand('workbench.action.openSettings', 'lifers')
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.openMarketplace', () =>
      vscode.env.openExternal(vscode.Uri.parse('https://marketplace.visualstudio.com/search?term=lifers&target=VSCode'))
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.openChat', () => controller.ensureChatPanel())
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.focusSessions', () => controller.focusSessionsPanel())
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.openTotalControl', () => controller.openTotalControlPanel())
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('lifers.agents.show', async () => {
      await controller.ensureChatPanel();
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidChangeWorkspaceFolders(() => {
      controller.resetSessionBootstrapFlag();
      const r = getBrainRoot();
      if (r) {
        setTimeout(() => controller.bootstrapDefaultSessionIfNeeded(r), 400);
      }
    })
  );

  const conf = vscode.workspace.getConfiguration('lifers');
  if (conf.get('openChatOnStartup') !== false) {
    setTimeout(() => controller.ensureChatPanel(), 600);
  }

  if (conf.get('openSessionSidebarOnStartup') !== false) {
    setTimeout(() => controller.revealSessionSidebar(), 480);
  }

  if (conf.get('openTotalPanelOnStartup') !== false) {
    setTimeout(() => controller.openTotalControlPanel(), 720);
  }

  setTimeout(() => {
    const r = getBrainRoot();
    if (r) {
      controller.bootstrapDefaultSessionIfNeeded(r);
    }
  }, 900);

}

function legacyHistoryPath(root) {
  return path.join(root, '.lifers', 'agents_history.json');
}

function legacyHistoryPathOld(root) {
  return path.join(root, '.lifers_ui', 'agents_history.json');
}

/**
 * Stable key for the same project on disk (Windows drive letter normalized).
 * @param {string} root
 */
function brainKeyForStorage(root) {
  let p = path.resolve(root);
  if (process.platform === 'win32' && p.length >= 2) {
    p = p.charAt(0).toLowerCase() + p.slice(1);
  }
  return p;
}

function hasAgentBridge(dir) {
  try {
    return fs.existsSync(path.join(dir, 'scripts', 'agent_bridge_once.py'));
  } catch {
    return false;
  }
}

function resolveBrainFolder(wsRoot) {
  const abs = path.resolve(wsRoot);
  if (hasAgentBridge(abs)) return abs;
  const nestedDirs = ['lifers', 'lifers_brain'];
  for (const seg of nestedDirs) {
    const nested = path.join(abs, seg);
    if (hasAgentBridge(nested)) return nested;
  }
  // 仅打开 rs0 等子目录时：在父目录旁查找 lifers_brain（与 lifers.code-workspace 多根布局一致）
  const parent = path.dirname(abs);
  if (parent && parent !== abs) {
    for (const seg of nestedDirs) {
      const sib = path.join(parent, seg);
      if (hasAgentBridge(sib)) return sib;
    }
    if (hasAgentBridge(parent)) return parent;
  }
  let cur = abs;
  for (let i = 0; i < 10; i++) {
    const p = path.dirname(cur);
    if (p === cur) break;
    for (const seg of nestedDirs) {
      const sib = path.join(p, seg);
      if (hasAgentBridge(sib)) return sib;
    }
    if (hasAgentBridge(p)) return p;
    cur = p;
  }
  for (const seg of nestedDirs) {
    const nested = path.join(abs, seg);
    try {
      if (fs.existsSync(nested) && fs.statSync(nested).isDirectory()) return nested;
    } catch {
      /* ignore */
    }
  }
  return abs;
}

/**
 * Expand ${workspaceFolder} / ${workspaceFolder:name} in a setting string (extension API often returns these verbatim).
 * @param {string} raw
 * @param {string} brainRoot
 */
function expandWorkspaceFolderVars(raw, brainRoot) {
  if (!raw || typeof raw !== 'string') return raw;
  let s = raw;
  const folders = vscode.workspace.workspaceFolders || [];
  s = s.replace(/\$\{workspaceFolder\}/gi, brainRoot);
  s = s.replace(/\$\{workspaceFolder:([^}]+)\}/gi, (_, name) => {
    const f = folders.find((x) => x.name === name);
    return f ? f.uri.fsPath : brainRoot;
  });
  return s;
}

/**
 * Resolve lifers.pythonPath for spawn: expand vars, map Windows venv layout to Unix, fall back if missing.
 * @param {unknown} confPython
 * @param {string} brainRoot
 */
function resolveLifersPythonExecutable(confPython, brainRoot) {
  let py = (confPython != null && String(confPython).trim()) || 'python';
  py = expandWorkspaceFolderVars(py, brainRoot);
  py = path.normalize(py);

  const venvWin = path.join(brainRoot, '.venv', 'Scripts', 'python.exe');
  const venvUnix3 = path.join(brainRoot, '.venv', 'bin', 'python3');
  const venvUnix = path.join(brainRoot, '.venv', 'bin', 'python');

  if (process.platform !== 'win32') {
    const lower = py.toLowerCase();
    if (lower.includes('scripts\\python.exe') || lower.includes('scripts/python.exe') || lower.endsWith('python.exe')) {
      if (fs.existsSync(venvUnix3)) return venvUnix3;
      if (fs.existsSync(venvUnix)) return venvUnix;
      return 'python3';
    }
  }

  if (py.includes('${')) {
    if (fs.existsSync(venvWin)) return venvWin;
    if (fs.existsSync(venvUnix3)) return venvUnix3;
    if (fs.existsSync(venvUnix)) return venvUnix;
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  if (!fs.existsSync(py) && (py === 'python' || py === 'python3')) {
    if (fs.existsSync(venvWin)) return venvWin;
    if (fs.existsSync(venvUnix3)) return venvUnix3;
    if (fs.existsSync(venvUnix)) return venvUnix;
  }

  return py;
}

function getBrainRoot() {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || !folders.length) return null;
  const firstPath = folders[0].uri.fsPath;
  const conf = vscode.workspace.getConfiguration('lifers');
  const override = String(conf.get('brainPathOverride') || '').trim();
  if (override) {
    const o = expandWorkspaceFolderVars(override, firstPath);
    const np = path.normalize(o);
    if (hasAgentBridge(np)) return np;
  }
  const prefer = (nm) => {
    const n = String(nm || '').toLowerCase();
    return n === 'lifers_brain' || n === 'lifers' || n === 'lifers brain' || n === 'rs';
  };
  if (folders.length > 1) {
    for (const f of folders) {
      if (!prefer(f.name)) continue;
      const r = resolveBrainFolder(f.uri.fsPath);
      if (hasAgentBridge(r)) return r;
    }
  }
  if (folders.length === 1) return resolveBrainFolder(folders[0].uri.fsPath);
  for (const f of folders) {
    const r = resolveBrainFolder(f.uri.fsPath);
    if (hasAgentBridge(r)) return r;
  }
  return resolveBrainFolder(folders[0].uri.fsPath);
}

function getNonce() {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

/**
 * @param {string} cmd
 * @param {{ cwd?: string, timeout?: number }} [opts]
 */
function execShell(cmd, opts) {
  return new Promise((resolve) => {
    cp.exec(
      cmd,
      {
        encoding: 'utf8',
        maxBuffer: 24 * 1024 * 1024,
        timeout: opts?.timeout ?? 180000,
        cwd: opts?.cwd,
      },
      (err, stdout, stderr) => {
        resolve({
          code: typeof err?.code === 'number' ? err.code : err ? 1 : 0,
          stdout: stdout || '',
          stderr: stderr || '',
        });
      }
    );
  });
}

function isPathInside(root, candidate) {
  const r = path.resolve(root);
  const c = path.resolve(candidate);
  return c === r || c.startsWith(r + path.sep);
}

/**
 * @param {string} brainRoot
 * @param {string} absDir
 * @param {{ maxFiles: number; maxDepth: number; skipDirs: Set<string> }} opts
 * @returns {string[]}
 */
function collectFilesUnderFolder(brainRoot, absDir, opts) {
  const rels = [];
  function walk(dir, depth) {
    if (rels.length >= opts.maxFiles) return;
    if (depth > opts.maxDepth) return;
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      if (rels.length >= opts.maxFiles) break;
      const full = path.join(dir, e.name);
      if (!isPathInside(brainRoot, full)) continue;
      if (e.isDirectory()) {
        const low = e.name.toLowerCase();
        if (opts.skipDirs.has(low)) continue;
        walk(full, depth + 1);
      } else {
        const rel = path.relative(brainRoot, full).replace(/\\/g, '/');
        if (rel && !rel.startsWith('..')) rels.push(rel);
      }
    }
  }
  walk(absDir, 0);
  return rels;
}

/**
 * @param {string[] | undefined} existing
 * @param {string[]} additions
 * @param {number} maxTotal
 */
function mergeContextPaths(existing, additions, maxTotal) {
  const set = new Set([...(existing || [])]);
  for (const a of additions) {
    if (a) set.add(a.replace(/\\/g, '/'));
  }
  return Array.from(set).slice(0, maxTotal);
}

/**
 * Bridge prints one JSON object; tolerate stray stderr lines or leading noise.
 * @param {string} out
 * @param {string} stderr
 * @returns {any}
 */
function parseBridgeStdout(out, stderr) {
  const t = (out || '').trim();
  if (!t) {
    const se = (stderr || '').trim();
    throw new Error(
      se
        ? `Bridge 无标准输出。stderr（节选）：${se.slice(0, 1200)}`
        : 'Bridge 无标准输出（请检查 Python 路径与 lifers_brain）'
    );
  }
  try {
    return JSON.parse(t);
  } catch (e1) {
    const lines = t.split(/\r?\n/).filter((x) => x.trim());
    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        return JSON.parse(lines[i].trim());
      } catch {
        /* next */
      }
    }
    const m = t.match(/\{[\s\S]*\}\s*$/);
    if (m) {
      try {
        return JSON.parse(m[0]);
      } catch {
        /* fall through */
      }
    }
    const se = (stderr || '').trim();
    throw new Error(
      `无法解析 Bridge 返回的 JSON：${e1 instanceof Error ? e1.message : String(e1)}` +
        (se ? ` · stderr（节选）：${se.slice(0, 600)}` : '')
    );
  }
}

class LifersLauncherViewProvider {
  static viewType = 'lifers.agentsLauncherView';
  /** @param {LifersAgentsController} controller */
  constructor(controller) {
    this._c = controller;
  }
  resolveWebviewView(webviewView) {
    this._c.setLauncherView(webviewView);
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._c.extensionUri],
    };
    webviewView.webview.html = this._c.getLauncherHtml(webviewView.webview);
    webviewView.webview.onDidReceiveMessage((msg) => this._c.handleMessage(msg, 'launcher'));
  }
}

class LifersControlPanelViewProvider {
  static viewType = 'lifers.controlPanel';
  /** @param {LifersAgentsController} controller */
  constructor(controller) {
    this._c = controller;
  }
  resolveWebviewView(webviewView) {
    this._c.setControlPanelView(webviewView);
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._c.extensionUri],
    };
    webviewView.webview.html = this._c.getControlPanelHtml(webviewView.webview);
    webviewView.webview.onDidReceiveMessage((msg) => this._c.handleControlPanelMessage(msg));
  }
}

class LifersAgentsController {
  /**
   * @param {vscode.Uri} extensionUri
   * @param {vscode.ExtensionContext} context
   */
  constructor(extensionUri, context) {
    this.extensionUri = extensionUri;
    this._context = context;
    /** @type {vscode.WebviewView | undefined} */
    this._launcherView = undefined;
    /** @type {vscode.WebviewPanel | undefined} */
    this._chatPanel = undefined;
    /** @type {vscode.WebviewView | undefined} */
    this._controlView = undefined;
    /** One-shot: auto-create first session when history empty (after webviews ready). */
    this._defaultSessionEnsured = false;
    /** @type {import('./session_tree.js').LifersSessionTreeProvider | undefined} */
    this._sessionTreeProvider = undefined;
  }

  /** @param {import('./session_tree.js').LifersSessionTreeProvider | undefined} p */
  setSessionTreeProvider(p) {
    this._sessionTreeProvider = p;
  }

  /**
   * @param {string | undefined} sessionId
   */
  activateSessionById(sessionId) {
    const root = getBrainRoot();
    if (!root || !sessionId) return;
    const data = this._readHistory(root);
    data.activeSessionId = sessionId;
    this._writeHistory(root, data);
    this._postState(root);
  }

  /** 左侧活动栏「Lifers」容器内「会话建立」树（与 Agents 同栏，避免 Secondary Side Bar 不可见）。 */
  async revealSessionSidebar() {
    try {
      await vscode.commands.executeCommand('workbench.view.extension.lifers-agents');
    } catch {
      /* ignore */
    }
    try {
      await vscode.commands.executeCommand('workbench.actions.treeView.lifers.sessionTree.focus');
    } catch {
      /* 部分宿主无该命令 */
    }
    /* 镜像视图：资源管理器内「会话建立」，活动栏 Lifers 被隐藏或未装时仍可用 */
    try {
      await vscode.commands.executeCommand('workbench.view.explorer');
    } catch {
      /* ignore */
    }
    try {
      await vscode.commands.executeCommand('workbench.actions.treeView.lifers.sessionTreeExplorer.focus');
    } catch {
      /* ignore */
    }
  }

  /** @param {vscode.WebviewView | undefined} v */
  setLauncherView(v) {
    this._launcherView = v;
  }

  /** @param {vscode.WebviewView | undefined} v */
  setControlPanelView(v) {
    this._controlView = v;
  }

  async openTotalControlPanel() {
    try {
      await vscode.commands.executeCommand('workbench.action.focusPanel');
    } catch {
      /* ignore */
    }
    try {
      await vscode.commands.executeCommand('workbench.view.extension.lifers-total-panel');
    } catch {
      /* 部分宿主面板 id 不同，忽略 */
    }
    try {
      await vscode.commands.executeCommand('workbench.actions.treeView.lifers.sessionTreeBottom.focus');
    } catch {
      /* ignore */
    }
  }

  /**
   * Durable path: globalStorage …/lifers/sessions/。工作区镜像：.lifers/agents_history.json；旧版 .lifers_ui 可读入并迁移。
   * @param {string} root
   */
  _primaryHistoryPath(root) {
    const base = this._context.globalStorageUri?.fsPath;
    if (!base) {
      return legacyHistoryPath(root);
    }
    const key = crypto.createHash('sha256').update(brainKeyForStorage(root)).digest('hex').slice(0, 24);
    return path.join(base, 'lifers', 'sessions', `history_${key}.json`);
  }

  /**
   * @param {string} root
   */
  _readHistory(root) {
    const primary = this._primaryHistoryPath(root);
    const legacy = legacyHistoryPath(root);
    const parseFile = (p) => {
      const raw = fs.readFileSync(p, 'utf8');
      const data = JSON.parse(raw);
      if (!data.sessions) data.sessions = [];
      return data;
    };
    if (fs.existsSync(primary)) {
      try {
        return parseFile(primary);
      } catch (e) {
        console.warn('[lifers-agents-ui] corrupt primary history, trying legacy', e);
      }
    }
    if (fs.existsSync(legacy) && primary !== legacy) {
      try {
        const data = parseFile(legacy);
        this._writeHistory(root, data);
        console.info('[lifers-agents-ui] migrated workspace mirror .lifers to primary store');
        return data;
      } catch (e) {
        console.warn('[lifers-agents-ui] corrupt legacy history', e);
      }
    }
    const legacyOld = legacyHistoryPathOld(root);
    if (fs.existsSync(legacyOld)) {
      try {
        const data = parseFile(legacyOld);
        this._writeHistory(root, data);
        console.info('[lifers-agents-ui] migrated session history from .lifers_ui to .lifers / primary store');
        return data;
      } catch (e) {
        console.warn('[lifers-agents-ui] corrupt .lifers_ui history', e);
      }
    }
    if (primary === legacy && fs.existsSync(legacy)) {
      try {
        return parseFile(legacy);
      } catch {
        /* empty below */
      }
    }
    return { sessions: [], activeSessionId: null };
  }

  /**
   * @param {string} root
   * @param {{ sessions: any[]; activeSessionId: string | null }} data
   */
  _writeHistory(root, data) {
    const json = JSON.stringify(data, null, 2);
    const primary = this._primaryHistoryPath(root);
    const legacy = legacyHistoryPath(root);
    const written = new Set();
    const writeOne = (p, label) => {
      if (written.has(p)) return true;
      try {
        fs.mkdirSync(path.dirname(p), { recursive: true });
        fs.writeFileSync(p, json, 'utf8');
        written.add(p);
        return true;
      } catch (e) {
        console.warn(`[lifers-agents-ui] history write failed (${label})`, e);
        return false;
      }
    };
    const okP = writeOne(primary, 'primary');
    if (primary !== legacy) {
      writeOne(legacy, 'legacy');
    }
    if (!okP && primary !== legacy && !writeOne(legacy, 'legacy-fallback')) {
      vscode.window.showErrorMessage(
        'Lifers: 会话无法保存到磁盘，请检查扩展目录与工作区权限。 | Cannot persist session history.'
      );
    }
  }

  async ensureChatPanel() {
    if (this._chatPanel) {
      this._chatPanel.reveal(vscode.ViewColumn.One, false);
      const root = getBrainRoot();
      if (root) this._postState(root);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      'lifersAgentsChat',
      'Agents Chat',
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [this.extensionUri],
      }
    );
    panel.webview.html = this.getChatHtml(panel.webview);
    panel.webview.onDidReceiveMessage((msg) => this.handleMessage(msg, 'chat'));
    panel.onDidDispose(() => {
      this._chatPanel = undefined;
    });
    this._chatPanel = panel;
    const root = getBrainRoot();
    if (root) this._postState(root);
  }

  /** 聚焦左侧 Lifers 栏中的「会话建立」树视图。 */
  async focusSessionsPanel() {
    await this.revealSessionSidebar();
  }

  /**
   * @param {any} msg
   * @param {'chat' | 'launcher'} source
   */
  async handleMessage(msg, source) {
    const root = getBrainRoot();

    if (msg.type === 'openExternal') {
      const u = String(msg.url || '').trim();
      if (u.startsWith('http://') || u.startsWith('https://')) {
        vscode.env.openExternal(vscode.Uri.parse(u));
      }
      return;
    }
    if (msg.type === 'openSettings') {
      vscode.commands.executeCommand('workbench.action.openSettings', 'lifers');
      return;
    }
    if (msg.type === 'openChat') {
      await this.ensureChatPanel();
      return;
    }
    if (msg.type === 'focusSessions') {
      await this.focusSessionsPanel();
      return;
    }

    if (msg.type === 'ready') {
      if (!root) {
        this._postEmptyState();
        vscode.window.showWarningMessage(
          '请先打开工作区：lifers.code-workspace 或 lifers_brain。 | Open lifers.code-workspace or lifers_brain folder.'
        );
      } else {
        this._postState(root);
        this.bootstrapDefaultSessionIfNeeded(root);
      }
      return;
    }

    if (!root) {
      vscode.window.showWarningMessage('请先打开工作区（lifers.code-workspace 或 lifers_brain 文件夹）。 | Open lifers.code-workspace or lifers_brain folder.');
      return;
    }

    if (msg.type === 'newSession') {
      this.newSession();
      return;
    }
    if (msg.type === 'selectSession') {
      this.activateSessionById(msg.sessionId);
      return;
    }
    if (msg.type === 'persistRequest') {
      this._postState(root);
      return;
    }
    if (msg.type === 'send') {
      if (source !== 'chat') return;
      await this._handleSend(root, msg.text, msg.sessionId, msg.contextFiles || []);
      return;
    }
    if (msg.type === 'openReadme') {
      const readme = path.join(root, 'README.md');
      if (fs.existsSync(readme)) {
        const doc = await vscode.workspace.openTextDocument(readme);
        await vscode.window.showTextDocument(doc, { preview: true });
      } else {
        vscode.window.showInformationMessage('工作区根目录无 README.md');
      }
      return;
    }
    if (msg.type === 'pickContextFiles') {
      await this.addContextFiles();
      return;
    }
    if (msg.type === 'pickContextFolder') {
      await this.addContextFolder();
      return;
    }
    if (msg.type === 'clearContext') {
      const data = this._readHistory(root);
      const sid = data.activeSessionId;
      const s = data.sessions.find((x) => x.id === sid);
      if (s) {
        s.contextFiles = [];
        s.updated = Date.now();
        this._writeHistory(root, data);
        this._chatPanel?.webview.postMessage({ type: 'context', files: [] });
        this._postState(root);
      }
      return;
    }
    if (msg.type === 'setLifersModel') {
      const raw = String(msg.model || '').toLowerCase();
      const m = raw === 'markov' ? 'markov' : raw === 'transformer' ? 'transformer' : 'lifers';
      await vscode.workspace.getConfiguration('lifers').update('model', m, vscode.ConfigurationTarget.Workspace);
      this._postConfigToChat();
      return;
    }
    if (msg.type === 'setLifersSandbox') {
      await vscode.workspace
        .getConfiguration('lifers')
        .update('sandbox', !!msg.sandbox, vscode.ConfigurationTarget.Workspace);
      this._postConfigToChat();
      return;
    }
    if (msg.type === 'fileCompletion') {
      const q = String(msg.query ?? '');
      const rid = msg.requestId;
      const files = this._pathsForFileCompletion(root, q);
      this._chatPanel?.webview.postMessage({ type: 'fileCompletionResult', requestId: rid, files });
      return;
    }
    if (msg.type === 'addContextRelPaths') {
      const rels = Array.isArray(msg.paths)
        ? msg.paths.map((p) => String(p).replace(/\\/g, '/').trim()).filter(Boolean)
        : [];
      if (!rels.length) return;
      const opts = this._getContextOpts();
      let data = this._readHistory(root);
      let s = data.sessions.find((x) => x.id === data.activeSessionId);
      if (!s) {
        this.newSession();
        data = this._readHistory(root);
        s = data.sessions.find((x) => x.id === data.activeSessionId);
      }
      if (!s) return;
      s.contextFiles = mergeContextPaths(s.contextFiles, rels, opts.maxFiles);
      s.updated = Date.now();
      this._writeHistory(root, data);
      this._chatPanel?.webview.postMessage({ type: 'context', files: s.contextFiles });
      this._postState(root);
      return;
    }
  }

  /**
   * Enumerate workspace-relative paths under brain root for @ completion (capped).
   * @param {string} root
   * @param {string} query
   * @returns {string[]}
   */
  _pathsForFileCompletion(root, query) {
    const opts = this._getContextOpts();
    const depthCap = Math.min(Math.max(opts.folderDepth + 3, 6), 14);
    const all = collectFilesUnderFolder(root, root, {
      maxFiles: 8000,
      maxDepth: depthCap,
      skipDirs: opts.skipDirs,
    });
    const q = query.trim().toLowerCase();
    let hits = q ? all.filter((p) => p.toLowerCase().includes(q)) : all.slice();
    hits.sort((a, b) => {
      const al = a.toLowerCase();
      const bl = b.toLowerCase();
      if (q) {
        const as = al.startsWith(q) ? 0 : al.includes('/' + q) ? 1 : 2;
        const bs = bl.startsWith(q) ? 0 : bl.includes('/' + q) ? 1 : 2;
        if (as !== bs) return as - bs;
      }
      return al.localeCompare(bl);
    });
    return hits.slice(0, 50);
  }

  _lifersConfigPayload() {
    const conf = vscode.workspace.getConfiguration('lifers');
    const m = conf.get('model');
    const model = m === 'markov' ? 'markov' : m === 'transformer' ? 'transformer' : 'lifers';
    return {
      model,
      sandbox: conf.get('sandbox') !== false,
    };
  }

  _postConfigToChat() {
    const conf = vscode.workspace.getConfiguration('lifers');
    const m = conf.get('model');
    const model = m === 'markov' ? 'markov' : m === 'transformer' ? 'transformer' : 'lifers';
    this._chatPanel?.webview.postMessage({
      type: 'lifersConfig',
      model,
      sandbox: conf.get('sandbox') !== false,
    });
  }

  _postEmptyState() {
    const cfg = this._lifersConfigPayload();
    const payload = {
      type: 'bootstrap',
      state: { sessions: [], activeId: null, contextFiles: [] },
      lifersConfig: cfg,
    };
    this._chatPanel?.webview.postMessage(payload);
    this._sessionTreeProvider?.refresh();
  }

  resetSessionBootstrapFlag() {
    this._defaultSessionEnsured = false;
  }

  /**
   * When history has zero sessions, create one so the UI always has an active thread.
   * Called from webview `ready` and from activate() — does not rely on Chat/Sessions being visible.
   * @param {string} root
   */
  bootstrapDefaultSessionIfNeeded(root) {
    if (!root || this._defaultSessionEnsured) return;
    const conf = vscode.workspace.getConfiguration('lifers');
    if (conf.get('autoCreateSessionOnStart') === false) {
      this._defaultSessionEnsured = true;
      return;
    }
    let d = this._readHistory(root);
    if (d.sessions && d.sessions.length > 0) {
      this._defaultSessionEnsured = true;
      return;
    }
    this.newSession();
    d = this._readHistory(root);
    this._defaultSessionEnsured = true;
    if (!d.sessions || !d.sessions.length) {
      vscode.window.showErrorMessage(
        'Lifers: 未能创建或保存会话（磁盘权限或路径异常）。请重载窗口；仍失败则用「文件 → 打开文件夹」打开 lifers 便携根或 lifers_brain。'
      );
    }
  }

  /**
   * @param {string} root
   */
  _postState(root) {
    const data = this._readHistory(root);
    const activeId = data.activeSessionId;
    const ctx = data.sessions.find((x) => x.id === activeId)?.contextFiles || [];
    const payload = {
      type: 'bootstrap',
      state: {
        sessions: data.sessions,
        activeId,
        contextFiles: ctx,
      },
      lifersConfig: this._lifersConfigPayload(),
    };
    this._chatPanel?.webview.postMessage(payload);
    this._sessionTreeProvider?.refresh();

    if (this._chatPanel) {
      const s = data.sessions.find((x) => x.id === activeId);
      const title = s?.title && s.title !== '新会话' ? s.title : 'Agents Chat';
      this._chatPanel.title = s ? `Agents · ${title}` : 'Agents Chat';
    }
  }

  newSession() {
    const root = getBrainRoot();
    if (!root) {
      vscode.window.showWarningMessage('请先打开工作区（lifers.code-workspace 或 lifers_brain 文件夹）。 | Open lifers.code-workspace or lifers_brain folder.');
      return;
    }
    const data = this._readHistory(root);
    const id = 'lifers_sess_' + Date.now();
    data.sessions.unshift({
      id,
      title: '新会话',
      created: Date.now(),
      updated: Date.now(),
      messages: [],
      contextFiles: [],
    });
    data.activeSessionId = id;
    this._writeHistory(root, data);
    this._postState(root);
    this.ensureChatPanel();
  }

  _getContextOpts() {
    const conf = vscode.workspace.getConfiguration('lifers');
    const skipRaw = conf.get('contextFolderSkipDirs');
    const arr = Array.isArray(skipRaw) ? skipRaw : [];
    return {
      maxFiles: Number(conf.get('contextMaxFiles')) || 48,
      bridgeMax: Number(conf.get('bridgeContextMaxFiles')) || 32,
      folderMaxCollect: Number(conf.get('contextFolderMaxCollect')) || 96,
      folderDepth: Number(conf.get('contextFolderMaxDepth')) || 6,
      skipDirs: new Set(arr.map((x) => String(x).toLowerCase())),
    };
  }

  async addContextFiles() {
    const root = getBrainRoot();
    if (!root) return;
    const opts = this._getContextOpts();
    const picks = await vscode.window.showOpenDialog({
      canSelectMany: true,
      canSelectFiles: true,
      canSelectFolders: false,
      openLabel: '添加上下文文件',
      defaultUri: vscode.Uri.file(root),
      filters: {},
    });
    if (!picks || !picks.length) return;
    const rels = picks
      .map((u) => path.relative(root, u.fsPath).replace(/\\/g, '/'))
      .filter((r) => r && !r.startsWith('..'));
    const data = this._readHistory(root);
    const sid = data.activeSessionId;
    const s = data.sessions.find((x) => x.id === sid);
    if (!s) {
      this.newSession();
      return this.addContextFiles();
    }
    s.contextFiles = mergeContextPaths(s.contextFiles, rels, opts.maxFiles);
    s.updated = Date.now();
    this._writeHistory(root, data);
    this._chatPanel?.webview.postMessage({ type: 'context', files: s.contextFiles });
    this._postState(root);
  }

  /**
   * 将桥接 / 解析失败写入会话为一条 assistant 消息，避免 Chat 只闪一下错误就被 bootstrap 清空。
   * @param {string} root
   * @param {string} sessionId
   * @param {string} detail
   */
  _recordAssistantFailure(root, sessionId, detail) {
    const line = '⚠️ ' + String(detail || 'unknown').replace(/\s+/g, ' ').trim().slice(0, 4000);
    const data = this._readHistory(root);
    const s2 = data.sessions.find((x) => x.id === sessionId);
    if (s2) {
      if (!s2.messages) s2.messages = [];
      s2.messages.push({ role: 'assistant', content: line });
      s2.updated = Date.now();
      this._writeHistory(root, data);
    }
  }

  async addContextFolder() {
    const root = getBrainRoot();
    if (!root) return;
    const opts = this._getContextOpts();
    const picks = await vscode.window.showOpenDialog({
      canSelectMany: false,
      canSelectFiles: false,
      canSelectFolders: true,
      openLabel: '添加目录为上下文',
      defaultUri: vscode.Uri.file(root),
    });
    if (!picks || !picks.length) return;
    const dir = picks[0].fsPath;
    if (!isPathInside(root, dir) || !fs.existsSync(dir)) {
      vscode.window.showErrorMessage(
        '目录必须在当前工作区（lifers_brain / rs）内且存在。 | Folder must exist under the open workspace.'
      );
      return;
    }
    try {
      if (!fs.statSync(dir).isDirectory()) {
        vscode.window.showErrorMessage('所选路径不是目录。');
        return;
      }
    } catch {
      vscode.window.showErrorMessage('无法读取目录。');
      return;
    }

    const collected = collectFilesUnderFolder(root, dir, {
      maxFiles: opts.folderMaxCollect,
      maxDepth: opts.folderDepth,
      skipDirs: opts.skipDirs,
    });

    if (!collected.length) {
      vscode.window.showInformationMessage(
        '未收集到文件（可提高深度或减少排除目录）。 | No files collected.'
      );
      return;
    }

    const data = this._readHistory(root);
    let s = data.sessions.find((x) => x.id === data.activeSessionId);
    if (!s) {
      this.newSession();
      return this.addContextFolder();
    }
    s.contextFiles = mergeContextPaths(s.contextFiles, collected, opts.maxFiles);
    s.updated = Date.now();
    this._writeHistory(root, data);
    this._chatPanel?.webview.postMessage({ type: 'context', files: s.contextFiles });
    this._postState(root);
    vscode.window.showInformationMessage(
      `目录已加入上下文：${collected.length} 个路径，会话保留 ${s.contextFiles.length} 条（上限 ${opts.maxFiles}）。`
    );
  }

  /**
   * @param {vscode.Webview} webview
   */
  getChatHtml(webview) {
    return this._htmlShell(webview, 'chat.css', 'chat.js', this._chatBody(webview));
  }

  /**
   * @param {vscode.Webview} webview
   */
  getLauncherHtml(webview) {
    return this._htmlShell(webview, 'launcher.css', 'launcher.js', this._launcherBody());
  }

  /**
   * @param {vscode.Webview} webview
   */
  getControlPanelHtml(webview) {
    return this._htmlShell(webview, 'control_panel.css', 'control_panel.js', this._controlPanelBody());
  }

  _controlPanelBody() {
    return `
  <div>
    <h1>Lifers 总控（Kali 算力 + 同步）</h1>
    <p class="hint">依赖本机 <code>ssh</code>；路径在设置 <code>lifers.kaliSshPrefix</code>、<code>lifers.kaliBrainPath</code>。
    底部同一栏另有「会话建立」树。物化闭环在仓库：<code>scripts/embodied_tick_once.py</code> + <code>stack.embodied_world</code>（先 pause 再同步）。</p>
    <div class="row">
      <button type="button" id="btn-status">状态</button>
      <button type="button" class="secondary" id="btn-pause">暂停训练</button>
      <button type="button" id="btn-run">继续 run</button>
      <button type="button" class="secondary" id="btn-stop">stop</button>
      <button type="button" id="btn-sync">本机打包+scp</button>
      <button type="button" class="secondary" id="btn-open-settings">扩展设置</button>
    </div>
    <textarea id="out" readonly rows="14" spellcheck="false"></textarea>
  </div>`;
  }

  /**
   * @param {any} msg
   */
  async handleControlPanelMessage(msg) {
    const post = (text) => {
      this._controlView?.webview.postMessage({ type: 'controlResult', text });
    };
    if (msg.type === 'openSettings') {
      vscode.commands.executeCommand('workbench.action.openSettings', 'lifers.kali');
      return;
    }
    if (msg.type === 'controlReady') {
      post('就绪。先点「状态」确认 SSH；暂停会在合适检查点由远端 train 脚本处理（需已部署新版）。');
      return;
    }
    if (msg.type !== 'controlAction') return;
    const act = String(msg.action || '').trim();
    if (act === 'sync') {
      post(await this._runWindowsPackAndScp());
      return;
    }
    if (!['status', 'pause', 'run', 'stop'].includes(act)) {
      post('未知操作');
      return;
    }
    post(await this._runRemoteKaliTrainCtl(act));
  }

  /**
   * @param {'status'|'pause'|'run'|'stop'} act
   */
  async _runRemoteKaliTrainCtl(act) {
    const conf = vscode.workspace.getConfiguration('lifers');
    const ssh = String(conf.get('kaliSshPrefix') || 'ssh -o BatchMode=yes -o ConnectTimeout=12 kali@192.168.234.152').trim();
    const bp = String(conf.get('kaliBrainPath') || '/home/kali/lifers/lifers_brain').trim();
    const br = JSON.stringify(bp);
    const lines = [
      `BR=${br}`,
      `mkdir -p "$BR/weights" || true`,
      `act=${JSON.stringify(act)}`,
      `case "$act" in`,
      `status) echo "=== .train_control ==="; cat "$BR/weights/.train_control" 2>/dev/null || echo "(none)"; echo "=== process ==="; pgrep -af train_lifers_escalate 2>/dev/null || true; echo "=== weights ==="; stat -c "%s %y" "$BR/weights/lifers_transformer.json" 2>/dev/null || echo missing;;`,
      `pause) printf "pause\\n" >"$BR/weights/.train_control" && echo OK_pause;;`,
      `run) printf "run\\n" >"$BR/weights/.train_control" && echo OK_run;;`,
      `stop) printf "stop\\n" >"$BR/weights/.train_control" && echo OK_stop;;`,
      `esac`,
    ];
    const inner = lines.join(' ');
    const cmd = `${ssh} bash -lc ${JSON.stringify(inner)}`;
    const r = await execShell(cmd, { timeout: 120000 });
    const out = [r.stdout, r.stderr].filter(Boolean).join('\n---\n');
    return `exit ${r.code}\n${out || '(no output)'}`;
  }

  async _runWindowsPackAndScp() {
    const root = getBrainRoot();
    if (!root) return '未打开 lifers_brain 工作区，无法打包。';
    const conf = vscode.workspace.getConfiguration('lifers');
    const custom = String(conf.get('windowsPackScpCommand') || '').trim();
    if (custom) {
      const r = await execShell(custom, { cwd: root, timeout: 600000 });
      return `custom exit ${r.code}\n${r.stdout}\n${r.stderr}`;
    }
    const scriptPath = path.join(root, 'scripts', 'package_rs_for_kali.ps1');
    if (!fs.existsSync(scriptPath)) return `找不到 ${scriptPath}`;
    const dest = String(conf.get('kaliScpDestination') || 'kali@192.168.234.152:/home/kali/lifers_kali.tar.gz').trim();
    const dist = path.join(root, 'dist', 'lifers_kali.tar.gz');
    const ps = [
      '$ErrorActionPreference="Stop"',
      `Set-Location -LiteralPath ${JSON.stringify(path.dirname(scriptPath))}`,
      `& ${JSON.stringify(scriptPath)}`,
      `if (!(Test-Path -LiteralPath ${JSON.stringify(dist)})) { throw "no tarball" }`,
      `scp -o BatchMode=yes -o ConnectTimeout=30 ${JSON.stringify(dist)} ${JSON.stringify(dest)}`,
    ].join('; ');
    const r = await execShell(`powershell.exe -NoProfile -NonInteractive -Command ${JSON.stringify(ps)}`, {
      cwd: root,
      timeout: 600000,
    });
    return `pack+scp exit ${r.code}\n${r.stdout}\n${r.stderr}`;
  }

  /**
   * @param {vscode.Webview} webview
   * @param {string} cssName
   * @param {string} jsName
   * @param {string} bodyInner
   */
  _htmlShell(webview, cssName, jsName, bodyInner) {
    const nonce = getNonce();
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', cssName));
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', jsName));
    const iconUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'icon.png'));
    const iconStr = iconUri.toString();
    const csp = [
      `default-src 'none'`,
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `img-src ${webview.cspSource} https:`,
      `script-src 'nonce-${nonce}'`,
    ].join('; ');
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${styleUri}" rel="stylesheet" />
  <script nonce="${nonce}">window.__LIFERS_BRAND_ICON__=${JSON.stringify(iconStr)};</script>
</head>
<body>
${bodyInner}
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }

  /**
   * @param {vscode.Webview} webview
   */
  _chatBody(webview) {
    const iconUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'icon.png'));
    return `
  <div class="chat-root">
    <header class="chat-toolbar">
      <span class="chat-toolbar-title">Agents Chat</span>
      <div class="chat-toolbar-actions">
        <button type="button" class="tb-btn" id="btn-open-docs" title="README">
          <img class="aw-inline-icon" src="${iconUri}" width="14" height="14" alt="" /> Docs
        </button>
        <button type="button" class="tb-btn" id="btn-market">Marketplace</button>
      </div>
    </header>

    <section class="composer-strip" aria-label="Status">
      <div class="strip-row strip-row--muted">
        <span class="strip-chev">▸</span>
        <span>后台终端 · 请用 VS Code / Cursor 内置终端运行命令</span>
      </div>
      <div class="strip-row strip-main">
        <button type="button" class="strip-fold" id="btn-ctx-fold" aria-expanded="false" aria-controls="ctx-expanded">
          <span class="strip-chev" id="ctx-chev">▸</span>
          <span class="strip-label">上下文文件</span>
          <span class="strip-badge" id="ctx-badge">0</span>
        </button>
        <div class="strip-actions">
          <button type="button" class="strip-link" id="btn-clear-all-ctx" title="清空当前会话上下文">清空</button>
          <button type="button" class="strip-review" id="btn-review-readme">README</button>
        </div>
      </div>
      <div id="ctx-expanded" class="ctx-expanded hidden" role="region">
        <div id="context-bar" class="aw-context-bar-inner"></div>
      </div>
    </section>

    <section class="aw-composer" aria-label="Chat">
      <div id="bridge-progress" class="bridge-progress hidden" aria-live="polite">
        <div class="bridge-progress-head">执行过程</div>
        <pre id="bridge-progress-text" class="bridge-progress-text" spellcheck="false"></pre>
      </div>
      <div id="messages" class="aw-messages"></div>
      <div class="composer-input-card">
        <label class="sr-only" for="input">Message</label>
        <textarea id="input" rows="4" placeholder="Plan, Build — 行首 / 打开命令；@ 引用路径加入上下文"></textarea>
        <div id="slash-pop" class="composer-pop hidden" role="listbox" aria-label="Slash commands"></div>
        <div id="at-pop" class="composer-pop hidden" role="listbox" aria-label="Path completion"></div>
        <div class="composer-toolbar">
          <div class="toolbar-left">
            <div class="pill-agent" title="本地模型（lifers.model）">
              <span class="pill-infty" aria-hidden="true">∞</span>
              <span class="pill-text">Agent</span>
              <select id="sel-agent" class="toolbar-select">
                <option value="lifers">Lifers</option>
                <option value="transformer">transformer</option>
                <option value="markov">markov</option>
              </select>
            </div>
            <select id="sel-sandbox" class="toolbar-select toolbar-select--mode" title="Sandbox（lifers.sandbox）">
              <option value="safe">Auto</option>
              <option value="unsafe">Unsafe</option>
            </select>
          </div>
          <div class="toolbar-right">
            <span id="send-spinner" class="send-spinner hidden" role="status" aria-label="Sending"></span>
            <button type="button" class="icon-btn" id="btn-attach-file" title="添加文件到上下文">📄</button>
            <button type="button" class="icon-btn" id="btn-attach-folder" title="添加目录到上下文（递归采集）">📁</button>
            <button type="button" class="icon-btn icon-btn--disabled" id="btn-voice" disabled title="语音输入（未接入）">🎤</button>
            <button type="button" class="aw-send" id="btn-send" title="发送">
              <img class="aw-send-icon" src="${iconUri}" width="18" height="18" alt="" />
            </button>
          </div>
        </div>
      </div>
    </section>
  </div>`;
  }

  _launcherBody() {
    return `
  <div class="launcher-root">
    <p class="launcher-hint">会话列表：<strong>左侧 Lifers →「会话建立」</strong>；<strong>资源管理器</strong>内也有镜像树；<strong>底部面板 → Lifers 总控</strong> 栏内还有一份「会话建立」+「算力与同步」总控（关掉侧栏后请用这里）。</p>
    <p class="launcher-hint">Sessions: Lifers bar, Explorer, or bottom panel Lifers 总控.</p>
    <button type="button" class="launcher-btn" id="btn-chat">打开 Agents Chat</button>
    <button type="button" class="launcher-btn secondary" id="btn-scroll-sessions">显示会话建立面板</button>
  </div>`;
  }

  /**
   * @param {string} root
   * @param {string} text
   * @param {string | null} sessionId
   * @param {string[]} contextFiles
   */
  async _handleSend(root, text, sessionId, contextFiles) {
    const conf = vscode.workspace.getConfiguration('lifers');
    const py = resolveLifersPythonExecutable(conf.get('pythonPath'), root);
    const model = conf.get('model') || 'lifers';
    const sandbox = conf.get('sandbox') !== false;
    const httpDirect = conf.get('httpDirect') !== false;
    const remoteChat = conf.get('remoteChat') === true;
    const chatApiUrl = String(conf.get('chatApiUrl') || 'https://integrate.api.nvidia.com/v1/chat/completions').trim();
    const chatModel = String(conf.get('chatModel') || 'meta/llama-3.1-8b-instruct').trim();
    const chatApiKeyEnv = String(conf.get('chatApiKeyEnv') || 'NVIDIA_API_KEY').trim();
    const bridgeMax = Number(conf.get('bridgeContextMaxFiles')) || 32;
    const timeoutMs = Math.max(5000, Number(conf.get('bridgeTimeoutMs')) || 600000);

    let data = this._readHistory(root);
    if (!data.sessions.length) {
      const id = 'lifers_sess_' + Date.now();
      data.sessions = [
        {
          id,
          title: '新会话',
          created: Date.now(),
          updated: Date.now(),
          messages: [],
          contextFiles: [],
        },
      ];
      data.activeSessionId = id;
      this._writeHistory(root, data);
      this._postState(root);
    }

    let s = data.sessions.find((x) => x.id === sessionId) || data.sessions.find((x) => x.id === data.activeSessionId);
    if (!s) {
      vscode.window.showErrorMessage('没有活跃会话');
      this._postReply({ type: 'reply', ok: false, error: '没有活跃会话' });
      return;
    }

    if (!s.messages) s.messages = [];
    s.messages.push({ role: 'user', content: text });
    if (s.title === '新会话' || s.title === '(untitled)') {
      s.title = text.slice(0, 48) || '会话';
    }
    s.updated = Date.now();
    s.contextFiles = contextFiles && contextFiles.length ? contextFiles : s.contextFiles || [];
    this._writeHistory(root, data);

    const script = path.join(root, 'scripts', 'agent_bridge_once.py');
    if (!fs.existsSync(script)) {
      const msg =
        '缺少 scripts/agent_bridge_once.py。请打开 lifers/lifers.code-workspace（或 rs.code-workspace / lifers_brain 文件夹）；仅打开 rs0 时设 lifers.brainPathOverride（如 /home/kali/lifers/lifers_brain）后 Reload Window。';
      this._recordAssistantFailure(root, s.id, msg);
      this._postState(root);
      this._postReply({ type: 'reply', ok: false, error: msg });
      return;
    }

    const payload = JSON.stringify({
      text,
      contextFiles: s.contextFiles || [],
    });

    try {
      this._postReply({ type: 'bridgeProgressClear' });
      const { out, errText } = await new Promise((resolve, reject) => {
        const proc = cp.spawn(py, ['-u', script], {
          cwd: root,
          env: {
            ...process.env,
            PYTHONUTF8: '1',
            PYTHONIOENCODING: 'utf-8',
            LIFERS_ROOT: root,
            SANDBOX: sandbox ? '1' : '0',
            MODEL: String(model),
            LIFERS_CONTEXT_MAX_FILES: String(bridgeMax),
            ...(httpDirect ? { LIFERS_HTTP_DIRECT: '1' } : {}),
            ...(remoteChat
              ? {
                  LIFERS_REMOTE_CHAT: '1',
                  LIFERS_CHAT_URL: chatApiUrl,
                  LIFERS_CHAT_MODEL: chatModel,
                  LIFERS_CHAT_API_KEY_ENV: chatApiKeyEnv,
                }
              : {}),
          },
        });
        let stdout = '';
        let stderr = '';
        let stderrBuf = '';
        let settled = false;
        const timer = setTimeout(() => {
          if (settled) return;
          settled = true;
          try {
            proc.kill('SIGTERM');
          } catch {
            /* ignore */
          }
          reject(
            new Error(
              `Bridge 超时（>${timeoutMs / 1000}s）。可在设置 lifers.bridgeTimeoutMs 中延长；复杂任务请改用终端运行脚本。`
            )
          );
        }, timeoutMs);

        proc.stdout.on('data', (d) => (stdout += d.toString('utf8')));
        proc.stderr.on('data', (d) => {
          const chunk = d.toString('utf8');
          stderr += chunk;
          stderrBuf += chunk;
          const parts = stderrBuf.split(/\r?\n/);
          stderrBuf = parts.pop() || '';
          for (const line of parts) {
            if (line.includes('LIFERS_PROGRESS')) {
              this._postReply({ type: 'bridgeProgress', line: line.trim() });
            }
          }
        });
        proc.on('error', (e) => {
          clearTimeout(timer);
          if (!settled) {
            settled = true;
            reject(e);
          }
        });
        proc.on('close', (code) => {
          clearTimeout(timer);
          if (settled) return;
          settled = true;
          const tail = (stderrBuf || '').trim();
          if (tail.includes('LIFERS_PROGRESS')) {
            this._postReply({ type: 'bridgeProgress', line: tail });
          }
          if (code !== 0 && !stdout.trim()) {
            reject(new Error(stderr.trim() ? stderr.trim().slice(0, 2000) : `Python 退出码 ${code}`));
            return;
          }
          resolve({ out: stdout.trim(), errText: stderr });
        });
        proc.stdin.write(payload, 'utf8');
        proc.stdin.end();
      });

      let parsed;
      try {
        parsed = parseBridgeStdout(out, errText);
      } catch (pe) {
        const msg = pe instanceof Error ? pe.message : String(pe);
        this._recordAssistantFailure(root, s.id, msg);
        this._postState(root);
        this._postReply({ type: 'reply', ok: false, error: msg });
        return;
      }

      if (!parsed.ok) {
        const failMsg = parsed.error || 'unknown';
        this._recordAssistantFailure(root, s.id, failMsg);
        this._postState(root);
        this._postReply({ type: 'reply', ok: false, error: failMsg });
        return;
      }

      const replyText =
        parsed.text != null && String(parsed.text).trim()
          ? String(parsed.text)
          : '（模型返回为空，可重试或切换 lifers.model）';

      const d3 = this._readHistory(root);
      const s2 = d3.sessions.find((x) => x.id === s.id);
      if (s2) {
        s2.messages.push({ role: 'assistant', content: replyText });
        s2.updated = Date.now();
        this._writeHistory(root, d3);
      }

      this._postState(root);
      this._postReply({ type: 'reply', ok: true, text: replyText });
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      this._recordAssistantFailure(root, s.id, err);
      this._postState(root);
      this._postReply({ type: 'reply', ok: false, error: err });
    }
  }

  /** @param {any} payload */
  _postReply(payload) {
    this._chatPanel?.webview.postMessage(payload);
  }
}

function deactivate() {}

module.exports = { activate, deactivate };
