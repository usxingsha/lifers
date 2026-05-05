// @ts-check
/**
 * Session list tree in the Lifers activity bar container (native TreeView, not webview).
 * Loaded by extension.js via require('./session_tree.js').
 */
const vscode = require('vscode');

/**
 * @param {any[]} sessions
 */
function bucketSessions(sessions) {
  const day = 86400000;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const t0 = today.getTime();
  const todayList = [];
  const weekList = [];
  const older = [];
  [...sessions]
    .sort((a, b) => (b.updated || 0) - (a.updated || 0))
    .forEach((s) => {
      const u = s.updated || 0;
      if (u >= t0) todayList.push(s);
      else if (u >= t0 - 7 * day) weekList.push(s);
      else older.push(s);
    });
  return { todayList, weekList, older };
}

/**
 * @param {any} s
 */
function sessionMetaLine(s) {
  const msgs = s.messages || [];
  let chars = 0;
  msgs.forEach((m) => {
    chars += (m.content || '').length;
  });
  const ctxN = (s.contextFiles || []).length;
  return '+' + chars.toLocaleString('en-US') + ' · ' + ctxN + ' file' + (ctxN === 1 ? '' : 's');
}

class LifersSectionTreeItem extends vscode.TreeItem {
  /**
   * @param {string} label
   * @param {string} sectionId
   * @param {any[]} sessions
   */
  constructor(label, sectionId, sessions) {
    super(label, vscode.TreeItemCollapsibleState.Expanded);
    this.sectionId = sectionId;
    this.sessions = sessions;
    this.contextValue = 'lifersSessionSection';
    this.id = `lifers-section-${sectionId}`;
    this.iconPath = new vscode.ThemeIcon('folder');
  }
}

class LifersSessionTreeItem extends vscode.TreeItem {
  /**
   * @param {any} session
   * @param {boolean} isActive
   */
  constructor(session, isActive) {
    super(session.title || '(untitled)', vscode.TreeItemCollapsibleState.None);
    this.sessionId = session.id;
    this.contextValue = isActive ? 'lifersSessionActive' : 'lifersSession';
    this.id = `lifers-sess-${session.id}`;
    this.description = sessionMetaLine(session);
    this.iconPath = isActive ? new vscode.ThemeIcon('debug-stackframe-focused') : new vscode.ThemeIcon('circle-outline');
    this.command = {
      command: 'lifers.agents.activateSession',
      title: '切换会话',
      arguments: [session.id],
    };
  }
}

class LifersSessionTreeProvider {
  /**
   * @param {{ _readHistory: (root: string) => any; }} controller
   * @param {() => string | null} getRoot
   */
  constructor(controller, getRoot) {
    this._c = controller;
    this._getRoot = getRoot;
    this._emitter = new vscode.EventEmitter();
    this.onDidChangeTreeData = this._emitter.event;
  }

  refresh() {
    this._emitter.fire(undefined);
  }

  /**
   * @param {vscode.TreeItem} element
   */
  getTreeItem(element) {
    return element;
  }

  /**
   * @param {vscode.TreeItem | undefined} element
   * @returns {vscode.ProviderResult<vscode.TreeItem[]>}
   */
  getChildren(element) {
    const root = this._getRoot();
    if (!root) {
      return [new vscode.TreeItem('请先打开工作区（lifers.code-workspace / lifers_brain）', vscode.TreeItemCollapsibleState.None)];
    }
    let data;
    try {
      data = this._c._readHistory(root);
    } catch {
      return [new vscode.TreeItem('无法读取会话记录', vscode.TreeItemCollapsibleState.None)];
    }
    const sessions = data.sessions || [];
    const activeId = data.activeSessionId;
    const { todayList, weekList, older } = bucketSessions(sessions);

    if (!element) {
      /** @type {vscode.TreeItem[]} */
      const sections = [];
      if (todayList.length) {
        sections.push(new LifersSectionTreeItem('Today · 今日', 'today', todayList));
      }
      if (weekList.length) {
        sections.push(new LifersSectionTreeItem('Last 7 days · 近 7 天', 'week', weekList));
      }
      if (older.length) {
        sections.push(new LifersSectionTreeItem('Earlier · 更早', 'older', older));
      }
      if (!sections.length) {
        const hint = new vscode.TreeItem('（无会话）使用标题栏「新建会话」', vscode.TreeItemCollapsibleState.None);
        hint.iconPath = new vscode.ThemeIcon('info');
        return [hint];
      }
      return sections;
    }

    if (element instanceof LifersSectionTreeItem) {
      return element.sessions.map((s) => new LifersSessionTreeItem(s, s.id === activeId));
    }

    return [];
  }
}

module.exports = { LifersSessionTreeProvider, bucketSessions };
