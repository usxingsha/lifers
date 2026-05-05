(function () {
  const vscode = acquireVsCodeApi();

  document.getElementById('btn-chat').addEventListener('click', () => {
    vscode.postMessage({ type: 'openChat' });
  });
  document.getElementById('btn-scroll-sessions').addEventListener('click', () => {
    vscode.postMessage({ type: 'focusSessions' });
  });

  vscode.postMessage({ type: 'ready', surface: 'launcher' });
})();
