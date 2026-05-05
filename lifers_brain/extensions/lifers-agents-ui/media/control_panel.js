(function () {
  const vscode = acquireVsCodeApi();

  function post(act) {
    document.getElementById('out').textContent = '… 执行中';
    vscode.postMessage({ type: 'controlAction', action: act });
  }

  document.getElementById('btn-status').addEventListener('click', () => post('status'));
  document.getElementById('btn-pause').addEventListener('click', () => post('pause'));
  document.getElementById('btn-run').addEventListener('click', () => post('run'));
  document.getElementById('btn-stop').addEventListener('click', () => post('stop'));
  document.getElementById('btn-sync').addEventListener('click', () => post('sync'));
  document.getElementById('btn-open-settings').addEventListener('click', () => vscode.postMessage({ type: 'openSettings' }));

  window.addEventListener('message', (ev) => {
    const m = ev.data;
    if (m && m.type === 'controlResult') {
      document.getElementById('out').textContent = m.text || '';
    }
  });

  vscode.postMessage({ type: 'controlReady' });
})();
