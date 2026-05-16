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
  const btnProgress = document.getElementById('btn-progress');
  if (btnProgress) btnProgress.addEventListener('click', () => post('progress'));
  document.getElementById('btn-open-settings').addEventListener('click', () => vscode.postMessage({ type: 'openSettings' }));

  function applyTrainProgress(d) {
    const po = document.getElementById('p-overall');
    const ps = document.getElementById('p-sgd');
    const to = document.getElementById('t-overall');
    const ts = document.getElementById('t-sgd');
    const meta = document.getElementById('train-meta');
    if (!po || !ps || !meta) return;
    if (!d || typeof d !== 'object' || d._parse_error) {
      meta.textContent = d && d._parse_error ? '进度 JSON 解析失败' : '无远端数据（SSH / kaliBrainPath / 是否在训）';
      po.value = 0;
      ps.value = 0;
      if (to) to.textContent = '—';
      if (ts) ts.textContent = '—';
      return;
    }
    const oc =
      typeof d.overall_pct_approx === 'number'
        ? Math.round(Math.min(100, Math.max(0, d.overall_pct_approx)))
        : null;
    const sp =
      d.sgd && typeof d.sgd.pct === 'number' ? Math.round(Math.min(100, Math.max(0, d.sgd.pct))) : null;
    po.value = oc != null ? oc : 0;
    ps.value = sp != null ? sp : 0;
    if (to) to.textContent = oc != null ? oc + '%' : '—';
    if (ts) ts.textContent = sp != null ? sp + '%' : '—';
    const parts = [];
    if (d.phase) parts.push(String(d.phase));
    if (d.updated_at) parts.push(String(d.updated_at));
    if (d.ramp) parts.push('ramp ' + d.ramp.iter + '/' + d.ramp.max);
    if (d.sgd) parts.push('sgd ' + d.sgd.step + '/' + d.sgd.total_steps);
    meta.textContent = parts.join(' · ') || '—';
  }

  window.addEventListener('message', (ev) => {
    const m = ev.data;
    if (m && m.type === 'controlResult') {
      document.getElementById('out').textContent = m.text || '';
    }
    if (m && m.type === 'trainProgress') {
      applyTrainProgress(m.data);
    }
  });

  vscode.postMessage({ type: 'controlReady' });
})();
