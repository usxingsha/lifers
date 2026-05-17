"""
Lifers Console — Web仪表盘
纯 stdlib HTTP + 嵌入式 HTML/CSS/JS，自给自足
"""

from __future__ import annotations

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Embedded dashboard HTML
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lifers Console</title>
<style>
:root { --bg: #0a0a14; --panel: #12122a; --accent: #00cc88; --warn: #ffaa00; --err: #ee4444; --text: #c8c8e0; --text2: #8888aa; }
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; height:100vh; display:flex; }
#sidebar { width:220px; background:var(--panel); padding:16px; display:flex; flex-direction:column; gap:8px; border-right:1px solid #1a1a3a; }
#sidebar h1 { color:var(--accent); font-size:20px; margin-bottom:12px; }
#sidebar .nav { display:flex; flex-direction:column; gap:4px; }
#sidebar .nav button { background:none; border:1px solid #2a2a4a; color:var(--text); padding:8px 12px; text-align:left; cursor:pointer; border-radius:4px; font-size:13px; }
#sidebar .nav button:hover, #sidebar .nav button.active { background:#1a1a3a; border-color:var(--accent); color:var(--accent); }
#sidebar .status { margin-top:auto; font-size:11px; color:var(--text2); }
#main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
#header { padding:12px 20px; background:var(--panel); border-bottom:1px solid #1a1a3a; display:flex; justify-content:space-between; align-items:center; }
#header h2 { font-size:16px; font-weight:500; }
#content { flex:1; padding:16px; overflow-y:auto; display:grid; gap:12px; }
.card { background:var(--panel); border:1px solid #1a1a3a; border-radius:6px; padding:14px; }
.card h3 { font-size:13px; color:var(--text2); margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }
.metric-row { display:flex; gap:16px; flex-wrap:wrap; }
.metric { text-align:center; padding:8px 12px; background:#0a0a20; border-radius:4px; min-width:100px; }
.metric .value { font-size:24px; font-weight:600; color:var(--accent); }
.metric .label { font-size:10px; color:var(--text2); margin-top:2px; }
.metric.warn .value { color:var(--warn); }
.metric.err .value { color:var(--err); }
.log-line { font-family:'Cascadia Code','Consolas',monospace; font-size:11px; padding:2px 0; border-bottom:1px solid #0e0e20; }
.log-line .ts { color:var(--text2); margin-right:8px; }
.log-line.info { color:var(--text); }
.log-line.warn { color:var(--warn); }
.log-line.err { color:var(--err); }
#chat-area { height:300px; overflow-y:auto; background:#0a0a20; border:1px solid #1a1a3a; border-radius:4px; padding:8px; margin-bottom:8px; font-size:12px; }
#chat-input { display:flex; gap:8px; }
#chat-input input { flex:1; background:#0a0a20; border:1px solid #2a2a4a; color:var(--text); padding:8px 12px; border-radius:4px; font-size:13px; }
#chat-input button { background:var(--accent); color:#000; border:none; padding:8px 16px; border-radius:4px; cursor:pointer; font-weight:600; }
#progress-bar { height:4px; background:#1a1a3a; border-radius:2px; overflow:hidden; margin-top:4px; }
#progress-bar .fill { height:100%; background:var(--accent); transition:width .5s; }
.agent-card { padding:8px; border-left:3px solid var(--accent); margin:4px 0; font-size:12px; }
.agent-card .role { color:var(--accent); font-weight:600; }
.agent-card .energy { font-size:10px; color:var(--text2); }
@media(max-width:768px){ body{flex-direction:column;} #sidebar{width:100%;flex-direction:row;flex-wrap:wrap;padding:8px;} #sidebar h1{margin:0;} }
</style>
</head>
<body>
<div id="sidebar">
  <h1>⚡ Lifers</h1>
  <span style="font-size:11px;color:var(--text2)">全能AI控制台</span>
  <div class="nav">
    <button class="active" data-view="overview">📊 总览</button>
    <button data-view="memory">🧠 记忆</button>
    <button data-view="swarm">🐝 群体</button>
    <button data-view="training">🏋️ 训练</button>
    <button data-view="chat">💬 对话</button>
    <button data-view="logs">📋 日志</button>
  </div>
  <div class="status" id="status-bar">连接中...</div>
</div>
<div id="main">
  <div id="header"><h2 id="view-title">📊 总览</h2><span id="clock" style="font-size:12px;color:var(--text2)"></span></div>
  <div id="content"></div>
</div>
<script>
const API = '/api';
let currentView = 'overview';
let statusEl, clockEl, titleEl, contentEl, progressEl;

function $(id){ return document.getElementById(id); }

function init(){
  statusEl = $('status-bar');
  clockEl = $('clock');
  titleEl = $('view-title');
  contentEl = $('content');
  document.querySelectorAll('.nav button').forEach(btn => {
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.nav button').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      currentView = btn.dataset.view;
      refresh();
    });
  });
  setInterval(refresh, 3000);
  setInterval(()=>{ clockEl.textContent = new Date().toLocaleString('zh-CN'); }, 1000);
  clockEl.textContent = new Date().toLocaleString('zh-CN');
  refresh();
}

async function refresh(){
  try {
    const r = await fetch(API + '/snapshot');
    const data = await r.json();
    statusEl.textContent = '✅ 在线 | v' + (data.version||'0.2') + ' | uptime ' + Math.round(data.uptime_sec||0) + 's';
    render(data);
  } catch(e) {
    statusEl.textContent = '⏳ 等待服务...';
  }
}

function render(data){
  const views = {
    overview: renderOverview,
    memory: renderMemory,
    swarm: renderSwarm,
    training: renderTraining,
    chat: renderChat,
    logs: renderLogs,
  };
  const fn = views[currentView] || renderOverview;
  titleEl.textContent = {'overview':'📊 总览','memory':'🧠 记忆','swarm':'🐝 群体','training':'🏋️ 训练','chat':'💬 对话','logs':'📋 日志'}[currentView] || '📊 总览';
  contentEl.innerHTML = fn(data);
}

function renderOverview(d){
  const m = d.metrics||{};
  return `<div class="card"><h3>关键指标</h3>
    <div class="metric-row">
      <div class="metric"><div class="value">${m.lifers_turns_total||0}</div><div class="label">总轮次</div></div>
      <div class="metric"><div class="value">${m.lifers_memory_items||0}</div><div class="label">记忆条目</div></div>
      <div class="metric"><div class="value">${m.lifers_vector_items||0}</div><div class="label">向量条目</div></div>
      <div class="metric"><div class="value">${m.lifers_errors_total||0}</div><div class="label">错误数</div></div>
      <div class="metric"><div class="value">${Math.round(m.lifers_uptime_sec||0)}s</div><div class="label">运行时间</div></div>
    </div>
    <div id="progress-bar"><div class="fill" style="width:${Math.min(100,((m.lifers_turns_total||0)%100))}%"></div></div>
    </div>
    <div class="card"><h3>推理延迟</h3><div class="metric-row">
      <div class="metric"><div class="value">${(d.inference_avg_ms||0).toFixed(0)}ms</div><div class="label">平均</div></div>
      <div class="metric"><div class="value">${(d.inference_p50_ms||0).toFixed(0)}ms</div><div class="label">P50</div></div>
      <div class="metric"><div class="value">${(d.inference_p99_ms||0).toFixed(0)}ms</div><div class="label">P99</div></div>
    </div></div>
    <div class="card"><h3>活跃追踪</h3><p style="font-size:12px">活跃Span: ${d.active_spans||0} | 已完成: ${d.completed_spans||0}</p></div>`;
}

function renderMemory(d){
  const m = d.memory||{};
  return `<div class="card"><h3>记忆状态</h3>
    <div class="metric-row">
      <div class="metric"><div class="value">${m.sql_count||0}</div><div class="label">SQL条目</div></div>
      <div class="metric"><div class="value">${m.vector_count||0}</div><div class="label">向量条目</div></div>
      <div class="metric"><div class="value">${m.vector_enabled?'✅':'❌'}</div><div class="label">向量启用</div></div>
      <div class="metric"><div class="value">${m.backend||'none'}</div><div class="label">后端</div></div>
    </div></div>
    <div class="card"><h3>最近记忆</h3>${(d.recent_memories||[]).map(r=>`<div class="log-line info"><span class="ts">${new Date(r.ts_ms).toLocaleTimeString()}</span> [${r.type}] ${JSON.stringify(r.content).substring(0,80)}</div>`).join('')}</div>`;
}

function renderSwarm(d){
  const agents = d.swarm_agents||{};
  return `<div class="card"><h3>智能体群体</h3>` +
    Object.entries(agents).map(([id,a])=>
      `<div class="agent-card"><span class="role">${a.role||id}</span> | 能量: ${(a.energy||0).toFixed(2)} | 知识: ${a.knowledge||0}条</div>`
    ).join('') +
    `<p style="font-size:11px;color:var(--text2);margin-top:8px">轮次: ${d.swarm_round||0} | 待处理消息: ${Object.values(d.pending_messages||{}).reduce((a,b)=>a+b,0)}</p></div>`;
}

function renderTraining(d){
  const t = d.training||{};
  return `<div class="card"><h3>训练状态</h3>
    <div class="metric-row">
      <div class="metric"><div class="value">${t.running?'🏋️':'⏸️'}</div><div class="label">状态</div></div>
      <div class="metric"><div class="value">${t.progress||0}%</div><div class="label">进度</div></div>
      <div class="metric"><div class="value">${t.loss||'-'}</div><div class="label">Loss</div></div>
      <div class="metric"><div class="value">${t.step||0}</div><div class="label">步数</div></div>
    </div></div>`;
}

function renderChat(d){
  return `<div class="card"><h3>AI 对话</h3>
    <div id="chat-area">${(d.chat_history||[]).map(m=>`<div class="log-line ${m.role==='user'?'warn':'info'}"><b>${m.role}:</b> ${m.text}</div>`).join('')}</div>
    <div id="chat-input"><input id="msg" placeholder="输入消息..." onkeydown="if(event.key==='Enter')sendMsg()"><button onclick="sendMsg()">发送</button></div></div>`;
}

async function sendMsg(){
  const input = $('msg');
  if(!input) return;
  const text = input.value.trim();
  if(!text) return;
  try {
    await fetch(API+'/chat', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    input.value = '';
    refresh();
  } catch(e){}
}

function renderLogs(d){
  return `<div class="card"><h3>最近日志</h3>` +
    (d.recent_logs||[]).map(l=>`<div class="log-line ${l.level||'info'}"><span class="ts">${new Date(l.ts_ms).toLocaleTimeString()}</span> ${l.message}</div>`).join('') +
    `</div>`;
}

window.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard Server
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardServer:
    """Simple HTTP dashboard for Lifers Console."""

    def __init__(self, host: str = "127.0.0.1", port: int = 55556) -> None:
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        # Data providers
        self._data_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._chat_handler: Optional[Callable[[str], str]] = None

    def set_data_provider(self, fn: Callable[[], Dict[str, Any]]) -> None:
        self._data_provider = fn

    def set_chat_handler(self, fn: Callable[[str], str]) -> None:
        self._chat_handler = fn

    def start(self) -> None:
        server = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(s, *args): pass  # silence logs

            def do_GET(s):
                if s.path == "/" or s.path == "/index.html":
                    s._serve_html()
                elif s.path == "/api/snapshot":
                    s._serve_json(server._get_snapshot())
                else:
                    s.send_response(404); s.end_headers()

            def do_POST(s):
                if s.path == "/api/chat":
                    length = int(s.headers.get("Content-Length", 0))
                    body = s.rfile.read(length)
                    data = json.loads(body)
                    reply = ""
                    if server._chat_handler:
                        reply = server._chat_handler(data.get("text", ""))
                    s._serve_json({"reply": reply})
                else:
                    s.send_response(404); s.end_headers()

            def _serve_html(s):
                s.send_response(200)
                s.send_header("Content-Type", "text/html; charset=utf-8")
                s.end_headers()
                s.wfile.write(_DASHBOARD_HTML.encode("utf-8"))

            def _serve_json(s, data):
                s.send_response(200)
                s.send_header("Content-Type", "application/json")
                s.send_header("Access-Control-Allow-Origin", "*")
                s.end_headers()
                s.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

        self._server = HTTPServer((self.host, self.port), Handler)

    def _get_snapshot(self) -> Dict[str, Any]:
        if self._data_provider:
            return self._data_provider()
        return {
            "version": "0.2.0",
            "uptime_sec": 0,
            "metrics": {},
            "memory": {},
            "swarm_agents": {},
            "training": {},
            "recent_logs": [],
            "recent_memories": [],
            "chat_history": [],
            "ts_ms": int(time.time() * 1000),
        }

    def serve_forever(self) -> None:
        self.start()
        print(f"Lifers Console → http://{self.host}:{self.port}")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            self._server.shutdown()
