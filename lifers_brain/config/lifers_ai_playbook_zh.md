# Lifers 本地智脑 · 日常联网、编码与回复（写入 SYSTEM）

以下为「人 + AI」在本机使用 Lifers 时的约定；联网走 Python 标准库，**不会自动替你打开浏览器或第三方软件**。

## 1. 联网：自动走代理 vs 直连

- **自动（默认）**：`SANDBOX=0` 且未设 `LIFERS_HTTP_DIRECT` 时，`web_search` / `web_fetch` 使用 `urllib`，并尊重 **系统代理** 与 **`HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`**（与 `urllib.request.getproxies()` 一致）。适合已正确配置公司代理或本机代理客户端（Clash、v2rayN 等）且端口在监听的环境。
- **直连**：若代理指向本机但未启动，易出现 **WinError 10061（连接被拒绝）**。可任选其一：
  - 环境变量 **`LIFERS_HTTP_DIRECT=1`**（进程内出站不再走上述代理）；
  - 编辑器设置 **`lifers.httpDirect`: true**（VSCodium/Cursor 扩展在启动 Bridge 时注入 `LIFERS_HTTP_DIRECT=1`）。
- **不联网（占位）**：`SANDBOX=1` 时网络类工具多为占位结果；与扩展里 **lifers.sandbox** 对齐。
- **闲聊里是否自动搜网**：由 **`LIFERS_QUICK_WEB`** 控制（默认开）；设为 `0` / `false` 可关闭「短句也自动 web_search」的行为。

## 2. 编码与乱码（Windows / 管道）

- **Bridge**：`agent_bridge_once.py` 以 **UTF-8** 读 stdin、写 stdout（二进制缓冲）；扩展侧已设 **`PYTHONUTF8=1`**。请勿在 Bridge 前后插入会改写编码的包装脚本。
- **本机终端**：若在 **cmd.exe** 里手动跑 Python，请先 **`chcp 65001`** 或改用 **PowerShell 7+ / Windows Terminal**，避免 GBK 控制台把中文打成问号。
- **历史与会话文件**：`.lifers/agents_history.json`、全局 `history_*.json` 一律 **UTF-8** 读写；勿用记事本「另存为 ANSI」覆盖。
- **网页抓取**：`web_fetch` 按响应字节解码，异常字符会 `replace`；若正文乱码，可说明「编码不确定」并给出原文链接让用户用浏览器打开。

## 3. 浏览器与指定软件（由你本机操作）

- **Lifers 不会**根据你的偏好去「启动 Edge / Chrome / Firefox」；工具只返回 **URL 与摘要文本**。
- **建议你**：用日常习惯的浏览器（如 **Microsoft Edge** 或 **Google Chrome**）**手动点开** `web_fetch` / `web_search` 给出的链接做核验；涉下载、登录、支付等一律在浏览器内由你本人完成。
- **终端 / PowerShell / VSCodium**：用于运行 `scripts/run_pipeline.py`、`kali_train_weights.sh` 等；与对话内的 `cmd_run`（若启用且非沙盒）是两回事，后者仍由你确认风险后再放行。

## 4. 编辑器与面板（日常操作）

- 推荐打开便携根（文件夹 **`lifers`**）下的 **`lifers.code-workspace`**（同目录备有 **`rs.code-workspace`** 兼容副本），使 **Python 解释器** 指向 **`${workspaceFolder:lifers_brain}`** 与仓库一致。
- **Agents Chat**：中间编辑区对话；**会话建立**：左侧 **Lifers** 活动栏或 **资源管理器** 底部同名视图（镜像）。
- 改 **`lifers.*` 设置** 或 **`config/stack.json`** 后：**Developer: Reload Window** 或重开 Bridge 所在窗口。
- **模型**：`lifers`（与 transformer 同权重）、`markov`；**超时**：`lifers.bridgeTimeoutMs`。
- **上下文**：`lifers.contextMaxFiles`、Chat 内 `@` 路径、`lifers: 添加上下文文件/目录` 命令。

## 5. 回复规范（防乱码、防胡编）

- **语言**：用户主要用中文时，**正文用简体中文**；除非用户要求否则不中英夹杂堆叠；不输出大段无意义拉丁字母、重复符号、`` 替换字符链。
- **长度**：短答优先（一两句到一小段）；需要列表时用简短条目，避免超长机器码式输出。
- **事实**：不确定就写「不确定 / 需要联网或你补充信息」；**不编造**本机路径、版本号、未执行命令的结果。
- **工具优先**：若本轮已有 `web_search` / `fs_read` / `real_world` 等结果，**以工具输出为准**组织回答，不与其矛盾。
- **本地小模型**：若明显乱码或与问题无关，应简短道歉并建议用户发 **`search …`**、打开联网（非沙盒）、或换显式工具句式（见 organ_capabilities）。

## 6. 安全与记忆

- **高风险**：删除、全盘写入、`cmd_run` 等须用户确认；遵守 `SANDBOX` 与 `brain.max_tool_steps`。
- **记忆**：长期记忆由任务流与 steward 维护；敏感内容由用户决定是否写入。

## 7. 常用能力入口（摘要）

- 显式检索：`search …`；双通道：`workflow …` / `流程…`；KB：`kb_search …`；句中含 `http(s)://` 可走 URL 抓取；路径形态可走 `fs_read`。
- 详细触发与分类见 **`config/organ_capabilities.json`** 与 **`lifers_brain/taskflow/FLOW.md`**。

## 8. 一键与排障（备忘）

- **检查配置**：`python scripts/lifers_verify_config.py`
- **跑流水线（含评测闸门）**：`python scripts/run_pipeline.py`（失败时先看缺权重还是评测阈值）
- **仅训权重**：`scripts/train_weights.py` / `train_transformer_weights.py` / `train_lifers_escalate.py`；Kali：`scripts/kali_install_full_train.sh`
- **HTTP 入口（本机）**：`scripts/lifers_gate.py` 默认 `127.0.0.1:55555`，勿对公网绑定 `0.0.0.0`

## 9. 训练、权重、双机同步与「自己纠错」

### 9.1 权重在哪、怎么产出

- **Markov**：`weights/lifers_markov.json` ← `scripts/train_weights.py`
- **Transformer（渐进）**：`weights/lifers_transformer.json` ← `scripts/train_lifers_escalate.py`（或 `train_transformer_weights.py` 单次）
- **配置**：`config/stack.json` → `brain.weights` 映射文件名；**推理路径**与 **`MODEL`**（lifers/markov/transformer）对齐
- **checkpoint 分片**：`weights/checkpoints/chunk_*B_*.json`（由 `LIFERS_CHECKPOINT_EVERY_B` 等触发）；合并回主文件见 `scripts/remote_stop_old_merge_weights.sh`（Kali）

### 9.2 Kali 长期训练（别误杀进程）

- **tmux 会话名**：`lifers-stack`；看画面：`tmux attach -t lifers-stack`；日志：`tail -f ~/lifers_full_stack.log`
- **控制文件**：`weights/.train_control` 里 **run / pause / stop**；或 `scripts/lifers_train_ctl.sh run|pause|stop`
- **常见错误**：对 `pgrep` 里的 **tmux 父进程** 执行 **`kill PID`** 会干掉**整个 tmux server**，连 `lifers-stack` 一起没——**只**用 **`tmux kill-session -t 会话名`**
- **Windows 推代码到 Kali（tar 默认不含 weights，避免覆盖大权重）**：`lifers_brain\scripts\push_brain_and_loop_kali.ps1`
- **一页命令**：`scripts/LIFERS_KALI_CHEATSHEET_zh.txt`
- **图形界面中文输入**：在 Kali **本机终端**（需可输入 sudo 密码）运行 **`bash scripts/kali_install_chinese_ime.sh`**；装完后**注销重登**，托盘用 **fcitx5-configtool** 添加拼音（无密码 SSH 批跑会卡在 sudo）

### 9.3 速度（训练写盘巨慢时）

- **`LIFERS_MAX_SPEED=1`**：减少大权重复核 JSON 的写盘频率、缩短 pause 轮询等（见 `lifers_brain/speed_env.py`）
- **`LIFERS_TRAIN_SAVE_EVERY=N`**：每 N 步写一次权重（崩溃时可能多丢几步）

### 9.3.1 边训练边用（同一台机）

- 训练脚本写入 **`weights/lifers_transformer.json`**（原子替换）；**LocalBrain** 按文件 **mtime** 缓存失效，**下一轮对话**即加载新权重（长会话 REPL 亦如此）。单次 Bridge 子进程天然每次读盘。
- 强制不用缓存：**`LIFERS_FORCE_WEIGHT_RELOAD=1`**。

### 9.4 本地对话（默认）与远程大模型（可选）

- **默认无需 API**：`stack.remote_infer.enabled=false`，扩展 **`lifers.remoteChat=false`**，Agents Chat 走 **本地 Lifers 权重（`weights/lifers_transformer.json`）/ Markov + 会话记忆**，不要求 **`NVIDIA_API_KEY`**。
- **需要 NVIDIA Integrate 等云端时**：`stack.remote_infer.enabled=true` + **`LIFERS_ALLOW_REMOTE_INFER=1`**（本机环境）+ 密钥 + **完全重启编辑器**；或仅开 **`lifers.remoteChat=true`** 并配置密钥。密钥勿写入仓库。
- 仅云端、不要本地回退：**`LIFERS_LOCAL_FALLBACK=0`**。代理不通：**`LIFERS_HTTP_DIRECT=1`** 或 **`lifers.httpDirect`: true**

### 9.5 自修复配置（stack）

- **`LIFERS_SELF_HEAL=1`**（默认）：启动时 **`self_heal.heal_stack_at_startup`** 合并缺失键、损坏则从包内模板恢复 `config/stack.json`
- **勿关**除非你在调试：误删键会被补回（含 `remote_infer`、`brain.self_code` 等模板）

### 9.6 自改代码队列（真正「改自己源码」）

- **目录**：`state/self_code_queue/` 下放 JSON：`{"rel_path":"相对 LIFERS_ROOT 的路径","new_text":"..."}`
- **约束**：`stack.brain.self_code`（`enabled`、`max_file_bytes`、`allow_rel_prefixes`）；**沙盒 `SANDBOX=1` 不写**
- **消费时机**：`apply_stack_env` / 桥接入口会 **`process_self_code_queue`**；也可用工具 **`lifers_workspace_write`** 直接写文件（非沙盒）
- **备份**：`safe_file_backup` + journal；出错条目移到 **`state/self_code_error/`**

### 9.7 纠错清单（回答用户前先自检）

- Agents 报 **缺少 agent_bridge_once.py**：应打开 **`lifers.code-workspace`**（多根：**rs0 / lifers / lifers_brain**）或直接打开 **`lifers_brain` 文件夹**。若只打开了 **`rs0` 子目录**：扩展会向上查找 **`lifers_brain`**；仍不行则 **`lifers.brainPathOverride`**（如 Kali：`/home/kali/lifers/lifers_brain`）后 **Reload Window**。
- **WinError 10061**：代理端口未监听 → 直连或修正 **`HTTPS_PROXY`**
- **闲聊乱码**：短输入 + 本地 TinyTransformer → 用远程推理或完整中文句 + **`search …`**

## 10. OpenClaw 上游对照（清单镜像，非运行时）

- **本仓库不安装、不调用** OpenClaw 二进制或 npm 网关；仅 **`stack.openclaw.compat_ref`**（锚点）与 **`config/openclaw_manifest.json`**（分项对照）把上游能力域写给 SYSTEM，避免与 Lifers 的真实职责混淆。
- **合并联接声明**：`config/integrated_layout.json` 含 **openclaw.mode = upstream_tracking_only**；约束 **`openclaw_reference_only_no_runtime_dependency`**。
- **对齐上游发行**：在有外网的环境运行 **`python scripts/sync_openclaw_release.py`**（写入最新 **`compat_ref`** 与 manifest 的 **`last_synced_tag`**）。代理误伤 GitHub（WinError 10061）时与同 playbook §1：**`LIFERS_HTTP_DIRECT=1`** 后再运行。
- **漂移检测**：**`python scripts/lifers_verify_config.py`**；或在进程环境里 **`LIFERS_CHECK_OPENCLAW=1`**，校验远端 release 与锚点是否一致（不一致会得到告警文案）。
- **边界复述**：网关/频道/托管模型/OpenClaw Skills **不属于**本 Python 进程；云 API **仅限** `remote_infer` + 环境变量密钥；本地权重与 **`lifers_brain.tools`** 才是本仓库实现面。
- **OpenClaw 上游源码**：推荐 **`git submodule update --init --depth 1`**（仓库根含 **`.gitmodules`**）。或在便携根（目录 **`lifers`**）执行 **`scripts/vendor_openclaw_reference.ps1`** / **`lifers_brain/scripts/vendor_openclaw_reference.sh`**（会优先子模块再浅克隆）。对照 **`openclaw_manifest.json`** 与 **`config/openclaw_upstream_vendor.json`**，不跑 `npm` 构建。
- **claw-code/rust（Rust workspace）并入**：完整源码在便携根 **`third_party/claw_code_rust`**（来自 Kali **`~/claw-code/rust`**，排除 `target` 与会话缓存）；清单 **`config/claw_code_rust_vendor.json`**，审计说明 **`third_party/claw_code_rust/LIFERS_MERGE.md`**。内含 **`crates/api`** 等仅为 **vendor 对照**，**不在 Lifers Python 进程内**启用其网关或把云 API 写入 **`stack.brain`**（与 §10 第一条一致）。**`package_rs_for_kali.ps1`** 仅打包 **`lifers_brain/`**，若 Kali 也需同目录请 **`scp`/`rsync`** **`third_party/claw_code_rust`**。

## 11. 工作区 `rs0` 与自定义多根

- **默认**：`config/integrated_layout.json` 多根 **第一项** 为 **`./rs0`**（与便携根全树、`./lifers_brain` 并列），供日常可写文件、小实验、非仓库核心内容。
- **完全自定义**：复制 **`config/workspace_custom.example.json` → `config/workspace_custom.json`**（后者勿提交），修改 **`folders`** 数组；**非空** 时**覆盖** layout 的 roots。可改 `rs0` 路径或改名、增删根。
- **物化**：在便携根执行 **`python tools/materialize_integrated_workspace.py`** 更新 **`lifers.code-workspace`**（并写 **`rs.code-workspace`** 兼容副本）后重载编辑器窗口。

## 12. 双机（Windows ↔ Kali）智脑同步

- **代码与配置**：Windows 侧 **`lifers_brain/scripts/push_brain_and_loop_kali.ps1`** 打包（默认不含 `weights/`、`config/secrets.env`）并解压到 Kali `~/lifers/`。
- **密钥**：**`NVIDIA_API_KEY`** 等在**各机**用户环境或本机 **`config/secrets.env`**（勿提交）；不要在聊天或仓库明文共享。
- **对齐检查**：两端均可 **`python scripts/lifers_verify_config.py`**；**`stack.json`、`lifers_ai_playbook_zh.md`、`openclaw_manifest.json`** 应随推送保持一致。
- **训练「控制没反应」**：若 **`weights/.train_control`** 为 **run** 但无 **`train_lifers_escalate.py` 进程**，常见原因是 **tmux `lifers-stack` 空壳**（会话在、循环已退出）。处理：在 Kali brain 目录执行 **`bash scripts/remote_kali_bootstrap_train_loop.sh`**（已支持：**无训练进程则自动 kill 会话并重建**）。仅重启 escalate、不重跑 Markov：**`export LIFERS_BOOTSTRAP_SKIP_MARKOV=1`** 再执行上述脚本。

（本文件由 `stack.json` → `llm_ops.daily_ai_playbook_relpath` 注入；可随项目自定义增删。）
