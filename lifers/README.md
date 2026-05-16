# Lifers

Python 包名 **`lifers`**；便携仓库根目录历史上曾名 **`rs`**，现工作区与配置统一以 **`lifers`** 为显示名（联运布局见便携根 **`../config/integrated_layout.json`**，物化 **`../lifers.code-workspace`**）。**产品向目标**：同一进程内「全能型」本地栈——自动化编排（taskflow / 桥 / gate）+ 可审计推理（LocalBrain ± 可选远程）+ 工具注册表 + 会话与长期记忆 + 本能与扩展（详见 **`config/lifers_ai_pipeline.json#omni_stack_zh`** 与 **`config/lifers_ai_playbook_zh.md` §13**）。Offline local agent, tools, eval/sim stubs.

**自测（会话分类 / CHAT_QUICK / Bridge JSON）**：在 **`lifers`** 目录执行 `PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py" -v`（见 `tests/test_session_dialogue.py`；不依赖 PyTorch）。

## Agents 侧栏

**VSCodium / 便携**：请用便携根下的 **`run_lifers_vscodium.bat`**（扩展目录 → **`data\extensions`**）。安装/更新后 **Developer: Reload Window**。

**一键全量同步 UI（推荐）**：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sync_lifers_agents_ui_windows_kali.ps1`  
（不打断 Kali 训练加 **`-SkipTrainPause`**；同步后让训练继续加 **`-ResumeKaliTrain`**；只看 Kali 训练状态：`scripts\kali_train_status.ps1`）

**对话质量预期**：`weights/lifers_transformer.json` 由 **`train_lifers_escalate.py` / `train_transformer_weights.py`** 等写入（同一 JSON 格式），质量随训练进度上升；**不是** ChatGPT 级开箱体验。短输入容易采样飘；请用**完整中文**或 **`search …`** 走联网检索；需要更强模型时再考虑可选云端（见下）。训练写入权重后，对话侧按文件更新时间热加载（边训边用）。

**Agents Chat「中文没反应」常见原因**：每次发送 Bridge **子进程**会重新加载 **`weights/lifers_markov.json`**；若该 JSON **体积极大（如训练写到数 GB）**，`json.loads` 会长时间卡住，界面像死机。对策：（1）**「你好」等寒暄**已在快路径直接中文回复（不读 Markov）；（2）默认 **`LIFERS_MARKOV_JSON_MAX_BYTES=128MB`** 超限则立即返回说明（可设 **`lifers.markovJsonMaxBytes`** 或环境变量提高上限，或压缩 Markov 文件 / 改用 transformer）；（3）同步 Kali 后 **Reload Window** 与新版扩展。

**训练/权重/纠错知识注入**：长文约定在 **`config/lifers_ai_playbook_zh.md` §9**（经 `llm_ops` 进入对话上下文）；含 Kali tmux、checkpoint、速度变量、常见误操作。**自修复**：`LIFERS_SELF_HEAL`（默认开）合并缺失 `stack` 键；**自改代码**：`state/self_code_queue/` JSON → `self_code_runner`（见 playbook §9.6）。

**Agents Chat 默认路径（无需 API 密钥）**：扩展向 Bridge 注入 **`MODEL=lifers`**、**`LIFERS_FORCE_LOCAL_ONLY`**：只用本地 **`weights/lifers_transformer.json`** + 工具链；联网事实检索用 **`SANDBOX=0`** 下的 **`web_search` / `web_fetch`**（你在本机浏览器里点开链接核验）。**可选云端 Chat（高级）**：仅在终端手动跑 Python 且不设 `LIFERS_FORCE_LOCAL_ONLY` 时，`stack.remote_infer` + `LIFERS_ALLOW_REMOTE_INFER` + 密钥才可能生效；详见 **`config/lifers_ai_playbook_zh.md` §9.4** 与 **`config/nvidia_api.env.example`**。

**速度（训练 / 执行 / HTTP）**：设环境变量 **`LIFERS_MAX_SPEED=1`**（或在 Kali/tmux 里 `export` 后再启训练）可启用：`train_sgd` **降低大权重复核 JSON 写盘频率**、pause 轮询更密、HTTP 超时上限收紧、本地 LM 生成长度上限缩小；详见 **`lifers/speed_env.py`** 与 **`scripts/LIFERS_KALI_CHEATSHEET_zh.txt`**。仍可用 **`LIFERS_TRAIN_SAVE_EVERY`**、**`LIFERS_PAUSE_POLL_SEC`**、**`LIFERS_HTTP_TIMEOUT_CAP`** 逐项覆盖。代理不通时扩展里 **`lifers.httpDirect`: true** 或 **`LIFERS_HTTP_DIRECT=1`**。

**训练吃满 CPU（Kali 上常仅 ~20–30%）**：`train_sgd` 的纯 Python 列表实现是**单核**的；安装 **NumPy** 后（`pip install numpy` 或 `apt install python3-numpy`）同一代码会走 **NumPy/BLAS 前向**（`transformer_lm.py`），再设 **`export OMP_NUM_THREADS=$(nproc)`**（或 `OPENBLAS_NUM_THREADS`）以用满多核。显式关闭加速路径：`LIFERS_USE_NUMPY=0`。

**每日 AI 健康与自动纠偏（本地）**：`python eval/daily_ai_health.py`（或 `scripts/run_daily_ai_health.ps1` / `scripts/run_daily_ai_health.sh`）汇总 **eval 套件（逻辑/格式/一致性代理）**、**`eval/full_system_check.py` 全链**、**桥接延迟**、**多文件上下文吞吐**（`LIFERS_CONTEXT_MAX_FILES` 扫描），报告写入 **`state/daily_health/latest.json`**（目录已 gitignore）。可选 **`LIFERS_DAILY_HEALTH_AUTO_REMEDIATE=1`**：仅在检查通过且负载偏高时，把 **`lifers.contextMaxFiles` / `lifers.bridgeContextMaxFiles` / `lifers.bridgeTimeoutMs`** 合并进 **`config/workspace_custom.json`**（已忽略提交）并尝试运行 **`tools/materialize_integrated_workspace.py`**。

**一键维护（拉远程 + Kali 权重 + 健康检查）**：`scripts/run_maintenance_all.ps1`（干净工作区会先 `git pull`；有本地改动请加 **`-SkipGit`**）。Windows 每日定时：`scripts/register_maintenance_task.ps1`（管理员 PowerShell）。

**对话推理分发**：`lifers/taskflow/dialogue_router.py` 的 **`infer_dialogue_route`** 按用户尾句决定 **CHAT_QUICK / WEB_SEARCH / FULL_PIPELINE …**；编排器写入 **`TaskContext.dialogue_route_*`**，并在 stderr 输出 **`LIFERS_PROGRESS dialogue_route`**（Agents Chat「执行过程」可见）。Kali 同步：照旧 **`scripts/lifers_sync_pause_and_brain.ps1`** 或 **`push_brain_and_loop_kali.ps1`**。

**布局（v0.4.5+）**：**Agents Chat** 在**中间编辑器**；**会话建立**（会话树）在**左侧活动栏 → Lifers** 下，与 **Agents** 启动器同栏（不再依赖 Secondary Side Bar）。命令：**Lifers: 打开 Agents Chat**、**Lifers: 显示会话建立**。**上下文**：**Lifers: 添加上下文文件**、**Lifers: 添加目录为上下文**（递归收集文件路径，受设置 `lifers.*` 限制）。

**Kali 训练权重**：`scripts/package_rs_for_kali.ps1` → `dist/lifers_kali.tar.gz`；在 Kali 上解压后跑 **`scripts/kali_install_full_train.sh`**（默认 `full` + 名义 **`LIFERS_TARGET_PARAM_B=20`** 的渐进放大，遇 **OOM** 停止；产物 **`weights/lifers_*.json`**）。详见 **`scripts/KALI_WEIGHTS.md`**。  
**一键推代码到 Kali 并开长期 escalate 循环**：`powershell -File scripts\push_brain_and_loop_kali.ps1`（tmux **`lifers-stack`**，日志 **`~/lifers/lifers_full_stack.log`**；tar **不含** `weights/`，不冲掉 Kali 已有大权重）。Kali 上一页命令表：**`scripts/LIFERS_KALI_CHEATSHEET_zh.txt`**（可 `scp` 到 `~/` 常看）。

### 随跑随用 · 每段 B 暂停同步 · 能力队列（不设「产品挡位」上限）

- **超过单一聊天大模型的底子**：靠 **整条栈** 叠起来——本地 **`lifers_transformer.json` 持续训练** + **stack.json → llm_ops** 外部强模型 + **taskflow**（检索/命令/沙箱）+ **eval**。本地 ramp 负责「可增量、可审计、可离线」；前沿对话/编程仍可接 API/自托管大模型，本地权重负责增量与编排。
- **每满约 1B 近似 float 预算**（`LIFERS_CHECKPOINT_EVERY_B`，默认 1）：`train_lifers_escalate.py` 写入 **`weights/checkpoints/chunk_*B_*.json`** + **`manifest.jsonl`**，并可设 **`LIFERS_POST_CHECKPOINT_CMD`**（例：`scripts/post_checkpoint_hook_example.sh`）做 scp/rsync。
- **暂停去更新**：设 **`LIFERS_PAUSE_ON_CHECKPOINT=1`**，在新 B 档 checkpoint 落盘且 post-cmd 跑完后，自动把 **`weights/.train_control`** 写成 **`pause`**；你同步/打包/换机拉取后，再 **`lifers_train_ctl.sh run`**（或 `echo run > …/.train_control`）继续。**不限训练目标 B**：配合 **`LIFERS_ESCALATE_UNLIMITED=1`** + 大 **`LIFERS_RAMP_MAX_ITERS`**（见 `train_lifers_escalate.py` 文档字符串），直到 OOM 或手动 stop。
- **跑完一段就切下一类能力**：编辑 **`config/capability_queue.json`**（编程 → 聊天 → 工具 → 检索 → 安全…），用 **`python scripts/lifers_capability_queue.py show|env|advance`**；`env` 子命令打印 **`LIFERS_TRAIN_SUITE_DIR`**，供下一轮 escalate 读取（各 `suite_dir` 下放你的 **jsonl** 语料）。

**配置**：便携根 **`config/integrated_layout.json`** → `workspace_settings`，物化 **`lifers.code-workspace`**（单文件）。对照：**`config/lifers_agents_ui.defaults.json`**（与 `lifers.*` 键对齐）。

**会话持久化（v0.3.2+）**：历史会话保存在编辑器 **全局存储**（按 Lifers 根目录绝对路径哈希），并**镜像**到 **`.lifers/agents_history.json`**。切换「只打开 brain」与「打开 **`lifers.code-workspace`**」只要指向同一物理 **`lifers`**，应加载同一会话。

扩展目录：`extensions/lifers-agents-ui`。安装：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/install_agents_extension.ps1
```

只用 **VSCodium**（便携 `data\extensions` + 本机 `~\.vscode-oss\extensions`），不往 Cursor / VS Code 市场目录拷：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/install_agents_extension.ps1 -EditorTargets Vscodium
```

或环境变量：`LIFERS_EXT_INSTALL_TARGETS=Vscodium`（再执行不带参数的脚本）。

仅检查：`... install_agents_extension.ps1 -VerifyOnly`（可加同一 `-EditorTargets`）。

然后 **Developer: Reload Window**。便携编辑器请将 `VSCODE_PORTABLE` 指向 **`data`** 或使用 **`tools\launch_rs_editor.cmd`**（仓库根 **`run_lifers_vscodium.bat`** 同逻辑）。

**不装 Microsoft VS Code、用自有壳 + VSCodium**：在 **`lifers/tools/lifers_editor/`** 运行 **`lifers-editor.ps1`**（Windows）或 **`lifers-editor.sh`**（Linux/Kali）。会同步 **`tools/vscodium_editor_defaults.json`** 到独立 user-data（若尚无 `User/settings.json`）、复制 Agents UI 到独立 extensions、优先 **`../shell/VSCodium`** 与系统 **VSCodium**，打开 **`lifers.code-workspace`**（若存在）。需临时用 Code/Cursor：PowerShell 加 **`-AllowProprietaryEditors`**；Bash 设 **`LIFERS_EDITOR_ALLOW_PROPRIETARY=1`**。

**自研 GUI + 自带 Bridge 宿主（不依赖 VSCodium 窗口）**：在 **`lifers`** 下执行 `PYTHONPATH=. python scripts/run_lifers_gui_host.py`（或 **`tools/lifers_gui_host/run_gui_host.ps1` / `run_gui_host.sh`**）。默认 **http://127.0.0.1:18765/**：左侧工作区（离线 **textarea** + **Monaco**：优先 **`static/vendor/monaco/min/vs`**，否则 jsDelivr）、右侧对话；`POST /api/bridge` 与 **`agent_bridge_once`** / **`lifers_gate`** 相同 JSON；字体主题读 **`tools/vscodium_editor_defaults.json`**。**离线 Monaco 物料**（需联网一次）：`PYTHONPATH=. python tools/lifers_gui_host/fetch_offline_monaco.py`（产物在 **`tools/lifers_gui_host/static/vendor/`**，已 **gitignore**，打包 Kali 时若本机已下载会一并进 tar）。**内嵌窗口**：`pip install -r tools/lifers_gui_host/optional_requirements.txt` 后加 **`--webview`**。**一键全测**：`PYTHONPATH=. python scripts/lifers_run_all_checks.py` 或 **`scripts/lifers_run_all_checks.ps1` / `.sh`**（unittest + **eval/full_system_check.py**，桥接默认开、联网检查默认关）。可选：`python tools/fetch_upstream_vscodium_refs.py` 缓存上游 **product.json** 到 **`state/vscodium_upstream/`**。

**暂停 Kali 训练并全量同步仓库到远端**：`lifers/scripts/lifers_sync_pause_and_brain.ps1`（封装 **`push_brain_and_loop_kali.ps1 -SkipBootstrap`**，默认先 pause 再打包 scp）。

侧栏不见时：**Ctrl+Shift+P** → **`Lifers: 打开 Agents 侧栏`**。打开 **`lifers.code-workspace`** 时扩展解析 **`lifers`**。冒烟测试（便携根目录）：`powershell -File tools\test_agents_ui_smoke.ps1`。

### Workspace / 工作区

- **EN:** Deploy: **`bootstrap_lifers.bat`** or **`powershell -File tools/bootstrap_lifers.ps1`** — materialize **`lifers.code-workspace`**, **`run_integrated_bootstrap.py`**, **Lifers Agents UI**, **`sync_cursor_settings_to_vscodium.py`**, optional **`link_lifers_app.ps1`**, **`test_agents_ui_smoke.ps1`**. Tasks: **lifers: Bootstrap**, etc.

- **EN:** Open **`lifers.code-workspace`**. Defaults: **`config/integrated_layout.json`**. Override: **`config/workspace_custom.json`**, then **`python tools/materialize_integrated_workspace.py`**.
- **中文：** 打开 **`lifers.code-workspace`**。自定义同上。

### Spell check / 拼写检查

- **EN:** Install recommended extensions (includes **Code Spell Checker**). Workspace uses **`cSpell.language`: `en,zh-CN`**. If Chinese suggestions are thin, use command **CSpell: Show Spell Checker Info** or add the **Chinese (Simplified)** dictionary from the extension’s dictionary manager.
- **中文：** 安装工作区推荐扩展（含 **Code Spell Checker**）。工作区已设 **`cSpell.language`: `en,zh-CN`**。若中文提示不全，用命令面板里的 **CSpell: Show Spell Checker Info** 或通过扩展的字典管理添加 **简体中文** 词典。
