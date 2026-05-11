# 在 Kali 上训练并取回权重

## 1. Windows 打包

在 Lifers 根下的 `scripts` 执行（仓库中多为 `lifers_brain\scripts`）：

```powershell
.\package_rs_for_kali.ps1
```

得到 `dist\lifers_kali.tar.gz`（相对当前 Lifers 根目录）。

### 1b. 一键推送到 Kali 并开跑「全套」长期训练（推荐）

在 **`lifers_brain\scripts`** 下（需本机 `ssh`/`scp` 可免密登录 Kali）：

```powershell
.\push_brain_and_loop_kali.ps1
```

会做：`package_rs_for_kali.ps1` → `scp` 到 Kali → 解压到 **`~/lifers/lifers_brain`**（**tar 不含 `weights/`，不覆盖 Kali 上已有大权重**）→ 执行 **`remote_kali_bootstrap_train_loop.sh`**：

- 刷新 **`weights/lifers_markov.json`**
- **`weights/.train_control`** 设为 **`run`**
- 在 **tmux 会话 `lifers-stack`** 里跑 **`kali_train_escalate_loop.sh`**（`LIFERS_ESCALATE_UNLIMITED=1`、`LIFERS_RAMP_MAX_ITERS=999999`，断 SSH 仍继续）

Kali 上查看：`tmux attach -t lifers-stack`，日志：`tail -f ~/lifers/lifers_full_stack.log`。仅解压不启动：`.\push_brain_and_loop_kali.ps1 -SkipBootstrap`。

若 tmux 会话已存在，引导脚本**不会**重复新建，避免顶掉正在跑的任务；需先 `tmux kill-session -t lifers-stack` 或换 `LIFERS_TRAIN_TMUX_SESSION`。

## 2. 传到 Kali

```bash
scp dist/lifers_kali.tar.gz user@kali-host:~/
```

## 3. Kali 解压并训练

### 3a. 快速（默认步数少，与旧版一致）

```bash
mkdir -p ~/lifers && tar -xzf ~/lifers_kali.tar.gz -C ~/lifers
cd ~/lifers/lifers_brain
chmod +x scripts/kali_train_weights.sh
LIFERS_KALI_TRAIN_MODE=fast bash scripts/kali_train_weights.sh
```

### 3b. 尽量跑全（推荐在 Kali 上）

一键安装 `python3`（若缺）、默认 **`LIFERS_KALI_TRAIN_MODE=full`**（较大 `TT_STEPS` / 词表与维度，仍为本仓库自带的小 Transformer，非大模型）：

```bash
mkdir -p ~/lifers && tar -xzf ~/lifers_kali.tar.gz -C ~/lifers
bash ~/lifers/lifers_brain/scripts/kali_install_full_train.sh
```

若压缩包不在默认路径，可传入：

```bash
bash ~/lifers/lifers_brain/scripts/kali_install_full_train.sh /path/to/lifers_kali.tar.gz
```

更激进（耗时更长）：`LIFERS_KALI_TRAIN_MODE=extreme bash .../kali_install_full_train.sh`

### `iter 14/999999` 是不是「从头训练」？

不是「从随机权重第 1 个 epoch 重来」的意思。`train_lifers_escalate.py` 里 **`iter N` 是 ramp 档位**（每档增大词表 / `d_model` / `steps` 等），`999999` 是 `LIFERS_RAMP_MAX_ITERS` 的安全上限。若已有 `weights/lifers_transformer.json` 且与当前 ramp 形状一致，会 **warm-start 并打印 `resume tier iter …`**；若语料或 `LIFERS_ESCALATE_*` 与权重不匹配，会 **从 tier 1 冷启动**（日志里会有说明）。单次 Python 进程内会连续跑多档，直到 OOM、`stop`、或你 `pause`。

可选：训练完成后注册 **用户 systemd 开机再跑一遍**（会覆盖 `weights/*.json`，适合你想每次开机刷新权重的场景）：

```bash
bash ~/lifers/lifers_brain/scripts/kali_install_full_train.sh --enable-boot
```

然后按需执行（无登录也跑权重训练）：

```bash
sudo loginctl enable-linger "$USER"
systemctl --user start lifers-kali-train.service   # 可选立即试跑
journalctl --user -u lifers-kali-train.service -e
```

环境变量说明：

- **`LIFERS_KALI_TRAIN_MODE`**：`fast` | `full` | `extreme`（`kali_train_weights.sh` 在未设置时默认为 **`fast`**；`kali_install_full_train.sh` 默认设为 **`full`**）。
- **Transformer 训练量**（任意模式均可覆盖）：`TT_STEPS`、`TT_VOCAB`、`TT_DMODEL`、`TT_DFF`、`TT_MAXSEQ`。
- **`LIFERS_TARGET_PARAM_B`**（默认 **20**）：`full` / `extreme` 且未设 `LIFERS_KALI_RAMP=0` 时走 **`train_lifers_escalate.py`**，按近似「浮点参数量」逐级放大直至达到目标、**MemoryError** 或迭代上限（名义 20B，实际受内存与纯 Python 算力限制）。
- **`LIFERS_KALI_RAMP=0`**：强制单次 `train_transformer_weights.py`，不逐级放大。
- **`LIFERS_ESCALATE_HONOR_TT_STEPS=1`**：ramp 每档沿用当前环境的 `TT_STEPS`/`TT_*`（与 `kali_train_weights` 的 full 预设叠加时可能极慢）。

产物：

- `weights/lifers_markov.json`
- `weights/lifers_transformer.json`

## 4. 拷回 Windows

将上述两个文件覆盖到本机仓库的 `lifers_brain\weights\`，与 `config/stack.json` 中 `brain.weights`（`lifers` 产品与 **transformer** 后端共用 `lifers_transformer.json`）一致即可。

完整流水线（含评测闸门）在 Linux 上也可跑：`python3 scripts/run_pipeline.py`（若 `exp_*` 闸门失败属预期，可只用 `kali_train_weights.sh` 生成权重）。

## 5. 运营速查（把流程「教给」Kali 操作者）

Kali 本机一页版（随仓库同步）：**`scripts/LIFERS_KALI_CHEATSHEET_zh.txt`**（`cat` 即可）。

| 目的 | 命令 |
|------|------|
| 看训练 | `tmux attach -t lifers-stack` |
| 看日志 | `tail -f ~/lifers/lifers_full_stack.log` |
| 暂停 / 继续 / 停 | `cd ~/lifers/lifers_brain` → `bash scripts/lifers_train_ctl.sh` + 参数 `pause` / `run` / `stop`；多路径同停可用 `bash /tmp/remote_pause_lifers_train.sh`（与 Windows 同步脚本一致） |
| Windows 推代码并带跑循环 | `powershell -File scripts\push_brain_and_loop_kali.ps1`（**先** `remote_pause_lifers_train.sh` 暂停，再 scp 解压；`-SkipPauseTrainFirst` 可跳过） |
| Windows 从 Kali 拉权重 | `.\sync_weights_from_kali.ps1`（**先** 远端 pause；`-SkipTrainPause` 跳过） |
| 停旧 tmux、备份主权重、合并 checkpoint（若有） | `bash scripts/remote_stop_old_merge_weights.sh` |
| 能力队列 | `python3 scripts/lifers_capability_queue.py` 子命令 `show` / `env` / `advance` |

**不要**对 `pgrep` 里看到的 **tmux 父进程** 随便 `kill`，否则会干掉整个 tmux server，连 `lifers-stack` 一起没；只杀会话用 **`tmux kill-session -t 会话名`**。

### 「离正常使用」要多久（务实预期）

- **编辑器里点得开、桥接能跑、工具能调**：装/同步 **Lifers Agents UI** 后 **Reload Window** → 通常 **几分钟到十几分钟**。
- **日常对话、编程像主流大模型那样顺手**：主要不取决于 `lifers_transformer.json` 训多久，而取决于 **`stack.json` → `llm_ops`**（或同类）是否接上 **外部大模型** 与网络；配置正确的话，往往 **一小时内** 就能当「主力」日常用。
- **只靠本仓库 TinyTransformer 权重变好**：纯 Python 小模型 + 当前语料，**数天～数周** 可能有「可感」的增量，但**天花板远低于** GPT 类产品；把它当 **离线补充 / 路由 / 记忆侧** 更现实。
- **Kali 上 escalate 一直跑**：属于 **长期占用**；单档迭代从日志里的 `iter …/999999` 与每档 `steps` 可粗看进度，**没有**「再训 X 小时一定达标」的固定公式。

结论：**要「正常当主力助手用」→ 优先把 API/代理链路打通；训练权重是并行加分项。**
