# Lifers

Portable **Lifers** root（目录名建议 **`lifers`**，与 GitHub 仓一致；智脑 `lifers` + 配置与工具）。运行时以 `lifers/config/stack.json` 为准。

若本机仍为旧文件夹名 **`rs`**：请先关闭占用该路径的编辑器，再运行 **`powershell -File tools/rename_portable_folder_rs_to_lifers.ps1`**（或在 **`Desktop\curku`** 下手动把 **`rs`** 重命名为 **`lifers`**）。

## 克隆

本仓库包含 **git 子模块**（OpenClaw 上游对照）：

```bash
git clone --recurse-submodules <你的仓库 URL>
# 若已克隆未带子模块：
git submodule update --init --depth 1
```

## 双 vendor 对照（非运行时）

| 路径 | 说明 |
|------|------|
| `third_party/openclaw` | `openclaw/openclaw` 子模块；能力域见 `lifers/config/openclaw_manifest.json` |
| `third_party/claw_code_rust` | 以 Kali **`~/lifers/third_party/claw-code/rust`** 为权威对照；见 `lifers/config/claw_code_rust_vendor.json` |
| `lifers/config/lifers_vendor_map.json` | 总表与策略（以 Lifers 为主） |

不在此仓库进程内安装或启动 OpenClaw / claw-code 网关；云 API 不入 `stack.brain`。

**源码并入说明**：`third_party/openclaw`（git 子模块）与 `third_party/claw_code_rust`（Kali 并入）均在仓库内，用于对照与审计；Python 智脑 **不会** 自动调用其中的 npm/Rust 网关。对话默认 **本地**（`stack.json` → `remote_infer.enabled=false`，扩展 `lifers.remoteChat=false`），无需 NVIDIA 密钥即可聊天；需要云端大模型时再开启并配置环境变量密钥。

**本地模型**：推理只认 **`lifers/weights/lifers_transformer.json`**（及 stack 配置路径）；已删除仓库内玩具后备 **`tiny_transformer_v001.json`**。Kali/本机 **`train_lifers_escalate.py`** 持续写入同一文件时，对话侧按文件更新时间自动加载新权重（边训边用）。**Cursor**：仓库含 **`.cursor/rules/*.mdc`**（免费、可版本化的项目规则）；不捆绑 Cursor 商业安装包。

## 文档入口

- 日常运维 AI：`lifers/config/lifers_ai_playbook_zh.md`（由 `stack.json` → `llm_ops` 注入）
- 合并联运：`config/integrated_layout.json`（**便携仓根** `lifers/config/`，与 `tools/` 同级；**已跟踪**，勿加入 `.gitignore`）
- 持久 HTTP 桥（与 stdin `agent_bridge_once` 同款 JSON）：`lifers/scripts/lifers_gate.py`（默认 `127.0.0.1:55555`，勿对公网暴露）
- 任务流与审计对照：`lifers/taskflow/FLOW.md`（含 **外部审计陈述对照** 表）

## 路径核对（防「文档写了但仓库没有」）

克隆后下列文件应存在（`git ls-files` 可查）：

| 用途 | 路径 |
|------|------|
| 联运物化 | `config/integrated_layout.json` |
| OpenClaw 清单 | `lifers/config/openclaw_manifest.json` |
| Rust vendor 清单 | `lifers/config/claw_code_rust_vendor.json` |
| 总 vendor 策略 | `lifers/config/lifers_vendor_map.json` |
| 运维长文 | `lifers/config/lifers_ai_playbook_zh.md` |

## rs0 是什么

多根工作区 **`lifers.code-workspace`** 的首文件夹 **`./rs0`**：可选本地笔记/小实验目录，**允许为空**；运行时与脚本以 **`lifers`** 为准。

## VS Code / VSCodium 扩展

根目录 **`.vscode/extensions.json`** 含 **`lifers.lifers-agents-ui`**。若市场或 Open VSX 未上架该 ID，请用 **`tools/install_agents_extension.ps1`**，或在编辑器中选择 **从文件夹安装扩展** → `lifers/extensions/lifers-agents-ui`。
