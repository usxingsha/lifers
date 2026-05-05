# Lifers

Portable **Lifers** root（目录名建议 **`lifers`**，与 GitHub 仓一致；智脑 `lifers_brain` + 配置与工具）。运行时以 `lifers_brain/config/stack.json` 为准。

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
| `third_party/openclaw` | `openclaw/openclaw` 子模块；能力域见 `lifers_brain/config/openclaw_manifest.json` |
| `third_party/claw_code_rust` | Kali **claw-code/rust** 并入；见 `lifers_brain/config/claw_code_rust_vendor.json` |
| `lifers_brain/config/lifers_vendor_map.json` | 总表与策略（以 Lifers 为主） |

不在此仓库进程内安装或启动 OpenClaw / claw-code 网关；云 API 不入 `stack.brain`。

**源码并入说明**：`third_party/openclaw`（git 子模块）与 `third_party/claw_code_rust`（Kali 并入）均在仓库内，用于对照与审计；Python 智脑 **不会** 自动调用其中的 npm/Rust 网关。对话默认 **本地**（`stack.json` → `remote_infer.enabled=false`，扩展 `lifers.remoteChat=false`），无需 NVIDIA 密钥即可聊天；需要云端大模型时再开启并配置环境变量密钥。

## 文档入口

- 日常运维 AI：`lifers_brain/config/lifers_ai_playbook_zh.md`（由 `stack.json` → `llm_ops` 注入）
- 合并联运：`config/integrated_layout.json`
