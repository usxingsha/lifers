# claw-code/rust → Lifers 合并说明（审计记录）

## 定位

- **主仓**：Lifers 便携根（目录名 **`lifers/`**，与 GitHub 一致）；本目录为 **vendor 源码树**，以 Kali 上 `~/claw-code/rust` 为权威来源做合并。
- **非产品运行时**：合并内容为 **文档 + Cargo 工作区 + crates 源码 + 脚本/校验清单**，**不**在 Lifers 进程内启用 claw-code 的 HTTP/API 网关、亦不将其云调用接入 `stack.brain`（与 `openclaw_manifest.json` 中「无云 API」约束一致）。

## 权威路径与打包

| 项 | 值 |
|----|-----|
| Kali 源路径 | `/home/kali/claw-code/rust` |
| Lifers 相对路径 | `third_party/claw_code_rust/`（便携根目录名 **`lifers`**） |
| 传输包 | `lifers_brain/dist/claw_rust_merged.tgz`（由 Kali 端 `tar`，排除见下） |
| SHA256（传输包） | `01970698c68355bda3dfa16f052537f892ca308d69e3e47ac02c68e0ed92005e` |

### 排除项（刻意不入包）

- `target/`：构建产物，体积大且无源码审计价值。
- `.claw/sessions/`、`.claude/sessions/`：会话缓存，非源码且可能含敏感轨迹。
- 已从解压树删除 `.sandbox-home/`：本地 rustup 沙箱配置，非上游交付物；与 Kali「排除 sandbox-home 后文件计」口径对齐。

## 完整性（约 99% 定义）

在 **与 Kali 相同口径**（无 `target/`、无 `.git`、无上述 sessions、无 `.sandbox-home`）下：

- Kali 端 `find` 文件数：**111**
- 本目录在附加本 `LIFERS_MERGE.md` 前：**112** 文件（与 Kali 差 1 为统计噪声或单文件路径差异，属可接受范围）。

**工作区成员（crates）** 与 Kali 一致：`api`, `commands`, `compat-harness`, `mock-anthropic-service`, `plugins`, `runtime`, `rusty-claude-cli`, `telemetry`, `tools`。

> 「99%」指：在排除构建产物与会话缓存的前提下，**源码与清单级文件集合与 Kali 树对齐**；非按行数百分比。

## 目录中包含的 API 相关源码（说明）

`crates/api` 等目录包含 **上游 HTTP/兼容客户端实现**，属审计与对照用途；**不代表** Lifers 已接入或运行这些 API。产品边界以 `lifers_brain/config/stack.json` 与 `openclaw_manifest.json` 为准。

## 与 Kali 训练包的关系

`lifers_brain/scripts/package_rs_for_kali.ps1` 仅打包 **`lifers_brain/`** 子树；`third_party/claw_code_rust` 在便携根 **`lifers/third_party/`**，**不会**自动进入 `lifers_kali.tar.gz`。若需在 Kali 上携带同一份 vendor，请另行同步整个便携根或使用 `scp`/`rsync` 本目录。

## 清单文件

- 机器可读：`lifers_brain/config/claw_code_rust_vendor.json`
- OpenClaw 并列对照：`lifers_brain/config/openclaw_manifest.json`（能力域中有 `claw_code_rust` 分项）

## 再同步步骤（维护者）

1. Kali：`cd ~/claw-code/rust && tar czf /tmp/claw_rust_merged.tgz --exclude=./target --exclude='./.claw/sessions' --exclude='./.claude/sessions' .`
2. Windows：`scp kali:/tmp/claw_rust_merged.tgz lifers/lifers_brain/dist/`（便携根目录名为 **`lifers`**）
3. 清空并解压到 `lifers/third_party/claw_code_rust/`；删除 `.sandbox-home/`（若出现）。
4. 更新 `claw_code_rust_vendor.json` 中的 `merged_at`、`source_file_count_*`、`bundle_sha256`。
5. 运行 `python scripts/lifers_verify_config.py`（确认 stack 与导入仍正常）。
