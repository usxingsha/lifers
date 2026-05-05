OpenClaw 上游：本仓库以 **git 子模块** 纳入 third_party/openclaw（见仓库根 .gitmodules、lifers_brain/config/openclaw_upstream_vendor.json）。克隆 Lifers 后请执行：git submodule update --init --depth 1。亦可仅用 scripts/vendor_openclaw_reference.ps1 / vendor_openclaw_reference.sh 在无子模块时拉浅克隆。不安装、不执行上游 npm/网关运行时，仅能力域与源码对照（与 openclaw_manifest.json 一致）。

claw-code/rust 源码并入（vendor）：目录 claw_code_rust/ 来自 Kali ~/claw-code/rust，排除 target 与会话缓存后打包同步；见 lifers_brain/config/claw_code_rust_vendor.json 与 claw_code_rust/LIFERS_MERGE.md。不在 Lifers 内启用其 HTTP/API 运行时。

双 vendor 总表：lifers_brain/config/lifers_vendor_map.json（以 Lifers 为主：运行时真相在 stack.json / lifers_brain）。package_rs_for_kali.ps1 仅打包 lifers_brain 子树；third_party 随整仓 git；若只要 claw Rust 可单独 scp/rsync。

主流开源 AI 生态（PyTorch / HF / vLLM / llama.cpp 等）的 **官方源、许可证与推荐集成方式** 见 lifers_brain/config/oss_ai_ecosystem_manifest.json（**不**整仓 vendor 进 Lifers）。可选浅克隆参考：`lifers_brain/scripts/vendor_oss_ai_reference_fetch.ps1 -IncludeIds nanoGPT` → third_party/_refs/（默认 .gitignore）。
