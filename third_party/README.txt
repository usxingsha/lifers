OpenClaw 上游：本仓库以 **git 子模块** 纳入 third_party/openclaw（见仓库根 .gitmodules、lifers_brain/config/openclaw_upstream_vendor.json）。克隆 Lifers 后请执行：git submodule update --init --depth 1。亦可仅用 scripts/vendor_openclaw_reference.ps1 / vendor_openclaw_reference.sh 在无子模块时拉浅克隆。不安装、不执行上游 npm/网关运行时，仅能力域与源码对照（与 openclaw_manifest.json 一致）。

claw-code/rust 源码并入（vendor）：目录 third_party/claw_code_rust 来自 Kali 上 claw-code/rust 工作区，排除 target 与会话缓存后打包同步；见 lifers_brain/config/claw_code_rust_vendor.json 与 third_party/claw_code_rust/LIFERS_MERGE.md。不在 Lifers 内启用其 HTTP/API 运行时。

双 vendor 总表：lifers_brain/config/lifers_vendor_map.json（以 Lifers 为主：运行时真相在 stack.json / lifers_brain）。Windows→Kali 不经 git：lifers_brain/scripts/package_rs_for_kali.ps1 从 **仓库根** 打 lifers_kali.tar.gz（含 lifers_brain、third_party 检出源码、tools、根级配置等），push_brain_and_loop_kali.ps1 负责 pause、scp、解压合并到 Kali 的 ~/lifers。third_party 仍随本仓 git 跟踪；若只要 claw Rust 可单独 scp/rsync。

与 PyTorch、Hugging Face、vLLM、llama.cpp 等工业栈的关系写在 lifers_vendor_map.json 的 external_ai_frameworks_posture_* 字段：本仓不整树复制那些上游；训练与权重格式以 lifers_brain 为准，大权重文件不入版本库。
