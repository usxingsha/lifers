OpenClaw 上游：本仓库以 **git 子模块** 纳入 third_party/openclaw（见仓库根 .gitmodules、lifers_brain/config/openclaw_upstream_vendor.json）。克隆 Lifers 后请执行：git submodule update --init --depth 1。亦可仅用 scripts/vendor_openclaw_reference.ps1 / vendor_openclaw_reference.sh 在无子模块时拉浅克隆。不安装、不执行上游 npm/网关运行时，仅能力域与源码对照（与 openclaw_manifest.json 一致）。

claw-code/rust 源码并入（vendor）：目录 claw_code_rust/ 来自 Kali ~/claw-code/rust，排除 target 与会话缓存后打包同步；见 lifers_brain/config/claw_code_rust_vendor.json 与 claw_code_rust/LIFERS_MERGE.md。不在 Lifers 内启用其 HTTP/API 运行时。

双 vendor 总表：lifers_brain/config/lifers_vendor_map.json（以 Lifers 为主：运行时真相在 stack.json / lifers_brain）。package_rs_for_kali.ps1 仅打包 lifers_brain 子树；third_party 随整仓 git；若只要 claw Rust 可单独 scp/rsync。
