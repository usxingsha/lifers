Lifers Editor（RS 专用壳）
=======================

这不是从零写的「新编辑器内核」，而是用系统里已安装的 **VSCodium / VS Code / Cursor**，
加上 **独立配置目录**（user-data-dir）与 **仅含 Lifers 扩展的 extensions-dir**，
得到一块单独给 lifers 用的桌面入口，行为上接近「专用小发行版」。

启动
----
Windows（在 lifers 仓库里）:
  powershell -ExecutionPolicy Bypass -File tools\lifers_editor\lifers-editor.ps1

Linux / Kali（在 lifers 根目录）:
  bash tools/lifers_editor/lifers-editor.sh

若提示找不到 codium/code/cursor，请先安装其一，例如 Kali:
  sudo apt update && sudo apt install -y vscodium

Kali 上「UI 在哪」
------------------
Lifers 界面是 **VS Code 类编辑器的扩展**，不是单独 .deb 程序名「rs」。

同步脚本安装后，扩展目录一般为:
  ~/.vscode/extensions/lifers.lifers-agents-ui-<版本>/
  ~/.vscode-oss/extensions/lifers.lifers-agents-ui-<版本>/

在 VSCodium 里 **打开文件夹** 指向你的 brain 根目录（例如）:
  /home/kali/lifers/lifers

然后:
  - 左侧活动栏图标 **「Lifers」** → Agents / 会话建立
  - 底部面板 **「Lifers 总控」**（算力与同步）
  - 命令面板搜索: Lifers / 总控

「RS 是否装在 Kali」
--------------------
通常有两部分:
  1) **lifers** 代码与权重: 由 tar/scp 解压到某路径（如 /home/kali/lifers/lifers）
  2) **lifers-agents-ui** 扩展: 在上面的 extensions 目录；必须在图形/远程图形里打开 VSCodium 才看得见

清理旧版 UI 目录
----------------
Windows:
  powershell -File scripts\cleanup_old_lifers_agents_ui.ps1

一键同步（删旧版 + 全路径安装 + 修复 extensions.json，含 rs\data、Flatpak 常见目录）:
  powershell -File scripts\sync_lifers_agents_ui_windows_kali.ps1
  （仅本机可加 -SkipKali；不新开 Cursor 加 -SkipCursor）
