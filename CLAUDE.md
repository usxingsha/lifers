# CLAUDE.md — Lifers 全能AI项目引导

## 项目概述

Lifers 是一个本地化全能型 AI 智能体项目，基于 NumPy 实现的 Deep Transformer 模型。
核心特性：ALBERT 风格权重共享、RoPE 位置编码、Pre-LayerNorm、GELU 激活、AdamW 优化器。
训练在 Kali Linux VM (192.168.234.152) 上持续进行，Windows 为开发主站。

## 目录结构

```
lifers/                          # 仓库根
  lifers/                        # Python 包
    agent.py                     # 主Agent（prompt构建→推理→工具→记忆）
    deep_transformer.py          # Deep Transformer 核心（前向+生成）
    deep_transformer_train.py    # 训练模块（反向传播+AdamW）
    tools.py                     # 工具注册表（build_default_registry）
    local_brain.py               # 本地推理引擎
    config/                      # 配置文件
      stack.json                 # 主配置（模型/工具/记忆/插件）
      omni_ai_prompts.py         # 全能AI提示词中心
      lifers_ai_playbook_zh.md   # AI操作手册
    scripts/                     # CLI和运维脚本
      cli.py                     # 主CLI（流式输出/对话历史/Tab补全）
      lifers_chat.py             # 聊天CLI（6种模式）
      train_deep_escalate.py     # Kali渐进训练入口
      auto_sync_to_kali.sh       # Windows→Kali代码同步
    taskflow/                    # 任务流编排
    tools/plugins/               # 工具插件
      lifers_omni_skills/        # 全能AI技能包（7个工具）
  config/                        # 仓库级配置
  tools/                         # 工具目录（含插件）
  weights/                       # 训练权重（Git忽略但不忽略目录）
```

## 开发铁律（Bug预防第一）

### 修改前必做
1. **Read 再 Edit** — 任何 Edit 前必须先 Read 目标文件
2. **检查依赖** — 修改函数签名时 grep 所有调用点
3. **考虑训练进程** — Kali 上 train_deep_escalate.py 长期运行，修改核心文件前确认不会影响训练

### 修改后必做
1. **语法检查** — `python -c "import py_compile; py_compile.compile('file.py', doraise=True)"`
2. **导入测试** — 确保修改的模块可正常 import
3. **Git diff 审查** — commit 前仔细看 `git diff --stat` 和 `git diff` 关键部分

### 禁止操作
- 禁止覆盖 Kali 权重文件 (`*.npz`, `lifers_deep_transformer.json`, `.train_*`)
- 禁止在 .gitignore 中误屏蔽 Python 源文件
- 禁止修改 `stack.json` 时引入 trailing comma（JSON 不允许）
- 禁止在 generation 函数中省略 `if not ids: ids = [0]` 空输入保护
- 禁止 commit 包含密码/密钥/API Key
- 禁止 `git push --force` 到 main

## Git 工作流

```bash
# 免密推送（Windows credential.helper=manager 已配置）
git push origin main

# Kali 拉取
ssh kali@192.168.234.152 "cd /home/kali/lifers && git pull origin main"

# 快速同步（scp 关键文件，排除权重）
bash lifers/scripts/auto_sync_to_kali.sh
```

## Kali 训练监控

```bash
# 训练状态
ssh kali@192.168.234.152 "cat /home/kali/lifers/weights/.train_status.json"

# 训练进程
ssh kali@192.168.234.152 "ps aux | grep train_deep | grep -v grep"

# 权重文件
ssh kali@192.168.234.152 "ls -lh /home/kali/lifers/lifers/weights/"
```

## 关键架构约束

1. **工具注册唯一入口**: `lifers/tools.py → build_default_registry()`
2. **提示词唯一入口**: `lifers/config/omni_ai_prompts.py`
3. **配置唯一入口**: `config/stack.json`
4. **插件注册**: `tools/plugins/<name>/plugin.py → register_plugin_tools(registry, root)`
5. **前向传播**: `deep_transformer.py → forward_deep()` — 3个入口参数模式
6. **生成函数**: `generate_text()` 和 `cli.py:_generate()` 都必须有空输入保护

## 已知陷阱（避免重复踩坑）

| 陷阱 | 症状 | 修复 |
|------|------|------|
| .gitignore `/lifers` 屏蔽 | 源文件不入库 | `!/lifers/` 负向排除 |
| 空prompt崩溃 | ValueError zero-size array | `if not ids: ids = [0]` |
| JSON trailing comma | stack.json 解析失败 | JSON不允许尾逗号 |
| Kali root权限污染 | git操作被拒 | `chown -R kali:kali .git/` |
| scp覆盖训练日志 | 幽灵文件写入 | 同步时排除 logs/ |
| RoPE缓存未复用 | 反向传播慢 | import _rope_cache |

## 中文回复

用户使用中文，所有回复用简体中文。
