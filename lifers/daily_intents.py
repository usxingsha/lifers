"""日常用语与任务类型对齐（无 agent 依赖，供 classify / Planner 共用）。"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# 否定 + 随后出现「做/写/游戏…」等：避免把「不要写游戏」误判为开发请求。
_BUILD_NEG = re.compile(
    r"(不要|请勿|别做|别写|不做|不写|取消)(?:.{0,24}(写|做|实现|开发|创建|搭建|游戏|程序|项目|网站|应用|脚本))",
    re.UNICODE,
)


def cn_web_query_line(text: str) -> Optional[str]:
    """若整句为「中文显式检索」则返回查询串，否则 None（与 `search …` 语义对齐）。"""
    s = text.strip()
    if not s:
        return None
    for head in ("帮我搜索", "帮我搜一下", "帮我查查", "帮我查询"):
        if s.startswith(head):
            q = s[len(head) :].strip()
            return q or None
    for head in ("搜索", "搜一下", "查查", "查询"):
        if s.startswith(head):
            q = s[len(head) :].strip()
            if q:
                return q
    if s.startswith("搜") and len(s) >= 2 and s[1].isspace():
        q = s[1:].strip()
        return q or None
    return None


def looks_like_build_or_code_request(text: str) -> bool:
    """实现 / 开发 / 做游戏或小型项目：宜走完整管线（Planner + 工具链），而非 CHAT_QUICK。"""
    s = text.strip()
    if not s or len(s) > 280:
        return False
    if _BUILD_NEG.search(s):
        return False
    low = s.lower()
    verbs = ("做", "写", "实现", "开发", "创建", "搭建", "生成", "弄", "编个", "写个", "给个")
    nouns = (
        "游戏",
        "程序",
        "软件",
        "网站",
        "网页",
        "应用",
        "项目",
        "脚本",
        "插件",
        "组件",
        "页面",
        "接口",
        "贪吃蛇",
        "俄罗斯方块",
        "飞机大战",
    )
    v_hit = any(x in s for x in verbs)
    n_hit = any(x in s for x in nouns)
    if v_hit and n_hit:
        return True
    if s.startswith(("做个", "写一个", "写个", "做一个", "搞个", "弄个", "来做个", "来做一个")):
        if n_hit or "小程序" in s or "html" in low or "css" in low or "script" in low:
            return True
    en_stems = ("build a ", "build an ", "make a ", "create a ", "implement ", "develop a ", "code a ")
    if any(low.startswith(st) for st in en_stems):
        if any(k in low for k in (" game", " app", " site", " script", " api", " program")):
            return True
    return False


def looks_like_rewrite_or_longform(text: str) -> bool:
    """总结 / 续写 / 翻译 / 诗词对联等创作：宜走完整管道与工具链。"""
    s = text.strip()
    if not s:
        return False
    low = s.lower()
    if low.startswith("translate ") or s.startswith("翻译"):
        return len(s) > 3
    heads = (
        "总结",
        "续写",
        "改写",
        "润色",
        "扩写",
        "缩写",
        "概括",
        "提炼要点",
        "写一段",
        "写一篇",
        # 诗词 / 韵文 / 故事：短句也易在 CHAT_QUICK 上采样塌缩，走完整管线更稳
        "写诗",
        "写首诗",
        "作诗",
        "作一首诗",
        "赋诗",
        "来首诗",
        "帮我写诗",
        "帮我写一首诗",
        "写首词",
        "填词",
        "写副对联",
        "写一副对联",
        "写春联",
        "写首歌",
        "写歌词",
        "写个故事",
        "写童话",
        "写寓言",
    )
    if any(s.startswith(h) for h in heads):
        return True
    # 「来一首…」「写一首…」等变体
    if re.match(r"^(来|写|作|帮我写|帮我作)一?[首篇段]", s):
        if any(k in s for k in ("诗", "词", "对联", "歌", "曲", "赋", "文", "故事", "童话")):
            return True
    return False


def looks_like_remember_or_todo_surface(text: str) -> bool:
    """备忘 / 待办类表面语句：走完整管道便于写入 KB 或工具链。"""
    s = text.strip()
    if not s:
        return False
    heads = ("备忘", "待办", "提醒我", "提醒我明天", "提醒我后天", "记事", "记录一下", "帮我记")
    return any(s.startswith(h) for h in heads)


def parse_workspace_write_message(text: str) -> Optional[Tuple[str, str]]:
    """
    首行 `rel_write|workspace_write|self_write <rel_path>`，自第二行起为完整文件正文。
    用于工具链直接改 LIFERS_ROOT 下任意相对路径（含自身源码）。
    """
    raw = text.strip()
    if "\n" not in raw:
        return None
    first, rest = raw.split("\n", 1)
    fl = first.strip()
    low = fl.lower()
    for prefix in ("rel_write ", "workspace_write ", "self_write "):
        if low.startswith(prefix):
            rel = fl[len(prefix) :].strip()
            if rel:
                return rel, rest
    return None
