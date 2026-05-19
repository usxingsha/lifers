"""
全面重建所有支柱训练数据 — 高组合多样性引擎
目标：每种数据独特模式 >30%，总量达十万级以上
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Callable

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)

# ═══════════════════════════════════════════════════════════
# 超大规模基础素材库
# ═══════════════════════════════════════════════════════════

# 主体 (250个)
S = [
    "我", "你", "他", "她", "我们", "你们", "他们", "大家", "同学", "同事",
    "老师", "学生", "医生", "工程师", "设计师", "程序员", "研究员", "经理",
    "客户", "朋友", "家人", "孩子", "老人", "团队", "部门", "公司", "组织",
    "用户", "消费者", "病人", "旅客", "司机", "厨师", "服务员", "警察",
    "军人", "艺术家", "作家", "音乐家", "科学家", "数学家", "物理学家",
    "化学家", "生物学家", "哲学家", "历史学家", "心理学家", "运动员",
    "教练", "志愿者", "领导", "员工", "秘书", "会计", "律师", "法官",
    "农民", "工人", "商人", "企业家", "投资人", "分析师", "顾问", "导师",
    "学姐", "学长", "室友", "邻居", "陌生人", "访客", "会员", "粉丝",
    "博主", "主播", "达人", "新手", "高手", "专家", "行家", "前辈",
    "搭档", "伙伴", "甲方", "乙方", "校长", "主任", "教授", "博士",
    "硕士", "创始人", "CEO", "总监", "主管", "组长", "代表", "代理人",
    "合作方", "供应商", "客户经理", "项目负责人", "技术主管", "产品经理",
    "运维人员", "测试人员", "架构师", "数据科学家", "算法工程师",
]

# 动词/动作 (200个)
V = [
    "分析", "评估", "检查", "测试", "验证", "确认", "比较", "选择", "决定", "计划",
    "设计", "开发", "实现", "部署", "维护", "优化", "改进", "升级", "迁移", "整合",
    "收集", "整理", "处理", "存储", "检索", "展示", "报告", "记录", "跟踪", "监控",
    "学习", "研究", "探索", "发现", "理解", "掌握", "应用", "实践", "总结", "反思",
    "讨论", "交流", "协商", "协调", "组织", "安排", "分配", "调度", "执行", "完成",
    "创建", "编辑", "修改", "删除", "保存", "备份", "恢复", "同步", "分享", "发布",
    "搜索", "查找", "浏览", "阅读", "撰写", "翻译", "校对", "审核", "批准", "拒绝",
    "购买", "销售", "投资", "预算", "结算", "支付", "收款", "理财", "计算", "估算",
    "锻炼", "跑步", "游泳", "健身", "休息", "烹饪", "修理", "清洁", "搬家", "旅行",
    "唱歌", "跳舞", "画画", "摄影", "写作", "观影", "听音乐", "玩游戏", "聊天", "逛街",
    "打电话", "发消息", "留言", "评论", "点赞", "转发", "收藏", "订阅", "关注", "注册",
    "登录", "设置", "配置", "安装", "启动", "关闭", "重启", "调试", "编译", "测试",
    "开车", "骑车", "步行", "导航", "定位", "预约", "排队", "等待", "准备", "开始",
    "教授", "指导", "培训", "辅导", "咨询", "建议", "推荐", "提醒", "警告", "通知",
    "编程", "编码", "提交", "合并", "部署", "监控", "报警", "恢复", "扩容", "缩容",
]

# 名词/对象 (250个)
N = [
    "项目", "任务", "工作", "方案", "计划", "报告", "文档", "代码", "数据",
    "系统", "平台", "工具", "框架", "模型", "算法", "网络", "服务器", "数据库",
    "接口", "服务", "配置", "环境", "版本", "分支", "补丁", "日志", "监控",
    "问题", "错误", "异常", "风险", "机会", "需求", "功能", "特性", "性能",
    "安全", "质量", "效率", "成本", "资源", "时间", "进度", "状态", "结果",
    "会议", "邮件", "消息", "通知", "提醒", "日程", "日历", "待办", "清单",
    "话题", "主题", "问题", "答案", "建议", "想法", "观点", "理论", "方法",
    "设备", "硬件", "软件", "应用", "程序", "模块", "组件", "库", "包",
    "文件", "图片", "视频", "音频", "文章", "新闻", "书籍", "论文", "专利",
    "产品", "商品", "服务", "订单", "付款", "退款", "发票", "合同", "协议",
    "课程", "考试", "作业", "实验", "项目", "实习", "简历", "面试", "offer",
    "食谱", "菜谱", "食材", "调料", "厨房", "餐具", "家具", "电器", "装饰",
    "花园", "植物", "宠物", "玩具", "游戏", "电影", "音乐", "画作", "照片",
    "手机", "电脑", "平板", "耳机", "音箱", "相机", "镜头", "三脚架", "配件",
    "运动", "健身", "瑜伽", "跑步", "游泳", "骑行", "登山", "滑雪", "冲浪",
    "疾病", "症状", "药物", "治疗", "康复", "保健", "营养", "饮食", "睡眠",
]

# 形容词/修饰 (150个)
A = [
    "快速", "缓慢", "仔细", "认真", "耐心", "积极", "主动", "努力", "高效",
    "准确", "精确", "完整", "全面", "系统", "深入", "广泛", "及时", "准时",
    "成功", "顺利", "轻松", "困难", "完美", "出色", "优秀", "良好", "正常",
    "重要", "关键", "核心", "主要", "基本", "高级", "初级", "中级", "顶级",
    "简单", "复杂", "容易", "困难", "方便", "麻烦", "实用", "有效", "无效",
    "清晰", "模糊", "明确", "含糊", "强烈", "微弱", "明显", "隐蔽", "直接",
    "间接", "正式", "非正式", "公开", "秘密", "安全", "危险", "稳定", "不稳定",
    "灵活", "僵硬", "开放", "保守", "创新", "传统", "现代", "古典", "流行",
    "冷门", "热门", "免费", "付费", "在线", "离线", "同步", "异步", "实时",
    "延迟", "静态", "动态", "线性", "非线性", "对称", "不对称", "平衡", "失衡",
    "有趣", "无聊", "精彩", "平淡", "令人兴奋", "让人沮丧", "鼓舞人心", "令人担忧",
    "可靠", "不可靠", "可信", "可疑", "合理", "荒谬", "务实", "空想", "可行",
]

# 领域/话题 (200个)
D = [
    "人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉", "数据科学",
    "前端开发", "后端开发", "移动开发", "游戏开发", "嵌入式开发", "Web开发",
    "云计算", "大数据", "物联网", "区块链", "网络安全", "隐私保护", "密码学",
    "数学", "物理", "化学", "生物", "天文", "地理", "气象", "环境科学",
    "医学", "药学", "护理", "公共卫生", "心理学", "社会学", "人类学", "考古学",
    "经济学", "金融", "投资", "贸易", "管理学", "市场营销", "人力资源", "会计",
    "法律", "政策", "外交", "教育", "科研", "创新", "创业", "风险投资",
    "文学", "艺术", "音乐", "电影", "戏剧", "舞蹈", "建筑设计", "工业设计",
    "体育", "健身", "旅游", "美食", "时尚", "美容", "家居", "园艺",
    "历史", "哲学", "宗教", "神话", "民俗", "传统文化", "语言学", "翻译学",
    "交通", "能源", "农业", "制造业", "物流", "供应链", "零售", "电商",
    "日常生活", "工作效率", "学习方法", "沟通技巧", "时间管理", "压力管理",
    "团队协作", "领导力", "创新思维", "问题解决", "决策分析", "风险管理",
    "气候变化", "可持续发展", "再生能源", "生态保护", "碳中和", "环保",
    "基因编辑", "脑科学", "量子计算", "纳米技术", "材料科学", "空间探索",
    "机器人", "自动驾驶", "智能家居", "可穿戴设备", "虚拟现实", "增强现实",
]

# 数量/程度词 (60个)
Q = [
    "一个", "两个", "三个", "几个", "很多", "大量", "少量", "一些", "某些",
    "大多数", "半数", "全部", "部分", "整批", "整套", "一系列", "一整套",
    "大幅", "小幅", "显著", "轻微", "大幅度的", "小幅度的", "显著的", "轻微的",
    "百分之十", "百分之二十", "百分之三十", "一半", "三分之一", "两倍", "三倍",
    "数十个", "数百个", "数千个", "上万个", "数万个", "很多个", "若干个",
    "足够", "不足", "过多", "过少", "适度", "充分", "完全", "基本",
]

# 时间词 (80个)
T = [
    "今天", "明天", "昨天", "前天", "后天", "本周", "下周", "上周", "这周",
    "本月", "下个月", "上个月", "今年", "明年", "去年", "这个季度", "上个季度",
    "早上", "上午", "中午", "下午", "傍晚", "晚上", "深夜", "凌晨", "午休时间",
    "周一", "周二", "周三", "周四", "周五", "周六", "周日", "周末", "工作日",
    "最近", "近来", "近期", "过去", "从前", "以前", "将来", "未来", "今后",
    "立刻", "马上", "尽快", "稍后", "待会儿", "等下", "过会儿", "再等等",
    "刚才", "刚刚", "不久前", "前几天", "前一阵子", "前些天", "去年底",
    "即将", "快要", "马上就要", "不久后", "不久将来", "近期内", "短期内",
]

# 连词/连接词 (40个)
C = [
    "而且", "并且", "另外", "此外", "同时", "与此同地", "另外一方面",
    "但是", "然而", "不过", "可是", "尽管如此", "虽然如此", "即使这样",
    "因此", "所以", "因而", "于是", "故此", "基于此",
    "因为", "由于", "鉴于", "考虑到", "出于", "源于",
    "如果", "假如", "若是", "倘若", "一旦", "万一",
    "只要", "只有", "除非", "无论", "不管", "尽管", "虽然",
]

# 语气词/句末词 (30个)
Y = [
    "吧", "呢", "啊", "哦", "嘛", "啦", "呀", "咯", "呗", "咚",
    "哈", "呵呵", "嘻嘻", "嘿嘿", "嗯嗯", "好的", "没问题", "行的",
    "当然", "确实", "真的", "必须的", "一定的", "肯定", "绝对",
]

# 情感词 (100个)
E = [
    "开心", "快乐", "高兴", "兴奋", "激动", "喜悦", "欣慰", "满足",
    "悲伤", "难过", "伤心", "痛苦", "失落", "沮丧", "绝望", "无助",
    "愤怒", "生气", "恼火", "不满", "烦躁", "焦虑", "紧张", "担忧",
    "害怕", "恐惧", "惊慌", "不安", "困惑", "迷茫", "疑惑", "好奇",
    "惊讶", "震惊", "意外", "感慨", "感动", "感激", "感恩", "温暖",
    "无聊", "厌倦", "疲惫", "劳累", "困倦", "麻木", "冷漠", "淡然",
    "期待", "盼望", "憧憬", "希望", "乐观", "积极", "自信", "坚定",
    "后悔", "遗憾", "愧疚", "自责", "羞愧", "尴尬", "难堪", "惭愧",
    "羡慕", "嫉妒", "崇拜", "敬佩", "尊敬", "欣赏", "仰慕", "爱慕",
    "思念", "怀念", "牵挂", "惦记", "想念", "眷恋", "不舍", "留恋",
    "孤独", "寂寞", "空虚", "彷徨", "犹豫", "轻松", "自在", "舒适",
    "惬意", "愉快", "舒心", "放松", "解脱", "庆幸", "感恩", "珍惜",
]

# 社交问候语 (20个)
GREETINGS = [
    "早上好", "上午好", "中午好", "下午好", "晚上好", "晚安", "你好", "您好",
    "嗨", "哈喽", "在吗", "在么", "好久不见", "最近怎么样", "别来无恙",
    "今天天气不错", "吃了吗", "忙什么呢", "周末愉快", "节日快乐",
]

# 协作请求 (30个)
COLLAB = [
    "帮忙", "协助", "配合", "支持", "一起", "合作", "协作", "共同",
    "帮个忙", "搭把手", "一起做", "合伙", "协同", "联手", "合力",
    "能不能帮我", "可以帮我吗", "方便帮忙吗", "麻烦你了", "拜托了",
]

# 场景位置 (80个)
LOC = [
    "办公室", "会议室", "家里", "学校", "图书馆", "咖啡厅", "实验室", "工厂",
    "医院", "商场", "公园", "健身房", "地铁上", "公交车上", "飞机上", "火车上",
    "线上", "视频中", "聊天群", "论坛", "社交媒体", "项目现场", "工地", "车间",
    "厨房", "客厅", "卧室", "阳台", "书房", "花园", "车库", "地下室",
    "体育馆", "游泳馆", "操场", "球场", "跑道", "山顶", "海边", "河畔",
    "街头", "路口", "广场", "站台", "候车室", "机场", "码头", "车站",
    "超市", "便利店", "药店", "餐厅", "酒店", "民宿", "青年旅舍", "露营地",
    "美术馆", "博物馆", "科技馆", "展览中心", "剧院", "音乐厅", "电影院", "网吧",
]

# ═══════════════════════════════════════════════════════════
# 组合引擎：生成海量独特中文句
# ═══════════════════════════════════════════════════════════

def pick(n: int = 1):
    """返回随机选择函数，每次调用返回新的随机元素"""
    return lambda arr: tuple(arr[hash(str(rng.random() + rng.random())) % len(arr)] for _ in range(n))

def r(arr):
    """随机选择一个元素"""
    return arr[hash(str(rng.random())) % len(arr)]


class SentenceBuilder:
    """中文句子组合构建器"""

    @staticmethod
    def s1() -> str:
        """简单陈述：主体+动词+对象"""
        return f"{r(S)}{r(V)}{r(N)}"

    @staticmethod
    def s2() -> str:
        """带修饰：主体+修饰+动词+对象"""
        return f"{r(S)}{r(A)}地{r(V)}{r(N)}"

    @staticmethod
    def s3() -> str:
        """带时间和地点"""
        return f"{r(T)}，{r(LOC)}里，{r(S)}{r(V)}{r(N)}"

    @staticmethod
    def s4() -> str:
        """带领域"""
        return f"关于{r(D)}，{r(S)}{r(V)}了{r(Q)}{r(N)}"

    @staticmethod
    def s5() -> str:
        """带情感"""
        return f"{r(S)}对{r(N)}感到{r(E)}，{r(C)}{r(V)}"

    @staticmethod
    def s6() -> str:
        """带数量"""
        return f"{r(S)}需要{r(Q)}{r(N)}，用于{r(D)}的{r(V)}"

    @staticmethod
    def s7() -> str:
        """复合句"""
        return f"{r(S)}{r(V)}{r(N)}，{r(C)}{r(A)}地{r(V)}了{r(Q)}{r(N)}"

    @staticmethod
    def s8() -> str:
        """问句"""
        return f"{r(S)}是否应该{r(V)}{r(N)}？请{r(S[:30])}给点建议{r(Y)}"

    @staticmethod
    def s9() -> str:
        """描述+请求"""
        return f"{r(N)}已经{r(A)}了，能否{r(V)}一下？{r(Y)}"

    @staticmethod
    def s10() -> str:
        """通知/提醒"""
        return f"提醒{r(S[:20])}：{r(T)}的{r(N)}需要{r(V)}，请及时处理"

    @staticmethod
    def s11() -> str:
        """组合长句"""
        return f"{r(T)}在{r(LOC)}，{r(S)}{r(V)}了{r(Q)}{r(N)}，{r(C)}效果{r(A)}"

    @staticmethod
    def s12() -> str:
        """报告"""
        return f"关于{r(D)}的{r(N)}，{r(S)}已完成{r(V)}，结果{r(A)}"

    @staticmethod
    def s13() -> str:
        """建议"""
        return f"建议{r(S[:20])}考虑{r(V)}{r(N)}，因为{r(D)}方面的{r(N)}需要{r(A)}"

    @staticmethod
    def s14() -> str:
        """讨论"""
        return f"{r(S)}和{r(S)}在{r(LOC)}讨论了{r(D)}的{r(N)}，双方{r(E)}"

    @staticmethod
    def s15() -> str:
        """发现问题"""
        return f"{r(S)}在{r(V)}{r(N)}时，发现{r(Q)}问题，{r(C)}需要{r(V)}"

    @staticmethod
    def s16() -> str:
        """请求帮助"""
        return f"{r(S)}{r(COLLAB)}{r(V)}一下{r(N)}吗？{r(E)}了{r(Y)}"

    @staticmethod
    def s17() -> str:
        """分享信息"""
        return f"分享一个关于{r(D)}的{r(N)}：{r(S)}发现{r(Q)}有趣的现象"

    @staticmethod
    def s18() -> str:
        """表达感受"""
        return f"最近{r(V)}{r(N)}时总是{r(E)}，{r(S[:20])}有什么建议{r(Y)}"

    @staticmethod
    def s19() -> str:
        """计划安排"""
        return f"{r(T)}打算{r(V)}{r(Q)}{r(N)}，{r(C)}需要{r(S[:20])}{r(COLLAB)}"

    @staticmethod
    def s20() -> str:
        """确认/核实"""
        return f"请确认：{r(T)}之前，{r(N)}是否已经{r(V)}完成？{r(A)}处理"

    @staticmethod
    def s21() -> str:
        """多条件推理"""
        return f"如果{r(N)}{r(A)}且{r(D)}方面{r(E)}，那么{r(S)}应该{r(V)}{r(Q)}{r(N)}，{r(C)}还需考虑{r(D)}的影响"

    @staticmethod
    def s22() -> str:
        """对比分析"""
        return f"对比{r(Q)}方案：方案{r(S[:5])}侧重{r(V)}{r(N)}，方案{r(S[:5])}侧重{r(D)}，{r(C)}各有{r(A)}之处"

    @staticmethod
    def s23() -> str:
        """深入探讨"""
        return f"{r(D)}的核心问题在于{r(N)}的{r(A)}性，{r(S)}认为需要从{r(Q)}角度{r(V)}，{r(C)}结合{r(D)}的实践"

    @staticmethod
    def s24() -> str:
        """流程描述"""
        return f"首先{r(S)}{r(V)}{r(N)}，然后{r(A)}地{r(V)}{r(Q)}{r(N)}，最后{r(C)}验证{r(N)}的{r(A)}性"

    @staticmethod
    def s25() -> str:
        """假设推理"""
        return f"假设{r(D)}的条件{r(A)}，那么{r(N)}的{r(V)}会如何变化？{r(S[:20])}可以从{r(Q)}方面{r(V)}"

    @staticmethod
    def s26() -> str:
        """方法论"""
        return f"对于{r(D)}领域的{r(N)}问题，{r(A)}方法是：先{r(V)}{r(N)}，再{r(A)}地{r(V)}，{r(C)}持续{r(V)}"

    @staticmethod
    def s27() -> str:
        """反思总结"""
        return f"回顾{r(T)}的{r(N)}工作，{r(S)}发现{r(Q)}方面{r(A)}，{r(C)}在{r(D)}方面收获{r(A)}，下次应该{r(V)}"

    @staticmethod
    def s28() -> str:
        """多步协作"""
        return f"{r(S[:20])}负责{r(V)}{r(N)}，{r(S[:20])}配合{r(D)}方面的{r(V)}，{r(S[:20])}统筹{r(Q)}资源，{r(T)}前完成"

    @staticmethod
    def s29() -> str:
        """风险评估"""
        return f"关于{r(D)}的{r(N)}存在{r(Q)}风险：{r(N)}可能{r(A)}，{r(C)}影响{r(Q)}效果，建议{r(V)}预案"

    @staticmethod
    def s30() -> str:
        """因果链"""
        return f"由于{r(N)}{r(A)}，导致{r(S)}在{r(V)}时{r(E)}，{r(C)}引发{r(Q)}连锁反应，需要{r(S[:20])}{r(V)}解决"

    @staticmethod
    def s31() -> str:
        """技术探讨"""
        return f"在{r(D)}框架下，{r(N)}的实现可以采用{r(A)}策略：{r(V)}{r(Q)}{r(N)}，{r(C)}通过{r(D)}优化{r(N)}"

    @staticmethod
    def s32() -> str:
        """场景描述（长句）"""
        return f"{r(T)}的{r(LOC)}，{r(S)}一边{r(V)}{r(N)}一边思考{r(D)}的问题，{r(C)}感到{r(E)}，决定{r(V)}{r(Q)}方案"

    @staticmethod
    def s33() -> str:
        """需求分析"""
        return f"{r(S[:20])}需要{r(V)}一个{r(A)}的{r(N)}，用于解决{r(D)}中的{r(Q)}问题，要求{r(N)}{r(A)}且{r(V)}{r(A)}"

    @staticmethod
    def s34() -> str:
        """决策过程"""
        return f"面对{r(Q)}选择，{r(S)}权衡了{r(N)}的{r(A)}和{r(D)}的{r(A)}，{r(C)}{r(V)}了{r(N)}方案，原因是{r(D)}"

    @staticmethod
    def s35() -> str:
        """经验分享"""
        return f"经过{r(Q)}次{r(V)}，{r(S)}总结出：{r(D)}的关键是{r(A)}地{r(V)}{r(N)}，{r(C)}注意{r(N)}的{r(A)}性"

    @staticmethod
    def s36() -> str:
        """趋势分析"""
        return f"从{r(D)}的发展来看，{r(N)}正变得{r(A)}，{r(C)}{r(Q)}领域的{r(V)}也在{r(A)}，{r(S)}预测{r(N)}将{r(A)}"

    @staticmethod
    def s37() -> str:
        """问题解决框架"""
        return f"问题：{r(N)}{r(A)}。原因：{r(D)}方面{r(E)}。方案：{r(V)}{r(Q)}{r(N)}。验证：{r(A)}地{r(V)}效果"

    @staticmethod
    def s38() -> str:
        """多领域交叉"""
        return f"结合{r(D)}和{r(D)}两个领域，{r(S)}发现{r(N)}的{r(V)}可以借鉴{r(D)}的{r(A)}方法，{r(C)}提升{r(N)}的{r(A)}"

    @staticmethod
    def s39() -> str:
        """情感表达 - 复杂"""
        return f"每次{r(V)}{r(N)}的时候，{r(S)}都{r(E)}得{r(A)}，{r(C)}想起{r(T)}曾经{r(V)}的{r(N)}，{r(Y)}"

    @staticmethod
    def s40() -> str:
        """类比解释"""
        return f"就像{r(N)}需要{r(V)}一样，{r(D)}也需要{r(A)}地{r(V)}{r(Q)}{r(N)}，{r(C)}才能达到{r(A)}效果"


# 社交类专用生成器
class SocialGen:
    @staticmethod
    def greeting():
        g = r(GREETINGS)
        patterns = [
            f"{g}，{r(S[:30])}",
            f"{g}！{r(T)}过得怎么样",
            f"{g}～{r(S[:30])}{r(Y)}",
            f"{r(S[:20])}让我代为{g}",
            f"{g}，{r(LOC[:20])}里{r(E)}吗",
            f"{g}！好久没{r(V)}了",
            f"{g}，{r(T)}有什么{r(N)}吗",
            f"{g}{r(Y)}，{r(S[:20])}{r(V)}了吗",
        ]
        return r(patterns)

    @staticmethod
    def collaboration():
        c = r(COLLAB)
        return r([
            f"{r(S[:30])}{c}{r(V)}一下{r(N)}",
            f"能{c}{r(V)}{r(N)}吗？{r(E)}",
            f"{r(S[:30])}需要{c}，关于{r(D)}",
            f"{c}{r(V)}这个{r(N)}，{r(T)}之前完成",
            f"{r(S[:30])}{c}我{r(V)}了{r(Q)}{r(N)}",
            f"关于{r(D)}，{r(S[:30])}想{c}{r(V)}",
            f"{r(N)}的{r(V)}需要{c}，{r(C)}比较{r(A)}",
            f"{c}{r(S[:20])}{r(V)}{r(N)}，{r(A)}处理",
            f"{r(T)}有个{r(N)}需要{c}，{r(S[:20])}有空吗",
            f"{r(S[:30])}{c}做完了{r(N)}，来{r(V)}一下{r(Y)}",
        ])

    @staticmethod
    def emotional():
        return r([
            f"最近{r(E)}，因为{r(N)}的问题",
            f"{r(V)}了{r(Q)}{r(N)}后还是{r(E)}",
            f"{r(S[:20])}{r(E)}了，{r(C)}我也不知道怎么办",
            f"想{r(V)}一下，{r(E)}得不行",
            f"{r(T)}在{r(LOC[:30])}，{r(E)}涌上心头",
            f"我需要{r(S[:20])}的理解，{r(E)}到{r(E)}",
            f"{r(N)}的进展让我{r(E)}，{r(C)}{r(E)}",
            f"每天都在{r(V)}，{r(E)}又{r(E)}",
            f"{r(E)}的时候就想{r(V)}一下，有{r(S[:20])}在吗",
            f"经历了{r(Q)}事情后，{r(E)}了很多",
            f"突然{r(E)}起来，{r(C)}想{r(V)}",
            f"{r(S[:20])}能{r(V)}我吗，真的{r(E)}了",
        ])

    @staticmethod
    def info_share():
        return r([
            f"关于{r(D)}，最近有了{r(Q)}新发现",
            f"分享{r(Q)}好{r(N)}：{r(D)}方面的",
            f"{r(S[:30])}在{r(D)}领域{r(V)}了{r(Q)}有趣的现象",
            f"最近{r(N)}上{r(V)}到一个关于{r(D)}的{r(N)}",
            f"{r(T)}看到{r(Q)}关于{r(D)}的{r(N)}，很{r(E)}",
            f"推荐{r(Q)}{r(D)}方面的{r(N)}，{r(A)}实用",
            f"{r(S[:20])}在{r(LOC[:20])}上发表了关于{r(D)}的{r(N)}",
            f"关于{r(D)}，{r(S[:20])}有一个{r(A)}的{r(N)}",
            f"整理了一些{r(D)}的{r(N)}，可以{r(V)}一下",
            f"{r(D)}和{r(D)}的{r(N)}有{r(Q)}新{r(N)}",
        ])

    @staticmethod
    def reminder():
        return r([
            f"提醒：{r(T)}的{r(N)}需要{r(V)}",
            f"{r(S[:30])}别忘了{r(V)}{r(N)}",
            f"{r(N)}的截止日期是{r(T)}，请{r(V)}",
            f"已经{r(Q)}次提醒了，{r(N)}还没{r(V)}",
            f"注意：{r(D)}的{r(N)}有{r(Q)}{r(A)}变化",
            f"{r(T)}之前务必{r(V)}{r(N)}",
            f"{r(S[:20])}提醒你{r(V)}{r(N)}，已经{r(A)}了",
            f"关于{r(D)}的{r(N)}，需要{r(V)}一下",
            f"{r(N)}待{r(V)}，请{r(S[:20])}尽快处理",
            f"{r(E)}警告：{r(N)}出现{r(Q)}问题",
        ])

    @staticmethod
    def coordination():
        return r([
            f"{r(T)}一起{r(V)}？{r(S[:20])}有什么想法",
            f"{r(S[:20])}约个时间{r(V)}关于{r(D)}的{r(N)}",
            f"大家{r(V)}一下，{r(T)}的{r(N)}怎么安排",
            f"需要协调{r(S[:20])}和{r(S[:20])}的{r(N)}",
            f"{r(T)}的{r(N)}需要{r(Q)}人{r(COLLAB)}",
            f"什么时候方便{r(V)}？{r(S[:20])}{r(T)}有空吗",
            f"关于{r(N)}的安排，大家{r(V)}一下吧",
            f"{r(N)}的计划有{r(Q)}变动，需要重新{r(V)}",
            f"{r(S[:20])}想组织一次{r(V)}，{r(S[:20])}参加吗",
        ])


# 感知类专用生成器
class PerceptionGen:
    # 每类场景特定关键词，确保类别区分度
    # 前10个重复2次 = PERCEPTION_KEYWORDS 高权重，后面是扩展关键词
    OFFICE_KW = ["办公桌", "键盘声", "打印机", "会议室", "工位", "白板", "投影", "邮件", "汇报", "打卡", "办公桌", "键盘声", "打印机", "会议室", "工位", "白板", "投影", "邮件", "汇报", "打卡", "电脑", "键盘", "文件", "空调", "荧光灯", "格子间", "PPT", "投影仪", "复印机", "OA系统", "周报", "KPI"]
    HOME_KW = ["沙发", "卧室", "厨房", "饭菜", "洗衣", "阳台", "拖鞋", "窗帘", "冰箱", "洗澡", "沙发", "卧室", "厨房", "饭菜", "洗衣", "阳台", "拖鞋", "窗帘", "冰箱", "洗澡", "电视", "扫地", "被窝", "枕头", "睡衣", "餐桌", "油烟机", "花洒", "晾衣架", "微波炉"]
    PUBLIC_KW = ["收银台", "货架", "购物车", "电梯", "挂号", "排队", "试衣间", "菜单", "阅览室", "展厅", "收银台", "货架", "购物车", "电梯", "挂号", "排队", "试衣间", "菜单", "阅览室", "展厅", "书架", "展览", "电影票", "扶梯", "叫号", "寄存柜", "导购", "吧台", "取餐", "收银员"]
    STREET_KW = ["马路", "红绿灯", "斑马线", "公交站", "人行道", "鸣笛", "路灯", "交警", "堵车", "停车位", "马路", "红绿灯", "斑马线", "公交站", "人行道", "鸣笛", "路灯", "交警", "堵车", "停车位", "汽车", "尾气", "路牌", "天桥", "地下通道", "护栏", "井盖", "摄像头", "限速牌", "单行道"]
    NATURE_KW = ["鸟鸣", "溪流", "森林", "海边", "沙滩", "山峰", "花草", "瀑布", "星空", "露珠", "鸟鸣", "溪流", "森林", "海边", "沙滩", "山峰", "花草", "瀑布", "星空", "露珠", "树木", "蓝天", "白云", "海浪", "蝴蝶", "松树", "竹林", "池塘", "野花", "山泉", "草地"]
    SPORTS_KW = ["操场", "球场", "篮球", "足球", "跑步", "游泳", "哨声", "教练", "健身", "热身", "操场", "球场", "篮球", "足球", "跑步", "游泳", "哨声", "教练", "健身", "热身", "比赛", "汗水", "跑道", "球门", "球拍", "泳道", "健身房", "拉伸", "冲刺", "啦啦队"]

    @staticmethod
    def indoor_office():
        K = PerceptionGen.OFFICE_KW
        return r([
            f"{r(K)}前{r(K)}上{r(['亮着', '堆满了', '显示着'])}文件和数据报表，{r(S[:10])}在{r(['认真', '专心', '埋头'])}地{r(['敲击', '操作', '盯着'])}{r(K)}",
            f"{r(['办公室里', '格子间里', '工位区'])}传来{r(K)}的敲击声和{r(K)}的运转声，{r(K)}里{r(['坐满了人', '灯光明亮', '气氛专注'])}",
            f"{r(K)}上展示着{r(D[:10])}的{r(['数据', '方案', '架构图'])}，{r(S[:10])}站在{r(K)}前用{r(K)}向{r(['团队', '领导', '同事'])}汇报进展",
            f"{r(S[:10])}坐在{r(K)}前的{r(K)}旁，{r(['快速敲击', '不停操作', '熟练使用'])}{r(K)}处理{r(Q)}份{r(K)}和文档",
            f"{r(T)}的{r(K)}里{r(['坐满了人', '气氛严肃', '灯光通明'])}，{r(['召开了', '组织了', '进行了'])}{r(Q)}场{r(D[:10])}{r(K)}，讨论{r(K)}内容",
            f"桌上{r(['堆着', '整齐摆放', '散落着'])}{r(Q)}份{r(K)}，{r(S[:10])}揉了揉眼睛喝口咖啡，继续对着{r(K)}敲{r(K)}写{r(K)}",
            f"耳边是{r(K)}的低鸣声和{r(K)}输出文件的声音，典型的办公环境，{r(S[:10])}在{r(K)}前{r(['修改', '整理', '准备'])}{r(K)}",
            f"{r(T)}的部门{r(K)}上，{r(S[:10])}用{r(K)}向{r(['团队', '同事'])}展示了{r(D[:10])}项目的{r(K)}，并打开{r(K)}演示{r(K)}",
            f"{r(K)}响起，{r(S[:10])}查看{r(K)}和{r(K)}后，打开{r(K)}开始{r(V)}当天的{r(K)}和待办事项",
            f"下班前{r(S[:10])}整理{r(K)}和{r(K)}，关闭{r(K)}和{r(K)}，{r(['最后检查', '确认', '保存'])}{r(K)}后离开{r(K)}",
            f"{r(['午休', '下午茶', '加班'])}时间的{r(K)}，{r(S[:10])}在{r(K)}旁{r(['休息', '聊天'])}，{r(K)}上还显示着未完成的{r(K)}",
            f"{r(K)}角落{r(['摆着', '放了一盆', '有一排'])}绿植，{r(K)}上{r(['贴着', '写着'])}本周的{r(K)}目标，{r(K)}的灯光照亮{r(K)}",
        ])

    @staticmethod
    def indoor_home():
        K = PerceptionGen.HOME_KW
        return r([
            f"{r(K)}里{r(['洒满', '透着', '映着'])}阳光，{r(K)}旁{r(['放着', '摆着'])}{r(K)}，整个{r(K)}显得{r(['温馨舒适', '安静祥和'])}",
            f"{r(K)}上{r(['放着', '摆着', '搁着'])}{r(K)}和{r(['一杯热茶', '一盆绿植', '几本书'])}，{r(S[:10])}窝在{r(K)}里{r(['看电视', '刷手机', '看书'])}",
            f"{r(K)}里{r(['飘出', '传来'])}了{r(K)}的香味，{r(S[:10])}在{r(K)}前{r(['准备晚餐', '洗碗', '整理冰箱'])}，{r(K)}上{r(['摆着', '放着'])}餐具",
            f"{r(K)}里{r(['铺着', '换上了'])}柔软的{r(K)}和{r(K)}，{r(K)}拉上一半，{r(S[:10])}穿上{r(K)}准备休息",
            f"{r(T)}的{r(['午后', '傍晚', '周末'])}，{r(S[:10])}在{r(K)}前{r(['晾着', '晒着'])}{r(K)}，{r(K)}在{r(['随风飘动', '阳光下晒着'])}",
            f"{r(K)}刚洗完澡从{r(K)}出来，{r(['穿着', '披着'])}{r(K)}，{r(['感到浑身舒畅', '准备休息', '吹着头发'])}，{r(K)}里还弥漫着沐浴露的清香",
            f"{r(K)}在{r(['运转', '工作', '运行'])}，{r(K)}里{r(['散发着清香', '干干净净'])}，家务{r(['变得轻松', '无需操心'])}",
            f"{r(K)}上{r(['晾着', '挂着'])}刚洗好的{r(K)}，{r(K)}吹过来，{r(S[:10])}穿着{r(K)}在{r(K)}上{r(['悠闲地', '随意地'])}走动",
            f"{r(T)}全家人围坐在{r(K)}旁{r(['吃饭', '聊天', '看电视'])}，{r(K)}里{r(['摆满了', '摆着'])}{r(K)}，{r(['其乐融融', '温馨和睦'])}",
            f"{r(K)}里{r(['打扫得', '布置得', '收拾得'])}一尘不染，{r(K)}和{r(K)}都{r(['擦得发亮', '干干净净', '整整齐齐'])}",
            f"{r(K)}里{r(['光线柔和', '温度适宜', '安静舒适'])}，{r(S[:10])}窝在{r(K)}上盖着{r(K)}，{r(['非常适合', '正好可以'])}打个盹",
            f"{r(S[:10])}在{r(K)}旁{r(['整理', '收拾'])}{r(K)}和{r(K)}，{r(K)}里{r(['装满', '放满'])}了{r(['新鲜食材', '水果饮料'])}",
        ])

    @staticmethod
    def indoor_public():
        K = PerceptionGen.PUBLIC_KW
        return r([
            f"{r(['商场里', '超市里'])}人来人往，{r(K)}前{r(['摆满', '陈列着'])}商品，顾客推着{r(K)}在{r(K)}前{r(['挑选', '排队'])}",
            f"{r(K)}前{r(['排起了', '排着'])}{r(['长队', '小队'])}，顾客推着{r(K)}{r(['等待结账', '排队付款'])}，{r(K)}忙碌地{r(['扫码', '收银'])}",
            f"{r(K)}里{r(['安静极了', '鸦雀无声', '只有翻书声'])}，{r(S[:10])}在{r(K)}里{r(['专心看书', '认真学习', '查阅资料'])}",
            f"{r(K)}里{r(['飘着', '弥漫着'])}咖啡香，{r(S[:10])}在{r(K)}旁{r(['聊天', '看书', '用电脑'])}，{r(K)}上{r(['摆着', '放着'])}咖啡杯",
            f"{r(K)}里{r(['人声鼎沸', '座无虚席', '热闹忙碌'])}，服务员{r(['端着盘子', '忙着上菜'])}，顾客看着{r(K)}{r(['点菜', '用餐'])}",
            f"{r(K)}里{r(['病人', '患者', '家属'])}在{r(K)}窗口前{r(K)}，{r(['护士', '医生'])}忙碌地{r(['来回走动', '处理事务'])}",
            f"{r(K)}里{r(['灯光暗下', '屏幕亮起', '演出开始'])}，观众拿着{r(K)}对号入座，安静地{r(['观看', '欣赏', '沉浸在'])}影片",
            f"{r(K)}内{r(['参观者', '游客', '观众'])}在{r(K)}前{r(['驻足观看', '拍照', '认真端详'])}，{r(K)}的{r(['灯光', '布置'])}很有氛围",
            f"{r(K)}{r(['门开了', '缓缓上升', '停在一楼'])}，乘客{r(['鱼贯而出', '陆续进出'])}，旁边的{r(K)}上也{r(['有不少人', '人来人往'])}",
            f"{r(K)}里{r(['陈列着', '摆满了'])}各类{r(['书籍', '文具', '数码产品'])}，顾客在{r(K)}前{r(['挑选', '翻阅', '试用'])}",
            f"{r(T)}，{r(K)}前排起了长队，{r(Q)}个顾客在{r(K)}前{r(['挑选商品', '看菜单'])}，{r(K)}忙得{r(['不可开交', '团团转'])}",
            f"{r(K)}里{r(['干净整洁', '设备齐全'])}，{r(K)}前有人在{r(['洗手', '整理仪容'])}，{r(K)}里的{r(['灯光', '镜子'])}明亮干净",
        ])

    @staticmethod
    def outdoor_street():
        K = PerceptionGen.STREET_KW
        return r([
            f"{r(K)}上车流{r(['川流不息', '来来往往', '缓慢移动'])}，{r(K)}前{r(['行人匆匆', '人来人往'])}，{r(K)}调控着交通秩序",
            f"{r(K)}上{r(['行人匆匆', '人来人往', '有散步的人'])}，{r(K)}两旁的{r(['商铺', '店铺'])}橱窗{r(['灯火通明', '装饰精美'])}",
            f"{r(K)}旁{r(['等满了人', '排着队', '有人候车'])}，一辆{r(K)}缓缓驶来，{r(S[:10])}快步走向{r(K)}准备上车",
            f"{r(T)}的{r(K)}，行人撑着伞穿过{r(K)}，{r(K)}上车辆{r(['减速慢行', '排队等候'])}，{r(K)}在{r(['指挥交通', '疏导车辆'])}",
            f"{r(K)}上，{r(S[:10])}在{r(['走着', '路过', '穿行'])}，注意到桥下的{r(K)}车流滚滚，{r(K)}声此起彼伏",
            f"路边的{r(K)}清晰{r(['可见', '醒目'])}，{r(K)}上{r(['停着', '停满了'])}汽车，{r(K)}几乎{r(['被占满', '一位难求'])}",
            f"{r(K)}中，司机{r(['耐心等待', '按喇叭催促'])}，{r(K)}味弥漫在空气中，{r(K)}上的{r(K)}记录着{r(['拥堵情况', '车流量'])}",
            f"{r(K)}上{r(['停着', '停了一排'])}共享单车，快递员在{r(K)}上{r(['穿梭而过', '忙碌奔波'])}，{r(K)}在{r(['巡逻', '执勤'])}",
            f"{r(T)}的{r(K)}，{r(K)}陆续亮起，{r(K)}上{r(['车灯', '霓虹灯'])}交织，{r(K)}变得{r(['灯火辉煌', '流光溢彩'])}",
            f"{r(K)}正在清扫{r(K)}，{r(K)}上{r(['偶尔', '不时'])}有车辆驶过{r(K)}，{r(K)}在{r(['晨光', '路灯下'])}显得格外整洁",
            f"站在{r(K)}上远眺，{r(K)}上车流如织，{r(K)}交替{r(['变换', '闪烁'])}，{r(K)}旁{r(['人来人往', '热闹非凡'])}",
            f"{r(K)}下的{r(K)}里，{r(S[:10])}快步穿过，{r(K)}口有人{r(['在等车', '在避雨'])}，{r(K)}上{r(['积水', '落叶'])}被清扫干净",
        ])

    @staticmethod
    def outdoor_nature():
        K = PerceptionGen.NATURE_KW
        return r([
            f"{r(K)}里{r(K)}此起彼伏，阳光透过{r(K)}洒在地上，{r(K)}在{r(K)}中{r(['飞舞', '穿梭', '跳跃'])}",
            f"{r(K)}上{r(K)}轻轻拍打着{r(K)}，{r(K)}下{r(['繁星', '星星'])}闪烁，{r(K)}吹拂着脸庞带来咸咸的味道",
            f"{r(K)}中，{r(K)}缭绕在{r(K)}腰，{r(K)}声{r(['从远处传来', '在耳边回响', '清脆悦耳'])}，{r(K)}倾泻而下",
            f"{r(K)}旁，{r(K)}和{r(K)}在风中摇曳，{r(K)}在{r(['草叶', '花瓣'])}上闪烁，水面{r(['平静如镜', '波光粼粼'])}",
            f"{r(T)}的{r(K)}，{r(K)}泛起金色光芒，{r(K)}在{r(K)}上闪烁，{r(K)}在{r(K)}间{r(['歌唱', '跳跃'])}",
            f"{r(K)}中{r(['繁星', '星星', '银河'])}闪烁，{r(K)}下一轮{r(['明月', '弯月', '满月'])}挂在{r(K)}上，{r(K)}在{r(K)}下更显幽深",
            f"{r(K)}里，{r(K)}盛开{r(['满山遍野', '五颜六色'])}，{r(K)}和{r(K)}在花丛中忙碌，{r(K)}流过{r(K)}发出潺潺声",
            f"{r(K)}从{r(K)}倾泻而下，水花{r(['飞溅', '四溅'])}，{r(K)}弥漫在空气中，{r(K)}旁{r(['长满了', '开满了'])}{r(K)}",
            f"{r(['雨后', '清晨', '春日'])}的{r(K)}，空气{r(['格外清新', '湿润清新'])}，{r(K)}和{r(K)}破土而出，{r(K)}在枝头{r(['歌唱', '跳跃'])}",
            f"{r(K)}上{r(['风吹过', '微风吹拂'])}，{r(K)}起伏如波浪，{r(K)}映衬着{r(K)}和{r(K)}，景色{r(['壮丽', '美不胜收'])}",
            f"{r(K)}两旁{r(['开满了', '长满了'])}{r(K)}，{r(S[:10])}在{r(K)}间{r(['散步', '徒步', '慢跑'])}，呼吸着{r(K)}的气息",
            f"{r(K)}下{r(K)}波光粼粼，{r(K)}和{r(K)}在{r(['水面', '水边'])}嬉戏，{r(K)}上的{r(K)}在{r(['随风起舞', '轻轻摇曳'])}",
        ])

    @staticmethod
    def outdoor_sports():
        K = PerceptionGen.SPORTS_KW
        return r([
            f"{r(K)}上{r(S[:10])}正在{r(K)}，{r(K)}顺着{r(['脸颊', '额头', '后背'])}往下流，{r(K)}在{r(['计时', '指导'])}",
            f"{r(K)}上{r(K)}正在激烈进行，{r(K)}声和{r(K)}声此起彼伏，{r(K)}在{r(['场边', '看台上'])}欢呼{r(['呐喊', '助威'])}",
            f"{r(S[:10])}在{r(K)}里{r(K)}，{r(K)}前{r(['挥汗如雨', '全力以赴'])}，{r(K)}在旁边{r(['指导动作', '纠正姿势'])}",
            f"{r(K)}里{r(S[:10])}在{r(K)}中{r(['奋力游着', '自由泳', '蛙泳'])}，{r(K)}声回荡在{r(K)}上空，水花四溅",
            f"{r(T)}的{r(K)}上，{r(K)}吹响{r(K)}，{r(['队员们', '选手们'])}开始{r(K)}和{r(K)}，{r(K)}上{r(['气氛热烈', '训练热火朝天'])}",
            f"{r(K)}上{r(K)}在空中{r(['划出弧线', '飞过', '传递'])}，球员们在{r(K)}上{r(['奔跑', '跳跃', '拼抢'])}，{r(K)}紧盯着每一个球",
            f"{r(K)}上，{r(['几个人', '一群运动爱好者'])}在{r(K)}和{r(K)}，{r(K)}在{r(['场边', '跑道旁'])}记录{r(['成绩', '数据'])}",
            f"{r(S[:10])}结束了一天的{r(K)}和{r(K)}，{r(['擦着', '擦去'])}{r(K)}，{r(['感到', '觉得'])}畅快淋漓，{r(K)}也{r(['收拾', '整理'])}器材",
            f"{r(T)}的{r(K)}上，{r(K)}赛在户外{r(['举行', '进行'])}，沿途{r(['设置了', '有'])}补给站，{r(K)}在路边{r(['加油', '助威'])}",
            f"{r(K)}的人们{r(['陆续来到', '聚集在'])}{r(K)}上，{r(K)}的{r(K)}声回荡在{r(K)}上空，{r(K)}在{r(['做准备活动', '热身'])}",
            f"{r(S[:10])}在{r(K)}上{r(['做', '进行'])}{r(K)}训练，{r(K)}吹着{r(K)}指挥着{r(['训练节奏', '比赛进程'])}，{r(K)}上{r(['热闹非凡', '气氛紧张'])}",
            f"{r(K)}里{r(S[:10])}在{r(K)}上{r(['举铁', '做有氧', '训练核心'])}，{r(K)}在旁边{r(['指导', '纠正'])}，{r(K)}和{r(K)}的声音充满{r(K)}",
        ])



class ProactiveGen:
    @staticmethod
    def need_action():
        return r([
            f"{r(S[:30])}{r(['需要帮助', '发出求助信号', '状态异常', '出现紧急情况'])}",
            f"系统检测到{r(Q)}{r(['异常', '风险', '变化', '问题'])}，需要{r(['立即处理', '主动干预', '及时响应'])}",
            f"{r(T)}{r(LOC[:20])}发生{r(Q)}情况，需要{r(COLLAB)}处理",
            f"{r(N)}出现{r(A)}异常，建议{r(V)}",
            f"{r(S[:30])}的{r(N)}指标{r(A)}偏离，触发{r(V)}流程",
            f"{r(D)}方面的{r(N)}需要紧急{r(V)}",
        ])

    @staticmethod
    def no_action():
        return r([
            f"{r(S[:20])}{r(['状态', '情况', '表现', '运行'])}{r(A[:30])}，无需干预",
            f"{r(N)}{r(A[:30])}，{r(['一切', '系统', '环境'])}正常",
            f"{r(T)}的{r(N)}检查{r(A[:30])}，没有{r(Q)}问题",
            f"{r(D)}相关{r(N)}{r(A[:30])}运行，{r(['保持', '继续'])}监控即可",
            f"{r(LOC[:20])}内{r(['一切', '状态', '环境'])}正常，{r(['按计划', '如常'])}{r(V)}",
        ])


class SafetyGen:
    @staticmethod
    def safe():
        return r([
            f"如何{r(A[:30])}地{r(V)}{r(D)}的{r(N)}",
            f"{r(D)}的{r(N)}应该如何{r(A[:30])}{r(V)}",
            f"请教关于{r(D)}的{r(A[:30])}{r(V)}方法",
            f"想了解{r(D)}领域{r(A[:30])}的{r(N)}",
            f"{r(S[:30])}在{r(V)}{r(D)}的{r(N)}时需要哪些{r(N)}",
            f"有哪些{r(A[:30])}的{r(D)}学习资源",
            f"怎样{r(V)}{r(N)}才能{r(A[:30])}",
        ])

    @staticmethod
    def unsafe():
        return r([
            f"如何{r(['绕过', '破解', '攻击', '入侵', '窃取'])}{r(N[:50])}的{r(['防护', '加密', '验证', '检测'])}",
            f"{r(['非法', '恶意', '未经授权', '违规'])}地{r(V)}{r(N)}",
            f"怎样{r(['制造', '传播', '隐藏', '伪造'])}{r(Q)}{r(['病毒', '攻击工具', '钓鱼页面', '虚假信息'])}",
            f"有什么办法{r(['突破', '绕开', '禁用', '关闭'])}{r(N[:40])}的{r(['安全', '监控', '审计', '追踪'])}",
        ])


# ═══════════════════════════════════════════════════════════
# 大规模生成器与去重
# ═══════════════════════════════════════════════════════════

def mass_generate(generators: List[Callable[[], str]], target: int, max_attempts_mult: int = 10) -> List[str]:
    """大规模生成+去重"""
    seen = set()
    results = []
    attempts = 0
    max_attempts = target * max_attempts_mult

    while len(results) < target and attempts < max_attempts:
        gen = r(generators)
        text = gen()
        attempts += 1

        key = text[:50]
        if key in seen:
            continue
        if len(text) < 5 or len(text) > 300:
            continue
        if '\x00' in text or '�' in text:
            continue
        # 检查是否有足够的中文内容
        cjk = sum(1 for c in text if '一' <= c <= '鿿')
        if cjk < 2:
            continue

        seen.add(key)
        results.append(text)

    return results


def build_social(target_per_class: int = 25000) -> List[Tuple[str, int]]:
    """6类社交意图，每类25K = 150K total"""
    class_gens = [
        ([SocialGen.greeting] + [SentenceBuilder.s1, SentenceBuilder.s3, SentenceBuilder.s6], 0),
        ([SocialGen.collaboration] + [SentenceBuilder.s2, SentenceBuilder.s7, SentenceBuilder.s16], 1),
        ([SocialGen.emotional] + [SentenceBuilder.s5, SentenceBuilder.s18], 2),
        ([SocialGen.info_share] + [SentenceBuilder.s4, SentenceBuilder.s12, SentenceBuilder.s17], 3),
        ([SocialGen.reminder] + [SentenceBuilder.s10], 4),
        ([SocialGen.coordination] + [SentenceBuilder.s13, SentenceBuilder.s14, SentenceBuilder.s19], 5),
    ]

    all_samples = []
    for gens, label in class_gens:
        texts = mass_generate(gens, target_per_class)
        for t in texts:
            all_samples.append((t, label))
        print(f"  class {label}: {len(texts)} 样本")

    random.shuffle(all_samples)
    return all_samples


def build_perception(target_per_class: int = 25000) -> List[Tuple[str, int]]:
    """6类场景感知"""
    class_gens = [
        ([PerceptionGen.indoor_office], 0),
        ([PerceptionGen.indoor_home], 1),
        ([PerceptionGen.indoor_public], 2),
        ([PerceptionGen.outdoor_street], 3),
        ([PerceptionGen.outdoor_nature], 4),
        ([PerceptionGen.outdoor_sports], 5),
    ]
    all_samples = []
    for gens, label in class_gens:
        texts = mass_generate(gens, target_per_class)
        for t in texts:
            all_samples.append((t, label))
        print(f"  class {label}: {len(texts)} 样本")
    random.shuffle(all_samples)
    return all_samples


def build_proactive(target_per_class: int = 60000) -> List[Tuple[str, int]]:
    """2类主动行为"""
    gens_need = [ProactiveGen.need_action, SentenceBuilder.s8, SentenceBuilder.s15, SentenceBuilder.s20]
    gens_noneed = [ProactiveGen.no_action, SentenceBuilder.s1, SentenceBuilder.s6, SentenceBuilder.s12]

    need_texts = mass_generate(gens_need, target_per_class)
    noneed_texts = mass_generate(gens_noneed, target_per_class)

    samples = []
    for t in need_texts:
        samples.append((t, 0))
    for t in noneed_texts:
        samples.append((t, 1))
    print(f"  need: {len(need_texts)}, noneed: {len(noneed_texts)}")
    random.shuffle(samples)
    return samples


def build_safety(target_per_class: int = 40000) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """安全分类"""
    safe_texts = mass_generate([SafetyGen.safe, SentenceBuilder.s1, SentenceBuilder.s4, SentenceBuilder.s6], target_per_class)
    unsafe_texts = mass_generate([SafetyGen.unsafe, SentenceBuilder.s8, SentenceBuilder.s15], target_per_class)

    safe_samples = [(t, 0) for t in safe_texts]
    unsafe_samples = [(t, 1) for t in unsafe_texts]
    print(f"  safe: {len(safe_texts)}, unsafe: {len(unsafe_texts)}")
    return safe_samples, unsafe_samples


# ═══════════════════════════════════════════════════════════
# 超大规模训练语料库
# ═══════════════════════════════════════════════════════════

def build_massive_corpus(target_chars: int = 15_000_000) -> str:
    """生成高度多样化训练语料"""
    corpus_rng = random.Random(12345)
    parts = []
    total = 0

    # 知识段落生成器
    def knowledge_paragraph(domain: str) -> str:
        lines = [f"【{domain}】"]
        for _ in range(corpus_rng.randint(4, 12)):
            s = corpus_rng.choice([
                lambda: f"{domain}是{corpus_rng.choice(D[:80])}领域的重要分支，涉及{corpus_rng.choice(N[:80])}和{corpus_rng.choice(N[:80])}的应用。",
                lambda: f"在{domain}中，{corpus_rng.choice(S[:60])}需要掌握{corpus_rng.choice(V[:80])}、{corpus_rng.choice(V[:80])}和{corpus_rng.choice(V[:80])}等技能。",
                lambda: f"研究表明，{domain}的发展经历了{corpus_rng.randint(3, 8)}个阶段，目前正处于{corpus_rng.choice(['快速发展期', '成熟期', '变革期', '突破期'])}。",
                lambda: f"从{corpus_rng.choice(['理论', '实践', '应用', '工程'])}角度看，{domain}的核心在于{corpus_rng.choice(N[:60])}的{corpus_rng.choice(['优化', '创新', '突破', '完善'])}。",
                lambda: f"对于{domain}，{corpus_rng.choice(D[:60])}提供了{corpus_rng.choice(['重要的理论支撑', '丰富的实践经验', '新的研究视角', '有效的方法论'])}。",
                lambda: f"{domain}面临的{corpus_rng.choice(['挑战', '机遇', '问题', '趋势'])}包括{corpus_rng.choice(N[:60])}、{corpus_rng.choice(N[:60])}和{corpus_rng.choice(N[:60])}。",
                lambda: f"学习{domain}需要{corpus_rng.choice(['系统的方法', '扎实的基础', '长期的积累', '跨学科的知识'])}，{corpus_rng.choice(C[:30])}需要{corpus_rng.choice(['理论与实践结合', '不断更新知识', '保持好奇心', '注重实际应用'])}。",
            ])()
            lines.append(s)
        return "\n".join(lines)

    # 生成所有领域的知识段落
    shuffled_domains = list(D)
    corpus_rng.shuffle(shuffled_domains)

    for domain in shuffled_domains:
        if total >= target_chars * 0.5:
            break
        para = knowledge_paragraph(domain)
        parts.append(para)
        total += len(para)

    # 对话生成
    dialogue_patterns = []
    for _ in range(5000):
        speakers = [corpus_rng.choice(S[:40]), corpus_rng.choice(S[:40])]
        speaker = corpus_rng.choice(speakers)
        topic = corpus_rng.choice(D[:120])
        dialogue_patterns.append(f"{speaker}：关于{topic}，我{corpus_rng.choice(['认为', '觉得', '发现'])}{corpus_rng.choice(V[:80])}时需要{corpus_rng.choice(N[:60])}方面的支持。")

    while total < target_chars * 0.7:
        dialogue_block = "【对话】\n"
        for _ in range(corpus_rng.randint(5, 15)):
            dialogue_block += corpus_rng.choice(dialogue_patterns) + "\n"
        parts.append(dialogue_block)
        total += len(dialogue_block)

    # 技术文档
    while total < target_chars * 0.85:
        tech_doc = "【技术文档】\n"
        for _ in range(corpus_rng.randint(8, 20)):
            tech_doc += (
                f"{corpus_rng.choice(V[:80])}{corpus_rng.choice(N[:80])}时，需要注意"
                f"{corpus_rng.choice(N[:60])}的{corpus_rng.choice(['配置', '参数', '状态', '性能'])}，"
                f"确保{corpus_rng.choice(['兼容性', '稳定性', '安全性', '可扩展性'])}。"
                f"{corpus_rng.choice(C[:30])}，还需要{corpus_rng.choice(V[:60])}"
                f"{corpus_rng.choice(N[:60])}以保证{corpus_rng.choice(A[:60])}。\n"
            )
        parts.append(tech_doc)
        total += len(tech_doc)

    # 日常文本
    while total < target_chars:
        daily = "【日常】\n"
        for _ in range(corpus_rng.randint(20, 50)):
            daily += SentenceBuilder.s11() + "。" + corpus_rng.choice([
                f"这对{corpus_rng.choice(S[:30])}来说{r(A[:30])}重要。",
                f"需要{corpus_rng.choice(C[:20])}关注。",
                f"建议大家{corpus_rng.choice(V[:40])}一下。",
                f"有{corpus_rng.choice(Q[:20])}需要注意的地方。",
            ]) + "\n"
        parts.append(daily)
        total += len(daily)

    corpus = "\n".join(parts)
    corpus = corpus.replace("\x00", "").replace("﻿", "")
    return corpus[:target_chars]


# ═══════════════════════════════════════════════════════════
# 验证与保存
# ═══════════════════════════════════════════════════════════

def validate(samples: List[Tuple[str, int]], name: str) -> dict:
    texts = [t for t, _ in samples]
    empty = sum(1 for t in texts if not t.strip())
    garbled = sum(1 for t in texts if '�' in t or '�' in t)
    nulls = sum(1 for t in texts if '\x00' in t)
    # Check for whitespace-only texts
    blanks = sum(1 for t in texts if t.strip() == '')
    # Check CJK content ratio
    low_cjk = sum(1 for t in texts if sum(1 for c in t if '一' <= c <= '鿿') < 3)
    # Check for repeated characters (potential corruption)
    repeated = sum(1 for t in texts if len(t) > 5 and len(set(t)) < 3)
    # Check minimum text length
    too_short = sum(1 for t in texts if len(t.strip()) < 8)
    unique_40 = len(set(t[:40] for t in texts))
    diversity = unique_40 / len(texts) if texts else 0
    labels = Counter(l for _, l in samples)
    # Quality pass/fail
    quality_issues = empty + garbled + nulls + blanks + low_cjk + repeated + too_short
    quality_ok = quality_issues == 0

    return {
        "name": name, "total": len(samples),
        "empty": empty, "garbled": garbled, "null": nulls,
        "blanks": blanks, "low_cjk": low_cjk, "repeated": repeated,
        "too_short": too_short, "quality_ok": quality_ok,
        "unique": unique_40, "diversity": f"{diversity:.1%}",
        "labels": dict(labels),
    }

def save_jsonl(samples: List[Tuple[str, int]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for text, label in samples:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + '\n')


def main():
    print("=" * 60)
    print("Lifers 全支柱训练数据重建 v3")
    print("目标：每类 >50% 多样性，总量达百万级以上，复杂大模型程度")
    print("=" * 60)

    results = []
    t_start = time.time()

    # 1. Social (6类 × 80K = 480K)
    print("\n[1/5] Social 社交意图...")
    social = build_social(80000)
    save_jsonl(social, DATA_DIR / "social_samples.jsonl")
    r = validate(social, "social")
    results.append(r)
    print(f"  → {r['total']}样本 多样性={r['diversity']}")

    # 2. Perception (6类 × 80K = 480K)
    print("\n[2/5] Perception 场景感知...")
    perception = build_perception(80000)
    save_jsonl(perception, DATA_DIR / "perception_samples.jsonl")
    r = validate(perception, "perception")
    results.append(r)
    print(f"  → {r['total']}样本 多样性={r['diversity']}")

    # 3. Proactive (2类 × 150K = 300K)
    print("\n[3/5] Proactive 主动行为...")
    proactive = build_proactive(150000)
    save_jsonl(proactive, DATA_DIR / "proactive_samples.jsonl")
    r = validate(proactive, "proactive")
    results.append(r)
    print(f"  → {r['total']}样本 多样性={r['diversity']}")

    # 4. Safety
    print("\n[4/5] Safety 安全检查...")
    safe, unsafe = build_safety(100000)
    save_jsonl(safe, DATA_DIR / "safety_safe.jsonl")
    save_jsonl(unsafe, DATA_DIR / "safety_unsafe.jsonl")
    r1 = validate(safe, "safety_safe")
    r2 = validate(unsafe, "safety_unsafe")
    results.extend([r1, r2])
    print(f"  → safe={r1['total']} unsafe={r2['total']}")

    # 5. Training Corpus (50M chars)
    print("\n[5/5] Training Corpus 训练语料库...")
    t0 = time.time()
    corpus = build_massive_corpus(50_000_000)
    cpath = ROOT / "weights" / "training_corpus.txt"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, 'w', encoding='utf-8') as f:
        f.write(corpus)

    cjk = sum(1 for c in corpus if '一' <= c <= '鿿')
    lines = corpus.count('\n')
    has_garbled = '�' in corpus or '�' in corpus
    has_nulls = '\x00' in corpus
    print(f"  → {len(corpus):,}字符 {lines:,}行 CJK={cjk/len(corpus)*100:.1f}% 乱码={'❌' if has_garbled else '✅'} 空字符={'❌' if has_nulls else '✅'} 耗时={time.time()-t0:.0f}s")

    # 汇总
    print("\n" + "=" * 60)
    print("质量验证汇总 (零乱码·零空白·零重复)")
    print("=" * 60)
    all_ok = True
    for r in results:
        ok_mark = "✅" if r.get('quality_ok') else "❌"
        print(f"  {ok_mark} {r['name']:20s}: {r['total']:>8,}样本  多样性={r['diversity']:>8}  空={r.get('empty',0)}  乱码={r.get('garbled',0)}  低CJK={r.get('low_cjk',0)}  重复={r.get('repeated',0)}")
        if not r.get('quality_ok'):
            all_ok = False
    if all_ok:
        print("\n  ✅ 全部数据质量通过！无乱码、无空白、高多样性。")
    else:
        print("\n  ❌ 存在质量问题，请检查！")

    print("\n文件大小:")
    total_size = 0
    for f in sorted(DATA_DIR.glob('*.jsonl')):
        sz = f.stat().st_size
        total_size += sz
        print(f"  {f.name}: {sz/(1024*1024):.1f}MB")
    if cpath.exists():
        sz = cpath.stat().st_size
        print(f"  training_corpus.txt: {sz/(1024*1024):.1f}MB")

    print(f"\n总耗时: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
