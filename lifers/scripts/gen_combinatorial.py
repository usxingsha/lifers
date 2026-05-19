#!/usr/bin/env python3
"""组合爆炸式语料生成 — 领域×主题×格式×风格 交叉生成海量训练数据"""
import sys, os, random, itertools

sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
CORPUS_PATH = Path(__file__).resolve().parent.parent / "weights" / "training_corpus.txt"

# ═══════════════════════════════════════════
# 组合生成引擎
# ═══════════════════════════════════════════

random.seed(42)

# 领域 × 子主题 矩阵
DOMAINS = {
    "AI安全": [
        "对抗样本防御", "模型鲁棒性", "差分隐私", "联邦学习安全",
        "供应链安全", "提示注入防御", "模型水印", "安全审计",
        "威胁情报", "漏洞管理", "访问控制", "加密通信",
        "身份认证", "安全日志", "入侵检测", "代码安全",
    ],
    "计算机视觉": [
        "图像分类", "目标检测", "语义分割", "实例分割",
        "姿态估计", "人脸识别", "OCR文字识别", "图像生成",
        "视频理解", "3D重建", "深度估计", "图像增强",
        "风格迁移", "超分辨率", "去噪去模糊", "场景理解",
    ],
    "自然语言处理": [
        "文本分类", "命名实体识别", "关系抽取", "文本摘要",
        "机器翻译", "问答系统", "情感分析", "文本生成",
        "对话系统", "信息检索", "知识图谱", "词向量",
        "句法分析", "指代消解", "篇章分析", "语言模型",
    ],
    "强化学习": [
        "Q学习", "策略梯度", "演员评论家", "DQN",
        "PPO算法", "SAC算法", "多智能体RL", "逆强化学习",
        "探索策略", "奖励设计", "元学习", "分层RL",
        "离线RL", "基于模型的RL", "模仿学习", "好奇心驱动",
    ],
    "预测与规划": [
        "时间序列预测", "需求预测", "异常检测", "因果推断",
        "动态规划", "蒙特卡洛", "贝叶斯优化", "A*搜索",
        "路径规划", "资源调度", "库存优化", "风险管理",
        "决策树", "集成学习", "梯度提升", "神经网络预测",
    ],
    "社交智能": [
        "对话策略", "情感识别", "共情回应", "个性化推荐",
        "社交网络分析", "社区发现", "影响力传播", "舆论分析",
        "信任建模", "协作决策", "谈判策略", "群体智慧",
        "用户画像", "意图理解", "多轮对话", "语境管理",
    ],
    "知识工程": [
        "本体构建", "实体链接", "关系抽取", "知识融合",
        "图神经网络", "链接预测", "推理引擎", "知识问答",
        "时序知识图谱", "多模态知识", "常识推理", "规则挖掘",
        "图嵌入", "路径排序", "归纳推理", "溯因推理",
    ],
    "分布式系统": [
        "共识算法", "数据分片", "负载均衡", "故障恢复",
        "CAP定理", "消息队列", "微服务", "容器编排",
        "服务发现", "配置中心", "链路追踪", "日志聚合",
        "缓存策略", "数据一致性", "流处理", "批处理",
    ],
    "软件工程": [
        "设计模式", "代码重构", "测试策略", "持续集成",
        "技术债务", "架构设计", "性能优化", "安全编码",
        "代码审查", "版本管理", "敏捷开发", "需求工程",
        "领域驱动设计", "事件驱动", "CQRS", "六边形架构",
    ],
    "语音处理": [
        "语音识别", "语音合成", "声纹识别", "情感语音",
        "噪声抑制", "回声消除", "声源定位", "语音增强",
        "说话人分离", "关键词检测", "语种识别", "音乐分析",
        "空间音频", "波束成形", "声学建模", "端到端ASR",
    ],
}

# 内容模板
TEMPLATES = [
    "## {domain}: {topic}\n\n{body}\n\n> 关键思考: {insight}\n",
    "### {topic} ({domain}领域)\n\n{body}\n\n*实践建议: {practice}*\n",
    "\n**{topic}**\n\n{body}\n\n> {insight}\n",
    "# {topic}\n\n{body}\n\n## 技术要点\n\n1. {point1}\n2. {point2}\n3. {point3}\n",
    "\n### 深入探讨: {topic}\n\n{body}\n\n```python\n{practice}\n```\n\n*{insight}*\n",
    "问: 请详细解释{topic}?\n\n答: {body}\n\n> {insight}\n\n*{practice}*\n",
    "\n## {domain}核心技术: {topic}\n\n{body}\n\n- 要点一: {point1}\n- 要点二: {point2}\n- 要点三: {point3}\n",
    "> **{topic}**\n>\n> {body}\n>\n> *——{insight}*\n",
    "\n### {topic}在实际中的应用\n\n{body}\n\n**最佳实践**\n\n{practice}\n",
    "\n## 技术深度分析\n\n### {domain}: {topic}\n\n{body}\n\n### 延伸思考\n\n{insight}\n\n### 实践指南\n\n{practice}\n",
]

# 生成体库
BODIES = {
    "short": [
        "{topic}是{domain}领域的核心研究方向之一。它涉及{aspect}等多个方面，在{application}等场景中有着广泛的应用前景。当前主流方法基于{method}，通过{approach}的方式实现。该方向的关键挑战在于{challenge}，研究者们正在探索{exploration}来应对这些挑战。",
    ],
    "medium": [
        "在{domain}领域中，{topic}的研究具有重要意义。该方向主要关注{aspect}，涉及从理论到实践的多个层面。\n\n从理论角度来看，{topic}建立在{foundation}的基础上。核心思想是通过{approach}来解决{challenge}的问题。研究表明，{method}是当前最有效的技术路线之一。\n\n在实际应用中，{topic}已经在{application}等场景中得到了验证。典型的实现流程包括：数据预处理、特征工程、模型训练和评估优化。每个环节都需要根据具体场景进行调整。\n\n未来发展方向包括：{exploration}，以及与{related}等领域的交叉融合。",
    ],
}

ASPECTS = [
    "理论基础与算法设计", "工程实现与性能优化", "安全性与鲁棒性",
    "可解释性与透明度", "数据效率与泛化能力", "计算效率与资源消耗",
    "公平性与伦理影响", "可扩展性与分布式部署",
]

APPLICATIONS = [
    "自动驾驶", "医疗诊断", "金融风控", "智能制造",
    "智慧城市", "社交媒体", "教育科技", "电子商务",
    "网络安全", "机器人", "智能家居", "游戏AI",
    "内容推荐", "语音助手", "搜索引擎", "物联网",
]

METHODS = [
    "深度学习", "强化学习", "图神经网络", "Transformer架构",
    "生成对抗网络", "变分自编码器", "注意力机制", "迁移学习",
    "联邦学习", "自监督学习", "元学习", "集成学习",
]

APPROACHES = [
    "端到端训练", "多任务联合学习", "知识蒸馏", "模型压缩",
    "对抗训练", "数据增强", "特征融合", "集成决策",
    "渐进式训练", "课程学习", "主动学习", "对比学习",
]

CHALLENGES = [
    "数据质量和标注成本", "模型的泛化能力", "计算资源的限制",
    "安全性和隐私保护", "可解释性的不足", "长尾分布的处理",
    "实时性的要求", "多模态信息的融合",
]

FOUNDATIONS = [
    "概率论与统计学", "线性代数与优化理论", "信息论与编码理论",
    "控制论与系统理论", "博弈论与决策理论", "图论与组合优化",
    "数理逻辑与形式化方法", "认知科学与神经科学",
]

EXPLORATIONS = [
    "更高效的训练算法", "更鲁棒的模型架构", "更智能的数据利用方式",
    "更透明的决策机制", "更公平的学习范式", "更高效的推理引擎",
    "跨模态的知识迁移", "人机协同的智能系统",
]

RELATED_FIELDS = [
    "自然语言处理", "计算机视觉", "强化学习", "知识图谱",
    "机器人学", "语音处理", "推荐系统", "运筹学",
    "控制理论", "认知科学", "博弈论", "信号处理",
]

INSIGHTS = [
    "理论和实践之间的鸿沟往往需要通过大量的实验和迭代来弥合",
    "简单的方法在实际中往往比复杂的方法更有效，关键在于正确的问题建模",
    "数据和模型同等重要，高质量的数据往往比更复杂的模型带来更大的提升",
    "系统的鲁棒性和泛化能力比在标准测试集上的性能指标更重要",
    "技术方案的选择应该由问题本身驱动，而非由最新的研究趋势决定",
    "好的工程实践和系统设计很多时候比算法的微小改进更有价值",
    "在大多数真实场景中，可解释性和可靠性比原始性能更重要",
    "跨领域的知识迁移常常能带来意想不到的创新和突破",
    "持续迭代和数据驱动的方法往往比一次性设计出完美方案更可行",
    "理解问题的本质比掌握最新技术更重要，技术是解决问题的工具而非目的",
]

PRACTICES = [
    "从简单的基线模型开始，逐步增加复杂度。每次增加前都应该验证新增部分确实带来了性能提升",
    "建立完善的评估流程和指标体系。没有好的评估标准，优化方向就无从谈起",
    "保持代码和实验记录的可复现性。使用版本控制管理代码、数据和配置",
    "在模型设计阶段就考虑部署和运维需求。模型上线只是开始，持续监控和更新才是常态",
    "重视数据质量而非仅仅追求数据数量。精心清洗的万条数据胜过粗制滥造的百万条数据",
    "建立系统化的调试方法论。面对问题时先检查数据，再检查代码，最后调整模型",
    "定期审视和更新技术路线。按月或按季度评估新技术新方法的适用性",
    "将安全性和伦理考量融入技术方案的每个环节而非事后补救",
]

POINTS = [
    "数据预处理是模型成功的基础，需要仔细处理缺失值、异常值和数据分布",
    "特征工程直接影响模型的学习能力，好的特征能让简单模型也能取得不错效果",
    "模型选择需要考虑精度、速度、内存和可解释性的综合权衡",
    "超参数调优不应该盲目，使用贝叶斯优化或HyperBand等结构化方法",
    "交叉验证是评估模型泛化能力的基本方法，注意数据泄露问题",
    "模型部署后需要持续监控性能指标和数据分布的变化",
    "文档和注释是代码可维护性的保障，写清楚为什么比写清楚是什么更重要",
]

def generate_body(topic, domain, style="medium"):
    """生成内容主体"""
    template = random.choice(BODIES[style])
    return template.format(
        topic=topic, domain=domain,
        aspect=random.choice(ASPECTS),
        application=random.choice(APPLICATIONS),
        method=random.choice(METHODS),
        approach=random.choice(APPROACHES),
        challenge=random.choice(CHALLENGES),
        foundation=random.choice(FOUNDATIONS),
        exploration=random.choice(EXPLORATIONS),
        related=random.choice(RELATED_FIELDS),
    )

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   组合爆炸式语料生成器                   ║")
    print("╚══════════════════════════════════════════╝")

    existing = ""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            existing = f.read()
    print(f"现有语料: {len(existing)/1024/1024:.1f}MB")

    parts = []
    total_chars = 0
    entry_count = 0

    # 对每个领域-主题组合生成内容
    for domain, topics in DOMAINS.items():
        parts.append(f"\n\n# {domain} 技术知识体系\n")
        for topic in topics:
            # 每个主题生成多种格式变体
            for _ in range(20):  # 20 variations per topic
                style = random.choice(["short", "medium"])
                body = generate_body(topic, domain, style)
                insight = random.choice(INSIGHTS)
                practice = random.choice(PRACTICES)

                tpl = random.choice(TEMPLATES)
                text = tpl.format(
                    domain=domain, topic=topic, body=body,
                    insight=insight, practice=practice,
                    point1=random.choice(POINTS),
                    point2=random.choice(POINTS),
                    point3=random.choice(POINTS),
                )
                parts.append(text)
                total_chars += len(text)
                entry_count += 1

        print(f"  {domain}: {len(topics)} 主题")

    print(f"\n生成 {entry_count} 个条目, {total_chars:,} 字符")

    # 组装
    header = "\n\n" + "="*60 + "\n"
    header += "# LIFERS 全领域组合语料库\n"
    header += "# 涵盖: " + ", ".join(DOMAINS.keys()) + "\n"
    header += "="*60 + "\n\n"

    combined = existing + header + '\n'.join(parts)
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"\n{'='*60}")
    print(f"语料总量: {len(combined):,} 字符 ({len(combined)/1024/1024:.1f}MB)")
    print(f"行数: {combined.count(chr(10)):,}")
    print(f"保存: {CORPUS_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
