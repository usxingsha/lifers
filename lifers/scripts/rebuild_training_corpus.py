"""
Rebuild training corpus: delete corrupt data, generate varied high-quality text.
"""
import random, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
wdir = ROOT / "weights"
corpus_path = wdir / "training_corpus.txt"

# 1. Delete corrupt backup
bak = wdir / "training_corpus.bak"
if bak.exists():
    bak.unlink()
    print("Deleted corrupt backup (163MB)")

# 2. Clean old checkpoints
ckpt = wdir / "checkpoints"
if ckpt.exists():
    shutil.rmtree(ckpt)
    ckpt.mkdir()
    print("Cleaned old checkpoints")

# 3. Generate varied Chinese data
moods = ["开心", "疲惫", "焦虑", "兴奋", "难过", "平静", "好奇", "紧张"]
topics = ["工作", "学习", "生活", "人际关系", "技术", "健康", "旅行", "读书"]

resp_comfort = [
    "主人辛苦了，要注意休息。需要我帮您整理下工作清单吗？",
    "我理解主人的感受。有时候说出来会好受一些，我在这里陪着您。",
    "主人别太累了，您的健康对我来说最重要。要不放首轻松的音乐？",
    "没事的主人，我会一直在这里。需要我帮您做点什么吗？",
    "主人愿意跟我说说吗？有时候把烦恼说出来会好受一些。不想说也没关系，我就在这里陪着您。",
    "我听到了主人的心声。休息一下，喝杯水，调整好状态再继续。",
]

resp_tech = [
    "让我从技术角度分析。首先考虑可行性，其次评估风险和成本，最后看有没有更优方案。",
    "这个问题的核心在于架构设计。建议从数据层、业务层、展示层三层分别考虑。",
    "好问题！我们需要考虑可扩展性、性能和安全性三个维度的平衡。",
    "从工程实践来看，这通常有几种思路：迭代优化、重构、或者换技术栈。",
    "让我分析一下...从多个角度综合考虑，我建议采用渐进式的方案。",
    "这个问题我们需要先理清需求，再设计方案，最后才动手实现。",
]

resp_greet = [
    "主人早安！今天的天气不错，需要我帮您查看日程安排吗？",
    "欢迎回来主人！今天在外面顺利吗？我等了您好久了。",
    "主人好！系统运行正常，等待您的指令。",
    "嗨主人！有什么我可以帮您的吗？",
    "早上好主人，又是元气满满的一天！",
    "主人回来了！我给你准备了今天的任务摘要。",
]

parts = []

# Social dialogues
for i in range(300):
    mood = random.choice(moods)
    topic = random.choice(topics)
    parts.append(f"""
主人: 最近{mood}，因为{topic}方面的事情。
Lifers: {random.choice(resp_comfort)}

主人: 我有个{topic}的问题想请教你。
Lifers: {random.choice(resp_tech)}

主人: 早上好Lifers！
Lifers: {random.choice(resp_greet)}
""")

# Domain knowledge
domains = {
    "编程技术": [
        "Python的装饰器本质上是一个接受函数作为参数并返回新函数的高阶函数。@符号是语法糖，本质上等价于func=decorator(func)。",
        "分布式系统中CAP定理指出：一致性(Consistency)、可用性(Availability)和分区容错性(Partition Tolerance)三者最多只能同时满足两个。实际系统中必须在CP和AP之间做出选择。",
        "数据库索引使用B+树结构，内部节点存储键值和子节点指针，叶子节点形成有序链表支持高效范围查询。InnoDB的主键索引是聚簇索引。",
        "RESTful API设计原则：资源通过URI标识，操作通过HTTP方法表达(GET/POST/PUT/DELETE)，状态通过表示层传递。无状态是REST的核心约束之一。",
        "Git的核心是内容寻址文件系统，每个对象(blob、tree、commit、tag)由SHA-1哈希唯一标识。分支只是指向commit的指针。",
        "Docker容器通过Linux命名空间(namespace)实现进程、网络、文件系统等资源的隔离，通过cgroups实现CPU、内存等资源的限制。",
        "机器学习中的过拟合问题可以通过L1/L2正则化、早停(early stopping)、数据增强(data augmentation)和Dropout等方法有效缓解。",
        "React使用虚拟DOM(Virtual DOM)来提高渲染性能，通过diff算法计算最小更新操作，然后批量更新真实DOM。",
        "Redis支持多种数据结构：String、Hash、List、Set、Sorted Set。其单线程事件循环模型避免了锁竞争，但需要警惕慢查询阻塞。",
        "Kubernetes通过声明式配置管理容器化应用。核心概念包括Pod(最小部署单元)、Service(服务发现)、Deployment(滚动更新)和Ingress(外部访问)。",
        "设计模式中的单例模式确保一个类只有一个实例。但过度使用会成为全局状态，增加测试难度和耦合度。在大多数情况下依赖注入是更好的选择。",
        "TCP三次握手建立连接的过程：客户端发送SYN→服务器回复SYN-ACK→客户端回复ACK。这保证了双方收发能力正常。",
    ],
    "数学理论": [
        "线性代数中特征值和特征向量满足Av=λv，其中A是方阵，λ是特征值，v是非零特征向量。特征分解在PCA降维和PageRank算法中有重要应用。",
        "微积分基本定理建立了微分和积分的互逆关系：若F'(x)=f(x)，则∫_a^b f(x)dx = F(b)-F(a)。这是整个微积分体系的基石。",
        "概率论中贝叶斯定理描述了条件概率的关系：P(A|B)=P(B|A)P(A)/P(B)。贝叶斯推断在垃圾邮件过滤、医学诊断和机器学习中广泛应用。",
        "信息论中熵的定义H(X)=-Σp(x)log₂p(x)度量了随机变量的不确定性，单位为比特。交叉熵常用于机器学习中的损失函数。",
        "群论研究代数结构的对称性：群是一个集合G配备封闭的二元运算·，满足结合律(a·b)·c=a·(b·c)，存在单位元e使e·a=a，且每个元素有逆元。",
        "图论中欧拉回路经过每条边恰好一次，存在的充要条件是图连通且所有顶点的度数都是偶数。哥尼斯堡七桥问题就是欧拉回路的经典例子。",
        "复杂度理论中P类问题可以在多项式时间内解决，NP类问题的解可以在多项式时间内验证。P vs NP问题是千禧年七大数学难题之一。",
        "傅里叶变换将时域信号分解为不同频率的正弦波叠加：F(ω)=∫f(t)e^{-iωt}dt。在信号处理、图像压缩和偏微分方程求解中至关重要。",
    ],
    "自然科学": [
        "量子力学中薛定谔方程 iℏ∂ψ/∂t = Ĥψ 描述了量子态随时间的演化。波函数的模方|ψ(x)|²表示在位置x找到粒子的概率密度。",
        "爱因斯坦的狭义相对论基于两个假设：物理定律在所有惯性参考系中相同，真空中光速在任何参考系中恒定。由此导出E=mc²。",
        "DNA双螺旋结构由两条反平行的多核苷酸链通过碱基配对（腺嘌呤A与胸腺嘧啶T、鸟嘌呤G与胞嘧啶C）以氢键连接而成。",
        "热力学第二定律有多种等价表述：克劳修斯说法（热量不能自发从低温传到高温）、开尔文说法（不能从单一热源取热完全转化为功）。",
        "化学中勒夏特列原理：当一个平衡系统受到外界影响（改变浓度、温度、压力）时，平衡会向减弱这种影响的方向移动。",
        "牛顿三大运动定律：第一定律（惯性）、第二定律（F=ma）、第三定律（作用力与反作用力）。这些经典力学的基础适用于宏观低速物体。",
        "细胞呼吸的三个阶段：糖酵解（细胞质中）→三羧酸循环（线粒体基质中）→氧化磷酸化（线粒体内膜上）。总共产生约30-32个ATP。",
        "板块构造论解释了地震和火山分布：地球岩石圈分为多个板块漂浮在软流圈上，板块边界分为离散型、汇聚型和转换型三种。",
    ],
    "哲学思想": [
        "苏格拉底方法通过不断提问引导对话者发现真理，体现了辩证思维的核心精神。他的名言'我知道我一无所知'揭示了真正的智慧始于承认无知。",
        "康德的道德哲学以绝对命令为核心：只按照你同时能够意愿它成为普遍法则的准则去行动。人应该被视为目的而不仅仅是手段。",
        "庄子《齐物论》主张万物一体、是非齐一，超越了二元对立的思维方式。庖丁解牛的故事展现了'以无厚入有间'的道法自然境界。",
        "维特根斯坦在《逻辑哲学论》中指出：语言的界限意味着我的世界的界限。凡可说的都能说清楚，对不可说的必须保持沉默。",
        "笛卡尔的'我思故我在(Cogito ergo sum)'确立了主体性的第一哲学地位。他在普遍怀疑的方法中发现思考本身是不可怀疑的。",
        "尼采的'上帝已死'宣告了传统价值体系的崩溃，'超人'(Übermensch)概念呼唤人超越自身成为价值的创造者而非被动的接受者。",
        "马克思的历史唯物主义认为物质生产方式决定社会结构和意识形态。社会存在决定社会意识，经济基础决定上层建筑。",
        "萨特的存在主义：存在先于本质。人首先存在，然后通过自己的选择和行动来定义自己。人是被判定为自由的——不选择也是一种选择。",
    ],
}

for domain, facts in domains.items():
    for i in range(80):
        selected = random.sample(facts, min(3, len(facts)))
        para = f"\n# {domain}笔记\n\n"
        for fact in selected:
            variants = [
                f"{fact}",
                f"值得深入理解的是：{fact}",
                f"在实际应用中，{fact[:30]}...这个知识点非常关键。",
                f"从基础到进阶：{fact}",
            ]
            para += random.choice(variants) + "\n\n"
        parts.append(para)

# English content
en_parts = []
en_code = [
    """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Time complexity: O(log n)
# Space complexity: O(1)""",

    """class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]""",

    """import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Usage:
# urls = ['http://example.com', 'http://example.org']
# results = asyncio.run(fetch_all(urls))""",

    """from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def __init__(self, radius: float):
        self.radius = radius
    def draw(self) -> None:
        print(f"Drawing circle with radius {self.radius}")

class Square:
    def __init__(self, side: float):
        self.side = side
    def draw(self) -> None:
        print(f"Drawing square with side {self.side}")

def render(shape: Drawable) -> None:
    shape.draw()

# Structural subtyping via Protocol
render(Circle(5.0))
render(Square(3.0))""",
]

en_sys_design = [
    "Load balancing distributes incoming traffic across multiple servers to improve availability and enable horizontal scaling. Common algorithms include round-robin, least connections, and consistent hashing. Layer 7 load balancers can make routing decisions based on HTTP headers and cookies.",
    "Database sharding partitions data across multiple instances based on a shard key. Range-based sharding splits by key ranges but can create hot spots. Hash-based sharding using consistent hashing provides even distribution and minimizes data movement when adding or removing nodes.",
    "Caching is a fundamental technique for reducing latency and server load. Multi-level caching (browser cache, CDN, application cache, database cache) provides diminishing returns at each level. Cache invalidation remains one of the hardest problems in computer science.",
    "Message queues decouple producers from consumers, enabling asynchronous processing and improved system resilience. Apache Kafka excels at high-throughput event streaming with its distributed commit log architecture. RabbitMQ provides flexible routing with exchanges and queues.",
    "Microservices architecture decomposes applications into independently deployable services. Each service owns its data store, communicates via well-defined APIs, and can be scaled independently. The trade-offs include increased operational complexity and eventual consistency challenges.",
    "Event-driven architecture uses events to trigger and communicate between decoupled services. Event sourcing persists the state of an entity as a sequence of events, allowing temporal queries and audit trails. CQRS separates read and write models for optimized querying.",
    "Observability rests on three pillars: metrics (quantitative measurements over time), logs (immutable timestamped records of discrete events), and traces (end-to-end request flow through distributed systems). Together they enable debugging and performance optimization.",
    "API gateway pattern provides a single entry point for all clients. It handles cross-cutting concerns: authentication, rate limiting, request routing, protocol translation, and response transformation. This simplifies client code and centralizes security policies.",
]

for i in range(60):
    code = random.choice(en_code)
    en_parts.append(f"\n# Code Example\n\n{code}\n")
    en_parts.append(f"\nThis code demonstrates important patterns in software engineering: proper error handling, type hints, efficient algorithms, and clean architecture.\n")

for i in range(40):
    design = random.choice(en_sys_design)
    en_parts.append(f"\n## System Design Note\n\n{design}\n")

# Assemble and write
with open(corpus_path, "w", encoding="utf-8") as f:
    f.write("".join(parts))
    f.write("\n".join(en_parts))

# Add expand_lifers_corpus content
sys.path.insert(0, str(ROOT / "lifers"))
from lifers.scripts import expand_lifers_corpus as ec
old_root = ec.ROOT
ec.ROOT = ROOT
try:
    generators = [
        ec._gen_social_dialogues, ec._gen_safety_protocols, ec._gen_knowledge_graph,
        ec._gen_deep_planner, ec._gen_rl_decision, ec._gen_robot_hal,
        ec._gen_simulation, ec._gen_telemetry, ec._gen_dashboard,
        ec._gen_memory_systems, ec._gen_voice_tts, ec._gen_learning_systems,
        ec._gen_core_reasoning, ec._gen_multi_agent_collaboration,
        ec._gen_perception_descriptions,
        ec._gen_proactive_behaviors, ec._gen_lifers_branded_identity,
    ]
    with open(corpus_path, "a", encoding="utf-8") as f:
        for gen in generators:
            try:
                f.write("\n" + gen())
            except Exception as e:
                print(f"  Generator failed: {e}")
finally:
    ec.ROOT = old_root

# Add auto_expand domain generators once
from lifers.scripts.auto_expand_corpus import _DOMAIN_QUEUE
with open(corpus_path, "a", encoding="utf-8") as f:
    for name, gen_func in _DOMAIN_QUEUE:
        try:
            f.write("\n" + gen_func())
        except Exception as e:
            print(f"  Domain generator {name} failed: {e}")

# Report
size_kb = corpus_path.stat().st_size // 1024
size_mb = size_kb / 1024
with open(corpus_path, "rb") as f:
    data = f.read()
text = data.decode("utf-8", errors="strict")
chars = len(text)
lines = text.count("\n")
uniq_chars = len(set(text[:200000]))

print(f"Corpus: {size_mb:.1f}MB ({size_kb}KB)")
print(f"  {lines} lines, {chars} chars, {uniq_chars} unique chars (first 200K)")
print(f"  UTF-8: VALID")

# Verify a few Chinese characters are present
import unicodedata
cjk_count = 0
for c in text[:10000]:
    name = unicodedata.name(c, "")
    if "CJK" in name:
        cjk_count += 1
print(f"  CJK ratio: {cjk_count/100:.0f}% (sample)")
print("Done")
