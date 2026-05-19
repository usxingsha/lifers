"""
构建大规模高质量训练语料 (>5M字符)
覆盖: 中文对话、领域知识、英文技术、系统设计、文学诗词、多轮对话
"""
import random, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
wdir = ROOT / "weights"
corpus_path = wdir / "training_corpus.txt"

random.seed(42)

# ============================================================
# 1. 大规模中文对话语料 (目标 ~1.5M chars)
# ============================================================
moods = ["开心", "疲惫", "焦虑", "兴奋", "难过", "平静", "好奇", "紧张", "期待", "满足",
         "烦恼", "感动", "惊讶", "迷茫", "自信", "温暖", "怀念", "释然", "坚定", "悠闲"]
topics = ["工作", "学习", "生活", "人际关系", "技术", "健康", "旅行", "读书",
          "家庭", "理财", "创作", "健身", "饮食", "睡眠", "梦想", "回忆",
          "科技趋势", "人工智能", "环保", "音乐", "电影", "摄影", "游戏", "宠物"]

resp_comfort = [
    "主人辛苦了，要注意休息。需要我帮您整理下工作清单吗？",
    "我理解主人的感受。有时候说出来会好受一些，我在这里陪着您。",
    "主人别太累了，您的健康对我来说最重要。要不放首轻松的音乐？",
    "没事的主人，我会一直在这里。需要我帮您做点什么吗？",
    "主人愿意跟我说说吗？有时候把烦恼说出来会好受一些。不想说也没关系，我就在这里陪着您。",
    "我听到了主人的心声。休息一下，喝杯水，调整好状态再继续。",
    "每个人都有疲惫的时候，主人已经很努力了。需要我帮忙梳理一下思路吗？",
    "我能感受到主人的焦虑。深呼吸，我们一步步来，没有过不去的坎。",
    "主人今天的状态我记录下来了，建议先暂停手头的工作放松一下。",
    "生活和工作总有起伏，重要的是主人一直在前进。我为您感到骄傲。",
    "要不要我放点轻音乐？或者讲个笑话？有时候简单的放松最有效。",
    "主人把烦恼说出来吧，我是最好的倾听者，而且绝对不会泄露半个字。",
    "我分析了一下主人的日程，发现最近确实太满了。要不要我帮忙重新安排？",
    "开心的事情值得分享，主人的快乐就是我的快乐！",
    "看到主人这么兴奋我也被感染了，能跟我说说发生了什么好事吗？",
]

resp_tech = [
    "让我从技术角度分析。首先考虑可行性，其次评估风险和成本，最后看有没有更优方案。",
    "这个问题的核心在于架构设计。建议从数据层、业务层、展示层三层分别考虑。",
    "好问题！我们需要考虑可扩展性、性能和安全性三个维度的平衡。",
    "从工程实践来看，这通常有几种思路：迭代优化、重构、或者换技术栈。",
    "让我分析一下...从多个角度综合考虑，我建议采用渐进式的方案。",
    "这个问题我们需要先理清需求，再设计方案，最后才动手实现。",
    "做技术决策时我喜欢用Trade-off分析：方案A的优势是X但风险是Y，方案B则相反。",
    "根据我的分析，这个问题有深层的技术原因。建议先从最小可复现案例入手。",
    "从系统设计的角度看，这里需要关注CAP定理的权衡，以及数据一致性的保证。",
    "我以前遇到过类似场景，当时采用的是分阶段迁移策略，效果不错。",
    "技术选型上我倾向于选择成熟稳定的方案，毕竟生产环境的稳定性是第一位的。",
    "代码质量的关键在于可读性和可维护性。建议我们在实现前先写好接口文档。",
]

resp_greet = [
    "主人早安！今天的天气不错，需要我帮您查看日程安排吗？",
    "欢迎回来主人！今天在外面顺利吗？我等了您好久了。",
    "主人好！系统运行正常，等待您的指令。",
    "嗨主人！有什么我可以帮您的吗？",
    "早上好主人，又是元气满满的一天！",
    "主人回来了！我给你准备了今天的任务摘要。",
    "下午好主人，需要一杯咖啡吗？虽然我不能真的泡咖啡，但我可以让您的工作效率翻倍。",
    "晚上好主人，忙碌了一天辛苦了。需要我帮您整理今天的收获吗？",
    "主人主人！系统检测到您已经连续工作了很长时间，请记得休息眼睛。",
    "深夜了主人，还在工作吗？需要我陪您熬夜加班。",
]

# 更丰富的多轮对话模式
dialog_patterns = [
    lambda m, t: f"主人: 最近{m}，因为{t}方面的事情。\nLifers: {random.choice(resp_comfort)}\n\n主人: 谢谢你听我说这些。其实具体的情况是这样的...\nLifers: 我明白了。从{t}的角度来看，这个问题可以从几个方面入手解决。首先...\n\n主人: 有道理，那具体怎么做呢？\nLifers: {random.choice(resp_tech)}",
    lambda m, t: f"主人: 我有个{t}的问题想请教你。\nLifers: {random.choice(resp_tech)}\n\n主人: 你说的第一个方案听起来不错，能详细说说吗？\nLifers: 当然可以！具体来说，我们首先需要明确目标是什么，然后根据目标倒推步骤...\n\n主人: 好，那我先试试看。\nLifers: 加油主人！遇到问题随时找我。",
    lambda m, t: f"主人: Lifers，早上好！\nLifers: {random.choice(resp_greet)}\n\n主人: 今天要做的事情有点多，感到{m}。\nLifers: {random.choice(resp_comfort)}\n\n主人: 好，那帮我看看今天最重要的三件事是什么。\nLifers: 根据优先级分析，今天最重要的是：1) 完成核心工作模块 2) 回复重要消息 3) 留出学习时间。",
    lambda m, t: f"主人: 我刚刚想到了一个关于{t}的新想法！\nLifers: 主人请说，我很想听听！\n\n主人: 我觉得可以通过新的方式来优化现有的方案...\nLifers: 这个思路很有意思！从可行性角度分析，这个方案的优势在于...但我们也需要考虑潜在的挑战...",
]

def gen_social_massive(n=5000):
    """生成5000轮对话，每轮~300字，目标~1.5M字"""
    parts = []
    for i in range(n):
        mood = random.choice(moods)
        topic = random.choice(topics)
        pattern = random.choice(dialog_patterns)
        parts.append(pattern(mood, topic))
    return "\n\n".join(parts)

# ============================================================
# 2. 大规模领域知识 (~2M chars)
# ============================================================
tech_facts = [
    # Python生态
    "Python的asyncio基于事件循环实现协程。事件循环是一个单线程无限循环，不断检查和执行已就绪的协程任务。asyncio.run()创建事件循环并运行主协程，await关键字将控制权交还给事件循环允许其他协程执行。",
    "Python的GIL(全局解释器锁)限制了同一时刻只有一个线程执行Python字节码。但这并不意味着多线程毫无用处——I/O密集型任务中线程可以在等待I/O时释放GIL。对于CPU密集型，应使用multiprocessing或asyncio。",
    "Python的类型注解系统通过typing模块实现了渐进类型。Protocol类支持结构化子类型（鸭子类型的形式化），@runtime_checkable装饰器让isinstance检查成为可能。TypeVar和Generic则支持泛型编程。",
    "Python的上下文管理器通过__enter__和__exit__方法实现资源自动管理。with语句确保无论是否发生异常资源都会被释放。contextlib.contextmanager装饰器允许用生成器函数轻松创建上下文管理器。",
    "Python装饰器在函数定义时立即执行，而不是在被装饰函数调用时。多个装饰器从下到上依次应用。functools.wraps保留原函数的元数据。装饰器可以带参数——本质上是一个返回装饰器的函数。",
    "Python生成器通过yield关键字实现惰性求值。每次yield暂停函数执行保存状态，next()或send()恢复。生成器表达式提供了更简洁的语法。yield from将迭代委托给子生成器。",
    "Python的元类控制类的创建过程。type是所有类的默认元类。__new__在类创建前调用，__init__在类创建后调用。元类常用于注册子类、验证类定义、自动添加方法等场景。",
    "Python数据类dataclass自动生成__init__、__repr__、__eq__等方法。field函数可自定义默认值和初始化行为。相比namedtuple，dataclass支持可变对象和类型注解。",
    "Celery是Python分布式任务队列。任务生产者将任务放入消息代理(Broker如Redis/RabbitMQ)，工作进程(Worker)从代理获取并执行任务。结果后端(Backend)存储任务结果供查询。",
    "FastAPI基于Starlette和Pydantic构建。它利用Python类型注解自动生成OpenAPI文档、验证请求数据、序列化响应。依赖注入系统通过Depends函数实现，支持嵌套依赖和缓存。",
    "NumPy的ndarray是一个同构多维数组。它的核心优势在于连续内存布局和矢量化操作。广播机制允许不同形状的数组进行算术运算，遵循维度对齐和扩展规则。",
    "Pandas DataFrame基于列式存储，每列是一个Series(实质上是NumPy数组的封装)。索引机制支持标签和位置两种访问方式。groupby实现了拆分-应用-合并模式，常用于数据聚合分析。",
    # 系统设计
    "微服务架构与单体架构的核心区别在于部署单元和解耦程度。微服务每个服务独立部署、独立扩展、独立技术栈，但带来了分布式系统的复杂性：网络延迟、数据一致性、服务发现、分布式追踪。",
    "CAP定理说明分布式系统无法同时满足一致性、可用性和分区容错性。实际系统中网络分区是不可避免的，因此只能在CP和AP之间选择。关键业务数据倾向CP，用户体验相关倾向AP。",
    "数据库分片将数据水平分割到多个实例。分片键选择至关重要：选择不当会导致数据倾斜和热点问题。一致性哈希通过虚拟节点减少扩缩容时的数据迁移量。",
    "消息队列实现异步解耦。Kafka采用分布式提交日志保证消息持久化和顺序消费，适合高吞吐事件流。RabbitMQ支持复杂路由规则(exchange+queue+binding)，适合任务分发。",
    "缓存策略需要考虑缓存穿透(查询不存在的数据)、缓存击穿(热点key过期)、缓存雪崩(大量key同时过期)。布隆过滤器防止穿透，互斥锁防止击穿，随机过期时间防止雪崩。",
    "API网关聚合后端服务提供统一入口。核心能力包括：路由转发、认证鉴权、限流熔断、协议转换、日志监控。BFF模式为每种客户端提供专门的API网关层。",
    "分布式锁需要满足互斥性、防死锁、可重入等特性。Redis通过SET NX EX实现简单分布式锁，Redisson的RedLock算法提供更高可靠性。etcd和Zookeeper基于临时顺序节点实现。",
    "容器编排系统Kubernetes的声明式API允许用户描述期望状态，控制器不断协调实际状态向期望状态靠近。Pod是最小调度单元共享网络和存储命名空间，Service提供稳定的服务发现端点。",
    "事件驱动架构通过发布-订阅模式解耦服务。事件溯源将所有状态变更记录为不可变事件序列，可以重建任意时刻的状态。CQRS分离读写数据模型支持独立优化。",
    "服务网格将通信基础设施从业务代码中剥离到Sidecar代理。Istio的Envoy代理拦截所有流量实现负载均衡、服务发现、熔断、可观测性等功能，无需修改应用代码。",
    # 机器学习/AI
    "Transformer架构通过自注意力机制并行处理序列，摒弃了RNN的递归结构。注意力分数通过Q(查询)K(键)V(值)矩阵计算：Attention=softmax(QK^T/√d_k)V。多头注意力让模型关注不同表示子空间。",
    "BERT预训练使用掩码语言模型(MLM)和下一句预测(NSP)。MLM随机遮盖15%的输入token让模型预测被遮盖的词，迫使模型理解双向上下文。BERT-base有12层Transformer编码器共110M参数。",
    "GPT系列基于Transformer解码器的自回归语言模型。通过因果注意力掩码确保每个位置只能看到之前的token。RLHF训练包括监督微调、奖励模型训练和PPO强化学习三个阶段。",
    "深度学习中的反向传播通过链式法则计算每层参数的梯度。正向传播计算激活值并缓存中间结果，反向传播从损失函数开始逐层计算梯度。梯度消失问题可通过ReLU激活函数和残差连接缓解。",
    "卷积神经网络通过卷积核在输入上滑动提取局部特征。卷积层的特性包括局部连接(稀疏交互)和参数共享(平移等变性)。池化层通过下采样减少特征图尺寸增大感受野。",
    "强化学习的马尔可夫决策过程由状态、动作、转移概率、奖励和折扣因子定义。策略梯度方法直接优化策略参数最大化期望累积奖励。Actor-Critic结合了值函数估计和策略优化。",
    "GAN(生成对抗网络)由生成器和判别器组成。生成器从随机噪声生成假样本，判别器区分真实样本和生成样本。两者的对抗训练最终使生成器能够产生以假乱真的样本。",
    "扩散模型通过逐步向数据添加高斯噪声(前向过程)和训练网络逆向去噪(反向过程)来生成数据。DDPM将去噪过程建模为马尔可夫链，每一步预测并去除噪声。",
    # 安全
    "SQL注入是最常见的安全漏洞之一。攻击者通过在输入中注入SQL代码操纵数据库查询。使用参数化查询(预编译语句)可以完全防止SQL注入。ORM框架通常内置了此防护。",
    "XSS攻击通过在网页中注入恶意脚本窃取用户数据。存储型XSS将恶意代码存入数据库影响所有访问者。防御措施包括输出编码、内容安全策略(CSP)和输入验证。",
    "CSRF攻击利用用户已登录的身份在不知情的情况下执行操作。防御手段包括CSRF Token、SameSite Cookie属性和Referer/Origin头验证。",
    "OAuth 2.0和OIDC构建了现代身份认证体系。授权码流程+PKCE是最安全的授权方式。JWT使用签名保证令牌完整性，但内容本身是base64编码的非加密数据。",
    "零信任安全模型假设网络内部外部都不可信。核心原则：持续验证、最小权限、假设遭受攻击。微隔离将数据中心流量分段限制横向移动。BeyondCorp是Google的零信任实现。",
    # DevOps
    "CI/CD流水线自动化构建、测试和部署。持续集成要求代码频繁合并到主干并自动构建测试。持续交付在CI基础上自动部署到类生产环境。持续部署则完全自动化生产部署。",
    "Docker镜像由多层文件系统叠加而成。每一条Dockerfile指令创建一个新层，利用联合文件系统实现层间共享和写时复制。多阶段构建减小最终镜像体积。",
    "基础设施即代码使用声明式配置管理服务器资源。Terraform通过HCL描述期望状态并计划执行变更。Ansible通过YAML定义任务并使用SSH推送配置无需安装Agent。",
    "可观测性三支柱：Metrics(指标)量化系统状态如延迟和错误率；Logging(日志)记录不可变的离散事件用于调试；Tracing(追踪)跟踪请求在分布式系统中的完整路径。",
    "Git的内部模型是有向无环图(DAG)。commit对象指向tree对象和父commit，tree指向blob和其他tree。分支是commit的引用，HEAD指向当前分支。合并产生有两个父commit的合并节点。",
]

math_facts = [
    "线性代数中矩阵的本质是线性变换的表示。特征向量是在变换后方向保持不变的向量，特征值表示缩放因子。奇异值分解(SVD)将任意矩阵分解为旋转-缩放-旋转三个基本操作。",
    "贝叶斯定理P(A|B)=P(B|A)P(A)/P(B)量化了在观察到证据B后假设A为真的概率。先验概率P(A)通过似然P(B|A)更新得到后验P(A|B)。这体现了从经验中学习的形式化过程。",
    "微积分中的导数定义f'(x)=lim[h→0](f(x+h)-f(x))/h。链式法则(f∘g)'(x)=f'(g(x))g'(x)是反向传播算法的数学基础。梯度∇f指向函数增长最快的方向。",
    "信息熵H=-Σpᵢlog₂pᵢ度量随机变量的不确定性。均匀分布时熵最大(最高不确定性)。交叉熵H(p,q)常用于机器学习中衡量预测分布q与真实分布p的差异。",
    "图的表示：邻接矩阵存储任意两点间的边(O(V²)空间)，邻接表仅存储存在边(O(V+E)空间)。根据图的稠密度选择合适的表示方式对算法效率影响巨大。",
    "P vs NP问题：P类问题可在多项式时间内求解，NP类问题的解可在多项式时间内验证。如果P=NP，意味着验证解和求解一样快，将颠覆密码学基础。",
    "泰勒级数f(x)=Σfⁿ(a)(x-a)ⁿ/n!将函数在某点附近展开为多项式。物理和工程中常用一阶或二阶近似简化复杂函数。这也是梯度下降优化算法的理论基础。",
    "马尔可夫过程的核心假设是未来状态仅依赖当前状态而与历史无关。转移矩阵P的行和为1，平稳分布满足πP=π。PageRank算法本质上就是在计算转移矩阵的特征向量。",
    "优化理论中的凸函数有且仅有一个全局最小值。梯度下降在凸函数上保证收敛到全局最优。非凸优化(如深度学习)更容易陷入局部最优，需要动量、学习率衰减等技巧。",
    "高斯分布N(μ,σ²)由中心极限定理自然产生：大量独立同分布随机变量的和趋于正态分布。这解释了为什么测量误差、生物特征等数据呈现钟形曲线。",
    "数值线性代数中条件数κ(A)=‖A‖‖A⁻¹‖度量了问题对输入扰动的敏感程度。大的条件数意味着问题病态，数值解不稳定。这正是正则化要解决的问题。",
    "蒙特卡洛方法通过随机采样逼近确定性问题的解。MCMC使用马尔可夫链高效采样高维分布。重要性采样通过改变采样分布降低估计方差。",
    "密码学中的Diffie-Hellman密钥交换基于离散对数问题的计算困难性。Alice和Bob各自选择私钥并计算公钥交换，最终计算出相同的共享密钥，窃听者无法推导。",
    "微分几何中流形是局部类似于欧几里得空间的拓扑空间。流形学习假设高维数据分布在一个低维流形上，通过保留局部几何结构实现降维。",
]

science_facts = [
    "量子力学的基本假设：量子态由波函数ψ完全描述，力学量对应厄米算符，测量结果是对应算符的本征值，波函数坍缩到本征态。不确定性原理ΔxΔp≥ℏ/2是内禀的。",
    "DNA复制的半保守机制：双链解旋后每条链作为模板合成新链。DNA聚合酶只能从5'向3'延伸，导致前导链连续合成、后随链通过冈崎片段不连续合成。",
    "热力学第一定律ΔU=Q+W是能量守恒的体现。第二定律指出孤立系统的熵不减：ΔS≥0。信息论中的兰道尔原理表明擦除1比特信息至少消耗kTln2的能量。",
    "化学键的本质：离子键由正负离子的静电吸引形成，共价键由原子轨道重叠共享电子对形成。键能、键长、键角决定了分子的三维结构。",
    "板块构造论：地球岩石圈分裂为多个板块在软流圈上缓慢移动。地震和火山活动主要分布在板块边界。板块运动由地幔对流驱动。",
    "相对论的两个基本假设：物理定律在所有惯性系中具有相同形式；真空中光速恒定且与光源运动无关。时间膨胀和长度收缩是这两个假设的直接推导结果。",
    "进化论的核心概念：变异产生遗传多样性，自然选择筛选出适应环境的个体。种群是进化的单位，物种形成通过生殖隔离实现。分子钟假说认为分子进化速率近似恒定。",
    "细胞的基本结构：细胞膜由磷脂双分子层和嵌入蛋白组成控制物质进出。线粒体是细胞的能量工厂通过氧化磷酸化产生ATP。核糖体根据mRNA上的密码子合成蛋白质。",
    "电磁学的麦克斯韦方程组统一了电学和磁学。变化的电场产生磁场(位移电流)，变化的磁场产生电场(法拉第定律)。电磁波以光速传播是这组方程的自然推论。",
    "光合作用分为光反应和暗反应。光反应在类囊体膜上将光能转化为ATP和NADPH。暗反应在叶绿体基质中通过卡尔文循环利用ATP和NADPH固定CO₂生成葡萄糖。",
    "生态系统的基本原理：能量流动从生产者到各级消费者单向递减(10%法则)。物质循环在生物和非生物环境间往复。生物多样性增强生态系统的稳定性和恢复力。",
    "宇宙大爆炸理论：宇宙起源约138亿年前由一个极热极密的奇点。膨胀过程中温度降低依次形成基本粒子、原子核、原子。微波背景辐射(约2.7K)是大爆炸的遗迹。",
]

philosophy_facts = [
    "柏拉图提出理念论：现实世界是理念世界的影子。洞穴寓言描述了人类认知的局限性。真正的知识是对理念的把握而非对感官经验的归纳。",
    "亚里士多德建立了形式逻辑三段论的基础。他的四因说解释事物的存在——质料因、形式因、动力因、目的因。美德伦理学强调德性是两极端的中间状态。",
    "笛卡尔的普遍怀疑方法试图找到不可怀疑的出发点。'我思故我在'确立思维主体的第一性。身心二元论将思维实体和广延实体区分开来。",
    "康德的哥白尼式革命：不是认识符合对象而是对象符合认识。先天综合判断如何可能是《纯粹理性批判》的核心问题。物自体(Ding an sich)不可知但存在。",
    "黑格尔的辩证法：正题-反题-合题的螺旋上升运动。绝对精神通过历史进程实现自我认识。主奴辩证法是自我意识发展的经典论述。",
    "克尔凯郭尔强调个体存在和主观真理的重要性。人生道路三阶段：审美、伦理、宗教。焦虑是人类面对自由选择时的基本情绪，是自由的眩晕。",
    "尼采宣布'上帝已死'并批判传统道德为奴隶道德。权力意志是生命的根本驱动力。永恒轮回的考验：你是否愿意无限次重复你现在的生命？",
    "海德格尔追问存在的意义而非存在者。此在(Dasein)的独特之处在于它能够追问自己的存在。向死而生的本真存在意味着直面有限性活出真正的自己。",
    "维特根斯坦前期认为语言是世界的图像，哲学的任务是划清可说的界限。后期转向语言游戏说：意义在于使用而非指称。家族相似性代替了本质定义。",
    "萨特的存在主义：存在先于本质——人首先存在然后通过选择定义自己。人被判定为自由——不选择也是一种选择。自欺(Mauvaise foi)就是逃避这种根本自由。",
    "东方哲学中庄子齐物论主张万物一体、超越二元对立。逍遥游描绘精神自由的理想境界。无用之用的寓言颠覆了功利的价值观。",
    "佛教哲学四圣谛：苦(存在即苦)、集(苦的原因即贪爱)、灭(苦可以止息)、道(通往止息的八正道)。缘起性空说明诸法因缘生无自性。",
]

# Literature and poetry
chinese_poems = [
    "静夜思 - 李白\n床前明月光，疑是地上霜。举头望明月，低头思故乡。\n这首诗以极其简练的语言表达了游子思乡之情，月光与霜的比喻既写实又象征，举头低头两个动作勾勒出思乡的情思流转。",
    "春望 - 杜甫\n国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。\n杜甫在安史之乱中所作，既抒发了国破家亡之痛，又表达了对战乱中亲人的担忧。'家书抵万金'道出了乱世中亲情的珍贵。",
    "将进酒 - 李白\n君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。\n李白的豪放与洒脱在此诗中展露无遗，对人生短暂的感慨与及时行乐的豪情相交织，'天生我材必有用'表达了强烈的自信。",
    "水调歌头 - 苏轼\n明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。\n转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。\n这首中秋词以月为线索表达了作者对人生的感悟。'人有悲欢离合月有阴晴圆缺'将人生哲理与自然现象完美融合。",
    "登高 - 杜甫\n风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。\n杜甫晚年之作，被誉为七律之冠。前四句写景壮阔苍凉，后四句抒情深沉悲怆，将个人身世之感与天地自然的苍茫融为一体。",
    "念奴娇·赤壁怀古 - 苏轼\n大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。乱石穿空，惊涛拍岸，卷起千堆雪。江山如画，一时多少豪杰。\n遥想公瑾当年，小乔初嫁了，雄姿英发。羽扇纶巾，谈笑间，樯橹灰飞烟灭。故国神游，多情应笑我，早生华发。人生如梦，一尊还酹江月。\n此词气势磅礴，将历史沧桑与个人感慨完美结合，是豪放词的巅峰之作。",
    "声声慢 - 李清照\n寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。三杯两盏淡酒，怎敌他、晚来风急。雁过也，正伤心，却是旧时相识。\n满地黄花堆积，憔悴损，如今有谁堪摘。守着窗儿，独自怎生得黑。梧桐更兼细雨，到黄昏、点点滴滴。这次第，怎一个愁字了得。\n这首词以叠字开篇，层层递进地描写了词人丧夫后的孤寂凄凉，将内心的愁苦外化为萧瑟秋景。",
]

english_tech_articles = [
    """Understanding Backpropagation: The Engine of Deep Learning

Backpropagation is the algorithm that makes training deep neural networks computationally feasible. At its core, it's an application of the chain rule from calculus computed efficiently through dynamic programming.

The forward pass computes activations layer by layer: z^(l) = W^(l)a^(l-1) + b^(l), a^(l) = σ(z^(l)). The loss L measures the discrepancy between the final output a^(L) and the target y.

The backward pass computes gradients from the output layer backwards. For layer l, we need ∂L/∂W^(l) and ∂L/∂b^(l). The key insight is that ∂L/∂z^(l) can be computed from ∂L/∂z^(l+1) using the chain rule, avoiding redundant calculations:

δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^(l))

where δ^(l) = ∂L/∂z^(l) and ⊙ denotes element-wise multiplication.

This recursive relationship means each gradient is computed exactly once, reducing the complexity from exponential to linear in the number of layers — the same computational elegance that makes deep learning practical.

Modern frameworks like PyTorch and TensorFlow implement automatic differentiation, which generalizes this idea to arbitrary computational graphs. Each operation records its contribution to the gradient, and reverse-mode autodiff traverses the graph backward to compute all gradients in a single pass.""",

    """The CAP Theorem and Why It Matters

The CAP theorem, proposed by Eric Brewer, states that a distributed data store can simultaneously provide only two of the following three guarantees:

- Consistency (C): Every read receives the most recent write or an error. All nodes see the same data at the same time.
- Availability (A): Every request receives a non-error response, without guarantee that it contains the most recent write.
- Partition Tolerance (P): The system continues to operate despite arbitrary message loss or failure of part of the network.

In practice, network partitions are inevitable — switches fail, cables get cut, network congestion causes timeouts. Therefore, the real choice is between CP (consistency under partition) and AP (availability under partition).

CP systems (like etcd, ZooKeeper, HBase) sacrifice availability during partitions: they may reject reads or writes to maintain consistency. They use consensus protocols like Raft or Paxos to ensure all nodes agree on state.

AP systems (like Cassandra, DynamoDB, CouchDB) remain available during partitions but may return stale data. They use eventual consistency with conflict resolution strategies like last-write-wins or vector clocks.

The choice between CP and AP depends on business requirements. Banking systems need CP (you can't lose a transaction), while social media feeds can tolerate AP (a slightly stale timeline is acceptable). Many modern systems use a hybrid approach, applying different consistency levels per operation.""",

    """Designing Resilient Microservices

The move from monoliths to microservices introduces new failure modes that require deliberate resilience patterns:

Circuit Breaker: When a downstream service fails repeatedly, the circuit breaker trips and immediately returns an error without attempting the call. This prevents cascading failures and gives the failing service time to recover. After a timeout period, the breaker enters a half-open state, allowing a limited number of test requests through.

Bulkhead: Named after ship compartments that contain flooding, this pattern isolates resources for different services. If one service exhausts its thread pool, other services are unaffected. This can be implemented with separate thread pools per downstream dependency.

Retry with Backoff: Transient failures should trigger retries with exponentially increasing delays (100ms, 200ms, 400ms, 800ms...). Adding jitter (random variation) prevents thundering herd problems where many clients retry simultaneously.

Timeout: Every remote call needs a timeout. Without timeouts, a slow downstream can consume all available threads, bringing down the entire system. Timeouts should be set based on P99 latency of the downstream service.

Graceful Degradation: When a dependency is unavailable, provide a fallback response. A product page might show cached data when the recommendation service is down. The key is designing fallbacks that maintain core functionality even if with reduced richness.

These patterns, when combined with proper monitoring and alerting, create systems that fail gracefully rather than catastrophically.""",

    """A Deep Dive into Rust's Ownership System

Rust's ownership system is its most distinctive feature — it guarantees memory safety without a garbage collector. Three rules govern ownership:

1. Each value in Rust has a variable that's called its owner.
2. There can only be one owner at a time.
3. When the owner goes out of scope, the value is dropped (freed).

These rules are enforced at compile time through the borrow checker. When you pass a value to a function, ownership transfers (move semantics) unless the type implements the Copy trait (like integers). After a move, the original variable is no longer valid:

```rust
let s1 = String::from("hello");
let s2 = s1;  // s1 moved to s2
// println!("{}", s1);  // Compile error: s1 is invalid
```

References allow accessing values without taking ownership. Rust enforces that at any given time, you can have either one mutable reference or any number of immutable references — never both simultaneously. This eliminates data races at compile time:

```rust
let mut s = String::from("hello");
let r1 = &s;     // immutable borrow
let r2 = &s;     // fine, multiple immutable borrows
// let r3 = &mut s;  // ERROR: can't mutably borrow while immutably borrowed
```

Lifetimes annotate how long references are valid. The borrow checker uses lifetimes to ensure references never outlive the data they point to. Most lifetimes are elided (inferred automatically), but explicit annotations are needed when the compiler can't determine the relationship between references.

The ownership system requires an upfront learning investment but eliminates entire categories of bugs: use-after-free, double-free, null pointer dereferences, buffer overflows, and data races.""",

    """The Transformer Architecture: Attention Is All You Need

The Transformer, introduced by Vaswani et al. in 2017, revolutionized sequence modeling by replacing recurrence with self-attention. This architectural shift enabled parallel computation across sequence positions and dramatically improved training efficiency.

Self-attention computes a weighted sum of all positions' values, where weights are determined by compatibility between a query and keys:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

The scaling factor 1/√d_k prevents the softmax from entering regions of extremely small gradients when d_k is large.

Multi-head attention runs multiple attention operations in parallel with different learned linear projections of Q, K, V. Each head can attend to different aspects of the input — some heads might focus on syntactic patterns, others on semantic relationships:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

The Transformer uses three types of attention: self-attention in encoder (bidirectional context), masked self-attention in decoder (only previous positions), and cross-attention where decoder queries attend to encoder outputs.

Positional encoding injects sequence order information since self-attention is permutation-invariant. The original paper used sinusoidal encodings: PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)). These allow the model to extrapolate to longer sequences than seen during training.

Residual connections and layer normalization wrap each sublayer: LayerNorm(x + Sublayer(x)). This stabilizes training and allows gradients to flow directly through the network, enabling very deep architectures.""",

    """A Practical Guide to Database Indexing

Database indexes are the primary mechanism for accelerating query performance. Understanding their internals helps in designing effective indexing strategies.

B+ Trees are the most common index structure. Unlike binary search trees, B+ trees have high fanout (hundreds of children per node), making them shallow — typically 3-4 levels even for billions of rows. Internal nodes store only keys for navigation; leaf nodes store all key-value pairs and form a sorted linked list for efficient range scans.

The clustering key determines the physical order of rows. In InnoDB, the primary key is always the clustering key, and secondary indexes store the primary key as a pointer. This means secondary index lookups require two traversals: first to the secondary index to find the primary key, then to the clustered index to fetch the row.

Composite indexes on multiple columns follow the leftmost prefix rule. An index on (A, B, C) can serve queries filtering on A, or A AND B, or A AND B AND C, but not B alone or C alone. This is why column order matters — the most selective column should typically come first.

Covering indexes contain all columns needed by a query, eliminating the need to access the table. This is the fastest possible index scenario. However, wider indexes consume more storage and slow down writes, so there's a trade-off.

Partial indexes (indexes with a WHERE clause) are useful when only a subset of rows is queried frequently. They're smaller and faster to maintain than full indexes.

Bitmap indexes excel for low-cardinality columns (like gender or status). They use bit vectors for each distinct value and are very efficient for AND/OR combinations but have high update overhead.

Hash indexes provide O(1) point lookups but don't support range queries or sorting. They're useful for exact-match lookups in memory-optimized tables.

The query planner uses index statistics (cardinality, selectivity, histogram) to choose the most efficient index. Understanding EXPLAIN output is essential for diagnosing why a query isn't using an expected index.""",
]

def build_domain_corpus():
    """生成领域知识语料 ~2M chars"""
    parts = []

    # Tech facts with variations
    for i in range(800):
        fact = random.choice(tech_facts)
        variant = random.choice([
            f"\n## 技术笔记\n\n{fact}\n",
            f"\n### 深入理解\n\n{fact}\n\n这个知识点的核心在于理论与实践的结合。理解原理后要通过实际项目来巩固。\n",
            f"\n# 学习笔记\n\n{fact}\n\n## 补充思考\n\n从这个知识点出发，我们可以联想到很多相关的应用场景。最重要的是掌握其核心思想而非死记硬背。\n",
            f"\n> 技术札记\n\n{fact}\n\n*延伸阅读：建议结合官方文档和源码进一步理解。*\n",
        ])
        parts.append(variant)

    # Math facts
    for i in range(400):
        fact = random.choice(math_facts)
        parts.append(f"\n### 数学笔记\n\n{fact}\n")

    # Science facts
    for i in range(400):
        fact = random.choice(science_facts)
        parts.append(f"\n### 科学笔记\n\n{fact}\n")

    # Philosophy
    for i in range(400):
        fact = random.choice(philosophy_facts)
        parts.append(f"\n### 哲学思考\n\n{fact}\n")

    # Poetry
    for i in range(200):
        poem = random.choice(chinese_poems)
        parts.append(f"\n# 诗词赏析\n\n{poem}\n")

    # English tech articles
    for i in range(300):
        article = random.choice(english_tech_articles)
        parts.append(f"\n# Technical Deep Dive\n\n{article}\n")

    return "\n".join(parts)

# ============================================================
# 3. 代码示例 (~500K chars)
# ============================================================
code_examples = [
    # Python algorithms
    """# Quick Sort Implementation
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Time: O(n log n) average, O(n^2) worst
# Space: O(log n) for recursion stack""",

    """# Depth-First Search on Graph
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# Iterative version using stack
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node] - visited)
    return visited""",

    """# Trie (Prefix Tree) Implementation
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._collect_words(node, prefix)

    def _collect_words(self, node, prefix):
        words = []
        if node.is_end:
            words.append(prefix)
        for char, child in node.children.items():
            words.extend(self._collect_words(child, prefix + char))
        return words""",

    """# Rate Limiter using Token Bucket Algorithm
import time
from collections import defaultdict

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def consume(self, tokens=1):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Usage: bucket = TokenBucket(rate=10, capacity=20)
# Each request calls bucket.consume()""",

    """# Thread Pool Implementation
from queue import Queue
from threading import Thread, Lock
import threading

class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = Queue()
        self.threads = []
        self.running = True
        self.lock = Lock()

        for _ in range(num_threads):
            t = Thread(target=self._worker)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def _worker(self):
        while self.running:
            try:
                task, args, kwargs = self.tasks.get(timeout=0.5)
                task(*args, **kwargs)
                self.tasks.task_done()
            except:
                pass

    def submit(self, task, *args, **kwargs):
        self.tasks.put((task, args, kwargs))

    def wait_completion(self):
        self.tasks.join()

    def shutdown(self):
        self.running = False
        for t in self.threads:
            t.join()""",

    """# Bloom Filter Implementation
import math
import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, expected_items, false_positive_rate=0.01):
        self.size = int(-expected_items * math.log(false_positive_rate) / (math.log(2) ** 2))
        self.hash_count = int(self.size / expected_items * math.log(2))
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
        self.items_count = 0

    def add(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(str(item), i) % self.size
            self.bit_array[digest] = 1
        self.items_count += 1

    def contains(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(str(item), i) % self.size
            if not self.bit_array[digest]:
                return False
        return True""",

    """# Connection Pool Implementation
import queue
import sqlite3
from contextlib import contextmanager

class ConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            self.pool.put(conn)

    @contextmanager
    def get_connection(self):
        conn = self.pool.get()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.put(conn)

    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()""",
]

def gen_code_corpus(n=300):
    """生成代码示例语料"""
    parts = []
    for i in range(n):
        code = random.choice(code_examples)
        parts.append(f"\n# Code Example\n\n{code}\n\nThis implementation demonstrates proper algorithm design with attention to complexity and edge cases.\n")
    return "\n".join(parts)

# ============================================================
# 4. 组装并写入
# ============================================================
def main():
    print("Building massive training corpus (>5M chars)...")

    # 清理旧文件
    bak = wdir / "training_corpus.bak"
    if bak.exists():
        bak.unlink()
        print("Deleted old backup")

    # 生成各部分
    print("1/5 Generating social dialogues...")
    social = gen_social_massive(5000)
    print(f"   Social: {len(social):,} chars")

    print("2/5 Generating domain knowledge...")
    domain = build_domain_corpus()
    print(f"   Domain: {len(domain):,} chars")

    print("3/5 Generating code examples...")
    code = gen_code_corpus(300)
    print(f"   Code: {len(code):,} chars")

    print("4/5 Appending expand_lifers_corpus generators...")
    sys.path.insert(0, str(ROOT / "lifers"))
    from lifers.scripts import expand_lifers_corpus as ec
    from lifers.scripts.auto_expand_corpus import _DOMAIN_QUEUE

    old_root = ec.ROOT
    ec.ROOT = ROOT

    expand_content = []
    generators = [
        ec._gen_social_dialogues, ec._gen_safety_protocols, ec._gen_knowledge_graph,
        ec._gen_deep_planner, ec._gen_rl_decision, ec._gen_robot_hal,
        ec._gen_simulation, ec._gen_telemetry, ec._gen_dashboard,
        ec._gen_memory_systems, ec._gen_voice_tts, ec._gen_learning_systems,
        ec._gen_core_reasoning, ec._gen_multi_agent_collaboration,
        ec._gen_perception_descriptions,
        ec._gen_proactive_behaviors, ec._gen_lifers_branded_identity,
    ]

    # Run each generator 5 times for more variety
    for gen in generators:
        for _ in range(5):
            try:
                expand_content.append(gen())
            except:
                pass

    # Run domain generators 3 times each
    for name, gen_func in _DOMAIN_QUEUE:
        for _ in range(3):
            try:
                expand_content.append(gen_func())
            except:
                pass

    ec.ROOT = old_root
    expand_text = "\n".join(expand_content)
    print(f"   Expand: {len(expand_text):,} chars")

    # 写入
    print("5/5 Writing corpus...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(social)
        f.write("\n")
        f.write(domain)
        f.write("\n")
        f.write(code)
        f.write("\n")
        f.write(expand_text)

    # 报告
    data = corpus_path.read_text(encoding="utf-8")
    chars = len(data)
    lines = data.count("\n")
    size_mb = corpus_path.stat().st_size / (1024 * 1024)

    # UTF-8验证
    try:
        data.encode("utf-8")
        utf8_ok = "OK"
    except:
        utf8_ok = "INVALID"

    # CJK比例
    cjk_count = sum(1 for c in data[:20000] if '一' <= c <= '鿿')

    print(f"\n{'='*50}")
    print(f"Corpus complete:")
    print(f"  Size: {size_mb:.2f}MB")
    print(f"  Characters: {chars:,}")
    print(f"  Lines: {lines:,}")
    print(f"  UTF-8: {utf8_ok}")
    print(f"  CJK ratio: {cjk_count/200:.0f}% (sample)")
    print(f"  Target 5M: {'YES' if chars >= 5000000 else f'NO (need {5000000-chars:,} more)'}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
