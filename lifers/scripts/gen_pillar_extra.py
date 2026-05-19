#!/usr/bin/env python3
"""
补充缺失支柱语料 — 视觉/语音/KG/群体/仿真/遥测/RL/仪表盘
使用单引号避免中文引号冲突，直接追加到主语料文件
"""
import sys, os, random, time

sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
CORPUS_PATH = Path(__file__).resolve().parent.parent / "weights" / "training_corpus.txt"
random.seed(42)

# ═══════════════════════════════════════════
# 中文补充领域 (8个新支柱领域)
# ═══════════════════════════════════════════

ZH_EXTRA_DOMAINS = {
    '计算机视觉与感知': [
        ('图像分类与目标检测',
         '计算机视觉是让机器理解视觉世界的核心技术。图像分类从早期的LeNet到ResNet再到Vision Transformer，经历了从浅层CNN到深层残差网络再到纯注意力机制的演进。目标检测回答了「哪里有什么」的问题——从两阶段的R-CNN系列到单阶段的YOLO、SSD，再到基于Transformer的DETR，检测速度和精度不断提升。特征金字塔网络(FPN)通过多尺度特征融合解决了小目标检测困难的问题。非极大值抑制(NMS)消除重复检测框。mAP(平均精度均值)是衡量检测器性能的标准指标。在实际应用中，目标检测是自动驾驶感知、安防监控和工业质检的基础能力。'),
        ('图像分割与场景理解',
         '语义分割为图像中的每个像素分配类别标签，是自动驾驶和医学影像分析的核心技术。全卷积网络(FCN)开创了端到端的像素级预测范式。U-Net通过编码器-解码器结构和跳跃连接，在医学图像分割中表现优异，成为生物医学影像分析的事实标准。DeepLab系列通过空洞卷积扩大感受野同时保持分辨率。实例分割进一步区分同一类别中的不同实例，Mask R-CNN在Faster R-CNN基础上增加分割分支实现了这一目标。全景分割统一了语义分割和实例分割，要求为每个像素分配语义类别和实例ID。'),
        ('3D视觉与深度估计',
         '从二维图像恢复三维信息是计算机视觉的核心挑战之一。双目立体视觉模拟人眼视差原理，通过匹配左右图像中的对应点计算深度。结构光通过投影已知图案并分析形变来重建表面形状，广泛用于工业精密测量。ToF飞行时间相机通过测量光脉冲往返时间直接获取深度图。单目深度估计使用深度学习从单张图像推断深度，虽然精度不及双目或结构光，但成本低、部署简单。点云处理是3D视觉的重要分支，PointNet开创了直接处理无序点云的深度学习方法。3D目标检测在自动驾驶和机器人抓取中至关重要。'),
        ('视觉SLAM与定位',
         '视觉SLAM通过相机传感器实现机器人的同步定位与地图构建。ORB-SLAM是视觉SLAM的标志性系统，通过ORB特征点进行跟踪、建图和回环检测。直接法SLAM跳过特征提取直接使用像素强度进行位姿估计，在纹理稀疏的环境中更有优势。视觉惯性里程计(VIO)融合相机和IMU数据，利用IMU的高频运动信息弥补视觉在快速运动时的不足。回环检测是消除累积漂移的关键，基于词袋模型(BoW)的场景识别是常用方案。视觉重定位通过图像检索或场景坐标回归估计相机六自由度位姿。'),
        ('视频理解与行为识别',
         '视频理解扩展了图像分析的时间维度，处理的是时空数据。3D卷积网络(C3D、I3D)通过三维卷积核同时提取空间和时间特征。双流网络分别处理RGB帧和光流，融合空间表观信息和运动信息。时序分割网络(TSN)通过稀疏采样长视频关键帧实现高效行为识别。SlowFast网络的双路径设计模拟了生物视觉系统中M细胞和P细胞的分工。视频目标分割(VOS)在半监督或无监督条件下从视频中分割出特定目标。时序行为检测不仅识别行为类别还定位时间边界，在安防监控和体育分析中应用广泛。'),
    ],
    '语音与音频智能': [
        ('语音识别技术',
         '自动语音识别(ASR)将音频信号转化为文本，是人机语音交互的第一步。传统ASR系统由声学模型、语言模型和发音词典三部分组成，GMM-HMM是经典框架。端到端ASR(如DeepSpeech、LAS)直接从音频学习到文本的映射，大幅简化了系统架构。CTC损失函数解决了输入输出序列长度不匹配的对齐问题。Transformer/Conformer架构在ASR中表现优异，Conformer通过卷积增强自注意力更好地捕捉局部和全局特征。Whisper等多语言ASR模型展现了强大的跨语言泛化能力。流式ASR要求在低延迟下输出部分识别结果，对模型架构和推理引擎都有特殊要求。'),
        ('语音合成技术',
         '文本到语音(TTS)将文字转化为自然流畅的语音。参数化TTS通过预测声学参数(基频、频谱)再通过声码器合成波形。端到端TTS(Tacotron、FastSpeech)直接从字符序列生成声学特征。FastSpeech通过非自回归架构解决了Tacotron的重复词和漏词问题，同时大幅提升推理速度。神经声码器(WaveNet、WaveGlow、HiFi-GAN)从声学特征生成高质量波形。多说话人TTS通过说话人嵌入实现单一模型多声音合成。零样本语音克隆从几秒音频即可学习目标说话人的音色特征。情感TTS控制合成语音的情感表达，让合成语音更自然、更具表现力。'),
        ('声音事件检测',
         '声音事件检测(SED)识别音频中发生的声音事件及其时间边界，在智能家居、工业监控和城市噪声管理中有着广泛应用。传统方法使用MFCC特征配合GMM或HMM进行分类。深度学习方法使用CNN或CRNN在梅尔频谱图上进行事件检测。注意力机制帮助模型关注相关的时间频率区域。弱监督SED仅需音频级别的标签而非精确的时间标注，大幅降低了数据标注成本。少样本声音检测通过学习声音类别的通用表示，在处理长尾分布的稀有声音事件时特别有价值。多通道声源定位结合麦克风阵列估计声源方向。'),
        ('说话人识别与声纹技术',
         '说话人识别通过声纹特征确认说话人身份。说话人验证判断声纹是否与声称的身份一致。说话人辨识从多个注册声纹中确定说话人身份。i-vector是经典的统计声纹方法，通过因子分析将变长语音映射到固定维度的身份向量。x-vector使用深度神经网络提取更具判别力的声纹嵌入。ECAPA-TDNN通过通道注意力和统计池化等机制在多个基准上达到最优性能。端到端声纹系统直接从语音波形学习到身份表示。抗欺骗检测区分真实语音和录音重放、语音合成、语音转换等欺骗攻击，是声纹系统安全性的关键保障。'),
        ('多模态语音理解',
         '多模态语音理解融合音频和视觉等多模态信息提升语音感知能力。视听语音识别(AVSR)结合唇动视觉信息提高在噪声环境中的识别准确率。McGurk效应证明了视觉信息对人类语音感知的影响。视听说话人分离通过视频中的面部信息辅助区分不同说话人。LipNet是首个端到端的句子级唇读模型。视听语音分离通过面部运动信息从混合音频中隔离目标说话人的语音。多模态情感识别融合语音、面部表情和文本信息更准确地识别说话人的情感状态。'),
    ],
    '知识图谱与推理': [
        ('知识图谱构建',
         '知识图谱将知识组织为实体-关系-实体的三元组结构，是让机器理解和推理世界知识的基础设施。知识抽取包括命名实体识别从文本中识别实体提及、关系抽取确定实体间的语义关系、实体消歧将文本提及链接到知识库中的正确实体。远程监督通过将已有知识库与文本对齐自动生成训练数据。开放信息抽取(OIE)不限于预定义关系，直接从文本中抽取任意关系三元组。知识融合解决不同来源知识的对齐和合并问题。时序知识图谱记录知识随时间的变化，支持历史回溯和趋势预测。'),
        ('图神经网络与推理',
         '图神经网络(GNN)在知识图谱上进行表示学习和推理。GCN通过邻居聚合更新节点表示，每一层聚合一跳邻居的信息。GAT引入注意力机制为不同邻居分配不同的重要性权重。R-GCN为不同关系类型学习不同的变换矩阵，适合知识图谱的多关系特点。知识图谱嵌入方法(TransE、RotatE、ComplEx)将实体和关系映射到低维向量空间，通过向量运算进行链接预测。基于GNN的推理可以从观察到的模式进行归纳推理，预测缺失的三元组。逻辑规则与GNN结合的神经符号推理兼具数据驱动的学习和逻辑驱动的可解释性。'),
        ('知识问答系统',
         '知识图谱问答(KBQA)使用自然语言问题从知识图谱中检索和推理答案。语义解析方法将自然语言问题转化为结构化查询(如SPARQL)，然后在知识图谱上执行。信息检索方法直接从知识图谱中检索候选答案，再通过排序模型选择最佳答案。多跳问答需要通过多个中间实体和关系的推理链才能到达最终答案，对模型的推理能力要求更高。时序问答涉及时间约束的推理。对话式问答在对话上下文中理解指代和省略。大语言模型与知识图谱的结合通过检索增强生成(RAG)将外部知识融入生成过程。'),
        ('常识推理',
         '常识推理是AI接近人类智能的关键门槛。ConceptNet是最大的常识知识库之一，包含了数十种关系类型的日常概念关联。ATOMIC知识库专门记录社会常识——关于事件的前因、后果和心理状态。COMET模型通过在常识知识库上训练GPT，能够自动生成关于事件前因后果的常识推理。物理常识推理要求理解物理世界的基本规律。社会常识推理涉及理解人类意图、情感和社交规范。时间常识推理需要理解事件的典型持续时间和先后顺序。常识推理的主要挑战在于常识是隐含的、语境依赖的，很少被明确写出。'),
        ('知识图谱应用',
         '知识图谱在工业界有着广泛的应用。搜索引擎使用知识图谱增强搜索结果，Google Knowledge Graph覆盖数十亿实体。推荐系统中知识图谱提供丰富的物品属性信息和用户-物品交互路径，提升推荐的准确性和可解释性。医疗知识图谱整合疾病、症状、药物和诊疗指南，辅助临床诊断和用药推荐。金融知识图谱通过分析企业、个人和交易的关系网络进行风控和反欺诈。法律知识图谱将法律条文、案例和司法解释结构化，辅助法律研究和判决预测。对话系统中知识图谱为聊天机器人提供背景知识和推理能力。'),
    ],
    '多智能体与群体智能': [
        ('多智能体强化学习',
         '多智能体强化学习(MARL)研究多个智能体在共享环境中同时学习和交互。每个智能体的策略变化改变了其他智能体所处的环境，破坏了单智能体RL中的马尔可夫性和平稳性假设。集中训练分散执行(CTDE)是主流范式，在训练时利用全局信息指导学习，执行时仅依赖局部观测。MADDPG为每个智能体学习集中的Q函数，在混合合作竞争场景中表现良好。QMIX通过单调混合网络分解联合Q值，保证集中策略与分散执行策略的一致性。MAPPO扩展PPO到多智能体场景，因其简洁和稳定而流行。'),
        ('群体智能与涌现行为',
         '群体智能研究大量简单个体通过局部交互涌现复杂集体行为的机制。蚁群算法模拟蚂蚁通过信息素通信实现最短路径搜索。粒子群优化(PSO)模拟鸟群觅食行为，每个粒子根据自身和群体的历史最佳位置更新搜索方向。Boids模型通过分离、对齐和凝聚三条简单规则模拟鸟群和鱼群的集群运动。涌现行为在没有集中控制的情况下自发形成——群体的能力超出了任何个体的能力之和。自组织关键性是某些复杂系统自发演化到临界状态的现象。'),
        ('协作与通信机制',
         '多智能体系统中的通信机制使智能体能够共享信息和协调行动。显式通信通过专门的通信信道传递消息，DIAL算法通过可微的通信通道端到端学习通信协议。隐式通信通过环境状态变化传递信息，无需专门通信信道。通信语言可以从连续向量演化为离散符号，从而形成类似人类语言的结构化通信。基于注意力的通信让智能体选择性地关注最重要的通信伙伴，提高通信效率。'),
        ('多机器人协作',
         '多机器人系统通过协作完成单机器人难以或无法独立完成的任务。任务分配解决哪个机器人执行哪个子任务的问题，市场机制通过拍卖将任务分配给出价最低的机器人。编队控制使多机器人保持期望的空间配置，领导者-跟随者法简洁高效。多机器人SLAM通过共享地图和位姿信息加速探索并提高建图精度。分布式目标跟踪通过融合多个机器人从不同视角的观测提高跟踪精度。群体操作任务如多机器人搬运要求精细的力协调。'),
        ('群体仿真与风险建模',
         '多智能体仿真用于模拟和预测复杂社会技术系统的群体行为。基于智能体的建模(ABM)为每个个体定义行为规则，观察宏观模式的涌现。人群疏散仿真模拟紧急情况下大量人群的行为，用于优化建筑物安全设计。交通流仿真用智能体模拟车辆和行人，研究交通拥堵的形成和缓解策略。金融市场仿真的智能体代表不同类型的交易者，研究市场微观结构和系统性风险的传播机制。流行病传播建模的智能体代表个体，模拟疾病在网络化人群中的传播动态。'),
    ],
    '仿真与数字孪生': [
        ('物理引擎与实时仿真',
         '物理引擎是仿真系统的核心组件，负责模拟真实世界的物理规律。刚体动力学模拟物体的运动、碰撞和接触——Bullet、PhysX和MuJoCo是机器人仿真常用的物理引擎。碰撞检测分为宽相(AABB包围盒)和窄相(GJK/EPA算法)两个阶段，前者快速排除不可能碰撞的物体对，后者精确计算碰撞点和穿透深度。柔体动力学使用有限元法或弹簧质点模型模拟可变形物体。流体仿真通过Navier-Stokes方程或其简化形式模拟液体和气体行为。实时仿真要求每帧计算在固定时间预算内完成，精度与速度的权衡是核心挑战。'),
        ('数字孪生与工业应用',
         '数字孪生为物理资产创建实时同步的虚拟副本，实现从监控到预测再到优化的闭环。数字孪生与仿真模型的关键区别在于前者与物理实体保持实时的数据同步。创建数字孪生需要融合多源数据：CAD模型提供几何信息，物联网传感器提供运行状态，历史数据提供退化模式。数字孪生在制造业中实现设备预测性维护和生产过程优化。数字孪生城市通过融合BIM、GIS和IoT数据支持城市规划和管理。医疗数字孪生为患者构建个性化生理模型。'),
        ('虚拟环境与场景生成',
         '虚拟环境为AI训练和测试提供安全可控的沙盒。程序化生成通过算法自动创建大规模多样化的虚拟场景。域随机化在仿真中变化视觉纹理、光照、物理参数等，使在仿真中训练的策略能够迁移到真实世界。Sim2Real迁移通过域适应、域随机化和辅助真实数据微调等方法弥合仿真与真实的鸿沟。Unity ML-Agents和Isaac Sim等平台将游戏引擎的渲染能力与强化学习框架集成。合成数据生成利用仿真器自动生成标注数据。'),
        ('多物理场耦合仿真',
         '多物理场仿真同时求解多个相互耦合的物理过程，对工程设计和科学研究至关重要。流固耦合模拟流体与固体结构的相互作用，是飞机机翼和桥梁风振分析的基础。热力耦合考虑温度变化引起的热应力和热变形。电磁-热-力耦合在电动机和变压器设计中，电磁损耗产热导致热膨胀和机械应力。多物理场仿真的挑战在于不同物理场的时空尺度差异巨大，以及耦合边界条件的高效处理。'),
        ('人在回路仿真',
         '人在回路仿真将人类操作者集成到仿真环境中，用于培训和系统评估。飞行模拟器是人在回路仿真的经典应用，飞行员在真实驾驶舱中操作，系统提供视觉、运动和力反馈。驾驶模拟器用于自动驾驶算法的人机交互测试和新手驾驶员培训。手术模拟器使用力反馈设备模拟手术器械与虚拟组织的交互，用于外科医生培训。军事仿真通过构建逼真的战场环境训练指战员的战术决策能力。人在回路强化学习通过人类反馈引导智能体在复杂环境中的探索和策略学习。'),
    ],
    '遥测与可观测性': [
        ('遥测数据采集与传输',
         '遥测系统从远程设备采集运行数据并传输到中央分析平台。传感器数据采集使用ADC将模拟信号数字化，采样率需满足奈奎斯特准则以保留信号信息。时间同步是分布式遥测的关键，NTP提供毫秒级精度，PTP(IEEE 1588)可达亚微秒级精度。数据压缩(有损和无损)减少传输带宽需求。MQTT和OPC UA是工业遥测的标准传输协议。边缘预处理在传输前进行滤波、聚合和异常检测，减少传输延迟和中心处理压力。'),
        ('时序异常检测',
         '时序异常检测从遥测数据中识别异常模式，是故障预警和系统健康管理的关键。统计方法(3-sigma、Grubbs检验)简单有效但假设数据服从特定分布。基于预测的方法(LSTM、Transformer)学习正常模式然后标记偏离预测置信区间的观测。基于重构的方法(自编码器、VAE)在正常数据上训练，异常数据产生较大重构误差。多维时序异常检测需要考虑维度间的相关性和因果结构。在线异常检测需要在检测精度和延迟之间权衡。'),
        ('分布式追踪',
         '分布式追踪记录请求在微服务架构中的完整调用链路。每个外部请求被分配一个唯一的Trace ID，内部RPC调用生成携带有Span ID和Parent Span ID的Span。Jaeger和Zipkin是广泛使用的开源分布式追踪系统。OpenTelemetry提供了统一的遥测数据采集标准，支持追踪、指标和日志三大信号。基于追踪的延迟分析识别调用链中的瓶颈服务。追踪数据的采样策略(头部采样、尾部采样、自适应采样)平衡了覆盖率和存储成本。'),
        ('可观测性工程',
         '可观测性通过外部输出了解系统内部状态的能力。三大支柱包括：指标提供聚合的数值测量(延迟、吞吐量、错误率、饱和度)，日志记录离散事件的详细上下文，分布式追踪关联跨服务的请求流。SLO/SLI/SLA体系定义了服务的可靠性目标和测量方式。错误预算是SLO与实际表现的差距，指导发布决策和风险承担。告警规则设计需要在信号和噪声之间取得平衡，避免告警疲劳。Grafana和Prometheus是主流的指标可视化和监控工具组合。'),
        ('健康监测与预测性维护',
         '设备健康监测通过分析遥测数据评估设备状态的退化趋势。特征提取从原始振动信号、温度、压力等中提取时域(均值、峰度、峭度)和频域(FFT、包络谱)健康指标。退化建模使用指数模型、威布尔分布或维纳过程描述设备退化轨迹。剩余使用寿命(RUL)预测估计设备距离功能失效的剩余时间。故障诊断识别已发生故障的类型和位置。故障预测在故障发生前预报即将出现的问题。维护策略从被动维修、预防性维护到预测性维护和规范维护不断演进。'),
    ],
    '强化学习与决策智能': [
        ('深度强化学习算法',
         'DQL使用深度神经网络近似Q函数，通过经验回放打破样本相关性，通过目标网络稳定训练过程。策略梯度方法直接优化策略参数，REINFORCE算法使用蒙特卡洛采样估计梯度。Actor-Critic方法结合值函数(Critic)和策略(Actor)的优点，减少策略梯度的方差。PPO通过裁剪目标函数约束策略更新幅度，是目前最广泛使用的RL算法之一。SAC在最大化回报的同时最大化策略熵，鼓励探索和提高鲁棒性。TD3通过双Q学习和目标策略平滑解决DDPG的过度估计问题。'),
        ('探索与利用',
         '探索与利用的权衡是强化学习的核心挑战。epsilon-贪心以固定概率随机探索。UCB选择不确定性与预估价值之和最大的动作。Boltzmann探索根据Q值的softmax分布采样。基于计数的探索给访问次数少的区域额外奖励。基于好奇心的探索使用预测误差作为内在奖励。基于信息论的探索直接最大化关于环境的信息增益。参数空间探索在参数层面而非动作层面加噪，产生更连贯的探索行为。'),
        ('离线强化学习',
         '离线强化学习从固定的历史数据集中学习策略，无需与环境进行在线交互。分布偏移是离线RL的核心挑战——学习的策略可能选择数据集中未见过的动作，导致Q值过度估计。CQL在标准Q学习损失上增加正则项，惩罚数据分布外动作的Q值。BCQ约束学习策略接近行为策略。IQL通过对期望分位数回归避免了OOD动作的查询。决策Transformer将RL问题重新表述为条件序列建模，使用Transformer在状态-动作-回报序列上进行自回归预测。'),
        ('基于模型的强化学习',
         'MBRL首先学习环境动力学模型，然后使用该模型进行规划或策略改进。集成模型通过多个网络的预测方差量化认知不确定性。概率模型(如概率集成、高斯过程)同时预测均值和不确定度。轨迹采样从模型中生成模拟轨迹用于策略训练，Dyna架构在模型学习和策略改进之间交替。模型预测控制(MPC)使用模型在有限时域内规划，执行第一步后重新规划。世界模型通过学习环境的紧凑潜在表示，在其中进行高效的模拟和规划。Dreamer在潜在空间中同时学习世界模型和策略。'),
        ('决策智能应用',
         'RL在工业中已取得了令人瞩目的实际成果。数据中心冷却使用RL将冷却能耗降低40%。推荐系统使用RL优化长期用户参与度而非单次点击率。供应链优化使用RL处理库存管理和运输调度中的序列决策。量化交易使用RL从市场微观结构中学习交易策略。芯片设计使用RL进行宏布局规划，在数小时内完成人类工程师数周的工作。游戏AI中AlphaZero从零开始通过自对弈学习，达到了超越人类和传统引擎的棋力水平。'),
    ],
    '仪表盘与可视化': [
        ('实时数据仪表盘',
         '仪表盘以直观的图形界面展示系统关键指标。仪表盘设计的第一原则是信息层次——最重要的指标放在最显眼的位置，按F型或Z型扫描模式布局。实时数据的流式处理使用WebSocket或Server-Sent Events推送更新到前端。时间窗口聚合计算滑动窗口内的统计量。阈值告警在指标越过预设阈值时触发视觉提示。健康评分将多维指标融合为单一的综合评分，方便快速判断系统整体状态。深色主题和合适的配色方案减少长时间监控的视觉疲劳。'),
        ('交互式可视化',
         '交互式可视化通过缩放、过滤、联动和钻取支持多维数据探索。刷选联动使用户在一个视图中选择数据子集时，其他视图自动同步过滤。滚轮缩放和拖拽平移是空间数据探索的基本交互。钻取操作从概览到细节逐级深入。散点图矩阵展示多维数据的两两关系。平行坐标图将高维数据映射到平行排列的坐标轴，用于多维模式识别。大屏可视化专为公共展示设计，需要自动刷新、动画过渡和远距离可读性。'),
        ('地理空间可视化',
         '地理空间数据可视化将数据与地理位置关联展示。热力图通过颜色密度表示点数据的空间聚集程度。等值区域图按行政区划着色表示区域统计数据。流向图使用线条或箭头表示物体或人流在不同位置间的移动。时空立方体将时间作为第三维展示轨迹和移动模式。栅格图层叠加在卫星影像和多源遥感数据上。Web GIS(如Leaflet、Mapbox GL JS)提供交互式地图基础组件。GPS轨迹可视化展示移动物体的路径、速度和停留点。'),
        ('可视化叙事',
         '数据叙事结合了数据可视化和故事叙述的技巧。好的数据故事有清晰的结构：情境、问题、分析、洞察和行动建议。注释和标注引导读者关注关键发现。动画和过渡在时间维度上展示变化过程。探索性可视化与解释性可视化服务于不同目的——前者帮助分析师发现未知模式，后者向受众传达已知结论。小倍数通过并排排列将不同条件下的图表做对比。可视化仪表盘与报告的区别在于前者支持交互式探索而后者是静态叙述。'),
        ('监控告警系统',
         '告警系统是监控数据到运维行动的桥梁。告警规则分为阈值告警(固定阈值)、趋势告警(变化率超过阈值)、同环比告警(与历史同期比较)和智能告警(机器学习检测)。告警分级(P0-P4)按严重程度从紧急到信息进行划分。告警抑制避免依赖故障导致的告警风暴。告警升级在未及时响应时将低级别告警自动升级。告警通知渠道(短信、电话、即时通讯、邮件)根据严重级别和时段选择。值班轮换和告警路由确保告警总是能找到对应的责任人。'),
    ],
}

# ═══════════════════════════════════════════
# English extra domains
# ═══════════════════════════════════════════

EN_EXTRA_DOMAINS = {
    'Computer Vision & Visual Intelligence': [
        ('Image Classification and Object Detection',
         'Computer vision enables machines to perceive and understand visual information. Image classification has evolved from LeNet through ResNet to Vision Transformers — from shallow CNNs to deep residual networks and pure attention mechanisms. Object detection answers the question of what is where — from two-stage R-CNN variants to single-stage YOLO and SSD, to Transformer-based DETR, detection speed and accuracy continuously improve. Feature Pyramid Networks (FPN) address the challenge of small object detection through multi-scale feature fusion. Non-Maximum Suppression (NMS) eliminates duplicate bounding boxes. Mean Average Precision (mAP) is the standard metric for detector performance. In practice, object detection is foundational for autonomous driving perception, surveillance, and industrial quality inspection.'),
        ('Image Segmentation and Scene Understanding',
         'Semantic segmentation assigns class labels to every pixel, forming the backbone of autonomous driving and medical image analysis. Fully Convolutional Networks (FCN) pioneered end-to-end pixel-level prediction. U-Net, with its encoder-decoder architecture and skip connections, excels in medical image segmentation and has become the de facto standard for biomedical image analysis. DeepLab series uses atrous convolutions to enlarge receptive fields while preserving resolution. Instance segmentation further distinguishes individual instances of the same class — Mask R-CNN achieves this by adding a segmentation branch to Faster R-CNN. Panoptic segmentation unifies semantic and instance segmentation, requiring every pixel to have both a semantic class and an instance ID.'),
        ('3D Vision and Depth Estimation',
         'Recovering 3D information from 2D images is a fundamental challenge in computer vision. Stereo vision simulates human binocular disparity by matching corresponding points between left and right images to compute depth. Structured light projects known patterns and analyzes deformation to reconstruct surface geometry, widely used in industrial precision measurement. Time-of-Flight (ToF) cameras directly measure depth by timing light pulse round trips. Monocular depth estimation uses deep learning to infer depth from a single image — less accurate than stereo or structured light but cheaper and easier to deploy. Point cloud processing is a major branch of 3D vision; PointNet pioneered deep learning directly on unordered point sets. 3D object detection is critical for autonomous driving and robotic grasping.'),
        ('Visual SLAM and Localization',
         'Visual SLAM achieves simultaneous localization and mapping using camera sensors. ORB-SLAM is a landmark system that uses ORB features for tracking, mapping, and loop closure. Direct SLAM methods skip feature extraction and directly use pixel intensities for pose estimation, offering advantages in low-texture environments. Visual-Inertial Odometry (VIO) fuses camera and IMU data, leveraging IMU high-frequency motion information to compensate for visual shortcomings during rapid movement. Loop closure detection eliminates accumulated drift and is key to global consistency; Bag-of-Words (BoW) based scene recognition is a common approach. Visual relocalization estimates 6-DOF camera pose through image retrieval or scene coordinate regression.'),
        ('Video Understanding and Action Recognition',
         'Video understanding extends image analysis into the temporal dimension, dealing with spatiotemporal data. 3D Convolutional Networks (C3D, I3D) simultaneously extract spatial and temporal features through 3D convolution kernels. Two-stream networks process RGB frames and optical flow separately, fusing spatial appearance with motion information. Temporal Segment Networks (TSN) achieve efficient action recognition by sparsely sampling keyframes from long videos. The SlowFast network dual-pathway design mimics the division of labor between M-cells and P-cells in biological vision. Temporal action localization identifies both action category and temporal boundaries, widely applied in surveillance and sports analysis.'),
    ],
    'Speech & Audio Intelligence': [
        ('Automatic Speech Recognition',
         'Automatic Speech Recognition (ASR) converts audio signals into text, the first step in human-machine voice interaction. Traditional ASR systems consist of three components: acoustic model, language model, and pronunciation dictionary, with GMM-HMM as the classical framework. End-to-end ASR (DeepSpeech, LAS) directly learns the mapping from audio to text, dramatically simplifying system architecture. CTC loss solves the alignment problem of mismatched input-output sequence lengths. Transformer/Conformer architectures excel in ASR — Conformer enhances self-attention with convolutions to better capture both local and global features. Multilingual ASR models like Whisper demonstrate strong cross-lingual generalization. Streaming ASR requires outputting partial recognition results with low latency.'),
        ('Text-to-Speech Synthesis',
         'Text-to-Speech (TTS) converts written text into natural, fluent speech. Parametric TTS predicts acoustic parameters (fundamental frequency, spectrum) and then synthesizes waveforms through a vocoder. End-to-end TTS (Tacotron, FastSpeech) directly generates acoustic features from character sequences. FastSpeech addresses Tacotron repetition and omission issues through non-autoregressive architecture while significantly improving inference speed. Neural vocoders (WaveNet, WaveGlow, HiFi-GAN) generate high-quality waveforms from acoustic features. Multi-speaker TTS enables a single model to synthesize multiple voices through speaker embeddings. Zero-shot voice cloning learns target speaker timbre from just seconds of audio. Emotional TTS controls the emotional expression of synthesized speech.'),
        ('Sound Event Detection',
         'Sound Event Detection (SED) identifies sound events and their temporal boundaries in audio, with wide applications in smart homes, industrial monitoring, and urban noise management. Traditional methods use MFCC features with GMM or HMM for classification. Deep learning approaches employ CNN or CRNN on mel-spectrograms for event detection. Attention mechanisms help models focus on relevant time-frequency regions. Weakly-supervised SED requires only audio-level labels rather than precise temporal annotations, significantly reducing data labeling costs. Few-shot sound detection learns universal representations of sound categories, particularly valuable for long-tail distributions of rare sound events. Multi-channel sound source localization estimates source direction using microphone arrays.'),
        ('Speaker Recognition and Voice Biometrics',
         'Speaker recognition confirms identity through voiceprint features. Speaker verification determines whether a voiceprint matches the claimed identity. Speaker identification identifies a speaker from multiple enrolled voiceprints. i-vector is a classic statistical voiceprint method, mapping variable-length utterances to fixed-dimension identity vectors through factor analysis. x-vector uses deep neural networks to extract more discriminative voiceprint embeddings. ECAPA-TDNN achieves state-of-the-art performance through channel attention and statistical pooling mechanisms. Anti-spoofing detection distinguishes genuine speech from replay attacks, speech synthesis, and voice conversion.'),
    ],
    'Knowledge Graph & Reasoning': [
        ('Knowledge Graph Construction',
         'Knowledge graphs organize knowledge as entity-relation-entity triple structures, serving as foundational infrastructure for machines to understand and reason about world knowledge. Knowledge extraction includes Named Entity Recognition from text, relation extraction determining semantic relationships between entities, and entity linking connecting text mentions to correct knowledge base entities. Distant supervision automatically generates training data by aligning existing knowledge bases with text — noisy but extremely cost-effective. Open Information Extraction (OIE) is not restricted to predefined relations, directly extracting arbitrary relational triples from text. Knowledge fusion addresses alignment and merging across different sources. Temporal knowledge graphs track how knowledge changes over time.'),
        ('Graph Neural Networks and Reasoning',
         'Graph Neural Networks (GNNs) perform representation learning and reasoning on knowledge graphs. GCN updates node representations through neighbor aggregation — each layer aggregates information from one-hop neighbors. GAT introduces attention mechanisms to assign different importance weights to different neighbors. R-GCN learns different transformation matrices for different relation types, suitable for multi-relational knowledge graphs. Knowledge graph embedding methods (TransE, RotatE, ComplEx) map entities and relations to low-dimensional vector spaces, performing link prediction through vector operations. Neural-symbolic reasoning combining logical rules with GNNs offers both data-driven learning and logic-driven interpretability.'),
        ('Knowledge-Based Question Answering',
         'Knowledge Base Question Answering (KBQA) uses natural language questions to retrieve and reason about answers from knowledge graphs. Semantic parsing methods convert natural language questions into structured queries (e.g., SPARQL) and execute them on the knowledge graph. Information retrieval methods directly retrieve candidate answers from the knowledge graph and select the best answer through ranking models. Multi-hop QA requires traversing multiple intermediate entities and relational reasoning chains to reach the final answer. Combining large language models with knowledge graphs through Retrieval-Augmented Generation (RAG) incorporates external knowledge into the generation process, improving factual accuracy.'),
        ('Commonsense Reasoning',
         'Commonsense reasoning represents a critical threshold for AI approaching human intelligence. ConceptNet is one of the largest commonsense knowledge bases, containing everyday concept associations across dozens of relation types. The ATOMIC knowledge base specifically records social commonsense — about event causes, effects, and mental states. The COMET model, trained on commonsense knowledge bases using GPT, can automatically generate commonsense inferences about event causes and effects. Physical commonsense reasoning requires understanding fundamental physical world laws. Social commonsense reasoning involves understanding human intentions, emotions, and social norms. The primary challenge of commonsense reasoning is that common sense is implicit, context-dependent, and rarely explicitly written down.'),
    ],
    'Multi-Agent & Swarm Intelligence': [
        ('Multi-Agent Reinforcement Learning',
         'Multi-Agent Reinforcement Learning (MARL) studies how multiple agents simultaneously learn and interact in a shared environment. Each agent policy changes alter the environment for other agents, violating Markov and stationarity assumptions in single-agent RL. Centralized Training with Decentralized Execution (CTDE) is the dominant paradigm — global information guides learning during training, but execution relies only on local observations. MADDPG learns centralized Q-functions for each agent, performing well in mixed cooperative-competitive settings. QMIX decomposes joint Q-values through monotonic mixing networks, ensuring consistency between centralized and decentralized policies. MAPPO extends PPO to multi-agent settings and is popular for its simplicity and stability.'),
        ('Swarm Intelligence and Emergent Behavior',
         'Swarm intelligence studies how large numbers of simple individuals, through local interactions, spontaneously generate complex collective behaviors. Ant colony algorithms simulate how ants discover shortest paths through pheromone communication, used for optimization and scheduling problems. Particle Swarm Optimization (PSO) simulates bird flock foraging behavior — each particle updates its search direction based on personal and swarm best positions. The Boids model simulates flocking behavior in birds and fish schools through three simple rules: separation, alignment, and cohesion. Emergent behavior arises spontaneously without centralized control — the collective capabilities exceed the sum of any individual abilities.'),
        ('Multi-Robot Collaboration',
         'Multi-robot systems accomplish tasks through collaboration that would be difficult or impossible for a single robot. Task allocation determines which robot performs which subtask — market-based mechanisms assign tasks to the lowest-bidding robot through auctions. Formation control maintains desired spatial configurations among multiple robots; the leader-follower approach is simple yet effective. Multi-robot SLAM accelerates exploration and improves mapping accuracy by sharing map and pose information. Distributed target tracking improves tracking accuracy by fusing observations from multiple robots with different perspectives. Cooperative manipulation tasks such as multi-robot transport require fine force coordination; impedance control and master-slave control are common approaches.'),
    ],
    'Simulation & Digital Twin': [
        ('Physics Engines and Real-Time Simulation',
         'Physics engines are core components of simulation systems, responsible for simulating real-world physical laws. Rigid body dynamics simulate object motion, collision, and contact — Bullet, PhysX, and MuJoCo are commonly used physics engines in robotics simulation. Collision detection consists of broad phase (AABB bounding boxes) and narrow phase (GJK/EPA algorithms) — the former quickly excludes impossible collision pairs, the latter precisely computes collision points and penetration depth. Soft body dynamics use finite element methods or mass-spring models to simulate deformable objects. Fluid simulation solves Navier-Stokes equations or their simplified forms to model liquid and gas behavior. The accuracy-speed tradeoff is the core challenge of real-time simulation.'),
        ('Digital Twin and Industrial Applications',
         'Digital twins create real-time synchronized virtual replicas of physical assets, enabling a closed loop from monitoring to prediction to optimization. The key distinction between digital twins and simulation models is that the former maintains real-time data synchronization with the physical entity. Creating a digital twin requires fusing multi-source data: CAD models provide geometric information, IoT sensors provide operational status, and historical data provides degradation patterns. In manufacturing, digital twins enable predictive equipment maintenance and production process optimization. Digital twin cities support urban planning and management by integrating BIM, GIS, and IoT data. Medical digital twins construct personalized physiological models for patients.'),
        ('Sim-to-Real Transfer',
         'Sim-to-Real transfer bridges the simulation-reality gap. Domain randomization varies visual textures, lighting, and physical parameters in simulation, enabling policies trained in simulation to transfer to the real world. System identification calibrates simulator parameters to match real-world observations. Domain adaptation aligns feature representations between simulated and real data. Platforms like Unity ML-Agents and NVIDIA Isaac Sim integrate game engine rendering capabilities with reinforcement learning frameworks, accelerating sim-to-real research and applications.'),
    ],
    'Telemetry & Observability': [
        ('Time Series Anomaly Detection',
         'Time series anomaly detection identifies abnormal patterns in telemetry data, key to fault early warning and system health management. Statistical methods (3-sigma, Grubbs test) are simple and effective but assume data follows specific distributions. Prediction-based methods (LSTM, Transformer) learn normal patterns and flag observations that deviate from prediction confidence intervals. Reconstruction-based methods (Autoencoder, VAE) are trained on normal data; anomalous data produces larger reconstruction errors. Multivariate time series anomaly detection must account for inter-dimensional correlations and causal structures. Online anomaly detection requires balancing detection accuracy and latency, crucial for industrial real-time monitoring.'),
        ('Distributed Tracing',
         'Distributed tracing records the complete call chain of requests through microservice architectures. Each external request is assigned a unique Trace ID, and internal RPC calls generate Spans carrying Span ID and Parent Span ID. Jaeger and Zipkin are widely used open-source distributed tracing systems. OpenTelemetry provides a unified standard for telemetry data collection, supporting the three pillars: traces, metrics, and logs. Trace-based latency analysis identifies bottleneck services in call chains. Trace sampling strategies (head sampling, tail sampling, adaptive sampling) balance coverage and storage cost.'),
        ('Observability Engineering',
         'Observability is the ability to understand system internal state through external outputs. The three pillars include: metrics providing aggregated numerical measurements (latency, throughput, error rate, saturation), logs recording detailed context of discrete events, and distributed tracing correlating request flows across services. The SLO/SLI/SLA framework defines service reliability targets and measurement methods. Error budgets represent the gap between SLO and actual performance, guiding release decisions and risk-taking. Alert rule design must balance signal and noise to avoid alert fatigue. Grafana and Prometheus are the mainstream combination for metrics visualization and monitoring.'),
        ('Predictive Maintenance',
         'Equipment health monitoring assesses degradation trends by analyzing telemetry data. Feature extraction derives time-domain (mean, kurtosis, crest factor) and frequency-domain (FFT, envelope spectrum) health indicators from raw vibration signals and operational parameters. Degradation modeling uses exponential models, Weibull distributions, or Wiener processes to describe equipment degradation trajectories. Remaining Useful Life (RUL) prediction estimates the remaining time until functional failure, forming the core of predictive maintenance. Fault diagnosis identifies the type and location of faults. Maintenance strategies have evolved from reactive and preventive maintenance to predictive and prescriptive maintenance.'),
    ],
    'Reinforcement Learning & Decision Intelligence': [
        ('Deep RL Algorithms',
         'DQL uses deep neural networks to approximate Q-functions, breaking sample correlations through experience replay and stabilizing training through target networks. Policy gradient methods directly optimize policy parameters; the REINFORCE algorithm estimates gradients using Monte Carlo sampling. Actor-Critic methods combine the advantages of value functions (Critic) and policies (Actor), reducing policy gradient variance. PPO constrains policy update magnitude by clipping the objective function, making it one of the most widely used RL algorithms. SAC maximizes both expected return and policy entropy, encouraging exploration and improving robustness. TD3 addresses DDPG overestimation through double Q-learning and target policy smoothing.'),
        ('Offline Reinforcement Learning',
         'Offline reinforcement learning learns policies from fixed historical datasets without online environment interaction. Distribution shift is the core challenge — the learned policy may select actions unseen in the dataset, causing Q-value overestimation. CQL adds a regularization term to the standard Q-learning loss, penalizing Q-values for out-of-distribution actions. BCQ constrains the learned policy to be close to the behavioral policy. IQL avoids querying OOD actions through expectile regression. Decision Transformers reframe the RL problem as conditional sequence modeling, using Transformers for autoregressive prediction on state-action-return sequences.'),
        ('Model-Based RL',
         'MBRL first learns an environment dynamics model, then uses that model for planning or policy improvement. Ensemble models quantify epistemic uncertainty through prediction variance across multiple networks. Probabilistic models simultaneously predict means and uncertainties. Trajectory sampling generates simulated trajectories from the model for policy training; the Dyna architecture alternates between model learning and policy improvement. Model Predictive Control (MPC) uses the model to plan over a finite horizon, replanning after executing the first step. World models learn compact latent representations of the environment for efficient simulation and planning.'),
        ('RL Industrial Applications',
         'RL has achieved impressive real-world industrial results. Data center cooling using RL reduced cooling energy consumption by 40%. Recommendation systems use RL to optimize long-term user engagement rather than single-click rates. Supply chain optimization uses RL for sequential decision-making in inventory management and transportation scheduling. Quantitative trading uses RL to learn trading strategies from market microstructure. Chip design uses RL for macro placement, completing in hours what takes human engineers weeks. In game AI, AlphaZero learns from scratch through self-play, achieving superhuman performance.'),
    ],
    'Dashboard & Visualization': [
        ('Real-Time Data Dashboards',
         'Dashboards display system key metrics through intuitive graphical interfaces. The primary principle of dashboard design is information hierarchy — the most important metrics go in the most prominent positions, following F-pattern or Z-pattern scanning layouts. Real-time data streaming uses WebSocket or Server-Sent Events to push updates to the frontend. Time window aggregation computes statistics over sliding windows. Threshold alerts trigger visual cues when metrics cross preset thresholds. Health scores fuse multi-dimensional metrics into a single composite score for quick assessment. Dark themes and appropriate color schemes reduce visual fatigue during long monitoring sessions.'),
        ('Interactive Visualization',
         'Interactive visualization enables multi-dimensional data exploration through zooming, filtering, brushing and linking, and drill-down. Brushing and linking synchronizes filtering across views — selecting a data subset in one view automatically filters other views. Scroll-wheel zooming and drag-panning are basic interactions for spatial data exploration. Drill-down operations progressively move from overview to detail. Scatter plot matrices display pairwise relationships in multi-dimensional data. Parallel coordinates map high-dimensional data to parallel axes for multi-dimensional pattern recognition. Large-screen visualization is designed for public displays, requiring auto-refresh, animated transitions, and long-distance readability.'),
        ('Geospatial Visualization',
         'Geospatial data visualization associates data with geographic locations. Heatmaps use color density to represent the degree of spatial clustering of point data. Choropleth maps color administrative regions to represent regional statistical data. Flow maps use lines or arrows to show object or people movement between different locations. Space-time cubes use time as a third dimension to display trajectories and movement patterns. Raster layer overlays combine satellite imagery with multi-source remote sensing data. Web GIS solutions like Leaflet and Mapbox GL JS provide interactive map base components. GPS trajectory visualization displays paths, speeds, and stopping points of moving objects.'),
        ('Monitoring and Alerting Systems',
         'Alerting systems bridge monitoring data to operational action. Alert rules are categorized as threshold alerts (fixed thresholds), trend alerts (rate of change exceeding thresholds), period-over-period alerts (comparison with historical data), and intelligent alerts (machine learning detection). Alert severity levels (P0-P4) range from emergency to informational based on criticality. Alert suppression prevents alert storms caused by dependency failures. Alert escalation automatically escalates lower-level alerts when not promptly responded to. Alert notification channels (SMS, phone call, instant messaging, email) are selected based on severity and time of day. On-call rotations and alert routing ensure alerts always reach responsible personnel.'),
    ],
}

# ═══════════════════════════════════════════
# Simple templates
# ═══════════════════════════════════════════

TEMPLATES_ZH = [
    '\n## {topic}\n\n{body}\n\n> {insight}\n',
    '\n### {topic}\n\n{body}\n\n*要点: {insight}*\n',
    '\n# 学习: {topic}\n\n{body}\n\n> {insight}\n',
    '\n### 问答: {topic}\n\n**问**: 请详细解释{topic}的核心概念和应用?\n\n**答**: {body}\n\n> {insight}\n',
    '\n## {topic} (深入分析)\n\n{body}\n\n### 关键总结\n\n{insight}\n',
]

TEMPLATES_EN = [
    '\n## {topic}\n\n{body}\n\n> {insight}\n',
    '\n### {topic}\n\n{body}\n\n*Key Point: {insight}*\n',
    '\n# Study Guide: {topic}\n\n{body}\n\n> {insight}\n',
    '\n### Q&A: {topic}\n\n**Q**: Explain the core concepts and applications of {topic}.\n\n**A**: {body}\n\n> {insight}\n',
    '\n## {topic} (Deep Dive)\n\n{body}\n\n### Key Takeaway\n\n{insight}\n',
]

INSIGHTS_ZH = [
    '真正的理解不在于记住了多少概念，而在于能够在陌生的问题中识别出熟悉的结构。',
    '理论和实践之间的鸿沟往往需要通过大量的实验和迭代来弥合。',
    '简单的方法在实际中往往比复杂的方法更有效，关键在于正确的问题建模。',
    '数据和模型同等重要，高质量的数据往往比更复杂的模型带来更大的提升。',
    '系统的鲁棒性和泛化能力比在标准测试集上的性能指标更重要。',
    '技术方案的选择应该由问题本身驱动，而非由最新的研究趋势决定。',
    '好的工程实践很多时候比算法的微小改进更有价值。',
    '在大多数真实场景中，可解释性和可靠性比原始性能更重要。',
]

INSIGHTS_EN = [
    'True understanding lies not in how many concepts you have memorized, but in the ability to recognize familiar structures in unfamiliar problems.',
    'The gap between theory and practice often requires extensive experimentation and iteration to bridge.',
    'Simple methods often outperform complex ones in practice — the key lies in correct problem formulation.',
    'Data and models are equally important; high-quality data often brings greater improvements than more complex models.',
    'System robustness and generalization matter more than performance metrics on standard benchmarks.',
    'The choice of technical approach should be driven by the problem itself, not by the latest research trends.',
    'Good engineering practices are often more valuable than minor algorithmic improvements.',
    'In most real-world scenarios, interpretability and reliability matter more than raw performance.',
]


def generate_entries(lang, domains_dict, templates, insights, repeat_count):
    """Generate entries for all domains"""
    all_entries = []
    total_chars = 0
    for domain, topics in domains_dict.items():
        all_entries.append(f'\n\n# {domain}\n')
        for topic_name, topic_body in topics:
            for _ in range(repeat_count):
                tpl = random.choice(templates)
                ins = random.choice(insights)
                entry = tpl.format(topic=topic_name, body=topic_body, insight=ins)
                all_entries.append(entry)
                total_chars += len(entry)
        print(f'  {domain}: {len(topics)} topics x {repeat_count}')
    return '\n'.join(all_entries), total_chars


def main():
    print('=' * 50)
    print('  补充支柱语料生成器')
    print('  视觉/语音/KG/群体/仿真/遥测/RL/仪表盘')
    print('=' * 50)

    existing_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024 if os.path.exists(CORPUS_PATH) else 0
    print(f'现有语料: {existing_mb:.0f}MB')

    # Write header
    header = '\n\n' + '=' * 60 + '\n# 全支柱补充语料 (视觉/语音/KG/群体/仿真/遥测/RL/仪表盘)\n' + '=' * 60 + '\n'
    with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
        f.write(header)

    total_new = 0

    # Chinese
    print('\n[中文补充]')
    for seed in [42, 123, 456, 789, 1011]:
        random.seed(seed)
        zh_text, zh_chars = generate_entries('zh', ZH_EXTRA_DOMAINS, TEMPLATES_ZH, INSIGHTS_ZH, 500)
        with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
            f.write(zh_text)
        total_new += zh_chars
        cur_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024
        print(f'  Seed {seed}: +{zh_chars/1024/1024:.1f}MB => {cur_mb:.0f}MB')

    # English
    print('\n[English Extra]')
    for seed in [42, 123, 456, 789, 1011]:
        random.seed(seed)
        en_text, en_chars = generate_entries('en', EN_EXTRA_DOMAINS, TEMPLATES_EN, INSIGHTS_EN, 500)
        with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
            f.write(en_text)
        total_new += en_chars
        cur_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024
        print(f'  Seed {seed}: +{en_chars/1024/1024:.1f}MB => {cur_mb:.0f}MB')

    final_mb = os.path.getsize(CORPUS_PATH) / 1024 / 1024
    print(f'\n===== 完成 =====')
    print(f'新增: {total_new/1024/1024:.1f}MB')
    print(f'总量: {final_mb:.0f}MB')


if __name__ == '__main__':
    main()
