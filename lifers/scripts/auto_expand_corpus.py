#!/usr/bin/env python3
"""
Auto-expanding training corpus: monitors loss plateau and adds new domain
content when the model stops improving, then triggers vocab rebuild.

Run alongside escalate loop:
  python3 scripts/auto_expand_corpus.py

Env:
  LIFERS_AUTO_EXPAND_PLATEAU_N   consecutive checkpoints for plateau (default 4)
  LIFERS_AUTO_EXPAND_MIN_IMPROV   min loss improvement to reset plateau (default 0.02)
  LIFERS_AUTO_EXPAND_KB_PER_ADD   KB to add per expansion (default 30)
  LIFERS_AUTO_EXPAND_MAX_ADDITIONS  max total expansions (default 100)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Domain content generators — each returns ~5-20KB of training text
# ---------------------------------------------------------------------------

_DOMAIN_QUEUE = [
    # Round 1: STEM depth
    ("mathematics_proofs", lambda: _gen_math()),
    ("physics_advanced", lambda: _gen_physics()),
    ("chemistry_organic", lambda: _gen_chemistry()),
    ("biology_molecular", lambda: _gen_biology()),
    # Round 2: CS & Engineering
    ("cs_algorithms", lambda: _gen_algorithms()),
    ("cs_systems", lambda: _gen_systems()),
    ("software_engineering", lambda: _gen_software()),
    ("networking_protocols", lambda: _gen_networking()),
    # Round 3: Languages & Humanities
    ("chinese_classics", lambda: _gen_chinese()),
    ("philosophy_ethics", lambda: _gen_philosophy()),
    ("world_history", lambda: _gen_history()),
    ("economics_finance", lambda: _gen_economics()),
    # Round 4: Applied skills
    ("coding_practice", lambda: _gen_coding()),
    ("data_science", lambda: _gen_data_science()),
    ("security_crypto", lambda: _gen_security()),
    ("robotics_embedded", lambda: _gen_robotics()),
    # Round 5: General knowledge
    ("medicine_anatomy", lambda: _gen_medicine()),
    ("law_contracts", lambda: _gen_law()),
    ("architecture_design", lambda: _gen_architecture()),
    ("music_theory", lambda: _gen_music()),
    # Round 6: English content
    ("en_cs_ai", lambda: _gen_en_cs_ai()),
    ("en_programming", lambda: _gen_en_programming()),
    ("en_mathematics", lambda: _gen_en_mathematics()),
    ("en_sciences", lambda: _gen_en_sciences()),
    ("en_humanities", lambda: _gen_en_humanities()),
    ("en_technology", lambda: _gen_en_technology()),
    ("en_writing", lambda: _gen_en_writing()),
    ("en_business", lambda: _gen_en_business()),
]

_domain_index = 0


def _next_domain() -> tuple:
    global _domain_index
    item = _DOMAIN_QUEUE[_domain_index % len(_DOMAIN_QUEUE)]
    _domain_index += 1
    return item


# ---------------------------------------------------------------------------
# Content generators
# ---------------------------------------------------------------------------

def _gen_math() -> str:
    return """
# 高等数学进阶

## 泛函分析
Banach空间是完备的赋范向量空间。Hilbert空间是内积诱导的完备空间。有界线性算子构成Banach代数。谱理论将线性代数特征值推广到无限维——紧算子的谱由特征值(有限重数)和可能的极限点0组成。Fredholm二择一定理：对于紧算子K，要么(I-K)x=y对每个y有唯一解，要么齐次方程(I-K)x=0有非零解。

Hahn-Banach定理保证了线性泛函的延拓——给定子空间上的有界线性泛函，可以保范延拓到全空间。这一定理是非线性分析、凸分析和最优化的基础。开映射定理：Banach空间之间的满射有界线性算子将开集映射为开集——其推论是Banach同构定理和闭图像定理。

## 微分几何
流形是局部同胚于欧氏空间的拓扑空间配备光滑结构。切空间是流形上某点的所有方向导数构成的向量空间。黎曼度量给每个切空间赋予内积使其成为黎曼流形。测地线是局部最短曲线满足测地线方程——∇_γ'γ'=0。曲率张量R(X,Y)Z=∇_X∇_Y Z-∇_Y∇_X Z-∇_{[X,Y]}Z衡量平行移动与路径的相关性。

Gauss-Bonnet定理将曲面的总高斯曲率(几何量)与Euler示性数(拓扑量)联系起来——∫_M K dA=2πχ(M)。这是几何与拓扑深刻联系的第一个例子。广义相对论中Einstein场方程的几何背景——时空是4维Lorentz流形曲率与能量动量张量相关。

## 概率论极限定理
强大数定律：独立同分布随机变量序列的样本均值几乎必然收敛到期望——P(lim X̄_n=μ)=1。中心极限定理在Lindeberg条件下的推广——不要求同分布只要求Lindeberg条件。Berry-Esseen定理给出CLT收敛速度的上界——|F_n(x)-Φ(x)|≤Cρ/(σ³√n)其中ρ=E|X-μ|³。

大偏差理论估计罕见事件概率的指数衰减速率——Cramér定理给出独立同分布序列经验均值的速率函数I(x)=sup_θ(θx-log E[e^{θX}])。Varadhan积分引理将Laplace渐近与大偏差速率函数关联。大偏差在统计物理、信息论和金融风险管理中有重要应用。

## 优化理论
凸优化的对偶理论：Lagrange对偶将约束优化转为无约束问题——Lagrangian L(x,λ,ν)=f₀(x)+Σλ_i f_i(x)+Σν_i h_i(x)。对偶函数g(λ,ν)=inf_x L(x,λ,ν)是原问题最优值的下界。Slater条件保证强对偶——对偶间隙为零。KKT条件是凸问题全局最优的充要条件。

内点法(Interior Point Method)通过障碍函数将约束优化转为无约束序列——log障碍项μΣlog(-f_i(x))参数μ→0。原对偶内点法同时跟踪原始和对偶变量沿中心路径(central path)向最优解收敛。现代优化求解器(CPLEX、Gurobi)大量使用原对偶内点法和单纯形法的混合策略。
"""


def _gen_physics() -> str:
    return """
# 物理学进阶

## 量子场论基础
场的量子化将经典场提升为作用于Fock空间的算符。标量场φ(x)的Klein-Gordon方程(∂_μ∂^μ+m²)φ=0来源于Lagrangian密度L=½(∂_μφ∂^μφ-m²φ²)。正则对易关系[φ(x),π(y)]=iδ³(x-y)在等时面上定义了量子化规则。产生和湮灭算符a^†_p和a_p在动量空间对角化Hamiltonian——a^†_p|0⟩是动量为p的单粒子态。

费米子场ψ(x)满足Dirac方程(iγ^μ∂_μ-m)ψ=0。等时反对易关系{ψ_α(x),ψ^†_β(y)}=δ_αβδ³(x-y)实现Pauli不相容原理。路径积分形式化将跃迁振幅表示为经典作用量的指数加权求和——Z=∫Dφ e^{iS[φ]/ħ}。微扰展开产生Feynman图——每条传播子对应Green函数的逆，每个顶点对应相互作用的耦合常数。

重整化处理量子场论中的无穷大。裸参数(质量、耦合常数)吸收发散使物理可观测量有限。重整化群方程描述耦合常数随能标的变化——在QCD中渐近自由意味着高能下耦合变弱。有效场论将对高能物理的效应参数化为低能理论中的高维算符——它是Wilson重整化精神的体现。

## 凝聚态物理
Bloch定理：周期势V(r+R)=V(r)中的电子波函数为ψ_{nk}(r)=e^{ik·r}u_{nk}(r)，其中u_{nk}有晶格周期性。能带结构由布里渊区中每个k的离散能级E_n(k)组成。导体、半导体和绝缘体的区别在于费米能级与能带的相对位置——部分填充带导电、满带不导电。

拓扑绝缘体的体态绝缘但表面态导电——由体能带的非平凡拓扑(Z₂不变量)保证。量子Hall效应的电导量子化为e²/h的整数倍源自Landau能级和拓扑——Thouless-Kohmoto-Nightingale-denNijs(TKNN)将其解释为第一Chern数。Majorana零模是非阿贝尔任意子——拓扑量子计算的基础可能依赖于Majorana零模的编织(Braiding)。

## 天体物理
恒星结构方程：质量连续dm/dr=4πr²ρ、流体静力平衡dP/dr=-Gmρ/r²、能量产生dL/dm=ε(核反应能量产率)、能量传输(辐射或对流)决定温度梯度dT/dr。在辐射区能量由光子扩散传递不透明度κ决定dT/dr=-(3κρL)/(16πacr²T³)。在太阳内部pp链将四个质子融合为He-4释放26.7MeV——CNO循环在大质量恒星中主导能量产生。

中子星由中子简并压支撑最大质量约2-3太阳质量(Tolman-Oppenheimer-Volkoff极限)。脉冲星是旋转的中子星发射如灯塔般的射电束——毫秒脉冲星的计时精度超过原子钟验证了引力波的存在(Hulse-Taylor双星)。黑洞的热力学——Bekenstein-Hawking熵S=A/4Għ与视界面积成正比，Hawking温度T=ħc³/8πGMk_B与质量成反比。
"""


def _gen_chemistry() -> str:
    return """
# 化学进阶

## 有机反应机理
周环反应的Woodward-Hoffmann规则：反应在基态下允许或禁阻取决于参与轨道的对称性。对于[4+2]环加成(Diels-Alder反应)：二烯的HOMO与亲二烯体的LUMO对称性匹配在基态下允许——同面-同面加成。对于[2+2]环加成：基态下同面-同面加热禁阻(对称性不匹配)但光化学允许。前线轨道理论(Fukui)用HOMO-LUMO相互作用统一理解反应性。

交叉偶联反应的催化循环：Suzuki偶联——Pd(0)对芳基卤化物进行氧化加成得到Pd(II)络合物、有机硼酸在碱活化下转金属化到Pd上、还原消除释放联芳基产物再生Pd(0)。Buchwald-Hartwig胺化：Pd催化芳基卤化物与胺的C-N键形成。Heck反应：Pd催化芳基卤化物与烯烃的C-C偶联。Grubbs烯烃复分解催化剂通过[2+2]环加成和逆[2+2]断裂交换烯烃的碳碳双键。

## 量子化学
Hartree-Fock近似将多电子波函数近似为单个Slater行列式——每个电子在其他电子的平均场中运动。Fock算符F=T+V_nuc+J-K包含动能、核吸引力、Coulomb排斥(J)和交换相互作用(K)。Roothaan方程将HF方程投影到基组上FC=SCε变为矩阵特征值问题。电子相关能是精确解与HF极限的差值——包含了瞬时电子-电子排斥(动态相关)和简并态混合(静态相关)。

密度泛函理论(DFT)将电子密度ρ(r)作为基本变量——总能量E[ρ]是密度的泛函包括动能、外部势、Coulomb能和交换相关能。Kohn-Sham方法引入非相互作用的辅助系统使精确动能近似可计算——交换相关泛函E_xc[ρ]是唯一未知项。Jacob的天梯对泛函分类：LDA(仅密度)→GGA(密度+梯度)→meta-GGA(+动能密度)→hybrid(+精确交换)→double hybrid(+MP2相关)。

## 生物化学
酶的催化机制：酶通过稳定过渡态降低活化能——过渡态与活性位点的互补性(形状、电荷、氢键)比底物更强。丝氨酸蛋白酶(胰蛋白酶、胰凝乳蛋白酶)的催化三联体Ser-His-Asp——His作为广义碱从Ser的羟基提取质子增强Ser的亲核性攻击肽键。碳酸酐酶的转换数(turnover number)高达10⁶ s⁻¹——Zn²⁺配位的水分子pKa从15.7降至约7使其在生理pH下产生高浓度的OH⁻亲核试剂。

蛋白质折叠的Anfinsen实验证明氨基酸序列包含折叠所需的所有信息(热力学假设)。Levinthal佯谬指出随机搜索所有构象需要宇宙年龄以上——蛋白质折叠必然是偏向的动力学过程。折叠漏斗模型将蛋白质能量面描述为朝向天然态的漏斗——能量和构象熵都向天然态递减。分子伴侣(Hsp70、GroEL-GroES)在细胞中辅助蛋白质正确折叠防止错误折叠和聚集。
"""


def _gen_biology() -> str:
    return """
# 分子生物学深入

## 基因表达调控
真核生物基因表达的调控层次：染色质结构(组蛋白修饰和DNA甲基化改变DNA的可及性)→转录起始(转录因子和增强子/启动子相互作用)→mRNA加工(剪接、5'加帽、3'加尾)→mRNA转运和定位→mRNA稳定性和降解→翻译调控(起始因子磷酸化、miRNA抑制)→翻译后修饰(磷酸化、泛素化、乙酰化)→蛋白质定位和降解。

增强子是远距离作用于启动子的DNA序列——通过染色质环化(CTCF和Cohesin介导)与启动子物理接触。超级增强子是大的增强子簇控制细胞身份基因——癌细胞获取超级增强子驱动癌基因表达(BRD4抑制剂BET域蛋白靶向超级增强子)。绝缘子(Insulator)阻断增强子-启动子相互作用定义调控域边界。

CRISPR干扰(CRISPRi)和CRISPR激活(CRISPRa)使用催化失活的dCas9融合转录抑制子(KRAB域)或激活子(VP64)靶向特定基因的启动子——可逆敲低或激活基因表达。碱基编辑器将dCas9或切口酶Cas9融合胞苷或腺苷脱氨酶实现C→T或A→G的精确碱基转换无需双链断裂和供体模板。

## 细胞信号转导
受体酪氨酸激酶(RTK)的二聚化是信号启动的关键步骤——配体(EGF)结合诱导受体构象变化暴露二聚化界面。自磷酸化在激酶域的活化环上产生磷酸化酪氨酸——这些pTyr作为细胞内信号蛋白(SH2域、PTB域)的停靠位点。Ras-MAPK级联放大信号：Ras(GTPase)激活Raf(MAPKKK)→磷酸化MEK(MAPKK)→磷酸化ERK(MAPK)→磷酸化数百种底物调控增殖和分化。

Wnt/β-catenin通路：无Wnt信号时β-catenin被破坏复合体(Axin、APC、GSK3β、CK1)磷酸化——磷酸化的β-catenin被泛素-蛋白酶体系统降解。Wnt结合Frizzled受体招募Dishevelled抑制破坏复合体——β-catenin积累进入细胞核与TCF/LEF转录因子共同激活Wnt靶基因。Wnt信号在胚胎发育、干细胞维持和癌症中至关重要——APC失活是结直肠癌最常见的起始事件。

## 系统生物学
基因调控网络(GRN)的建模：布尔网络将基因状态简化为ON/OFF使用逻辑规则描述调控关系——尽管简化但能重现细胞分化的多稳态。微分方程模型用Hill函数描述转录调控的连续动力学——d[mRNA]/dt=α·(TF/K_d)^n/(1+(TF/K_d)^n)-γ[mRNA]。随机模型(Gillespie算法)模拟mRNA和蛋白质分子的离散产生和降解——解释了基因表达噪音的来源和传播。

代谢网络的通量平衡分析(FBA)基于化学计量矩阵S和稳态假设S·v=0最大化生物量产生通量。约束条件包括热力学可行性(不可逆反应)、酶容量限制和营养摄取速率。FBA成功预测了大肠杆菌在多种碳源下的生长速率和基因敲除效应——约束的代谢模型已扩展到人类代谢网络(Recon3D)支持癌症代谢和药物靶点的研究。
"""


def _gen_algorithms() -> str:
    return """
# 算法设计进阶

## 近似算法
对于NP难优化问题近似算法在多项式时间内给出可证明的近似比。顶点覆盖的2-近似：每次选一条边的两个端点加入覆盖删除所有相邻边——每条选中的边必须覆盖至少一个端点所以2倍内最优。旅行商问题的Christofides算法：在满足三角不等式的度量TSP上达到3/2近似比——先找MST、在奇数度节点上找最小完美匹配、找到Euler回路、通过shortcut得到Hamilton回路。

集合覆盖问题的贪心近似给出H_d近似比其中d是最大集的大小——每步选择覆盖最多未覆盖元素的集合。贪心算法的分析使用每次迭代的边际收益论证——每次选择至少覆盖剩余未覆盖元素的1/OPT部分。最大割问题的半定规划(SDP)基于Goemans-Williamson的随机舍入——0.878近似比是NP难问题可近似性的里程碑。

## 在线算法
在线算法必须在输入到达时立即做出决策不知道未来的输入。确定性在线算法的竞争力(competitive ratio)是离线最优解与在线解的比值在最坏情况下的上界。滑雪租赁问题：每天决定是继续租(每天$1)还是买($B)——确定性算法的最优竞争力是2-1/B。平衡策略在租了B-1天后购买达到2-1/B竞争比。

分页问题的标记算法(Marking Algorithm)：将内存页面分为标记(自上次标记重置后访问过的)和未标记。驱逐未标记页面任意一个。在缓存大小k下标记算法是k-竞争的——最优离线算法(Belady算法驱逐未来最远才使用的页面)。工作集大小动态变化时LRU在平稳分布下性能良好——但最坏情况下的竞争力是k。

## 流算法
数据流模型处理大规模序列：算法只能一次扫描数据使用亚线性(通常polylog)空间。哈希函数和随机化是流算法的核心工具。多数元素检测(Misra-Gries)：维护k-1个计数器跟踪超过n/k频率的元素——每次k个不同元素同时减少所有计数器。Count-Min Sketch估计元素频率使用d个独立哈希函数映射到w宽度数组——查询返回d个计数器的最小值给出频率的上界。

HyperLogLog估计集合中不同元素的个数——使用前导零的最大个数近似log N空间复杂度O(log log N)。AMS采样维护L2范数的近似每个元素的生存概率与其当前值成正比。滑动窗口上的流算法维护时间衰减的摘要——指数直方图维护最近W个元素的统计量的(1+ε)近似。
"""


def _gen_systems() -> str:
    return """
# 系统设计与分布式系统

## 一致性协议深入
Raft的领导者选举使用随机选举超时防止分裂投票。每个跟随者在选举超时(150-300ms随机)后成为候选者增加当前任期号向其他节点索要投票。候选者获得多数票则成为领导者开始发送心跳(appendEntries)确立权威。如果两个候选者同时索取投票且票数分裂——等待另一个随机超时重新选举。

Multi-Paxos与Raft的比较：两者都依赖选举领导者优化连续决策的性能。Raft的设计强调可理解性——将共识分解为领导者选举、日志复制和安全性三个独立部分。Paxos允许乱序接受提案(Multi-Paxos需要额外的领导者机制)但Raft严格顺序——领导者只追加日志不修改现有条目。Raft的日志匹配性质保证如果两条日志条目有相同的任期和索引则它们及之前的所有条目完全相同。

## 数据库事务
快照隔离(Snapshot Isolation)允许每个事务看到一致的数据库快照消除了大部分读-写冲突但不防止写偏斜(Write Skew)。PostgreSQL使用多版本并发控制(MVCC)：每个元组有创建事务ID(xmin)和删除事务ID(xmax)——事务只看到xmin<当前快照且xmax未设置或在快照之后提交的元组。Oracle使用回滚段(Rollback Segment)存储旧版本实现读一致性。

可串行化(Serializable)是最强隔离级别防止所有并发异常。串行化图测试检查事务的读写依赖是否形成环——依赖环中至少有一个事务回滚。两阶段锁(2PL)保证可串行化：事务在执行读/写前获取共享/排他锁并在提交时释放所有锁。谓词锁锁定满足WHERE条件的行而不仅是存在的行防止幻读。

## 负载均衡
一致性哈希将服务器和键映射到同一个环上——键被分配给环上顺时针方向的第一台服务器。添加或删除服务器时只有K/N个键需要重新分配。虚拟节点(vnode)为每个物理服务器分配多个在环上的位置平滑化负载分布。Chord DHT使用一致性哈希实现分布式查找每个节点维护O(log N)大小的路由表(Finger Table)查找O(log N)跳。

最少连接(Least Connections)将请求发送到活跃连接数最少的后端——适用于请求处理时间高度分散的场景(L7 HTTP)。加权轮询(Weighted Round Robin)根据后端权重分配请求——每次选择weighted current最大的后端并将所有后端的current累加权重每当选完后端时将其current减去权重总和。一致性哈希与权重的结合使加权一致性哈希同时提供粘性会话和加权负载分配。
"""


def _gen_software() -> str:
    return """
# 软件工程最佳实践

## 代码审查清单
功能正确性：代码是否实现了需求的预期行为？是否有缺失的边界条件？错误处理和回滚逻辑是否完整？是否有隐藏的假设(如非空、已排序)未被验证？输入验证是否覆盖所有外部输入点(API参数、用户输入、文件内容)？

代码可维护性：命名是否清晰传达意图？函数是否做一件事并做好？是否有重复代码应被提取？依赖方向是否正确(高层→低层还是相反)？类/模块的内聚性高吗？耦合性低吗？

性能与安全：数据库查询是否有适当的索引？是否有N+1查询问题？循环中的重复计算是否可以提升？敏感数据是否被正确保护(传输加密、存储散列、日志脱敏)？是否有SQL注入、XSS或其他OWASP Top 10漏洞？

## 重构技巧
提取函数(Extract Function)：当一段代码可以被一个好名字描述其意图时提取为单独函数。以旧代码为委托保持旧接口兼容然后逐步迁移调用者到新函数。内联函数(Inverse of Extract Function)：当函数体与其名称一样清晰时消除不必要的间接层。

以多态替换条件(Replace Conditional with Polymorphism)：当switch/if-else链基于类型代码选择行为时使用子类多态——每个子类覆写对应行为。策略模式提供对象组合版本的多态——适合行为需要在运行时改变的场合。

分解条件(Decompose Conditional)：将复杂条件表达式提取为有名称的布尔函数——if(isEligibleForDiscount(user, order))比if(user.age>18 && user.memberSince.before(oneYearAgo) && order.total>100)更可读。

## 测试策略
合同测试(Contract Tests)验证服务间的API合同：消费者定义期望的请求/响应模式，提供者验证其满足这些合同。这比端到端测试更快更可靠——每个服务独立测试但验证集成点的兼容性。

属性基础测试(Property-Based Testing)：定义输入生成器和应保持的不变性质而非指定具体的输入输出对。例如：对任何字符串s，encode(decode(s))==s(往返不变性)；对任何两个列表，sort(a++b)==merge(sort(a),sort(b))(排序分配律)。测试框架(QuickCheck、Hypothesis)自动生成边界情况和缩小最小反例。

蜕变测试(Metamorphic Testing)：在没有测试预言(oracle)的情况下利用输入变化与输出变化的已知关系。例如：神经网络中两个输入在嵌入空间中接近则它们的预测标签应相似；搜索排名中如果位置1和2的结果交换则总体相关性评分应变化。
"""


def _gen_networking() -> str:
    return """
# 网络协议深入

## HTTP/3与QUIC
QUIC在UDP之上实现了类TCP的可靠传输同时消除队头阻塞。HTTP/2在单个TCP连接上多路复用多个请求——但TCP层的丢包会阻塞该连接上的所有流。QUIC为每个流提供独立的可靠交付——流的丢包只影响该流不影响其他流。0-RTT握手：如果客户端之前连接过服务器它可以使用预共享的密钥在第一条消息中就发送加密数据减少延迟。

连接迁移：QUIC连接由连接ID标识而非(源IP,源端口,目的IP,目的端口)四元组。客户端切换网络(Wi-Fi到蜂窝)时使用相同的连接ID继续连接无需重新握手——对移动应用体验至关重要。内置TLS 1.3加密：QUIC的每个包(除了少数初始包)都经过认证和加密——没有像TCP那样可以被中间件检查的明文部分。

## BGP深度
BGP属性决定路由选择顺序：最高Local Preference(本地优先级控制出站流量)、最短AS Path(路径长度)、最低Origin Type(IGP<EGP<Incomplete)、最低MED(Multi-Exit Discriminator控制入站流量)、eBGP优先于iBGP、最低IGP度量到下一跳、最低路由器ID。每条路由策略修改这些属性影响选路结果。

BGP的安全问题：路由劫持——AS错误地通告不属于自己的前缀(如2008年巴基斯坦电信劫持YouTube前缀导致全球中断)。ROA(Route Origin Authorization)使用RPKI验证AS是否被授权通告某个前缀。BGPsec扩展提供路径验证——每个AS在路由通告上签名包含下一个AS使路径不可伪造。然而BGPsec的部署面临计算开销和增量部署的挑战。

## 网络拥塞控制
BBR(Bottleneck Bandwidth and Round-trip propagation time)基于模型而非丢包信号控制发送速率。BBR周期性地探测：以估计的瓶颈带宽发送(测量最大交付速率)→以更高的速率短暂探测(寻找更大的带宽)→以较低的速率短暂探测(测量最小RTT刷新RTprop估计)。BBR在有一定丢包率的链路(如无线)上比基于丢包的算法(CUBIC)表现更好但可能导致延迟增加和公平性问题。

Swift是Google为Google数据中心间WAN设计的拥塞控制——使用基于延迟的目标(而非丢包)在极高速链路(100Gbps+)上实现低延迟和高利用率。Swift使用RTT的增量(而非绝对RTT)作为拥塞信号减少了延迟测量的噪声影响。PCC(Performance-oriented Congestion Control)使用在线学习选择效用最高的发送速率——每次尝试不同速率观察性能指标(吞吐量、延迟、丢包)然后收敛到优化效用函数的速率。
"""


def _gen_chinese() -> str:
    return """
# 中国典籍精华

## 《孙子兵法》选读
兵者国之大事也死生之地存亡之道不可不察也。故经之以五校之计而索其情一曰道二曰天三曰地四曰将五曰法。道者令民与上同意可与之死可与之生而不畏危也。天者阴阳寒暑时制也。地者远近险易广狭死生也。将者智信仁勇严也。法者曲制官道主用也。

知彼知己百战不殆不知彼而知己一胜一负不知彼不知己每战必败。故善战者立于不败之地而不失敌之败也。是故胜兵先胜而后求战败兵先战而后求胜。善用兵者修道而保法故能为胜败之政。

兵者诡道也。故能而示之不能用而示之不用近而示之远远而示之近。利而诱之乱而取之实而备之强而避之怒而挠之卑而骄之佚而劳之亲而离之。攻其无备出其不意此兵家之胜不可先传也。

## 《史记》选读
太史公曰：诗有之高山仰止景行行止虽不能至然心乡往之。余读孔氏书想见其为人。适鲁观仲尼庙堂车服礼器诸生以时习礼其家余祗回留之不能去云。天下君王至于贤人众矣当时则荣没则已焉。孔子布衣传十余世学者宗之。自天子王侯中国言六艺者折中于夫子可谓至圣矣。

屈原至于江滨被发行吟泽畔颜色憔悴形容枯槁。渔父见而问之曰子非三闾大夫与何故而至于斯。屈原曰举世混浊而我独清众人皆醉而我独醒是以见放。渔父曰圣人不凝滞于物而能与世推移。世人皆浊何不淈其泥而扬其波众人皆醉何不哺其糟而歠其酾何故深思高举自令放为。

## 唐宋八大家文选
韩愈《师说》：古之学者必有师师者所以传道受业解惑也人非生而知之者孰能无惑惑而不从师其为惑也终不解矣。生乎吾前其闻道也固先乎吾吾从而师之生乎吾后其闻道也亦先乎吾吾从而师之。吾师道也夫庸知其年之先后生于吾乎是故无贵无贱无长无少道之所存师之所存也。

柳宗元《小石潭记》：从小丘西行百二十步隔篁竹闻水声如鸣佩环心乐之。伐竹取道下见小潭水尤清冽。全石以为底近岸卷石底以出为坻为屿为嵁为岩。青树翠蔓蒙络摇缀参差披拂。潭中鱼可百许头皆若空游无所依日光下澈影布石上佁然不动俶尔远逝往来翕忽似与游者相乐。
"""


def _gen_philosophy() -> str:
    return """
# 哲学思想体系

## 存在主义
萨特的"存在先于本质"：人与物不同——人的存在(他存在的事实)先于他的本质(定义他是什么)。没有预先给定的人性——人首先存在然后在世界中行动选择并通过这些选择创造自己的本质。人是被判定为自由的——他不能选择不存在或逃避选择因为不选择也是一种选择。

海德格尔的此在(Dasein)分析：人不是与世界对立的孤立主体——此在本来就是"在世界之中的存在"(Being-in-the-world)。日常此在以"常人"(das Man)的方式存在陷入闲谈、好奇和两可之中。向死而存在(Being-towards-death)使此在从常人的沉沦中唤醒——直面自己的死亡揭示存在的本真可能性。

## 分析哲学
维特根斯坦的语言图像论(早期《逻辑哲学论》)：命题是事实的逻辑图像——语言和世界共享逻辑形式使语言能够描绘世界。凡可说的都能说清楚，对不可说的必须保持沉默。"我的语言的界限意味着我的世界的界限。"后期《哲学研究》转向语言游戏：语言的意义在于其使用——词语在各自语言游戏中获得意义就像棋子在各自的棋盘上获得意义。不存在语言的本质只有家族相似性联结的多种语言活动。

蒯因的自然化认识论：认识论不应寻求先验的知识基础而应作为心理学的一章——研究人类如何从感觉刺激构建世界理论。分析-综合区分的否定——没有纯粹的分析真理所有知识都在经验面前可修正。翻译的不确定性论题：没有唯一正确的意义事实——不同的翻译手册可以与所有行为证据相容但彼此不相容。

## 政治哲学
罗尔斯的正义论：正义的两个原则——每个人都有权拥有最广泛的基本自由相容于他人同等的自由(自由原则)；社会和经济不平等应安排得使它们对最不利者最有利(差别原则)并在公平的机会平等条件下开放给所有人。原始状态和无知之幕：在原初状态中各方不知道自己在社会中将占据什么位置——在此条件下选择的原则是公平的。

诺齐克的自由至上主义反驳：持有正义理论——获取正义(最初以正义方式获取财产)、转让正义(以自愿交换转让财产)、矫正正义(纠正过去的非正义)。只有最弱意义的国家(仅限于保护公民免受暴力、盗窃和欺诈)是正义的——任何更强大的国家都会侵犯个人权利。模式化的分配原则(如按需分配)不断被人们的自由行为打乱——自由颠覆模式。
"""


def _gen_history() -> str:
    return """
# 世界历史关键转折

## 农业革命
约公元前10000年新石器革命：人类从狩猎采集转向定居农业。肥沃新月地带(Fertile Crescent)最早驯化了小麦、大麦、豌豆、小扁豆以及山羊、绵羊、猪和牛。农业的出现导致人口密度剧增、劳动分工、社会阶层化和国家形成——但也造成营养多样性下降和传染病增加。贾雷德·戴蒙德在《枪炮、病菌与钢铁》中论证地理因素(可驯化的动植物物种、大陆轴线方向)解释了各大洲文明发展的差异。

## 轴心时代
公元前800-200年卡尔·雅斯贝尔斯所称的"轴心时代"——中国(孔子、老子)、印度(佛陀、耆那教大雄)、波斯(琐罗亚斯德)、以色列(先知)、希腊(苏格拉底、柏拉图、亚里士多德)同时出现了革命性的思想。轴心时代的共同特征是超越性——从神话思维转向理性反思和伦理普遍主义。这些思想奠定了至今影响人类的主要宗教和哲学传统。

## 科学革命
哥白尼《天体运行论》(1543)提出日心说挑战了托勒密的地心宇宙体系——将地球从宇宙中心移除动摇了中世纪的等级宇宙观。开普勒用椭圆轨道代替圆形轨道发现了行星运动三大定律——行星在椭圆轨道上运行、在相等时间内扫过相等面积、周期的平方与半长轴的立方成正比(T²∝a³)。伽利略通过望远镜观察(木星的卫星、金星的相位、月球的山脉)提供了日心说的经验证据并使用斜面实验开创了定量实验方法。

牛顿的《自然哲学的数学原理》(1687)统一了天文学和力学——万有引力定律F=Gm₁m₂/r²和三大运动定律将天体的运动和地上物体的运动纳入统一的数学框架。牛顿力学在18世纪由欧拉、拉格朗日、拉普拉斯等人发展为分析力学——拉格朗日方程和哈密顿原理提供了比牛顿定律更强大的推广和量子力学的出发点。
"""


def _gen_economics() -> str:
    return """
# 经济学原理

## 微观经济理论
消费者理论：在预算约束p·x≤w下最大化效用U(x)。Marshall需求x(p,w)是价格和收入的可观测函数。Hicks需求h(p,u)是达到给定效用水平u下成本最小化的商品束。Slutsky方程将价格变化的需求效应分解为替代效应(沿无差异曲线移动)和收入效应(购买力变化)：∂x_i/∂p_j=∂h_i/∂p_j-x_j∂x_i/∂w。Shephard引理：支出函数对价格的导数等于Hicks需求∂e(p,u)/∂p_i=h_i(p,u)。Roy恒等式：间接效用对价格的导数与对收入的导数之比等于Marshall需求。

一般均衡理论：Walras提出经济中所有市场同时出清的价格向量存在性由Arrow-Debreu(1954)使用Kakutani不动点定理证明——在凸性、连续性和严格正价格的假设下。福利经济学第一基本定理：竞争均衡是Pareto有效的(看不见的手定理)。第二基本定理：任何Pareto有效配置可以通过初始禀赋的适当重分配由竞争均衡实现——将效率与分配问题分离。

## 宏观经济理论
IS-LM模型(希克斯1937)：IS曲线表示投资=储蓄时GDP与利率的组合——向下倾斜(利率下降刺激投资和GDP)。LM曲线表示货币供给=货币需求时GDP与利率的组合——向上倾斜(GDP上升增加货币交易需求利率随之上调)。IS-LM的交点确定短期均衡的GDP和利率水平。财政扩张(增加G或减税)向右移动IS曲线增加GDP和利率。货币扩张(增加M)向右移动LM曲线增加GDP降低利率。

Solow增长模型：人均资本积累dk/dt=sf(k)-(n+δ)k其中s是储蓄率f(k)是生产函数n是人口增长率δ是折旧率。稳态满足sf(k*)=(n+δ)k*——在稳态人均资本和人均产出不再增长。Solow残差：经验分解发现产出的增长只有一小部分可由资本和劳动的投入增长解释——剩余的"全要素生产率(TFP)"反映技术进步。内生增长理论(Romer、Lucas)将技术进步内生化——知识溢出和人力资本积累产生持续增长而非收敛到稳态。
"""


def _gen_coding() -> str:
    return """
# 编程实践进阶

## 并发模式的实现
读写锁(RWLock)的实现考虑写者饥饿问题。基本实现：读者获取共享锁在有写者等待时新的读者被阻塞——防止写者无限等待(写者优先)。乐观锁使用版本号：读取数据时记录版本号，写入时检查版本号是否变化(比较并交换)。如果版本号相同则更新数据并递增版本号；如果版本号不同则放弃整个操作并重试。乐观锁在争用低时效率高(无锁等待)争用高时重试浪费。

发布-订阅模式的核心设计决策：推送vs拉取——推送模式实现在发布者端将消息复制到多个订阅者队列(扇出)；拉取模式下订阅者按自己的节奏消费消息。主题/队列命名约定——分级命名(trading.equities.us)支持通配符匹配(trading.*.us)。死信队列处理无法被消费的消息(格式错误、超出重试次数)——运营人员可检查死信队列排查问题。

## 性能反模式
过早优化的陷阱：在没有测量数据的情况下"优化"代码——开发者的直觉经常错误地识别瓶颈。先让代码正确工作再使用profiler(perf、pprof、flamegraph)找到真正的瓶颈。优化的目标应该是热度最高的代码路径(Amdahl定律)——优化1%执行频率的代码100%提升不如优化50%频率的代码10%提升。保持原始未"优化"的版本作为正确性基准用于回归测试。

对象生命周期管理：频繁的分配/释放导致内存碎片和GC压力。对象池(object pool)复用已分配的对象避免分配开销——适合大对象、昂贵初始化或高频率分配的场景(database connection、thread、buffer)。减少引用的深度和数量——深层次的对象图在GC标记阶段遍历所有引用增加GC暂停时间。最终化(finalization)的不确定性——析构函数/终结器的调用时机不确定不应用于资源清理(C#的using、Java的try-with-resources、Python的with是确定的)。

## 错误处理哲学
异常的正确使用：只在异常情况下使用异常——正常控制流(如用户未找到)不应使用异常而应使用返回值(Result类型、Optional)。异常应包含足够的上下文信息帮助调试——操作类型、输入参数、系统状态。捕获具体异常类型而非泛化的Exception——不同异常需要不同的恢复策略。异常安全性的三个保证级别：基本保证(不泄漏资源不变式保持)、强保证(操作要么成功要么回滚到操作前状态)、不抛出保证(永远不会失败)。

Go的错误处理模式：显式的error返回值而非try-catch——if err!=nil {return fmt.Errorf("context: %w", err)}。错误包装(wrapping)使用%w动词保留原始错误的类型和上下文——errors.Is用于检查错误链中是否存在特定错误、errors.As提取错误链中特定类型的错误。对于不可恢复的错误使用panic/recover——recover只在当前goroutine中有效且只在deferred函数中有效。
"""


def _gen_data_science() -> str:
    return """
# 数据科学与机器学习实践

## 特征工程
数值特征的处理：分箱(Binning)将连续值离散化为有序类别——等宽分箱(固定区间大小)简单但受异常值影响；等频分箱(固定样本数)自适应数据分布但丢失边界信息。分位数分箱使用数据分布的分位数作为边界。对数变换(log(x+1))处理长尾分布(如收入、点击量)——缩小高值的影响范围使线性模型更有效。Box-Cox变换(Y^λ-1)/λ对λ从-5到5搜索最佳的归一化变换。

类别特征编码：独热编码(One-Hot Encoding)为每个类别创建二进制列——简单但类别多时维度爆炸(高维稀疏矩阵)。目标编码(Target Encoding)用每个类别对应的目标变量均值替代类别标签——包含正则化以处理小类别(使用全局均值作为先验均值shrinkage)。Count Encoding用每个类别在训练数据中出现的次数替代标签——简单但可能泄露测试数据信息。Embedding编码将每个类别映射为可训练的密集向量——捕获类别之间的语义相似性。

## 模型评估
交叉验证的策略选择：K折交叉验证将数据分为K份每份轮流作为验证集——K=5或10是常见选择平衡偏差和方差。分层K折(Stratified K-Fold)保证每折中目标变量的分布与整体相同——对不平衡分类数据集至关重要。时间序列的交叉验证使用扩展窗口(不断累积历史数据)或滚动窗口(固定大小的历史窗口向前移动)——不能随机分割因为时间顺序包含信息。

离线评估与在线评估的差距：离线评估使用历史数据在固定的数据集上测量模型性能——反映了模型在历史数据上的表现但无法衡量部署后的真实影响(分布偏移、用户行为改变)。在线A/B测试将用户随机分配到实验组和对照组测量实际业务指标(点击率、转化率、留存率)——是模型效果的黄金标准。Interleaving实验交替展示两模型的结果测量用户偏好——敏感度比A/B测试高出多个数量级。

## AutoML
超参数优化方法：网格搜索(Grid Search)穷举搜索预设的超参数组合——维度灾难使全面搜索不可行。随机搜索(Random Search)从超参数空间中随机采样——比网格搜索更高效(大多数超参数的重要性低几个重要超参数获得更多尝试)。贝叶斯优化使用高斯过程或树形Parzen估计器(TPE)建模超参数与性能的关系——采样期望改善(EI)最大的点平衡探索和利用。HyperBand将连续减半(Successive Halving)与随机搜索结合——早期停止性能差的配置将资源集中到有前途的配置上。
"""


def _gen_security() -> str:
    return """
# 信息安全深入

## 密码分析
差分密码分析(Biham-Shamir 1990)：通过分析明文对的特定差异在密文对中的传播来恢复密钥。对DES的攻击：选择具有特定XOR差异的明文对观察经过各轮后差异的传播——某些差异路径(差分特征)以高于随机概率连接明文差异和密文差异。正确密钥猜测会使预期差分出现得更频繁——错误密钥下差异分布均匀。线性密码分析(Matsui 1993)：用明文、密文和密钥位的线性近似(P[某些位XOR=0]与½的偏差)逐步恢复密钥位。

侧信道防御的恒定时间实现：避免基于秘密值的数据依赖分支和内存访问。对于RSA的模幂：使用Montgomery梯子(无论密钥位是0还是1都执行相同的乘法和平方操作)。对于AES：使用常数时间S盒实现(避免查找表——在支持AES-NI的CPU上使用硬件加速AES指令)。内存访问模式也应独立于秘密——避免在数组索引中使用秘密值(如查找S盒)导致缓存时序泄露。

## 后量子密码学
基于格的密码学：Learning With Errors(LWE)问题——给定A和b=As+e(其中e是小误差向量)恢复s。Regev(2005)证明LWE至少和GapSVP(最坏情况格问题)一样难。Ring-LWE使用多项式环R_q=Z_q[x]/(x^n+1)上的运算利用NTT(数论变换)实现高效实现。NTRU基于在多项式环中找到短多项式的困难性。Kyber(CRYSTALS-Kyber)是NIST选定的基于格的密钥封装机制(KEM)使用Module-LWE(介于LWE和Ring-LWE之间)。

基于哈希的签名：Merkle签名方案(MSS)使用Merkle树将多个一次性签名(如Winternitz OTS)组合成多次使用的公钥。SPHINCS+是NIST选定的无状态基于哈希的签名方案——使用FORS(Forest of Random Subsets)处理少量签名和XMSS处理大量签名的分层Merkle树。基于哈希的签名的主要缺点是签名大小(SPHINCS+-128s约8KB)和签名数量有限(MSS)。

## Android安全
Android的权限模型：安装时权限(正常权限自动授予如INTERNET、危险权限必须在运行时动态请求如CAMERA、签名权限仅授予与声明者同签名的应用)。Android 6.0前后：之前权限在安装时一次性授予不可撤销；之后危险权限分组运行时动态请求用户可以按权限单独撤销。

Android的沙盒机制：每个应用以独立的Linux UID运行——内核在进程级别实施隔离。每个应用有独立的/data/data/<package>目录只有该应用可以访问。应用组件(Activity、Service、BroadcastReceiver、ContentProvider)通过Intent通信——Intent Filter定义组件可接收的Intent类型。导出的ContentProvider应验证调用者的权限和输入数据的格式防止SQL注入和路径遍历。
"""


def _gen_robotics() -> str:
    return """
# 机器人学与控制

## 运动学与动力学
正向运动学(FK)：给定关节角度θ计算末端执行器的位姿。对串联机械臂使用Denavit-Hartenberg(DH)参数——每个关节用4个参数(a_i、α_i、d_i、θ_i)描述。变换矩阵T_i^{i-1}将坐标系i中的坐标变换到坐标系i-1：绕z轴旋转θ_i、沿z轴平移d_i、沿x轴平移a_i、绕x轴旋转α_i。将所有变换矩阵相乘T_n^0=T_1^0 T_2^1 ... T_n^{n-1}得到末端位姿。

逆向运动学(IK)：给定末端位姿求解关节角度。对于6自由度机械臂：当最后三个关节轴交于一点(腕部中心)时解耦为位置IK(前三个关节)和方向IK(后三个关节)。解析解存在但可能有多组解(如肘部上/下、腕部翻转)。数值IK使用雅可比矩阵的伪逆迭代更新关节角——Δθ=J⁺Δx其中J⁺=J^T(JJ^T)^{-1}。雅可比矩阵将关节速度映射到末端速度ẋ=J(θ)θ̇。

## 路径规划
基于采样的规划器：PRM(概率路线图)在配置空间中随机采样无碰撞配置并将邻近的配置用局部规划器连接——构建路线图然后使用图搜索(A*)在此路线图上找到路径。RRT(快速探索随机树)从起始配置向外生长树——随机采样配置将树中最近的节点向采样方向延伸固定步长。RRT-Connect同时从起始和目标生长两棵树交替扩展——一棵树的新节点成为另一棵树的目标偏向生长。

轨迹优化：CHOMP(协变Hamilton优化)将路径表示为一组等间隔时间点的状态使用梯度下降同时优化平滑性和避碰。TEB(时间弹性带)将路径表示为位姿和时间间隔的序列——优化目标包括路径长度、执行时间、与障碍物的距离和速度/加速度限制。在动态环境中MPC(模型预测控制)在每个时间步求解有限时域的优化问题应用第一步控制然后重新规划(滚动时域)。

## 状态估计
卡尔曼滤波：预测步骤x̂^-_k=Ax̂_{k-1}+Bu_k、P^-_k=AP_{k-1}A^T+Q。更新步骤K_k=P^-_kH^T(HP^-_kH^T+R)^{-1}、x̂_k=x̂^-_k+K_k(z_k-Hx̂^-_k)、P_k=(I-K_kH)P^-_k。卡尔曼增益K_k根据预测的协方差和测量的噪声平衡预测和测量——测量精确(小R)时更信任测量、模型精确(小Q)时更信任预测。

粒子滤波：使用加权的随机样本(粒子)表示任意概率分布。重要性采样根据建议分布生成粒子并根据观测概率赋予权重——w_k=w_{k-1}p(z_k|x_k)。重采样步骤消除权重退化的粒子复制高权重粒子——有效粒子数N_eff=1/Σw_i²低于阈值时触发重采样。粒子滤波适用于非线性非高斯系统(全球定位、多目标跟踪)但粒子数需要随状态维度指数增长(维度诅咒)。
"""


def _gen_medicine() -> str:
    return """
# 医学基础

## 病理生理学
炎症的五个经典标志：红(rubor)、热(calor)、肿(tumor)、痛(dolor)、功能丧失(functio laesa)。急性炎症的血管反应：血管扩张(组胺、一氧化氮)增加血流导致红和热；血管通透性增加(组胺、缓激肽、白三烯)使血浆渗出导致肿；炎症介质(前列腺素、缓激肽)刺激神经末梢导致痛。白细胞招募的级联：选择素介导的滚动→趋化因子激活整合素→牢固粘附→跨内皮迁移(游出)。

动脉粥样硬化的进展：内皮功能障碍→LDL渗透进入内膜→LDL被氧化(oxLDL)→单核细胞招募→转化为巨噬细胞→吞噬oxLDL形成泡沫细胞→脂肪条纹→平滑肌细胞迁移和增殖→细胞外基质沉积→纤维斑块。不稳定斑块的特征：大脂质核心、薄纤维帽、大量炎症细胞(巨噬细胞分泌基质金属蛋白酶降解纤维帽)、斑块内出血。斑块破裂暴露促凝血组织因子引发血栓形成导致急性血管事件(心肌梗死、卒中)。

## 药理学原理
药代动力学四个阶段：吸收(给药部位到体循环——口服药物的生物利用度F=AUC_oral/AUC_iv受首过代谢影响)、分布(药物从血液进入组织——分布容积V_d=Dose/C_0越大组织结合越多)、代谢(主要在肝脏中通过I相氧化还原水解(CYP450)和II相结合(葡萄苷酸化、硫酸化)增加水溶性促进排泄)、排泄(主要通过肾脏——肾小球滤过→肾小管重吸收→肾小管分泌)。

药物作用的量效关系：EC50产生50%最大效应的浓度。K_d是药物-受体复合物的解离常数——亲和力越高K_d越小。治疗指数TI=TD50/ED50(中毒剂量与有效剂量之比)衡量药物的安全性——TI越宽越安全。华法林治疗窗窄TI低需要定期监测INR。激动剂与受体结合激活信号转导——完全激动剂产生最大效应、部分激动剂产生亚最大效应、反向激动剂在固有活性受体上产生负效应。拮抗剂结合但不激活受体阻断激动剂的作用——竞争性拮抗剂可被高浓度激动剂克服而非竞争性拮抗剂不能。

## 神经科学
动作电位的离子基础：静息膜电位(约-70mV)由K+的Nernst平衡电位主导——静息K+通道开放维持K+外流。去极化至阈电位(约-55mV)时电压门控Na+通道激活——Na+内流快速去极化细胞(上升相)。Na+通道失活和电压门控K+通道延迟激活导致K+外流——细胞复极化(下降相)。不应期：绝对不应期(Na+通道全部失活无论多大刺激无法触发动作电位)、相对不应期(需更大刺激触发部分Na+通道已恢复但K+通道仍开放)。
"""


def _gen_law() -> str:
    return """
# 法律学基础

## 宪法原理
权力分立(Montesquieu)：立法权(议会制定法律)、行政权(政府执行法律)、司法权(法院解释和适用法律)应分配于不同机关相互制衡。美国宪法的制衡机制：总统否决国会法案(立法制约行政)、国会以三分之二多数推翻总统否决(行政制约立法)、联邦法官由总统提名参议院确认(行政+立法制约司法)、司法审查(Marbury v. Madison 1803确立法院有权审查法律是否违宪)。

基本权利的审查标准(德国比例原则)：适当性原则(手段有助于目的实现)、必要性原则(没有更轻微的手段能同样有效地实现目的)、均衡性原则(狭义比例性——手段造成的损害与实现目的的利益不成比例)。美国的三层审查标准：严格审查(基本权利如言论自由和可疑分类如种族——法律必须服务于压倒性的政府利益且措施是限制性最小的)、中度审查(性别分类——法律必须实质关联于重要的政府利益)、合理基础审查(经济和社会立法——法律必须合理关联于正当的政府利益)。

## 刑法基础
犯罪的构成要件：客观构成要件(行为、结果、因果关系)、主观构成要件(故意或过失)、违法性(不存在正当防卫、紧急避险等正当化事由)、有责性(行为人具有刑事责任能力且不存在不可避免的违法性认识错误)。故意分为：一级直接故意(意图实现构成要件)、二级直接故意(确知行为导致构成要件)、间接故意(容忍构成要件的发生)。过失分为：有认识的过失(预见可能性但轻信能够避免)和无认识的过失(应当预见但没有预见)。

刑法中的因果关系：条件说(but-for test)——没有行为就没有结果时行为与结果有因果关系。相当因果关系说——根据一般人的经验判断行为导致结果的相当性排除偶然因果链。客观归责理论：行为创造了法不容许的风险且该风险在构成要件结果中实现——如果结果的发生不在规范的保护目的范围内即使有因果关系也不能归责(如闯红灯的行人恰好被遵守交规但刹车距离不足的车辆撞死不应归责于闯红灯行为)。

## 合同法
合同的成立：要约(一方以缔约为目的向相对方发出的确定的意思表示)与承诺(受要约人完全接受要约内容的意思表示)达成合意。要约与要约邀请的区别——要约邀请(如商品陈列、广告)是希望他人向自己发出要约。对价(Consideration)是英美合同法的特有要件——各方必须提供有价值的事物作为交换但不需要相等(胡椒规则——一粒胡椒也可以构成充分对价)。允诺禁反言(Promissory Estoppel)在缺乏对价但受诺人合理信赖允诺而受损时强制执行允诺。
"""


def _gen_architecture() -> str:
    return """
# 建筑学原理

## 结构体系
框架结构由梁和柱组成的骨架承受荷载——荷载路径：楼板→梁→柱→基础→地基。框架对侧向力(风、地震)的抵抗依赖刚性节点(弯矩框架)或支撑系统(斜撑、剪力墙)。剪力墙是垂直的钢筋混凝土墙体抵抗水平力——在平面上对称布置避免扭转。核心筒结构将电梯井、楼梯间和机电设备集中在中央形成刚度极大的竖向悬臂抵抗侧向力。

拱结构利用几何形状将弯矩转化为轴向力——适合使用抗压强度高但抗拉强度低的材料(石、砖、混凝土)。抛物线拱在均布荷载下无弯矩——理想拱轴线是压力线。穹顶是旋转产生的外壳结构——顶部受压环底部受拉环需要环形链或扶壁抵抗外推。薄壳结构(如悉尼歌剧院的壳形屋顶)利用双曲率获得刚度——厚度与跨度之比可以达到鸡蛋壳的比例。

## 建筑环境
自然通风的驱动机制：风压驱动——迎风面正压背风面负压形成压差驱动穿堂风需要建筑两边有开口。热压驱动(烟囱效应)——室内外温差产生密度差热空气上升从高处开口排出冷空气从低处开口补充。通风口面积应为地板面积的至少5%实现有效自然通风。中庭作为热压通风的烟囱——顶部开口和底部开口的高度差和温差决定驱动力。

日照与热工：南向开口(北半球)获取冬季低角度太阳的被动式太阳能——水平遮阳阻挡夏季高角度太阳同时允许冬季低角度太阳进入。玻璃的性能指标：U值(导热系数越低越好W/m²K)、SHGC(太阳得热系数控制传入的太阳辐射)、VT(可见光透过率)。Low-E涂层反射红外辐射降低U值同时允许可见光透过。相变材料(PCM)在固液相变中吸收或释放大量潜热平稳室内温度波动——石蜡基PCM的相变温度可定制(20-30℃范围适合建筑应用)。
"""


def _gen_music() -> str:
    return """
# 音乐理论基础

## 和声学
功能和声的三大功能：主功能(Tonic I,vi,iii)稳定、归属功能(Dominant V,vii°)强烈倾向解决到主、下属功能(Subdominant IV,ii)创造远离主的运动准备属功能。和弦进行的语法：I-任何和弦、任何和弦-V-I。五度圈进行(I-IV-vii°-iii-vi-ii-V-I)是和声最强的推动力每步都是上四度(或下五度)进行。

爵士和声的扩展和弦：七和弦(maj7、7、m7、m7♭5、dim7)在基础三和弦上增加第四音。九和弦、十一和弦和十三和弦进一步叠加三度形成丰富的和声色彩。Tritone替代：任何属七和弦可被相距三全音(增四度/减五度)的属七和弦替代——G7(D-F#-A-C)被D♭7替代因为导音(F#→G和C→B共同音)。这种替代创造半音进行的低音线条——Dm7-D♭7-Cmaj7。

## 数字信号处理在音乐中的应用
傅里叶变换在音频中的应用：短时傅里叶变换(STFT)在重叠窗口上运行FFT生成频谱图——横轴时间、纵轴频率、颜色表示幅度。从频谱图可以看出音符的基频和谐波结构——谐波在基频的整数倍上泛音的相对强度决定音色(如单簧管的奇数倍谐波强度更大创造出空洞的音色)。梅尔频率倒谱系数(MFCC)模拟人耳对频率的非线性感知——低频比高频有更高的频率分辨率广泛用于语音识别和音乐信息检索。

物理建模合成(Physical Modeling Synthesis)：使用数字波导(Digital Waveguide)模拟弦乐器的振动——延迟线模拟波的传播、滤波器模拟反射和损耗。Karplus-Strong算法是最简单的波导合成——短噪声激励通过低通滤波器反馈产生拨弦音色。波导模型也可以模拟管乐器(cylindrical vs conical bore)、人声(声道滤波器)、打击乐器(二维膜振动)。与采样合成相比物理建模占用更少内存但计算量更大——允许实时修改物理参数(弦长度、材料、激发位置)产生连续变化的音色。
"""


# ---------------------------------------------------------------------------
# English content generators (Round 6)
# ---------------------------------------------------------------------------

def _gen_en_cs_ai() -> str:
    return """
# Computer Science and Artificial Intelligence

## Deep Learning Architecture Design

The transformer architecture introduced self-attention as an alternative to recurrence. Given input sequence X of length n with d-dimensional embeddings, self-attention computes Q=XW_Q, K=XW_K, V=XW_V where W_Q, W_K, W_V are learned projection matrices of shape d x d_k, d x d_k, d x d_v. Attention weights are A = softmax(QK^T/sqrt(d_k)), producing context-aware representations Z = AV. Multi-head attention runs h parallel attention heads, concatenating results: MultiHead(X) = Concat(head_1,...,head_h)W_O. Each head captures different relationship types — syntactic dependencies, semantic similarity, positional proximity.

Layer normalization stabilizes training by normalizing activations across the feature dimension. Pre-LayerNorm (applied before attention/FFN sublayers) empirically yields smoother gradients and allows larger learning rates compared to post-LayerNorm. The feed-forward network FFN(x) = GELU(xW_1+b_1)W_2+b_2 with inner dimension d_ff typically 4x d_model provides position-wise nonlinear transformation.

RoPE (Rotary Position Embedding) encodes position information by rotating query and key vectors: for position m, dimension pair (2i, 2i+1), the rotation angle is m*theta_i where theta_i = base^{-2i/d}. This naturally captures relative position — the dot product between query at m and key at n depends only on (m-n). RoPE generalizes to longer sequences than those seen during training since relative positions are bounded.

## Reinforcement Learning Fundamentals

The Markov Decision Process (MDP) framework: states S, actions A, transition function P(s'|s,a), reward function R(s,a,s'), discount factor gamma. A policy pi(a|s) maps states to action distributions. The state value V^pi(s) = E[sum_{t=0}^{inf} gamma^t R_t | s_0=s, pi] is expected cumulative discounted reward. The action-value Q^pi(s,a) = E[R_0 + gamma V^pi(s_1) | s_0=s, a_0=a].

The Bellman optimality equation: Q*(s,a) = E[R + gamma max_{a'} Q*(s',a')]. Value iteration and policy iteration converge to Q* for tabular MDPs. Q-learning uses temporal difference updates: Q(s,a) <- Q(s,a) + alpha[R + gamma max_{a'}Q(s',a') - Q(s,a)]. The TD error delta = R + gamma max Q(s',a') - Q(s,a) drives learning — positive delta increases Q(s,a), negative decreases it.

Deep Q-Networks (DQN) approximate Q* with neural networks, using experience replay (store transitions in buffer, sample random minibatches for training) and target networks (periodically copy online network to target network for stable TD targets). Double DQN reduces overestimation bias by decoupling action selection (online network) from value estimation (target network). Dueling DQN splits the Q-network into value stream V(s) and advantage stream A(s,a), combining them as Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).

Policy gradient methods directly optimize policy parameters: gradient J(theta) = E[sum_t grad(log pi_theta(a_t|s_t)) * R_tau]. The REINFORCE estimator has high variance mitigated by subtracting a baseline b(s_t) (typically V(s_t)). Actor-critic methods combine a policy network (actor) with a value network (critic), using the critic's TD error or advantage estimate A(s,a) = Q(s,a) - V(s) to reduce policy gradient variance.

Proximal Policy Optimization (PPO) constrains policy updates using a clipped surrogate objective: L = E[min(r_t(theta)*A_t, clip(r_t(theta), 1-eps, 1+eps)*A_t)] where r_t = pi_theta(a_t|s_t)/pi_old(a_t|s_t) is the probability ratio. This prevents destructively large policy updates while being simpler to implement than TRPO's constrained optimization.

## Database Systems Internals

B+ trees are the standard index structure in relational databases. Internal nodes store key-pointer pairs for navigation; leaf nodes store key-value pairs (or key-rowid) linked as a sorted doubly-linked list for range scans. Fanout f determines depth — a B+ tree with fanout 500 and 3 levels can index 125 million records. Insert and delete operations maintain balance via node splitting (when occupancy exceeds capacity) and merging/redistribution (when occupancy drops below threshold).

Write-Ahead Logging (WAL) ensures atomicity and durability: before modifying data pages, log records describing the changes are written to sequential WAL files. On recovery after crash: (1) redo phase replays all committed transactions from the last checkpoint, (2) undo phase rolls back uncommitted transactions. The log sequence number (LSN) monotonically increases, ordering all modifications. ARIES (Algorithm for Recovery and Isolation Exploiting Semantics) implements this with dirty page tables and precise undo/redo bookmarking.

MVCC (Multi-Version Concurrency Control) allows readers to see a consistent snapshot without blocking writers. Each row has associated transaction IDs (xmin for creation, xmax for deletion/update). A transaction sees rows where xmin <= snapshot_xmin AND (xmax is null OR xmax > snapshot_xmax). Writers create new row versions; old versions are garbage-collected by VACUUM when no active snapshot needs them. This enables high concurrency — readers never wait for writers and writers don't block readers.
"""


def _gen_en_programming() -> str:
    return """
# Software Engineering and Programming

## System Design Principles

SOLID principles: Single Responsibility — a class should have exactly one reason to change. Open/Closed — open for extension, closed for modification (achieved through abstractions and polymorphism). Liskov Substitution — subtypes must be substitutable for their base types without altering program correctness (derived classes must not strengthen preconditions or weaken postconditions relative to the base contract). Interface Segregation — many specific interfaces are better than one general-purpose interface; clients should not depend on methods they don't use. Dependency Inversion — high-level modules should not depend on low-level modules; both should depend on abstractions.

The Clean Architecture (Robert C. Martin) organizes code into concentric circles: Entities (enterprise business rules, innermost), Use Cases (application-specific business rules), Interface Adapters (controllers, gateways, presenters), Frameworks & Drivers (web, database, UI, outermost). Dependencies point inward — outer layers depend on inner layers, never the reverse. This allows the core business logic to be framework-agnostic, database-agnostic, and UI-agnostic.

## Concurrency and Parallelism

Thread safety guarantees in Java's memory model: volatile ensures visibility (writes are immediately visible to all threads) but not atomicity. synchronized provides mutual exclusion + visibility — the monitor lock guarantees that only one thread executes the critical section. java.util.concurrent provides higher-level abstractions: ExecutorService for thread pool management, ConcurrentHashMap with striped locking for high-concurrency maps, CountDownLatch/CyclicBarrier for coordination, CompletableFuture for composable asynchronous computation.

The Actor model (Erlang, Akka) avoids shared mutable state entirely. Each actor has a mailbox and processes messages sequentially. Actors can: send messages to other actors, create new actors, designate behavior for the next message. Supervision hierarchies — parent actors monitor child actors and handle failures (restart, stop, escalate). This containment of failure enables building fault-tolerant systems where failures in one component don't cascade.

Go's concurrency model: goroutines are lightweight user-space threads multiplexed onto OS threads by the Go runtime scheduler (M:N scheduling). Channels provide typed communication between goroutines — ch := make(chan int, bufferSize). The select statement multiplexes across multiple channel operations. The principle: "Do not communicate by sharing memory; instead, share memory by communicating." Context propagation carries deadlines, cancellation signals, and request-scoped values through call chains.

## Testing Strategies

The testing pyramid: unit tests (fast, numerous, test individual functions/methods in isolation), integration tests (test interactions between modules, slower), end-to-end tests (test entire system from user perspective, slowest, fewest). Unit tests should avoid I/O, database access, and network calls — use test doubles (stubs for providing canned answers, mocks for verifying interactions). Property-based testing (QuickCheck, Hypothesis) generates random inputs and verifies invariants hold: for any integer list, reverse(reverse(xs)) == xs. Mutation testing evaluates test suite quality by injecting artificial bugs and checking if tests catch them.
"""


def _gen_en_mathematics() -> str:
    return """
# Mathematics

## Linear Algebra

A vector space V over field F is a set with vector addition and scalar multiplication satisfying 8 axioms: associativity, commutativity of addition, additive identity, additive inverses, scalar multiplicative identity, scalar associativity, and distributivity (two forms). Linear independence: a set of vectors {v_1,...,v_n} is independent if sum(c_i v_i) = 0 implies all c_i = 0. A basis is a maximal linearly independent set (or minimal spanning set). Dimension is the cardinality of any basis — all bases of a finite-dimensional vector space have the same size.

Eigenvalues and eigenvectors: for a linear operator T: V -> V, a nonzero vector v is an eigenvector with eigenvalue lambda if Tv = lambda v. The characteristic polynomial p(lambda) = det(T - lambda I) has eigenvalues as roots. Diagonalization: if T has n linearly independent eigenvectors, T = PDP^{-1} where D is diagonal with eigenvalues and P has eigenvectors as columns. Symmetric matrices are diagonalizable by an orthogonal matrix (Spectral Theorem). The Singular Value Decomposition (SVD) generalizes eigendecomposition to rectangular matrices: A = U Sigma V^T where U and V are orthogonal, Sigma is diagonal with nonnegative singular values.

## Real Analysis

A sequence (x_n) converges to L if for every epsilon > 0 there exists N such that |x_n - L| < epsilon for all n > N. Cauchy sequences: sequences where terms become arbitrarily close to each other (for all epsilon > 0, exists N such that |x_m - x_n| < epsilon for m,n > N). In complete metric spaces (including R^n), a sequence converges iff it is Cauchy. Compactness in R^n (Heine-Borel): a set is compact iff it is closed and bounded.

Continuity: f is continuous at a if lim_{x->a} f(x) = f(a). Equivalently, for every epsilon > 0 there exists delta > 0 such that |x-a| < delta implies |f(x)-f(a)| < epsilon. Uniform continuity: delta depends only on epsilon, not on a (stronger condition). The Intermediate Value Theorem: a continuous function on [a,b] takes every value between f(a) and f(b). The Extreme Value Theorem: a continuous function on a compact set attains its maximum and minimum.

Differentiability in R^n: the derivative of f: R^n -> R^m at point a is a linear transformation Df(a): R^n -> R^m satisfying lim_{h->0} ||f(a+h)-f(a)-Df(a)h||/||h|| = 0. The Jacobian matrix J_f entries are partial derivatives J_ij = partial f_i / partial x_j. The gradient of scalar function f is a row vector (or column depending on convention). The Hessian H_f is the matrix of second partial derivatives — symmetry: (H_f)_ij = (H_f)_ji when second partials are continuous.

## Information Theory

Entropy H(X) = -sum_x p(x) log p(x) measures uncertainty in bits (log base 2) or nats (natural log). Joint entropy H(X,Y). Conditional entropy H(Y|X) = H(X,Y) - H(X). Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y) measures shared information between X and Y. The Data Processing Inequality: if X -> Y -> Z forms a Markov chain, then I(X;Y) >= I(X;Z) — post-processing cannot increase information.

The Asymptotic Equipartition Property (AEP): for i.i.d. sequences, the typical set A_epsilon^n contains sequences whose sample entropy is within epsilon of the true entropy H. As n grows, the typical set contains almost all probability mass while occupying a fraction approaching 2^{-nH} of the sample space. This underlies Shannon's source coding theorem: the fundamental limit of lossless compression is H bits per symbol.
"""


def _gen_en_sciences() -> str:
    return """
# Natural Sciences

## Physics

Maxwell's equations in differential form describe classical electromagnetism: Gauss's law div E = rho/epsilon_0 (electric charges are sources of electric fields), Gauss's law for magnetism div B = 0 (no magnetic monopoles), Faraday's law curl E = -dB/dt (changing magnetic fields produce electric fields), Ampere-Maxwell law curl B = mu_0(J + epsilon_0 dE/dt) (currents and changing electric fields produce magnetic fields). The displacement current term epsilon_0 dE/dt was Maxwell's addition — it predicts electromagnetic waves traveling at c = 1/sqrt(mu_0 epsilon_0), the speed of light.

The photoelectric effect demonstrated light's particle nature: electrons are ejected from metal when light frequency exceeds a threshold f_0 regardless of intensity. Below f_0, no electrons are emitted no matter how bright the light. Einstein explained this with photons of energy E = hf — each photon can eject at most one electron. The electron's kinetic energy K_max = hf - phi where phi is the work function (binding energy) of the metal. This directly supports the quantum hypothesis — energy is quantized in discrete packets.

Quantum mechanics postulates: (1) The state of a system is represented by a unit vector |psi> in a complex Hilbert space. (2) Observables correspond to Hermitian operators. (3) Measurement outcomes are eigenvalues of the corresponding operator, with probability |<eigenvalue|psi>|^2. (4) After measurement, the state collapses to the corresponding eigenstate (the measurement postulate). (5) Time evolution is unitary: i*hbar d|psi>/dt = H|psi> (Schrodinger equation).

Heisenberg's uncertainty principle: sigma_x * sigma_p >= hbar/2, where sigma_x and sigma_p are standard deviations of position and momentum. This is not a limitation of measurement technology but a fundamental property of quantum states — position and momentum eigenstates are incompatible. More generally, for any two observables A and B, sigma_A * sigma_B >= |<[A,B]>|/2 where [A,B] = AB - BA is the commutator. The Pauli exclusion principle follows from antisymmetrization of fermion wavefunctions.

## Chemistry

Chemical equilibrium and Le Chatelier's principle: if a system at equilibrium is disturbed (change in concentration, temperature, pressure), the equilibrium shifts to partially counteract the disturbance. For exothermic reactions (Delta H < 0), increasing temperature shifts equilibrium toward reactants. For reactions where number of gas molecules decreases, increasing pressure shifts toward products. The equilibrium constant K = exp(-Delta G^o / RT) relates standard Gibbs free energy change to equilibrium position.

Enzyme kinetics — Michaelis-Menten model: E + S <-> ES -> E + P. The steady-state approximation (d[ES]/dt = 0) yields reaction rate v = V_max[S] / (K_M + [S]) where V_max = k_cat[E]_total and K_M = (k_{-1} + k_cat)/k_1. K_M is the substrate concentration at half-maximum velocity. Competitive inhibition: inhibitor binds to active site, effectively increases K_M (needs more substrate to overcome), V_max unchanged. Noncompetitive inhibition: inhibitor binds elsewhere, reduces functional enzyme — V_max decreases, K_M unchanged.

## Biology

The central dogma of molecular biology: DNA -> RNA -> Protein. DNA replication is semiconservative — each strand serves as template for a new complementary strand (Meselson-Stahl experiment confirmed this with N-14/N-15 isotope labeling). Transcription: RNA polymerase synthesizes mRNA complementary to the template DNA strand. In eukaryotes, pre-mRNA undergoes splicing (introns removed, exons joined via spliceosome), 5' capping (7-methylguanosine), and 3' polyadenylation.

Translation: ribosomes read mRNA codons (triplets of bases) and match them to tRNA anticodons carrying specific amino acids. The genetic code is degenerate — most amino acids are encoded by multiple codons (usually differing in the third base, the "wobble" position). Start codon AUG (methionine), stop codons UAA, UAG, UGA. Protein folding is driven by hydrophobic collapse, hydrogen bonding, and van der Waals interactions — the folded state is a local free energy minimum in the complex energy landscape (Levinthal's paradox: random search would take longer than the age of the universe; folding follows funnel-shaped energy landscapes).
"""


def _gen_en_humanities() -> str:
    return """
# Humanities and Social Sciences

## Philosophy

Epistemology — the study of knowledge. The JTB (Justified True Belief) account of knowledge held that knowing P requires: (1) P is true, (2) S believes P, (3) S is justified in believing P. Gettier cases challenge this: scenarios where justified true belief seems not to constitute knowledge (e.g., S believes "the person who gets the job has 10 coins in their pocket" based on strong but misleading evidence — the belief happens to be true because S themselves gets the job, but S didn't know about their own coins). Post-Gettier theories add a fourth condition: no defeaters, reliable causal chain, or tracking the truth across possible worlds.

Ethics frameworks: Utilitarianism (Bentham, Mill) evaluates actions by consequences — maximize overall happiness/well-being. Act utilitarianism judges individual acts; rule utilitarianism judges rules that, if universally followed, would maximize utility. Deontology (Kant) evaluates actions by their adherence to moral rules/duties regardless of consequences — the Categorical Imperative: "Act only according to that maxim by which you can at the same time will that it should become a universal law." The second formulation: "Act so that you treat humanity, whether in your own person or in that of another, always as an end and never as a means only."

Virtue ethics (Aristotle) focuses on character rather than actions or consequences. Virtues are character traits that enable flourishing (eudaimonia): courage (mean between cowardice and recklessness), temperance, justice, practical wisdom (phronesis). A virtuous person reliably perceives what is morally relevant and acts accordingly — ethics is cultivated through habituation and practice, like learning a craft.

## Economics

Market efficiency: the First Welfare Theorem states that competitive equilibrium is Pareto efficient — no one can be made better off without making someone else worse off (under perfect competition, no externalities, complete markets). The Second Welfare Theorem: any Pareto efficient allocation can be achieved as a competitive equilibrium given appropriate lump-sum transfers (separating efficiency from equity). Market failures: externalities (costs/benefits not reflected in prices — pollution, education), public goods (non-rival, non-excludable — national defense, basic research), asymmetric information (adverse selection — Akerlof's "Market for Lemons"; moral hazard — insurance changes behavior).

Game theory: Nash equilibrium — a strategy profile where no player can improve their payoff by unilaterally changing strategy. Pure strategy equilibrium may not exist (matching pennies); mixed strategy equilibrium always exists in finite games (Nash's existence theorem). Prisoner's dilemma: individually rational strategies (defect) lead to collectively worse outcome than cooperation. Pareto optimality in games differs from equilibrium. Subgame perfect equilibrium (backward induction) refines Nash equilibrium by requiring rationality at every decision node in sequential games.

## History

The Scientific Revolution (16th-17th centuries) transformed understanding through systematic observation, experimentation, and mathematical description. Key shifts: Copernicus proposed heliocentric model (1543) — the Sun, not Earth, is the center; Galileo's telescopic observations (1609) supported Copernican theory (moons of Jupiter, phases of Venus, lunar mountains); Kepler's elliptical orbits replaced circular orbits with three empirical laws; Newton's Principia (1687) unified terrestrial and celestial mechanics under universal gravitation and three laws of motion.

The Industrial Revolution (late 18th-19th centuries) transformed production, transportation, and social organization. Key technologies: steam engine (Watt's separate condenser, 1769) converted thermal to mechanical energy efficiently; mechanized textile production (spinning jenny, power loom); railroads and steamships revolutionized transportation. Social consequences: urbanization, factory system replacing domestic industry, new class structure (industrial bourgeoisie, proletariat), labor movements, reforms addressing working conditions and child labor.
"""


def _gen_en_technology() -> str:
    return """
# Modern Technology

## Distributed Systems

CAP theorem (Brewer's conjecture, Gilbert-Lynch proof): a distributed data store can simultaneously provide at most two of: Consistency (all nodes see the same data at the same time), Availability (every request receives a non-error response), Partition tolerance (system continues operating despite network partitions). Since partitions are inevitable in distributed systems, the practical choice is CP (sacrifice availability) or AP (sacrifice strong consistency). Real systems offer tunable consistency — DynamoDB allows choosing consistency level per request.

Consensus algorithms enable replicated state machines to agree on a sequence of commands. Paxos uses prepare-promise and accept-accepted phases with majority quorums: proposer sends prepare(n) — acceptors promise not to accept proposals <n, respond with highest accepted proposal <n — proposer sends accept(n, value) using the value from the highest previous proposal or its own. Raft decomposes consensus into leader election (randomized timeouts, RequestVote RPC), log replication (AppendEntries RPC, leader forces followers' logs to match), and safety (leader can only commit entries from its current term). Raft's explicit leader and log matching rules make it easier to understand and implement than Paxos.

## Cryptography

Public-key cryptography: RSA generates two large primes p,q, computes n=pq, phi(n)=(p-1)(q-1). Select public exponent e coprime to phi(n) (commonly 65537). Compute private key d = e^{-1} mod phi(n). Encryption: c = m^e mod n. Decryption: m = c^d mod n. Security relies on the hardness of factoring n = p*q. If large-scale quantum computers capable of running Shor's algorithm exist, RSA is broken (integer factorization in polynomial time). Key size recommendations: 2048-bit minimum, 4096-bit for long-term security.

Elliptic curve cryptography (ECC) provides equivalent security with smaller keys. An elliptic curve over finite field F_p is the set of points (x,y) satisfying y^2 = x^3 + ax + b (mod p) plus point at infinity. Point addition forms an abelian group. Scalar multiplication Q = kP = P + P + ... + P (k times) is easy; the discrete log problem (finding k given P and Q) is computationally hard. ECDH key exchange: Alice computes k_A * Bob's_public, Bob computes k_B * Alice's_public — both get k_A*k_B*G as shared secret. Ed25519 uses twisted Edwards curve providing high performance and resistance to timing side-channel attacks.

## Web Architecture

REST (Representational State Transfer) architectural constraints: Client-Server (separation of concerns), Stateless (each request contains all necessary information, no server-side session state), Cacheable (responses explicitly mark themselves cacheable or not), Uniform Interface (resource identification through URIs, manipulation through representations, self-descriptive messages, hypermedia as engine of application state HATEOAS), Layered System (intermediaries like proxies, gateways transparent to clients), Code-on-Demand (optional — server can extend client functionality by transferring executable code).

HTTP semantics: GET is safe (no side effects) and idempotent; POST creates resources (not idempotent); PUT replaces resource (idempotent); DELETE removes resource (idempotent); PATCH partially updates (not necessarily idempotent). Status codes: 2xx success, 3xx redirection, 4xx client error, 5xx server error. Content negotiation via Accept/Content-Type headers. Caching via ETag (entity tag for version comparison), Last-Modified, Cache-Control directives (max-age, no-cache, no-store, public/private).
"""


def _gen_en_writing() -> str:
    return """
# Writing and Communication

## Technical Writing

Effective technical documentation follows the principle of progressive disclosure — present the simplest thing that works first, then layer on complexity. A good getting-started guide: (1) state prerequisites clearly with exact versions, (2) show a minimal working example the user can copy-paste-run in under 5 minutes, (3) explain what happened and why, (4) point to next steps and deeper documentation. Error messages should diagnose the problem ("Expected a string, got integer 5"), explain why it's wrong ("This API requires names as text"), and suggest a fix ("Convert to string with str() first").

API documentation structure: Overview (what this API does, when to use it), Authentication (how to get credentials), Endpoints (grouped by resource, each with HTTP method, path, parameters table, request/response examples, error codes), Rate Limits, SDK examples, Changelog. Code samples must be copy-paste runnable — include imports, setup, error handling, cleanup. Write descriptions in active voice present tense: "Returns a list of users" not "A list of users will be returned."

## Essay Structure

The classical five-paragraph essay structure generalizes to most persuasive writing: Introduction (hook + thesis statement + roadmap), Body paragraphs (claim + evidence + analysis + transition), Conclusion (restate thesis + synthesize key points + broader implications). Each paragraph should develop exactly one idea. Topic sentences state the paragraph's claim; supporting sentences provide evidence and reasoning; concluding sentences link back to the thesis or forward to the next paragraph.

The thesis statement is the essay's central claim — specific, arguable, and scope-defined. Weak: "Social media has effects on society." Strong: "Instagram's algorithmic amplification of extreme fitness content disproportionately increases body image dissatisfaction among adolescent females compared to male peers, as demonstrated by three longitudinal studies (2019-2024)." Coherence devices: transitional phrases (furthermore, however, consequently, in contrast), parallel structure (repeating grammatical patterns across sentences), lexical cohesion (reusing key terms rather than varying for thesaurus-driven elegance).

## Persuasive Speaking

Aristotle's three modes of persuasion: Ethos (credibility — demonstrate expertise, acknowledge counterarguments, show good character), Pathos (emotional appeal — use vivid imagery, tell stories, connect to audience's values and fears), Logos (logical argument — use data, statistics, causal reasoning, deductive and inductive logic). The best persuasion weaves all three: an ethos-built frame, a logos-structured argument, a pathos-delivered close.

Story structure for presentations: The Hero's Journey adapted for business communication — (1) The Ordinary World: current state and its limitations, (2) The Call to Adventure: the opportunity or problem, (3) The Ordeal: challenges and obstacles, (4) The Reward: proposed solution and its benefits, (5) The Return: implementation plan and call to action. Keep slides visual — one idea per slide, images over bullet points, data visualizations over tables. The 10/20/30 rule (Kawasaki): 10 slides, 20 minutes, 30-point minimum font size.
"""


def _gen_en_business() -> str:
    return """
# Business and Entrepreneurship

## Strategy

Porter's Five Forces analyze industry competition: (1) Threat of New Entrants — barriers to entry (capital requirements, economies of scale, brand loyalty, regulation, access to distribution). High barriers protect incumbent profitability. (2) Bargaining Power of Suppliers — concentrated suppliers, few substitutes, high switching costs, ability to forward-integrate all increase supplier power. (3) Bargaining Power of Buyers — concentrated buyers, standardized products, low switching costs, ability to backward-integrate increase buyer power. (4) Threat of Substitutes — products from different industries that satisfy the same need (videoconferencing substitutes for business travel). (5) Rivalry Among Existing Competitors — many competitors, slow industry growth, high exit barriers, low differentiation all increase rivalry intensity.

Blue Ocean Strategy (Kim & Mauborgne): Instead of competing in crowded "red oceans" (existing market boundaries, beat competition, exploit existing demand), create "blue oceans" (uncontested market space, make competition irrelevant, create new demand). The strategy canvas maps key competing factors and their offering levels across competitors. Value innovation — simultaneous pursuit of differentiation and low cost by eliminating/reducing factors the industry competes on while raising/creating factors the industry has never offered. Example: Cirque du Soleil eliminated animal acts and star performers (high cost, limited appeal), reduced thrill/ danger, raised artistic elements and unique venue, created theatrical storyline — competing in neither circus nor theater industries but creating a new entertainment category.

## Product Management

The product development lifecycle: Discovery (identify user problems through interviews, analytics, support tickets), Validation (test whether solving this problem drives business outcomes — build prototypes, run experiments, measure), Delivery (build the solution incrementally, ship frequently, gather feedback), Growth (optimize adoption, engagement, retention, monetization). The MVP (Minimum Viable Product) tests the riskiest assumption with the least effort — not "the smallest product we can ship" but "the fastest way to learn whether we should build this." Types of MVPs: concierge (manually deliver the service), Wizard of Oz (users think it's automated but humans are behind the scenes), single-feature (do one thing well), landing page (gauge interest before building).

OKRs (Objectives and Key Results): Objectives are qualitative, inspirational, time-bound goals (what we want to achieve). Key Results are quantitative, measurable outcomes (how we know we're achieving it). Good KRs are specific, measurable, have a target and current baseline. Example: Objective: "Deliver a best-in-class onboarding experience." KR1: "Increase new user activation rate (completed 3+ actions in first session) from 45% to 70%." KR2: "Reduce time-to-first-value from 12 minutes to under 4 minutes." KR3: "Achieve NPS >= 40 from users in their first 30 days." OKRs are cascaded (company -> team -> individual) but should involve bottom-up input — teams should set ~60% of their OKRs based on what they believe matters most.

## Leadership

Situational Leadership (Hersey-Blanchard): leadership style should match the follower's development level on each specific task. Development level D1 (low competence, high commitment) -> Telling/Directing style (high task direction, low relationship). D2 (some competence, low commitment) -> Selling/Coaching (high task, high relationship). D3 (moderate-high competence, variable commitment) -> Participating/Supporting (low task, high relationship). D4 (high competence, high commitment) -> Delegating (low task, low relationship). Leaders must diagnose development level per task — the same person may be D4 on technical work and D1 on giving presentations.
"""


# ---------------------------------------------------------------------------
# Loss plateau detection
# ---------------------------------------------------------------------------

class PlateauDetector:
    def __init__(self, patience: int = 4, min_improvement: float = 0.02):
        self.patience = patience
        self.min_improvement = min_improvement
        self.losses: List[float] = []
        self._last_avg: float | None = None
        self._plateau_checks = 0

    def record(self, loss: float) -> bool:
        """Return True if plateau detected (window-average improvement < min_improvement)."""
        self.losses.append(loss)
        needed = self.patience * 2
        if len(self.losses) > needed:
            self.losses = self.losses[-needed:]
        if len(self.losses) < needed:
            return False  # not enough data yet
        # Compare avg of recent half vs older half
        mid = self.patience
        old_avg = sum(self.losses[:mid]) / mid
        new_avg = sum(self.losses[mid:]) / mid
        improvement = old_avg - new_avg  # positive = getting better
        if improvement >= self.min_improvement:
            self._plateau_checks = 0
            return False
        self._plateau_checks += 1
        return self._plateau_checks >= self.patience


# ---------------------------------------------------------------------------
# Inline corpus expander (called from training loop, no separate process needed)
# ---------------------------------------------------------------------------

class CorpusExpander:
    """内联语料扩展器 — 从训练循环中直接调用，无需独立 watchdog 进程。"""

    def __init__(self, root: Path, patience: int = 4, min_improvement: float = 0.02,
                 max_additions: int = 100):
        self.root = root
        self.detector = PlateauDetector(patience=patience, min_improvement=min_improvement)
        self.additions = 0
        self.max_additions = max_additions

    def check_and_expand(self, loss: float, ramp_iter: int = 0) -> bool:
        """记录 loss，若检测到平台期则自动扩展语料。返回 True 表示已扩展。"""
        if self.additions >= self.max_additions:
            return False
        if not self.detector.record(loss):
            return False
        domain_name, gen_func = _next_domain()
        content = gen_func()
        new_size = _append_corpus(self.root, content)
        self.additions += 1
        print(f"[corpus-expand] 平台期检测 iter={ramp_iter} loss={loss:.4f} "
              f"→ 已添加 {domain_name} (+{len(content)}字符, 语料={new_size/1024:.0f}KB "
              f"{self.additions}/{self.max_additions})", flush=True)
        self.detector = PlateauDetector(
            patience=self.detector.patience,
            min_improvement=self.detector.min_improvement,
        )
        return True


# ---------------------------------------------------------------------------
# Main supervisor loop
# ---------------------------------------------------------------------------

def _read_heartbeat(root: Path) -> Optional[dict]:
    hb = root / "weights" / ".train_heartbeat.json"
    if not hb.is_file():
        return None
    try:
        with open(hb, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _append_corpus(root: Path, content: str) -> int:
    corpus = root / "weights" / "training_corpus.txt"
    with open(corpus, "a", encoding="utf-8") as f:
        f.write(content)
    return corpus.stat().st_size


def main():
    root = ROOT
    patience = int(os.environ.get("LIFERS_AUTO_EXPAND_PLATEAU_N", "4"))
    min_imp = float(os.environ.get("LIFERS_AUTO_EXPAND_MIN_IMPROV", "0.02"))
    kb_per_add = int(os.environ.get("LIFERS_AUTO_EXPAND_KB_PER_ADD", "30"))
    max_adds = int(os.environ.get("LIFERS_AUTO_EXPAND_MAX_ADDITIONS", "100"))

    print(f"[auto-expand] watching {root}/weights/.train_heartbeat.json")
    print(f"[auto-expand] plateau: patience={patience} min_improvement={min_imp}")
    print(f"[auto-expand] will add ~{kb_per_add}KB per expansion, max {max_adds} times")

    detector = PlateauDetector(patience=patience, min_improvement=min_imp)
    last_hb_ts = None
    last_hb_time = time.time()
    last_pid = None
    same_ts_count = 0
    additions = 0

    while additions < max_adds:
        hb = _read_heartbeat(root)
        if hb is None:
            # Check staleness
            stale_sec = time.time() - last_hb_time
            if stale_sec > 600 and last_hb_time > 0:
                print(f"[auto-expand] ALERT: No heartbeat for {stale_sec:.0f}s (>10min) — training may be dead",
                      flush=True)
                last_hb_time = time.time()  # prevent spam
            time.sleep(30)
            continue

        last_hb_time = time.time()
        ts = hb.get("ts", "")
        current_pid = hb.get("pid", 0)

        # Detect PID change (training restarted)
        if last_pid and current_pid != last_pid:
            print(f"[auto-expand] Training restarted (PID {last_pid} -> {current_pid}) — resetting detector",
                  flush=True)
            detector = PlateauDetector(patience=patience, min_improvement=min_imp)
            same_ts_count = 0
        last_pid = current_pid

        if ts == last_hb_ts:
            same_ts_count += 1
            if same_ts_count > 10:
                print(f"[auto-expand] WARNING: Heartbeat timestamp unchanged for {same_ts_count} checks — training may be stuck",
                      flush=True)
                same_ts_count = 0  # prevent spam
            time.sleep(30)
            continue
        same_ts_count = 0

        last_hb_ts = ts
        loss = hb.get("loss", float("inf"))
        ramp_iter = hb.get("ramp_iter", 0)

        if detector.record(loss):
            domain_name, gen_func = _next_domain()
            content = gen_func()
            new_size = _append_corpus(root, content)
            additions += 1
            print(f"[auto-expand] PLATEAU detected (loss={loss:.4f}, iter={ramp_iter})")
            print(f"[auto-expand] added domain: {domain_name} (+{len(content)} chars)")
            print(f"[auto-expand] corpus now {new_size/1024:.0f} KB ({additions}/{max_adds} additions)")
            # Reset detector after expansion
            detector = PlateauDetector(patience=patience, min_improvement=min_imp)
        time.sleep(60)

    print(f"[auto-expand] reached max additions ({max_adds}) — exiting")


if __name__ == "__main__":
    main()
