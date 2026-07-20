# 架构是否困在 Adam 的局部最优?—— Orabona conjecture 的后续检验

> 起点:Francesco Orabona, *Neural Networks (Maybe) Evolved to Make Adam The Best Optimizer* (2020)
> 问题:在 Muon 等强非-Adam 优化器出现后,神经网络架构在多大程度上是 Adam/AdamW 下的局部最优?
>
> **图例约定**:结论后标注 `[确证]`(有大规模实证)/ `[增强]`(证据比 2020 更强,但结论向"更强的耦合"演化)/ `[猜想]`(机制假设或证据有限)。

---

## 1. 背景与动机

Orabona 的原始论断不是"Adam 是最优优化器",而是一个**幸存者偏差(survivorship bias)+ 协同演化(co-evolution)** 的元论断:

- 深度学习社区像一个巨大的遗传算法,半随机地探索架构空间,保留 work 的、丢弃不 work 的。
- 但探索时**优化器几乎恒定为 Adam**(因为 Adam 是"默认优化器")。
- 因此:那些"需要同时设计新架构 + 新优化器"的方向被**过早丢弃**了——因为同时设计两者是极难的任务。
- 推论:我们看到的架构谱系,是被 Adam 隐式塑造的一个子空间,而非"优化器无关"的全局最优。

原博客给的唯一证据是**间接**的:Adam 在非-神经网络的简单凸/非凸问题上表现很差(引 Vaswani et al., 2019),暗示 Adam 的"魔力"依赖于神经网络特有的初始化、权重尺度、损失函数等。

**为什么现在能重新检验**:2024 年底 Muon 出现,并在 2025 年被大规模采用。这是第一个足够强、被工业界真正使用的非-Adam 优化器,提供了做"反事实实验"的抓手——我们终于可以问:换一个优化器,哪些"架构性质"会消失?

---

## 2. 问题在领域中的定位

这个问题横跨三条正在收敛的研究线:

| 研究线 | 核心关切 | 代表工作 |
|---|---|---|
| **架构-优化器耦合的实证测绘** | 哪些架构和哪些优化器"绑定"? | BOCB (2410.06373);Nested Learning 续作 |
| **Muon 机制理解** | Muon 为何/如何优于 Adam?找到的解有何不同? | OSP (2506.19697);QK-Clip 分析;Muon 尾部联想记忆 (2509.26030) |
| **联合搜索(architecture × optimizer)** | 能否同时演化架构和优化器? | CliffSearch (2604.01210);evolutionary optimizer search |

关键定位判断:**前两条线已有丰富成果,第三条(也是 Orabona 论断最核心的反事实)几乎空白。** 这决定了下面结论的确证程度分布。

---

## 3. 核心内容:简单数学 + 直觉

### 3.1 前提本身已被证伪:Adam 不是终点 `[确证]`

Orabona 反问"我们几年前就到达优化顶峰,这有多可能?"——Muon 直接回答了。

- FLOP 效率:16B 参数、5.7T token 上,Muon 只需约 **52%** 的 FLOPs 达到 AdamW 同等性能。
- 规模验证:已用于 32B 激活 / 1T 总参数的前沿模型(Kimi K2)。
- 内存:Muon 只维护一阶动量,比 AdamW 更省。

**Muon 更新规则**(矩阵参数 $W_t \in \mathbb{R}^{m \times n}$):

$$
M_t = \beta M_{t-1} + G_t, \qquad O_t = \mathrm{NS}(M_t), \qquad W_{t+1} = W_t - \eta\, O_t
$$

符号定义:
- $G_t \in \mathbb{R}^{m \times n}$:第 $t$ 步的梯度矩阵,形状同 $W_t$。
- $M_t \in \mathbb{R}^{m \times n}$:动量缓冲(一阶),$\beta$ 为动量系数(标量,典型 0.95)。
- $\mathrm{NS}(\cdot)$:Newton-Schulz 迭代,近似把输入矩阵映射到最近的**半正交矩阵**(即让其奇异值全部趋近 1)。这是 Muon 的核心——它把梯度替换为"归一化正交因子之和"。
- $\eta$:学习率(标量);$O_t$:正交化后的更新方向。

**直觉**:Adam 把每个参数当作独立标量做 per-coordinate 自适应缩放;Muon 承认权重是**有几何结构的矩阵**,做谱范数下的最速下降(steepest descent under spectral norm),即在保持 operator norm 有界的前提下最大化损失下降。实践中 Muon 只用于隐藏层矩阵,embedding/norm/bias 仍用 Adam(常称 MuonAdam)。

### 3.2 最强证据:activation outlier 是 Adam 的指纹,不是 transformer 的性质 `[确证]`

这是最直接验证 Orabona 论断的一例。长期以来 activation outliers(异常大的激活通道)被当作大型 transformer 的**内在性质**,是低比特量化困难的根源。

**OSP (Outlier-Safe Pre-Training) 的发现**:把 Adam 换成 Muon + Single-Scale RMSNorm + 可学习 embedding projection,可以从源头**阻止** outlier 产生。

- 诊断:softmax 只依赖 logit 的相对差,模型可以不把 logit 推向 $-\infty$ 就实现"no-op"。容易形成 outlier 的模型倾向于采用"推向负无穷"策略,而这只在**偏好集中式通道激活**的训练动力学下发生——这种动力学正是 Adam 的 per-coordinate 自适应带来的。
- 结果:1.4B 模型 / 1T token,相比 Adam 仅 2% 开销,量化鲁棒性从根本上改善。

**论断映射**:这是 Orabona 论点的一个精确镜像——不是"架构迁就 Adam",而是**我们把 Adam 造成的 artifact 误认为架构的性质**。

伏笔工作(2405.19279)方向一致:大的**对角**自适应学习率是 outlier 出现的关键;非对角 preconditioner(SOAP、Shampoo)能同时减少 outlier 并改善收敛。

### 3.3 双向耦合:BOCB 把论断从"单向"升级为"根本纠缠" `[增强]`

**BOCB (Backbone-Optimizer Coupling Bias)** 建了 20 backbone × 20 optimizer 的 benchmark(CIFAR-100 / ImageNet-1K / COCO):

- 经典 CNN(VGG、ResNet)⟷ SGD 家族共依赖;现代架构(ViT、ConvNeXt)⟷ 自适应学习率优化器紧密耦合。
- **因果方向是双向的**:BOCB 既可由优化器引入,也可由某些 backbone 设计引入。
- 代价:强 BOCB 导致额外调参成本 + 更差泛化,限制预训练模型的实际部署。
- 机制:耦合可能源于架构演化带来的优化复杂化——现代 backbone 的复杂 token-mixer 和 block-wise 异构结构塑造了更复杂的优化地形。

**Nested Learning 续作**把架构和优化器都建模为嵌套联想记忆系统,论证其协同依赖是根本性而非偶然。它的第一个核心问题几乎是本主题的学术改写:*Transformer 与 Adam/AdamW 的配对是任意的,还是反映底层结构-动力学兼容性?*

**升级点**:Orabona 说"架构迁就优化器"(单向);BOCB/NL 说"两者互相塑造到无法独立设计"(双向)。这比原论断更强。

### 3.4 微妙之处:稳定性补丁在 Muon 下**改变**而非消失 `[增强]` / `[猜想]`

与你的 Qwen3 QK-LayerNorm、DeepSeek MLA 研究直接相关。QK-norm / logit soft-capping / σReparam / QK-clip 这一整套技巧,都是 Adam 时代为解决 attention logit 爆炸发明的。自然猜想"换 Muon 就不需要了"——但证据**混合甚至反向**。

**Logit 爆炸的数学根源**(Cauchy-Schwarz):

$$
|q_i \cdot k_j| \le \|q_i\|\,\|k_j\| = \|x_i W_q\|\,\|x_j W_k\| \le \|x_i\|\,\|x_j\|\,\|W_q\|\,\|W_k\|
$$

符号定义:
- $q_i = x_i W_q$、$k_j = x_j W_k$:第 $i$/$j$ 个 token 的 query/key 向量($x_i \in \mathbb{R}^{d_{\text{model}}}$ 为输入,$W_q, W_k \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$ 为投影矩阵)。
- $\|\cdot\|$:向量取 $\ell_2$ 范数,矩阵取谱范数(最大奇异值)。

由于 $x$ 通常经过 RMSNorm,$\|x_i\|\|x_j\|$ 不会爆炸,所以 **MaxLogit 爆炸 ⟺ $\|W_q\|$、$\|W_k\|$ 的谱范数无界增长**。这正是 MaxLogit 作为"outlier 指标"的含义。

**Kimi 的关键观察(反直觉)**:Muon 反而**加剧** logit 爆炸风险——它的**全秩更新**增加了与已有奇异方向对齐(碰撞)的概率,而 Adam 的更新更"低秩"。虽然碰撞是低概率事件(所以即使 Muon 下也只有少数 head 爆炸),但相对风险更高。这正是他们发明 **MuonClip** 的动机。

**AdamW/MHA 的分歧**:AdaMuon 报告在标准 MHA 下**没有**观察到过大 logit,并归因于架构——先前研究用的是 **MLA**,其结构设计本身放大 logit 幅度。

**含义**:换优化器不会自动消除稳定性技巧的需求,只是改变了需要**哪种**。而"MLA 在不同优化器下稳定性行为不同"本身就是架构 × 优化器**不可分离**的又一证据。方向仍支持 Orabona,只是走向"互相纠缠"的更强结论。

### 3.5 Muon 收敛到"另一个盆地" `[确证]`

不是"同一最优点、不同路径",而是**性质不同的解**:

- Muon 解的 outlier 通道激活数量**更少**(→ 量化误差更小,压缩后精度更高)。
- Muon 权重的**奇异值熵(≈ 有效秩)** 与 Adam 不同。

---

## 4. 未解决 / 开放问题

1. **核心反事实几乎无人做** `[空白]`:"如果整个 transformer 谱系(pre-norm、QK-norm、初始化尺度、residual 缩放……)从零在 Muon 下演化,会长成什么样?" 现有工作全是"固定架构换优化器"或"固定优化器换架构",**没有真正的联合搜索**。这恰恰验证了 Orabona 说的"同时设计架构+优化器极难"——CliffSearch 也只能冻结架构、单独演化优化器规则。

2. **QK-norm 之于 Muon 的必要性无定论** `[矛盾]`:Kimi 说 Muon 更需要 clip,AdaMuon 说标准 MHA 下没问题,分歧点集中在 **MLA**。需要在受控设置下系统 ablation。

3. **"大问题 vs 小问题"是否本质不同**:Orabona 本人质疑"小优化问题与大问题有根本不同特性"这一常见信念缺乏真正证据。样本量增加甚至可能**增加**目标函数的光滑性,让优化更易——这个假设至今未被严格检验。

4. **幸存者偏差的不可证伪性**:这是论点最漂亮也最棘手之处——survivorship bias 的本质就是**你看不到那些没被生出来的架构**。即使联合搜索也只能探索被搜索空间覆盖的部分,无法排除"存在某个用拟牛顿法能线性收敛的架构,但社区因太聚焦于一阶方法而永远找不到它"(Orabona 在评论区的原话)。

---

## 5. 参考来源

**直接检验 conjecture:**
- Li et al., *Unveiling the Backbone-Optimizer Coupling Bias in Visual Representation Learning*, arXiv:2410.06373 (BOCB, ICLR)。最系统的正面 benchmark。
- Tian/Li/Wang et al., *Backbone-Optimizer Coupling Bias: The Hidden Co-Design Principle* (HuggingFace blog, 2025-12) + Nested Learning 续作。双向耦合的理论化。

**Muon 机制 / 最强证据:**
- *Outlier-Safe Pre-Training*, arXiv:2506.19697。最干净的因果证据(outlier = Adam 产物)。
- Soket AI, *QK-Clip: Taking Muon one step further* + Kimi K2 报告。Muon 如何改变而非消除稳定性需求。
- *Muon Outperforms Adam in Tail-End Associative Memory Learning*, arXiv:2509.26030。
- Anson & Aitchison, *Controlling changes to attention logits*, arXiv:2511.21377(µP-inspired,MLA 兼容的替代方案)。

**Muon 规模化证据:**
- Essential AI, *Practical Efficiency of Muon for Pretraining*, arXiv:2505.02222(52% FLOPs;muP 校准)。
- Moonshot, *Muon is Scalable for LLM Training* (Moonlight 16B)。

**联合搜索(仍空白):**
- CliffSearch, arXiv:2604.01210(冻结架构、只演化优化器)。
- Marfinetz, *Evolving Deep Learning Optimizers*, arXiv:2512.11853。

**outlier 伏笔:**
- *Understanding and Minimising Outlier Features in Transformer Training*, arXiv:2405.19279(对角 vs 非对角 preconditioner)。

**原始出处:**
- Orabona, *Neural Networks (Maybe) Evolved to Make Adam The Best Optimizer*, parameterfree.com (2020-12)。

---

*注:本主题与你正在做的 Qwen3 entropy/KL 爆炸诊断(QK-LayerNorm)、MLA 深研究属于同一条线。§3.4 的 logit 爆炸机制与你在 GRPO 训练中观察到的稳定性问题共享同一套谱范数直觉。*