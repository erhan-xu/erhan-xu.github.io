# 大模型优化领域全景 Landscape

> 一份把苏剑林《让炼丹更科学一些》系列锚定进当前大模型优化研究版图的地图。
> 所有 arXiv/blog 链接均经独立 web 检索确认存在；凡属推断（谁引谁、谁属同一条线）之处均已标注。
> 记号约定：$\theta\in\mathbb{R}^N$ 参数，$L(\theta)$ 总体损失，$\eta_t$ 第 $t$ 步学习率，$G$ 梯度范数上界，$T$ 总步数，$D=\|\theta_1-\theta^*\|$ 初始点到最优点距离。
>
> **关于"发表形式"的一个注记**：这个领域一个耐人寻味的社会学现象是，最有影响力的两项工作都**不是传统论文**。Muon 只有一篇 blog + 一条 Twitter thread，作者 Keller Jordan 公开表示拒绝为 Muon 写论文（"与其发一篇大概率被埋没在 arXiv 的论文，不如继续老实做研究"，二作 Yuchen Jin 的话是"发论文 ≠ 影响力"），凭这篇 blog 进了 OpenAI，Muon 可能已用于 GPT-5 训练。苏神系列本身也只是 blog。这条"blog/thread 而非 paper"的路线，和 DeepSeek 开源的逻辑一样，指向同一件事：在快速迭代的 AI 领域，开放、社区共建、快速响应，可能比传统论文范式更有效。下面参考文献里凡是 blog/thread 的都已注明，不要误当 arXiv 论文去引。
>
> *本版修订（v2）：新增 §7 社区评价与价值光谱；补齐全部缺失链接；修正 Muon 为 blog 而非论文。*

---

## 0. 一句话地图

整个领域可以看成一张 $2\times 2$：**横轴**是"改什么"——改**学习率随时间的形状**（schedule）还是改**更新方向的几何**（optimizer / norm）；**纵轴**是"靠什么说话"——**凸优化的最坏情况理论**还是**大模型的经验 scaling law**。苏神系列坐落在左下（schedule × 凸理论），而当前前沿的活力主要在四个角之间的**桥**上：凸理论居然能预测 LLM 的 loss 曲线（Schaipp），schedule 与权重平均是同一个东西（Defazio / WSM），norm 的选择决定最优 scaling（Bernstein / Pethick / Muon），以及超参能否跨规模迁移（µP / weight decay）。

---

## 1. 背景与动机（为什么这件事重要）

训练一个大模型，真正能自由拧的旋钮其实很少：**学习率的峰值**、**学习率随时间怎么衰减**、**用哪个优化器**、**batch size**、**weight decay**。这几个旋钮里，学习率 schedule 是最玄学、也最贵的——一次全尺寸训练动辄几十上百万美元，schedule 选错就是纯烧钱。

历史上这些选择几乎全靠经验（cosine decay 到峰值的 10%，warmup 占 0.1–0.5% 步数），教科书理论则被认为"太理想、没用"（苏神系列开篇的原话吐槽）。**这一领域近两三年最大的转变**，就是发现经典凸优化理论——恰恰是苏神系列在讲的那套——**出人意料地能定量预测甚至指导真实 LLM 训练**。这从两头同时发生：理论端把 last-iterate / 无界域 / 线性衰减做干净了；经验端则用 scaling law 精确拟合 loss 曲线，反过来发现最优 schedule 长得和理论预言一样。

---

## 2. 领域定位（这张版图的四个坐标轴）

| 轴 | 问题 | 代表工作 |
|---|---|---|
| **A. Schedule 形状** | $\eta_t$ 随 $t$ 怎么变最优？ | 线性衰减 (Defazio 2023)、WSD (Hu 2024)、WSM (Tian 2025)、D2Z (Bergsma 2025) |
| **B. 收敛理论** | 最坏情况下能证明多快？上界 vs 下界 | Orabona 2020、Zamani–Glineur 2023、Agarwal 2010 (下界) |
| **C. Optimizer / norm 几何** | 更新方向该在哪个范数下取最速下降？ | Adam、Muon、Scion、Bernstein–Newhouse (modular norm) |
| **D. 超参迁移 / scaling** | 小模型调好的超参能否搬到大模型？ | µP (Yang 2021)、u-µP、weight decay 与 batch size 的 scaling law |
| **桥** | 理论↔经验为何吻合？ | Schaipp 2025 (凸理论预测 LLM loss)、FSL / Multi-Power-Law scaling |

苏神系列 = **轴 A + 轴 B 的左下角**（schedule 的凸理论），下面每一节展开一个轴，并标出苏神系列在其中的位置。

---

## 3. 核心内容：五条线 + 简单数学与直觉

### 线一：Schedule 的凸优化理论（苏神系列的正脉）

**核心不等式**（苏神系列反复用的，也是这条线的心脏）：对凸 $L$、梯度有界 $\|g_t\|\le G$、SGD 迭代，

$$
\sum_{t=1}^T \eta_t\,\mathbb{E}[L(\theta_t)-L(\varphi)] \;\le\; \frac{\|\theta_1-\varphi\|^2}{2} + \frac{G^2}{2}\sum_{t=1}^T \eta_t^2.
$$

这里 $\varphi$ 是任意比较点。**直觉**：左边是"你比 $\varphi$ 好多少"的加权累积，右边第一项是"起点离 $\varphi$ 有多远"（一次性预算），第二项是"每步噪声代价"的累积。整个证明就是 telescoping + 凸性，不需要有界域、不需要梯度无偏（详见你我前几轮讨论的"路线 A"）。

**关键结论演化**：
- **平均损失**（系列一）：$\frac1T\sum_t \mathbb{E}[L(\theta_t)-L(\theta^*)]$ 至多 $\mathcal{O}(\sqrt{\ln T/T})$。
- **无界域**（系列二）：去掉有界域 + 投影这根拐杖，右边用 $\|\theta_1-\varphi\|^2$ 而非 $R^2$。源头是 Orabona 的 blog。
- **终点损失**（系列三）：从平均损失转成 $\mathbb{E}[L(\theta_T)-L(\theta^*)]$，更贴合实践（大家用的是最后一步的权重，不是平均）。源头是 Orabona 那篇的思路。
- **线性衰减最优**（系列四/五/六）：最优 schedule 是 $\eta_t \propto 1-t/T$，达到 $\mathcal{O}(1/\sqrt{T})$，比常数、比 $1/t$、比 $1/\sqrt t$ 都好。源头是 Defazio 2023。

**为什么是线性衰减？** 最坏情况下，$1-t/T$ 恰好平衡了"早期要快速远离初始点"（大 LR）和"后期要压住梯度噪声"（小 LR）。这个 $1-t/T$ 的形状会在下面每一条线里反复出现——它是这个领域的一个"吸引子"。

### 线二：Last-iterate 与上界/下界（判断"最优"的标尺）

"最优"这个词只有配上**下界**才有意义。三块拼图：
- **上界**：Orabona (2020) 无界域 last-iterate；Zamani–Glineur (2023) 给出 subgradient 方法 last-iterate 的**精确**速率。
- **下界**：Agarwal–Bartlett–Ravikumar–Wainwright (2010) 的信息论下界——任何算法在 oracle 模型下都不可能更快。有了它，说线性衰减的 $\mathcal{O}(1/\sqrt T)$"最优"才站得住（= 碰到了下界）。
- **精确常数**：Defazio 2023 之所以能宣称 $1-t/T$ 最优，靠的是把上界的常数抠到与下界匹配（PEP / performance estimation 那套）。

**直觉**：last-iterate（最后一步）比 averaged-iterate（平均）难分析得多，因为平均自带平滑、噪声被抵消，而最后一步"记得"所有历史抖动。历史上大家先证平均（容易），近几年才把最后一步做干净——而实践恰恰用最后一步，所以这个 gap 的弥合直接有实用价值。

### 线三：Schedule ⟷ 权重平均的对偶（这条线最深刻）

**核心洞察**：一个学习率 schedule，等价于对历史更新（或历史 checkpoint）做某种**加权平均**。这是理解 D2Z、Schedule-Free、WSM 的总钥匙。

写出来：参数 $\theta_t = \theta_1 - \sum_{i<t}\eta_i g_i$，把不同 schedule 代入，就得到"第 $i$ 次更新对最终 $\theta_T$ 的贡献系数 $c_{t,i}$"。**LR 衰减得越猛，后期更新的权重越小**——这正是 D2Z 论文里 AdamW-as-EMA 的图 2 讲的事。

三种落地：
- **Schedule-Free (Defazio 2024, NeurIPS)**：干脆不用 schedule，用一种迭代平均隐式实现衰减，好处是不用预先知道 $T$。缺点：经验上略逊 WSD（Hägele 2024）。
- **WSM (Tian 2025)**：把"衰减≈平均"实现为**显式 checkpoint 合并**——训练时 LR 不衰减（decay-free），事后把最近几个 checkpoint 按理论权重合并，离线合成出"仿佛衰减过"的解。发现**merge duration（合并窗口）**是最关键因素。Ling-1T 万亿模型实际用了它。
- **Anytime / horizon-free + weight averaging**：把 schedule-free 的思路推广到任意训练时长。

**直觉**：cosine / linear decay 的末段"退火"，本质是在做 bias-variance 权衡的加权平均；既然如此，为何不直接平均？这就打通了"调度"和"平均"两个看似无关的传统。

### 线四：Optimizer 与 norm 几何（改方向而非改步长）

前三条线都在调 $\eta_t$（标量步长），这条线改的是**更新方向**。统一视角来自 Bernstein–Newhouse 的"steepest descent under a norm"：

$$
\Delta\theta = \arg\min_{\|\delta\|\le 1}\langle g, \delta\rangle,
$$

选不同的范数 $\|\cdot\|$ 就得到不同优化器。
- **SGD** = 欧氏范数下的最速下降。
- **Adam / Signum**（无 EMA 时）≈ $\ell_\infty$（逐坐标 max）下的最速下降。
- **Muon** (Jordan 2024) = 矩阵**谱范数**（RMS→RMS）下的最速下降，用 Newton-Schulz 迭代对梯度做正交化。经验上比 AdamW 快约 **2×**，已用到 1T 参数规模（Moonlight 16B MoE、Kimi 系列）。
- **Scion** (Pethick 2025) = 给每层指定各自的范数（输入/输出层用 $\ell_\infty$，隐藏层用谱范数），比 Muon 更好解释、可分层调 LR。
- **Gluon / modular norm**：把"每层一个范数"推广成模块化框架。

**Muon 为何更快的理论解释**（尚在争论，属 conjecture 与初步证明混合）：Shen et al. (2025) 证明 Muon 的收敛由**谱范数下的 Lipschitz 常数**控制，这个常数在 Hessian 低秩/块对角时可远小于欧氏对应，故更快。另一支（关联记忆视角）认为 Muon 在重尾/长尾知识上尤其占优。**要点**：谱范数 smoothness 比欧氏 smoothness 更贴合深度网络的真实几何。

**关键工程细节**（Liu 2025, "Muon is Scalable"）：Muon 要 scale 到大模型必须 (1) 加 weight decay，(2) 仔细调每参数更新幅度。Muon 通常只用于隐藏层，输入/输出层仍用 AdamW（混合）。

**与你研究的接口**：这条线是你 memory 里"流形上的最速下降"、Muon 谱范数、QK-Clip 那一串的正主。

### 线五：超参迁移与 scaling law（小模型调好搬到大模型）

**µP / µTransfer (Yang 2021)**：重参数化初始方差和分层 LR，使得**最优超参不随宽度变化**——在小 proxy 模型上调好，零样本搬到大模型。理论上对宽度（Yang 2022）和深度（Yang 2023）都有保证。GPT-4 报告引了 µP，Llama4 的 MetaP、Grok-2 配置都疑似用了类似技术。

**近期修正（重要）**：
- **Weight decay 可能比 µP 更关键**（arXiv 2510.19093, 2025）：µP 的几何对齐假设**只在训练最初几步成立**；之后真正稳住不同宽度更新动态的是 **independent weight decay**，不是 µP。这是对 µP 叙事的实质性修正。
- **u-µP (Blake 2025, ICLR)**：µP + unit scaling，更简单、默认超参更好、天然支持 FP8 低精度训练。
- **critical batch size (McCandlish 2018)**：若 proxy 模型的 batch size 小于临界值，LR 迁移到大 batch 会失准。µP 与 batch size 必须一起考虑。

**Scaling law for 超参本身**：
- **Power Lines (Bergsma 2025)**：weight decay 与 batch size 的 scaling law。
- **Scaling optimal LR across token horizons (Bjorck 2025, ICLR)**：最优 LR 如何随 token 数变。
- **DeepSeek / Hu (MiniCPM)**：把最优超参写成 compute / loss / 模型大小的函数。

### 桥：凸理论为何能预测 LLM（这是当前最激动人心的部分）

**Schaipp et al. (2025), "The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training"** (arXiv 2501.18965)：
- 把凸优化的 last-iterate bound（就是苏神系列那个 bound 的精细版）拿来，**直接拟合真实 LLM 的 loss 曲线**，形状吻合得出奇地好。
- 用理论预测的最优 schedule 和 base LR 去训 124M/210M Llama-style transformer，**真的改善了 pretraining**。
- 解释了 WSD 的 cooldown 分数、continual training 该怎么调——这些都能从凸理论推出来。
- 关键论断：LLM 虽非凸，但其 loss 动态"惊人地"贴合凸优化的收敛模式，尤其在线性衰减的有效性上。

**Scaling-law 侧的对应物**：
- **Functional Scaling Law, FSL (Li et al. 2025)**：在特征空间线性回归 + power-law 结构下，把 loss 写成 LR schedule 的**解析泛函**；经验上这个泛函对真实 LLM 也准。
- **Optimal LRS under FSL (arXiv 2602.06797, 2026)**：在 FSL 框架下推最优 schedule，发现由 source 指数 $s$ 和 capacity 指数 $\beta$ 两个指数控制，存在**相变**——不同 $(s,\beta)$ 区间最优 schedule 形状会突变（power decay ↔ WSD）。
- **Multi-Power Law (arXiv 2503.12811)**：跨 schedule 预测 loss 曲线的多幂律，反推出的最优 schedule 自发长成 WSD 的样子。
- **river valley 视角 (Wen 2025)**：猜想 loss landscape 有"河谷"结构，解释 WSD 为何有效。

**直觉**：为什么最坏情况凸理论能预测显然非凸的 LLM？一种解释是 LLM 训练的有效动态被少数几个"慢方向"（power-law 谱）主导，而这些方向的行为接近凸/二次——于是凸理论抓住了主项。这仍是**开放的解释性问题**（下节展开）。

---

## 4. 开放 / 未解问题（诚实区分 conjecture 与已证）

**已经比较扎实的**：
- 凸 + Lipschitz 下线性衰减 last-iterate 最优（Defazio 2023，配 Agarwal 2010 下界）——**已证**。
- Muon = 谱范数最速下降，且谱 smoothness 可小于欧氏——**已证**（Shen 2025）。
- µP 的宽度/深度迁移在其假设下**已证**（Yang 2022/2023）。

**仍是 conjecture 或经验观察**：
- **凸理论为何能预测非凸 LLM**：Schaipp 2025 是经验发现 + 部分理论，机制未彻底讲清。属当前最重要的开放问题之一。
- **Muon 相对 Adam 的优越性来源**：多个互相竞争的解释（谱 smoothness / 关联记忆 / MoE 谱结构），**尚无共识**。
- **weight decay vs µP 谁才是 LR 迁移的真因**：2510.19093 挑战了 µP 叙事，但这是新结果，尚待复现与辩论。
- **FSL 的相变边界**在真实 LLM 中的位置：理论预言了相变，但 $(s,\beta)$ 在真实模型中怎么测、边界在哪，未定。
- **schedule-free / WSM 与显式 schedule 的最终优劣**：经验证据互相打架（Hägele 说 schedule-free 逊于 WSD，WSM 说自己胜过 WSD），依赖设置。
- **batch size × LR × weight decay 的联合 scaling**：Smith–Le 的 SDE 视角只对 SGD，对 Adam/Muon 的联合 scaling **理论上仍很欠缺**（Pethick 2025 明说"data scaling 理论上 poorly understood"）。
- **非凸、随机性、重尾噪声下的 last-iterate**：绝大多数干净结论仍在凸 + 有界方差假设下，重尾噪声（LLM 梯度实测重尾）下的理论刚起步。

**你可能感兴趣的研究缺口**（结合你的 RL / alignment 背景）：
- 上述几乎全是 pretraining / supervised 的 schedule 理论；**RL post-training（GRPO/PPO）的 LR schedule 理论几乎空白**。on-policy、非平稳目标、KL 正则下，"线性衰减最优"还成不成立？这是一个开放且与你 memory 里 BASIS/GRPO 工作直接相关的方向。
- Schedule ⟷ 平均对偶在 RL 里的对应物（policy averaging？）是否存在类似定理？

---

## 5. 参考文献（全部经检索确认链接）

> **实验规模标注**（每条前缀）：基于对各文最大实验规模的逐篇核实。口径——【生产级】≥1B 参数且在真实 LLM 预训练验证；【中等】约 100M–1B，或 GPT-2 规模（NanoGPT 124M）级；【小/toy】<100M，或仅线性回归/二次型/CIFAR/合成数据；【理论】纯理论、无 LLM 实验。**只对直接以 LLM/大模型为主题的论文分档**，纯理论/toy 论文标【理论】或【小/toy】并附一句实验设定说明。规模是**经验排名类主张**（优化器 A 胜 B、schedule X 最优）的可信度前提，但**不**是理论/机理类论文的评判标准——读的时候分开看。两处规模未能精确核实的已标"待核实"。

### 苏神系列本身
- 让炼丹更科学一些（一）SGD平均损失收敛：https://kexue.fm/archives/9902
- （二）将结论推广到无界域：https://kexue.fm/archives/11469
- （三）SGD终点损失收敛：https://kexue.fm/archives/11480
- （四）新恒等式，新学习率：https://kexue.fm/archives/11494
- （五）基于梯度精调学习率 / （六）自上而下的精妙构造（archives 号未从检索片段确认，见站内目录；六篇正文明确"依然出自"Defazio 2023 那篇）
- （七）步长调度与权重平均：https://kexue.fm/archives/11804 【**直接就是本文线三"schedule ⟷ 权重平均"对偶的中文推导**；核心不等式 $\frac{1}{w_{1:T}}\sum_t w_t\mathbb{E}[g\cdot(\psi_t-\theta^*)]\le\frac{1}{2w_{1:T}}(R^2+\sum_t w_t^2 G_t^2)$，$R=\|\theta_1-\theta^*\|$。这篇把 schedule 和 averaging 统一，正对应 Defazio 2024 / WSM 那条线】（理论推导为主，无独立大规模实验）

### 线一 + 线二：Schedule 的凸理论与上下界
- 【理论】Orabona, *Last Iterate of SGD Converges (Even in Unbounded Domains)* (2020) — https://parameterfree.com/2020/08/07/last-iterate-of-sgd-converges-even-in-unbounded-domains/ 【系列二/三的源头，blog 非论文；纯凸优化理论，无实验】
- 【中等】Defazio, Cutkosky, Mehta, Mishchenko, *Optimal Linear Decay Learning Rate Schedules and Further Refinements* (2023) — https://arxiv.org/abs/2310.07831 【系列四/五/六的核心，第六篇正文明确"依然出自"；作者自称"迄今最全面的 schedule 评测"跨 10 个深度学习问题 + 一系列 LLM + logistic regression，最大 LLM 为 GPT-2 级，未给单一最大参数量（待核实）；线性衰减最优是**理论结论**，被 trillion-token 规模采用】
- 【理论】Agarwal, Bartlett, Ravikumar, Wainwright, *Information-theoretic lower bounds on the oracle complexity of stochastic convex optimization* (2010) — https://arxiv.org/abs/1009.0571 【下界基准；纯信息论理论，无实验】
- 【理论】Zamani, Glineur, *Exact convergence rate of the last iterate in subgradient methods* (2023) — https://arxiv.org/abs/2307.11134 【与 Defazio 2023 并列被引为"schedule 即可达最坏最优率"双源之一；Defazio 的线性衰减 last-iterate 结果正是对它的 online-to-batch 推广。纯凸理论，无实验。待正文确认苏神是否直接引】

### 线三：Schedule ⟷ 权重平均
- 【中等】Defazio et al., *The Road Less Scheduled* (NeurIPS 2024) — https://arxiv.org/abs/2405.15682 【Schedule-Free；跨 28 个问题（logistic regression→CIFAR/ImageNet/NanoGPT LM/MAE/DLRM/fastMRI），拿了 MLCommons 2024 AlgoPerf 自调优冠军；最大 LM 为 NanoGPT/BERT/GPT 级，未给单一最大参数量】
- 【中等】Bergsma, Dey, Gosal, Gray, Soboleva, Hestness, *Straight to Zero: Why Linearly Decaying the LR to Zero Works Best for LLMs* (ICLR 2025) — https://arxiv.org/abs/2502.15938 【D2Z，AdamW-as-EMA 对偶；最大 **610M**，跨 TPP 20–200、多 batch/词表/数据集，真实 LLM 预训练但 <1B】
- 【中等】Tian et al., *WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging* (2025) — https://arxiv.org/abs/2507.17634 【衰减=合并，Ling-1T 采用；真实 LLM 预训练 + MoE 实验，+3.5% MATH/+2.9% HumanEval/+5.5% MMLU-Pro over WSD，约 1B 级但**最大规模/token 数待核实**，若确证 ≥1B 应升为生产级】
- 【中等】Anytime Pretraining: Horizon-Free LR Schedules with Weight Averaging (2026) — https://arxiv.org/abs/2602.03702 【最大 **300M**（另 150M），1–32× Chinchilla；constant-LR + 权重平均 vs 调优 cosine，附过参数化线性回归理论】

### 线四：Optimizer / norm
- 【中等】Jordan, Jin, Boza, You, Cesista, Newhouse, Bernstein, *Muon: An optimizer for hidden layers in neural networks* (2024) — **blog，非 arXiv 论文**：https://kellerjordan.github.io/posts/muon/ （另有 Twitter/X thread 作为首发）。作者明确拒绝写成论文。blog 内规模：CIFAR-10 speedrun + NanoGPT 124M，并明确 scale 到 **774M / 1.5B**（1.5B 训到 GPT-2 XL HellaSwag 水平）；16B+ 来自后续 Moonlight。BibTeX 见该页 `@misc{jordan2024muon,...}`。
  - 【理论】Jeremy Bernstein, *Deriving Muon*（2025，blog）— https://jeremybernste.in/writing/deriving-muon 【Muon 的理论推导，配合 modular duality；无实验】
  - 【理论】Bernstein, Newhouse, *Old Optimizer, New Norm: An Anthology* (2024) — https://arxiv.org/abs/2409.20325 【"最速下降 under a norm"统一视角的源头，SGD/Adam/Muon 都是特例；纯框架/理论，无实验】
  - 【理论】Bernstein, Newhouse, *Modular Duality in Deep Learning* (2024) — https://arxiv.org/abs/2410.21265 【modular norm / 对偶，解释为何 embedding 层动态应不同；仅引用外部 NanoGPT 速度记录，无自有实验】
- 【生产级】Liu et al., *Muon is Scalable for LLM Training* (2025) — https://arxiv.org/abs/2502.16982 【Moonlight **3B/16B MoE（3B active / 16B total），5.7T tokens**；≈2× compute 效率 over AdamW；但摘要自陈"小规模有效、大规模尚未证明"，需加 weight decay + per-param update-scale 才能 scale——小规模成功≠充分】
- 【中等→生产级 borderline】Pethick et al., *Training Deep Learning Models with Norm-Constrained LMOs* (Scion) (2025) — https://arxiv.org/abs/2502.07529 【每层指定各自的范数：输入/输出层 $\ell_\infty$、隐藏层谱范数，基于 Frank-Wolfe / 约束优化视角；NanoGPT 64M→1B 做 LR 迁移，另有 **3B GPT** 大模型实验，3B 上匹配/略胜调优 Muon】
- （Bernstein–Newhouse 的"新范数"统一视角与 modular duality 见上方 Muon 条目下的子项）
- 【理论】Su Jianlin（苏剑林）, *AdamW 与 Muon 的 update RMS 匹配* — https://kexue.fm/archives/11267 【苏神本人关于 Muon 缩放因子 $\sqrt{(1-\beta_1)/(1+\beta_1)}$ 的推导，被 NVIDIA NeMo 等实现直接引用；推导为主】
- 【理论】*Preconditioning Benefits of Spectral Orthogonalization in Muon* (2026) — https://arxiv.org/abs/2601.13474 【纯理论：矩阵分解 + 线性 transformer 的 in-context learning 两个 case study 证线性收敛，无 LLM 实验】
- 【小/toy】关联记忆视角：*Muon Outperforms Adam in Tail-End Associative Memory Learning* (2025) — https://arxiv.org/abs/2509.26030 【机理论文：核心理论是**单层关联记忆模型** + 类不平衡/重尾合成数据，实证仅在小 LLM 上做组件消融，无大规模预训练】

### 线五：µP / 超参迁移 / scaling
- 【生产级】Yang, Hu, Babuschkin et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (µP / µTransfer, 宽度) (2022) — https://arxiv.org/abs/2203.03466 【从 40M proxy 迁移超越 **GPT-3 6.7B**（调参成本仅 7%），及 BERT-large 350M（从 13M 迁移）；"调小部署大"的典范】
- 【混合 理论+toy】Yang, Yu, Zhu, Hayou, *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks* (µP 深度迁移) (2023) — https://arxiv.org/abs/2310.02244 【深度-µP 理论；验证以深度刻画（up to 2^10=1024 blocks），实验多为 toy 残差网/MLP（CIFAR 级）+ 一个 Common Crawl 上的 Megatron transformer（无 headline 参数量）】
- 【小/toy】Yang, Simon, Bernstein, *A Spectral Condition for Feature Learning* (2023) — https://arxiv.org/abs/2310.17813 【µP 的谱范数视角，连接到 Muon；实验极简：3 层 MLP（宽 16–4096）在 200 例 CIFAR-10 子集上，无 transformer/LLM】
- 【生产级】Blake et al., *u-µP: The Unit-Scaled Maximal Update Parametrization* (ICLR 2025) — https://openreview.net/forum?id=P7KRIiLM8T （arXiv: https://arxiv.org/abs/2407.17465）【µP + Unit Scaling，开箱 FP8 训练验证 **到 7B**（base 消融多在 WikiText-103 较小规模）】
- 【中等】*Weight Decay may matter more than µP for LR Transfer in Practice* (2025) — https://arxiv.org/abs/2510.19093 【对 µP 叙事的修正；最大 **1B（宽 2048），20B tokens** LLaMA-style + AdamW，论证真正驱动 LR 迁移的是 weight decay 而非 µP】
- 【生产级】Bergsma et al., *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training* (2025) — https://arxiv.org/abs/2505.13738 【111M/266M/610M/1.7B/**3.3B**，TPP 20–1280；最优 AdamW timescale τ=B/(ηλD) 呈 TPP 幂律】
- 【生产级】Bjorck et al., *Scaling Optimal LR across Token Horizons* (ICLR 2025) — https://arxiv.org/abs/2409.19913 【最优 LR 随 token 数下降，$\beta\approx0.32$ 幂律；50M–2.7B 扫描 + 6.7B + **7B LLaMA** 留出验证，token 至 **800B**，>250 runs】
- 【N/A pre-LLM】McCandlish, Kaplan, Amodei et al., *An Empirical Model of Large-Batch Training* (critical batch size) (2018) — https://arxiv.org/abs/1812.06162 【critical batch size / 梯度噪声尺度奠基作；前 LLM 时代，跨 MNIST/CIFAR/ImageNet/Billion-Word LM + Atari/Dota2 RL + 生成模型，batch 从数万到数百万；无现代 transformer LLM，故不套 LLM 分档】
- 【中等→borderline】Zhang et al., *How Does Critical Batch Size Scale in Pre-training?* (2025) — https://arxiv.org/abs/2410.21676 【CBS 的现代 scaling 版本；85M–**1.2B** transformer LM on C4，结论：CBS 主要随数据量而非模型大小 scale】
- 【生产级】Power Scheduler (batch/token-agnostic, 2024) — https://arxiv.org/abs/2408.13359 【PowerLM-**3B**（1T tokens）+ PowerMoE-3B（800M active，2.5T tokens）；power-law LR + µP】

### 桥：理论↔经验
- 【中等 · 小而关键】Schaipp et al., *The Surprising Agreement Between Convex Optimization Theory and LR Scheduling for Large Model Training* (2025) — https://arxiv.org/abs/2501.18965 【最重要的桥。注：它选用 Defazio 2023 而非 Orabona 2020 的 bound，因为前者"对最后一步仍有意义"。**最大仅 124M/210M Llama-type**——这篇高影响力的"桥"本身是刻意小规模的，属"小规模、高影响"典型】
- 【理论】*Optimal Learning-Rate Schedules under Functional Scaling Laws* (2026) — https://arxiv.org/abs/2602.06797 【FSL + 相变：easy-task 区 power decay，hard-task 区 WSD 型；理论 + 特征空间线性回归/kernel regression 数值，LLM 相关性靠 FSL 框架继承而非新 LLM 实验】
- 【中等】*A Multi-Power Law for Loss Curve Prediction Across LR Schedules* (2025) — https://arxiv.org/abs/2503.12811 【主实验 **400M Llama-2，12B tokens**；预测未见 schedule 的 loss 曲线，自发得出 WSD 型最优】
- 【生产级】Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD 原始提出) (2024) — https://arxiv.org/abs/2404.06395 【MiniCPM **1.2B / 2.4B** 非嵌入参数（另 MoE、128K）；提出 WSD scheduler 与"模型风洞"小规模→大规模迁移方法论——是"设计小规模实验以迁移"的成功范例】
- 【中等】Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (WSD 大规模对比) (2024) — https://arxiv.org/abs/2405.18392 【33M–**360M**，token 0.3B–10B；constant-LR + cooldown / SWA 作 cosine 替代，作者注趋势"在现代规模下可能改变"】
- 【中等】Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective* (2024) — https://arxiv.org/abs/2410.05192 【**0.1B–1.2B**（WSD-S）+ bigram toy 模型展示 river-valley 地貌；小规模、高影响】
- 【小/中等】*Through the River: Understanding the Benefit of Schedule-Free Methods for LM Training* (2025) — https://arxiv.org/abs/2507.09846 【river-valley 视角下重新理解 schedule-free；最大 **124M** + toy 2D + CIFAR MLP，作者自陈"受算力所限仅小规模"】

---

## 7. 社区评价与价值光谱（哪些最值钱，哪些偏观赏）

> 本节是**声誉/判断**问题，非唯一答案。下面把"有明确共识信号的"（谁被前沿模型采用、谁被反复当基线、谁被明确写 underperforms）和"我的综合判断"分开标。

### 价值光谱一句话

**改方向的几何（Muon）> 工程灵活的调度（WSD / µP）> 解释性的桥（Schaipp）> 纯最坏情况理论（线性衰减、schedule-free、经典 SGD 收敛）。** 越靠"能被前沿模型直接采用"越值钱，越靠"漂亮的最坏情况定理"越偏学术观赏。

### 最有价值一档（强共识）

- **Muon / 谱范数优化器 —— 全场最硬的赢家**。唯一从 benchmark 真正走进**万亿级生产模型**的：Kimi K2/2.5、GLM-4.5/4.7、Moonlight 16B MoE、DeepSeek 都在用。scaling law 显示达 AdamW 同等性能只需约 **52% FLOPs**（Liu 2025）。共识信号极强——多个独立前沿团队真金白银投票，而非一篇论文自夸。
- **WSD schedule —— schedule 侧的实际赢家**。价值高于线性衰减纯理论，因为**工程灵活**：constant + 随时 cooldown，不锁定 $T$，支持 early-stop / 续训。被几乎所有后续 schedule 论文当头号基线。
- **µP / µTransfer —— 已成基础设施**。经典中的经典，已沉淀为标准工具（GPT-4 报告引用，Llama4 MetaP、Grok-2 疑似用）。价值在把调超参成本从全尺寸降到小 proxy。

### 中间档（有价值但有明确保留）

- **"凸理论惊人吻合 LLM"（Schaipp 2025）—— 理解价值 > 直接实用**（我的判断）。漂亮解释了线性衰减为何有效，但 bound 含 $G_t$、$D$ 等实践未知量，作者自承要额外想办法才能落地。共识是"重要的桥、启发性强"，但没人靠它直接调线上模型。是苏神这条纯理论线**最有生命力的延伸**。
- **WSM / 线性衰减理论（Defazio 2023）—— 有效但非颠覆**。WSM 有硬数字（+3.5% MATH / +2.9% HumanEval / +5.5% MMLU-Pro over WSD，Ling-1T 采用），但是 WSD 的改良而非另起炉灶。Defazio 2023 是整条 schedule 理论的根，但"最优"只在最坏情况凸下成立，实践中 WSD/cosine 常打平或更好。

### 偏鸡肋 / 不实用一档（有明确负面信号）

- **Schedule-Free（Defazio 2024）—— 最典型的"叫好不叫座"**。多篇直接写它逊于 WSD（Hägele 2024）。拿了 MLCommons 2024 AlgoPerf 自调优赛道冠军、学术上受尊敬，但**实际 LLM 预训练采用率低**，需要一堆后续修补（ScheduleFree+、Through the River）才追上 WSD。讽刺：它催生的**修补工作比它本身更被认可**。
- **纯经典 SGD 收敛结论（= 苏神系列一/二/三的核心本体）**。**苏神自己开篇就吐槽**："这些经典优化结论所依赖的条件过于苛刻，跟实际应用相去甚远，进入 LLM 时代后参考价值似乎更加有限"。凸 + Lipschitz + 有界方差，真实 LLM 一条都不严格满足。价值不在直接指导训练，而在 (a) 理解框架的地基，(b) 通过 Schaipp 那条桥间接发挥作用。SGD 本身在语言任务上也打不过 Adam。
- **Muon 用于 fine-tuning（相对 pretraining）—— 实用性受限**。存在"优化器不匹配"：用 Muon 微调 Adam 预训练的模型效果劣于 Adam，反之亦然；大多数开源模型是 Adam 预训练的，这严重限制 Muon 在微调场景的用处，需 LoRA-Muon 之类补救。

### 苏神的"暴论"：2026 年了，不 scale 的小 batch 实验论文还有必要吗？——辩证看

苏神原话是句带刺的调侃（kexue.fm 评论）："咱就是说，都2026年了，还'we use a fixed batch size of 512…'"。**字面**打的是"只用小固定 batch"的实验，**不是**否定一切小规模工作。分两种读法各有胜负：

**字面读法（只用小固定 batch 的经验排名）——证据充分，苏神基本对：**
- **critical batch size 直接证伪小 batch 排名**。McCandlish 2018 / Zhang 2024 / Essential AI 2025 都表明：低于 vs 高于临界 batch，compute/data 权衡乃至优化器排名会变；Essential AI 的头条正是"Muon 在**超**critical batch 后才胜 AdamW"——这个 regime 在小 batch 实验里根本看不见。这是对"fixed batch size 512"最直接的打脸。
- **小规模默认值把错误选择传播到大规模**。D2Z 指出无处不在的"cosine 衰减到 10%"默认，导致 Llama2-7B 训了 286 TPP 却严重欠衰减——一个小规模惯例在大规模是错的。
- **Muon 不 scale-free**。Moonlight 自陈"小规模有效、大规模未证"，需加 weight decay + per-param scale 才 work——小规模成功是必要非充分。
- **Schedule-Free 在 LLM 规模翻车**（Hägele 2024），小规模/AlgoPerf 上看着能打，LM 预训练被 WSD 超。

**宽读法（否定一切小规模/理论工作）——证据不支持，苏神过了：**
- **µP 本身**：从无限宽理论 + 13M–40M proxy 验证，零样本迁移到 **GPT-3 6.7B**。整个卖点就是"调小部署大"，是最硬的反例。
- **Muon 的出身**：始于 CIFAR-10 / NanoGPT 124M speedrun，才 scale 到 16B。小规模"望远镜"发现现象，大规模确认。Keller Jordan 甚至专门跑了 1.5B 实验回应"只在小模型 work 不会 scale"的质疑，并论证 NanoGPT speedrun 的价值恰在**便宜、可复现、可证伪**（"你根本不用信我"）。
- **凸理论预测 LLM loss 曲线**：Schaipp（≤210M）用凸 bound 复现 WSD/cosine 与 cooldown 行为——小规模/理论结果有直接的大模型解释力。
- **MiniCPM 的"模型风洞"**：显式地设计小规模实验以迁移，产出了现在广用的 WSD。

**综合判断**（我的，非共识）：苏神的话最好读作一个**在职实践者对"经验排名类主张"的门槛要求**——你要宣称优化器/schedule A 胜 B，就得在真实（大）batch regime 下验证，否则排名可能在临界 batch 处翻转。但它**不适用于**理论/机理/scaling-law 类工作，那些的价值在于严谨性和下游被采用度，而非参数量。注意苏神自己是 Moonlight 共同作者——他的门槛针对的是"经验落地",不是"数学漂不漂亮"。这也正对应本文参考文献里的**双轴标注**：规模档位管经验排名的可信度，理论/机理条目标【理论】豁免。



苏神系列讲的东西恰恰落在这个光谱**偏观赏的一端**——但这不代表没用。它的价值是**教育性和框架性**的：吃透这套理论，你才能读懂 Schaipp 为何"惊人"、判断 D2Z/WSM 的对偶从哪来。它是**理解这个领域的地图，而非领域本身的前沿**。对你（做 RL/alignment 理论、建知识体系）这恰恰是对的用法——先有地图再谈开疆。

### 相关补充参考（本节引用）

- 【中等模型/极小数据 · blog】Nishith Jain, *Muon vs MuonClip vs Muon+AdamW for Fine-Tuning*（HF blog, 2025）— https://huggingface.co/blog/KingNish/optimizer-part1 【实测：Muon+AdamW 混合最稳，胜过纯 Muon / MuonClip / 纯 AdamW；Qwen3-4B + 约 10k 指令行，仅微调、数据极小，非同行评审】
- 【生产级】*Practical Efficiency of Muon for Pretraining* (Essential AI, 2025) — https://arxiv.org/abs/2505.02222 【Muon 扩展 AdamW 的 compute-time Pareto 前沿；**到 4B** 参数，batch 至 16M tokens，证 Muon 在超 critical batch size 后仍保数据效率；µP telescoping HP 迁移】
- 【中等→borderline】*Can Muon Fine-tune Adam-Pretrained Models?* (2026) — https://arxiv.org/abs/2605.10468 【优化器不匹配问题 + LoRA-Muon 补救；微调至 **LLaMA-2-13B**，受控 from-scratch 预训练用两个 561M NanoChat（~11B tokens）】

---

## 8. 给 Zettelkasten 的建议结构

三层卡：
1. **MOC（内容地图）卡**：本文件即是。作为整个"大模型优化"主题的入口。
2. **五张"线"卡**（每条线一张）：线一凸理论 / 线二上下界 / 线三平均对偶 / 线四 norm 几何 / 线五 µP 迁移。互相双链，都反链到 MOC。
3. **论文原子卡**（每篇一张）：BibTeX + 一句话角色 + 所属线 + 与你 BASIS/GRPO 工作的接口。

核心双链关系（建议在卡间显式连出）：
- Orabona 2020 →（喂）→ 苏神系列二、三
- Defazio 2023 →（喂）→ 苏神系列四五六 + D2Z + WSM + Schaipp
- "线性衰减 $1-t/T$" 这个形状串起：Defazio 理论 → D2Z 实证 → Schaipp 桥接 → FSL 相变
- "schedule=平均" 串起：Schedule-Free → WSM → Anytime
- "norm 选择" 串起：Bernstein 统一视角 → SGD/Adam/Muon/Scion
- 与你前几轮讨论的接口：路线 A/B → 线一；有界域/投影 → 线一 + Orabona；无偏性 → 线一

**你的研究缺口卡**（值得单独立一张）：RL post-training 的 schedule 理论空白 + schedule/平均对偶在 policy 空间的对应物——直接挂到你的 BASIS/GRPO variance 工作上。