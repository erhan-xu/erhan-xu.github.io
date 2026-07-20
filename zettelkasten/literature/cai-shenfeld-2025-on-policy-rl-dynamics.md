---
type: literature
created: 2026-07-06
tags: [rl, alignment, rlhf, interpretability, scaling, low-rank, on-policy]
status: seedling
confidence: high
links: []
source: "Cai et al. 2025 (arXiv:2510.00553); Shenfeld et al. 2025 (arXiv:2509.04259)"
authors: [Cai, Shenfeld]
year: 2025
---

# On-policy RL 更新为何特别：分布空间与参数空间的互补切片

> **TLDR**: SFT 和 on-policy RL 的梯度结构根本不同——区别不是 reweighting vs 非 reweighting，而是 **reweight 谁**。on-policy 的乘性更新 $\pi^\star \propto \pi_0 e^{A/\beta}$ 让 logit 更新小且保支撑（分布空间，RL's Razor）；verifiable reward 的稀疏性让 advantage 有效秩低，从而 $\Delta W$ rank-1（参数空间，Cai et al.）。两者独立：范数 $\perp$ 秩。

---

## 1. 背景与问题定位

RL（尤其 RLVR / GRPO 类）已经是 LLM 推理能力的主要驱动力，但**训练过程中参数在权重空间里到底发生了什么，几乎是黑箱**。多数可解释性工作看的是**终点**（激活探针、SAE），而对**训练动态**、特别是**权重空间的动态**研究稀少。

核心问题：**什么让 on-policy RL 更新如此特别？** 它坐落在三条已有脉络的交叉口——

- **遗忘 / 对齐**：为什么 RL 比 SFT 更少灾难性遗忘？
- **低秩微调**：为什么微调发生在低维子空间？（内在维度、LoRA）
- **权重编辑**：$\Delta W$ 作为可编辑、可组合的对象（ROME、task vector）

两篇论文分别落在**不同的空间**：RL's Razor 在**分布空间**（量 KL），Rank-1 在**参数空间**（量 $\Delta W$ 的谱）。二者不是谁的推论，而是对同一现象的互补切片。

## 2. 数学与精确逻辑

### 2.1 两者的梯度核：区别在 reweight 谁

**SFT 梯度**（logits 空间，$p=\mathrm{softmax}(z)$，$c$ 为目标 token）：

$$-\nabla_z L = e_c - p$$

朝**外生**目标（数据）搬质量，不动点 $= \hat{p}_{\mathcal{D}}$，$\pi_0$ 缺席 → 可跑远、满秩、大范数。

**Policy gradient**（$A$ 为 advantage，$\bar{A}=\mathbb{E}_{y\sim p}[A]$ 为 baseline）：

$$\nabla_z J = p \odot (A - \bar{A}\mathbf{1})$$

被当前分布 $p$ **门控**（每分量乘 $p_i$），只 reweight 自己；不动点

$$\pi^\star(y) \propto \pi_0(y)\, e^{A(y)/\beta}$$

$\pi_0$ 是**乘性锚** → 保支撑、KL 小。

> **核心命题**：区别不是 reweighting vs 非 reweighting，而是 **reweight 谁**——reweight 自己（基座在不动点里当因子）还是 reweight 到别人（数据分布）。

### 2.2 RL's Razor：分布空间的结论

on-policy 隐式选 KL 最小的解。在 $\pi_0$ 附近做 Fisher 二次型展开（$F$ 为 Fisher 信息矩阵）：

$$\mathrm{KL}(\pi_{\theta_0+\Delta\theta} \,\|\, \pi_{\theta_0}) \approx \tfrac{1}{2}\,\Delta\theta^\top F\, \Delta\theta$$

**小 KL ⟹ 小 $\|\Delta W\|$**（幅度约束）。

关键：**这个性质来自 on-policy 采样本身**（乘性 reweighting-within-support 结构），**不依赖显式 KL penalty**。RL's Razor 自己的 oracle 解耦实验用的就是**无 KL penalty、无 clipping 的 GRPO**，仍然观察到 KL-minimal bias。

- **静态**结论（终点是哪个解）
- 定理 + oracle 解耦，因果论证闭合

### 2.3 Rank-1 / AlphaRL：参数空间的结论

小 KL 只给**幅度**（小范数），给不出**秩**（秩 $\perp$ 范数：小范数可满秩，大范数可 rank-1）。低秩需要额外条件。

由乘性形式：

$$\Delta\log\pi \approx \eta A \quad\Longrightarrow\quad \mathrm{rank}(\Delta W) \;\leftrightarrow\; \text{advantage } A \text{ 的有效秩}$$

verifiable reward（0/1）让 $A$ 近似秩 1 ⟹ $\Delta W$ rank-1。

> **核心命题**：低秩是**奖励结构**的事，不是 on-policy 本身的事。on-policy 给出干净的乘性框架，秩是不是 1 取决于奖励。

- **含动态**（路径线性：$\theta_t - \theta_0 \propto (1 - e^{-\lambda t})$，早期 $\approx \lambda t$）
- 实证 + 相关，因果论证开放

### 2.4 两篇论文的精确对比

| | RL's Razor | Rank-1 / AlphaRL |
|---|---|---|
| 空间 | 分布 / 函数（KL） | 参数 / 权重（奇异谱） |
| 量 | $\mathrm{KL}(\pi_\theta\|\pi_0)$ | $\Delta W$ 的秩 |
| 依赖的旋钮 | on-policy **采样**（隐式 bias） | advantage **低维**（奖励结构） |
| 静 / 动态 | 静态（终点是哪个解） | 含动态（路径线性） |
| 预测 | 遗忘（旧能力保留） | 学习曲线 + 加速（新任务预测） |
| 因果论证 | 定理 + oracle 解耦（闭合） | 实证 + 相关（开放） |
| 逻辑关系 | 小 KL ⟹ 小范数 | **额外**需要 $A$ 稀疏 ⟹ 低秩 |
| 必要性 | on-policy 是充分条件（不需要显式 KL penalty） | on-policy + 稀疏奖励（两个都需要） |

### 2.5 Binary reward 如何产生 per-prompt rank-1（精确推导）

**一句话直觉**：binary reward 每个 prompt 只问模型一个问题——"这次对了吗？"。一个问题 = 一个比特 = 一个可教的对比方向（往"对"的分布靠、离"错"的分布远）。稠密奖励问很多个分级的问题 → 很多方向 → 高秩。

**记号**：prompt $x$，回答 $y=(y_1,\dots,y_T)$ 是 token 序列，策略 $\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t})$。binary verifiable reward $r(x,y)\in\{0,1\}$（对/错）——这是关键的结构输入。

GRPO：每个 prompt 采一组 $G$ 个回答，advantage 是组内归一化奖励：
$$A(x,y^{(i)})=\frac{r(x,y^{(i)})-\bar{r}_x}{\sigma_x},\quad \bar{r}_x=p_x=\tfrac{1}{G}\textstyle\sum_i r,\ \ \sigma_x=\sqrt{p_x(1-p_x)}$$
$p_x$ 是这个 prompt 的组内正确率，$\sigma_x$ 是组内标准差。

**严格能证的核：binary reward ⟹ 每个 prompt 的梯度 rank-1**

**第一步——advantage 只取两个值。** 因为 $r\in\{0,1\}$，advantage 是一个两值阶梯函数：
$$A_+=\frac{1-p_x}{\sigma_x}=\sqrt{\tfrac{1-p_x}{p_x}}\ \ (\text{答对}),\qquad A_-=\frac{-p_x}{\sigma_x}=-\sqrt{\tfrac{p_x}{1-p_x}}\ \ (\text{答错})$$
per prompt 只有一个自由度：你在阶梯的哪一侧。这就是低维的种子。

**第二步——把梯度按对/错拆开。** policy gradient：
$$\nabla_\theta J=\mathbb{E}_x\,\mathbb{E}_{y\sim\pi_\theta}\!\big[A(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)\big]$$
给定 $x$，按"答对/答错"做条件期望。记：
$$g_+^x=\mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta\mid \text{答对}],\qquad g_-^x=\mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta\mid \text{答错}]$$
（分别是"正确回答的平均 score 向量"和"错误回答的平均 score 向量"）。于是：
$$\mathbb{E}_{y\sim\pi_\theta}[A\nabla\log\pi\mid x]=\underbrace{p_x A_+}_{}\,g_+^x+\underbrace{(1-p_x)A_-}_{}\,g_-^x$$

**第三步——那两个系数恰好抵成 $\pm\sigma_x$。** 直接算（可自行验证）：
$$p_x A_+=p_x\sqrt{\tfrac{1-p_x}{p_x}}=\sqrt{p_x(1-p_x)}=\sigma_x,\qquad (1-p_x)A_-=-\sigma_x$$
代回：
$$\boxed{\ \nabla_\theta J=\mathbb{E}_x\big[\,\sigma_x\,(g_+^x-g_-^x)\,\big]\ }$$

这就是核。每个 prompt 的梯度贡献 $=$ 一个标量 $\sigma_x$ 乘以单个差向量 $(g_+^x-g_-^x)$——"抬升正确回答对数似然的平均方向" 减 "抬升错误回答的平均方向"。**per prompt 严格 rank-1**：binary reward 把该 prompt 的全部信号坍缩成一个"正确 − 错误"对比方向。以上没有任何近似，是精确代数。

**为什么稠密奖励会破坏它：** 如果 $r$ 取多个值（或 PRM 给逐步奖励），advantage 不再两值，第二步就变成对很多不同 $(A_k,g_k)$ 的加权和：
$$\nabla_\theta J=\mathbb{E}_x\Big[\textstyle\sum_k A_k\, g_k^x\Big]$$
是很多方向的组合 → 高秩。所以"binary"不是可有可无的——正是奖励只有一个比特，才把 per-prompt 贡献压成单一对比。这也给出了可证伪推论：换 PRM，rank-1 应变弱。

**从"per-prompt rank-1"到"$\Delta W$ rank-1"——这里进入假设：** $\nabla_\theta J=\sum_x\sigma_x(g_+^x-g_-^x)$ 是 $N$ 个 prompt 的求和，一般是 rank $\min(N,d)$，不是 rank-1。要坍缩到一个方向，还需两个结构事实：

**假设 1（跨 prompt 共线）**：不同 prompt 的对比方向近似平行：
$$g_+^x-g_-^x\approx c_x\,d\quad(\text{共享方向 }d,\ \text{prompt 专属标量 }c_x)$$
直觉/证据：区分"对的推理链 vs 错的推理链"的东西，与具体是哪道数学题无关——由一小撮通用行为承载（论文 Appendix D 的 "Alright/Let" 定策略、"But/Wait" 自我纠错）。若成立：
$$\nabla_\theta J\approx\Big(\textstyle\sum_x\sigma_x c_x\Big)\,d=\text{标量}\cdot d$$
聚合梯度 rank-1，方向就是那个通用的"推理能力"方向 $d$。binary reward 给每个 prompt 一个对比向量；对比方向的跨 prompt 通用性（经验事实）把求和坍缩成一个方向。两者缺一不可。

**假设 2（方向稳定 / lazy 区）**：$\Delta W=\sum_t\eta_t\nabla_t$ 是逐步累积。若每步方向都 $\approx d$ 且训练中 $d$ 不怎么转（近基座的 lazy 区，正是 RL's Razor 小 KL 保证的），则 $\Delta W\approx(\sum_t\eta_t\text{标量}_t)\,d$ 仍 rank-1。若方向大幅旋转（feature learning），即便每步 rank-1，累积的 $\Delta W$ 也会升秩——这就接回未解问题 #3。

**更底层的一层（连 Appendix D 到矩阵秩）**：对某个具体线性层 $W\in\mathbb{R}^{m\times n}$，单 token 的 score 是外积 $\nabla_W\log\pi_\theta=\sum_t \delta_t\otimes h_t$（$\delta_t$ 是输出侧梯度、$h_t$ 是输入激活，各自 $m$ 维 / $n$ 维）。每个 token 贡献一个 rank-1 外积；求和的有效秩 $\approx$ "起作用的 token 数"。Appendix D 说只有少数关键 token 起作用 → 每个样本的有效贡献本就近似 rank-(少数)，再叠加"正确 − 错误"的单一对比 → 进一步集中。这把"token 稀疏"机制地翻译成了"矩阵低秩"。

**诚实的账本**：
- **精确成立**：binary reward ⟹ advantage 两值 ⟹ per-prompt 梯度 $=\sigma_x(g_+^x-g_-^x)$，per-prompt rank-1。纯代数，无假设。
- **额外需要（有经验支撑、未证明）**：① 跨 prompt 对比方向共线（Appendix D）；② 训练中方向稳定（lazy / 小 KL）。这两条才把 per-prompt rank-1 放大成 $\Delta W$ rank-1。

所以准确的说法不是"rank-1 来源于 binary reward"，而是：**binary reward 提供了 per-prompt 的 rank-1 种子（稠密奖励会破坏它），通用性 + lazy 动力学把它放大到 $\Delta W$ rank-1。** binary 是必要的种子，但不充分；充分性来自那两条几何假设。

### 2.5 DAPO / Dr.GRPO 无 KL penalty 的分析

**经验事实**：DAPO（无 KL penalty）的 rank-1 恢复率是**最强的**——Table 8 显示 MATH-500 上 rank-1 恢复 100%，正文说 "in RLOO, GRPO, and DAPO, it can even surpass the fully trained model"。

**框架精炼与验证**：这个结果**验证**（而非反驳）我们的分析，关键在于正确识别机制：

1. **小 KL 来自 on-policy 采样，不来自显式 KL penalty**。RL's Razor 自己的 oracle 解耦实验用的就是**无 KL penalty、无 clipping 的 GRPO**，仍然观察到 KL-minimal bias。乘性更新结构 $\pi_{\text{new}} \propto \pi_{\text{old}}\, e^{\eta A}$ 本身就是"reweight 自己、保支撑、KL 小"的来源——这是采样机制给的，不需要显式 KL 项。显式 KL penalty 只是在它之上再加一道**次要约束**。

2. **DAPO/Dr.GRPO 去掉的是次要约束，主机制原封未动**。它们仍然是 on-policy + binary reward，所以 rank-1 当然不弱。这反过来精炼了我们框架的必要条件：不是"KL penalty"，而是"on-policy 采样"。

3. **反直觉的细节：KL penalty 可能不降反升秩**。显式 KL penalty 项 $-\beta\,\mathrm{KL}(\pi_\theta\|\pi_0)$ 的梯度 $\nabla_\theta\mathrm{KL}$ 是一个朝 $\pi_0$ 拉回的复原力，其方向一般不与 advantage 驱动的 rank-1 方向对齐。所以总更新：
$$\Delta W = \underbrace{(\text{reward 驱动的 rank-1 方向})}_{\propto\,\eta A} + \underbrace{(\text{KL 复原方向})}_{\text{未必 rank-1 对齐}}$$
加 KL penalty 反而可能掺进一个偏离主方向的分量、轻微抬高有效秩；去掉它（DAPO）留下的是更纯粹的 rank-1。这和 DAPO rank-1=100% 是方向一致的。（当然 GRPO 的 $\beta=0.001$ 极小，效应很弱，但方向性论证成立：加 KL penalty 不会增强 rank-1，反而可能略微削弱。）

**朴素直觉的错误**：直觉上可能认为"无 KL penalty → 更新更激进 → rank-1 更弱"，这是把"显式 penalty"错当成了小 KL 的来源。实际上小 KL 来自 on-policy 采样的隐式偏置，显式 penalty 只是次要加固。

**诚实的 caveat**：在论文的 regime 内 DAPO 强 rank-1；但如果 DAPO 训练得很激进（clip-higher, 动态采样, 很多步）推离 base model 很远（离开 lazy/near-base regime），rank-1 是否仍然成立是**未检验的**——连接到 open problem #3（lazy vs feature-learning 的张力）。

## 3. 直觉层

### 一句话理解

SFT 说"变成那个数据"，RL 说"在已有的东西里挑好的"——前者可以走很远（满秩大范数），后者只在自己身上微调（小范数），而且当"好"只有一个维度时（对/错），调整也只有一个方向（rank-1）。

### 范数 ⊥ 秩的正交性

小范数可满秩（每个方向都动一点点），大范数可 rank-1（一个方向动很多）。RL's Razor 解释的是范数，Cai et al. 解释的是秩。混淆两者是常见的直觉错误。

### 可解释性接口

- **ROME / key-value**：rank-1 = 学出来的 ROME 式权重编辑，$v_1$ = key（触发语境）/ $u_1$ = value（写入行为）
- **内在维度 / LoRA**：无约束 RL 已 rank-1，是内在维度现象的涌现强化版 → 低秩 RL 应近乎无损
- **SAE / thought-anchors**：$u_1$ 应对应少数 SAE 特征；rank-1 作用位点应 = thought anchors（Alright / But / Wait 等 token）。*可证伪。*
- **发展式可解释性 / SLT**：(动态 × 权重) 象限最空白，最贴合，用 LLC 给"有效秩"原则性定义

## 4. 开放问题与可检验预测

### 4.1 跨 prompt 共线性检验（假设 1 的直接度量）

**目标**：验证"不同 prompt 的对比方向近似平行"（假设 1），这是把 per-prompt rank-1 种子放大成 ΔW rank-1 的关键一环。

**实验设计**：

**Step 1: 估计 per-prompt 对比向量**

从训练日志（或 checkpoint 重算），对每个 prompt $x$：
$$v^x = g_+^x - g_-^x = \mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta \mid r=1] - \mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta \mid r=0]$$

对每个 prompt 采样一组回答（或用已有 rollout），按 reward 分成对/错两堆，分别算平均 score 向量再相减。

**Step 2: 堆成矩阵做 PCA**

把 $N$ 个 prompt 的 $v^x$ 堆成矩阵 $V \in \mathbb{R}^{N \times d}$（$d$ = 参数维度）。

**Step 3: 三个诊断量**

1. **第一主成分方差占比**：$\lambda_1 / \sum_i \lambda_i$
   - 若 >90% → 强共线，支持假设 1

2. **有效秩（participation ratio）**：$\exp(H(p))$，其中 $p_i = \lambda_i / \sum_j \lambda_j$
   - 若 ≈ 1-2 → 支持假设 1

3. **两两余弦相似度分布**：$\cos(v^x, v^{x'})$ 对所有 prompt 对
   - 若集中在 ±1 附近 → 支持假设 1
   - 若分散 → 反对假设 1

**预期结果**：
- $v^x$ 高度共线 → 假设 1 成立 → "通用推理能力方向 $d$" 存在 → per-prompt rank-1 种子确实能放大成 ΔW rank-1
- $v^x$ 分散 → 假设 1 不成立 → 需要其他机制解释 ΔW rank-1（或者 rank-1 本身就是近似的，不是严格的）

**计算成本**：中等。需要 per-sample 梯度（或从 checkpoint 重算），但只需要一个训练快照，不需要完整训练轨迹。GRPO 的 rollout 通常已经记录了，可以直接用。

**价值**：这是比"换 PRM 看 rank-1 变化"（Claim B 检验）更便宜的诊断——后者需要重新训练，这个只需要分析现有日志。如果共线性成立，就直接证了"通用推理方向 $d$"的存在，补上从 §2.5 的种子到全局 rank-1 的关键一环。

### 4.2 秩 ← 奖励结构（Claim B 的检验）

把 "verifiable / 稀疏 reward ⟹ advantage 有效秩低 ⟹ $\Delta W$ rank-1" 做成带假设的命题。

**判据实验**：固定 on-policy GRPO，只变奖励密度：
- Binary verifiable reward (0/1) → 预测 rank-1 强
- Process reward model (PRM, dense per-step) → 预测 rank-1 弱（advantage 散布到更多方向）
- Continuous / shaped reward → 中间

如果 rank-1 dominance 随 reward 密度增加而降低（on-policy 固定），那就是 Claim B 的直接证据。

### 4.3 Advantage 有效秩的直接度量（smoking gun）

$\Delta\log\pi \approx \eta A$ ⟹ $\mathrm{rank}(\Delta W) \leftrightarrow \mathrm{rank}(A)$。

从训练日志中直接度量 advantage 矩阵 $A(x, \text{token})$ 的有效秩（stable rank / participation ratio），检查它是否在方法间追踪 $\Delta W$ 的 rank-1%。如果追踪，就是 Claim B 的 smoking gun。计算量低，可直接验证。

### 4.4 Lazy 还是 feature-learning？

Property 2 的线性能否由"小 KL → lazy/NTK 区"推出？早期 $\theta_t-\theta_0 \propto (1-e^{-\lambda t}) \approx \lambda t$ 近似线性。但 lazy 预测"**方向固定 + 幅度增长**"，与论文观测的"**方向演化**"有张力——RL 更新落在两者之间何处，**未知**。

### 4.5 性能梯度是否 rank-1

该验证的是性能梯度 $\nabla_W P$ 的秩，而非 $\Delta W$ 的秩；范数 rescale（$\alpha=\|\Delta W\|/\|\Delta W^{(1)}\|$）让"恢复 99%"名不副实的问题**未解**。

### 4.6 无未来信息的预测 + 误差界

AlphaRL 用 $y^*=1$（终点相对准确率）反推，泄漏未来信息。应改用轨迹**自饱和**预测终点，并给外推置信区间：

$$\text{预测误差} \lesssim \underbrace{\text{窗口内残差}}_{\text{拟合优度}} + \underbrace{\text{曲率}\times(\text{外推距离})^2}_{\text{非线性}} + \underbrace{\text{advantage 噪声}}_{\propto 1/G}$$

### 4.7 有效维度的严谨度量

用奇异学习理论的 **LLC / RLCT** 替代 ad-hoc 的 SVD-rank，并检验可证伪预测——"**RL 平滑发展 vs SFT 分阶段相变**"。

### 4.8 On→off-policy sweep（Claim A 的检验）

保持 reward 固定，逐步 off-policy（stale rollouts, replay buffer, $\pi_\theta$-vs-$\pi_{\text{behavior}}$ gap 增大）→ 预测 $\|\Delta W\|$ 增长且 rank-1 减弱。论文的 off-policy DPO（norm 0.79, rank-1 62%）是一个数据点；受控的 on→off sweep 可以画出完整曲线。

### 4.9 已有支持证据

Table 7/8 的范数列已部分支持 Claim A：
- DIST (21.24), SFT (10.75)：巨大（off-policy / 非 RL）
- RL DAPO (0.01)：极小（on-policy）
- DPO (0.79)：中间（有 on-policy 成分但非纯 on-policy）
- on-policy distill (1.46) vs offline distill (21.24)：on-policy 小 15×

DPO 和 on-policy distill 的 rank-1 是中间值（56%, 62%），符合"需要两个旋钮都到位才能 rank-1"。

## References

- Cai et al., *On Predictability of Reinforcement Learning Dynamics for Large Language Models*, arXiv:2510.00553
- Shenfeld et al., *RL's Razor: Why Online Reinforcement Learning Forgets Less*, arXiv:2509.04259
- Geva et al. 2021, *Transformer Feed-Forward Layers Are Key-Value Memories*, arXiv:2012.14913
- Meng et al. 2022, *Locating and Editing Factual Associations in GPT* (ROME), arXiv:2202.05262
- Aghajanyan et al. 2020, *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, arXiv:2012.13255
- Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685
- Watanabe, *Singular Learning Theory*; 发展式可解释性 / LLC / RLCT
