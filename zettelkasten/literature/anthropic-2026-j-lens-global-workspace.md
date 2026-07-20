---
type: literature
created: 2026-07-15
tags: [interpretability, jacobian, global-workspace, sparse-subspace, consciousness-analog, mechanistic-interpretability]
status: seedling
confidence: medium
links: [cai-shenfeld-2025-on-policy-rl-dynamics, park-2024-linear-representation-hypothesis, zhang-2026-is-one-layer-enough, gradient-2024-shape-symmetry-structure]
source: "https://www.anthropic.com/research/global-workspace; Transformer Circuits Thread 2026-07-06"
authors: [Anthropic]
year: 2026
---

# J-Lens & Global Workspace：Jacobian 视角下的 Claude 内部工作空间

> **TLDR**: Anthropic 用 Jacobian lens（$\partial \text{logits}/\partial h_\ell$）发现 Claude 内部存在一个稀疏子空间（J-space），其中少量激活方向对输出有决定性影响。这个 J-space 满足 Global Workspace Theory 的 5 个功能特性（broadcast、flexible routing、limited capacity、conscious access analog、integration）。干预 J-space 可以改变 Claude 的"沉默推理"而不影响其表面行为。

---

## 1. 核心方法：J-Lens

### 定义
J-lens 计算从某一层激活 $h_\ell$ 到最终输出 logits 的 Jacobian：
$$J_\ell = \frac{\partial \text{logits}}{\partial h_\ell}$$

**注意**：这不是说 block 是线性的。而是说在给定输入 $x$ 附近，激活值的**微小变化**如何线性地传播到输出——标准的局部线性近似。

### 关键发现：J-space 的定义与稀疏性

**定义**：将 Jacobian 作用于 unembedding 矩阵：
$$D_\ell = J_\ell W_U \in \mathbb{R}^{d_{model} \times |V|}$$
$D_\ell$ 的每一列是某个 vocabulary token 在激活空间 $h_\ell$ 中的"对应方向"。

**J-space = $D_\ell$ 列向量的稀疏非负线性组合所张成的子空间。**

- 任何时刻只有 ~25 个方向活跃
- J-space 不超过任何层激活方差的 10%

**SVD 的角色（诊断工具，不是定义）**：对 $J_\ell$ 做 SVD：$J_\ell = U \Sigma V^\top$

- 大部分奇异值接近零（大部分激活方向对输出无影响）
- 少数大奇异值（少数方向"重要"）
- SVD 揭示了 J-space **为什么**稀疏，但 J-space 本身 ≠ top-$k$ right singular vectors 的 span

> ⚠️ **注意区分**：J-Lens 是技术（应用 Jacobian + LayerNorm + unembedding），J-Space 是空间（$JW_U$ 列的稀疏组合张成的子空间）。SVD 是谱分析工具。

### 操作含义
- J-space 中的方向 = 对输出"重要"的方向
- 干预 J-space → 改变模型的"沉默推理"
- 干预 J-space 之外的方向 → 几乎不影响输出

---

## 2. Global Workspace 的 5 个功能特性

Anthropic 验证 J-space 满足 Global Workspace Theory (Baars 1988) 的：

1. **Broadcast**：J-space 中的信息可被多种下游过程访问
2. **Flexible routing**：同一 J-space 表征可服务于不同任务
3. **Limited capacity**：J-space 是稀疏的（只有少数方向活跃）
4. **Conscious access analog**：J-space 中的信息是模型"可以报告但尚未说出"的
5. **Integration**：J-space 整合来自多个模块的信息

**关键实验**：阻止 Claude 使用 J-space → 仍然能正常交互，但丧失高阶认知功能。

**替换 J-space 内容** → 可以重定向 Claude 的沉默推理。

---

## 3. 与已有笔记的连接

### ↔ Park (Linear Representation Hypothesis)
- Park：概念在 **Mahalanobis 度量** $M$ 下正交
- J-lens：概念的"重要性"由 Jacobian 奇异值决定
- **可能的统一**：$M \approx J^\top J$（Jacobian 的 Gram 矩阵编码了"重要性"的度量结构）
- 如果成立：Park 的"概念在 $M$ 下正交" ↔ "概念在 Jacobian 奇异向量下正交"

### ↔ Cai-Shenfeld (rank-1)
- RL 的 $\Delta W$ rank-1 → 训练只在一个方向上大幅改变权重
- J-space sparse 但不一定 rank-1 → 有多个"重要方向"
- **可能的调和**：RL 主要修改 J-space 中最重要的那个方向（rank-1 update 对齐第一奇异向量），其他方向来自 pre-training

### ↔ Zhang (layer contribution)
- 中间层 (40-60% 深度) 贡献最大
- J-space 是否也集中在中间层？
- 如果 J-space 是"全局广播"的 workspace，源头可能在中间层（信息瓶颈位置）

### ↔ Gradient-2024 (数学工具)
- Jacobian = 曲率/几何信息的具体实现
- Fisher Information ≈ $J^\top J$（在 MLE 下）→ J-space ≈ Fisher 信息的主子空间
- 自然梯度 = Fisher 逆 × 普通梯度 → 自然梯度优先更新 J-space 方向

---

## 4. ⚠️ 理解张力与待精炼点

> 以下记录讨论中暴露的理解矛盾和 gap，需后续 refine。

### 张力 1：线性近似 vs 非线性真实

**Erhan 的直觉**：Anthropic 把一个 transformer block "unreasonably simply to a linear transformation"。

**校正**：不是把 block 简化成线性，而是用局部线性近似做**诊断工具**。关键洞察不在于"线性"本身，而在于 Jacobian 的**谱结构**是稀疏的。

**待 refine**：
- 为什么局部线性近似就够了？是否因为模型在 lazy/near-base regime 运行？
- 如果模型远离 base（深度 feature learning），Jacobian 的稀疏性是否消失？
- 这和 Edge of Stability 有什么关系？（大 LR 下 Jacobian 可能不再稀疏）

### 张力 2："方向 = 概念" vs "Jacobian = 概念的重要性"

**Erhan 的直觉**：方向/Jacobian 就是概念。

**校正**：
- 概念 = 激活空间中的方向 $v$
- Jacobian $J$ = 从激活空间到输出空间的**映射**
- Jacobian 告诉我们哪些方向"重要"（$\|Jv\|$ 大）
- 更精确：$J$ 不是概念，是概念的**重要性评估器**

**待 refine**：
- $M \approx J^\top J$ 这个猜想是否有文献支持？
- 如果成立，Park 的 SAE-in-Mahalanobis 建议就变成了 SAE-in-Jacobian-metric
- 这和 "natural gradient = 在 Fisher 度量下的最速下降" 是否统一？

### 张力 3：J-space 的容量 vs rank-1

- J-space 是 sparse（~几十个方向）
- Cai-Shenfeld 说 RL update 是 rank-1（1 个方向）
- **问题**：RL 是否只修改 J-space 的第一奇异向量？还是 rank-1 update 和 J-space 是不同层次的概念？

**待 refine**：
- J-space 的有效维度（participation ratio）是多少？
- 不同训练阶段（pre-training vs SFT vs RL）J-space 是否变化？
- RL 的 rank-1 update 是否在 pre-training 建立的 J-space 中？

---

## 5. 开放问题

→ J-space 的空间分布：是否集中在 40-60% 深度（中间层）？
→ J-space 的时间动态：生成过程中 J-space 内容如何变化？
→ 和 SAE 的关系：SAE features 是否集中在 J-space 中？
→ 和 thought anchors (Alright/But/Wait) 的关系：这些 token 是否对应 J-space 的切换？
→ Consciousness 问题：Anthropic 明确区分了 access consciousness（功能性的）和 phenomenal consciousness（主观体验）。J-space 只对应前者。

---

*讨论记录 2026-07-15，理解张力待后续 refine*
