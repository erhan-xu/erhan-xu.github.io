---
type: literature
created: 2026-07-08
tags: [geometry, symmetry, group-theory, category-theory, representation-theory, loss-landscape, interpretability, fiber-bundle]
status: seedling
confidence: medium
links: [cai-shenfeld-2025-on-policy-rl-dynamics, park-2024-linear-representation-hypothesis]
source: "https://thegradient.pub/shape-symmetry-structure/"
authors: [The Gradient]
year: 2024
---

# Shape, Symmetries, and Structure: 数学在 ML 研究中的角色演变

> **TLDR**: ML 社区过去十年从"数学辅助调参"走向"数学定义问题本身"。四个关键工具——几何（intrinsic dimension/curvature/topology）、对称性（群/equivariance/表示论）、模型内部对称性（permutation symmetry/loss landscape/LMC）、范畴论（fiber bundle/Bundle Networks）——正在从边缘走向核心。

---

## 1. 几何工具：高维空间的直觉替代品

### Intrinsic Dimension（内蕴维度）
- 模型参数空间维度极高，但实际训练轨迹的有效维度很低
- 直接联系：LoRA 有效性的理论基础——微调只需要低维子空间
- **与 RL 训练动力学的联系**：on-policy RL 的 rank-1 更新暗示训练轨迹的 intrinsic dimension 可能更低（rank-1 ⟹ 每步只沿一个方向移动）

### Curvature（曲率）
- Hessian 特征值分布刻画 loss landscape 的局部几何
- **Edge of Stability** (Cohen et al. 2021)：大学习率下 loss 在曲率方向上震荡但不发散——RL 训练中是否也有类似的 edge of stability？
- Fisher Information Matrix ≈ Hessian（在 MLE 下），而 RL 的 policy gradient 也有 Fisher 的自然梯度——曲率信息编码在训练动力学中

### Topology / Homology（拓扑/同调）
- Persistent homology 用于分析 loss landscape 的拓扑结构（连通分量、环、空洞）
- 不同优化器可能找到拓扑结构不同的解

---

## 2. 对称性：群、等变性、表示论

### Equivariance 的核心交换图
$$f(g \cdot x) = g' \cdot f(x)$$
- $f$：网络，$g$：输入空间的群作用，$g'$：输出空间的群作用
- CNN：$g$ = 平移，$g'$ = 平移 → translation-equivariant
- GNN：$g$ = 节点置换 → permutation-equivariant

### 表示论与不可约表示（Irreps）
- 所有线性等变映射可由表示论完全分类
- **关键文献**：Park, Choe, Veitch (2024) 用表示论统一了 LLM 中几种"线性表征假说"——见 [[park-2024-linear-representation-hypothesis]]

### AlphaFold3 的反直觉
- 理论上应该用等变架构（SE(3)-equivariant），但 AF3 用非等变架构仍然有效
- **scale vs. built-in prior 的权衡**：当数据量足够大时，data-driven symmetry learning 可以超越 hand-coded equivariance
- 开放问题：是否存在某个临界数据量，使得 built-in prior 总是更好？

---

## 3. 模型内部对称性

### Permutation Symmetry（置换对称）
- 同一层内的神经元可以任意置换（配合上下游权重调整）→ loss landscape 有大量对称的鞍点
- **Git Re-Basin** (Ainsworth et al. 2022)：通过 neuron alignment 找到两个独立训练模型的置换映射，然后在权重空间做线性插值 → loss 几乎不升高
- 这就是 **Linear Mode Connectivity (LMC)** 的基础

### LMC 与 Loss Landscape
- 如果两个解可以通过线性路径连接而不经过高 loss 区域，说明它们在同一个 basin 内
- **与 RL 的联系**：SFT → RL 的路径是否也在同一个 basin 内？RL 是否只是在 SFT 解的 basin 内做微小调整？如果 rank-1 更新为真，那 $\Delta W$ 的低秩性质暗示 RL 不会跳出 basin

### Neuron 语义与 Superposition
- 单个神经元通常不对应清晰的语义概念（polysemanticity）
- SAE (Sparse Autoencoder) 试图从 superposition 中解出 monosemantic features
- **Park et al. 的贡献**：用 Mahalanobis 内积分清了"unembedding 的表征侧"、"probing 的线性可分性"和"steering 的可加性"——这些概念在普通欧氏内积下容易混淆

---

## 4. 范畴论视角

### Product 的泛性质（Universal Property）
- 不定义 product "是什么"，而是定义 product "能做什么"
- 数学中的 product 由投影映射 + 唯一性定义，不依赖具体构造

### Fiber Bundle（纤维丛）
- 直觉：在每个底空间点上"粘"一个纤维空间
- $E \xrightarrow{\pi} B$，$E$ = total space，$B$ = base space，$F$ = fiber
- **Bundle Networks**：把数据流形上每点的特征看作纤维，网络学习如何在纤维间一致地传递信息

### 从抽象到代码
- 范畴论提供了"接口定义"（泛性质），fiber bundle 提供了"数据结构"（局部 trivialization）
- 问题：如何让神经网络原生地操作这些结构？→ Group Equivariant Networks, Bundle Networks

---

## 5. 开放问题与个人研究方向联系

### RL 训练动力学 × 几何
- RL 的 loss landscape 是否和 SFT 有本质不同的几何结构？
- on-policy 更新的 rank-1 性质是否意味着 RL 在 fiber bundle 上沿特定 connection 移动？
- Edge of Stability 是否在 RL 训练中也有对应现象？

### 对称性 × 可解释性
- SAE features 的 superposition 本质上是表示论问题——Park et al. 的 Mahalanobis 内积提供了一个几何框架
- 如果 RL 的 rank-1 更新保持某些对称性不变，那这些不变量可能就是 RL "学到的东西"

---

## 参考论文（文中引用或提及的关键文献）

| 论文 | 核心贡献 |
|------|---------|
| Park, Choe, Veitch 2024 (arXiv:2311.03658) | Mahalanobis 内积统一 linear representation hypothesis |
| Park, Choe, Veitch 2024 (arXiv:2406.01506) | 分类/层级概念的 simplex/polytope 几何 |
| Ainsworth et al. 2022 | Git Re-Basin: neuron alignment + LMC |
| Cohen et al. 2021 | Edge of Stability |
| Bronstein et al. 2021 | Geometric Deep Learning (book) |
| Batzner 2022 | e(3)nn library (equivariant networks) |
