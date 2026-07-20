---
type: literature
created: 2026-07-08
tags: [interpretability, representation-theory, linear-representation, SAE, steering, probing, geometry, LLM]
status: seedling
confidence: medium
links: [gradient-2024-shape-symmetry-structure, cai-shenfeld-2025-on-policy-rl-dynamics]
source: "https://arxiv.org/abs/2311.03658"
authors: [Park, Choe, Veitch]
year: 2024
venue: ICML 2024
---

# The Linear Representation Hypothesis and the Geometry of Large Language Models

> **TLDR**: LLM 中"概念由方向表征"这一假说存在多个相互混淆的版本（unembedding 表征侧、probing 线性可分性、steering 可加性、SAE monosemanticity）。Park et al. 用**表示论**分类了这些版本，发现它们由一个 **Mahalanobis 型内积** $\langle u, v \rangle_M = u^\top M v$ 统一——概念在这个内积意义下正交，而非普通欧氏内积。

---

## 1. 问题：线性表征假说的混乱

"LLM 内部用方向（direction）表征概念"——这个直觉在多种场景下被使用，但不同场景说的其实是不同的东西：

| 场景 | 说的是 | 数学上 |
|------|--------|--------|
| **Unembedding 表征** | 输出空间中 $w_{\text{unembed}}$ 方向对应概念 | $\text{logit}(x) = w^\top h(x)$ |
| **Probing** | 存在线性分类器区分概念 | $f(h) = v^\top h > 0$ |
| **Steering / Activation Addition** | 沿某方向加向量可以控制生成 | $h' = h + \alpha \cdot d$ |
| **SAE Monosemanticity** | 从 superposition 中解出单义特征 | $h \approx \sum_i f_i d_i$（稀疏激活） |

**混淆点**：这些"方向"是同一个方向吗？在什么内积意义下？如果不区分，就会出现"probing 有效但 steering 无效"等矛盾现象。

---

## 2. 核心贡献：Mahalanobis 内积统一框架

### 关键发现
不同线性操作对应**不同的几何结构**：
- Probing 方向 $v_{\text{probe}}$ 和 steering 方向 $v_{\text{steer}}$ 一般**不在同一条线上**
- 它们通过一个矩阵 $M$（与模型的 residual stream 几何相关）联系：$v_{\text{steer}} \propto M^{-1} v_{\text{probe}}$

### 直觉
- 模型的 hidden state 空间不是欧氏的——它有由 LayerNorm、attention 结构等引入的**非平凡度量**
- "正交"应该在**这个度量**下定义，而非 $\langle u, v \rangle = u^\top v$
- 概念在这个度量下正交 ⟹ 它们可以被独立操作（steering 不干扰其他概念）

### Superposition 的几何重述
- Superposition = 概念数 > 有效维度 → 概念方向被迫不正交
- SAE 试图恢复 monosemantic features = 试图在这个非平凡度量下找到正交基
- **但 SAE 隐含假设了欧氏正交**——Park et al. 暗示应该在 Mahalanobis 内积下做

---

## 3. 与 RL 训练动力学的联系

### On-policy RL × 表征几何
- Cai-Shenfeld 证明了 RL 的 $\Delta W$ 是 rank-1 的
- 如果 RL 主要修改的是某个**方向**上的权重，那这个方向是什么？
- 猜想：RL 可能在对齐/调整模型内部的 **Mahalanobis 度量 $M$**——改变"什么算正交"，从而改变概念的独立性

### 与 reward hacking 的联系
- Reward hacking 可能是某些概念方向的 **over-rotation**——在 Mahalanobis 度量下，RL 让某些方向过度对齐了 reward signal
- EleutherAI 的 reward hacking 研究可能提供实证

---

## 4. 后续工作

- **Park, Choe, Veitch 2024 (arXiv:2406.01506)**: 将框架扩展到**分类概念**（categorical）和**层级概念**（hierarchical），用 simplex 和 polytope 的几何来刻画
  - 分类概念 ⟹ simplex 结构（$n$ 个互斥类别 → $n$-simplex 的顶点）
  - 层级概念 ⟹ 嵌套 polytope 结构

---

## 5. 开放问题

1. **$M$ 是什么？** — Park et al. 给出了存在性证明，但 $M$ 的计算方式、与模型架构的关系还不完全清楚
2. **RL 是否改变 $M$？** — SFT 和 RL 对 Mahalanobis 度量的影响是否不同？
3. **SAE 在 Mahalanobis 内积下做会怎样？** — 用正确的度量重做 SAE 是否能得到更好的 monosemantic features？
