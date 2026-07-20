---
type: literature
created: 2026-07-20
tags: [information-theory, mdl, data-selection, pre-training, compression, compute-bound]
status: seedling
confidence: medium
links: [cai-shenfeld-2025-on-policy-rl-dynamics, park-2024-linear-representation-hypothesis]
source: "https://arxiv.org/abs/2601.03220"
authors: [Marc Finzi]
year: 2026
---

# Epiplexity：从熵到计算受限智能的信息度量

> **TLDR**: 传统信息论（entropy, KL）是为**模型选择**设计的——给定数据，哪个模型压缩得最好？Epiplexity 翻转了这个视角：给定**计算约束**，如何**选择/生成/变换数据**以最大化学习效果？它将 compute 视为信息的维度，从 MDL 原则推导出一个面向数据选择的新度量。

---

## 1. 背景：模型选择 vs 数据选择

- **经典框架**（Shannon entropy, KL divergence, MDL）：以数据为固定输入，优化模型
  - MDL: $\min_M \; L(M) + L(D|M)$ — 选压缩数据最好的模型
  - 隐式假设：compute 无限，瓶颈在模型容量
- **现代现实**：模型架构已趋同（transformer），compute 是真正的瓶颈
  - 问题变成：给定固定 compute budget，哪些数据最值得训练？
  - 经典信息论无法回答这个问题——它没有 compute 维度

## 2. 核心概念：Epiplexity

Epiplexity 将 **compute** 作为信息的显式维度：

$$\text{Epiplexity}(D) = \text{从数据 } D \text{ 中提取的压缩效率，给定 compute budget } C$$

**与 MDL 的关系**：
- MDL: 给定数据，最小化 $L(\text{model}) + L(D|\text{model})$ → **模型选择**
- Epiplexity: 给定 compute，最大化从 $D$ 中提取的预测信号 → **数据选择**

> ⚠️ **TODO**: 精确公式待确认——核心公式是 MDL 框架的变体，将 data 作为优化变量而非固定输入。

## 3. 实际估算：用 Loss 代替真实描述长度

真实的 epiplexity 难以直接计算（需要穷举所有压缩方案）。实践中的估算：

$$\text{Epiplexity}(D) \approx -\mathcal{L}(D) \cdot \text{(scaling factor)}$$

其中 $\mathcal{L}(D)$ 是模型在数据 $D$ 上的 loss。

- **直觉**：loss 高 = 数据还没被压缩好 = 还有学习空间 = epiplexity 高
- **scaling laws** 可用于外推：在小模型上测 loss → 预测大模型上的 epiplexity

## 4. 应用：Pre-training 数据选择

- **数据过滤**：按 epiplexity 排序，优先训练高 epiplexity 样本
- **Curriculum design**：按 epiplexity 递减顺序安排训练数据
- **合成数据生成**：生成高 epiplexity 的合成样本（compute 转化为信息）
- **数据去重/去噪**：低 epiplexity 的数据 ≈ 已被压缩的冗余信息

## 5. 与已有笔记的连接

### ↔ Cai-Shenfeld (rank-1 / on-policy RL)
- RL 的 rank-1 update 本质上是在做"数据选择"——只保留 advantage 最大的方向
- Epiplexity 提供了一个 pre-training 层面的类比：选择哪些数据能产生最大的"advantage"

### ↔ Park (Linear Representation Hypothesis)
- 如果概念在激活空间中线性表征，epiplexity 可能衡量的是数据对新概念的"解锁"能力
- 高 epiplexity 数据 ↔ 激活空间中尚未被覆盖的方向

---

## 6. ⚠️ 待确认点

- [ ] 精确的核心公式（MDL 变体的具体形式）
- [ ] Loss 估算的具体 scaling factor 是什么？
- [ ] 论文中是否有实验结果（哪些数据集上验证的）？
- [ ] 与 data pruning / DSIR / DCLM 等已有 data selection 方法的比较

---

*阅读记录 2026-07-20，基于初步理解*
