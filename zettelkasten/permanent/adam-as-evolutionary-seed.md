---
type: permanent
created: 2026-07-07
tags: [meta-ml, adam, survivorship-bias, architecture-search, optimizer-coupling, scaling-laws]
status: seedling
confidence: medium
links: [su-2026-optimization-theory, llm-optimization-landscape, adam-coevolution-conjecture]
source: "Orabona (parameterfree.com) influential post"
---

# ML 社区是广义遗传算法：Adam 作为选择压力的元观察

> **后续深度调研**:详见 `references/adam-coevolution-conjecture.md`(Muon 出现后的系统性检验,含 OSP/BOCB/QK-Clip 等确证证据)

## 核心洞察

ML 社区的架构演进本质上是一个**广义遗传算法**——架构被"选择"，而**选择压力来自 Adam**。

**推论链**：
1. Adam 是一个重要的 **seed**（初始条件/选择算子）
2. 在 Adam 上表现不好的 architecture 被 discard（自然选择）
3. 因此我们探索的 (architecture, optimizer) 空间只是**小小一隅**
4. 所有结论——包括 scaling laws——都**隐式依赖这个区域**
5. 这个区域在完整空间里到底好不好？**未知**

## 类比结构

```
遗传算法           ↔  ML 社区
─────────────────────────────────
个体/基因型        ↔  神经网络架构
适应度函数         ↔  Adam 下的 benchmark 表现
选择算子           ↔  论文发表 + 开源采用
变异/交叉          ↔  架构微调（attention 变体、norm 位置等）
```

## 这与 Muon/Scion 线的关系

Bernstein 的"最速下降 under a norm"统一视角揭示了一个关键事实：**不同的 norm 选择 → 不同的最优架构**。

- 如果 Muon（谱范数）真的 2× 于 AdamW，那可能存在一批**专门为谱范数优化器设计的架构**，在 Muon 下远优于当前所有架构
- 但因为社区从未用 Muon 做大规模架构搜索（Muon 2024 才出现），这些架构从未被发现
- 当前的 "最优" transformer 变体可能只是 **"Adam-最优"**，不是真正的最优

## 与 scaling law 的关系

Scaling laws（Kaplan 2020, Chinchilla 2022）都是在 Adam + 标准 transformer 下拟合的。如果：
- 换优化器 → 不同的 $L(N, D, C)$ 曲面
- 换架构 → 不同的 power-law 指数

那所有 scaling law 结论都有**条件依赖**：$L = f(N,D,C \mid \text{Adam, standard transformer})$。跨出这个区域，law 可能完全不同。

## 开放问题
→ 有没有人在非 Adam 优化器下做过大规模架构搜索？（Muon 出现后的自然实验）
→ "Adam-最优架构"和"SGD-最优架构"有多大的重叠？（早期有一些工作，但 transformer 时代似乎没有系统对比）
→ 这个"小小一隅"的边界在哪里？能否估计 (architecture, optimizer) 空间的有效维度？
→ 如果 Muon/Scion 真的被广泛采用，是否会触发新一轮架构进化？

## 元层次
这个观察本身也是 ML 社会学的有趣案例：**工具塑造了被工具优化的对象**——不是工具适应问题，而是问题（被选择的架构子集）适应了工具。
