---
type: literature
created: 2026-07-07
tags: [optimization, convex-analysis, learning-rate, convergence, information-theory, llm-training]
status: seedling
confidence: medium
links: []
source: "苏剑林 · 科学空间 · 让炼丹更科学系列（前三篇）"
authors: [苏剑林 (Su Jianlin)]
---

# 苏剑林「让炼丹更科学」优化理论系列（前3篇）

## 核心内容

### 1. 关键恒等式
- **last-iterate vs average-iterate**：关于末次迭代值与平均迭代值之间的恒等式
  - 这个关系很有意思——收敛分析的两种不同视角
  - average-iterate 通常有更好的理论保证，但 last-iterate 才是实际使用的
- （手写笔记中有更详细推导，待整理）

### 2. 强假设的必要性 & 如何 argue
- **凸性 (convexity)**：大部分经典优化理论需要 loss function 凸性
- 非凸情形下理论保证大幅削弱
- LLM 的 loss landscape 高度非凸——理论如何 bridge 这个 gap？

### 3. 学习率选择 → 不同收敛速度
| 策略 | 衰减形式 | 收敛速度 |
|---|---|---|
| Constant | $\eta = \text{const}$ | — |
| Sqrt-t decay | $\eta_t \propto 1/\sqrt{t}$ | 经典 |
| 数学游戏法 | $\eta_t \propto 1/\sqrt{t \ln t}$ | 略有改进 |

每种对应不同的收敛速率保证。

### 4. 收敛速度下界（信息论不等式）
- 论文涉及收敛速度的 **下界**（不是上界——说明"不可能更快"）
- 作者：**Wainwright**（高维统计大爹）+ **Agarwal**
- 看起来很难，但应该非常有意义
- 连接了信息论与优化的 fundamental limits

## 与 LLM 训练的接口
- 经典优化理论（凸、smooth、SGD）vs LLM 训练（非凸、AdamW、cosine schedule）
- 理论 gap 有多大？哪些结论能 carry over？
- last-iterate vs average-iterate 在 LLM fine-tuning 中的实际含义？

## Landscape 全景文档（Claude 整理，37KB）
→ `references/llm-optimization-landscape.md`

核心结构：**2×2 矩阵**（横轴 = schedule vs optimizer/norm；纵轴 = 凸理论 vs scaling law）

五条线：
1. **Schedule 凸理论**（苏神正脉）— 核心不等式 + 线性衰减最优
2. **Last-iterate 上下界** — Orabona / Zamani-Glineur / Agarwal-Wainwright
3. **Schedule ⟷ 权重平均对偶** — Defazio Schedule-Free / WSM / D2Z
4. **Optimizer/norm 几何** — Muon (谱范数) / Scion / Bernstein 统一视角
5. **超参迁移 / scaling** — µP / u-µP / weight decay 修正

桥：Schaipp 2025 — 凸理论竟然能定量预测 LLM loss 曲线

**开放缺口**（与 RL/alignment 直接相关）：
- RL post-training（GRPO/PPO）的 LR schedule 理论**几乎空白**
- Schedule/平均对偶在 policy 空间的对应物是否存在？

## TODO
- [ ] 等 Claude 整理的 LLM + 优化 landscape 文档到了再细化 → ✅ 已收到，见上方
- [ ] 整理手写笔记中的恒等式推导
- [ ] 找 Wainwright & Agarwal 的收敛下界论文精读
- [ ] 细读 landscape 文档中五条线的关键论文
