---
type: fleeting
created: 2026-07-07
tags: [randomized-linear-algebra, preconditioning, hadamard, numerical-methods]
status: seedling
confidence: low
links: []
source: "random browsing"
---

# Randomized Linear Algebra 浅触

## 了解到的点

- **Randomized Hadamard preconditioning**：用随机 Hadamard 变换作为预条件，替代 Cholesky / QR 分解
- 核心思想：Hadamard 变换可以"搅匀"矩阵的条件数分布，使得后续求解更快收敛
- 属于 **randomized numerical linear algebra** 这个大类（Halko, Martinsson, Tropp 2011 是经典 survey）

## 为什么有意思
→ 如果 Muon 的 Newton-Schulz 正交化本身就是一种预条件，那 randomized Hadamard 可能是更便宜的替代？
→ 和 Cai-Shenfeld rank-1 的潜在联系：随机投影能否揭示权重更新的低秩结构？

## 待做
- [ ] 找 Halko-Martinsson-Tropp survey 速读
- [ ] 了解 randomized SVD 和 randomized Hadamard 的关系
