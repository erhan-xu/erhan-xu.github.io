---
type: literature
created: 2026-07-13
tags: [rl, post-training, layer-contribution, mechanistic-interpretability, optimization, single-layer-training]
status: seedling
confidence: high
links: [cai-shenfeld-2025-on-policy-rl-dynamics, park-2024-linear-representation-hypothesis]
source: "Zhang et al. 2026 (arXiv:2607.01232)"
authors: [Zhang, Hu, Glentis, Li, Yau, Lin, Hong]
year: 2026
arxiv: 2607.01232v2
affiliations: [University of Minnesota, Peking University, Amazon]
---

# 论文阅读：Is One Layer Enough?

## 内容总览

**发表时间**：2026-07-01（v2: 2026-07-02）

**发表团队**：University of Minnesota + Peking University + Amazon 

**论文地址**：https://arxiv.org/abs/2607.01232

**成果概括**：发现<font style="color:#DF2A3F;">单独训练一个中间 Transformer 层可以恢复甚至超越全参数 RL 训练的效果</font>，引入 Layer Contribution 量化指标，跨 7 个模型、3 种 RL 算法、多任务域验证了"中间层主导"的稳定模式。

**文章结构：**
- Section 2: Layer Contribution 定义与实验设置
- Section 3: 主要发现（4 个核心发现）
- Section 4: Layer contribution-guided 训练策略
- Section 5: 相关工作
- Section 6: 讨论与结论

## 核心概念：Layer Contribution

$$C(k) = \frac{S_k - S_{base}}{S_{full} - S_{base}}$$

- $S_k$: 只训练第 k 层时的得分（其他层冻结）
- $S_{base}$: 基础模型得分（未训练）
- $S_{full}$: 全参数 RL 训练得分

**物理意义**: 单独训练第 k 层能恢复多少比例的完整 RL 训练收益。

> **Erhan's remark**: 比较 naive 的构造。注意实际上层贡献度依赖于模型（size 和结构）、训练集和测试集，即 $C = C(\pi_\theta, D_{train}, D_{test})$。这意味着 C(k) 不是一个稳定的模型属性，而是训练-测试配置的函数。

![Layer Contribution 概念图](/images/is-one-layer-enough/custom_concept_diagram.png)

## 背景铺垫：RL Post-Training 工作流

在介绍本文之前，先回顾目前标准的 RL post-training 做法——

1. **选择 base model**：通常是 SFT 后的模型（如 Qwen3-Base）
2. **选择 RL 算法**：GRPO、GiGPO、Dr. GRPO 等
3. **全参数更新**：所有层同时训练，<font style="background-color:#FBDE28;">隐含假设每层贡献类似</font>
4. **评估**：在数学/代码/Agent 基准上测试

本文挑战了第 3 步的隐含假设。

## 核心发现

### 发现 1: Single Layer ≈ Full RL

单独训练一个中间层可以恢复 80-100% 的全参数 RL 收益，在某些情况下甚至超越：

| Model | Base | Full RL | Best Single Layer | C(k)_max |
|-------|------|---------|-------------------|----------|
| Qwen3-1.7B | 44.1 | 52.2 | 50.8 | 0.97 |
| Qwen3-4B | 58.0 | 63.0 | 64.3 | **1.13** |
| Qwen3-8B | 66.4 | 65.9 | 69.1 | **1.18** |

![Figure 1(a): Layer Contribution across all models](/images/is-one-layer-enough/fig1a.png)

### 发现 2: Middle Layers Dominate

高贡献层集中在网络的 **40-60%** 深度位置。输入层和输出层贡献显著较低。

![Cross-model consistency](/images/is-one-layer-enough/fig2.png)

### 发现 3: Universal Pattern

模式跨以下维度保持一致：
- 模型规模 (1.7B / 4B / 8B)
  - <font style="color:#DF2A3F;">但注意细节差异</font>：1.7B 下初始层贡献度也很大，而 8B 下初始层贡献度 < 0（tune 这层比 base 更差）。尽管都有 Middle Layer Dominance，至少初始层的 pattern 随规模已经发生了明显变化。
- 模型家族 (Qwen3 / Qwen2.5)
  - <font style="color:#8A8F8D;">⚠️ 本质全是 Qwen，且全是 Dense 没有 MoE。选择 DeepSeek-Distilled-Qwen-7B 作为另外的模型家族比较 suspicious——Why not Llama/Gemma？此外，并没有使用之前主实验的 Numina-CoT Dataset，未控制变量。</font>
- RL 算法 (GRPO / GiGPO / Dr. GRPO)
  - <font style="color:#8A8F8D;">⚠️ 不同模型用不同算法，没有控制变量，也没有考虑 GPG、GSPO、PPO 等其他算法。</font>
- 任务类型 (数学推理 / 代码生成 / Agent 决策)
  - <font style="color:#8A8F8D;">仔细读图可以发现，同样是数学训练集，DeepScaleR 的末尾层出现了大量的负贡献度，但在 Numina 上仍然是非常正的。一个数据集上的层贡献度为负无法说明该层对某个任务完全无效——间接说明这个指标的设计比较 naive，对数据集很敏感。</font>

![Figure 5: Agentic task](/images/is-one-layer-enough/fig5.png)
![Figure 6: DeepSeek-Qwen-7B](/images/is-one-layer-enough/fig6.png)

### 发现 4: Weight Change Concentration

全参数训练时权重变化也集中在中间层，但单层训练时变化完全局限在训练的那一层。

> **Erhan's insight**: 可能层的平均参数变化（2-范数）的粒度还是太粗了。按照 <font style="background-color:#FBDE28;">Manifold 假设</font>，对应的参数构成的几何应该会有特定的变化，可能需要考虑其他能够体现几何特征的 measurement（如 geodesic distance、curvature）才有机会发现和 Mid Layer Dominance 相一致的不同层参数变化的异质性。

**关于 OOD 泛化的补充**：仔细读表可以发现，日常推理和语言作为 OOD 测试集时的 Layer Contribution 可能是比较 uniform 的。Coding 有一定的 Middle Layer Dominance，但只有 Math 的 variety 是最显著的——这也许是为什么作者只汇报了 Math 单独的层贡献度和四项平均的层贡献度。（如果在 language and general reasoning 上训练，还有一样的 pattern 吗？未必）

![Figure 9: Weight change magnitude](/images/is-one-layer-enough/fig9.png)

## 实用训练策略

![三种策略对比](/images/is-one-layer-enough/custom_three_strategies.png)

1. **Boosting**: 增加高贡献层学习率
2. **Selective**: 只训练高贡献层
3. **Ensemble**: 多层 specialized models 投票（Figure 8 展示 OlympiadBench 上进一步提升）

**具体实验配置**（对应 Figure 7 的颜色）：
- 蓝色：实际仍是全量更新，但学习率差异化。Top5/top10 高贡献层 LR=1e-5 (2x)，其他 5e-6
- 红色 1（对照组）：低贡献度层 LR=1e-5，其他 5e-6
- 绿色：固定只训练 top5/top10 贡献度层
- 红色 2（对照组）：worst-k 贡献度层
- 紫色：固定只训练中间 5 层

<font style="color:#DF2A3F;">结论：三种启发式 RL 算法都能超越全量 RL，但哪种最优随模型 size 变化，没有定论。</font> 本质上全是 post-hoc 研究——训练的事前并不能精确知道要训练的层的子集，只有 heuristic 算法。

![Training strategies comparison](/images/is-one-layer-enough/fig7.png)

## 实验流程图

![实验设置流程](/images/is-one-layer-enough/custom_experiment_flowchart.png)

## 🧠 个人评价

**优点：**
- 实验覆盖面广：7 个模型 × 3 种算法 × 多任务，结果高度一致
- Layer Contribution 定义简洁有力，是一个可复用的分析工具
- C(k) > 1.0 的发现非常有冲击力，直接挑战了"更多参数=更好"的直觉

**局限：**
- 缺乏机制解释——为什么是 40-60% 深度？只给出经验观察，没有理论或 mechanistic 解释
- 模型规模限于 ≤8B，不清楚在 70B+ 模型上是否仍然成立
- 单层训练只测了"一层"，没有测"2-3 层组合"的上界

**与我的研究的关联：**
- **RL 动力学**: 中间层可能是 RL adaptation 的"瓶颈层"，这与 Cai-Shenfeld 的 rank-1 结构假说可能有关——如果 RL 更新主要集中在某些 rank-1 方向，而中间层恰好对这些方向更敏感
- **Mechanistic Interpretability**: 可以用 SAE 分析高贡献层的 features，看它们是否学到了特定的"RL-relevant features"
- **Advantage structure**: 不同层的 advantage function 可能不同——中间层可能有更大的 advantage variance，因此 RL 信号更强
- **潜在研究**: Layer Contribution × Thought Anchors / SAE features / rank-1 weight structure

### 🔗 相关工作地图
- **前置工作**：GRPO (DeepSeek), GiGPO, Dr. GRPO, LoRA 层选择相关工作
- **后续进展**：待观察（论文刚发布）
- **平行工作**（详细横向对比）：

| 论文 | 探针 (metric 算子) | 训练/推理阶段 | "重要性"定义 | 声称的层结构 |
|------|-------------------|--------------|-------------|-------------|
| **本文** | 单层 RL 训练（只更新一层） | RL 后训练 | 单层能恢复多少全参 RL 增益 | 中层最高 |
| Nepal 2506.22638 | zero-ablation（整层置零） | 推理时 | 移除后掉多少分 | 少数中后层关键 |
| Song 2510.02091 | layer/head pruning | 推理时 | 移除后掉多少分（依 metric） | 依 metric 而变：似然→浅层，生成→中深层 |
| Shi 2410.17875 (ILA) | 学习可微二值掩码 | SFT 对齐 | 掩码把哪些层的 $\Delta\theta$ 保留 | 跨数据集稳定，冻结 ~25% 不重要层反而更好 |
| Zhang 2024 (BlackboxNLP) | zero-ablation（参数矩阵置零） | 推理时 | "cornerstone layers" 移除即崩 | 少数关键层 |
| Pan 2024 (LISA) | 权重范数 skewness → 重要性采样 | SFT | LoRA 下各层权重范数 | 两端(emb/head)范数大，中层小 |

<font style="color:#8A8F8D;">看起来矛盾的结论可能是因为不同的探针，以及一些比较微妙的训练集和测试集的异质性导致的。一些结论的 external generality 可能并没有论文自身 claim 的那么强。另外这些项目的代码开源也相对较差。</font>

## 实验设置细节 (Experimental Details)

### 核心实验：Layer Contribution 测量

**固定项 (Fixed):**
- RL 算法选择：GRPO (DeepSeek), GiGPO, Dr. GRPO — 不同模型用不同算法
- 训练数据：NuminaMath-CoT (数学), Skywork-math, ALFWorld (Agent)
- 评估基准：Math benchmarks (GSM8K, MATH, etc.), Code benchmarks, Agent benchmarks
- 训练轮数：主实验 Qwen3 系列为 **4 epochs**
- 学习率：与对应模型/数据集下，先搜索全参数 RL 的最优学习率，然后单层训练保持一致（**5e-6**）
- Batch size: 512 / minibatch 128，optimizer 未提及（大概率 AdamW）

**变化项 (Varied):**
- 每次只解冻一层，其他所有层冻结

### 超参数敏感性

**论文补充的信息（来自汇报 PPT）：**
- RL 训练 epochs: 4
- 学习率: 5e-6（全参数最优 LR，单层保持一致）
- Batch size: 512/minibatch 128
- Optimizer: 未明确（推测 AdamW）

**仍未报告的：**
- 学习率调度策略
- 是否做了超参数搜索（仅全参数 LR 搜索了，单层 LR 直接沿用）
- 随机种子和方差

**为什么这很重要：**
如果单层训练用了更多 steps 或更高学习率，那"单层≈全参数"可能是因为单层获得了更多计算预算，而不是因为该层真的更重要。

## 🧠 批判性分析 (Critical Analysis)

### 结论支持度评估

**强支持 (Strong Evidence):**
- ✅ "中间层贡献最大" — 跨 7 个模型、3 种算法、多任务都一致，robust pattern
- ✅ "C(k) > 1.0 在某些情况" — 数值证据清晰

**中等支持 (Moderate Evidence):**
- ⚠️ "单层训练≈全参数训练" — 需要确认实验条件是否公平
  - 如果单层训练用了相同 compute budget，结论成立
  - 如果单层训练用了更多 steps，可能是 overfitting 到单层
- ⚠️ "40-60% 深度的规律" — 经验观察，但没有理论解释
  - 为什么不是 30-50% 或 50-70%？
  - 是否与 transformer 的信息处理流程有关？

**弱支持 (Weak Evidence):**
- ❌ "机制解释" — 论文完全没有解释 **为什么** 中间层贡献最大
  - 是中间层表征能力更强？
  - 是中间层在信息流中的瓶颈位置？
  - 是中间层的梯度/可塑性更好？
  - 论文只给经验观察，没有 mechanistic 或 theoretical 解释

### 潜在问题 (Potential Issues)

**1. 实验公平性 (Experimental Fairness)**

```
问题：单层训练 vs 全参数训练是否用了相同的 compute budget?

场景 A：两者都用相同 steps/epochs
  → 结论可信：单层真的更高效

场景 B：单层训练用了更多 steps（因为只更新一层，每 step 更快）
  → 结论可疑：可能是因为更多 compute，而不是层本身更重要

论文没有明确说明，这是关键遗漏。
```

**2. 单层 vs 多层组合 (Single vs Multi-Layer)**

```
论文只测了"一层"，没有测"2-3层组合"。

问题：如果训练 top-3 层会怎样？
  - 可能更好（组合效应）
  - 可能更差（层间干扰）
  - 可能差不多（收益递减）

这限制了实用价值：如果 2-3 层组合更好，那"只训练一层"不是最优策略。
```

**3. 模型规模限制 (Scale Limitation)**

```
所有模型 ≤ 8B，没有 70B+ 的大模型。

问题：在更大模型上，pattern 是否仍然成立？
  - 可能：更大模型有更多冗余，单层贡献更小
  - 可能：更大模型的中间层更关键，单层贡献更大
  - 未知：需要实验验证

这限制了泛化性结论。
```

**4. C(k) > 1.0 的解释 (Interpretation of C(k) > 1.0)**

```
Qwen3-4B/8B 出现 C(k) > 1.0，单层超越全参数。

可能解释：
  A. 全参数训练 overfit 或陷入局部最优
  B. 单层训练避免了层间干扰，学到了更好的表示
  C. 全参数训练的超参数不是最优

论文没有深入分析，只报告了现象。

这对实践的意义：
  - 如果 A/B 成立：单层训练是更好的策略
  - 如果 C 成立：只是超参数问题，调好超参数后全参数仍然更好
```

### 与我的研究的关联 (Research Connections)

**直接可探索的问题：**

1. **为什么是 40-60% 深度？**
   - 假设：中间层是信息处理的"瓶颈层"
   - 验证：用 SAE 分析中间层的 features，看是否学到了"RL-relevant features"
   - 工具：SAE + circuit analysis

2. **Layer Contribution × Rank-1 Structure (Cai-Shenfeld)**
   - 假设：高贡献层对应特定的 rank-1 weight directions
   - 验证：分析 ΔW = W_trained - W_base 的 SVD，看是否 rank-1 主导
   - 预期：如果 rank-1 主导，说明 RL 更新集中在少数方向

3. **Advantage Structure Across Layers**
   - 假设：中间层的 advantage variance 更大，RL 信号更强
   - 验证：分析每层的 advantage function A(s,a) 的分布
   - 预期：中间层 advantage 更"尖锐"，学习更高效

4. **Thought Anchors × Layer Contribution**
   - 假设：高贡献层对应特定的 thought patterns
   - 验证：用 attention pattern 分析，看高贡献层是否关注特定 token
   - 预期：高贡献层可能 focus on "reasoning tokens"

### 缺失的分析 (Missing Analyses)

**论文应该但没有做的：**

1. **2-3 层组合实验** — 只训练多层 vs 单层 vs 全参数
2. **Compute budget 对比** — 确保单层和全参数用相同 FLOPs
3. **机制解释** — 为什么中间层贡献最大？（SAE, gradient analysis, attention pattern）
4. **超参数敏感性** — 学习率、steps 对 C(k) 的影响
5. **大模型验证** — 70B+ 模型是否仍然成立
6. **Layer interaction** — 解冻相邻层是否有组合效应

## 复现建议 (Reproduction Guide)

### 最小可行复现 (Minimal Viable Reproduction)

```python
# 验证 Qwen3-1.7B 上的 layer contribution
model = "Qwen3-1.7B-Base"
dataset = "NuminaMath-CoT"
rl_algo = "GRPO"
eval_bench = "GSM8K"

# Baseline
S_base = evaluate(load_model(model), eval_bench)  # ~44.1
S_full = train_and_evaluate(model, rl_algo, dataset, eval_bench, train_all=True)  # ~52.2

# Layer contribution for middle layer (e.g., layer 14/28)
for k in [14, 15, 16]:  # 中间层
    model_k = load_model(model)
    freeze_all_except(model_k, layer_k)
    S_k = train_and_evaluate(model_k, rl_algo, dataset, eval_bench)
    C_k = (S_k - S_base) / (S_full - S_base)
    print(f"Layer {k}: C(k) = {C_k:.2f}")
```

**关键控制变量：**
- 固定学习率（如 1e-5）
- 固定训练 steps（如 1000 steps）
- 固定 batch size
- 记录每层的训练时间（单层应该更快）

**预期计算资源：**
- 单层训练：~2-4 A100 hours/layer
- 全参数训练：~8-16 A100 hours
- 总共：~100 A100 hours（如果扫所有 28 层）

### 关键复现问题 (Key Reproduction Questions)

1. **训练 steps 是多少？** — 论文没说，需要联系作者或猜
2. **学习率是否统一？** — 如果不同层用不同 LR，结果不可比
3. **是否做了多次实验取平均？** — 如果只跑一次，方差可能很大
4. **代码是否开源？** — 截至 2026-07-13，未见开源代码

## 总结评价 (Overall Assessment)

**优点：**
- 实验覆盖面广，pattern robust
- Layer Contribution 定义简洁，可复用
- C(k) > 1.0 的发现冲击力大

**主要缺陷：**
- **缺乏机制解释** — 只给经验观察，没有回答"为什么"
- **实验细节不充分** — 关键超参数（steps, LR）未报告
- **没有多层组合实验** — 限制了实用价值
- **模型规模有限** — 只在 ≤8B 上验证

**对研究的价值：**
- **高**：提供了一个新的分析工具 (Layer Contribution)
- **高**：揭示了 RL training 的 layer-wise 不均匀性
- **中**：实用指导有限（没有说"最优策略是什么"）
- **低**：理论深度不足（没有解释机制）

**下一步研究方向：**
1. 用 mechanistic interpretability 工具分析高贡献层
2. 探索 layer contribution × rank-1 structure 的关系
3. 设计基于 layer contribution 的自适应训练策略
4. 在大模型 (70B+) 上验证 pattern
5. **Layer Contribution 的参数化建模**：如果考虑层的相对位置（归一化到 $[0,1]$），或许可以将 $C(k/L)$ 参数化成 <font style="background-color:#FBDE28;">Beta 分布</font> $\text{Beta}(\alpha, \beta)$，用 $\alpha, \beta$ 两个参数刻画不同模型/算法/数据集下的"中间层集中度"，比逐层报告 $C(k)$ 更紧凑，也方便跨设置比较。

---

**ZK Links**: #RL #post-training #layer-contribution #mechanistic-interpretability #optimization
