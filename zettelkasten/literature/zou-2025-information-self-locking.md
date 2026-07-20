---
type: literature
created: 2026-07-08
tags: [rl, active-reasoning, agent, credit-assignment, reward-shaping, POMDP, self-locking, coding-agent, process-reward]
status: evergreen
confidence: high
links: [cai-shenfeld-2025-on-policy-rl-dynamics, gradient-2024-shape-symmetry-structure]
source: "https://arxiv.org/abs/2603.12109"
authors: [Zou, Chen, Feng, Li, Li, Gong, Cheng]
year: 2025
venue: "ICML submission (predecessor T^3 = ICLR 2026 Oral)"
---

# On Information Self-Locking in RL for Active Reasoning of LLM Agents

> Zou, Chen, Feng, Li, Li, Gong, Cheng, arXiv 2603.12109 (投 ICML)
> 前作：$T^3$ = *Truncating Belief-Trapped Trajectories*, arXiv 2510.12264 (ICLR 2026 Oral)
> 团队：CUHK (James Cheng) × Kun Zhang 因果组 (CMU/MBZUAI) × Georgia Tech (Pan Li) × 字节
> 代码：`github.com/unimpor/T3` (verl fork), 数据 HF `WorkingOut/T3_data`

**核心论断**：仅以 outcome reward 训练主动推理 agent 会导致信息自锁 (information self-locking) —— agent 逐渐停止提出有信息量的问题，且不再将已获取的反馈整合进信念。作者将 agent 能力分解为 Action Selection (AS) 与 Belief Tracking (BT)，通过零和逐步 critique 对 advantage 进行重分配 ($\hat A_t = A_t + \lambda u_t$) 以破锁。方法极简，工程扎实；理论部分本质为二维线性动力系统的局部分析，形式意义大于实质。

---

## 1. 任务定义与实验域

**主动推理 (active reasoning)**：任务信息不完备，agent 需通过多轮交互主动查询以补齐缺失信息，最终作答。共同结构为"反复提问 → 环境反馈 → 维护信念向量 → 稀疏 outcome reward"。

**Preference Estimation (PE-G / PE-F)**：电影偏好估计。用户具有隐藏偏好 $\mathbf w^\star\in\mathbb R^D$，agent 每轮选择一对电影及属性子空间进行比较，从反馈中还原 $\mathbf w^\star$。
- PE-G (gated)：属性子空间受限 ($S=2/3$)，侧重考察 AS
- PE-F (full)：全属性可用 ($D=6/8$)，侧重考察 BT
- $\mathbf w^\star$ 已知 → AS 信息量与 BT 准确性可精确计算，为主诊断床

**MediQ (Medical Diagnosis)**：基于原子事实的 4 选 1 诊断任务，agent 向 LLM 模拟病人 (Qwen2.5-14B) 问诊，病人仅回应已问事实。检验合成设定外的迁移性。

**FloDial (Troubleshooting)**：基于流程图的排障对话，agent 提出 yes/no 诊断问题，模拟用户按参考表回答 Yes/No/Unknown。分 Easy/Hard，验证跨域普适性。

---

## 2. 问题诊断：信息自锁的机制分析

### 2.1 现象

仅以 outcome reward 训练时，reward 上升但 agent 行为退化：不再提出有信息量的问题，且不将反馈整合进信念。关键证据：MediQ 中将病人反馈全部替换为 "Unknown" 后，RL 训练模型性能下降幅度反而更小（表明 agent 已不依赖交互），同时信念一致性从 78.7 升至 92.8。

### 2.2 AS/BT 分解与遮蔽效应

- **Action Selection (AS)**：提问策略 $\pi^Q_\omega(\cdot\mid b_t)$，决定信息流
- **Belief Tracking (BT)**：信念更新 $b_{t+1}=\pi^U_\omega(b_t,a_t,o_t)$，决定反馈整合

**核心洞察 —— 遮蔽 (masking)**：固定同一提问序列，将反馈分别交给 agent 自身的 BT 与更强的 BT（人工规则/前沿模型），发现 AS 质量与最终 reward 的相关性在强 BT 下显著更高。这表明"好问题不涨分"的根因并非问题本身差，而是弱 BT 遮蔽了 AS 的信用 —— 这是 credit assignment 层面的解释，比笼统的"探索崩溃"更为精细。反向地，AS 过于保守使 BT 缺乏可学习的证据。两者互相拖累 → 自锁。

### 2.3 POMDP 形式化

主动推理建模为 POMDP $(\mathcal S,\mathcal Q,\mathcal O,T,O,R,\gamma)$，其中 $\mathcal S$ 为不可观测的隐状态。信念 $b_t\in\Delta(\mathcal S)$ 为历史的充分统计量，将 POMDP 转化为 belief-MDP (Åström 1965; Kaelbling–Littman–Cassandra 1998)。AS/BT 分解对应控制论中估计器/控制器的经典分离 (separation principle) —— 概念上自然。

需注意：LLM 无显式 belief，AS/BT 切分为 post-hoc 人为叠加，proxy 度量逐任务手工设计。

### 2.4 与 reward shaping 的关系

自锁根因为稀疏 outcome reward 无法为中间步骤分配信用。Potential-based reward shaping (PBRS) $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ 是唯一保证不改最优策略的中间奖励形式 (Ng–Harada–Russell 1999)。AReW 借鉴 PBRS 的差分/零和直觉，但**故意不满足保最优条件** —— 自锁的动机正是要改变原有的坏 fixed point。

---

## 3. 算法：AReW

### 3.1 方法

**Step 1 — 逐步 critique**：偶数步 (AS) 产出 $z^Q_t\in\{-1,0,+1\}$（+1 问出新信息，−1 无信息/重复）；奇数步 (BT) 产出 $z^U_t=\mathrm{sign}(\hat\Psi_{t+1}-\hat\Psi_t)$（置信读数升降）。各数据集中 critique label 来自环境 oracle（PE 用真 $\mathbf w^\star$，MediQ 用反事实对照，FloDial 规则化）。

**Step 2 — 零和归一化**：$u_t\in\{+\frac1{|\mathcal P|},-\frac1{|\mathcal N|},0\}$，$\sum_t u_t=0$。

**Step 3 — Advantage 重加权**：
$$\boxed{\;\widehat A_t \leftarrow A_t + \lambda\,u_t\;}$$
$\lambda$ 逐任务调参 (PE-G 0.2, PE-F/MediQ 1.0, FloDial 0.5)。Reward、critic、RL 算法均不变。

### 3.2 零和的安全性质 (梯度视角)

将 $u_t$ 分解为共模 + 差分 $u_t=\bar u+\tilde u_t$，新增梯度中 $\bar u\,\nabla\log p(\tau)$ 项因 $\sum u_t=0\Rightarrow\bar u=0$ 被消除。含义：错误 critique 最坏仅在轨迹内部重分配梯度，无法对整条轨迹施加净推力，轨迹整体方向仍由真实 $A_t$ 决定 → 防 proxy-hacking。

### 3.3 非 potential-based 的 reward shaping

AReW 可精确写为 reward shaping ($F_t=\lambda(u_t-u_{t+1})$ telescope 回 $\lambda u_t$)，但不满足 PBRS 条件：(i) BT 的 sign 量化破坏 telescoping；(ii) $u_t$ 由整轨迹计数归一化，非 Markov；(iii) AS critique 依赖历史。因此不享有保最优定理 —— 这是有意为之。Prop 4.1 给出数据驱动的精度保证：一阶增益 $\propto (2\,\mathrm{Acc}-1)$，critique 排序准确率 $>1/2$ 即朝正向。

### 3.4 理论评价 (Thm 3.4)

Thm 3.4 将 AS 质量 $\mathcal I_{\text{th}}$ 与 BT 质量 $C_{\text{BT}}$ 的一步漂移建模为
$$\begin{pmatrix}\Delta_Q\mathcal I_{\text{th}}\\ \Delta_U C_{\text{BT}}\end{pmatrix}\preceq \eta\begin{pmatrix}0&\alpha\\ \beta_I&\beta_C\end{pmatrix}\begin{pmatrix}\mathcal I_{\text{th}}\\ C_{\text{BT}}\end{pmatrix}+o(\eta).$$

左上角 0 的含义："AS 不能自举，只能依赖 BT"。但该定理存在以下局限：
- 为漂移**上界** ($\preceq$)，非精确动态；局部一阶 ($o(\eta)$)，仅在小邻域成立
- 承重假设 (iii)（自毁漂移与 query 无关）在真实 LLM 中基本不成立 —— 确认偏误会以依赖 query 的方式破坏信念，放松此假设则 AS 可自举，陷阱不再是严格陷阱
- 假设未经实证检验，常数 $\alpha,\beta$ 未量化

**结论**：理论是装饰性的。AReW 作为"绕开坏管道的稠密信号注入"这一 heuristic 不依赖该定理即可成立。实质贡献在 masking 诊断 (§2.2) 与工程 (§4)。

---

## 4. 实验

**设置**：3 域 7 任务；策略模型 Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct；PPO + GAE + 学习式 critic；rollout $n=1$ (on-policy)；8 卡单节点。

| | PE-G S=3 (Qwen) | PE-G S=3 (Llama) |
|---|---|---|
| Vanilla PPO | 18.3 | 11.0 |
| AReW-AS only | 32.0 | 73.0 |
| AReW-AS+BT | **80.3** | **77.7** |

**正面结果**：7 任务 × 2 模型上 AReW 全部正向提升，AS+BT > AS-only > vanilla 单调（消融干净）；训练动态上 AS/BT 均恢复增长。

**可靠性问题**：
1. **Critique 依赖 oracle**：主表 $z_t$ 使用真 $\mathbf w^\star$/真诊断/真故障，"easy-to-obtain" 实为特权信息，无 ground-truth 的真实任务中卖点不成立
2. **Baseline 单薄**：仅对比 vanilla PPO，缺少 process-reward / turn-level advantage / entropy bonus 等对照，无法排除"任意稠密信号均有效"的可能
3. **无多 seed / error bar**：单 run 结果，variance 未知；vanilla 本身存在不稳迹象

**Tau2Bench**：唯一非 oracle 实验 (telecom 客服工单)，critique 从 agent-工具-用户交互轨迹弱推（命中预期动作 = 正，工具报错/重复/无效 = 负），是"critique 在现实设定中可廉价获取"的最强证据。使用 `step_abs_sum` 归一化变体。

**与前作 $T^3$ 的关系**：同一问题的两种干预。$T^3$ 为减法（检测信念陷阱后截断尾部，省 token）；AReW 为加法（注入零和 critique 重分配信用）。互补，可叠加。

---

## 5. Coding Agent 领域的现象验证与方法迁移边界

本节系统评估 AReW/$T^3$ 框架在 coding agent 任务中的适用性。遵循"立现象 → 划边界 → 评估迁移"的分析路径。

### 5.1 现象层：自锁行为被多篇独立记录

Coding agent 中存在与 self-locking 高度相似的行为模式，已被多个研究团队独立观察到：

**循环卡死与提前放弃**
- 连续三轮以上使用相同参数调用同一工具被定义为"重复动作"，在不同模型中分别出现于 12.4% / 28.9% 的样本，且常作为"直接放弃"的前置触发器 (arXiv:2601.15679)
- SWE-EVO 的失败分类体系中明确包含 *Stuck in Loop* 与 *Gave Up Prematurely* 两类 (arXiv:2512.18470)

**信念追踪失败**
- MIRAGE-Bench 观察到 agent 误读反馈、遗忘先前动作、推理不一致、陷入无效循环等现象，而人类在类似失败后会主动调整策略 (arXiv:2507.21017)

**二维漂移刻画的独立验证**
- TACT 研究 coding agent 的 *agent drift*，聚焦 **overthinking** (反复处理已有信息) 与 **overacting** (不整合观测即行动) 两类失败
- 发现隐藏态沿这两条轴线性可分 (AUC≈0.9)，与 AS/BT 分解形成独立旁证 (arXiv:2605.05980)

**Outcome reward 与 process credit 的错位**
- 在 1750 条 SWE-bench 轨迹上，submit rate 与 test-verified resolve rate 严重背离 (GPT-5 提交率 100% 但解决率仅 44%)
- 失败集中于 *silent semantic failure* ——"同一误读被重复而非随机错误"，占失败样本的 68-80% (arXiv:2603.25764)
- 同一研究的探针实验显示，面对已修复的 bug，多数模型仍会修改代码，暗示缺乏"何时不应行动"的判断能力

### 5.2 边界层：三处结构性错位

尽管现象层存在相似性，直接将 AReW 框架迁移至 coding agent 面临以下结构性障碍：

**1. 信念状态不可度量**
- PE/MediQ 中信念为低维向量 (≤4 个假设)，BT 质量可通过对真假设的 margin 变化精确量化：$\Psi(b)=b(s^\star)$
- Coding agent 的"信念"是对代码库与 bug 的高维弥散理解，无 ground-truth 隐状态，无干净 potential function
- **可迁移的是 AS/BT 分析透镜，而非论文的 proxy/oracle 度量机制**

**2. 失败谱系更宽：overacting 比 self-locking 更致命**
- Self-locking 特指"被动" (不探索 + 不吸收)
- Coding agent 同样普遍且更致命的是 **overacting / action bias / silent semantic failure** ——"主动但错误"
- TACT 的两轴结构表明 coding agent 退化张成更大空间，论文框架仅覆盖"绕圈/放弃"这一半

**3. 时态错位：推理时 vs 训练时**
- 论文的 self-locking 机制针对 RL 训练过程中的坍缩
- 上述实证证据多来自推理时行为观察
- 训练时的对应现象 (如 RAGEN 报道的 "reasoning collapse in agentic RL") 证据较为薄弱
- 讨论"coding RL 会自锁"时需明确区分**推理时绕圈**与**训练时坍缩**

### 5.3 可迁移组件评估

基于上述边界，以下组件具有迁移可行性：

**AS/BT 二轴分析框架**
- AS: 读哪个文件 / grep 什么 / 跑哪个测试 / 问澄清问题
- BT: 维护"代码库状态 + bug 假设 + 已排除项"的内部模型
- TACT 的独立验证佐证了这一分解的自然性

**BT 反幻觉规则**
- "无新证据则不改结论/不行动"
- 比 AS proxy 更具可迁移性，不依赖任务特定的 ground-truth
- 已被独立研究以不同形式提出 (arXiv:2603.25764)

**Outcome 定义修正**
- 按 test-verified resolve rate 而非 submit rate 计分
- 实证表明两者严重背离，outcome 定义本身需要审慎选择

**诊断性埋点**
- 分别追踪"是否在获取新信息"与"是否在正确更新计划"
- 两条曲线背离可作为卡死前兆信号，比仅观察最终成功率更早发现问题
- 此方法不依赖 RL 训练，可用于纯推理场景

**零和重分配原则**
- 若需提供稠密过程信号，采用轨迹内零和 ($\sum u_t=0$) 而非直接加正 reward
- 避免 agent 刷测试 / 反复修改文件等 reward hacking 行为

### 5.4 $T^3$ 与 AReW 在 coding 场景的对比评估

$T^3$ (arXiv:2510.12264, ICLR 2026 Oral) 在 5 个 active-reasoning 任务上验证：检测 Belief-Trap Region 后早截断坏尾、保留信息量高的前缀，可同时提升训练稳定性与最终性能，token 成本最多降低 34%，且跨多种 RL 算法成立。

**$T^3$ 在 coding 场景的适配性优势**

| 维度 | $T^3$ | AReW |
|------|-------|------|
| **触发信号** | 可直接观测 (连续 $N$ 次相同工具调用 / edit 后文件状态不变 / 测试输出无信息 / 反复失败 patch) | 主表依赖 oracle ($\mathbf w^\star$ / 真诊断 / 真故障) |
| **判断成本** | "是否在绕圈" — 程序化可检出 | "信念更新是否正确" — 需 ground-truth |
| **算力收益** | 直接省 token (−34%)，coding rollout 长且贵时收益更大 | 无直接算力节省 |
| **信用净化** | 截断绝望尾部，避免污染 RL 回报信用 | 在前缀主体内重分配信用 |
| **算法耦合** | 算法无关，可直接接入 GRPO/PPO 管线 | 需修改 advantage 计算 |

**核心判断**：$T^3$ 要判断的"是否在绕圈"比 AReW 要判断的"信念更新是否正确"在 coding 场景便宜得多、鲁棒得多；而绕圈恰恰是 coding 中最有据、最耗 token 的失败模式 (§5.1)。

**$T^3$ 的局限**：截断无法处理 overacting 类失败 (silent semantic failure —— 自信改错、submit 过但 test 不过)。这类失败需要 BT 反幻觉规则 + test-verified 计分。

**两种方法的分工与顺序**
- $T^3$: 砍掉绝望尾部 (治绕圈/放弃)
- AReW: 在存活前缀主体内重分配信用 (给好中间步补分)
- 两者互补，均仅使用可观测弱信号
- 建议顺序：**先 $T^3$ 后 AReW** —— 先止血 (省 token、去污染)，再精修 (补 credit)

---

## 总结

**核心贡献**：AS/BT 分解与遮蔽诊断 (§2.2)；零和 shaping 防 reward hacking (§3.2–3.3)；Tau2Bench 式非 oracle 弱信号 (§4)。

**局限**：理论为二维线性玩具 (§3.4)；主表 critique 依赖 oracle、baseline 单薄、无 variance (§4)。

**资料**：论文 arXiv 2603.12109；前作 $T^3$ arXiv 2510.12264 (ICLR 2026 Oral)；代码 `github.com/unimpor/T3` (verl fork)，数据 HF `WorkingOut/T3_data`；背景文献 Ng, Harada & Russell 1999 (PBRS)，Kaelbling, Littman & Cassandra 1998 (POMDP)。