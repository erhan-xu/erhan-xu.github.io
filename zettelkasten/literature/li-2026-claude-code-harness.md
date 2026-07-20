---
type: literature
created: 2026-07-07
tags: [agent-design, harness, claude-code, prompt-cache, multi-agent, memory, safety, anti-distil]
status: seedling
confidence: medium
links: []
source: "https://01.me/2026/04/claude-code-harness-genai-2026/#more"
authors: [Li Bojie]
---

# Claude Code Harness 设计与 Agent 工程要点

## 核心洞察

### 1. Prompt Cache 是物理约束，不是优化点
- 系统设计**最开始**就要考虑 prompt cache 的物理限制
- 不是后期性能优化，而是架构决策的硬约束
- 影响：context window 管理、tool description 设计、system prompt 分层

### 2. 多 Agent 架构 & Compact
- compact 机制的细节（context 压缩/摘要策略）
- 多 agent 间的协调与 context 传递

### 3. Memory 设计：FS of Markdown
- 文件系统即 memory——markdown 文件作为持久化层
- 与我们的 Zettelkasten 方案同源，但 Claude Code 的 harness 把这个做成了 agent 的 core infra

### 4. Agent 安全
- **错误恢复**：graceful degradation, retry 策略
- **能力分区**：不同 agent/工具的能力隔离
- **Ablation study**：系统性拆解各组件贡献
- **Bun / GrowthBook**：编译时 flag + 运行时 flag，精细控制能力开关

### 5. Anti-Distillation 机制
- 防止模型输出被蒸馏/复制的保护策略
- 恶心但现实——商业护城河的技术实现

### 6. AI Native Company & 人类价值
- 人类的特殊价值与护城河在哪里？
- 哪些角色/能力可取代，哪些不可
- AI native 公司的组织形态

## 开放问题
→ Prompt cache 的物理限制具体是什么量级？对 harness 设计有哪些具体约束？
→ Compact 策略是摘要式还是选择式？信息损失如何量化？
→ Anti-distil 的实际效果如何？有无已知的绕过方式？
→ "FS of markdown" 与我们 Zettelkasten 的差别在哪？harness 层如何消费这些文件？
