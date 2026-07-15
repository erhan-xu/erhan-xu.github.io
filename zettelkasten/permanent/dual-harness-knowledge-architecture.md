---
type: permanent
created: 2026-07-06
tags: [zettelkasten, workflow, hermes, chezmoi, architecture]
status: seedling
confidence: high
links: []
---

# Dual Harness 知识管理架构

## Core Idea (one paragraph)

个人知识管理分为两层：**配置层**（Hermes skills, cron, SOUL.md → chezmoi dotfiles）和**内容层**（Zettelkasten 笔记 → 独立 git repo）。配置变更慢、需版本控制；内容增长快、需独立管理。企业凭证（API keys, tokens）永远不进 git，通过 setup script 在新机器上交互式配置。

## Elaboration

### 为什么双仓库？

1. **变更频率不同**
   - 配置（skills, cron, templates）：月级别变更
   - 笔记（literature, permanent）：日级别增长
   - 混在一起会导致 git history 噪音大

2. **体积差异**
   - 配置：< 1MB（skill markdown + 小脚本）
   - 笔记：可能增长到几百 MB（PDF, 图片, 大量 markdown）
   - 独立仓库可以独立 clone/sync 策略

3. **安全边界**
   - dotfiles 可能被公开（你的 dotfiles repo 是 public）
   - Zettelkasten 是私有的研究笔记
   - 凭证在两者之外（.env 文件）

### 安全模型

```
dotfiles/ (public, chezmoi)
├── .hermes/config.yaml.tmpl    ← 模板，无密钥
├── .hermes/skills/              ← 可公开的工作流
├── .hermes/scripts/setup-env.sh ← 交互式向导，不含实际密钥
└── .chezmoiignore               ← 排除 .env, memory, sessions

zettelkasten/ (private, independent)
├── inbox/
├── literature/
├── permanent/
└── references/                  ← PDFs, images

~/.hermes/.env (NEVER in git)
├── DOGFOODING_API_KEY=***├── TMCP_DINGTALK_TOKEN=***└── DISCORD_TOKEN=***```

### 工作 → 个人切换

chezmoi 的 `isWork` 变量自动切换：
- **工作**：Qwen3.7-Max (dogfooding) + DingTalk
- **个人**：gemma3:12b (Ollama) + Discord

同一套 skills/cron 定义，不同的 provider 和 channel。

## Connections
- Supports: [[inbox/2026-07-06-zettelkasten-workflow]] — 这是工作流设计的落地实现
- Extends: [[permanent/hermes-openclaw-dual-harness]] — 双 Harness 架构的知识管理层

## Open Questions
- Zettelkasten 是否需要 Obsidian 作为 viewer？还是纯 Neovim + grep 够用？
- Hugo blog skill 的最佳触发方式：手动命令 vs 自动从 permanent 笔记中检测成熟笔记？
- 是否需要一个 "weekly review" cron 任务来提醒处理 inbox 中的 seedling 笔记？
