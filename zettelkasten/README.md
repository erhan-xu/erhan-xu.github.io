# Zettelkasten 🗃️

A git-managed, bilingual (EN/ZH) Zettelkasten for research notes, fleeting thoughts, and permanent knowledge.

## Structure

```
zettelkasten/
├── inbox/              # 闪念笔记 — capture anything, process within 48h
│   └── YYYY-MM-DD-slug.md
├── literature/         # 文献笔记 — reading notes from papers/books/blogs
│   └── author-year-keyword.md
├── permanent/          # 永久笔记 — distilled insights, your own thinking
│   └── slug.md
├── projects/           # 项目笔记 — active research threads, will be archived
│   └── project-name/
├── references/         # 参考资料 — PDFs, images, data files
│   └── ...
├── daily/              # 日记/日志 — daily reflections (optional)
│   └── YYYY-MM-DD.md
├── templates/          # 笔记模板
├── INDEX.md            # 主索引 — entry points and clusters
└── tags.md             # 标签索引
```

## Conventions

### File Naming
- **inbox**: `YYYY-MM-DD-slug.md` (e.g., `2026-07-06-rlhf-scaling.md`)
- **literature**: `author-year-keyword.md` (e.g., `ouyang-2024-rlhf-survey.md`)
- **permanent**: `slug.md` (e.g., `reward-model-as-compass.md`)
- Use lowercase, hyphens, no spaces

### Frontmatter
Every note has YAML frontmatter:
```yaml
---
type: fleeting | literature | permanent | project
created: 2026-07-06
tags: [rl, alignment, rlhf]
status: seedling | evergreen | reviewed
confidence: high | medium | low | frontier
links: [related-note-1, related-note-2]
---
```

### Linking
Use wiki-style links: `[[note-slug]]` or `[[note-slug|display text]]`

### Status Lifecycle
- **seedling** 🌱 — just captured, needs processing
- **evergreen** 🌿 — reviewed, connected, valuable
- **reviewed** 🌳 — mature, may become blog post

### Confidence Levels
- **high** — I understand this well
- **medium** — rough understanding, details may be off
- **low** — speculative, skeleton only
- **frontier** — beyond my training data, AI provides scaffolding questions

## Workflow

1. **Capture** → `inbox/` (anything that catches your attention)
2. **Process** → Turn into `literature/` (from reading) or `permanent/` (from thinking)
3. **Connect** → Add links, update INDEX.md
4. **Distill** → Mature permanent notes become blog posts (via Hermes blog skill)

## Tools

- **Primary**: Any text editor (Neovim, VS Code, Obsidian)
- **AI Assistant**: Hermes Agent (reads/writes notes, helps with connections)
- **Sync**: git push/pull
- **Blog export**: Hugo blog at erhan-xu.github.io (via Hermes blog-publish skill)
