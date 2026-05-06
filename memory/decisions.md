# 技术决策记录

> 每次做出重要技术选择时记录，避免下次重新讨论。

---

## 决策模板

```
## YYYY-MM-DD：[决策主题]
- **选择**：XXX
- **备选方案**：YYY
- **理由**：为什么选 XXX
- **影响**：影响哪些文件/模块
```

---

## 2026-04-13：项目记忆系统架构

- **选择**：CLAUDE.md（主记忆）+ memory/（详细记录）+ hooks（自动化）
- **备选方案**：纯 TodoWrite、纯聊天记录
- **理由**：CLAUDE.md 由 Claude Code 自动加载，零配置，最可靠；hooks 提供自动化辅助
- **影响**：`.claude/settings.json`、`.claude/hooks/`、`memory/`

---

## 待决策

- [ ] 编程语言和依赖管理工具（建议 Python + uv）
- [ ] 数据存储格式（CSV / Parquet / 数据库）
- [ ] 是否使用 Hydra 管理实验配置
