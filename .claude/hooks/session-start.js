#!/usr/bin/env node
/**
 * session-start.js
 * Triggered by UserPromptSubmit (first message of each session).
 * Reads memory files and injects current project state into Claude's context.
 *
 * Claude Code hook protocol:
 *   - stdout → injected as context into Claude's next turn
 *   - stderr → shown to user as a warning (not injected)
 *   - exit 0  → proceed normally
 *   - exit 1  → block the prompt (use sparingly)
 */

const fs = require("fs");
const path = require("path");

// ── Config ────────────────────────────────────────────────────────────────────
const PROJECT_ROOT = path.resolve(__dirname, "../..");
const SESSION_FLAG = path.join(__dirname, ".session_active");
const MEMORY_DIR   = path.join(PROJECT_ROOT, "memory");

// ── Helper ────────────────────────────────────────────────────────────────────
function readFile(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
}

function flag(exists) {
  if (exists) {
    fs.writeFileSync(SESSION_FLAG, new Date().toISOString(), "utf8");
  } else {
    try { fs.unlinkSync(SESSION_FLAG); } catch {}
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
const isNewSession = !fs.existsSync(SESSION_FLAG);

if (isNewSession) {
  flag(true); // mark session as started

  const progress = readFile(path.join(MEMORY_DIR, "progress.md"));
  const paperNotes = readFile(path.join(MEMORY_DIR, "paper-notes.md"));

  const lines = [
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    "🔄  新会话已启动 — 项目记忆已加载",
    "    项目：带耦合多层网络 / Bonaccolto et al. (2019) 复现",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  ];

  if (progress) {
    // Extract the last session block (everything after the last "## 20" heading)
    const blocks = progress.split(/(?=^## \d{4}-\d{2}-\d{2})/m).filter(Boolean);
    const lastBlock = blocks[blocks.length - 1] || "";
    lines.push("\n📋 上次进度摘要：\n");
    lines.push(lastBlock.trim());
  }

  lines.push(`\n\u{1F4A1} 提示：会话结束时请发送 "保存进度" 或 "save progress"，我会自动更新记忆文件。`);
  lines.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

  process.stdout.write(lines.join("\n") + "\n");
}

process.exit(0);
