#!/usr/bin/env node
/**
 * session-end.js
 * Triggered by the Stop event (Claude finishes responding / session ends).
 * Cleans up the session flag so next startup shows fresh memory summary.
 *
 * Claude Code hook protocol:
 *   - stdout → shown to user after Claude's final response
 *   - exit 0  → normal
 */

const fs   = require("fs");
const path = require("path");

const SESSION_FLAG = path.join(__dirname, ".session_active");

// Remove the flag so the next UserPromptSubmit is treated as a new session
try { fs.unlinkSync(SESSION_FLAG); } catch {}

// Remind the user to save if they haven't yet
const reminder = [
  "",
  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  `💾  会话结束提醒`,
  `    如果本次做了重要进展，请在关闭前发送：`,
  `    "保存进度" 或 "save progress"`,
  `    Claude 会自动更新 memory/ 和 CLAUDE.md，`,
  `    下次重启后无缝接续。`,
  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  "",
].join("\n");

process.stdout.write(reminder);
process.exit(0);
