import test from "node:test";
import assert from "node:assert/strict";
import { createPosition, computeHash, legalMoves, DEFAULT_CONFIG } from "../src/engine/rules.ts";
import { EngineSession } from "../src/engine/api.ts";
import { findForcedKill } from "../src/engine/vcf.ts";
import { findWinningMove } from "../src/engine/search.ts";
import { candidates } from "../src/engine/candidates.ts";
import { profileFor, normalizeVersion, depthForProfile } from "../src/engine/profiles.ts";

function killBoard() {
  const p = createPosition();
  for (const r of [6, 7, 8, 9]) p.board[r * 15 + 4] = 2;
  p.board[4 * 15 + 4] = 2;
  p.player = 1; p.turnCount = 10; p.whiteTurns = 10;
  p.hash = computeHash(p.board);
  return p;
}

test("VCF：找到占领+安置的强制杀", () => {
  const p = killBoard();
  assert.equal(findForcedKill(p, DEFAULT_CONFIG, { maxDepth: 3, deadline: Date.now() + 1000, nodeCap: 20000 }), 4 * 15 + 4);
  assert.equal(findWinningMove(p, DEFAULT_CONFIG), 4 * 15 + 4);
});

test("档位：别名、VCF/开局库/蒸馏能力", () => {
  assert.equal(normalizeVersion("new"), "advanced");
  assert.equal(normalizeVersion("大师"), "hell");
  assert.equal(normalizeVersion("unknown"), "advanced");
  assert.equal(profileFor("beginner").supportsVcf, false);
  assert.equal(profileFor("beginner").supportsBook, false);
  assert.equal(profileFor("mid").supportsVcf, true);
  assert.equal(profileFor("advanced").distill, 1);
  assert.equal(profileFor("legend").distill, 2);
  assert.equal(depthForProfile("advanced", 20), 8);
  assert.equal(depthForProfile("legend", 8), 10);
});

test("提示：返回合法候选且按分值降序，pos_v=榜首", () => {
  const s = new EngineSession();
  s.handle({ cmd: "new" });
  for (const m of [112, 52, 172, 34, 190, 76]) s.handle({ cmd: "move", idx: m });
  const r = s.handle({ cmd: "hint", depth: 2, budget: 1, engine: "mid", vcf: false });
  assert.ok(r.moves.length > 0 && r.moves.length <= 5);
  assert.equal(r.moves[0].v, r.pos_v);
  for (let i = 1; i < r.moves.length; i++) assert.ok(r.moves[i - 1].v >= r.moves[i].v);
  const legal = legalMoves(s.pos, s.config);
  for (const m of r.moves) assert.ok(legal.includes(m.idx));
});

test("开局库只对中级及以上生效", () => {
  const adv = new EngineSession();
  assert.equal(adv.handle({ cmd: "ai", depth: 2, engine: "advanced", unlimited: true, vcf: false }).move, 14);
  const beg = new EngineSession();
  const m = beg.handle({ cmd: "ai", depth: 2, engine: "beginner", unlimited: true, vcf: false }).move;
  assert.notEqual(m, 14);
});

test("棋谱导出/导入保持状态", () => {
  const s = new EngineSession();
  s.handle({ cmd: "new" });
  for (const m of [112, 52, 172, 34, 190, 76]) s.handle({ cmd: "move", idx: m });
  const before = s.handle({ cmd: "state" }).state;
  const text = s.handle({ cmd: "export" }).text;
  assert.match(text, /\[Moves 6\]/);
  const t = new EngineSession();
  const imported = t.handle({ cmd: "import", text }).state;
  assert.deepEqual(imported.board, before.board);
  assert.equal(imported.player, before.player);
  assert.equal(imported.move_count, before.move_count);
  assert.deepEqual(imported.moves, before.moves);
});

test("候选生成：全部合法、按分值降序、根含 diversity", () => {
  const s = new EngineSession();
  s.handle({ cmd: "new" });
  for (const m of [112, 52, 172, 34, 190, 76]) s.handle({ cmd: "move", idx: m });
  const list = candidates(s.pos, s.config, 8, true, 8);
  assert.ok(list.length > 0);
  const legal = legalMoves(s.pos, s.config);
  for (const c of list) assert.ok(legal.includes(c.move));
  for (let i = 1; i < list.length; i++) assert.ok(list[i - 1].score >= list[i].score);
});
