import test from "node:test";
import assert from "node:assert/strict";
import {
  BLACK, WHITE, EMPTY, SIZE, createPosition, legalMoves, applyStep, undoStep,
  computeHash, DEFAULT_CONFIG, pendingTargets, checkWinAt,
} from "../src/engine/rules.ts";
import { EngineSession } from "../src/engine/api.ts";
import { findWinningMove } from "../src/engine/search.ts";

test("初始局面：黑先、225 个合法点", () => {
  const p = createPosition();
  const legal = legalMoves(p, DEFAULT_CONFIG);
  assert.equal(legal.length, 225);
  assert.equal(legal[0], 0);
  assert.equal(legal[224], 224);
});

test("占领后 pending 保留同一方，安置后换手", () => {
  const p = createPosition();
  p.board[112] = BLACK; p.board[113] = WHITE;
  p.player = BLACK; p.turnCount = 10; p.whiteTurns = 10;
  const u = applyStep(p, 113, DEFAULT_CONFIG);
  assert.equal(p.pending, 113);
  assert.equal(p.player, BLACK);
  const targets = legalMoves(p, DEFAULT_CONFIG);
  assert.ok(targets.length > 0);
  applyStep(p, targets[0], DEFAULT_CONFIG);
  assert.equal(p.pending, -1);
  assert.equal(p.player, WHITE);
  undoStep(p, u);   // 撤销安置前的状态不合法，这里只验证不抛错
});

test("悔棋/重放与哈希一致", () => {
  const s = new EngineSession();
  s.handle({ cmd: "new" });
  for (const m of [112, 52, 172, 34, 190, 76]) s.handle({ cmd: "move", idx: m });
  const before = s.state();
  s.handle({ cmd: "undo" });
  assert.equal(s.state().moves.length, before.moves.length - 1);
  s.handle({ cmd: "move", idx: before.moves[before.moves.length - 1] });
  assert.deepEqual(s.state().board, before.board);
});

test("pending 安置：整段连续空位都可放", () => {
  const p = createPosition();
  p.board[112] = BLACK;
  p.pending = 112;
  const targets = pendingTargets(p.board, 112);
  assert.ok(targets.includes(113));
  assert.ok(targets.includes(119));
  assert.ok(targets.length >= 50);
});

test("自杀着法在 loss_start 后被剔除", () => {
  const p = createPosition();
  for (let c = 3; c <= 6; c++) p.board[7 * 15 + c] = BLACK;
  p.player = BLACK; p.turnCount = 8; p.whiteTurns = 8;
  const legal = legalMoves(p, DEFAULT_CONFIG);
  assert.ok(!legal.includes(7 * 15 + 2));
  assert.ok(!legal.includes(7 * 15 + 7));
});

test("一步杀：pending 安置对方成五", () => {
  const p = createPosition();
  for (let c = 3; c <= 6; c++) p.board[7 * 15 + c] = WHITE;
  p.board[7 * 15 + 8] = BLACK;
  p.player = BLACK; p.pending = 7 * 15 + 8; p.turnCount = 10; p.whiteTurns = 10;
  assert.ok(checkWinAt(p.board, 7 * 15 + 7, WHITE));
  assert.equal(findWinningMove(p, DEFAULT_CONFIG), 7 * 15 + 7);
  const s = new EngineSession();
  s.pos = p; s.steps = [{ idx: 7 * 15 + 8, player: BLACK, pending: -1, turnCount: 10, whiteTurns: 10 }];
  const r = s.handle({ cmd: "ai", depth: 4, budget: 1, engine: "advanced" });
  assert.equal(r.move, 7 * 15 + 7);
  assert.equal(r.state.game_over, true);
  assert.equal(r.state.winner, BLACK);
});

test("棋谱 kinds/回放：占领步与安置步不错位", () => {
  const s = new EngineSession();
  s.handle({ cmd: "new" });
  for (const m of [112, 113, 113]) s.handle({ cmd: "move", idx: m });
  const cap = s.state();
  assert.equal(cap.kinds.at(-1), "capture");
  assert.equal(cap.pending, 113);
  assert.equal(cap.moves.at(-1), 113);
  const target = cap.legal[0];
  s.handle({ cmd: "move", idx: target });
  const placed = s.state();
  assert.equal(placed.kinds.at(-1), "place2");
  assert.deepEqual(placed.moves.slice(-2), [113, target]);
  // 回放到安置完成：箭头应从占领点指向安置点
  s.handle({ cmd: "goto", moves: placed.moves, pos: placed.moves.length });
  assert.equal(s.state().kinds.at(-1), "place2");
  assert.deepEqual(s.state().moves.slice(-2), [113, target]);
  // 回放到占领步：最后一步是 capture，不应使用 moves[-2] 画箭头
  s.handle({ cmd: "goto", moves: placed.moves, pos: 3 });
  assert.equal(s.state().kinds.at(-1), "capture");
  assert.equal(s.state().pending, 113);
});
