import test from "node:test";
import assert from "node:assert/strict";
import { createPosition, computeHash, DEFAULT_CONFIG } from "../src/engine/rules.ts";
import { EngineSession } from "../src/engine/api.ts";
import { searchBest, sampleRoot } from "../src/engine/search.ts";
import { mulberry32, varietyFor, varietyAt, normalizeVariety, randomSeed, sanitizeParams } from "../src/engine/variety.ts";

// 固定前缀:黑 112(天元)、白 52、黑 172、白 34 —— 空盘附近的中腹手顺。
const PREFIX = [112, 52, 172, 34];

function playAi(prefix, variety, seed) {
  const s = new EngineSession();
  s.handle({ cmd: "new", seed: seed ?? undefined });
  for (const m of prefix) s.handle({ cmd: "move", idx: m });
  const legal = s.handle({ cmd: "state" }).state.legal;
  const r = s.handle({
    cmd: "ai", depth: 4, engine: "advanced", unlimited: true, vcf: false, variety,
  });
  return { ...r, legalBefore: legal };
}

/** 占领+安置的一手杀态势(与 ai.test.mjs 同构)。 */
function killBoard() {
  const p = createPosition();
  for (const r of [6, 7, 8, 9]) p.board[r * 15 + 4] = 2;
  p.board[4 * 15 + 4] = 2;
  p.player = 1; p.turnCount = 10; p.whiteTurns = 10;
  p.hash = computeHash(p.board);
  return p;
}

test("变化档位：别名归一化与开局窗口", () => {
  assert.equal(normalizeVariety(undefined), "off");
  assert.equal(normalizeVariety(""), "off");
  assert.equal(normalizeVariety("高"), "high");
  assert.equal(normalizeVariety("unknown"), "off");
  assert.equal(varietyFor("mid").bookChance < 1, true);
  // 开局阶段用更大的温度/窗口/候选上限,中盘回到档位基础值。
  const spec = varietyFor("mid");
  assert.deepEqual(varietyAt(spec, 0), [spec.openingTemperature, spec.openingWindow, spec.openingPoolCap]);
  assert.deepEqual(varietyAt(spec, spec.openingPlies + 2), [spec.temperature, spec.window, spec.poolCap]);
  assert.deepEqual(varietyAt(varietyFor("off"), 0), [0, 0, 1]);
});

test("随机源：同种子可复现、异种子不同", () => {
  const a = mulberry32(12345);
  const b = mulberry32(12345);
  const c = mulberry32(12346);
  const seqA = [a(), a(), a()];
  assert.deepEqual(seqA, [b(), b(), b()]);
  assert.notDeepEqual(seqA, [c(), c(), c()]);
  assert.ok(randomSeed() > 0);
});

test("采样器：差距远超温度时退化为 argmax，近同分才可能换手", () => {
  const rng = mulberry32(7);
  const far = [{ move: 1, score: 0.5 }, { move: 2, score: 0.1 }];
  let hits = 0;
  for (let i = 0; i < 200; i++) if (sampleRoot(far, 0.05, 0.2, rng) === 2) hits++;
  assert.equal(hits, 0);
  const near = [{ move: 1, score: 0.5 }, { move: 2, score: 0.45 }];
  const seen = new Set();
  for (let i = 0; i < 200; i++) seen.add(sampleRoot(near, 0.2, 0.3, rng));
  assert.equal(seen.size, 2);
  // cap 收口:第 3 个着法分值虽在窗口内,但被 K 挡在池外(稀疏局面防走角点)。
  const wide = [{ move: 1, score: 0.5 }, { move: 2, score: 0.45 }, { move: 0, score: 0.44 }];
  const capped = new Set();
  for (let i = 0; i < 200; i++) capped.add(sampleRoot(wide, 0.2, 0.9, rng, 2));
  assert.deepEqual([...capped].sort(), [1, 2]);
});

test("off 档位：同一手顺跨会话完全一致（旧行为不变）", () => {
  const moves = [];
  for (let k = 0; k < 5; k++) moves.push(playAi(PREFIX, "off").move);
  assert.equal(new Set(moves).size, 1, `off 档位应完全确定,实际 ${moves.join(",")}`);
  // 缺省(不传 variety)也必须保持确定性。
  const dflt = [];
  for (let k = 0; k < 3; k++) dflt.push(playAi(PREFIX, undefined).move);
  assert.equal(new Set(dflt).size, 1);
});

test("mid 档位：同一手顺多次应手不再固定", () => {
  const moves = [];
  let legal = null;
  for (let k = 0; k < 10; k++) {
    const r = playAi(PREFIX, "mid");
    moves.push(r.move);
    legal = r.legalBefore;
  }
  assert.ok(new Set(moves).size >= 2, `mid 档位应出现不同应手,实际 ${moves.join(",")}`);
  for (const m of moves) assert.ok(legal.includes(m), `应手 ${m} 必须合法`);
});

test("变化档位：白方首应手(开局库)也会变化", () => {
  const first = [];
  for (let k = 0; k < 12; k++) first.push(playAi([112], "high").move);
  assert.ok(new Set(first).size >= 2, `白方首应手应出现不同,实际 ${first.join(",")}`);
});

test("变化档位：强制杀仍然确定（不因随机丢掉必胜）", () => {
  const rng = mulberry32(99);
  for (let k = 0; k < 5; k++) {
    const r = searchBest(killBoard(), DEFAULT_CONFIG, {
      maxDepth: 3, timeMs: 0, engine: "advanced",
      variety: varietyFor("high"), rng,
    });
    assert.equal(r.move, 4 * 15 + 4);
  }
});

test("变化档位：hint/分析不被打散", () => {
  const runs = [];
  for (let k = 0; k < 3; k++) {
    const s = new EngineSession();
    s.handle({ cmd: "new" });
    for (const m of PREFIX) s.handle({ cmd: "move", idx: m });
    // 即使同一会话跑过变化档位 AI，hint 仍应给出同一份分析。
    s.handle({ cmd: "hint", depth: 2, budget: 1, engine: "mid", vcf: false });
    runs.push(JSON.stringify(s.handle({ cmd: "hint", depth: 2, budget: 1, engine: "mid", vcf: false })));
  }
  assert.equal(new Set(runs).size, 1);
});

test("变化档位：一局内每一步都合法且能正常终局", () => {
  const s = new EngineSession();
  s.handle({ cmd: "new", config: { white_restrict_turns: 0 }, seed: 2026 });
  for (let ply = 0; ply < 60; ply++) {
    const st = s.handle({ cmd: "state" }).state;
    if (st.game_over) break;
    const r = s.handle({
      cmd: "ai", depth: 2, engine: "advanced", unlimited: true, vcf: false, variety: "high",
    });
    assert.ok(r.move === null || st.legal.includes(r.move));
    if (r.move === null) break;
  }
});

test("自定义参数：夹紧越界、丢弃非有限值、整数取整", () => {
  assert.deepEqual(sanitizeParams({ window: 99, poolCap: 2.7, temperature: -1, openingPlies: "7.4" }),
    { window: 3, poolCap: 3, temperature: 0, openingPlies: 7 });
  assert.deepEqual(sanitizeParams({ window: NaN, poolCap: Infinity, bogus: 5 }), {});
  assert.deepEqual(sanitizeParams(null), {});
  assert.deepEqual(sanitizeParams("abc"), {});
});

test("自定义档位：只覆盖给到的字段，预设档位忽略参数", () => {
  const custom = varietyFor("custom", { temperature: 0.5, poolCap: 5 });
  assert.equal(custom.level, "custom");
  assert.equal(custom.temperature, 0.5);
  assert.equal(custom.poolCap, 5);
  // 未给到的字段继承 custom 模板(= mid 数值)
  assert.equal(custom.window, 0.3);
  assert.equal(custom.bookChance, 0.35);
  // 预设档位不受 params 影响(避免 {level:"high", params:{temperature:0}} 这类组合被悄悄改掉)
  const high = varietyFor("high", { temperature: 0, poolCap: 1 });
  assert.equal(high.temperature, 0.2);
  assert.equal(high.poolCap, 4);
  assert.equal(normalizeVariety("自定义"), "custom");
  assert.equal(normalizeVariety("4"), "custom");
});

test("自定义档位：T=0/K=1 退化为确定性，T 拉大则出现多解", () => {
  const frozen = [];
  for (let k = 0; k < 3; k++) {
    const s = new EngineSession();
    s.handle({ cmd: "new", seed: 100 + k });
    for (const m of PREFIX) s.handle({ cmd: "move", idx: m });
    frozen.push(s.handle({
      cmd: "ai", depth: 4, engine: "advanced", unlimited: true, vcf: false,
      variety: "custom", varietyParams: { temperature: 0, window: 0, poolCap: 1 },
    }).move);
  }
  assert.equal(new Set(frozen).size, 1);

  const moves = new Set();
  for (let k = 0; k < 8; k++) {
    const s = new EngineSession();
    s.handle({ cmd: "new", seed: 500 + k });
    for (const m of PREFIX) s.handle({ cmd: "move", idx: m });
    moves.add(s.handle({
      cmd: "ai", depth: 4, engine: "advanced", unlimited: true, vcf: false,
      variety: "custom",
      varietyParams: { temperature: 0.35, window: 1.2, poolCap: 5, bookChance: 0 },
    }).move);
  }
  assert.ok(moves.size > 1, `期望自定义高档位出现多解，实际只有 ${[...moves]}`);
});
