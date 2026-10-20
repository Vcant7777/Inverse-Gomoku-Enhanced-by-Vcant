/**
 * 变化(去固定化)档位探针。
 *
 *   node scripts/variety-probe.mjs              # 多样性:同一手顺的应手分布
 *   node scripts/variety-probe.mjs --deviation  # 价值偏差:选中手离最优手差多少
 *   node scripts/variety-probe.mjs --ab         # 棋力 A/B:同节点数,配对交换先后手
 *
 * 环境变量:
 *   TRIALS=12   多样性每档重复次数        DEPTH=4   搜索深度
 *   PROBE_NODES=60000  多样性每步节点上限
 *   GAMES=30    A/B 对数(每对两局)        NODES=3000  A/B 每步节点上限
 *   OPEN_PLIES=2  A/B 随机开局手数        AB_LEVELS=mid,high  A/B 的变化档位(基准恒为 off)
 */
import { EngineSession } from "../src/engine/api.ts";
import { applyStep, createPosition, DEFAULT_CONFIG } from "../src/engine/rules.ts";
import { searchBest, clearTT } from "../src/engine/search.ts";
import { mulberry32, varietyAt, varietyFor } from "../src/engine/variety.ts";

const AB = process.argv.includes("--ab");
const DEVIATION = process.argv.includes("--deviation");
const TRIALS = Number(process.env.TRIALS ?? 12);
const DEPTH = Number(process.env.DEPTH ?? 4);
const PROBE_NODES = Number(process.env.PROBE_NODES ?? 60000);
const GAMES = Number(process.env.GAMES ?? 30);
const NODES = Number(process.env.NODES ?? 3000);
const OPEN_PLIES = Number(process.env.OPEN_PLIES ?? 2);
/** 开局种子起点:多进程并行时用不同 SEED0 分区,再手工汇总胜/负/和。 */
const SEED0 = Number(process.env.SEED0 ?? 90001);
const AB_LEVELS = (process.env.AB_LEVELS ?? "mid,high").split(",").filter(Boolean);

/** 中腹附近固定手顺:黑天元、白邻位、黑再展、白再应。 */
const PREFIX = [112, 52, 172, 34];
const LEVELS = ["off", "low", "mid", "high"];

function aiCall(session, variety, nodeLimit) {
  return session.handle({
    cmd: "ai", depth: DEPTH, engine: "advanced", unlimited: true, vcf: false,
    ...(nodeLimit ? { nodeLimit } : {}), variety,
  });
}

function probeDiversity(prefix, level) {
  const counts = new Map();
  for (let k = 0; k < TRIALS; k++) {
    const s = new EngineSession();
    s.handle({ cmd: "new" });
    for (const m of prefix) s.handle({ cmd: "move", idx: m });
    const mv = aiCall(s, level, PROBE_NODES).move;
    counts.set(mv, (counts.get(mv) ?? 0) + 1);
  }
  return counts;
}

function printDiversity() {
  for (const level of LEVELS) {
    for (const [name, prefix] of [["中腹4手", PREFIX], ["白方首应手", [112]]]) {
      const counts = probeDiversity(prefix, level);
      const dist = [...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([mv, n]) => `${mv}×${n}`)
        .join(" ");
      console.log(`变化=${level.padEnd(4)} ${name.padEnd(6)} 不同应手 ${counts.size}/${TRIALS}  ${dist}`);
    }
  }
}

function randomPrefix(plies, seed) {
  const s = new EngineSession();
  s.handle({ cmd: "new", seed });
  const rng = mulberry32(seed);
  const out = [];
  for (let i = 0; i < plies; i++) {
    const legal = s.handle({ cmd: "state" }).state.legal;
    if (!legal.length) break;
    const pick = legal[Math.floor(rng() * legal.length)];
    s.handle({ cmd: "move", idx: pick });
    out.push(pick);
  }
  return out;
}

/** 返回 { winner, turns }(winner: 1 黑胜 / 2 白胜 / 0 和/未终局)。 */
function playGame(prefix, blackVariety, whiteVariety) {
  const s = new EngineSession();
  s.handle({ cmd: "new" });
  for (const m of prefix) s.handle({ cmd: "move", idx: m });
  for (let ply = 0; ply < 400; ply++) {
    const st = s.handle({ cmd: "state" }).state;
    if (st.game_over) return { winner: st.winner, turns: st.move_count };
    const variety = st.player === 1 ? blackVariety : whiteVariety;
    const r = s.handle({
      cmd: "ai", depth: DEPTH, engine: "advanced", unlimited: true, vcf: true,
      nodeLimit: NODES, variety,
    });
    if (r.move === null) {
      const fin = s.handle({ cmd: "state" }).state;
      return { winner: fin.winner, turns: fin.move_count };
    }
  }
  return { winner: 0, turns: 400 };
}

function scoreForBlack(winner) {
  return winner === 1 ? 1 : winner === 2 ? 0 : 0.5;
}

function printAb() {
  for (const level of AB_LEVELS) {
    let score = 0, wins = 0, losses = 0, draws = 0, turns = 0, games = 0;
    for (let p = 0; p < GAMES; p++) {
      const seed = SEED0 + p * 7919;
      const prefix = randomPrefix(OPEN_PLIES, seed);
      // 同一开局下交换颜色各下一盘,抵消先后手偏差。
      for (const varietyIsBlack of [true, false]) {
        const r = playGame(prefix, varietyIsBlack ? level : "off", varietyIsBlack ? "off" : level);
        const s = varietyIsBlack ? scoreForBlack(r.winner) : 1 - scoreForBlack(r.winner);
        score += s;
        if (s === 1) wins++;
        else if (s === 0) losses++;
        else draws++;
        turns += r.turns;
        games++;
      }
    }
    const pct = (score / games) * 100;
    const elo = score > 0 && score < games ? -400 * Math.log10(1 / (score / games) - 1) : 0;
    console.log(`A/B 变化=${level} vs off: ${wins}胜 ${losses}负 ${draws}和 (${games}局, 每步≤${NODES}节点) `
      + `得分 ${pct.toFixed(1)}% elo ${elo.toFixed(0)} 平均 ${(turns / games).toFixed(1)} 手`);
  }
}


/* ---------------------------------------------------------------- 价值偏差 */

/** 用规则层重放前缀,拿到可交给 searchBest 的局面。 */
function positionAfter(prefix) {
  // applyStep 就地改局面并返回 undo 记录,不能把返回值当局面。
  const pos = createPosition();
  for (const m of prefix) applyStep(pos, m, DEFAULT_CONFIG);
  return pos;
}

function searchOnce(pos, level, seed) {
  // 每次测量都清空置换表:rng 只影响采样,同局面同参数的结果必须完全一致,
  // 否则跨次搜索的 TT 复用会把「深度/表差异」混进「采样偏差」的统计里。
  clearTT();
  const variety = varietyFor(level);
  const r = searchBest(pos, DEFAULT_CONFIG, {
    maxDepth: DEPTH,
    timeMs: 0,
    engine: "advanced",
    maxNodes: PROBE_NODES,
    variety,
    rng: variety.level === "off" ? undefined : mulberry32(seed),
  });
  return { ...r, level: variety.level };
}

/**
 * 对每个局面/档位:先用 off 拿到根着法精确分值做参照,再用 N 个种子实际抽样。
 * 判定用「每次搜索自己的根分值表」,与本轮深度无关(节点上限可能让不同种子完成
 * 不同层,拿固定基线表去比会把正常的深度差异误判成越界):
 *   ① 选中的手必须落在本次搜索自己的候选池里(窗口+K 之后);
 *   ② 选中手与本次最优手的偏差必须 ≤ 该档窗口。
 * 另统计相对 off 基线的「选中最优率 / 偏差」,给人看的棋力口径。
 */
function printDeviation() {
  const positions = [
    ["白首应手(黑112)", [112]],
    ["中腹4手", PREFIX],
    ["中盘8手", [112, 52, 172, 34, 127, 97, 82, 66]],
  ];
  for (let i = 0; i < 4; i++) {
    const prefix = randomPrefix(7 + i * 3, 4001 + i * 131);
    if (prefix.length >= 3) positions.push([`随机${prefix.length}手#${i + 1}`, prefix]);
  }

  let outside = 0, samples = 0, singletonBad = 0, singletonSamples = 0, depthDrift = 0;
  console.log(`== 价值偏差(深度 ${DEPTH},每步 ≤${PROBE_NODES} 节点,${TRIALS} 个种子) ==`);
  console.log("局面                 档位   池  选中最优    偏差均值/最大   窗口  最优-次优  深度");
  for (const [name, prefix] of positions) {
    const pos = positionAfter(prefix);
    if (pos.gameOver) continue;
    const base = searchOnce(pos, "off", 0);
    const table = base.rootScored;
    if (table.length < 2) continue;
    const sorted = [...table].sort((a, b) => b.score - a.score);
    const bestScore = sorted[0].score;
    for (const level of ["low", "mid", "high"]) {
      const [, window, cap] = varietyAt(varietyFor(level), pos.moveCount);
      const pool = sorted.slice(0, cap).filter((s) => bestScore - s.score <= window);
      let hitBest = 0, gapSum = 0, gapMax = 0, dMin = 99, dMax = 0, sameDepth = 0;
      for (let k = 0; k < TRIALS; k++) {
        const r = searchOnce(pos, level, 7001 + k * 9173);
        samples++;
        dMin = Math.min(dMin, r.depth);
        dMax = Math.max(dMax, r.depth);
        // ① ② 只跟本次搜索自己的表比:这是抽样的真实约束。
        const own = [...r.rootScored].sort((a, b) => b.score - a.score);
        const ownBest = own.length > 0 ? own[0].score : r.score;
        const ownPool = own.slice(0, cap).filter((s) => ownBest - s.score <= window);
        const ownPick = own.find((s) => s.move === r.move);
        if (!ownPool.some((s) => s.move === r.move) || !ownPick || ownBest - ownPick.score > window + 1e-9) {
          outside++;
        }
        if (ownPool.length === 1) {
          singletonSamples++;
          if (r.move !== ownPool[0].move) singletonBad++;
        }
        // 相对 off 基线的口径(深度可能不同,仅作参考)。
        const got = table.find((s) => s.move === r.move);
        if (r.depth === base.depth) {
          const gap = got ? bestScore - got.score : 0;
          gapSum += gap;
          sameDepth++;
          if (gap > gapMax) gapMax = gap;
        } else {
          depthDrift++;
        }
        if (r.move === sorted[0].move) hitBest++;
      }
      console.log(
        `${name.padEnd(20)} ${level.padEnd(4)} ${String(pool.length).padStart(2)}`
        + `  ${String(hitBest).padStart(3)}/${TRIALS}`
        + `   ${sameDepth > 0 ? `${(gapSum / sameDepth).toFixed(3)} / ${gapMax.toFixed(3)}` : "   n/a    "}`
        + `   ${window.toFixed(2)}`
        + `   ${(bestScore - sorted[1].score).toFixed(3)}`
        + `  ${base.depth}${dMin !== dMax ? `~${dMin}-${dMax}` : ""}`,
      );
    }
  }
  console.log(`自身池外采样 ${outside}/${samples} 次(必须为 0);`
    + `自身池=1 时选中最优 ${singletonSamples - singletonBad}/${singletonSamples} 次(必须全中)`);
  console.log(`与 off 基线完成层数不同的样本 ${depthDrift}/${samples} 次`
    + `(这些样本的「相对基线偏差」只作参考,越界判定用的是各自搜索自己的表)`);
  if (outside > 0 || singletonBad > 0) process.exitCode = 1;
}

if (AB) printAb();
else if (DEVIATION) printDeviation();
else printDiversity();
