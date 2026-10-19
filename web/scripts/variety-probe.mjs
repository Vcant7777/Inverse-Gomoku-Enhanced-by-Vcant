/**
 * 变化(去固定化)档位探针。
 *
 *   node scripts/variety-probe.mjs              # 多样性:同一手顺的应手分布
 *   node scripts/variety-probe.mjs --ab         # 棋力 A/B:同节点数,配对交换先后手
 *
 * 环境变量:
 *   TRIALS=12   多样性每档重复次数        DEPTH=4   搜索深度
 *   PROBE_NODES=60000  多样性每步节点上限
 *   GAMES=30    A/B 对数(每对两局)        NODES=3000  A/B 每步节点上限
 *   OPEN_PLIES=2  A/B 随机开局手数        AB_LEVELS=mid,high  A/B 的变化档位(基准恒为 off)
 */
import { EngineSession } from "../src/engine/api.ts";
import { mulberry32 } from "../src/engine/variety.ts";

const AB = process.argv.includes("--ab");
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

if (AB) printAb();
else printDiversity();
