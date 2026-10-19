// 两个引擎检出之间的自对弈 A/B(旧检出 vs 当前工作树),用于引擎改动的棋力回归。
//
// 用法:
//   OLD_ENGINE_ROOT=<旧引擎检出目录> node scripts/selfplay-ab.mjs
//   旧检出目录需包含 web/src/engine/api.ts(通常就是本仓库的历史版本)。
// 可调环境变量:TIERS(默认 legend,hell,advanced,mid,beginner)、GAMES_PER_TIER(20)、
//   NODE_LIMIT(20000)、WORKERS(CPU-2,上限 20)、OPEN_PLIES(2)、PAIRED(1=交换先后手)、
//   OUT(结果 JSON 路径,默认系统临时目录下的 selfplay-ab.json)。
import { Worker } from "node:worker_threads";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";

const TIERS = (process.env.TIERS ?? "legend,hell,advanced,mid,beginner").split(",").filter(Boolean);
const GAMES_PER_TIER = Number(process.env.GAMES_PER_TIER ?? 20);
const NODE_LIMIT = Number(process.env.NODE_LIMIT ?? 20000);
const WORKERS = Number(process.env.WORKERS ?? Math.min(20, Math.max(2, os.cpus().length - 2)));
const OPEN_PLIES = Number(process.env.OPEN_PLIES ?? 2);
const OLD_ROOT = process.env.OLD_ENGINE_ROOT ?? "";
const OUT = process.env.OUT ?? path.join(os.tmpdir(), "selfplay-ab.json");

if (!OLD_ROOT) {
  console.error("缺少 OLD_ENGINE_ROOT:请指向旧引擎检出目录(需含 web/src/engine/api.ts)。");
  process.exit(2);
}
const oldApiPath = path.join(OLD_ROOT, "web", "src", "engine", "api.ts");
if (!fs.existsSync(oldApiPath)) {
  console.error(`OLD_ENGINE_ROOT 下找不到 ${path.join("web", "src", "engine", "api.ts")}:${oldApiPath}`);
  process.exit(2);
}

const specs = [];
const PAIRED = process.env.PAIRED !== "0";
for (const tier of TIERS) {
  if (PAIRED) {
    // 每个开局种子下交换颜色各下一盘，抵消先后手偏差。
    const pairs = Math.ceil(GAMES_PER_TIER / 2);
    for (let p = 0; p < pairs; p++) {
      const seed = (TIERS.indexOf(tier) + 1) * 100000 + p * 7919 + 1;
      for (const oldIsBlack of [true, false]) {
        specs.push({
          tier, game: specs.filter((x) => x.tier === tier).length + 1,
          pair: p + 1, oldIsBlack, nodeLimit: NODE_LIMIT, openPlies: OPEN_PLIES, seed,
        });
      }
    }
  } else {
    for (let i = 0; i < GAMES_PER_TIER; i++) {
      specs.push({
        tier, game: i + 1, oldIsBlack: i % 2 === 0, nodeLimit: NODE_LIMIT,
        openPlies: OPEN_PLIES, seed: (TIERS.indexOf(tier) + 1) * 100000 + i * 104729 + 17,
      });
    }
  }
}

let next = 0, running = 0, done = 0;
const results = [];
const t0 = Date.now();
await new Promise((resolve) => {
  const launch = () => {
    while (running < WORKERS && next < specs.length) {
      const spec = specs[next++];
      const w = new Worker(new URL("./selfplay-ab-worker.mjs", import.meta.url), {
        workerData: { oldRoot: OLD_ROOT, spec },
      });
      running++;
      w.on("message", (r) => {
        results.push(r);
        running--; done++;
        console.log(`[${done}/${specs.length}] ${r.tier} #${r.game} ${r.oldIsBlack ? "旧黑" : "旧白"} -> ${r.result} (${r.turns}手, ${(r.oldMs/1000).toFixed(1)}s/${(r.newMs/1000).toFixed(1)}s)${r.error ? " ERR:" + r.error : ""}`);
        if (done === specs.length) resolve();
        else launch();
      });
      w.on("error", (e) => {
        results.push({ ...spec, result: "error", error: String(e), turns: 0 });
        running--; done++;
        console.log(`[${done}/${specs.length}] ${spec.tier} #${spec.game} ERROR ${e}`);
        if (done === specs.length) resolve();
        else launch();
      });
    }
  };
  launch();
});

const out = { nodeLimit: NODE_LIMIT, gamesPerTier: GAMES_PER_TIER, workers: WORKERS, openPlies: OPEN_PLIES, paired: PAIRED, elapsedMs: Date.now() - t0, results };
fs.writeFileSync(OUT, JSON.stringify(out, null, 2));

for (const tier of TIERS) {
  const rs = results.filter((r) => r.tier === tier);
  const oldW = rs.filter((r) => r.result === "old").length;
  const newW = rs.filter((r) => r.result === "new").length;
  const draw = rs.filter((r) => r.result === "draw").length;
  const err = rs.filter((r) => r.result === "error").length;
  const avgTurns = rs.reduce((a, r) => a + (r.turns ?? 0), 0) / rs.length;
  console.log(`RESULT ${tier.padEnd(9)} old ${oldW} new ${newW} draw ${draw} err ${err} avgTurns ${avgTurns.toFixed(1)}`);
}
const scored = results.filter((r) => r.result !== "error");
const oldW = scored.filter((r) => r.result === "old").length;
const newW = scored.filter((r) => r.result === "new").length;
const draw = scored.filter((r) => r.result === "draw").length;
const p = scored.length ? (newW + draw * 0.5) / scored.length : 0;
const elo = p > 0 && p < 1 ? -400 * Math.log10(1 / p - 1) : p === 1 ? 999 : -999;
console.log(`RESULT total    old ${oldW} new ${newW} draw ${draw} err ${results.length - scored.length} newScore ${(p*100).toFixed(1)}% elo ${elo.toFixed(0)} elapsed ${(out.elapsedMs/1000).toFixed(1)}s`);
