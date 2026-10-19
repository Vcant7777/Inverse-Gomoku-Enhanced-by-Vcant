// selfplay-ab.mjs 的 worker:一个 worker 下一盘旧引擎 vs 新引擎的对局。
// 依赖 workerData.oldRoot(旧引擎检出根目录),由 selfplay-ab.mjs 校验后传入。
import { parentPort, workerData } from "node:worker_threads";
import { pathToFileURL } from "node:url";
import path from "node:path";

const { oldRoot, spec } = workerData;
const oldApi = pathToFileURL(path.join(oldRoot, "web/src/engine/api.ts")).href;
const newApi = pathToFileURL(path.resolve(import.meta.dirname, "..", "src/engine/api.ts")).href;
const { EngineSession: OldEngine } = await import(oldApi);
const { EngineSession: NewEngine } = await import(newApi);

function play() {
  const oldS = new OldEngine();
  const newS = new NewEngine();
  for (const s of [oldS, newS]) s.handle({ cmd: "new" });
  const oldColor = spec.oldIsBlack ? 1 : 2;
  let seed = (spec.seed ?? 1) >>> 0;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  let turn = 0;
  let oldNodes = 0, newNodes = 0, oldMs = 0, newMs = 0;
  const moves = [];
  // 随机开局，避免同色对局完全重复。
  try {
    for (let k = 0; k < (spec.openPlies ?? 0); k++) {
      const st = oldS.state();
      if (st.game_over || !st.legal.length) break;
      const idx = st.legal[Math.floor(rnd() * st.legal.length)];
      oldS.handle({ cmd: "move", idx });
      newS.handle({ cmd: "move", idx });
      moves.push(idx);
      turn++;
    }
  } catch (err) {
    return { ...spec, result: "error", error: String(err), turns: turn, moves, oldNodes: 0, newNodes: 0, oldMs: 0, newMs: 0 };
  }
  let result = "unfinished";
  try {
    while (turn < 225) {
      const oldTurn = (turn % 2 === 0) === spec.oldIsBlack;
      const actor = oldTurn ? oldS : newS;
      const other = oldTurn ? newS : oldS;
      const t0 = performance.now();
      const r = actor.ai({
        depth: 16, budget: 0, engine: spec.tier, vcf: true, nodeLimit: spec.nodeLimit,
      });
      const ms = performance.now() - t0;
      if (oldTurn) { oldNodes += r.nodes; oldMs += ms; } else { newNodes += r.nodes; newMs += ms; }
      if (!(r.move >= 0)) throw new Error(`bad move ${r.move}`);
      other.handle({ cmd: "move", idx: r.move });
      moves.push(r.move);
      turn++;
      const st = actor.state();
      if (st.game_over) {
        if (st.is_draw) result = "draw";
        else if (st.winner === oldColor) result = "old";
        else result = "new";
        break;
      }
    }
    if (result === "unfinished") result = "draw";
  } catch (err) {
    result = "error";
    return { ...spec, result, error: String(err), turns: turn, moves, oldNodes, newNodes, oldMs, newMs };
  }
  return { ...spec, result, turns: turn, moves, oldNodes, newNodes, oldMs, newMs };
}

parentPort.postMessage(play());
