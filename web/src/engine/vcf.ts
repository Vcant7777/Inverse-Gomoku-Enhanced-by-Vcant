/** 强制杀链搜索（VCF 类），移植自 Python antifive/threat_search.py。 */
import { EMPTY, N, applyStep, legalMoves, other, undoStep } from "./rules.ts";
import type { Config, Position } from "./rules.ts";
import { jumpThrough, killCaptures, killPairs, killPlacementsConfig } from "./tactics.ts";

export interface VcfOptions {
  maxDepth?: number;
  deadline?: number;
  nodeCap?: number;
  onProgress?: (nodes: number) => void;
}

interface Ctx {
  deadline: number;
  nodeCap: number;
  nodes: number;
  aborted: boolean;
  memoA: Map<string, boolean>;
  memoD: Map<string, boolean>;
  killCache: Map<string, ReturnType<typeof killPairs>>;
  onProgress?: (nodes: number) => void;
}

const MEMO_CAP = 50_000;
const KILL_CACHE_CAP = 4096;

function boardKey(pos: Position): string {
  let s = "";
  for (let i = 0; i < N * N; i++) s += String.fromCharCode(pos.board[i]);
  return s;
}

function tick(ctx: Ctx): boolean {
  ctx.nodes++;
  if (ctx.onProgress && (ctx.nodes & 255) === 0) ctx.onProgress(ctx.nodes);
  if (ctx.nodeCap && ctx.nodes >= ctx.nodeCap) ctx.aborted = true;
  else if ((ctx.nodes & 31) === 0 && Date.now() >= ctx.deadline) ctx.aborted = true;
  return ctx.aborted;
}

function memoStore(map: Map<string, boolean>, key: string, value: boolean): void {
  if (map.size >= MEMO_CAP) map.clear();
  map.set(key, value);
}

export function hasKill(pos: Position, config: Config): boolean {
  if (pos.turnCount < config.loss_start_turns) return false;
  if (pos.pending >= 0) return killPlacementsConfig(pos, config.loss_start_turns).length > 0;
  return killCaptures(pos, config).length > 0;
}

function cachedPairs(ctx: Ctx, pos: Position, attacker: number, config: Config) {
  const key = boardKey(pos) + "|" + attacker + "|" + pos.turnCount + "|" + pos.whiteTurns;
  const hit = ctx.killCache.get(key);
  if (hit) return hit;
  const saved = pos.player;
  pos.player = attacker;
  const pairs = killPairs(pos, config);
  pos.player = saved;
  if (ctx.killCache.size >= KILL_CACHE_CAP) {
    const first = ctx.killCache.keys().next().value;
    if (first !== undefined) ctx.killCache.delete(first);
  }
  ctx.killCache.set(key, pairs);
  return pairs;
}

/** 行棋方（无 pending）所有能制造“对手可杀四”的完整回合。 */
function attackerThreatMoves(
  pos: Position, config: Config, ctx: Ctx,
  visit: (immediateWin: boolean) => boolean,
): boolean {
  const defender = other(pos.player);
  const attacker = pos.player;
  for (const cap of legalMoves(pos, config)) {
    if (pos.board[cap] !== defender) continue;
    const u1 = applyStep(pos, cap, config);
    if (!pos.gameOver) {
      const placements = legalMoves(pos, config);
      for (const p of placements) {
        const u2 = applyStep(pos, p, config);
        let stop = false;
        if (!pos.gameOver || pos.loser === defender) {
          const win = pos.gameOver && pos.loser === defender;
          if (win) stop = visit(true);
          else {
            const pr = (p / N) | 0, pc = p - pr * N;
            if (jumpThrough(pos.board, pr, pc, defender) >= 4 &&
                cachedPairs(ctx, pos, attacker, config).length > 0) {
              stop = visit(false);
            }
          }
        }
        undoStep(pos, u2);
        if (stop) { undoStep(pos, u1); return true; }
      }
    }
    undoStep(pos, u1);
  }
  return false;
}

interface Reach { [target: number]: number[] }

function defenderReach(pos: Position, config: Config): Reach {
  const defender = pos.player;
  const attacker = other(defender);
  const reach: Reach = {};
  for (const x of legalMoves(pos, config)) {
    if (pos.board[x] !== attacker) continue;
    const r = (x / N) | 0, c = x - r * N;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]] as const) {
      let nr = r + dr, nc = c + dc;
      while (nr >= 0 && nr < N && nc >= 0 && nc < N && pos.board[nr * N + nc] === EMPTY) {
        const t = nr * N + nc;
        (reach[t] ??= []).push(x);
        nr += dr; nc += dc;
      }
    }
  }
  return reach;
}

/** 防守方最终落点 = target 的所有完整回合；返回 true 表示找到逃生。 */
function defenderReplies(
  pos: Position, config: Config, target: number, reach: Reach, ctx: Ctx,
  movesLeft: number, attacker: number,
): boolean {
  const defender = pos.player;
  const legal = legalMoves(pos, config);
  const full = (): boolean => !pos.board.includes(EMPTY);
  if (pos.board[target] === EMPTY && legal.includes(target)) {
    const u = applyStep(pos, target, config);
    let escape = false;
    if (pos.gameOver) {
      if (pos.loser === defender) escape = false;
      else if (pos.loser === attacker) escape = true;
      else escape = true;
    } else if (full()) escape = true;
    else escape = !attackerWinsIn(pos, config, ctx, movesLeft);
    undoStep(pos, u);
    if (escape) return true;
  }
  for (const x of reach[target] ?? []) {
    if (pos.board[x] !== attacker) continue;
    const u1 = applyStep(pos, x, config);
    if (pos.gameOver) { undoStep(pos, u1); continue; }
    if (pos.board[target] !== EMPTY) { undoStep(pos, u1); continue; }
    const pm = legalMoves(pos, config);
    if (!pm.includes(target)) { undoStep(pos, u1); continue; }
    const u2 = applyStep(pos, target, config);
    let escape = false;
    if (pos.gameOver) {
      if (pos.loser === defender) escape = false;
      else if (pos.loser === attacker) escape = true;
      else escape = true;
    } else if (full()) escape = true;
    else escape = !attackerWinsIn(pos, config, ctx, movesLeft);
    undoStep(pos, u2);
    undoStep(pos, u1);
    if (escape) return true;
  }
  return false;
}

export function attackerWinsIn(
  pos: Position, config: Config, ctx: Ctx, movesLeft: number,
): boolean {
  if (tick(ctx)) return false;
  const key = boardKey(pos) + "|A|" + movesLeft;
  const hit = ctx.memoA.get(key);
  if (hit !== undefined) return hit;
  if (hasKill(pos, config)) { memoStore(ctx.memoA, key, true); return true; }
  if (movesLeft <= 0) { memoStore(ctx.memoA, key, false); return false; }
  const defender = other(pos.player);
  let found = false;
  attackerThreatMoves(pos, config, ctx, (immediateWin) => {
    if (immediateWin) { found = true; return true; }
    if (!defenderSurvives(pos, config, ctx, defender, movesLeft - 1)) {
      found = true; return true;
    }
    if (ctx.aborted) { found = false; return true; }
    return false;
  });
  if (ctx.aborted) return false;
  memoStore(ctx.memoA, key, found);
  return found;
}

export function defenderSurvives(
  pos: Position, config: Config, ctx: Ctx, defender: number, movesLeft: number,
): boolean {
  if (tick(ctx)) return true;
  const key = boardKey(pos) + "|D|" + movesLeft;
  const hit = ctx.memoD.get(key);
  if (hit !== undefined) return hit;
  if (hasKill(pos, config)) { memoStore(ctx.memoD, key, true); return true; }
  const legal = legalMoves(pos, config);
  if (!legal.length) { memoStore(ctx.memoD, key, false); return false; }
  const attacker = other(defender);
  const pairs = cachedPairs(ctx, pos, attacker, config);
  if (!pairs.length) { memoStore(ctx.memoD, key, true); return true; }
  let inter = new Set<number>(pairs[0].route);
  inter.add(pairs[0].target);
  for (let i = 1; i < pairs.length; i++) {
    const next = new Set<number>(pairs[i].route);
    next.add(pairs[i].target);
    for (const v of Array.from(inter)) if (!next.has(v)) inter.delete(v);
    if (!inter.size) break;
  }
  if (!inter.size) { memoStore(ctx.memoD, key, false); return false; }
  const reach = defenderReach(pos, config);
  for (const target of inter) {
    if (ctx.aborted) return true;
    if (defenderReplies(pos, config, target, reach, ctx, movesLeft, attacker)) {
      memoStore(ctx.memoD, key, true);
      return true;
    }
    if (ctx.aborted) return true;
  }
  memoStore(ctx.memoD, key, false);
  return false;
}

function makeCtx(opts: VcfOptions): Ctx {
  return {
    deadline: opts.deadline ?? Infinity,
    nodeCap: opts.nodeCap ?? 60_000,
    nodes: 0,
    aborted: false,
    memoA: new Map(),
    memoD: new Map(),
    killCache: new Map(),
    onProgress: opts.onProgress,
  };
}

/** 尝试证明行棋方存在强制杀链，返回起始 step；未证明返回 -1。 */
export function findForcedKill(
  pos: Position, config: Config, opts: VcfOptions = {},
): number {
  if (pos.turnCount < config.loss_start_turns) return -1;
  const depth = opts.maxDepth ?? 8;
  const ctx = makeCtx(opts);
  const defender = other(pos.player);
  const attacker = pos.player;
  if (pos.pending >= 0) {
    const kills = killPlacementsConfig(pos, config.loss_start_turns);
    if (kills.length) return kills[0];
    for (const p of legalMoves(pos, config)) {
      const u = applyStep(pos, p, config);
      let result = -1;
      if (pos.gameOver) {
        if (pos.loser === defender) result = p;
      } else {
        const pr = (p / N) | 0, pc = p - pr * N;
        if (jumpThrough(pos.board, pr, pc, defender) >= 4 &&
            cachedPairs(ctx, pos, attacker, config).length > 0 &&
            !defenderSurvives(pos, config, ctx, defender, depth - 1)) {
          result = p;
        }
      }
      undoStep(pos, u);
      if (result >= 0) return result;
      if (ctx.aborted) return -1;
    }
    return -1;
  }
  // 与 Python find_forced_kill 一致：先信任 kill_captures 的第一个两步杀。
  const quick = killCaptures(pos, config);
  if (quick.length) return quick[0][0];

  for (const cap of legalMoves(pos, config)) {
    if (pos.board[cap] !== defender) continue;
    const u1 = applyStep(pos, cap, config);
    if (!pos.gameOver) {
      const placements = legalMoves(pos, config);
      for (const p of placements) {
        const u2 = applyStep(pos, p, config);
        let hit = false;
        if (pos.gameOver && pos.loser === defender) hit = true;
        else if (!pos.gameOver) {
          const pr = (p / N) | 0, pc = p - pr * N;
          if (jumpThrough(pos.board, pr, pc, defender) >= 4 &&
              cachedPairs(ctx, pos, attacker, config).length > 0 &&
              !defenderSurvives(pos, config, ctx, defender, depth - 1)) {
            hit = true;
          }
        }
        undoStep(pos, u2);
        if (hit) { undoStep(pos, u1); return cap; }
        if (ctx.aborted) { undoStep(pos, u1); return -1; }
      }
    }
    undoStep(pos, u1);
  }
  return -1;
}
