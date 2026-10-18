/** 迭代加深 alpha-beta + 置换表 + 战术杀棋。 */
import {
  applyStep, checkWinAt, legalMoves, other, undoStep,
} from "./rules.ts";
import type { Position, Config } from "./rules.ts";
import { MATE, evaluate, features } from "./eval.ts";
import { killCaptures, killPlacementsConfig } from "./tactics.ts";
import { profileFor } from "./profiles.ts";
import { candidates } from "./candidates.ts";
import type { Candidate } from "./candidates.ts";
import { jumpThrough } from "./tactics.ts";

const INF = 2 * MATE;
const now = (): number =>
  typeof performance !== "undefined" ? performance.now() : Date.now();

export interface SearchOptions {
  maxDepth: number;
  timeMs: number;
  engine?: import("./profiles.ts").EngineVersion;
  onProgress?: (nodes: number) => void;
  interruptBuffer?: Int32Array | null;
  maxNodes?: number;
}

export interface SearchResult {
  move: number;
  score: number;
  depth: number;
  nodes: number;
  elapsed: number;
}

interface TTEntry {
  depth: number;
  value: number;
  flag: 0 | 1 | 2;   // exact / lower / upper
  best: number;
}

const TT_LIMIT = 300_000;
const tt = new Map<number, TTEntry>();

function ttKey(pos: Position): number {
  let h = pos.hash | 0;
  h ^= Math.imul(pos.player + 1, 0x9e3779b1);
  h ^= Math.imul(pos.pending + 2, 0x85ebca6b);
  h ^= Math.imul(pos.turnCount + 1, 0xc2b2ae35);
  h ^= Math.imul(pos.whiteTurns + 1, 0x27d4eb2f);
  return h | 0;
}

function terminalScore(pos: Position, ply: number): number {
  if (pos.isDraw) return 0;
  return pos.loser === pos.player ? -(MATE - ply) : MATE - ply;
}

/** 当前持子方可一步安置杀（放对方子成五）。 */
export function placementKills(pos: Position, config: Config): number[] {
  if (pos.pending < 0 || pos.turnCount < config.loss_start_turns) return [];
  const opp = other(pos.player);
  return legalMoves(pos, config).filter((t) => checkWinAt(pos.board, t, opp));
}

/** 当前方可否本回合强制杀（持子安置杀，或占领+安置两步杀）。返回首步。 */
export function findWinningMove(pos: Position, config: Config): number {
  if (pos.pending >= 0) {
    const kills = placementKills(pos, config);
    return kills.length ? kills[0] : -1;
  }
  if (pos.turnCount < config.loss_start_turns) return -1;
  const opp = other(pos.player);
  for (const cap of legalMoves(pos, config)) {
    if (pos.board[cap] !== opp) continue;
    const u = applyStep(pos, cap, config);
    const kills = pos.gameOver ? [] : placementKills(pos, config);
    undoStep(pos, u);
    if (kills.length) return cap;
  }
  return -1;
}

function moveScore(pos: Position, idx: number, ttBest: number): number {
  let s = 0;
  if (idx === ttBest) s += 1e12;
  if (pos.pending >= 0) {
    if (checkWinAt(pos.board, idx, other(pos.player))) s += 1e11;
    s += 14 - Math.abs(((idx / 15) | 0) - 7) - Math.abs((idx % 15) - 7);
    return s;
  }
  if (pos.board[idx] === other(pos.player)) s += 1e9;
  if (checkWinAt(pos.board, idx, other(pos.player))) s += 1e10;
  if (checkWinAt(pos.board, idx, pos.player)) s -= 1e10;
  s += 14 - Math.abs(((idx / 15) | 0) - 7) - Math.abs((idx % 15) - 7);
  return s;
}

function orderedMoves(pos: Position, config: Config, ttBest: number, limit = Infinity): number[] {
  const moves = legalMoves(pos, config);
  if (moves.length <= 1) return moves;
  if (pos.pending >= 0) {
    const kills = killPlacementsConfig(pos, config.loss_start_turns);
    if (kills.length) return kills;
  }
  const scored = moves.map((m) => [moveScore(pos, m, ttBest), m] as const);
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, limit).map((x) => x[1]);
}

function innerLimit(depth: number): number {
  if (depth <= 2) return 3;
  if (depth <= 4) return 4;
  return 6;
}

const MATE_LIMIT = MATE - 450;
const PVS_EPS = 1e-6;

function includeExtras(cands: Candidate[], extras: Array<number | null>, legal: number[]): void {
  const present = new Set(cands.map((x) => x.move));
  const valid: number[] = [];
  for (const m of extras) {
    if (m === null || m < 0 || m >= 225 || present.has(m)) continue;
    if (!legal.includes(m)) continue;
    valid.push(m);
    present.add(m);
  }
  if (!valid.length) return;
  let at = 0;
  while (at < cands.length && cands[at].score >= 8e4) at++;
  for (const m of valid) cands.splice(at++, 0, { score: 1e-6, move: m });
}

function hasKillable4(pos: Position, config: Config, color: number): boolean {
  const f = features(pos, config);
  return (color === 1 ? f.black : f.white).killable4 > 0;
}

function boardFull(board: Int8Array): boolean {
  for (let i = 0; i < 225; i++) if (board[i] === 0) return false;
  return true;
}

export function searchBest(
  root: Position,
  config: Config,
  opts: SearchOptions,
): SearchResult {
  const t0 = now();
  const deadline = opts.timeMs > 0 ? t0 + opts.timeMs : Infinity;
  const distill = profileFor(opts.engine).distill;
  let nodes = 0;
  let aborted = false;
  let interrupted = false;
  let bestMove = -1;
  let bestScore = -INF;
  let completed = 0;
  const nodeLimit = opts.maxNodes && opts.maxNodes > 0 ? opts.maxNodes : Infinity;

  const win = findWinningMove(root, config);
  if (win >= 0) {
    return { move: win, score: MATE, depth: 1, nodes: 0, elapsed: now() - t0 };
  }
  const rootMoves = legalMoves(root, config);
  if (!rootMoves.length) {
    return { move: -1, score: root.gameOver ? terminalScore(root, 0) : -MATE,
      depth: 0, nodes: 0, elapsed: now() - t0 };
  }
  bestMove = rootMoves[0];

  const killers: Array<[number, number]> = Array.from({ length: 64 }, () => [-1, -1]);
  const history = new Float64Array(225);
  let checkCounter = 0;

  const checkTime = () => {
    if (now() > deadline) aborted = true;
    if (opts.interruptBuffer && Atomics.load(opts.interruptBuffer, 0) !== 0) {
      interrupted = true;
      aborted = true;
    }
  };

  function search(
    pos: Position, depth: number, alpha: number, beta: number, ply: number,
    ext: number, oppLast: number,
  ): number {
    nodes++;
    if (nodes >= nodeLimit) { aborted = true; return 0; }
    if ((nodes & 1023) === 0) opts.onProgress?.(nodes);
    if (++checkCounter >= 1024) {
      checkCounter = 0;
      checkTime();
      if (aborted) return 0;
    }
    if (pos.gameOver) return terminalScore(pos, ply);
    if (depth <= 0) return evaluate(pos, config, distill);

    const key = ttKey(pos);
    const entry = tt.get(key);
    let ttMove: number | null = null;
    if (entry) {
      ttMove = entry.best >= 0 ? entry.best : null;
      if (entry.depth >= depth) {
        if (entry.flag === 0) return entry.value;
        if (entry.flag === 1 && entry.value >= beta) return entry.value;
        if (entry.flag === 2 && entry.value <= alpha) return entry.value;
      }
    }

    const legal = legalMoves(pos, config);
    const [k1, k2] = killers[depth & 63];
    const cands = candidates(pos, config, innerLimit(depth), false, null, history, oppLast, legal);
    includeExtras(cands, [ttMove, k1, k2], legal);
    if (!cands.length) {
      if (boardFull(pos.board)) return 0;
      return -(MATE - ply);
    }

    const forced = cands.length <= 2 && cands[0].score >= 8e4;
    const mover = pos.player;
    const alpha0 = alpha;
    let best = -1e18;
    let bestMoveHere = -1;
    let searched = false;
    let extUsed = false;

    for (let idx = 0; idx < cands.length; idx++) {
      if ((idx & 3) === 0 && now() > deadline) { aborted = true; return best; }
      const m = cands[idx].move;
      const parentPending = pos.pending;
      const r = (m / 15) | 0;
      const c = m % 15;
      const u = applyStep(pos, m, config);
      const loser = pos.gameOver ? pos.loser : 0;
      if (loser === mover) { undoStep(pos, u); continue; }
      let v: number;
      if (loser === other(mover)) {
        v = MATE - ply;
      } else if (boardFull(pos.board)) {
        v = 0;
      } else if (pos.pending >= 0) {
        v = search(pos, depth - 1, alpha, beta, ply + 1, ext, oppLast);
      } else if (jumpThrough(pos.board, r, c, mover) >= 4
        && hasKillable4(pos, config, mover)) {
        v = -(MATE - (ply + 2));
      } else {
        let d2 = depth - 1;
        let childExt = ext;
        const opp = other(mover);
        if (ext > 0 && (forced || (parentPending >= 0
          && jumpThrough(pos.board, r, c, opp) >= 4
          && hasKillable4(pos, config, opp)))) {
          d2 = depth;
          childExt = 0;
          ext = 0;
          extUsed = true;
        }
        if (!searched) {
          v = -search(pos, d2, -beta, -alpha, ply + 1, childExt, parentPending >= 0 ? m : m);
        } else {
          v = -search(pos, d2, -alpha - PVS_EPS, -alpha, ply + 1, childExt, m);
          if (alpha < v && v < beta) {
            v = -search(pos, d2, -beta, -alpha, ply + 1, childExt, m);
          }
        }
      }
      undoStep(pos, u);
      if (aborted) return best;
      searched = true;
      if (v > best) {
        best = v;
        bestMoveHere = m;
        if (v > alpha) alpha = v;
        if (alpha >= beta) {
          if (Math.abs(best) < MATE_LIMIT) history[m] += depth * depth;
          const ks = killers[depth & 63];
          if (ks[0] !== m) { ks[1] = ks[0]; ks[0] = m; }
          break;
        }
      }
    }
    if (!aborted && Math.abs(best) < MATE_LIMIT) {
      const flag: 0 | 1 | 2 = best <= alpha0 ? 2 : best >= beta ? 1 : 0;
      if (tt.size >= TT_LIMIT) tt.clear();
      tt.set(key, { depth: extUsed ? depth - 1 : depth, value: best, flag, best: bestMoveHere });
    }
    return best;
  }

  for (let depth = 1; depth <= opts.maxDepth; depth++) {
    aborted = false;
    let localBest = -INF;
    let localMove = bestMove;
    const rootLegal = legalMoves(root, config);
    const list = candidates(root, config, 8, true, 8, null, -1, rootLegal);
    const ordered = list.length ? list.map((x) => x.move) : rootLegal;
    if (bestMove >= 0) {
      const at = ordered.indexOf(bestMove);
      if (at > 0) { ordered.splice(at, 1); ordered.unshift(bestMove); }
    }
    const mover = root.player;
    for (const m of ordered) {
      if (aborted || nodes >= nodeLimit) { aborted = true; break; }
      const u = applyStep(root, m, config);
      let child: number;
      if (root.gameOver) child = terminalScore(root, 1);
      else if (root.player === mover) child = search(root, depth - 1, -INF, INF, 1, 1, -1);
      else child = search(root, depth - 1, -INF, INF, 1, 1, -1);
      const score = root.player === mover ? child : -child;
      undoStep(root, u);
      if (aborted) break;
      if (score > localBest) { localBest = score; localMove = m; }
    }
    if (aborted) break;
    bestMove = localMove;
    bestScore = localBest;
    completed = depth;
    if (bestScore >= MATE - 100) break;
  }

  if (interrupted) throw new Error("搜索已中断");
  return { move: bestMove, score: bestScore, depth: completed, nodes, elapsed: now() - t0 };
}

export function clearTT(): void {
  tt.clear();
}

export interface AnalysisResult {
  posV: number;
  moves: { idx: number; v: number }[];
  depth: number;
  nodes: number;
  elapsed: number;
}

function rootCandidates(pos: Position, config: Config, k: number): number[] {
  const legal = legalMoves(pos, config);
  if (!legal.length) return [];
  if (pos.pending >= 0) {
    const kills = killPlacementsConfig(pos, config.loss_start_turns);
    if (kills.length) return kills.slice(0, k);
  } else {
    const kc = killCaptures(pos, config);
    if (kc.length) return kc.map((x) => x[0]).slice(0, k);
  }
  return orderedMoves(pos, config, -1, Math.max(k, 1)).slice(0, Math.max(k, 1));
}

/** 与 Python analysis_moves 同形的根分析：全窗口根候选价值 + 排序。 */
export function analyze(
  root: Position,
  config: Config,
  opts: SearchOptions & { k?: number },
): AnalysisResult {
  const t0 = now();
  const deadline = opts.timeMs > 0 ? t0 + opts.timeMs : Infinity;
  const k = Math.max(1, opts.k ?? 5);
  const terminal = (): AnalysisResult => {
    if (!root.gameOver) return { posV: 0, moves: [], depth: 0, nodes: 0, elapsed: 0 };
    const side = root.isDraw ? 0 : (root.loser === root.player ? -MATE : MATE);
    let cell = 0;
    for (let i = 0; i < root.board.length; i++) if (root.board[i] === 0) { cell = i; break; }
    return { posV: side, moves: [{ idx: cell, v: side }], depth: 0, nodes: 0, elapsed: 0 };
  };
  if (root.gameOver) return terminal();
  const candidates = rootCandidates(root, config, Math.max(k, 8));
  if (!candidates.length) return { posV: 0, moves: [], depth: 0, nodes: 0, elapsed: 0 };
  let bestMoves: { idx: number; v: number }[] = [];
  let posV = 0;
  let completed = 0;
  let nodes = 0;
  const mover = root.player;

  for (let depth = 1; depth <= opts.maxDepth; depth++) {
    const scored: { idx: number; v: number }[] = [];
    let aborted = false;
    for (const m of candidates) {
      if (now() > deadline) { aborted = true; break; }
      const u = applyStep(root, m, config);
      let v: number;
      if (root.gameOver) v = root.loser === mover ? -MATE : MATE;
      else {
        const remain = Number.isFinite(deadline)
          ? Math.max(1, (deadline - now()) / Math.max(1, candidates.length))
          : 0;
        const r = searchBest(root, config, {
          maxDepth: Math.max(1, depth - 1),
          timeMs: remain,
          engine: opts.engine,
          interruptBuffer: opts.interruptBuffer,
        });
        nodes += r.nodes;
        v = root.player === mover ? r.score : -r.score;
      }
      undoStep(root, u);
      scored.push({ idx: m, v });
    }
    if (aborted || !scored.length) break;
    scored.sort((a, b) => b.v - a.v);
    bestMoves = scored.slice(0, k);
    posV = bestMoves[0].v;
    completed = depth;
    opts.onProgress?.(nodes);
    if (posV >= MATE - 100) break;
  }
  return { posV, moves: bestMoves, depth: completed, nodes, elapsed: now() - t0 };
}
