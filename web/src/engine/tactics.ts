/** 逆五子棋战术层：安置杀、占领+安置两步杀、跳形延伸。 */
import {
  DEFAULT_CONFIG, EMPTY, N, checkWinAt, legalMoves, other,
} from "./rules.ts";
import type { Config } from "./rules.ts";
import type { Position } from "./rules.ts";

const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]] as const;

export interface KillPair {
  capture: number;
  target: number;
  route: number[];
}

/** 与 Python tactics.kill_placements 等价：需要 config 判断判负窗口。 */
export function killPlacementsConfig(pos: Position, lossStart: number, legalInput?: number[]): number[] {
  if (pos.pending < 0 || pos.turnCount < lossStart) return [];
  const opp = other(pos.player);
  const legal = legalInput ?? legalMoves(pos, {
    white_restrict_turns: 0, loss_start_turns: lossStart, mask_suicide: true,
  });
  return legal.filter((t) => checkWinAt(pos.board, t, opp));
}

/** 若在 (r,c) 落 color 子，经过该格的最大子数（允许一个空档）。 */
export function jumpThrough(board: Int8Array, r: number, c: number, color: number): number {
  let best = 0;
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]] as const) {
    let lo = 0;
    let hi = 0;
    let nr = r - dr, nc = c - dc;
    while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === color) {
      lo++; nr -= dr; nc -= dc;
    }
    nr = r + dr; nc = c + dc;
    while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === color) {
      hi++; nr += dr; nc += dc;
    }
    let extra = 0;
    let gr = r - (lo + 1) * dr, gc = c - (lo + 1) * dc;
    if (gr >= 0 && gr < N && gc >= 0 && gc < N && board[gr * N + gc] === EMPTY) {
      let lo2 = 0;
      nr = gr - dr; nc = gc - dc;
      while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === color) {
        lo2++; nr -= dr; nc -= dc;
      }
      extra = Math.max(extra, lo2);
    }
    gr = r + (hi + 1) * dr; gc = c + (hi + 1) * dc;
    if (gr >= 0 && gr < N && gc >= 0 && gc < N && board[gr * N + gc] === EMPTY) {
      let hi2 = 0;
      nr = gr + dr; nc = gc + dc;
      while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === color) {
        hi2++; nr += dr; nc += dc;
      }
      extra = Math.max(extra, hi2);
    }
    best = Math.max(best, 1 + lo + hi + extra);
  }
  return best;
}

function capturePaths(board: Int8Array, ci: number): { target: number; route: number[] }[] {
  const out: { target: number; route: number[] }[] = [];
  const r = (ci / N) | 0;
  const c = ci - r * N;
  for (const [dr, dc] of DIRS) {
    const route: number[] = [];
    let nr = r + dr, nc = c + dc;
    while (nr >= 0 && nr < N && nc >= 0 && nc < N) {
      const cell = nr * N + nc;
      if (board[cell] !== EMPTY) break;
      out.push({ target: cell, route: route.slice() });
      route.push(cell);
      nr += dr; nc += dc;
    }
  }
  return out;
}

/** 只计算空位上的 line5（killPairs 的目标只能是空位）。 */
function line5Empty(board: Int8Array, color: number): Uint8Array {
  const out = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) {
    if (board[i] === EMPTY && checkWinAt(board, i, color)) out[i] = 1;
  }
  return out;
}

/** 全部两步杀对（占领 idx、安置 idx、封堵 route）。 */
export function killPairs(pos: Position, config: Config = DEFAULT_CONFIG, legalInput?: number[], l5Input?: Uint8Array, skip?: boolean): KillPair[] {
  const board = pos.board;
  const p = pos.player;
  if (skip || pos.pending >= 0 || pos.turnCount < config.loss_start_turns) return [];
  const legal = legalInput ?? legalMoves(pos, config);
  const opp = other(p);
  const l5 = l5Input ?? line5Empty(board, opp);
  const danger: number[] = [];
  for (let i = 0; i < N * N; i++) if (l5[i]) danger.push(i);
  if (!danger.length) return [];

  // 每个危险格：在哪些轴上落子即连五（lo/hi 为两侧对方子数 + 1）
  const axisInfo = new Map<number, { dr: number; dc: number; lo: number; hi: number }[]>();
  for (const t of danger) {
    const tr = (t / N) | 0, tc2 = t - tr * N;
    const info: { dr: number; dc: number; lo: number; hi: number }[] = [];
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]] as const) {
      let lo = 1, hi = 1;
      let nr = tr - dr, nc = tc2 - dc;
      while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === opp) { lo++; nr -= dr; nc -= dc; }
      nr = tr + dr; nc = tc2 + dc;
      while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === opp) { hi++; nr += dr; nc += dc; }
      if (lo + hi - 1 >= 5) info.push({ dr, dc, lo, hi });
    }
    axisInfo.set(t, info);
  }

  const out: KillPair[] = [];
  for (const ci of legal) {
    if (board[ci] !== opp) continue;
    const cix = (ci / N) | 0, ciy = ci - cix * N;
    for (const { target: t, route } of capturePaths(board, ci)) {
      if (!l5[t]) continue;
      const tr = (t / N) | 0, tc2 = t - tr * N;
      let ok = false;
      for (const ax of axisInfo.get(t) ?? []) {
        if ((cix - tr) * ax.dc === (ciy - tc2) * ax.dr) {
          const s = ax.dr ? cix - tr : ciy - tc2;
          if (-(ax.lo - 1) <= s && s <= ax.hi - 1) continue;
        }
        ok = true;
        break;
      }
      if (ok) out.push({ capture: ci, target: t, route });
    }
  }
  return out;
}

/** 两步杀：每个占领格只取第一个安置点。 */
export function killCaptures(pos: Position, config: Config = DEFAULT_CONFIG, legalInput?: number[], l5Input?: Uint8Array, skip?: boolean): [number, number][] {
  const out: [number, number][] = [];
  const seen = new Set<number>();
  for (const k of killPairs(pos, config, legalInput, l5Input, skip)) {
    if (seen.has(k.capture)) continue;
    seen.add(k.capture);
    out.push([k.capture, k.target]);
  }
  return out;
}
