/** 线级材料/位置评估：移植 Python heuristic 的核心项，保持逆五子棋视角。 */
import {
  BLACK, EMPTY, N, checkWinAt, legalMoves, other,
} from "./rules.ts";
import type { Config, Position } from "./rules.ts";
import { valueCorrection } from "./distill.ts";
import type { DistillMode } from "./distill.ts";

const LINES: number[][] = [];
{
  for (let r = 0; r < N; r++) {
    const line: number[] = [];
    for (let c = 0; c < N; c++) line.push(r * N + c);
    LINES.push(line);
  }
  for (let c = 0; c < N; c++) {
    const line: number[] = [];
    for (let r = 0; r < N; r++) line.push(r * N + c);
    LINES.push(line);
  }
  for (let k = -(N - 1); k <= N - 1; k++) {
    const line: number[] = [];
    for (let r = 0; r < N; r++) {
      const c = r + k;
      if (c >= 0 && c < N) line.push(r * N + c);
    }
    LINES.push(line);
  }
  for (let k = 2; k <= 2 * N - 2; k++) {
    const line: number[] = [];
    for (let r = 0; r < N; r++) {
      const c = k - r;
      if (c >= 0 && c < N) line.push(r * N + c);
    }
    LINES.push(line);
  }
}

const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]] as const;
const CENTER = new Float64Array(N * N);
for (let r = 0; r < N; r++) {
  for (let c = 0; c < N; c++) {
    CENTER[r * N + c] = 1 - Math.max(Math.abs(r - 7), Math.abs(c - 7)) / 7;
  }
}

interface Mat {
  killable4: number;
  dead4: number;
  open3: number;
  jump3: number;
  closed3: number;
}

const emptyMat = (): Mat => ({ killable4: 0, dead4: 0, open3: 0, jump3: 0, closed3: 0 });

export interface Four {
  run: Set<number>;
  ends: number[];
}

interface Features {
  black: Mat;
  white: Mat;
  dangBlack: Set<number>;
  dangWhite: Set<number>;
  foursBlack: Four[];
  foursWhite: Four[];
}

function reachable(board: Int8Array, T: number, color: number, run: Set<number>): boolean {
  const r = (T / N) | 0;
  const c = T - r * N;
  for (const [dr, dc] of DIRS) {
    let nr = r + dr;
    let nc = c + dc;
    while (nr >= 0 && nr < N && nc >= 0 && nc < N) {
      const idx = nr * N + nc;
      const v = board[idx];
      if (v === EMPTY) {
        nr += dr;
        nc += dc;
        continue;
      }
      if (v === color && !run.has(idx)) return true;
      break;
    }
  }
  return false;
}

function tallyFour(
  board: Int8Array, cells: number[], m: Mat, dang: Set<number>, color: number,
  s: number, e: number, lOpen: boolean, rOpen: boolean, gap: number | null,
  canCapture: boolean, fours: Four[],
): void {
  const ends: number[] = [];
  const run = new Set<number>();
  if (gap !== null) {
    ends.push(cells[gap]);
    for (let i = s; i < gap; i++) run.add(cells[i]);
    for (let i = gap + 1; i < e; i++) run.add(cells[i]);
  } else {
    if (lOpen) ends.push(cells[s - 1]);
    if (rOpen) ends.push(cells[e]);
    for (let i = s; i < e; i++) run.add(cells[i]);
  }
  let killable = false;
  if (canCapture) {
    for (const t of ends) {
      if (reachable(board, t, color, run)) { killable = true; break; }
    }
  }
  if (killable) m.killable4++;
  else m.dead4++;
  for (const t of ends) dang.add(t);
  fours.push({ run, ends });
}

export function features(pos: Position, config: Config): Features {
  const mB = emptyMat();
  const mW = emptyMat();
  const dB = new Set<number>();
  const dW = new Set<number>();
  const foursB: Four[] = [];
  const foursW: Four[] = [];
  if (pos.turnCount < config.loss_start_turns) {
    return { black: mB, white: mW, dangBlack: dB, dangWhite: dW, foursBlack: foursB, foursWhite: foursW };
  }
  const board = pos.board;
  const capBlack = pos.whiteTurns >= config.white_restrict_turns;
  const capWhite = true;
  for (const cells of LINES) {
    const n = cells.length;
    let i = 0;
    while (i < n) {
      const v = board[cells[i]];
      let j = i + 1;
      while (j < n && board[cells[j]] === v) j++;
      const s = i;
      const e = j;
      i = j;
      if (v === EMPTY) continue;
      const L = e - s;
      const isBlack = v === BLACK;
      const m = isBlack ? mB : mW;
      const dang = isBlack ? dB : dW;
      const cap = isBlack ? capBlack : capWhite;
      const lOpen = s > 0 && board[cells[s - 1]] === EMPTY;
      const rOpen = e < n && board[cells[e]] === EMPTY;
      if (L === 4) {
        tallyFour(board, cells, m, dang, v, s, e, lOpen, rOpen, null, cap, isBlack ? foursB : foursW);
      } else if (L === 3) {
        if (lOpen && rOpen) m.open3++;
        else m.closed3++;
      }
      if (e + 2 < n && board[cells[e]] === EMPTY && board[cells[e + 1]] === v) {
        let j2 = e + 1;
        while (j2 < n && board[cells[j2]] === v) j2++;
        const total = L + (j2 - e - 1);
        if (total >= 4) {
          tallyFour(board, cells, m, dang, v, s, j2, lOpen,
            j2 < n && board[cells[j2]] === EMPTY, e, cap, isBlack ? foursB : foursW);
        } else if (total === 3) {
          m.jump3++;
        }
      }
    }
  }
  return { black: mB, white: mW, dangBlack: dB, dangWhite: dW, foursBlack: foursB, foursWhite: foursW };
}

interface PosSummary {
  adjB: number; adjW: number; clB: number; clW: number; ctB: number; ctW: number;
}

function positional(board: Int8Array): PosSummary {
  let adjB = 0, adjW = 0, clB = 0, clW = 0, ctB = 0, ctW = 0;
  for (let i = 0; i < N * N; i++) {
    const v = board[i];
    if (v === EMPTY) continue;
    const r = (i / N) | 0;
    const c = i - r * N;
    let nSame = 0;
    for (const [dr, dc] of DIRS) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === v) nSame++;
    }
    const center = CENTER[i];
    if (v === BLACK) {
      adjB += nSame;
      clB += nSame * center;
      ctB += center;
    } else {
      adjW += nSame;
      clW += nSame * center;
      ctW += center;
    }
  }
  return { adjB: adjB >> 1, adjW: adjW >> 1, clB, clW, ctB, ctW };
}

const E_KILLABLE4 = 0.9, E_DEAD4 = 0.2, E_OPEN3 = 0.35, E_JUMP3 = 0.22, E_CLOSED3 = 0.1;
const E_DANGER = 0.02;
const E_SELF_KILLABLE4 = 0.95, E_SELF_DEAD4 = 0.3, E_SELF_OPEN3 = 0.5, E_SELF_JUMP3 = 0.25, E_SELF_CLOSED3 = 0.1;
const E_DOUBLE_KILL = 1.2, E_KILL_PLUS3 = 0.5, E_PENDING_KILL = 0.95;
const E_ADJ = 0.02, E_CENTER_OPP = 0.70, E_CENTER_SELF = 0.80, E_CLUSTER = 0.03;

export const MATE = 1_000_000;

/** 从 pos.player 视角评估。distill=1/2 时叠加 KataGo 价值校正。 */
export function evaluate(pos: Position, config: Config, distill: DistillMode = 0): number {
  if (pos.gameOver) {
    if (pos.isDraw) return 0;
    return pos.loser === pos.player ? -MATE : MATE;
  }
  const me = pos.player;
  const f = features(pos, config);
  const mine = me === BLACK ? f.black : f.white;
  const theirs = me === BLACK ? f.white : f.black;
  const dangerCnt = me === BLACK ? f.dangWhite.size : f.dangBlack.size;
  const dangerMine = me === BLACK ? f.dangBlack.size : f.dangWhite.size;
  let s = 0;
  s += E_KILLABLE4 * theirs.killable4 + E_DEAD4 * theirs.dead4;
  s += E_OPEN3 * theirs.open3 + E_JUMP3 * theirs.jump3 + E_CLOSED3 * theirs.closed3;
  s -= E_SELF_KILLABLE4 * mine.killable4 + E_SELF_DEAD4 * mine.dead4;
  s -= E_SELF_OPEN3 * mine.open3 + E_SELF_JUMP3 * mine.jump3 + E_SELF_CLOSED3 * mine.closed3;
  if (theirs.killable4 >= 2) s += E_DOUBLE_KILL;
  else if (theirs.killable4 === 1 && theirs.open3 + theirs.jump3 >= 1) s += E_KILL_PLUS3;
  if (mine.killable4 >= 2) s -= E_DOUBLE_KILL;
  else if (mine.killable4 === 1 && mine.open3 + mine.jump3 >= 1) s -= E_KILL_PLUS3;
  const empty = N * N - countStones(pos.board);
  s += E_DANGER * Math.min(dangerCnt, 20);
  s -= E_DANGER * Math.min(dangerMine, 20);
  const p = positional(pos.board);
  if (me === BLACK) {
    s += E_ADJ * (p.adjW - p.adjB);
    s += E_CENTER_OPP * p.ctW - E_CENTER_SELF * p.ctB;
    s += E_CLUSTER * (p.clW - p.clB);
  } else {
    s += E_ADJ * (p.adjB - p.adjW);
    s += E_CENTER_OPP * p.ctB - E_CENTER_SELF * p.ctW;
    s += E_CLUSTER * (p.clB - p.clW);
  }
  const theirSafe = empty - dangerCnt;
  if (theirSafe <= 2) s += 0.8;
  else if (theirSafe <= 5) s += 0.4;
  else if (theirSafe <= 10) s += 0.2;
  const mySafe = empty - dangerMine;
  if (mySafe <= 2) s -= 0.8;
  else if (mySafe <= 5) s -= 0.4;
  if (pos.pending >= 0 && pos.turnCount >= config.loss_start_turns) {
    const targets = legalMoves(pos, config);
    const opp2 = other(pos.player);
    if (targets.some((t) => checkWinAt(pos.board, t, opp2))) s += E_PENDING_KILL;
  }
  if (distill) {
    const adjM = me === BLACK ? p.adjB : p.adjW;
    const adjT = me === BLACK ? p.adjW : p.adjB;
    const clM = me === BLACK ? p.clB : p.clW;
    const clT = me === BLACK ? p.clW : p.clB;
    const ctM = me === BLACK ? p.ctB : p.ctW;
    const ctT = me === BLACK ? p.ctW : p.ctB;
    const f: number[] = [1,
      mine.killable4, mine.dead4, mine.open3, mine.jump3, mine.closed3,
      theirs.killable4, theirs.dead4, theirs.open3, theirs.jump3, theirs.closed3,
      dangerMine, dangerCnt, empty,
      adjM, adjT, clM, clT, ctM, ctT];
    if (distill === 2) {
      f.push(pos.pending >= 0 ? 1 : 0);
      f.push(Math.min(pos.turnCount, 63) / 63);
      f.push(Math.min(pos.whiteTurns, 63) / 63);
      f.push((empty - dangerMine) / (N * N));
      f.push((empty - dangerCnt) / (N * N));
    }
    s += valueCorrection(distill, f);
  }
  return s;
}

function countStones(board: Int8Array): number {
  let n = 0;
  for (let i = 0; i < N * N; i++) if (board[i] !== EMPTY) n++;
  return n;
}

export function evaluateLeaf(pos: Position, config: Config, distill: DistillMode = 0): number {
  return evaluate(pos, config, distill);
}
