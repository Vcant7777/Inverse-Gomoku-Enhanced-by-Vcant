/** 逆五子棋规则核心：与 Python antifive.reversegomoku 语义对齐。 */

export const BLACK = 1;
export const WHITE = 2;
export const EMPTY = 0;
export const N = 15;
export const SIZE = N * N;
export const LOSS_START_TURNS = 8;
export const NO_PENDING = -1;

export interface Config {
  white_restrict_turns: number;
  loss_start_turns: number;
  mask_suicide: boolean;
}

export const DEFAULT_CONFIG: Config = {
  white_restrict_turns: 2,
  loss_start_turns: LOSS_START_TURNS,
  mask_suicide: true,
};

export interface Position {
  board: Int8Array;
  player: number;
  pending: number;
  turnCount: number;
  whiteTurns: number;
  moveCount: number;
  gameOver: boolean;
  loser: number;
  isDraw: boolean;
  resigned: number;
  hash: number;
}

export interface Undo {
  idx: number;
  prevPlayer: number;
  prevPending: number;
  prevTurn: number;
  prevWhite: number;
  prevMoveCount: number;
  prevGameOver: boolean;
  prevLoser: number;
  prevIsDraw: boolean;
  wasCapture: boolean;
}

export const other = (c: number): number => (c === BLACK ? WHITE : BLACK);

const ZOBRIST = (() => {
  const table = new Int32Array(3 * SIZE);
  let x = 0x9e3779b9 | 0;
  for (let i = 0; i < table.length; i++) {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    table[i] = x | 0;
  }
  return table;
})();

export function computeHash(board: Int8Array): number {
  let h = 0;
  for (let i = 0; i < SIZE; i++) {
    const v = board[i];
    if (v) h ^= ZOBRIST[v * SIZE + i];
  }
  return h | 0;
}

export function createPosition(): Position {
  return {
    board: new Int8Array(SIZE),
    player: BLACK,
    pending: NO_PENDING,
    turnCount: 0,
    whiteTurns: 0,
    moveCount: 0,
    gameOver: false,
    loser: 0,
    isDraw: false,
    resigned: 0,
    hash: 0,
  };
}

export function clonePosition(p: Position): Position {
  return { ...p, board: p.board.slice() };
}

/** 以 color 落在 idx 后是否连五；idx 当前可以是空/对方子。 */
export function checkWinAt(board: Int8Array, idx: number, color: number): boolean {
  const r = (idx / N) | 0;
  const c = idx - r * N;
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]] as const) {
    let count = 1;
    let nr = r + dr;
    let nc = c + dc;
    while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === color) {
      count++;
      nr += dr;
      nc += dc;
    }
    nr = r - dr;
    nc = c - dc;
    while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === color) {
      count++;
      nr -= dr;
      nc -= dc;
    }
    if (count >= 5) return true;
  }
  return false;
}

/** 在 idx 落 color 子即连五的全部格子（不筛空位，含对方子位）。 */
export function line5Map(board: Int8Array, color: number): Uint8Array {
  const out = new Uint8Array(SIZE);
  for (let i = 0; i < SIZE; i++) {
    const v = board[i];
    if (v === color) continue;       // 已经是己方子，落子不改变局面
    if (checkWinAt(board, i, color)) out[i] = 1;
  }
  return out;
}

function hasAdjacentEmpty(board: Int8Array, idx: number): boolean {
  const r = (idx / N) | 0;
  const c = idx - r * N;
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === EMPTY) {
        return true;
      }
    }
  }
  return false;
}

/** 安置步：从 pending 格沿 8 方向可放到的空位（末端无空位则不可放）。 */
export function pendingTargets(board: Int8Array, pending: number): number[] {
  const out: number[] = [];
  const r = (pending / N) | 0;
  const c = pending - r * N;
  for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]] as const) {
    const line: number[] = [];
    let nr = r + dr;
    let nc = c + dc;
    while (nr >= 0 && nr < N && nc >= 0 && nc < N && board[nr * N + nc] === EMPTY) {
      line.push(nr * N + nc);
      nr += dr;
      nc += dc;
    }
    // Python: run_at >= k（k 为目标距离），整段连续空位均可放
    for (const cell of line) out.push(cell);
  }
  return out;
}

export function whiteRestricted(player: number, whiteTurns: number, config: Config): boolean {
  return config.white_restrict_turns > 0 && player === WHITE && whiteTurns < config.white_restrict_turns;
}

/** 当前合法落子/占领/安置格（升序）。 */
export function legalMoves(pos: Position, config: Config): number[] {
  const { board, player, pending, turnCount, whiteTurns } = pos;
  if (pending >= 0) return pendingTargets(board, pending).sort((a, b) => a - b);

  const out: number[] = [];
  for (let i = 0; i < SIZE; i++) if (board[i] === EMPTY) out.push(i);

  if (!whiteRestricted(player, whiteTurns, config)) {
    const opp = other(player);
    for (let i = 0; i < SIZE; i++) {
      if (board[i] === opp && hasAdjacentEmpty(board, i)) out.push(i);
    }
  }

  if (config.mask_suicide && turnCount >= config.loss_start_turns) {
    const l5 = line5Map(board, player);
    return out.filter((i) => l5[i] === 0).sort((a, b) => a - b);
  }
  return out.sort((a, b) => a - b);
}

/** 落子/占领/安置一步；返回可撤销信息。 */
export function applyStep(pos: Position, idx: number, config: Config): Undo {
  const u: Undo = {
    idx,
    prevPlayer: pos.player,
    prevPending: pos.pending,
    prevTurn: pos.turnCount,
    prevWhite: pos.whiteTurns,
    prevMoveCount: pos.moveCount,
    prevGameOver: pos.gameOver,
    prevLoser: pos.loser,
    prevIsDraw: pos.isDraw,
    wasCapture: false,
  };
  const board = pos.board;
  const player = pos.player;
  const opp = other(player);
  let loser = 0;
  const checkLoss = pos.turnCount >= config.loss_start_turns;

  if (pos.pending < 0) {
    if (board[idx] === opp) {
      u.wasCapture = true;
      pos.hash ^= ZOBRIST[opp * SIZE + idx] ^ ZOBRIST[player * SIZE + idx];
      board[idx] = player;
      pos.pending = idx;
      if (checkLoss && checkWinAt(board, idx, player)) loser = player;
    } else {
      pos.hash ^= ZOBRIST[player * SIZE + idx];
      board[idx] = player;
      pos.pending = NO_PENDING;
      pos.turnCount++;
      if (player === WHITE) pos.whiteTurns++;
      pos.player = opp;
      if (checkLoss && checkWinAt(board, idx, player)) loser = player;
    }
  } else {
    pos.hash ^= ZOBRIST[opp * SIZE + idx];
    board[idx] = opp;
    pos.pending = NO_PENDING;
    pos.player = opp;
    pos.turnCount++;
    if (player === WHITE) pos.whiteTurns++;
    if (checkLoss && checkWinAt(board, idx, opp)) loser = opp;
  }
  pos.moveCount++;
  if (loser) {
    pos.gameOver = true;
    pos.loser = loser;
  } else if (!board.includes(EMPTY)) {
    pos.gameOver = true;
    pos.isDraw = true;
    pos.loser = 0;
  }
  return u;
}

export function undoStep(pos: Position, u: Undo): void {
  if (u.wasCapture) {
    const old = pos.board[u.idx];
    pos.board[u.idx] = other(u.prevPlayer);
    pos.hash ^= ZOBRIST[old * SIZE + u.idx] ^ ZOBRIST[other(u.prevPlayer) * SIZE + u.idx];
  } else {
    const old = pos.board[u.idx];
    pos.board[u.idx] = EMPTY;
    if (old) pos.hash ^= ZOBRIST[old * SIZE + u.idx];
  }
  pos.player = u.prevPlayer;
  pos.pending = u.prevPending;
  pos.turnCount = u.prevTurn;
  pos.whiteTurns = u.prevWhite;
  pos.moveCount = u.prevMoveCount;
  pos.gameOver = u.prevGameOver;
  pos.loser = u.prevLoser;
  pos.isDraw = u.prevIsDraw;
}

export function winnerOf(p: Position): number {
  if (!p.gameOver || p.isDraw || !p.loser) return 0;
  return other(p.loser);
}

export function isStuck(pos: Position, config: Config): boolean {
  return legalMoves(pos, config).length === 0 && pos.board.includes(EMPTY);
}
