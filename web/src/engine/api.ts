/** 网页版 TypeScript 引擎会话层：与 Python web_api 的命令/状态形状对齐。 */
import {
  BLACK, EMPTY, N, SIZE, WHITE, applyStep, createPosition,
  legalMoves, line5Map, other, pendingTargets, winnerOf,
} from "./rules.ts";
import type { Config, Position, Undo } from "./rules.ts";
import { DEFAULT_CONFIG } from "./rules.ts";
import { searchBest, clearTT, analyze } from "./search.ts";
import { openingMove } from "./openbook.ts";
import { depthForProfile, profileFor, vcfSeconds } from "./profiles.ts";
import { findForcedKill } from "./vcf.ts";
import { MATE } from "./eval.ts";
import { killCaptures } from "./tactics.ts";

export interface StepRecord {
  idx: number;
  player: number;
  pending: number;
  turnCount: number;
  whiteTurns: number;
}

export interface SessionOptions {
  config?: Partial<Config>;
}

type Req = Record<string, unknown>;

export class EngineSession {
  config: Config;
  pos: Position = createPosition();
  steps: StepRecord[] = [];
  lastNodes = 0;
  lastElapsed = 0;
  resigned = 0;
  lastEngine = "advanced";
  onProgress: ((nodes: number) => void) | null = null;
  interruptBuffer: Int32Array | null = null;

  constructor(opts: SessionOptions = {}) {
    this.config = { ...DEFAULT_CONFIG, ...(opts.config ?? {}) };
  }

  reset(config?: Partial<Config>): void {
    if (config) this.config = { ...DEFAULT_CONFIG, ...config };
    this.pos = createPosition();
    this.steps = [];
    this.lastNodes = 0;
    this.lastElapsed = 0;
    this.resigned = 0;
    this.lastEngine = "advanced";
    clearTT();
  }

  private progress = (nodes: number): void => {
    this.lastNodes = nodes;
    this.onProgress?.(nodes);
  };

  private pushStep(u: Undo): void {
    // 与 Python history 一致：记录该步“走之前”的状态，
    // kinds 才能区分 capture(走后 pending 指向本步) 与 place2(走前 pending 指向本步)。
    this.steps.push({
      idx: u.idx,
      player: u.prevPlayer,
      pending: u.prevPending,
      turnCount: u.prevTurn,
      whiteTurns: u.prevWhite,
    });
  }

  /** 从着法序列重建；非法/终局处停止。 */
  replay(moves: number[]): void {
    const cfg = this.config;
    this.pos = createPosition();
    this.steps = [];
    this.resigned = 0;
    for (const m of moves) {
      const legal = legalMoves(this.pos, cfg);
      if (!legal.length) break;
      if (!legal.includes(m)) throw new Error(`棋谱第 ${this.steps.length + 1} 手非法`);
      this.pushStep(applyStep(this.pos, m, cfg));
    }
  }

  private finishStuck(): boolean {
    if (this.pos.gameOver) return false;
    const legal = legalMoves(this.pos, this.config);
    if (legal.length || !this.pos.board.includes(EMPTY)) return false;
    const mover = this.pos.player;
    if (this.pos.pending >= 0 || this.pos.turnCount < this.config.loss_start_turns) {
      this.pos.gameOver = true;
      this.pos.loser = mover;
      this.pos.isDraw = false;
      return true;
    }
    const winner = other(mover);
    const saved = this.pos.player;
    this.pos.player = winner;
    const kills = killCaptures(this.pos, this.config);
    if (kills.length) {
      const [cap, place] = kills[0];
      this.pushStep(applyStep(this.pos, cap, this.config));
      if (!this.pos.gameOver) {
        this.pushStep(applyStep(this.pos, place, this.config));
      }
      if (this.pos.gameOver && !this.pos.isDraw) return true;
    }
    this.pos.player = saved;
    this.pos.gameOver = true;
    this.pos.loser = mover;
    this.pos.isDraw = false;
    return true;
  }

  private kinds(): string[] {
    return this.steps.map((s, i) => {
      if (s.pending >= 0) return "place2";
      const next = this.steps[i + 1];
      if ((next && next.pending === s.idx) || (i === this.steps.length - 1 && this.pos.pending === s.idx)) {
        return "capture";
      }
      return "place";
    });
  }

  gameResult(): string {
    if (!this.pos.gameOver) return "未完";
    if (this.pos.isDraw) return "和棋";
    if (this.resigned) return this.resigned === BLACK ? "黑认输" : "白认输";
    return this.pos.loser === BLACK ? "白胜" : "黑胜";
  }

  state(): Record<string, unknown> {
    const p = this.pos;
    const me = p.player;
    const opp = other(me);
    const legal = legalMoves(p, this.config);
    const numbers = new Array<number>(SIZE).fill(0);
    this.steps.forEach((s, i) => { numbers[s.idx] = i + 1; });
    const dme = line5Map(p.board, me);
    const dopp = line5Map(p.board, opp);
    for (let i = 0; i < SIZE; i++) {
      if (p.board[i] !== EMPTY) dme[i] = 0;
      if (p.board[i] !== EMPTY) dopp[i] = 0;
    }
    return {
      board: Array.from(p.board),
      player: me,
      pending: p.pending,
      pending_color: p.pending >= 0 ? opp : 0,
      turn_count: p.turnCount,
      white_turns: p.whiteTurns,
      move_count: p.moveCount,
      game_over: p.gameOver,
      loser: p.loser,
      is_draw: p.isDraw,
      resigned: this.resigned,
      winner: winnerOf(p),
      stuck: legal.length === 0 && p.board.includes(EMPTY),
      result: this.gameResult(),
      legal,
      moves: this.steps.map((s) => s.idx),
      kinds: this.kinds(),
      numbers,
      danger_me: Array.from(dme),
      danger_opp: Array.from(dopp),
      config: {
        white_restrict_turns: this.config.white_restrict_turns,
        loss_start_turns: this.config.loss_start_turns,
        mask_suicide: this.config.mask_suicide,
      },
    };
  }

  private ensurePlayable(): void {
    if (this.pos.gameOver) throw new Error("对局已结束");
  }

  newGame(config?: Partial<Config>): Record<string, unknown> {
    this.reset(config);
    return { state: this.state() };
  }

  move(idx: number, allowSuicide = false): Record<string, unknown> {
    this.ensurePlayable();
    let legal = legalMoves(this.pos, this.config);
    if (!legal.includes(idx) && allowSuicide) {
      legal = legalMoves(this.pos, { ...this.config, mask_suicide: false });
    }
    if (!legal.includes(idx)) {
      if (!legal.length && this.pos.board.includes(EMPTY)) {
        this.finishStuck();
        return { state: this.state(), finished: true };
      }
      throw new Error(`非法着法 idx=${idx}`);
    }
    this.pushStep(applyStep(this.pos, idx, this.config));
    let finished = false;
    if (!this.pos.gameOver && !legalMoves(this.pos, this.config).length) {
      finished = this.finishStuck();
    }
    return { state: this.state(), finished };
  }

  undo(): Record<string, unknown> {
    if (!this.steps.length) throw new Error("没有可悔的着法");
    const moves = this.steps.map((s) => s.idx);
    moves.pop();
    this.replay(moves);
    return { state: this.state() };
  }

  resign(player = 0): Record<string, unknown> {
    this.ensurePlayable();
    const who = player || this.pos.player;
    this.resigned = who;
    this.pos.gameOver = true;
    this.pos.loser = who;
    this.pos.isDraw = false;
    return { state: this.state() };
  }

  ai(req: Req): Record<string, unknown> {
    this.ensurePlayable();
    const profile = profileFor(req.engine);
    if (profile.version !== this.lastEngine) {
      clearTT();
      this.lastEngine = profile.version;
    }
    if (profile.supportsBook) {
      const book = openingMove(this.pos);
      if (book >= 0 && legalMoves(this.pos, this.config).includes(book)) {
        this.pushStep(applyStep(this.pos, book, this.config));
        return { state: this.state(), move: book, nodes: 0, elapsed: 0 };
      }
    }
    const depth = Math.max(1, Math.floor(Number(req.depth ?? 4)));
    const baseDepth = depthForProfile(profile.version, depth);
    const budget = req.budget ? Number(req.budget) : 0;
    const t0 = Date.now();
    this.lastNodes = 0;

    const vcfEnabled = profile.supportsVcf &&
      (req.vcf === undefined ? profile.defaultVcf : Boolean(req.vcf));
    if (vcfEnabled) {
      const vcfMove = findForcedKill(this.pos, this.config, {
        maxDepth: profile.vcfDepth,
        deadline: Date.now() + vcfSeconds(budget) * 1000,
        nodeCap: profile.vcfNodes,
        onProgress: this.progress,
      });
      if (vcfMove >= 0) {
        this.pushStep(applyStep(this.pos, vcfMove, this.config));
        this.lastElapsed = (Date.now() - t0) / 1000;
        return { state: this.state(), move: vcfMove,
          nodes: this.lastNodes, elapsed: this.lastElapsed };
      }
    }

    const dynamic = budget > 0 && profile.supportsVcf
      ? (profile.version === "mid" ? Math.max(baseDepth, 8) : Math.max(baseDepth, 16))
      : baseDepth;
    const result = searchBest(this.pos, this.config, {
      maxDepth: dynamic,
      timeMs: budget > 0 ? Math.max(1, budget * 1000 - (Date.now() - t0)) : 0,
      engine: profile.version,
      onProgress: this.progress,
      interruptBuffer: this.interruptBuffer,
      maxNodes: req.nodeLimit ? Number(req.nodeLimit) : undefined,
    });
    this.lastNodes = result.nodes;
    this.lastElapsed = (Date.now() - t0) / 1000;
    if (result.move < 0) {
      this.finishStuck();
      return { state: this.state(), move: null, nodes: this.lastNodes, elapsed: this.lastElapsed };
    }
    this.pushStep(applyStep(this.pos, result.move, this.config));
    return {
      state: this.state(), move: result.move,
      nodes: this.lastNodes, elapsed: this.lastElapsed,
    };
  }

  hint(req: Req): Record<string, unknown> {
    this.ensurePlayable();
    const profile = profileFor(req.engine);
    if (profile.version !== this.lastEngine) {
      clearTT();
      this.lastEngine = profile.version;
    }
    const depth = Math.max(1, Math.floor(Number(req.depth ?? 2)));
    const budget = req.budget === undefined || req.budget === null ? 2 : Number(req.budget);
    const vcfEnabled = profile.supportsVcf &&
      (req.vcf === undefined ? profile.defaultVcf : Boolean(req.vcf));
    if (vcfEnabled) {
      const vcfMove = findForcedKill(this.pos, this.config, {
        maxDepth: profile.vcfDepth,
        deadline: Date.now() + vcfSeconds(budget) * 1000,
        nodeCap: profile.vcfNodes,
        onProgress: this.progress,
      });
      if (vcfMove >= 0) {
        return { pos_v: MATE - 1, moves: [{ idx: vcfMove, v: MATE - 1 }] };
      }
    }
    const r = analyze(this.pos, this.config, {
      maxDepth: depthForProfile(profile.version, depth),
      timeMs: budget > 0 ? budget * 1000 : 0,
      k: 5,
      engine: profile.version,
      interruptBuffer: this.interruptBuffer,
      onProgress: this.progress,
    });
    return { pos_v: r.posV, moves: r.moves };
  }

  rays(idx: number): Record<string, unknown> {
    if (this.pos.pending >= 0) return { rays: [] };
    if (idx < 0 || idx >= SIZE) return { rays: [] };
    const opp = other(this.pos.player);
    if (this.pos.board[idx] !== opp) return { rays: [] };
    if (!legalMoves(this.pos, this.config).includes(idx)) return { rays: [] };
    return { rays: pendingTargets(this.pos.board, idx).sort((a, b) => a - b) };
  }

  goto(moves: number[], pos: number, result: string): Record<string, unknown> {
    const n = Math.max(0, Math.min(pos, moves.length));
    this.replay(moves.slice(0, n));
    if (n === moves.length) {
      const loser = result === "黑认输" ? BLACK : result === "白认输" ? WHITE : 0;
      if (loser && !this.pos.gameOver) this.resign(loser);
    }
    return { state: this.state(), pos: n };
  }

  exportText(movesIn?: number[], resultIn?: string): Record<string, unknown> {
    let moves = movesIn ?? this.steps.map((s) => s.idx);
    let result = resultIn ?? this.gameResult();
    if (movesIn && movesIn.length && resultIn === undefined) result = "未完";
    if (this.pos.pending >= 0) {
      if (!this.pos.gameOver) throw new Error("移子待定:请先完成安置再保存");
      if (!movesIn) moves = moves.slice(0, -1);
    }
    return { text: dumps(moves.slice(), this.config, result) };
  }

  importText(text: string): Record<string, unknown> {
    const rec = loads(text);
    if (!rec.moves.length) throw new Error("棋谱中没有着法");
    this.config = rec.config;
    this.replay(rec.moves);
    const loser = rec.result === "黑认输" ? BLACK : rec.result === "白认输" ? WHITE : 0;
    if (loser && !this.pos.gameOver) this.resign(loser);
    return { state: this.state(), result: rec.result };
  }

  handle(req: Req): Record<string, unknown> {
    const cmd = String(req.cmd ?? "");
    switch (cmd) {
      case "new": return this.newGame(req.config as Partial<Config> | undefined);
      case "state": return { state: this.state() };
      case "move": return this.move(Number(req.idx), Boolean(req.allow_suicide));
      case "undo": return this.undo();
      case "resign": return this.resign(Number(req.player ?? 0));
      case "ai": return this.ai(req);
      case "hint": return this.hint(req);
      case "rays": return this.rays(Number(req.idx ?? -1));
      case "goto": return this.goto((req.moves as number[] | undefined) ?? [], Number(req.pos ?? 0), String(req.result ?? ""));
      case "export": return this.exportText(req.moves as number[] | undefined, req.result as string | undefined);
      case "import": return this.importText(String(req.text ?? ""));
      case "stuck":
        this.finishStuck();
        return { state: this.state() };
      default: throw new Error(`未知命令: '${cmd}'`);
    }
  }
}

// ---- .afg 记录 ----
export function idxToCoord(idx: number): string {
  const r = (idx / N) | 0;
  return `${String.fromCharCode(65 + (idx - r * N))}${r}`;
}

export function coordToIdx(token: string): number {
  const t = token.trim().toUpperCase();
  if (t.length < 2 || t[0] < "A" || t[0] > "O") throw new Error(`非法坐标: '${token}'`);
  const c = t.charCodeAt(0) - 65;
  const r = Number(t.slice(1));
  if (!Number.isInteger(r) || r < 0 || r >= N) throw new Error(`坐标越界: '${token}'`);
  return r * N + c;
}

export function dumps(moves: number[], config: Config, result: string): string {
  const coords = moves.map(idxToCoord).join(" ");
  return `[AntiFive 1.0]\n[Config 15x15 white_restrict=${config.white_restrict_turns} ` +
    `loss_start=${config.loss_start_turns} mask_suicide=${config.mask_suicide ? 1 : 0}]\n` +
    `[Result ${result}]\n[Moves ${moves.length}]\nmoves:\n${coords}\n`;
}

export interface LoadedRecord { moves: number[]; config: Config; result: string; }

export function loads(text: string): LoadedRecord {
  let config: Config = { ...DEFAULT_CONFIG };
  let result = "未完";
  let inAnalysis = false;
  const moves: number[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("[")) {
      const body = line.replace(/^\[/, "").replace(/\]$/, "").trim();
      inAnalysis = body.startsWith("Analysis");
      if (body.startsWith("Config")) {
        for (const tok of body.slice(6).trim().split(/\s+/)) {
          const [k, v] = tok.split("=");
          if (k === "white_restrict") config.white_restrict_turns = Number(v);
          else if (k === "loss_start") config.loss_start_turns = Number(v);
          else if (k === "mask_suicide") config.mask_suicide = Number(v) !== 0;
        }
      } else if (body.startsWith("Result")) result = body.slice(6).trim();
      continue;
    }
    if (inAnalysis) continue;
    if (line.toLowerCase().startsWith("moves")) continue;
    for (const tok of line.split(/\s+/)) moves.push(coordToIdx(tok));
  }
  return { moves, config, result };
}
