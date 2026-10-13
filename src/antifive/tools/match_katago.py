"""KataGo(等时)vs 启发式 AI 对拍:配对开局、可断点续跑、带置信区间

用途:按"配对开局 + 黑白互换"的规程,让启发式引擎(默认高级:VCF + 开局库 +
KataGo 蒸馏价值校正 + data/params.json 调优参数,深度 8)与 KataGo(GTP,
默认最新自训练权重)在相同每步墙钟时限下对弈,输出棋力对比统计与逐局记录。

默认口径:
- 双方每步 12 秒(启发式 --heur-budget;KataGo --katago-time);KataGo 的
  maxVisits 被覆盖为大值,确保按时间停(GTP 配置自带的 8000 visits 会提前
  触发,使"等时"名不副实);
- 规则 white_restrict=2、loss_start=8(判负检查优化)、mask_suicide=True
  (与 GUI/自对弈一致:启发式不生成自杀着;KataGo 着法按引擎规则允许自杀,
  走到即负)。注意:不能对启发式用 mask_suicide=False——该配置下"占领即
  成己方五连"的自杀着不会被候选过滤,引擎会走出必败着(冒烟实测);
- 96 局 = 48 个随机开局(seed 生成,D4 去重)× 黑白各一局。

输出目录(默认 records/katago_vs_advanced_<时间戳>/):
  run_meta.json     运行参数、引擎/权重路径、启发式功能开关快照
  openings.json     配对开局(复现与续跑依据)
  games/g####.afg   逐局棋谱(可载入 GUI 复盘)
  games/g####.json  逐局结果 + 逐步耗时/节点/visits
  results.jsonl     每局一行汇总(含 aborted)
  summary.json      胜率、Wilson 95% CI、近似 Elo、耗时与访问统计

用法:
    python -m antifive.tools.match_katago --games 96 --workers 2
    python -m antifive.tools.match_katago --games 2 --katago-time 1 \
        --heur-budget 1 --heur-depth 4 --out records/smoke   # 冒烟

中断后对同一 --out 重跑即续跑(已完成的对局自动跳过;aborted 局会重试)。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .. import record as record_mod
from ..heuristic_versions import normalize
from ..openbook import canonical
from ..paths import config_dir, data_dir, resolve
from ..reversegomoku import (
    BLACK, BOARD_SIZE, GameConfig, ReverseGomoku, WHITE,
)

LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
KATA_OPT = "K"
HEUR_OPT = "H"

# 非默认可能改变引擎行为的启发式环境变量:启动时打印,防止"最强配置"被
# 残留环境意外关掉(如 ANTIFIVE_NO_DISTILL=1)。
_ENV_KEYS = ("ANTIFIVE_NO_PARAMS", "ANTIFIVE_NO_BOOK", "ANTIFIVE_NO_DISTILL",
             "ANTIFIVE_PRIOR_W", "ANTIFIVE_VALUE_W", "ANTIFIVE_VCF",
             "ANTIFIVE_VCF_DEPTH", "ANTIFIVE_VCF_BUDGET", "ANTIFIVE_TT_BITS")


# --------------------------------------------------------------------------
# GTP 坐标(与 gui.py 一致:列 A-O 跳过 I,行号顶行=15)
# --------------------------------------------------------------------------
def idx_to_gtp(idx: int) -> str:
    r, c = divmod(int(idx), BOARD_SIZE)
    return f"{LETTERS[c]}{BOARD_SIZE - r}"


def gtp_to_idx(coord: str) -> int:
    coord = coord.strip().upper()
    c = LETTERS.index(coord[0])
    r = BOARD_SIZE - int(coord[1:])
    return r * BOARD_SIZE + c


def _parse_top_move(line: str):
    """kata-genmove_analyze 的 info 行 → (首选着法坐标, 该着法 visits);解析失败 None。

    fork 在必杀/必败证明搜索(utility 接近 ±1)中 visits 计数会异常放大(实测
    安置步达百万级),并非正常搜索量:此时 visits 置 None,避免污染访问统计。"""
    toks = line.split()
    try:
        i = toks.index("move")
        j = toks.index("visits")
        coord, visits = toks[i + 1].upper(), int(toks[j + 1])
    except (ValueError, IndexError):
        return None
    try:
        utility = float(toks[toks.index("utility") + 1])
    except (ValueError, IndexError):
        utility = None
    if utility is not None and abs(utility) >= 0.999:
        visits = None
    return coord, visits


# --------------------------------------------------------------------------
# KataGo 引擎/权重/配置默认值(engines.local.json 优先,auto_latest 解析)
# --------------------------------------------------------------------------
def _load_engine_entries() -> list:
    for name in ("engines.local.json", "engines.json"):
        p = config_dir() / name
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
        except Exception:
            continue
    return []


def _probe_file(candidates: tuple) -> str:
    from ..paths import roots
    for root in roots():
        for rel in candidates:
            p = root / rel
            if p.is_file():
                return str(p)
    return ""


def _probe_weight() -> str:
    from ..paths import roots
    for root in roots():
        for rel in ("models/katago", "katago/models", "katago"):
            d = root / rel
            if not d.is_dir():
                continue
            cands = sorted(d.glob("*.bin"), key=lambda f: f.stat().st_mtime,
                           reverse=True)
            trained = [f for f in cands if "init" not in f.name.lower()]
            picked = trained or cands
            if picked:
                return str(picked[0])
    return ""


def _kata_defaults() -> dict:
    """读取 KataGo 配置(engines.local.json 覆盖 engines.json,与 GUI 同逻辑)。"""
    def _resolve(path: str) -> str:
        if path and not os.path.isabs(path):
            return resolve(path)
        return path

    found = {"engine": "", "config": "", "weight": ""}
    for e in _load_engine_entries():
        if e.get("kind") == "heuristic":
            continue
        wp = _resolve(e.get("weight_path", ""))
        if e.get("auto_latest"):
            d = os.path.dirname(wp)
            try:
                cands = sorted(
                    (os.path.join(d, f) for f in os.listdir(d)
                     if f.startswith("antifive15-") and f.endswith(".bin")
                     and not f.endswith("antifive15-init.bin")),
                    key=os.path.getmtime, reverse=True)
                if cands:
                    wp = cands[0]
            except OSError:
                pass
        found = {"engine": _resolve(e.get("engine_path", "")),
                 "config": _resolve(e.get("config_path", "")),
                 "weight": wp}
        break
    if not (found["engine"] and os.path.exists(found["engine"])):
        found["engine"] = _probe_file(("katago/katago.exe", "katago.exe")) \
            or found["engine"]
    if not (found["config"] and os.path.exists(found["config"])):
        found["config"] = _probe_file(("config/gtp_test.cfg", "gtp_test.cfg")) \
            or found["config"]
    if not (found["weight"] and os.path.exists(found["weight"])):
        found["weight"] = _probe_weight() or found["weight"]
    return found


def _heuristic_status(version: str, use_vcf: bool, depth: int,
                      budget: float) -> dict:
    from .. import distill, openbook, threat_search
    v = normalize(version)
    return {
        "version": v,
        "depth": depth,
        "time_budget": budget,
        "vcf": bool(use_vcf and v != "beginner"),
        "vcf_depth": int(threat_search.VCF_MAX_DEPTH),
        "vcf_budget_s": float(threat_search.VCF_TIME_CAP),
        "book": bool(openbook._BOOK_ENABLED),
        "book_entries": len(openbook.OPENING_BOOK),
        "distill_enabled": bool(distill.ENABLED),
        "distill_value_w": float(getattr(distill, "VALUE_W", 0.0) or 0.0),
        "distill_prior_w": float(getattr(distill, "PRIOR_W", 0.0) or 0.0),
        "params_file": str(data_dir() / "params.json"),
        "env_overrides": {k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
    }


# --------------------------------------------------------------------------
# 配对开局生成
# --------------------------------------------------------------------------
def canonical_board_key(board: np.ndarray) -> bytes:
    """棋盘的 D4 八重对称规范化字节(用于开局结构去重)。"""
    return canonical(np.asarray(board, dtype=np.int8).reshape(-1))[0]


def generate_openings(n: int, plies: int, seed: int,
                      config: GameConfig) -> list:
    """生成 n 个 plies 步随机开局(双方均已走过,D4 结构互异)。

    构造时用默认 mask_suicide(前 plies 步远在判负窗口内,掩码无差异);
    要求结束时无待安置、非终局、白方已完成至少 1 回合。"""
    rng = np.random.default_rng(seed)
    seen, openings, guard = set(), [], 0
    while len(openings) < n:
        guard += 1
        if guard > n * 200:
            raise RuntimeError(f"开局生成失败:仅得到 {len(openings)}/{n} 个"
                               f"(plies={plies} 约束过紧?)")
        g = ReverseGomoku(GameConfig(white_restrict_turns=config.white_restrict_turns,
                                     loss_start_turns=config.loss_start_turns))
        for _ in range(plies):
            legal = np.nonzero(g.legal_mask())[0]
            if not legal.size or g.game_over:
                break
            g.make_move(int(rng.choice(legal)))
        if (g.game_over or g.pending >= 0 or g.move_count != plies
                or g.white_turns < 1):
            continue
        key = canonical_board_key(g.board)
        if key in seen:
            continue
        seen.add(key)
        openings.append(g.record())
    return openings


# --------------------------------------------------------------------------
# KataGo GTP 子进程(带读线程与超时;kata-genmove_analyze 记录访问数)
# --------------------------------------------------------------------------
class KataGoGTP:
    """GTP 封装:每局一进程,避免 fork 的跨局状态问题。

    关键语义(与 GUI `_KataGoEngine` 相同):fork 的 genmove 已在引擎内部
    自落盘,启发式着法必须逐"步" play 给引擎;catpure(占领)步用行棋方颜色。
    """

    def __init__(self, engine: str, model: str, cfg: str, max_time: float,
                 max_visits: int, log_dir: str):
        overrides = [f"maxTime={max_time}",
                     f"maxVisits={int(max_visits)}",
                     "ponderingEnabled=false"]
        if log_dir:
            overrides.append(f"logDir={log_dir}")
        self.proc = subprocess.Popen(
            [engine, "gtp", "-config", cfg, "-model", model,
             "-override-config", ",".join(overrides)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            encoding="utf-8", errors="replace")
        self._lines: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._timeout = max(60.0, max_time * 2 + 120.0)
        self._analyze_ok = True
        self._cmd(f"boardsize {BOARD_SIZE} {BOARD_SIZE}")
        self._cmd("clear_board")

    def _reader(self):
        try:
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                self._lines.put(line.rstrip("\n"))
        finally:
            self._lines.put(None)

    def _wait(self, deadline: float) -> str:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError("katago 响应超时")
            try:
                line = self._lines.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("katago 进程退出")
            return line

    def _cmd(self, line: str) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError("katago 进程已退出")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + self._timeout
        while True:
            out = self._wait(deadline)
            if out == "=" or out.startswith("= "):
                return out[2:] if out.startswith("= ") else ""
            if out == "?" or out.startswith("? "):
                raise RuntimeError(
                    f"GTP 错误 [{line}]: {out[2:].strip() if out.startswith('? ') else ''}")

    def play(self, color: str, coord: str):
        self._cmd(f"play {color} {coord}")

    def _genmove_analyze(self, color: str):
        """返回 (坐标, 选中着法的访问数);访问数解析失败为 None。"""
        self.proc.stdin.write(f"kata-genmove_analyze {color} 50\n")
        self.proc.stdin.flush()
        deadline = time.time() + self._timeout
        top, visits = None, None
        while True:
            out = self._wait(deadline)
            if out.startswith("play "):
                coord = out[len("play "):].strip()
                if top is not None and coord.upper() != top:
                    visits = None           # 最终着法与 info 首选不一致时不可信
                return coord, visits
            if out == "?" or out.startswith("? "):
                raise RuntimeError(f"GTP 错误 [kata-genmove_analyze]: "
                                   f"{out[2:].strip() if out.startswith('? ') else ''}")
            if out.startswith("info "):
                parsed = _parse_top_move(out)
                if parsed:
                    top, visits = parsed

    def genmove(self, color: str):
        if self._analyze_ok:
            try:
                return self._genmove_analyze(color)
            except RuntimeError:
                self._analyze_ok = False
        return self._cmd(f"genmove {color}"), None

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            time.sleep(0.05)
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------
# 单局对弈(worker 进程)
# --------------------------------------------------------------------------
def _game_task(job: dict) -> dict:
    from .. import heuristic_versions as hv
    from ..heuristic import _TT

    gid = job["game_id"]
    opening = job["opening"]
    kata_color = job["kata_color"]
    cfg = GameConfig(loss_start_turns=job["loss_start"],
                     white_restrict_turns=job["white_restrict"],
                     mask_suicide=True)
    kg = None
    try:
        g = ReverseGomoku(cfg)
        for m in opening:
            g.make_move(m)

        kg = KataGoGTP(job["engine"], job["model"], job["engine_cfg"],
                       job["kata_time"], job["kata_visits"],
                       str(Path(job["out_dir"]) / "gtp_logs"))
        for idx, player, *_ in g.history:
            kg.play("B" if player == BLACK else "W", idx_to_gtp(idx))

        rng = np.random.default_rng(job["seed"])
        tt = _TT()
        step_log, resigned = [], False
        while not g.game_over and g.move_count < job["max_steps"]:
            color = "B" if g.current_player == BLACK else "W"
            if g.current_player == kata_color:
                t0 = time.perf_counter()
                coord, visits = kg.genmove(color)
                dt = time.perf_counter() - t0
                if coord.lower() in ("resign", "pass"):
                    g.game_over = True
                    g.resigned = g.current_player
                    resigned = True
                    break
                a = gtp_to_idx(coord)
                g.make_move(a, allow_suicide=True)
                step_log.append([g.move_count, KATA_OPT, round(dt, 3), visits])
            else:
                nodes = [0]
                t0 = time.perf_counter()
                a = hv.choose(job["heur_version"], g.board, g.current_player,
                              g.pending, g.turn_count, g.white_turns, g.config,
                              rng, depth=job["heur_depth"],
                              time_budget=job["heur_budget"],
                              progress_cb=lambda n: nodes.__setitem__(0, n),
                              tt=tt, use_vcf=job["use_vcf"])
                dt = time.perf_counter() - t0
                if a is None:
                    g.game_over = True
                    g.loser = g.current_player
                    break
                kg.play(color, idx_to_gtp(a))
                g.make_move(a)
                step_log.append([g.move_count, HEUR_OPT, round(dt, 3),
                                 int(nodes[0])])
        if not g.game_over:
            g.game_over = True
            g.is_draw = True

        result = record_mod.game_result(g)
        if resigned:
            winner = "heuristic" if g.resigned == kata_color else "kata"
        elif g.is_draw:
            winner = "draw"
        else:
            winner = "heuristic" if g.loser == kata_color else "kata"
        moves = g.record()
        if g.pending >= 0 and moves:
            moves = moves[:-1]           # 未完成安置的占领步不记谱(与 GUI 一致)

        # 记录自检:重放落谱着法必须复现终局(防御同步/记录幻影,成本毫秒级)
        if not resigned and not g.is_draw:
            chk = ReverseGomoku(cfg)
            for m in moves:
                chk.make_move(m, allow_suicide=True)
            if not ((chk.game_over and chk.loser == g.loser)
                    or ((not chk.game_over) and chk.is_stuck())):
                raise RuntimeError(
                    f"记录自检失败:重放未复现终局(loser={g.loser}, "
                    f"pending={g.pending}, steps={len(moves)})")

        games_dir = Path(job["out_dir"]) / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        stem = f"g{gid:04d}"
        with open(games_dir / f"{stem}.afg", "w", encoding="utf-8") as f:
            f.write(record_mod.dumps(moves, g.config, result))

        kata_steps = [s for s in step_log if s[1] == KATA_OPT]
        heur_steps = [s for s in step_log if s[1] == HEUR_OPT]
        kata_sec = sum(s[2] for s in kata_steps)
        heur_sec = sum(s[2] for s in heur_steps)
        kata_visits = [s[3] for s in kata_steps if s[3]]
        rec = {
            "game_id": gid,
            "status": "ok",
            "opening_id": job["opening_id"],
            "kata_color": "B" if kata_color == BLACK else "W",
            "result": result,
            "winner": winner,
            "kata_resigned": resigned,
            "steps": g.move_count,
            "turns": g.turn_count,
            "kata_seconds": round(kata_sec, 2),
            "heur_seconds": round(heur_sec, 2),
            "kata_avg_seconds": round(kata_sec / max(len(kata_steps), 1), 3),
            "heur_avg_seconds": round(heur_sec / max(len(heur_steps), 1), 3),
            "kata_visits_avg": round(sum(kata_visits) / len(kata_visits), 1)
                if kata_visits else None,
            "kata_visits_per_sec": round(sum(kata_visits) / kata_sec, 1)
                if kata_visits and kata_sec > 0 else None,
            "step_log": step_log,
        }
        with open(games_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        return rec
    except Exception as e:                    # noqa: BLE001 - 上报给主进程
        return {"game_id": gid, "status": "aborted",
                "error": f"{type(e).__name__}: {e}"}
    finally:
        if kg is not None:
            kg.close()


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def wilson_ci(score: float, n: int, z: float = 1.96) -> tuple:
    """比分的 Wilson 95% 置信区间(和棋按 0.5 计入,方差近似)。"""
    if n <= 0:
        return 0.0, 0.0
    denom = 1.0 + z * z / n
    center = (score + z * z / (2 * n)) / denom
    half = z * math.sqrt(score * (1.0 - score) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def elo_diff(score: float):
    """比分(启发式视角,0.5=同档)→ 近似 Elo 差;0/1 返回 None(±∞)。"""
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def summarize(records: list, aborted: list, meta: dict, wall_seconds: float) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    n = len(ok)
    hw = sum(1 for r in ok if r["winner"] == "heuristic")
    kw = sum(1 for r in ok if r["winner"] == "kata")
    dr = n - hw - kw
    score = (hw + 0.5 * dr) / n if n else 0.0
    lo, hi = wilson_ci(score, n)
    by_color = {}
    for color in ("B", "W"):
        sub = [r for r in ok if r["kata_color"] == color]
        sub_h = sum(1 for r in sub if r["winner"] == "heuristic")
        sub_d = sum(1 for r in sub if r["winner"] == "draw")
        by_color[color] = {"games": len(sub), "heur_wins": sub_h,
                           "draws": sub_d, "kata_wins": len(sub) - sub_h - sub_d}

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    visits_rates = [r["kata_visits_per_sec"] for r in ok
                    if r.get("kata_visits_per_sec")]
    summary = {
        "games_ok": n,
        "games_aborted": len(aborted),
        "heuristic_wins": hw,
        "kata_wins": kw,
        "draws": dr,
        "heuristic_score": round(score, 4),
        "score_ci95": [round(lo, 4), round(hi, 4)],
        "elo_diff": (round(e, 1) if (e := elo_diff(score)) is not None else None),
        "elo_ci95": [round(elo_diff(lo), 1) if lo > 0 else None,
                     round(elo_diff(hi), 1) if hi < 1 else None],
        "by_kata_color": by_color,
        "kata_resigns": sum(1 for r in ok if r.get("kata_resigned")),
        "avg_steps": _avg([r["steps"] for r in ok]),
        "kata_avg_seconds_per_step": _avg([r["kata_avg_seconds"] for r in ok]),
        "heur_avg_seconds_per_step": _avg([r["heur_avg_seconds"] for r in ok]),
        "kata_visits_per_sec_avg": round(sum(visits_rates) / len(visits_rates), 1)
            if visits_rates else None,
        "wall_seconds": round(wall_seconds, 1),
        "meta": meta,
    }
    return summary


def _print_summary(s: dict):
    print("\n===== 对拍结果 =====")
    print(f"有效对局 {s['games_ok']} | 启发式胜 {s['heuristic_wins']} | "
          f"KataGo 胜 {s['kata_wins']} | 和棋 {s['draws']} | "
          f"放弃 {s['games_aborted']}")
    print(f"启发式比分 {s['heuristic_score']:.1%} "
          f"(95% CI {s['score_ci95'][0]:.1%}~{s['score_ci95'][1]:.1%})")
    e = s["elo_diff"]
    elo_ci = s["elo_ci95"]
    print(f"近似 Elo 差: {'%.0f' % e if e is not None else '±∞'}"
          f"(95% CI {elo_ci[0] if elo_ci[0] is not None else '-∞'} ~ "
          f"{elo_ci[1] if elo_ci[1] is not None else '+∞'})")
    print(f"平均局长 {s['avg_steps']} 步 | KataGo 认输 {s['kata_resigns']} | "
          f"平均步时 K {s['kata_avg_seconds_per_step']}s / "
          f"H {s['heur_avg_seconds_per_step']}s | "
          f"KataGo 访问 {s['kata_visits_per_sec_avg']} visits/s | "
          f"用时 {s['wall_seconds']:.0f}s")
    print(f"按 KataGo 执色: 黑 {s['by_kata_color']['B']} | 白 {s['by_kata_color']['W']}")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    defaults = _kata_defaults()
    p = argparse.ArgumentParser(description="KataGo(等时)vs 启发式 AI 对拍")
    p.add_argument("--engine", default=defaults["engine"], help="katago 可执行文件")
    p.add_argument("--model", default=defaults["weight"], help="KataGo 权重 .bin")
    p.add_argument("--engine-cfg", default=defaults["config"], help="GTP 配置")
    p.add_argument("--games", type=int, default=96,
                   help="总局数(偶数;= 开局数 × 2 色)")
    p.add_argument("--opening-plies", type=int, default=4,
                   help="配对开局的步数(双方均已走过,避开开局库)")
    p.add_argument("--katago-time", type=float, default=12.0,
                   help="KataGo 每步墙钟上限(秒)")
    p.add_argument("--katago-visits", type=int, default=100_000_000,
                   help="KataGo visits 上限(默认极大值,保证按时间停)")
    p.add_argument("--heur-budget", type=float, default=12.0,
                   help="启发式每步限时(秒)")
    p.add_argument("--heur-depth", type=int, default=8)
    p.add_argument("--heur-version", default="advanced",
                   help="启发式版本(advanced/mid/beginner/hell)")
    p.add_argument("--no-vcf", action="store_true", help="关闭启发式 VCF")
    p.add_argument("--max-steps", type=int, default=450)
    p.add_argument("--white-restrict", type=int, default=2)
    p.add_argument("--loss-start", type=int, default=8)
    p.add_argument("--workers", type=int, default=2,
                   help="并行对局进程数(等时口径下 GPU 争用会降低 KataGo "
                        "visits/s;要求绝对强度用 1)")
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--out", default="",
                   help="输出目录(默认 records/katago_vs_advanced_<时间戳>)")
    args = p.parse_args()

    for label, path in (("katago 可执行文件", args.engine),
                        ("KataGo 权重", args.model),
                        ("GTP 配置", args.engine_cfg)):
        if not (path and os.path.exists(path)):
            p.error(f"{label}不存在: {path!r}(请用命令行参数或 "
                    f"config/engines.local.json 指定)")
    if args.games < 2 or args.games % 2:
        p.error("--games 必须为不小于 2 的偶数")

    out_dir = Path(args.out) if args.out else Path(
        "records") / f"katago_vs_advanced_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "games").mkdir(exist_ok=True)
    (out_dir / "gtp_logs").mkdir(exist_ok=True)

    n_openings = args.games // 2
    cfgs = GameConfig(loss_start_turns=args.loss_start,
                      white_restrict_turns=args.white_restrict)
    openings_path = out_dir / "openings.json"
    if openings_path.exists():
        with open(openings_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("seed") != args.seed or len(saved.get("openings", [])) != n_openings:
            p.error(f"{openings_path} 与当前参数不符(seed/games 变化?)"
                    f":请换 --out 或使用相同参数续跑")
        openings = saved["openings"]
        print(f"续跑:{openings_path} 已有 {len(openings)} 个开局")
    else:
        t0 = time.perf_counter()
        openings = generate_openings(n_openings, args.opening_plies,
                                     args.seed, cfgs)
        with open(openings_path, "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "plies": args.opening_plies,
                       "openings": openings}, f, ensure_ascii=False)
        print(f"已生成 {len(openings)} 个配对开局"
              f"(plies={args.opening_plies}, {time.perf_counter() - t0:.1f}s)")

    use_vcf = not args.no_vcf
    hstatus = _heuristic_status(args.heur_version, use_vcf,
                                args.heur_depth, args.heur_budget)
    meta = {
        "args": vars(args),
        "engine": args.engine, "model": args.model,
        "engine_cfg": args.engine_cfg,
        "heuristic": hstatus,
        "kata_override": f"maxTime={args.katago_time},"
                         f"maxVisits={args.katago_visits},ponderingEnabled=false",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"输出目录: {out_dir}")
    print(f"KataGo: {os.path.basename(args.model)} | {args.engine}")
    print(f"启发式: {hstatus['version']} depth={hstatus['depth']} "
          f"budget={hstatus['time_budget']}s vcf={hstatus['vcf']} "
          f"book={hstatus['book']}({hstatus['book_entries']}) "
          f"distill_value_w={hstatus['distill_value_w']} "
          f"distill_prior_w={hstatus['distill_prior_w']}")
    if hstatus["env_overrides"]:
        print(f"注意:检测到启发式环境变量覆盖 {hstatus['env_overrides']}")

    jobs, done = [], []
    for i, opening in enumerate(openings, 1):
        for color in (BLACK, WHITE):
            gid = 2 * i - (0 if color == BLACK else 1)
            jf = out_dir / "games" / f"g{gid:04d}.json"
            if jf.exists():
                try:
                    with open(jf, encoding="utf-8") as f:
                        rec = json.load(f)
                    if rec.get("status") == "ok":
                        done.append(rec)   # 含认输局:均为有效结果,续跑不重打
                        continue
                except Exception:
                    pass
            jobs.append({
                "game_id": gid, "opening_id": i, "opening": opening,
                "kata_color": color, "seed": args.seed + gid * 7919,
                "engine": args.engine, "model": args.model,
                "engine_cfg": args.engine_cfg, "out_dir": str(out_dir),
                "heur_version": args.heur_version, "use_vcf": use_vcf,
                "heur_depth": args.heur_depth, "heur_budget": args.heur_budget,
                "kata_time": args.katago_time, "kata_visits": args.katago_visits,
                "max_steps": args.max_steps, "loss_start": args.loss_start,
                "white_restrict": args.white_restrict,
            })

    print(f"共 {args.games} 局:已完成 {len(done)} | 待对弈 {len(jobs)}"
          f"(workers={args.workers})")
    t_start = time.perf_counter()
    records, aborted = list(done), []
    counter = [len(done)]

    def handle(rec):
        counter[0] += 1
        tag = f"[{counter[0]}/{args.games}]"
        if rec.get("status") != "ok":
            aborted.append(rec)
            print(f"{tag} g{rec['game_id']:04d} 中止: {rec.get('error')}",
                  flush=True)
            return
        records.append(rec)
        w = rec["winner"]
        label = {"heuristic": "启发式胜", "kata": "KataGo 胜",
                 "draw": "和棋"}[w]
        if rec.get("kata_resigned"):
            label += "(认输)"
        print(f"{tag} g{rec['game_id']:04d} 开局#{rec['opening_id']:02d} "
              f"KataGo={rec['kata_color']} → {rec['result']}({label}) "
              f"{rec['steps']}步 {rec['kata_seconds'] + rec['heur_seconds']:.0f}s",
              flush=True)

    try:
        if args.workers > 1 and jobs:
            with mp.Pool(args.workers) as pool:
                for rec in pool.imap_unordered(_game_task, jobs):
                    handle(rec)
        else:
            for job in jobs:
                handle(_game_task(job))
    except KeyboardInterrupt:
        print("\n收到中断:已完成的局已落盘,再次运行同一 --out 可续跑")

    result_path = out_dir / "results.jsonl"
    with open(result_path, "w", encoding="utf-8") as f:
        for rec in records + aborted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = summarize(records, aborted, meta, time.perf_counter() - t_start)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    _print_summary(summary)
    print(f"结果: {result_path} | {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
