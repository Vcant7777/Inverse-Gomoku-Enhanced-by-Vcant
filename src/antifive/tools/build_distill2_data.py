"""KataGo 蒸馏 v2 数据构建:规模化采样 + 两层标注(批量低访问 + 硬样本高访问)

相对 v1(`build_distill_data.py`)的升级:
- 采样密度:每局 step=2/cap=40/start=6(v1 为 step=3/cap=12/start=10);
- 对局来源:启发式自对弈 + KataGo vs 启发式(默认 48 局,压迫/败局局面)
  + `--replay-dir` 回放既有 `.afg` 棋谱(如 records/stage2_12s 的 0:96 受迫局);
- 两层标注:批量 `--visits`(默认 2000)标注全部采样点;凡"实际着法不在
  KataGo top-`--hard-rank` 或与首着胜率差 > `--hard-gap`"的硬样本,当场用
  `--hard-visits`(默认 20000)重标并标记 `tier=1`(每局至多
  `--hard-per-game` 个),训练端对 tier=1 加权;
- 样本额外记录 `played`(该局面实际落子,回放局为棋谱着法)。

产出 `checkpoints/distill2_pool.npz`:boards/players/pends/tcs/wts/game_ids/
sources/lmoves/lvisits/lwr/played/tier,供 `tools/train_distill2.py` 训练。
按"局"并行、断点续跑(分片 `checkpoints/distill2_parts/gXXXXX.npz`)。

用法(分块运行,到点自动退出并合并):
    python -m antifive.tools.build_distill2_data --games-selfplay 300 \
        --games-katago 48 --workers 4 --minutes 40
    python -m antifive.tools.build_distill2_data --replay-dir records/stage2_12s/games
    python -m antifive.tools.build_distill2_data ... --only katago   # 只补跑某段
    python -m antifive.tools.build_distill2_data --merge-only
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import struct
import time
from pathlib import Path

import numpy as np

from .. import record as record_mod
from ..heuristic import _TT, choose_heuristic_move
from ..reversegomoku import BLACK, GameConfig, ReverseGomoku
from .build_distill_data import (
    Engine, _engine_defaults, _parse_info, gtp_to_idx, idx_to_gtp,
)

MAX_CAND = 20


def _sample_ply(ply: int, start: int, step: int, n: int, cap: int) -> bool:
    return ply >= start and (ply - start) % step == 0 and n < cap


def _valid_cands(cands: list) -> list:
    """[(move_idx, visits, winrate)] 按访问降序,剔除 pass/resign。"""
    out = []
    for c in sorted(cands, key=lambda d: -int(d.get("visits", 0))):
        if "move" not in c:
            continue
        mi = gtp_to_idx(c["move"])
        if mi < 0:
            continue
        out.append((mi, int(c.get("visits", 0)), float(c.get("winrate", 0.5))))
    return out


def _is_hard(cands: list, played: int, rank: int, gap: float) -> bool:
    """实际着法是否为"硬样本":不在 top-rank 或与首着胜率差超阈值。"""
    if played < 0:
        return False
    valid = _valid_cands(cands)
    if not valid:
        return False
    for i, (m, _, w) in enumerate(valid):
        if m == played:
            return i >= rank or (valid[0][2] - w) > gap
    return True


def _pack_sample(g, wr_coord, cands, spec, played: int, tier: int) -> dict:
    lmoves = np.full(MAX_CAND, -1, dtype=np.int16)
    lvisits = np.zeros(MAX_CAND, dtype=np.int32)
    lwr = np.zeros(MAX_CAND, dtype=np.float32)
    n = 0
    for mi, v, w in _valid_cands(cands):
        if n >= MAX_CAND:
            break
        lmoves[n] = mi
        lvisits[n] = v
        lwr[n] = w
        n += 1
    return {
        "board": g.board.copy(),
        "player": np.int8(g.current_player),
        "pending": np.int16(g.pending),
        "tc": np.int16(g.turn_count),
        "wt": np.int16(g.white_turns),
        "game_id": np.int32(spec["idx"]),
        "source": np.int8(spec["source"]),
        "lmoves": lmoves,
        "lvisits": lvisits,
        "lwr": lwr,
        "played": np.int16(played),
        "tier": np.int8(tier),
    }


def _label_once(eng: Engine, color: str, spec: dict, played: int,
                hard_budget: list) -> tuple:
    """批量分析 + 硬样本当场重标;返回 (cands, tier)。

    `kata-genmove_analyze` 会在服务端实际落子:本函数返回后棋盘上**恰好多
    一个子**(硬样本路径已 undo 后重析),调用方必须再 undo 一次恢复。"""
    _, cands = eng.analyze(color)
    tier = 0
    if hard_budget[0] > 0 and spec["hard_visits"] > spec["visits"] \
            and _is_hard(cands, played, spec["hard_rank"], spec["hard_gap"]):
        eng.simple("undo")                   # 撤销批量分析落下的子
        _, cands = eng.analyze(color, visits=spec["hard_visits"])
        tier = 1
        hard_budget[0] -= 1
    return cands, tier


def _play_game(eng: Engine, spec: dict, cfg: GameConfig) -> list:
    """自对弈/KataGo 对局:先标注当前局面(如需)再落子;返回样本列表。"""
    kind = spec["kind"]
    rng = np.random.default_rng(spec["seed"])
    g = ReverseGomoku(cfg)
    tt = _TT()
    eng.simple("clear_board")
    samples = []
    heur_black = spec.get("heur_black", True)
    hard_budget = [spec["hard_per_game"]]
    while not g.game_over and g.move_count < spec["max_steps"]:
        color = "B" if g.current_player == BLACK else "W"
        is_katago_turn = (kind == "katago" and
                          (g.current_player == BLACK) == (not heur_black))
        sampled = _sample_ply(g.move_count, spec["start"], spec["step"],
                              len(samples), spec["cap"])
        if is_katago_turn:
            if sampled:
                cands, tier = _label_once(eng, color, spec, -1, hard_budget)
                eng.simple("undo")           # 若未走硬样本路径,analyze 的子仍在
            coord = eng.genmove(color, spec["play_visits"])
            mv = gtp_to_idx(coord)
            if sampled:
                samples.append(_pack_sample(g, coord, cands, spec, mv, tier))
            if mv < 0:
                break
        else:
            mv = choose_heuristic_move(
                g.board, g.current_player, g.pending, g.turn_count,
                g.white_turns, g.config, rng, depth=8,
                time_budget=spec["budget"], tt=tt, use_vcf=True)
            p = -1 if mv is None else int(mv)
            if sampled:
                cands, tier = _label_once(eng, color, spec, p, hard_budget)
                eng.simple("undo")
                samples.append(_pack_sample(g, None, cands, spec, p, tier))
            if mv is None:
                break
            # 启发式落子必须同步到 KataGo 棋盘,否则后续 genmove 会在缺子的
            # 局面上选点(返回已占位置 → 非法着法提前终局,0 样本)
            eng.simple(f"play {color} {idx_to_gtp(mv)}")
        if mv < 0 or not g.legal_mask()[mv]:
            break
        g.make_move(int(mv))
    return samples


def _replay_game(eng: Engine, spec: dict, cfg: GameConfig) -> list:
    """回放 .afg:逐步同步 GTP 棋盘,按 step 采样并标注;`played` 为棋谱着法。"""
    rec = record_mod.load_game(spec["path"])
    if not rec.moves:
        return []
    g = ReverseGomoku(rec.config)
    eng.simple("clear_board")
    samples = []
    hard_budget = [spec["hard_per_game"]]
    for ply, m in enumerate(rec.moves):
        if g.game_over or len(samples) >= spec["cap"]:
            break
        mask = g.legal_mask()
        mi = int(m)
        if mask.any() and not mask[mi]:
            break                            # 谱与当前规则不兼容(白禁推等)
        color = "B" if g.current_player == BLACK else "W"
        if _sample_ply(ply, spec["start"], spec["step"], len(samples),
                       spec["cap"]):
            cands, tier = _label_once(eng, color, spec, mi, hard_budget)
            eng.simple("undo")
            samples.append(_pack_sample(g, None, cands, spec, mi, tier))
        eng.simple(f"play {color} {idx_to_gtp(mi)}")
        g.make_move(mi)
    return samples


def _worker(wargs: dict) -> int:
    exe, weight, cfgpath = wargs["engine"], wargs["weight"], wargs["config"]
    cfg = GameConfig()
    done = 0
    eng = Engine(exe, weight, cfgpath, wargs["visits"])
    eng.simple("boardsize 15 15")
    try:
        for spec in wargs["specs"]:
            if time.time() > wargs["deadline"]:
                break
            path = Path(wargs["parts"]) / f"g{spec['idx']:05d}.npz"
            if path.exists():
                continue
            t0 = time.perf_counter()
            try:
                if spec["kind"] == "replay":
                    samples = _replay_game(eng, spec, cfg)
                else:
                    samples = _play_game(eng, spec, cfg)
            except Exception as e:           # 单局失败(坏棋谱等)不拖垮整批
                print(f"[w{wargs['wid']}] game {spec['idx']} 失败: {e}",
                      flush=True)
                eng.simple("clear_board")
                samples = []
            _save_part(path, samples)
            done += 1
            n_hard = sum(int(s["tier"]) for s in samples)
            print(f"[w{wargs['wid']}] game {spec['idx']} "
                  f"({spec['kind']}, {len(samples)} samples, "
                  f"hard {n_hard}, {time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        eng.close()
    return done


def _save_part(path: Path, samples: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        np.savez(path, n=np.array([0]))
        return
    np.savez(
        path,
        boards=np.stack([s["board"] for s in samples]).astype(np.int8),
        players=np.array([s["player"] for s in samples], dtype=np.int8),
        pends=np.array([s["pending"] for s in samples], dtype=np.int16),
        tcs=np.array([s["tc"] for s in samples], dtype=np.int16),
        wts=np.array([s["wt"] for s in samples], dtype=np.int16),
        game_ids=np.array([s["game_id"] for s in samples], dtype=np.int32),
        sources=np.array([s["source"] for s in samples], dtype=np.int8),
        lmoves=np.stack([s["lmoves"] for s in samples]).astype(np.int16),
        lvisits=np.stack([s["lvisits"] for s in samples]).astype(np.int32),
        lwr=np.stack([s["lwr"] for s in samples]).astype(np.float32),
        played=np.array([s["played"] for s in samples], dtype=np.int16),
        tier=np.array([s["tier"] for s in samples], dtype=np.int8),
    )


_KEYS = ("boards", "players", "pends", "tcs", "wts", "game_ids", "sources",
         "lmoves", "lvisits", "lwr", "played", "tier")


def _sample_key(arr: dict, i: int) -> bytes:
    b = arr["boards"][i].tobytes()
    meta = struct.pack("<Bhhh", int(arr["players"][i]) & 0xFF,
                       int(arr["pends"][i]), int(arr["tcs"][i]),
                       int(arr["wts"][i]))
    return b + meta


def _merge(parts: Path, out: Path) -> int:
    files = sorted(parts.glob("g*.npz"))
    arrays = {k: [] for k in _KEYS}
    for f in files:
        z = np.load(f)
        if "boards" not in z:
            continue
        for k in _KEYS:
            arrays[k].append(z[k])
    if not arrays["boards"]:
        print("暂无已完成对局")
        return 0
    merged = {k: np.concatenate(v) for k, v in arrays.items()}
    seen: dict = {}
    for i in range(len(merged["boards"])):
        key = _sample_key(merged, i)
        j = seen.get(key)
        if j is None:
            seen[key] = i
        elif int(merged["tier"][i]) > int(merged["tier"][j]):
            seen[key] = i                  # 同一局面硬样本优先
    keep = np.asarray(sorted(seen.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: merged[k][keep] for k in _KEYS})
    n_hard = int(np.count_nonzero(merged["tier"][keep] >= 1))
    print(f"合并 {len(files)} 个分片 → {out}:{len(keep)} 个局面"
          f"(硬样本 {n_hard})")
    return len(keep)


def _build_specs(args) -> list:
    specs = []
    idx = 0
    for i in range(args.games_selfplay):
        specs.append({"idx": idx, "kind": "selfplay", "source": 0,
                      "seed": 70000 + i * 17,
                      "budget": 0.5 if i % 2 == 0 else 2.0,
                      "start": args.sample_start, "step": args.sample_step,
                      "cap": args.sample_cap, "play_visits": args.play_visits,
                      "max_steps": args.max_steps,
                      "visits": args.visits, "hard_visits": args.hard_visits,
                      "hard_rank": args.hard_rank, "hard_gap": args.hard_gap,
                      "hard_per_game": args.hard_per_game})
        idx += 1
    for i in range(args.games_katago):
        specs.append({"idx": idx, "kind": "katago", "source": 1,
                      "seed": 90000 + i * 29,
                      "budget": 1.0 if i % 2 == 0 else 2.0,
                      "heur_black": i % 2 == 0,
                      "start": args.sample_start, "step": args.sample_step,
                      "cap": args.sample_cap, "play_visits": args.play_visits,
                      "max_steps": args.max_steps,
                      "visits": args.visits, "hard_visits": args.hard_visits,
                      "hard_rank": args.hard_rank, "hard_gap": args.hard_gap,
                      "hard_per_game": args.hard_per_game})
        idx += 1
    seen_paths = set()
    for d in args.replay_dir:
        for f in sorted(Path(d).glob("*.afg")):
            if f in seen_paths:
                continue
            seen_paths.add(f)
            specs.append({"idx": idx, "kind": "replay", "source": 2,
                          "path": str(f), "start": args.replay_start,
                          "step": args.replay_step, "cap": args.replay_cap,
                          "max_steps": args.max_steps,
                          "visits": args.visits,
                          "hard_visits": args.hard_visits,
                          "hard_rank": args.hard_rank,
                          "hard_gap": args.hard_gap,
                          "hard_per_game": args.hard_per_game})
            idx += 1
    return specs


def main() -> None:
    d = _engine_defaults()
    p = argparse.ArgumentParser(description="KataGo 蒸馏 v2 数据构建")
    p.add_argument("--engine", default=d["engine"])
    p.add_argument("--weight", default=d["weight"])
    p.add_argument("--config", default=d["config"])
    p.add_argument("--visits", type=int, default=2000, help="批量标注访问数")
    p.add_argument("--hard-visits", type=int, default=20000,
                   help="硬样本重标访问数(≤visits 时关闭两层标注)")
    p.add_argument("--hard-rank", type=int, default=5,
                   help="实际着法跌出 top-N 视为硬样本")
    p.add_argument("--hard-gap", type=float, default=0.15,
                   help="实际着法与首着胜率差阈值")
    p.add_argument("--hard-per-game", type=int, default=6)
    p.add_argument("--play-visits", type=int, default=2000,
                   help="KataGo 对局落子访问数")
    p.add_argument("--games-selfplay", type=int, default=200)
    p.add_argument("--games-katago", type=int, default=48)
    p.add_argument("--replay-dir", action="append", default=[],
                   help="回放 .afg 目录(可多次指定)")
    p.add_argument("--replay-start", type=int, default=6)
    p.add_argument("--replay-step", type=int, default=2)
    p.add_argument("--replay-cap", type=int, default=20,
                   help="每局回放最多采样数")
    p.add_argument("--sample-start", type=int, default=6)
    p.add_argument("--sample-step", type=int, default=2)
    p.add_argument("--sample-cap", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=450)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--minutes", type=float, default=15.0,
                   help="本次运行时间上限(到点后不再开新局)")
    p.add_argument("--only", default="",
                   help="只跑指定类别(逗号分隔 selfplay,katago,replay;空=全部;"
                        "编号保持不变,适配分段断点续跑)")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    parts = Path(args.out_dir) / "distill2_parts"
    pool = Path(args.out_dir) / "distill2_pool.npz"

    if args.merge_only:
        _merge(parts, pool)
        return
    if not (args.engine and args.weight and args.config):
        p.error("未找到 KataGo 路径:请用 --engine/--weight/--config 指定")

    specs = _build_specs(args)
    if args.only:
        wanted = {k.strip() for k in args.only.split(",") if k.strip()}
        unknown = wanted - {"selfplay", "katago", "replay"}
        if unknown:
            p.error(f"--only 含未知类别: {sorted(unknown)}")
        specs = [s for s in specs if s["kind"] in wanted]
    done = len(list(parts.glob("g*.npz")))
    print(f"总局数 {len(specs)},已完成 {done},开始并行标注"
          f"(workers={args.workers}, visits={args.visits}, "
          f"hard={args.hard_visits})", flush=True)
    deadline = time.time() + args.minutes * 60
    wargs = []
    for w in range(args.workers):
        wargs.append({"wid": w, "specs": specs[w::args.workers],
                      "deadline": deadline, "parts": str(parts),
                      "engine": args.engine, "weight": args.weight,
                      "config": args.config, "visits": args.visits})
    with mp.Pool(args.workers) as pool_mp:
        counts = pool_mp.map(_worker, wargs)
    print("本次完成对局:", sum(counts), flush=True)
    _merge(parts, pool)
    done = len(list(parts.glob("g*.npz")))
    print(f"进度 {done}/{len(specs)}", flush=True)


if __name__ == "__main__":
    main()
