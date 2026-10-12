"""match_katago 对拍工具纯函数测试:坐标转换、开局生成、统计汇总。

不依赖 KataGo 可执行文件与 pygame/torch;真实对拍由命令行工具承担。
"""

from __future__ import annotations

import pytest

from antifive.reversegomoku import BOARD_SIZE, GameConfig, ReverseGomoku
from antifive.tools.match_katago import (
    _parse_top_move, canonical_board_key, elo_diff, generate_openings,
    gtp_to_idx, idx_to_gtp, summarize, wilson_ci,
)


def test_gtp_roundtrip():
    for idx in range(BOARD_SIZE * BOARD_SIZE):
        assert gtp_to_idx(idx_to_gtp(idx)) == idx


def test_gtp_known_corners():
    assert idx_to_gtp(0) == "A15"
    assert idx_to_gtp(14) == "P15"         # 列 A-P 跳过 I,第 15 列为 P
    assert idx_to_gtp(210) == "A1"
    assert idx_to_gtp(224) == "P1"
    assert idx_to_gtp(8) == "J15"
    assert gtp_to_idx("H8") == 7 * 15 + 7


def test_parse_top_move():
    line = ("info move D4 visits 156 winrate 0.51 order 0 pv D4 "
            "info move E5 visits 120 winrate 0.5 order 1 pv E5")
    assert _parse_top_move(line) == ("D4", 156)
    assert _parse_top_move("info visits 3") is None
    assert _parse_top_move("") is None
    # 必杀证明搜索:visits 异常放大,应丢弃
    mate = "info move B11 visits 2366499 utility 0.999375 winrate 0.999688"
    assert _parse_top_move(mate) == ("B11", None)


def test_wilson_ci():
    lo, hi = wilson_ci(0.5, 100)
    assert lo == pytest.approx(0.404, abs=0.01)
    assert hi == pytest.approx(0.596, abs=0.01)
    assert wilson_ci(1.0, 10)[1] == 1.0
    assert wilson_ci(0.5, 0) == (0.0, 0.0)


def test_elo_diff():
    assert elo_diff(0.5) == pytest.approx(0.0, abs=1e-9)
    assert elo_diff(0.75) == pytest.approx(190.85, abs=0.1)
    assert elo_diff(0.0) is None
    assert elo_diff(1.0) is None


def test_generate_openings():
    cfg = GameConfig()
    ops = generate_openings(4, 4, seed=7, config=cfg)
    assert len(ops) == 4
    assert all(len(o) == 4 for o in ops)
    keys = set()
    for o in ops:
        g = ReverseGomoku(cfg)
        for m in o:
            g.make_move(m)
        assert not g.game_over and g.pending < 0 and g.white_turns >= 1
        keys.add(canonical_board_key(g.board))
    assert len(keys) == 4                  # D4 结构互异
    assert generate_openings(4, 4, seed=7, config=cfg) == ops


def _rec(**kw):
    base = {"status": "ok", "winner": "heuristic", "kata_color": "B",
            "steps": 60, "kata_resigned": False, "kata_avg_seconds": 12.0,
            "heur_avg_seconds": 12.0, "kata_visits_per_sec": 1000.0}
    base.update(kw)
    return base


def test_summarize():
    records = [
        _rec(winner="heuristic", kata_color="B"),
        _rec(winner="kata", kata_color="W"),
        _rec(winner="draw", kata_color="B"),
        _rec(winner="kata", kata_color="W", kata_resigned=True),
    ]
    s = summarize(records, [{"game_id": 9, "status": "aborted"}], {"x": 1}, 10.0)
    assert s["games_ok"] == 4
    assert s["games_aborted"] == 1
    assert s["heuristic_wins"] == 1
    assert s["kata_wins"] == 2
    assert s["draws"] == 1
    assert s["heuristic_score"] == pytest.approx(0.375)
    assert s["kata_resigns"] == 1
    assert s["by_kata_color"]["B"]["games"] == 2
    assert s["score_ci95"][0] < s["heuristic_score"] < s["score_ci95"][1]
    assert s["elo_diff"] is not None
