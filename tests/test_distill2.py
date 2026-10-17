"""KataGo 蒸馏 v2(distill2)回归:特征维度/模式编码/回退与集成开关。

不锁定训练质量指标(随数据变化),只锁定可验证的接口语义:
- 5 格模式 token 的形状/取值域/确定性;
- v2 特征维度与蒸馏数据模块对齐,且前 14/20 列与 v1 完全一致;
- 数据缺失/维度不符/关闭开关时全部退化为零,不改变引擎行为;
- 开关打开时 `_prior_map`/`evaluate` 确实走 distill2 分支。
"""

from __future__ import annotations

import numpy as np
import pytest

from antifive import distill2
from antifive import heuristic as h
from antifive.reversegomoku import BLACK, GameConfig, ReverseGomoku

CFG = GameConfig(loss_start_turns=8, white_restrict_turns=2, mask_suicide=True)


def _position(n_moves: int = 8) -> ReverseGomoku:
    g = ReverseGomoku(CFG)
    for m in (112, 113, 111, 114, 97, 127, 98, 128, 96, 126, 82, 142)[:n_moves]:
        if not g.game_over and g.legal_mask()[m]:
            g.make_move(m)
    return g


def _features(g: ReverseGomoku):
    mb, mw, _, _, db, dw = h._board_features(g.board, g.white_turns,
                                             g.turn_count, g.config)
    danger_cnt = len(dw) if g.current_player == BLACK else len(db)
    danger_mine = len(db) if g.current_player == BLACK else len(dw)
    empty = int(np.count_nonzero(g.board == 0))
    adj_b, adj_w, cl_b, cl_w, ct_b, ct_w = h._positional_summary(g.board)
    return (mb, mw, danger_cnt, danger_mine, empty, adj_b, adj_w, cl_b, cl_w,
            ct_b, ct_w)


def test_pattern_tokens5_shape_and_range():
    g = _position()
    legal = np.nonzero(g.legal_mask())[0][:9]
    tk = distill2.pattern_tokens5(g.board, g.current_player, legal)
    assert tk.shape == (len(legal), 8)
    assert tk.dtype == np.uint16
    assert int(tk.min()) >= 0
    assert int(tk.max()) < 4 ** 5
    tk2 = distill2.pattern_tokens5(g.board, g.current_player, legal)
    assert np.array_equal(tk, tk2)


def test_prior2_features_dims_and_v1_prefix():
    g = _position()
    legal = np.nonzero(g.legal_mask())[0]
    sc2, tk = h._prior2_features(g.board, g.current_player, g.pending,
                                 g.turn_count, g.white_turns, g.config, legal)
    assert sc2.shape == (len(legal), h._PRIOR2_DIM)
    assert tk.shape == (len(legal), 8)
    sc1, _ = h._prior_features(g.board, g.current_player, g.pending,
                               g.turn_count, g.white_turns, g.config, legal)
    assert np.allclose(sc2[:, :h._PRIOR_DIM], sc1)
    assert np.all((sc2[:, 14] == 0.0) | (sc2[:, 14] == 1.0))
    if distill2.HAS_PRIOR:
        assert h._PRIOR2_DIM == distill2.PRIOR_IN


def test_value2_features_dim_and_v1_prefix():
    g = _position()
    args = _features(g)
    (mb, mw, danger_cnt, danger_mine, empty, adj_b, adj_w, cl_b, cl_w,
     ct_b, ct_w) = args
    f2 = h._value2_features(g.current_player, mb, mw, danger_cnt, danger_mine,
                            empty, adj_b, adj_w, cl_b, cl_w, ct_b, ct_w,
                            g.pending, g.turn_count, g.white_turns)
    f1 = h._value_features(g.current_player, mb, mw, danger_cnt, danger_mine,
                           empty, adj_b, adj_w, cl_b, cl_w, ct_b, ct_w)
    assert f2.shape == (h._VALUE2_DIM,)
    assert np.allclose(f2[:len(f1)], f1)
    assert 0.0 <= f2[21] <= 1.0 and 0.0 <= f2[22] <= 1.0
    assert 0.0 <= f2[23] <= 1.0 and 0.0 <= f2[24] <= 1.0
    if distill2.HAS_VALUE:
        assert h._VALUE2_DIM == distill2.VALUE_IN
        assert np.isfinite(distill2.value_score(f2))


def test_distill2_graceful_fallback(monkeypatch):
    g = _position()
    legal = np.nonzero(g.legal_mask())[0][:5]
    sc, tk = h._prior2_features(g.board, g.current_player, g.pending,
                                g.turn_count, g.white_turns, g.config, legal)
    monkeypatch.setattr(distill2, "HAS_PRIOR", False)
    assert np.allclose(distill2.prior_scores(sc, tk), 0.0)
    monkeypatch.setattr(distill2, "HAS_VALUE", False)
    assert distill2.value_score(np.zeros(h._VALUE2_DIM)) == 0.0
    monkeypatch.setattr(distill2, "ENABLED", False)
    assert np.allclose(distill2.prior_scores(sc, tk), 0.0)


def test_prior_map_uses_distill2_when_active(monkeypatch):
    g = _position()
    legal = np.nonzero(g.legal_mask())[0]
    monkeypatch.setattr(h, "_ACTIVE2", True)
    monkeypatch.setattr(distill2, "HAS_PRIOR", True)
    monkeypatch.setattr(distill2, "prior_scores",
                        lambda sc, tk: np.ones(len(sc)))
    pm = h._prior_map(g.board, g.current_player, g.pending, g.turn_count,
                      g.white_turns, g.config, legal)
    assert np.allclose(pm[legal], 1.0)
    assert np.allclose(np.delete(pm, legal), 0.0)


def test_evaluate_replaces_v1_value_when_v2_active(monkeypatch):
    """v2 激活时替换 v1(而非叠加):evaluate = v1 版基础 - v1 项 + v2 项。"""
    from antifive import distill

    g = _position()
    (mb, mw, danger_cnt, danger_mine, empty, adj_b, adj_w, cl_b, cl_w,
     ct_b, ct_w) = _features(g)
    v1_part = distill.value_score(h._value_features(
        g.current_player, mb, mw, danger_cnt, danger_mine, empty,
        adj_b, adj_w, cl_b, cl_w, ct_b, ct_w))
    base = h.evaluate(g.board, g.current_player, g.pending, g.turn_count,
                      g.white_turns, g.config)
    monkeypatch.setattr(h, "_ACTIVE2", True)
    monkeypatch.setattr(distill2, "HAS_VALUE", True)
    monkeypatch.setattr(distill2, "value_score", lambda feats: 0.5)
    got = h.evaluate(g.board, g.current_player, g.pending, g.turn_count,
                     g.white_turns, g.config)
    assert got == pytest.approx(base - v1_part + 0.5)


def test_distill2_default_off():
    assert h._ACTIVE2 == (h.USE_DISTILL2 and distill2.ENABLED)


def test_distill_data2_arrays_consistent():
    if not distill2.HAS_VALUE and not distill2.HAS_PRIOR:
        pytest.skip("distill_data2 未生成(仅训练后存在)")
    if distill2.HAS_VALUE:
        assert distill2.V_W1.shape == (distill2.VALUE_H, distill2.VALUE_IN)
        assert distill2.V_MEAN.shape == (distill2.VALUE_IN,)
        assert distill2.V_STD.shape == (distill2.VALUE_IN,)
    if distill2.HAS_PRIOR:
        assert distill2.P_W1.shape == (distill2.PRIOR_H, distill2.PRIOR_IN)
        assert distill2.W_STRAIGHT.shape == (1024,)
        assert distill2.W_DIAG.shape == (1024,)
