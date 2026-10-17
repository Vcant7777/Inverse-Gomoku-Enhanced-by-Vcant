"""KataGo 蒸馏 v2 推理(numpy):候选策略先验 + 局面价值校正(小 MLP)。

与 v1(`distill.py` + `distill_data.py`)平行、互不影响:冻结快照
(`tools/heuristic_v5.py`/`heuristic_v6.py`)继续走 v1,避免新模型污染
既有回归基线;大师档(`antifive.heuristic`)在 `ANTIFIVE_DISTILL2=1` 时启用
v2(**价值项替换 v1,而非叠加**;叠加配置等时对拍净负),传奇快照
(`tools/heuristic_v7.py`)默认常开(`ANTIFIVE_DISTILL2=0` 可关)。

v2 相对 v1 的升级:
- 特征:先验 18 维(候选类型/占领可达安置/杀格重叠),价值 25 维(回合
  状态、手中棋子、双方安全空位);模式窗口 4→5 格(base-4 码 ≤1023,
  直线/斜线各一张 1024 项权重表);
- 模型:标量部分由线性升级为单隐层 MLP(先验/价值各一个),离线训练端
  `tools/train_distill2.py` 导出包内 `distill_data2.py`(纯数据,网页可用)。

权重数据缺失/损坏/维度不符时全部退化为零,引擎行为与未启用 v2 时一致。

环境变量(均可选):
- `ANTIFIVE_DISTILL2=1|0`      在 `antifive.heuristic` 中启用/禁用 v2(默认关);
- `ANTIFIVE_NO_DISTILL2=1`     本模块整体禁用(即使快照内常开);
- `ANTIFIVE_PRIOR2_W` / `ANTIFIVE_VALUE2_W`  运行期缩放混合权重(A/B 用)。
"""

from __future__ import annotations

import os

import numpy as np

from .reversegomoku import BOARD_SIZE, EMPTY, PLACE_SIZE

_SMALL = float(1e-9)

try:                                     # 模型缺失时静默降级(回退到未启用)
    from .distill_data2 import (         # type: ignore
        SPEC, VALUE_IN, VALUE_H, V_W1, V_B1, V_W2, V_B2, V_MEAN, V_STD,
        PRIOR_IN, PRIOR_H, P_W1, P_B1, P_W2, P_B2, P_MEAN, P_STD,
        W_STRAIGHT, W_DIAG, PRIOR_W, VALUE_W,
    )
    has_value = (
        getattr(V_W1, "shape", ()) == (VALUE_H, VALUE_IN)
        and np.asarray(V_B1).shape == (VALUE_H,)
        and np.asarray(V_W2).shape == (VALUE_H,)
        and np.asarray(V_MEAN).shape == (VALUE_IN,)
        and np.asarray(V_STD).shape == (VALUE_IN,)
    )
    has_prior = (
        getattr(P_W1, "shape", ()) == (PRIOR_H, PRIOR_IN)
        and np.asarray(P_B1).shape == (PRIOR_H,)
        and np.asarray(P_W2).shape == (PRIOR_H,)
        and np.asarray(P_MEAN).shape == (PRIOR_IN,)
        and np.asarray(P_STD).shape == (PRIOR_IN,)
        and np.asarray(W_STRAIGHT).shape == (1024,)
        and np.asarray(W_DIAG).shape == (1024,)
    )
    HAS_VALUE = bool(has_value)
    HAS_PRIOR = bool(has_prior)
except Exception:                        # pragma: no cover - 数据缺失/损坏
    SPEC = 0
    VALUE_IN = VALUE_H = PRIOR_IN = PRIOR_H = 0
    V_W1 = V_B1 = V_W2 = V_B2 = V_MEAN = V_STD = None
    P_W1 = P_B1 = P_W2 = P_B2 = P_MEAN = P_STD = None
    W_STRAIGHT = W_DIAG = None
    PRIOR_W = 0.0
    VALUE_W = 0.0
    HAS_PRIOR = False
    HAS_VALUE = False

_DISABLED = os.environ.get("ANTIFIVE_NO_DISTILL2", "").strip().lower() \
    in ("1", "true", "yes", "on")
ENABLED = (HAS_PRIOR or HAS_VALUE) and not _DISABLED

# 标准化折叠进首层权重:z=(x-mean)/std 的除法与减均值在加载期一次性折进
# (W1/std, B1-(W1/std)@mean),推理路径只做一次矩阵乘(叶节点高频调用)。
if HAS_VALUE:
    _V_W1S = np.asarray(V_W1, dtype=np.float64) / np.asarray(V_STD)[None, :]
    _V_B1S = np.asarray(V_B1, dtype=np.float64) - _V_W1S @ np.asarray(V_MEAN)
else:
    _V_W1S = _V_B1S = None
if HAS_PRIOR:
    _P_W1S = np.asarray(P_W1, dtype=np.float64) / np.asarray(P_STD)[None, :]
    _P_B1S = np.asarray(P_B1, dtype=np.float64) - _P_W1S @ np.asarray(P_MEAN)
else:
    _P_W1S = _P_B1S = None


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return default if v is None or v == "" else float(v)
    except ValueError:
        return default


# 混合权重可用环境变量临时缩放,便于 A/B 选值(不改动权重文件)
PRIOR_W = _env_float("ANTIFIVE_PRIOR2_W", PRIOR_W)
VALUE_W = _env_float("ANTIFIVE_VALUE2_W", VALUE_W)

PATTERN_CELLS = 5                        # 模式窗口:每侧 5 格(base-4 码 0..1023)
_PAT = 4 ** PATTERN_CELLS
_PAD = PATTERN_CELLS
_STATE_OFF = 3                           # 棋盘外
_AXES = ((0, 1), (1, 0), (1, 1), (1, -1))


def pattern_tokens5(board: np.ndarray, player: int, moves) -> np.ndarray:
    """(M,8) uint16:每步 4 轴 × 两侧的 5 格 base-4 编码(0 空/1 己/2 对方/3 界外),
    列序与 v1 一致(0-3 直线轴,4-7 斜线轴)。"""
    flat = np.asarray(board, dtype=np.int8).reshape(-1)
    rel = np.zeros(PLACE_SIZE, dtype=np.int8)
    rel[flat == player] = 1
    rel[(flat != player) & (flat != EMPTY)] = 2
    size = BOARD_SIZE + 2 * _PAD
    # int32:5 格码的位权 k 最高 4**4=256,int8 会触发 numpy2 标量越界(NEP50)
    pad = np.full((size, size), _STATE_OFF, dtype=np.int32)
    pad[_PAD:_PAD + BOARD_SIZE, _PAD:_PAD + BOARD_SIZE] = \
        rel.reshape(BOARD_SIZE, BOARD_SIZE)
    mv = np.asarray(moves, dtype=np.int64)
    r, c = np.divmod(mv, BOARD_SIZE)
    r = r + _PAD
    c = c + _PAD
    out = np.empty((len(mv), 8), dtype=np.uint16)
    for a, (dr, dc) in enumerate(_AXES):
        cp = np.zeros(len(mv), dtype=np.int32)
        cn = np.zeros(len(mv), dtype=np.int32)
        for i in range(PATTERN_CELLS):
            k = 4 ** i
            cp += k * pad[r + (i + 1) * dr, c + (i + 1) * dc]
            cn += k * pad[r - (i + 1) * dr, c - (i + 1) * dc]
        out[:, 2 * a] = cp
        out[:, 2 * a + 1] = cn
    return out


def value_score(feats) -> float:
    """局面价值校正(未启用时返回 0.0)。feats 顺序见 heuristic._value2_features。

    返回已乘混合权重 VALUE_W(训练端在离线回归输出上标定)。"""
    if not ENABLED or not HAS_VALUE:
        return 0.0
    x = np.asarray(feats)
    if x.shape != (VALUE_IN,):
        return 0.0                       # 维度不符(旧调用方)→ 静默降级
    h = np.maximum(_V_W1S @ x + _V_B1S, 0.0)
    return VALUE_W * float(h @ V_W2 + V_B2)


def prior_scores(scalars: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """策略先验 MLP + 5 格模式表线性项(未启用时返回全 0)。"""
    sc = np.asarray(scalars, dtype=np.float64)
    tk = np.asarray(tokens)
    if not ENABLED or not HAS_PRIOR or sc.ndim != 2 \
            or sc.shape[1] != PRIOR_IN or tk.shape != (len(sc), 8):
        return np.zeros(len(sc), dtype=np.float64)
    h = np.maximum(sc @ _P_W1S.T + _P_B1S, 0.0)
    s = h @ P_W2 + P_B2
    s = s + W_STRAIGHT[tk[:, 0:4].astype(np.int64)].sum(axis=1)
    s = s + W_DIAG[tk[:, 4:8].astype(np.int64)].sum(axis=1)
    return s
