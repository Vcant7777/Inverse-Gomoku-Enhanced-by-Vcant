"""KataGo 蒸馏 v2 训练:策略先验 MLP(+5 格模式)与价值校正 MLP

读取 `build_distill_data.py`/`build_distill2_data.py` 产出的蒸馏池
(默认 `checkpoints/distill_pool.npz`,若存在 `checkpoints/distill2_pool.npz`
则优先;后者带 `tier` 硬样本标记),按对局切分 train/val,训练:

- 先验:每候选 18 维标量特征 + 5 格 base-4 模式表(直线/斜线各 1024),
  listwise softmax 对齐 KataGo 候选访问分布;硬样本(tier=1)损失加权;
- 价值:25 维局面特征 → KataGo 根胜率价值 ((wr-0.5)*4) 回归。

训练集内做 D4 八重对称增广(局面/手中棋子/着法同步变换,先验特征按变换后
局面重算);验证集按对局切分且不增广,防止泄漏。

写出包内 `src/antifive/distill_data2.py`(纯数据模块,网页/Pyodide 可用)
与 `checkpoints/distill2_report.json`。权重可用环境变量
`ANTIFIVE_PRIOR2_W`/`ANTIFIVE_VALUE2_W` 在推理端临时缩放,便于 A/B。

用法:
    python -m antifive.tools.train_distill2
    python -m antifive.tools.train_distill2 --pool checkpoints/distill2_pool.npz \
        --val-ratio 0.15 --epochs 300 --value-hidden 64 --prior-hidden 48
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from antifive import heuristic as h                          # noqa: E402
from antifive.reversegomoku import (                         # noqa: E402
    BLACK, BOARD_SIZE, PLACE_SIZE, SYM_FLAT, GameConfig, ReverseGomoku,
)

P_PAT = 4 ** 5                        # 5 格模式表规模(与 distill2.PATTERN_CELLS 对齐)
HARD_WEIGHT = 3.0                     # tier=1 硬样本损失权重


def _load_pool(path: Path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _split_by_game(pool: dict, val_ratio: float, seed: int):
    games = np.unique(pool["game_ids"])
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_ratio))
    val_games = set(int(g) for g in games[:n_val])
    idx = np.arange(len(pool["boards"]))
    val = np.array([i for i in idx if int(pool["game_ids"][i]) in val_games])
    train = np.array([i for i in idx if int(pool["game_ids"][i]) not in val_games])
    return train, val


def _tier_weight(pool: dict, i: int) -> float:
    if "tier" not in pool:
        return 1.0
    return HARD_WEIGHT if int(pool["tier"][i]) >= 1 else 1.0


def _transform(board: np.ndarray, pending: int, moves: np.ndarray, t: int):
    """D4 对称变换 t:棋盘/手中棋子/着法同步变换(约定见 openbook.canonical)。"""
    if t == 0:
        return board, pending, moves
    tb = np.zeros(PLACE_SIZE, dtype=np.int8)
    tb[SYM_FLAT[t]] = board.reshape(-1)
    p2 = int(SYM_FLAT[t][pending]) if pending >= 0 else pending
    m2 = SYM_FLAT[t][np.asarray(moves, dtype=np.int64)]
    return tb.reshape(BOARD_SIZE, BOARD_SIZE), p2, m2


def _d1_dataset(pool: dict, indices, augment: bool) -> list:
    """先验样本列表 [(sc (M,18) float32, tk (M,8) uint16, y (M,) float32, w), ...]。"""
    cfg = GameConfig()
    out = []
    n_aug = 8 if augment else 1
    for i in indices:
        board = np.asarray(pool["boards"][i], dtype=np.int8)
        player = int(pool["players"][i])
        pend, tc, wt = (int(pool["pends"][i]), int(pool["tcs"][i]),
                        int(pool["wts"][i]))
        legal = np.nonzero(ReverseGomoku.legal_mask_for(
            board, player, pend, wt, cfg, tc))[0]
        if not legal.size:
            continue
        y = np.zeros(len(legal), dtype=np.float32)
        for j in range(pool["lmoves"].shape[1]):
            m = int(pool["lmoves"][i][j])
            if m < 0 or pool["lvisits"][i][j] <= 0:
                continue
            pos = int(np.searchsorted(legal, m))
            if pos < len(legal) and legal[pos] == m:
                y[pos] = float(pool["lvisits"][i][j])
        if y.sum() <= 0:
            continue
        y = y / y.sum()
        w = _tier_weight(pool, i)
        for t in range(n_aug):
            b2, p2, mv2 = _transform(board, pend, legal, t)
            sc, tk = h._prior2_features(b2, player, p2, tc, wt, cfg, mv2)
            out.append((sc.astype(np.float32), tk.astype(np.int64),
                        y, w))
    return out


def _d2_dataset(pool: dict, indices):
    """价值样本 (X (N,25) float32, Y (N,), W (N,))。D4 不变特征,不做增广。"""
    cfg = GameConfig()
    X, Y, W = [], [], []
    for i in indices:
        board = np.asarray(pool["boards"][i], dtype=np.int8)
        player = int(pool["players"][i])
        tc, wt = int(pool["tcs"][i]), int(pool["wts"][i])
        pend = int(pool["pends"][i])
        mb, mw, _, _, db, dw = h._board_features(board, wt, tc, cfg)
        danger_cnt = len(dw) if player == BLACK else len(db)
        danger_mine = len(db) if player == BLACK else len(dw)
        empty = int(np.count_nonzero(board == 0))
        adj_b, adj_w, cl_b, cl_w, ct_b, ct_w = h._positional_summary(board)
        f = h._value2_features(player, mb, mw, danger_cnt, danger_mine, empty,
                               adj_b, adj_w, cl_b, cl_w, ct_b, ct_w,
                               pend, tc, wt)
        wr = 0.5
        for j in range(pool["lmoves"].shape[1]):
            if pool["lvisits"][i][j] > 0:
                wr = float(pool["lwr"][i][j])
                break
        X.append(f)
        Y.append((wr - 0.5) * 4.0)
        W.append(_tier_weight(pool, i))
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32),
            np.asarray(W, dtype=np.float32))


def _std_arrays(feats: list) -> tuple:
    allf = np.concatenate([f[0] for f in feats], axis=0)
    mean = allf.mean(axis=0)
    std = allf.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def _train_value(Xtr, Ytr, Wtr, Xva, Yva, Wva, hidden, epochs, seed, device):
    import torch
    torch.manual_seed(seed)
    mean = Xtr.mean(axis=0).astype(np.float64)
    std = Xtr.std(axis=0).astype(np.float64)
    std[std < 1e-6] = 1.0
    dev = torch.device(device)
    xt = torch.tensor((Xtr - mean) / std, dtype=torch.float32, device=dev)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    wt = torch.tensor(Wtr, dtype=torch.float32, device=dev)
    xv = torch.tensor((Xva - mean) / std, dtype=torch.float32, device=dev)
    yv = torch.tensor(Yva, dtype=torch.float32, device=dev)
    model = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2 * wt).sum() / wt.sum()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(xv).squeeze(-1) - yv) ** 2).mean())
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xv).squeeze(-1).cpu().numpy()
    mse = float(np.mean((pred - Yva) ** 2))
    sign = float(np.mean(np.sign(pred) == np.sign(Yva)))
    W1 = model[0].weight.detach().cpu().numpy().astype(np.float64)
    B1 = model[0].bias.detach().cpu().numpy().astype(np.float64)
    W2 = model[2].weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    B2 = float(model[2].bias.detach().cpu().numpy()[0])
    return dict(mse=mse, sign=sign, W1=W1, B1=B1, W2=W2, B2=B2,
                mean=mean, std=std, epochs_ran=ep + 1)


def _train_prior(data_tr, data_va, hidden, epochs, seed, device, batch=64):
    import torch
    torch.manual_seed(seed)
    mean, std = _std_arrays(data_tr)
    dev = torch.device(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(h._PRIOR2_DIM, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1)).to(dev)
    w_pat_s = torch.nn.Parameter(torch.zeros(P_PAT, device=dev))
    w_pat_d = torch.nn.Parameter(torch.zeros(P_PAT, device=dev))
    opt = torch.optim.Adam(list(model.parameters()) + [w_pat_s, w_pat_d],
                           lr=3e-3, weight_decay=1e-5)
    m_ = torch.tensor(mean, dtype=torch.float32, device=dev)
    s_ = torch.tensor(std, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)

    def batch_tensors(items):
        mmax = max(len(sc) for sc, _, _, _ in items)
        B = len(items)
        X = np.zeros((B, mmax, h._PRIOR2_DIM), dtype=np.float32)
        TK = np.zeros((B, mmax, 8), dtype=np.int64)
        Y = np.zeros((B, mmax), dtype=np.float32)
        M = np.zeros((B, mmax), dtype=bool)
        W = np.zeros(B, dtype=np.float32)
        for b, (sc, tk, y, w) in enumerate(items):
            n = len(sc)
            X[b, :n] = sc
            TK[b, :n] = tk
            Y[b, :n] = y
            M[b, :n] = True
            W[b] = w
        return (torch.tensor(X, device=dev), torch.tensor(TK, device=dev),
                torch.tensor(Y, device=dev), torch.tensor(M, device=dev),
                torch.tensor(W, device=dev))

    def scores(X, TK, M):
        z = (X - m_) / s_
        hh = torch.relu(model[0](z))
        s = model[2](hh).squeeze(-1)
        s = s + w_pat_s[TK[..., 0:4]].sum(dim=-1)
        s = s + w_pat_d[TK[..., 4:8]].sum(dim=-1)
        return s.masked_fill(~M, -1e9)

    def evaluate(items):
        model.eval()
        top1 = top3 = 0
        nll = 0.0
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            X, TK, Y, M, _ = batch_tensors(chunk)
            with torch.no_grad():
                s = scores(X, TK, M)
                logp = torch.log_softmax(s, dim=-1)
                nll += float((-(Y * logp).sum(dim=-1)).sum())
                order = torch.argsort(s, dim=-1, descending=True)
                tgt = torch.argmax(Y, dim=-1)
                top1 += int((order[:, 0] == tgt).sum())
                top3 += int((order[:, :3] == tgt.unsqueeze(1)).any(dim=1).sum())
        n = max(len(items), 1)
        return top1 / n, top3 / n, nll / n

    best, best_state, bad = float("inf"), None, 0
    order = np.arange(len(data_tr))
    for ep in range(epochs):
        model.train()
        rng.shuffle(order)
        for i in range(0, len(order), batch):
            chunk = [data_tr[j] for j in order[i:i + batch]]
            X, TK, Y, M, W = batch_tensors(chunk)
            opt.zero_grad()
            logp = torch.log_softmax(scores(X, TK, M), dim=-1)
            loss = (-(Y * logp).sum(dim=-1) * W).sum() / W.sum().clamp(min=1e-6)
            loss.backward()
            opt.step()
        _, _, v = evaluate(data_va)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = ({k: t.detach().clone() for k, t in model.state_dict().items()},
                          w_pat_s.detach().clone(), w_pat_d.detach().clone())
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state[0])
    w_pat_s.data, w_pat_d.data = best_state[1], best_state[2]
    tr1, tr3, trn = evaluate(data_tr)
    va1, va3, van = evaluate(data_va)
    P_W1 = model[0].weight.detach().cpu().numpy().astype(np.float64)
    P_B1 = model[0].bias.detach().cpu().numpy().astype(np.float64)
    P_W2 = model[2].weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    P_B2 = float(model[2].bias.detach().cpu().numpy()[0])
    return dict(top1_train=tr1, top3_train=tr3, nll_train=trn,
                top1_val=va1, top3_val=va3, nll_val=van,
                W1=P_W1, B1=P_B1, W2=P_W2, B2=P_B2, mean=mean, std=std,
                W_STRAIGHT=w_pat_s.detach().cpu().numpy().astype(np.float64),
                W_DIAG=w_pat_d.detach().cpu().numpy().astype(np.float64),
                epochs_ran=ep + 1)


def _fmt_1d(a, fmt="{:.7g}"):
    return "[" + ", ".join(fmt.format(float(x)) for x in np.asarray(a).reshape(-1)) + "]"


def _fmt_2d(a):
    rows = ",\n    ".join(_fmt_1d(r) for r in np.asarray(a))
    return "np.array([\n    " + rows + "], dtype=np.float64)"


def _write_module(path: Path, val: dict, pri: dict, prior_w: float,
                  value_w: float) -> None:
    lines = [
        '"""KataGo 蒸馏权重 v2(由 tools/train_distill2.py 生成,勿手改)。',
        "",
        "先验:score = MLP18(scalars) + W_STRAIGHT/W_DIAG[5 格 base-4 模式码]",
        "价值:value = VALUE_W * MLP25(feats)(特征顺序见 heuristic._value2_features)",
        '"""',
        "",
        "import numpy as np",
        "",
        "SPEC = 2",
        f"VALUE_IN = {val['W1'].shape[1]}",
        f"VALUE_H = {val['W1'].shape[0]}",
        f"V_W1 = {_fmt_2d(val['W1'])}",
        f"V_B1 = np.array({_fmt_1d(val['B1'])}, dtype=np.float64)",
        f"V_W2 = np.array({_fmt_1d(val['W2'])}, dtype=np.float64)",
        f"V_B2 = {val['B2']!r}",
        f"V_MEAN = np.array({_fmt_1d(val['mean'])}, dtype=np.float64)",
        f"V_STD = np.array({_fmt_1d(val['std'])}, dtype=np.float64)",
        f"PRIOR_IN = {pri['W1'].shape[1]}",
        f"PRIOR_H = {pri['W1'].shape[0]}",
        f"P_W1 = {_fmt_2d(pri['W1'])}",
        f"P_B1 = np.array({_fmt_1d(pri['B1'])}, dtype=np.float64)",
        f"P_W2 = np.array({_fmt_1d(pri['W2'])}, dtype=np.float64)",
        f"P_B2 = {pri['B2']!r}",
        f"P_MEAN = np.array({_fmt_1d(pri['mean'])}, dtype=np.float64)",
        f"P_STD = np.array({_fmt_1d(pri['std'])}, dtype=np.float64)",
        f"W_STRAIGHT = np.array({_fmt_1d(pri['W_STRAIGHT'])}, dtype=np.float64)",
        f"W_DIAG = np.array({_fmt_1d(pri['W_DIAG'])}, dtype=np.float64)",
        f"PRIOR_W = {prior_w:.6g}",
        f"VALUE_W = {value_w:.6g}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    p = argparse.ArgumentParser(description="KataGo 蒸馏 v2 训练(先验 MLP + 价值 MLP)")
    p.add_argument("--pool", default="",
                   help="蒸馏池 npz(默认优选 checkpoints/distill2_pool.npz,"
                        "缺失回退 checkpoints/distill_pool.npz)")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "distill_data2.py"))
    p.add_argument("--report", default="checkpoints/distill2_report.json")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--prior-w", type=float, default=0.0)
    p.add_argument("--value-w", type=float, default=0.5,
                   help="v2 价值混合权重默认值(替换 v1 语义;12s 等时对拍 0.5 "
                        "优于 1.0:78.1% vs 70%)")
    p.add_argument("--value-hidden", type=int, default=64)
    p.add_argument("--prior-hidden", type=int, default=48)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--prior-epochs", type=int, default=120)
    p.add_argument("--augment", type=int, default=1, help="D4 增广 1/0")
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pool_path = Path(args.pool) if args.pool else None
    if pool_path is None:
        cand2 = Path("checkpoints/distill2_pool.npz")
        cand1 = Path("checkpoints/distill_pool.npz")
        pool_path = cand2 if cand2.exists() else cand1
    if not pool_path.exists():
        p.error(f"蒸馏池不存在: {pool_path}")
    pool = _load_pool(pool_path)
    n = len(pool["boards"])
    train_idx, val_idx = _split_by_game(pool, args.val_ratio, args.seed)
    print(f"池 {pool_path}:局面 {n}(train {len(train_idx)} / val {len(val_idx)})"
          f" | device={device} augment={'D4x8' if args.augment else 'off'}"
          f" | tier={'有' if 'tier' in pool else '无'}", flush=True)

    t0 = time.perf_counter()
    data_tr = _d1_dataset(pool, train_idx, bool(args.augment))
    data_va = _d1_dataset(pool, val_idx, False)
    print(f"D1 数据集:{len(data_tr)}/{len(data_va)}"
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    Xtr, Ytr, Wtr = _d2_dataset(pool, train_idx)
    Xva, Yva, Wva = _d2_dataset(pool, val_idx)

    val = _train_value(Xtr, Ytr, Wtr, Xva, Yva, Wva, args.value_hidden,
                       args.epochs, args.seed, device)
    print(f"D2 价值:val MSE={val['mse']:.4f} 符号准确率={val['sign']:.1%} "
          f"(epochs={val['epochs_ran']})", flush=True)
    pri = _train_prior(data_tr, data_va, args.prior_hidden,
                       args.prior_epochs, args.seed, device)
    print(f"D1 先验:top1 train {pri['top1_train']:.1%} val {pri['top1_val']:.1%}"
          f" | top3 val {pri['top3_val']:.1%} | NLL val {pri['nll_val']:.3f}"
          f"(epochs={pri['epochs_ran']})", flush=True)

    _write_module(Path(args.out), val, pri, args.prior_w, args.value_w)
    report = {
        "pool": str(pool_path), "positions": int(n),
        "train": int(len(train_idx)), "val": int(len(val_idx)),
        "augment": bool(args.augment), "hidden_value": args.value_hidden,
        "hidden_prior": args.prior_hidden, "seed": args.seed,
        "d2_mse_val": val["mse"], "d2_sign_val": val["sign"],
        "d1_top1_train": pri["top1_train"], "d1_top1_val": pri["top1_val"],
        "d1_top3_train": pri["top3_train"], "d1_top3_val": pri["top3_val"],
        "d1_nll_train": pri["nll_train"], "d1_nll_val": pri["nll_val"],
        "prior_w": args.prior_w, "value_w": args.value_w,
        "elapsed_s": time.perf_counter() - t0,
    }
    Path(args.report).write_text(json.dumps(report, indent=1),
                                 encoding="utf-8")
    print(f"已写出 {args.out} 与 {args.report}"
          f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
