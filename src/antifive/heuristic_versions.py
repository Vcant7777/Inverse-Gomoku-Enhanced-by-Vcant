"""启发式引擎版本调度(桌面 GUI / 网页共用)

四档(名字与 GUI 一致):
- beginner(初级):`antifive.tools.heuristic_v2` 冻结经典版(无 VCF/开局库/蒸馏);
- mid(中级):`antifive.tools.heuristic_v5` 冻结快照(搜索一致性修复 + VCF v5.1
  深杀 + KataGo 开局库 + 防守 VCF,无蒸馏);
- advanced(高级):`antifive.tools.heuristic_v6` 冻结快照(中级 + KataGo 蒸馏价值
  校正,即此前 `antifive.heuristic` 的行为基线);
- hell(地狱):当前强化版 `antifive.heuristic`(在高级基础上增加防守强化:
  深化 VCF(d12/24 万节点/≤1.8s)、严格时限预算管理与自适应防守预留、
  防守校验全候选、时限驱动的动态加深;P1/P2 代码保留但默认关闭)。

统一入口,避免 GUI / web_api 各自处理多套函数签名差异。
兼容旧值:"old"→初级、"new"→高级。
"""

from __future__ import annotations

VERSION_BEGINNER = "beginner"
VERSION_MID = "mid"
VERSION_ADVANCED = "advanced"
VERSION_HELL = "hell"
VERSION_CHOICES = (VERSION_BEGINNER, VERSION_MID, VERSION_ADVANCED,
                   VERSION_HELL)
VERSION_LABELS = {VERSION_BEGINNER: "初级", VERSION_MID: "中级",
                  VERSION_ADVANCED: "高级", VERSION_HELL: "地狱"}

_ALIASES = {
    "old": VERSION_BEGINNER,
    "new": VERSION_ADVANCED,
    "beginner": VERSION_BEGINNER,
    "mid": VERSION_MID,
    "advanced": VERSION_ADVANCED,
    "hell": VERSION_HELL,
    "地狱": VERSION_HELL,
}

_OLD_MODULE = None
_MID_MODULE = None
_ADV_MODULE = None


def label(version: str) -> str:
    return VERSION_LABELS.get(normalize(version), VERSION_LABELS[VERSION_ADVANCED])


def normalize(version: str | None) -> str:
    return _ALIASES.get(str(version), VERSION_ADVANCED)


def supports_vcf(version: str) -> bool:
    """初级(v2 冻结版)没有 VCF;中级/高级/地狱均支持。"""
    return normalize(version) != VERSION_BEGINNER


def _old():
    global _OLD_MODULE
    if _OLD_MODULE is None:
        from .tools import heuristic_v2
        _OLD_MODULE = heuristic_v2
    return _OLD_MODULE


def _mid():
    global _MID_MODULE
    if _MID_MODULE is None:
        from .tools import heuristic_v5
        _MID_MODULE = heuristic_v5
    return _MID_MODULE


def _adv():
    global _ADV_MODULE
    if _ADV_MODULE is None:
        from .tools import heuristic_v6
        _ADV_MODULE = heuristic_v6
    return _ADV_MODULE


def choose(version: str, board, player, pending, turn_count, white_turns,
           config, rng, depth=None, time_budget=None, progress_cb=None,
           tt=None, use_vcf: bool | None = None):
    """统一着法选择:按版本分发;中级/高级/地狱支持 tt/use_vcf。"""
    v = normalize(version)
    if v == VERSION_BEGINNER:
        return _old().choose_heuristic_move(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, progress_cb=progress_cb)
    if v == VERSION_MID:
        return _mid().choose_heuristic_move(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, progress_cb=progress_cb,
            tt=tt, use_vcf=use_vcf)
    if v == VERSION_ADVANCED:
        return _adv().choose_heuristic_move(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, progress_cb=progress_cb,
            tt=tt, use_vcf=use_vcf)
    from .heuristic import choose_heuristic_move
    return choose_heuristic_move(
        board, player, pending, turn_count, white_turns, config, rng,
        depth=depth, time_budget=time_budget, progress_cb=progress_cb,
        tt=tt, use_vcf=use_vcf)


def analysis(version: str, board, player, pending, turn_count, white_turns,
             config, rng, depth=None, time_budget=None, k=8, progress_cb=None,
             tt=None, use_vcf: bool | None = None):
    """统一局面分析(初级无 tt/use_vcf 参数,自动降级)。"""
    v = normalize(version)
    if v == VERSION_BEGINNER:
        return _old().analysis_moves(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, k=k, progress_cb=progress_cb)
    if v == VERSION_MID:
        return _mid().analysis_moves(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, k=k, progress_cb=progress_cb,
            tt=tt, use_vcf=use_vcf)
    if v == VERSION_ADVANCED:
        return _adv().analysis_moves(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, k=k, progress_cb=progress_cb,
            tt=tt, use_vcf=use_vcf)
    from .heuristic import analysis_moves
    return analysis_moves(
        board, player, pending, turn_count, white_turns, config, rng,
        depth=depth, time_budget=time_budget, k=k, progress_cb=progress_cb,
        tt=tt, use_vcf=use_vcf)
