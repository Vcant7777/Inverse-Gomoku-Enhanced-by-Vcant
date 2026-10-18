# 网页版 TypeScript 引擎

网页版已从 Pyodide + numpy + Python 搜索迁移为 TypeScript 原生引擎，运行在 Web Worker 中。
构建产物约 90KB，不再下载 Python 运行时、numpy 或 wheel。

## 目录

| 文件 | 职责 |
| --- | --- |
| `web/src/engine/rules.ts` | 15x15 棋盘、占领/安置、白棋禁推、自杀筛、判负、增量 Zobrist、合法着法 |
| `web/src/engine/eval.ts` | 线级材料 + 位置摘要；移植 Python 评估，并支持蒸馏价值校正 |
| `web/src/engine/tactics.ts` | 安置杀、占领+安置两步杀、杀对 route、跳形延伸 |
| `web/src/engine/vcf.ts` | 多层强制杀链（VCF），移植 `threat_search.py` 的攻防证明与预算 |
| `web/src/engine/profiles.ts` | 五档引擎：初级/中级/高级/大师/传奇，VCF/开局库/蒸馏能力 |
| `web/src/engine/distill.ts` | KataGo 蒸馏 v1 线性 / v2 MLP 价值校正 |
| `web/src/engine/search.ts` | 迭代加深 alpha-beta、置换表、根/内部候选裁剪、根分析 |
| `web/src/engine/openbook.ts` | KataGo 20k 首手/白应手表（0/1 子） |
| `web/src/engine/api.ts` | 会话命令、`state()` 字段、`.afg` 导入导出、AI/提示/射线 |
| `web/src/worker.ts` | Worker 协议：init/call → ready/status/progress/result/fatal |

## 一致性

- 规则与合法着法：600 个随机状态 + 40 局随机对局与 Python `reversegomoku` 对拍，0 差异。
- 状态/命令：同手顺的 `state()`（board/player/pending/turn/legal/kinds/numbers/danger）与 Python `web_api` 对拍一致。
- 评估：25 个随机局面在无蒸馏/v1/v2 三种模式下与 Python `heuristic.evaluate` 逐值一致，最大误差 < 4e-7。
- 战术：80 个随机局面的 `kill_pairs` 与 Python `tactics` 完全一致；30 个局面 VCF 深度 1–3 与 Python `find_forced_kill` 完全一致。
- 档位：`beginner` 无 VCF/开局库/蒸馏；`mid` 无蒸馏；`advanced` v1；`legend` v2；`hell` 默认无 v2。
- 搜索：深度 1–3 与无置换表暴力 negamax 分值一致；根与内部候选均按 Python `_candidates` 移植（空位/占领各 k、杀着/防守路线、diversity、history/killer 补充），内部搜索加入 PVS、TT 边界、killer、强制扩展与 `_jump_through` 送杀判定；20 局面深度 4 与 Python 同着法 17/20。
- 提示：`hint` 走根候选全窗口分析，返回 `pos_v` 与按分值排序的前 5 着，形状与 Python `analysis_moves` 一致。

## 性能

Chrome（headless，固定局面）实测：

| 模式 | 节点 | 墙钟 | 节点/秒 |
| --- | ---: | ---: | ---: |
| 无预算、候选裁剪 | 0.1–1 万 | < 0.1s | 10–100 万 |
| 10s 预算（含 VCF/蒸馏/PVS） | ~62 万 | 10.0s | ~6.2 万 |

旧 Pyodide 版约 2.3k 节点/秒；单节点成本提升约 20–40 倍，首屏资源从约 17MB 降到约 90KB。

## 固定节点对照（旧网页 15s）

旧网页 20 局面实测 15s 平均约 2.84 万节点（21,696–33,082），平均墙钟 13.5s。
把 TS 每步硬限制到对应旧节点数后，平均墙钟约 0.78s，达到深度 6–7，与旧版同着法 12/20。
固定节点对局（旧 Pyodide 15s vs TS 28,422 节点）两局 **1–1**，两局均为白方获胜；
样本很小，只能说明未观察到明显棋力差，不能据此判定强弱。

## 命令

```bash
cd web
npm run build   # tsc --noEmit && vite build
npm test        # Node 规则/API/战术/档位/棋谱测试
npm run smoke   # 浏览器冒烟测试（需 Chrome/Edge）
npm run bench   # 浏览器搜索基准
```
