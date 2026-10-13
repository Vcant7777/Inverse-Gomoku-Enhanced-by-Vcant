# AI 引擎与 KataGo 配置

本项目包含四类对局引擎,均通过对战接口 `choose(game) -> 着法 idx` 与 GUI 解耦。

| 引擎 | 依赖 | 强度 | 说明 |
|---|---|---|---|
| 随机先验 MCTS | numpy | 低 | 无需任何额外文件,均匀先验搜索,默认兜底 |
| 自研神经网络 MCTS | numpy + torch + 权重 | **不足(训练失败)** | 最多稍弱于启发式,仅供研究 |
| 启发式 AI | numpy | 中高 | 迭代加深 alpha-beta + 全局评估 + 置换表,深度 1–8 |
| KataGo(逆五) | 外部 katago.exe + `.bin` 权重 | 高 | 自训练逆五规则权重,推荐实用引擎 |

## 自研神经网络(失败尝试)

权重位于 `models/warm_model.pth`、`models/best_model.pth`。训练总体失败,
强度不足,详见 `docs/training-log.md`。加载方式:

```bash
python -m antifive.gui --model models/best_model.pth
```

未安装 torch 时神经网络引擎自动禁用(设置面板中不显示该项),其余功能不受影响。

## 启发式 AI

内置算法,无需额外文件;GUI 中可选深度 1–8(默认 8 + AI 读秒 12 秒,四档通用;
地狱档在时限内动态加深,不受 8 层硬上限),并可切换四档引擎版本与 VCF 强制
杀链开关(初级无 VCF;默认高级 + VCF 开):

| 版本 | 实现 | 组成 |
|---|---|---|
| 初级 | `tools/heuristic_v2.py` 冻结经典版 | 老算法,无 VCF/开局库/蒸馏 |
| 中级 | `tools/heuristic_v5.py` 冻结快照 | 搜索一致性修复 + VCF v5.1 + 开局库 + 防守 VCF |
| 高级(默认) | `tools/heuristic_v6.py` 冻结快照 | 中级 + KataGo 蒸馏价值校正(旧 `antifive.heuristic` 行为) |
| 地狱 | `antifive.heuristic` | 高级 + 防守强化:深化 VCF(d12/24 万节点/≤1.8s)、P1 预防性防守候选、P0 防守校验全候选与预算对等 + 时限动态加深、P2 潜伏威胁评估 |

网页端难度第 4 档「地狱」= 深度 8 + 强制地狱引擎(困难之上);网页/桌面默认
仍为困难/高级,不默认地狱。地狱档默认与高级共用 `data/params.json`;若存在
`data/params_hell.json` 则优先读取(参数隔离机制保留,默认不提供该文件)。
A/B 开关:`ANTIFIVE_PREGIFT=1` / `ANTIFIVE_GIFT3=1`(P1/P2,默认关闭)、
`ANTIFIVE_NO_DYNAMIC=1`、`ANTIFIVE_NO_LMR=1`、`ANTIFIVE_NO_ROOT_PVS=1`
(实验特性)、VCF 参数 `ANTIFIVE_HELL_VCF_DEPTH/NODES/TIME`。

命令行行为 `--engine heuristic --heuristic-depth N --limit none|ai --ai-time S
--heuristic-version beginner|mid|advanced|hell --no-vcf`(兼容旧值 old/new)。
参数可被 `data/params.json` 复现,调优工具见 `src/antifive/tools/param_tune.py`。

### 四档等时对拍(12s/步 × 96 局/对,seed 0,严格时限)

早期 1s 口径下低档被时钟饿死(高级对初级 98% 等);12s 口径下人人能完成
深度 8,档间差距大幅收敛。2026-10-13 实测(`match_ai` 直接对拍,96 局/对;
地狱档已修复"进攻 VCF 不计入本步预算"的超时问题,与对手同为严格 12s):

| 配对(新 vs 旧) | 比分 | 近似 Elo 差(95% CI) |
|---|---|---|
| 地狱 vs 高级 | 63 : 33(65.6%) | +112(+41 ~ +184) |
| 地狱 vs 中级 | 67 : 29(69.8%) | +145(+72 ~ +219) |
| 地狱 vs 初级 | 76 : 20(79.2%) | +232(+149 ~ +314) |
| 高级 vs 中级 | 48 : 48(50.0%) | 0(−68 ~ +68) |
| 高级 vs 初级 | 51 : 45(53.1%) | +22(−46 ~ +89) |
| 中级 vs 初级 | 68 : 28(70.8%) | +154(+78 ~ +225) |

以初级为 1000 的相对 Elo(12s 口径,直接对初级换算):中级 ≈1154、
高级 ≈1022、地狱 ≈1232;旧三档在 12s 下差距收敛且存在风格不一致
(高级对中级 50%,中级对初级 70.8%),单对 CI 约 ±70 Elo。**地狱在三个
直接配对中均显著领先(对高级 +112)**,是四档中的最强者。

> 复盘注:地狱档此前的"57.3% vs 高级"含预算超时(进攻 VCF 的时间未计入
> 本步预算,平均多出数秒/步);修复后须在严格时限内取胜,当前成绩为
> 修复后重新实测。

### 防守强化机制(地狱档相对高级)

- **深化 VCF**:强制杀链从 d8/6 万节点/0.6s 提升到 d12/24 万节点/≤1.8s,
  攻防两侧共用同一预算(`_vcf_slice`);无对局限时改用确定性节点上限
  (`VCF_NODE_CAP_UNLIMITED`),保证 choose 与 analysis 结果一致;这是
  对高级 +112 Elo 的主要来源;
- **预算对等与严格计时**(P0):根防守校验覆盖全部候选、节点上限与进攻
  一致、共享同一 deadline;本步总墙钟从 choose 入口计时,"进攻 VCF + 主
  搜索 + 防守校验"合计 ≤ 时限(修复此前平均数秒/步的超时);
- **自适应防守预留**:仅当己方已有危险四连(对方下一步可完成)或对方已有
  可杀四时预留大额防守 VCF 预算,常规中盘只留 0.1s,回收时钟给搜索;
- **时限动态加深**:`MAX_DYNAMIC_DEPTH=16`,预算用满前逐层加深;
- **P1 预防性防守候选**(`ANTIFIVE_PREGIFT=1` 启用):枚举对方一步"投送
  己方子成四/五"的枢纽(连续形 + 跳形缺口,jump_through 判定),对确成杀的
  投送收集射线封堵格,按覆盖数排序补入根候选。严格等时对拍显示其净负
  (根候选变多、搜索变浅),默认关闭;
- **P2 潜伏威胁评估**(`ANTIFIVE_GIFT3=1` 启用):`E_SELF_GIFT3` 惩罚己方
  三连"可被投送"负债;严格等时对拍净负,默认关闭;
- **P3 复核**:多回合"以攻代守"在杀对已成立时不成立(不封堵则进攻方下一
  回合立即完成杀),仅有"立刻反杀"是逃生(原 has_kill 检查已覆盖),故不做
  不成立的放宽,并补回归测试锁定该语义;
- **已否决的实验**:根 α 传播 + PVS 与 LMR 速度收益大(有效深度 +1~3 层),
  但当前搜索树依赖搜索历史(`_include_extras` 会把 TT/killer 着法补进搜索
  集合),零窗试搜与全窗搜索走不同的树,边界失效(实测把最优着法从 0.871
  低估到 0.412,对拍 25%~38%),默认关闭(`USE_ROOT_PVS=False`、`USE_LMR=False`),
  代码与开关保留待确定性改造后再启用。

开局库:`antifive/openbook_data.py` 由 `tools/openbook_gen.py` 用 KataGo 生成
(默认 20k visits;黑空盘首手 + 每个 D4 等价类首手的白方应手),运行时按 D4
对称规范化查表,`ANTIFIVE_NO_BOOK=1` 可关闭。重新生成(需本地 katago):

```bash
python -m antifive.tools.openbook_gen --visits 20000
```

### KataGo 蒸馏(策略先验 + 价值校正)

`antifive/distill_data.py` 由 `tools/build_distill_data.py`(并行采样局面并用
KataGo 20k visits 标注 top-20 候选分布,可断点续跑)与 `tools/train_distill.py`
(列表式 softmax 策略先验 + 价值线性回归)生成:

```bash
python -m antifive.tools.build_distill_data --workers 4 --visits 20000 \
    --games-selfplay 200 --games-katago 12 --minutes 22   # 分块运行,续跑
python -m antifive.tools.train_distill --pool checkpoints/distill_pool.npz
```

运行时:**价值校正默认开启**(`evaluate` 混入,权重 `ANTIFIVE_VALUE_W` 默认
0.5,`ANTIFIVE_NO_DISTILL=1` 关闭);策略先验默认关闭(`ANTIFIVE_PRIOR_W`
默认 0,设为正数开启)。等时对拍:价值校正 vs 修复前 144 局 129:15(89.6%,
两个时限),策略先验单独启用 96 局 47:49(49%,故默认关闭)。

对拍(等时强弱回归;冻结快照:`heuristic_v2`=初级、`heuristic_v5`=中级、
`heuristic_v6`=高级、`antifive.heuristic`=地狱、`heuristic_v4`=搜索一致性
修复前的旧快照):

```bash
# 地狱 vs 高级(防守强化回归)
python -m antifive.tools.match_ai --games 96 --depth 8 --budget 12 --workers 8 \
    --new-module antifive.heuristic --old-module antifive.tools.heuristic_v6
# 高级 vs 中级
python -m antifive.tools.match_ai --games 96 --depth 8 --budget 12 --workers 8 \
    --new-module antifive.tools.heuristic_v6 \
    --old-module antifive.tools.heuristic_v5
# 中级 vs 初级
python -m antifive.tools.match_ai --games 96 --depth 8 --budget 12 --workers 8 \
    --new-module antifive.tools.heuristic_v5 \
    --old-module antifive.tools.heuristic_v2
```

## KataGo(逆五规则)

### 权重

- `models/katago/antifive15-init.bin`:初始权重(随机初始化,棋力无意义);
- `models/katago/antifive15-20260821-083223.bin`:自训练权重(最新)。

训练规则:逆五子棋(连五者输,`white_restrict=2`),基于
[hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo)(KataGo 分支,MIT)。

### 可执行文件(不随仓库分发)

`katago.exe` 与 CUDA/cuDNN 运行库因 NVIDIA 许可与体积原因不随仓库分发,
KataGo 的训练程序与依赖也不在开源范围内(本仓库只含运行配置与成品权重)。
运行环境自检可执行 `python scripts/setup_env.py`。
使用步骤:

1. 从 [hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo) 获取源码,
   编译支持逆五规则(white-restrict / `newgame_gomokuamazons` 相关改动)的
   `katago.exe`;
2. 将 `katago.exe` 及依赖 DLL 放到仓库根目录 `katago/`;
3. 依赖:GPU 版需要 CUDA/cuDNN(NVIDIA 官方渠道获取,不随仓库分发);
4. 在 `config/engines.json` 中确认路径。

### 配置

`config/engines.json`(相对路径以仓库根/EXE 所在目录解析):

```json
[
  { "name": "启发式", "kind": "heuristic", "depth": 8, "budget": 2.0 },
  {
    "name": "逆五KataGo",
    "engine_path": "katago/katago.exe",
    "weight_path": "models/katago/antifive15-init.bin",
    "config_path": "config/gtp_test.cfg",
    "auto_latest": true
  }
]
```

- `auto_latest=true`:自动改用权重目录中最新的 `antifive15-*.bin`
  (排除 `antifive15-init.bin`,避免随机初始化权重被误选);
- `config/gtp_test.cfg`:GTP 最小配置(`whiteRestrictTurns=2`、`maxVisits`、
  搜索线程、`maxTime` 等;GUI 会按读秒/模拟次数追加 `-override-config`);
- **本机路径覆盖**:`config/engines.local.json`(格式同上,已被 gitignore)
  优先于 `engines.json` 读取,适合放置机器相关的 `katago.exe` 绝对路径;
  在 GUI「设定 → KataGo」中用「浏览」选好路径后点「确定」会自动写入该文件。
  相对路径按 EXE 同级目录、`_internal`、仓库根、当前目录依次探测解析;
  配置指向的文件缺失时,还会在 `katago/` 等常见位置自动探测可执行文件与权重。

### 自训练与对拍

- 训练/自对弈实验使用 KataGomo 的 `selfplay` 与 GTP 工具,不属于本仓库;
- 分级检验(随机→d2→d4→d8 阶梯):
  `python -m antifive.tools.katago_vs_heur --model models/katago/antifive15-init.bin
  --engine katago/katago.exe --engine-cfg config/gtp_test.cfg`;
- 等时正式对拍(配对开局、黑白互换、断点续跑、落谱自检):
  `python -m antifive.tools.match_katago --games 96 --katago-time 12
  --heur-budget 12 --workers 2`;
  输出 `records/katago_vs_advanced_<时间戳>/`(.afg 棋谱 + 逐局 JSON +
  Wilson 置信区间的 summary.json);中断后对同一 `--out` 重跑即续跑。
  注意:该工具对启发式使用 `mask_suicide=True`(GUI 同款);不能改为
  `False`——否则启发式的候选生成会走出"占领即成己方五连"的自杀着。

### 高级启发式 vs KataGo(12s 等时,96 局)

2026-10-11~12,RTX 5070 Laptop;KataGo v1.12.3(CUDA 编译)+ 权重
`antifive15-20260821-083223.bin`;高级启发式 = VCF + 开局库 + 蒸馏价值校正 +
`data/params.json`,深度 8;双方每步 12 秒墙钟,48 个配对开局黑白互换,
数据在 `records/stage2_12s/`(不入库):

| 指标 | 结果 |
|---|---|
| 局分 | **启发式 0 : 96 KataGo**(KataGo 执黑 48:0、执白 48:0) |
| 比分 95% CI | 0%(0% ~ 3.9%);近似 Elo 差 ≤ −559 |
| 终局形态 | 96/96 由 KataGo 的"占领+安置"移子送连五给启发式落败;无认输/和棋 |
| 局长 | 中位 40 步(21~189) |
| 步时 | 启发式仅 21% 的步吃满 12s(迭代加深提前搜完);KataGo 全部 12s |
| 访问量 | KataGo 约 600~2400 visits/s(2 局并发,GPU 共享) |

**结论**:当前自训练权重的最强 KataGo 在 12s 等时下对"最强启发式"取得
96:0,强度差距大于约 560 Elo;该差距主要来自战术强制力(移子送礼连五),
而非启发式自损着(全程 0 次自我致命着)。
