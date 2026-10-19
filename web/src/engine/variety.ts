/**
 * AI 变化(去固定化)档位。
 *
 * 网页版引擎是纯确定性的迭代加深 alpha-beta:同一局面 + 同一深度/时限 + 同一
 * 置换表状态 ⇒ 同一着法。玩家连下同样的手顺就会看到完全一样的应手,体验上像
 * 「背棋谱」。这里给出可控的随机性:
 *
 *   1. 根着法软化采样:同一深度下每个根着法都有精确分值(根节点全窗口搜索),
 *      只在与最优着法差距 ≤ window 的着法之间按 exp((v - v_max)/T) 加权抽样。
 *      差距远大于 T 时最优着法权重≈1 → 棋力基本无损;接近同分时才换着。
 *   2. 开局偏移:前 openingPlies 手用更大的 T/window,让开局不再千篇一律。
 *   3. 开局库概率化:bookChance<1 时按概率跳过 KataGo 开局库,交由搜索决定。
 *
 * 强制杀(VCF / 一手安置杀 / 已知必胜必败)不参与抽样,始终走最快杀/最长抵抗,
 * 保证「能赢的局不会因为随机而赢不下来」。
 *
 * 随机源是每局一次的种子 RNG(见 `mulberry32`),因此:同一局内可复现、跨局不同,
 * 且不依赖 `Math.random` 的全局状态。
 */

export type VarietyLevel = "off" | "low" | "mid" | "high" | "custom";

export interface VarietySpec {
  level: VarietyLevel;
  /** 根着法价值软化温度(eval 单位:一个「可杀四」权重 ≈ 0.9)。 */
  temperature: number;
  /** 只与最优着法差距 ≤ window 的着法参与抽样,防止随机走坏棋。 */
  window: number;
  /**
   * 只在前 K 个最优根着法里抽样(按搜索分值排序,同分保持候选启发式顺序)。
   * 稀疏局面(空盘/一子)评估差普遍 < 0.1,单靠 window 会把棋盘另一端毫无关系的
   * 远端点也算成「同分」——实测真出现过角点。K 收口后,池内只剩候选启发式
   * 排序靠前的点(中心优先),diversity 补充的远端点不会进入池子。
   */
  poolCap: number;
  /** 前 N 个半手(落子/占领/安置各计一步)视为开局。 */
  openingPlies: number;
  openingTemperature: number;
  openingWindow: number;
  openingPoolCap: number;
  /** 开局库启用概率(1=永远用库)。 */
  bookChance: number;
}

const ALIASES: Record<string, VarietyLevel> = {
  "": "off", off: "off", none: "off", "0": "off", "关": "off", "关闭": "off",
  low: "low", "1": "low", "低": "low",
  mid: "mid", medium: "mid", "2": "mid", "中": "mid",
  high: "high", "3": "high", "高": "high",
  custom: "custom", "4": "custom", "自定义": "custom",
};

const SPECS: Record<VarietyLevel, VarietySpec> = {
  off: {
    level: "off", temperature: 0, window: 0, poolCap: 1, openingPlies: 0,
    openingTemperature: 0, openingWindow: 0, openingPoolCap: 1, bookChance: 1,
  },
  low: {
    level: "low", temperature: 0.06, window: 0.15, poolCap: 2, openingPlies: 4,
    openingTemperature: 0.12, openingWindow: 0.3, openingPoolCap: 3, bookChance: 0.55,
  },
  mid: {
    level: "mid", temperature: 0.10, window: 0.3, poolCap: 3, openingPlies: 4,
    openingTemperature: 0.22, openingWindow: 0.5, openingPoolCap: 3, bookChance: 0.35,
  },
  high: {
    level: "high", temperature: 0.20, window: 0.6, poolCap: 4, openingPlies: 6,
    openingTemperature: 0.35, openingWindow: 0.8, openingPoolCap: 4, bookChance: 0.2,
  },
  // 自定义 = 在 mid 基础上逐项覆盖(见 varietyFor 的 params 参数)。
  custom: {
    level: "custom", temperature: 0.10, window: 0.3, poolCap: 3, openingPlies: 4,
    openingTemperature: 0.22, openingWindow: 0.5, openingPoolCap: 3, bookChance: 0.35,
  },
};

export type VarietyParamKey =
  | "temperature" | "window" | "poolCap" | "openingPlies"
  | "openingTemperature" | "openingWindow" | "openingPoolCap" | "bookChance";

/** 可调参数(单一事实来源:引擎校验与设置面板共用)。 */
export interface VarietyParamField {
  key: VarietyParamKey;
  label: string;
  min: number;
  max: number;
  step: number;
  integer?: boolean;
}

export const PARAM_FIELDS: readonly VarietyParamField[] = [
  { key: "temperature", label: "中盘 T", min: 0, max: 1, step: 0.02 },
  { key: "window", label: "中盘 window", min: 0, max: 3, step: 0.05 },
  { key: "poolCap", label: "中盘 K", min: 1, max: 8, step: 1, integer: true },
  { key: "openingPlies", label: "开局手数", min: 0, max: 20, step: 1, integer: true },
  { key: "openingTemperature", label: "开局 T", min: 0, max: 1.5, step: 0.02 },
  { key: "openingWindow", label: "开局 window", min: 0, max: 4, step: 0.05 },
  { key: "openingPoolCap", label: "开局 K", min: 1, max: 8, step: 1, integer: true },
  { key: "bookChance", label: "开局库概率", min: 0, max: 1, step: 0.05 },
];

const FIELD_BY_KEY = new Map<VarietyParamKey, VarietyParamField>(
  PARAM_FIELDS.map((f) => [f.key, f]),
);

export type VarietyParams = Partial<Record<VarietyParamKey, number>>;

/** 校验并夹紧自定义参数:非有限值丢弃,整数项取整,其余夹到 [min,max]。 */
export function sanitizeParams(raw: unknown): VarietyParams {
  if (!raw || typeof raw !== "object") return {};
  const out: VarietyParams = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    const field = FIELD_BY_KEY.get(key as VarietyParamKey);
    if (!field) continue;
    const n = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(n)) continue;
    const clamped = Math.min(field.max, Math.max(field.min, n));
    out[field.key] = field.integer ? Math.round(clamped) : clamped;
  }
  return out;
}

export function normalizeVariety(v: unknown): VarietyLevel {
  const key = String(v ?? "").trim().toLowerCase();
  return ALIASES[key] ?? "off";
}

/**
 * 未指定/无法识别 = off(引擎默认保持完全确定性,脚本与测试口径不变)。
 * params 只覆盖 custom 档(其余档位忽略,避免预设被悄悄改掉)。
 */
export function varietyFor(v: unknown, params?: unknown): VarietySpec {
  const level = normalizeVariety(v);
  const spec = SPECS[level];
  if (level !== "custom" || params === undefined) return spec;
  const clean = sanitizeParams(params);
  return { ...spec, ...clean };
}

/** 当前是否处于开局阶段(按半手数)。 */
export function openingPhase(spec: VarietySpec, moveCount: number): boolean {
  return spec.openingPlies > 0 && moveCount < spec.openingPlies;
}

/** 同一档位在当前局面下实际使用的 (temperature, window, poolCap)。 */
export function varietyAt(spec: VarietySpec, moveCount: number): [number, number, number] {
  if (spec.level === "off") return [0, 0, 1];
  return openingPhase(spec, moveCount)
    ? [spec.openingTemperature, spec.openingWindow, spec.openingPoolCap]
    : [spec.temperature, spec.window, spec.poolCap];
}

/** mulberry32:32 位种子 → [0,1) 均匀随机;同一 seed 序列可复现。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 取一枚不可预测的 32 位种子(浏览器/Worker/Node 均可用)。 */
export function randomSeed(): number {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c && typeof c.getRandomValues === "function") {
    const buf = new Uint32Array(1);
    c.getRandomValues(buf);
    if (buf[0]) return buf[0] >>> 0;
  }
  return ((Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0) || 1;
}
