/** KataGo 蒸馏价值校正（v1 线性 / v2 单隐层 MLP）。 */
import {
  V1_B_VALUE, V1_VALUE_W, V1_W_VALUE,
  V2_V_B1, V2_V_B2, V2_V_MEAN, V2_V_STD, V2_V_W1, V2_V_W2, V2_VALUE_W,
} from "./distill_data.ts";

export type DistillMode = 0 | 1 | 2;

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** feats: v1 20 维；v2 25 维（前 20 维同 v1）。 */
export function valueCorrection(mode: DistillMode, feats: number[]): number {
  if (mode === 1) {
    return V1_VALUE_W * (dot(feats, V1_W_VALUE) + V1_B_VALUE);
  }
  if (mode === 2) {
    const h = new Array<number>(V2_V_B1.length);
    for (let i = 0; i < h.length; i++) {
      let v = V2_V_B1[i];
      const row = V2_V_W1[i];
      for (let j = 0; j < row.length; j++) {
        v += row[j] * ((feats[j] - V2_V_MEAN[j]) / V2_V_STD[j]);
      }
      h[i] = Math.max(0, v);
    }
    return V2_VALUE_W * (dot(h, V2_V_W2) + V2_V_B2);
  }
  return 0;
}
