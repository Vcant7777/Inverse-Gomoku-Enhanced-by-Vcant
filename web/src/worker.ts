/** TypeScript 原生引擎 Worker：不再加载 Pyodide。 */
import { EngineSession } from "./engine/api.ts";

const ctx = self as unknown as {
  postMessage(message: unknown): void;
  onmessage: ((event: MessageEvent) => void) | null;
};

const session = new EngineSession();

function errText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

ctx.onmessage = (event: MessageEvent) => {
  const msg = event.data;
  try {
    if (msg.type === "init") {
      if (msg.interruptBuffer) session.interruptBuffer = new Int32Array(msg.interruptBuffer);
      ctx.postMessage({ type: "status", status: { stage: "engine" } });
      ctx.postMessage({ type: "ready" });
      return;
    }
    if (msg.type !== "call") return;
    if (session.interruptBuffer) Atomics.store(session.interruptBuffer, 0, 0);
    session.onProgress = (nodes: number) => ctx.postMessage({ type: "progress", nodes });
    const value = session.handle({ cmd: msg.cmd, ...msg.payload });
    ctx.postMessage({ type: "result", id: msg.id, ok: true, value });
  } catch (error) {
    if (msg.type === "init") {
      ctx.postMessage({ type: "fatal", error: errText(error) });
    } else {
      ctx.postMessage({ type: "result", id: msg.id, ok: false, error: errText(error) });
    }
  }
};
