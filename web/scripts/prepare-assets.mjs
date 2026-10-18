/** TypeScript 引擎不再需要 Pyodide/numpy/wheel 资源；清理旧构建残留。 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(here, "..", "public");
for (const name of ["pyodide", "antifive.whl"]) {
  fs.rmSync(path.join(publicDir, name), { recursive: true, force: true });
}
fs.mkdirSync(publicDir, { recursive: true });
console.log("TypeScript engine: no external runtime assets");
