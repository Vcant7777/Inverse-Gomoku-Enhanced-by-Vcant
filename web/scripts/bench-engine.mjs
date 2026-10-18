import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.resolve(here, "..", "dist");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "") || "index.html";
  const file = path.join(dist, rel);
  if (!file.startsWith(dist) || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});
const port = await new Promise((r) => server.listen(0, "127.0.0.1", () => r(server.address().port)));
const chrome = process.env.CHROME_PATH ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--no-sandbox"] });
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => window.__antifive?.ready, { timeout: 60000 });
const result = await page.evaluate(async () => {
  const app = window.__antifive;
  await app.engine.call("new", {});
  for (const idx of [112, 52, 172, 34, 190, 76]) await app.engine.call("move", { idx });
  const out = [];
  for (const depth of [4, 5, 6]) {
    await app.engine.call("new", {});
    for (const idx of [112, 52, 172, 34, 190, 76]) await app.engine.call("move", { idx });
    const t0 = performance.now();
    const r = await app.engine.call("ai", { depth, budget: 10, engine: "advanced", vcf: true }, 30000);
    out.push({ depth, ms: performance.now() - t0, nodes: r.nodes, elapsed: r.elapsed, move: r.move });
  }
  return out;
});
console.log(JSON.stringify(result, null, 2));
await browser.close();
server.close();
