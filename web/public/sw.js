const VERSION = new URL(self.location.href).searchParams.get("v") || "dev";
const CACHE = `antifive-web-${VERSION}`;

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

function cacheFirst(request) {
  return caches.open(CACHE).then(async (cache) => {
    const hit = await cache.match(request);
    if (hit) return hit;
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  });
}

// GitHub Pages 无法自定义响应头:由 SW 给文档响应补 COOP/COEP,
// 启用跨源隔离(SharedArrayBuffer),使 Worker 搜索可被中断(悔棋即时生效)。
async function withIsolationHeaders(request, extra) {
  // 文档与 Worker 脚本一律绕过 HTTP 缓存:GitHub Pages 对 HTML 有 max-age,
  // 否则新部署后用户可能长时间看到旧页面(例如看不到新增的难度档)。
  const response = await fetch(request, { cache: "no-store" });
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  if (extra) {
    for (const [key, value] of Object.entries(extra)) headers.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  const sameOrigin = url.origin === self.location.origin;
  if (sameOrigin && event.request.mode === "navigate") {
    event.respondWith(withIsolationHeaders(event.request));
    return;
  }
  // 跨源隔离下,Worker 脚本响应自身也需带 COEP(否则 ERR_BLOCKED_BY_RESPONSE)
  if (sameOrigin && (event.request.destination === "worker" || event.request.destination === "sharedworker")) {
    event.respondWith(withIsolationHeaders(event.request, { "Cross-Origin-Resource-Policy": "same-origin" }));
    return;
  }
  if (!sameOrigin) return;
  event.respondWith(cacheFirst(event.request));
});
