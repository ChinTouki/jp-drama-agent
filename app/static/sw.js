/* JP-Drama Agent SW v1.0.0 */
const CACHE_VER = "v1.0.0";
const APP_CACHE = `jp-agent-${CACHE_VER}`;

const APP_SHELL = [
  "/playground",
  "/static/manifest.webmanifest",
  "/static/register-sw.js",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/maskable-192.png",
  "/static/icons/maskable-512.png"
];

self.addEventListener("install", (evt) => {
  evt.waitUntil(caches.open(APP_CACHE).then((c) => c.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (evt) => {
  evt.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map(k => k !== APP_CACHE && caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (evt) => {
  const req = evt.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const isHtmlNav = req.mode === "navigate" || (req.headers.get("accept") || "").includes("text/html");

  if (isHtmlNav && url.pathname.startsWith("/playground")) {
    evt.respondWith((async () => {
      const cache = await caches.open(APP_CACHE);
      const cached = await cache.match("/playground");
      const fetched = fetch(req).then(async (res) => {
        try { await cache.put("/playground", res.clone()); } catch {}
        return res;
      }).catch(() => cached);
      return cached || fetched;
    })());
    return;
  }

  if (url.origin === location.origin) {
    evt.respondWith((async () => {
      const cache = await caches.open(APP_CACHE);
      const cached = await cache.match(req);
      const fetching = fetch(req).then(async (res) => {
        if (res && res.status === 200) {
          try { await cache.put(req, res.clone()); } catch {}
        }
        return res;
      }).catch(() => cached);
      return cached || fetching;
    })());
    return;
  }

  evt.respondWith((async () => {
    try { return await fetch(req); }
    catch {
      const cache = await caches.open(APP_CACHE);
      const cached = await cache.match(req);
      return cached || new Response("", { status: 504, statusText: "offline" });
    }
  })());
});

self.addEventListener("message", (evt) => {
  if (evt.data && evt.data.type === "SKIP_WAITING") self.skipWaiting();
});
