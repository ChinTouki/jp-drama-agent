// 注册 Service Worker
(() => {
  if (!("serviceWorker" in navigator)) return;
  window.addEventListener("load", async () => {
    try {
      const reg = await navigator.serviceWorker.register("/static/sw.js");
      if (reg.waiting) {
        reg.waiting.postMessage({ type: "SKIP_WAITING" });
      }
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        // 新 SW 接管时可刷新；这里先静默
      });
      console.log("[PWA] SW registered:", reg.scope);
    } catch (e) {
      console.warn("[PWA] SW registration failed:", e);
    }
  });
})();
