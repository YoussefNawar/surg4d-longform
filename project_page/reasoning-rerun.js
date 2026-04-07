/**
 * Lazy-load Rerun WebViewer for any element with [data-rrd-url].
 * Gives the host an explicit pixel size at start() and refreshes on resize
 * so the canvas fills the allocated tile (avoids default 300×150-style sizing).
 */
import { WebViewer } from "https://esm.sh/@rerun-io/web-viewer@0.30.2";

const baseOpts = { hide_welcome_screen: true, allow_fullscreen: true };
/**
 * Side chrome: wide viewports default the left blueprint panel to expanded.
 * Bottom "Streams" tree: not the blueprint — it lives in the time panel when that
 * panel is expanded (Rerun uses expanded time when window height > ~600px).
 * Collapsing the time panel keeps play/pause, timeline, and scrubber; expanded adds the streams UI.
 */
const hidePanelsDefault = ["blueprint", "selection", "top"];
const hidePanelsInteractive = ["blueprint", "selection", "top"];

function panelsToHide(host) {
  return host.closest(".rerun-overview-row") ? hidePanelsInteractive : hidePanelsDefault;
}

function applyPanelHides(viewer, host) {
  for (const p of panelsToHide(host)) {
    try {
      viewer.override_panel_state(p, "hidden");
    } catch (_) {
      /* ignore */
    }
  }
  try {
    viewer.override_panel_state("time", "collapsed");
  } catch (_) {
    /* ignore */
  }
}

/** Re-apply after RRD/async layout; do not pass panel_state_overrides into start() — it can break wasm init (stuck on “Loading Rerun…”). */
function schedulePanelHideRetries(viewer, host) {
  for (const ms of [0, 80, 250, 700, 2000]) {
    setTimeout(() => applyPanelHides(viewer, host), ms);
  }
}

const started = new WeakSet();
const viewers = new WeakMap();

let queue = Promise.resolve();

function pixelBox(el) {
  const r = el.getBoundingClientRect();
  const w = Math.max(2, Math.floor(r.width));
  const h = Math.max(2, Math.floor(r.height));
  return { width: `${w}px`, height: `${h}px` };
}

function stretchCanvas(host) {
  const canvas = host.querySelector("canvas");
  if (!canvas) return;
  canvas.style.position = "absolute";
  canvas.style.left = "0";
  canvas.style.top = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.maxWidth = "none";
  canvas.style.maxHeight = "none";
  canvas.style.display = "block";
}

function notifyViewerResize(host) {
  const v = viewers.get(host);
  if (!v) return;
  for (const name of ["handle_resize", "resize"]) {
    if (typeof v[name] === "function") {
      try {
        v[name]();
      } catch (_) {
        /* ignore */
      }
      break;
    }
  }
  stretchCanvas(host);
}

async function waitForLayout(host) {
  for (let i = 0; i < 5; i++) {
    const r = host.getBoundingClientRect();
    if (r.width >= 32 && r.height >= 32) return;
    await new Promise((res) => requestAnimationFrame(res));
  }
}

function scheduleStart(host) {
  queue = queue.then(async () => {
    if (!host.isConnected || started.has(host)) return;
    const rawUrl = host.getAttribute("data-rrd-url");
    if (!rawUrl) return;
    // Newer web-viewer versions can mis-handle "./..." inputs; resolve explicitly.
    const url = new URL(rawUrl, window.location.href).href;
    started.add(host);

    try {
      await waitForLayout(host);
      const dims = pixelBox(host);
      const v = new WebViewer();
      await v.start(url, host, { ...baseOpts, ...dims });
      viewers.set(host, v);

      applyPanelHides(v, host);
      schedulePanelHideRetries(v, host);
      v.on("recording_open", () => applyPanelHides(v, host));

      stretchCanvas(host);
      notifyViewerResize(host);

      let hideDebounce = null;
      const ro = new ResizeObserver(() => {
        stretchCanvas(host);
        notifyViewerResize(host);
        clearTimeout(hideDebounce);
        hideDebounce = setTimeout(() => applyPanelHides(v, host), 200);
      });
      ro.observe(host);
    } catch (e) {
      console.warn("Rerun viewer failed:", host.id || host.className, e);
    }
  });
}

const io = new IntersectionObserver(
  (entries) => {
    for (const ent of entries) {
      if (!ent.isIntersecting) continue;
      const host = ent.target;
      io.unobserve(host);
      scheduleStart(host);
    }
  },
  { root: null, rootMargin: "120px 0px", threshold: 0.05 }
);

for (const host of document.querySelectorAll("[data-rrd-url]")) {
  io.observe(host);
}
