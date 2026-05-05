(function () {
  function readCfg() {
    return window.__LIFERS_BRAND__ || {};
  }

  function pickSrc() {
    const b = readCfg();
    if (!b.kitten) return '';
    if (b.mode === 'kitten') return b.kitten;
    if (b.mode === 'phoenix') return b.phoenix || b.kitten;
    var ms = Math.max(1000, Number(b.intervalMs) || 4000);
    return Math.floor(Date.now() / ms) % 2 === 0 ? b.kitten : (b.phoenix || b.kitten);
  }

  function applyToImages() {
    var src = pickSrc();
    if (!src) return;
    document.querySelectorAll('img.lifers-brand-img').forEach(function (img) {
      img.src = src;
    });
  }

  function start() {
    applyToImages();
    var b = readCfg();
    if (b.mode !== 'alternate') {
      if (window.__LIFERS_BRAND_TIMER__) {
        clearInterval(window.__LIFERS_BRAND_TIMER__);
        window.__LIFERS_BRAND_TIMER__ = null;
      }
      return;
    }
    var ms = Math.max(1000, Number(b.intervalMs) || 4000);
    if (window.__LIFERS_BRAND_TIMER__) {
      clearInterval(window.__LIFERS_BRAND_TIMER__);
    }
    window.__LIFERS_BRAND_TIMER__ = setInterval(applyToImages, ms);
  }

  window.applyLifersBrandIcons = start;

  window.addEventListener('message', function (ev) {
    var m = ev.data;
    if (m && m.type === 'lifersBrand' && m.brand) {
      window.__LIFERS_BRAND__ = Object.assign({}, window.__LIFERS_BRAND__ || {}, m.brand);
      window.__LIFERS_BRAND_ICON__ = window.__LIFERS_BRAND__.kitten || window.__LIFERS_BRAND_ICON__;
      start();
    }
  });

  function boot() {
    start();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
