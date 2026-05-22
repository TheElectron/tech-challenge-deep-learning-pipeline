"""Self-contained HTML dashboard served at GET /."""

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LSTM Stock Predictor</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:          #08080e;
      --surface:     #10101a;
      --surface2:    #181824;
      --border:      #252535;
      --text:        #e2e2f0;
      --muted:       #7878a0;
      --accent:      #00d4aa;
      --accent-dim:  rgba(0,212,170,.12);
      --accent-glow: rgba(0,212,170,.25);
      --danger:      #ff4757;
      --warn:        #ffa502;
      --radius:      14px;
      --radius-sm:   9px;
      --shadow:      0 8px 32px rgba(0,0,0,.5);
      --font:        -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono:        "SF Mono", "Fira Code", "Consolas", monospace;
      --transition:  0.18s ease;
    }

    html { scroll-behavior: smooth; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      line-height: 1.5;
    }

    /* ── Header ─────────────────────────────────────────── */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0.9rem 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      position: sticky;
      top: 0;
      z-index: 200;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .logo-mark {
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #00d4aa, #00a882);
      border-radius: 10px;
      display: grid;
      place-items: center;
      font-size: 1.2rem;
      flex-shrink: 0;
      box-shadow: 0 0 16px var(--accent-glow);
    }

    .logo-text h1 {
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: -.02em;
      line-height: 1.2;
    }

    .logo-text span {
      font-size: 0.75rem;
      color: var(--muted);
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.35rem 0.85rem;
      border-radius: 100px;
      font-size: 0.78rem;
      font-weight: 600;
      border: 1px solid transparent;
      transition: var(--transition);
    }
    .badge.ok     { background: rgba(0,212,170,.1); border-color: rgba(0,212,170,.3); color: var(--accent); }
    .badge.error  { background: rgba(255,71,87,.1);  border-color: rgba(255,71,87,.3);  color: var(--danger); }
    .badge.loading{ background: rgba(120,120,160,.1);border-color: rgba(120,120,160,.3);color: var(--muted); }

    .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
    .dot.pulse { animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

    /* ── Layout ─────────────────────────────────────────── */
    main {
      max-width: 1080px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    /* ── Card ───────────────────────────────────────────── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.5rem;
      box-shadow: var(--shadow);
    }

    .card-title {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .1em;
      color: var(--muted);
      margin-bottom: 1.1rem;
    }

    /* ── Model info grid ────────────────────────────────── */
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.85rem;
    }

    .info-tile {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 1.1rem 1.2rem;
    }

    .info-tile-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: .07em;
      color: var(--muted);
      margin-bottom: 0.45rem;
    }

    .info-tile-value {
      font-family: var(--mono);
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text);
      line-height: 1.15;
    }
    .info-tile-value.accent { color: var(--accent); }
    .info-tile-value.sm     { font-size: 1rem; word-break: break-all; }

    /* ── Form ───────────────────────────────────────────── */
    .form-cols {
      display: grid;
      grid-template-columns: 1fr 2fr;
      gap: 1.25rem;
      align-items: start;
    }
    @media (max-width: 680px) { .form-cols { grid-template-columns: 1fr; } }

    .form-group { display: flex; flex-direction: column; gap: 0.45rem; }

    label {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--muted);
      letter-spacing: .02em;
    }

    input, textarea {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text);
      font-family: var(--font);
      font-size: 0.93rem;
      padding: 0.65rem 0.9rem;
      outline: none;
      width: 100%;
      transition: border-color var(--transition), box-shadow var(--transition);
    }
    textarea {
      font-family: var(--mono);
      font-size: 0.8rem;
      min-height: 130px;
      resize: vertical;
      line-height: 1.65;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    input::placeholder, textarea::placeholder { color: var(--muted); opacity: .6; }

    .hint {
      font-size: 0.72rem;
      color: var(--muted);
      transition: color var(--transition);
    }
    .hint.ok { color: var(--accent); }

    /* Quick ticker chips */
    .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.25rem; }

    .chip {
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--muted);
      border-radius: 5px;
      padding: 0.18rem 0.55rem;
      font-size: 0.75rem;
      font-family: var(--mono);
      cursor: pointer;
      transition: all var(--transition);
    }
    .chip:hover { border-color: var(--accent); color: var(--accent); }

    /* ── Sparkline ──────────────────────────────────────── */
    #chart-wrap {
      margin-top: 0.9rem;
      display: none;
    }
    #chart-meta {
      font-size: 0.7rem;
      color: var(--muted);
      margin-bottom: 0.4rem;
    }
    #sparkline {
      width: 100%;
      height: 72px;
      border-radius: var(--radius-sm);
      overflow: hidden;
      display: block;
    }

    /* ── Buttons ────────────────────────────────────────── */
    .btn-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-top: 1.2rem; align-items: center; }

    .btn {
      background: var(--accent);
      color: #080810;
      border: none;
      border-radius: var(--radius-sm);
      font-size: 0.9rem;
      font-weight: 700;
      padding: 0.72rem 1.6rem;
      cursor: pointer;
      transition: filter var(--transition), transform var(--transition);
      letter-spacing: .01em;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }
    .btn:hover:not(:disabled) { filter: brightness(1.1); transform: translateY(-1px); }
    .btn:active:not(:disabled){ transform: translateY(0); }
    .btn:disabled { opacity: .45; cursor: not-allowed; }

    .btn-ghost {
      background: var(--surface2);
      color: var(--text);
      border: 1px solid var(--border);
    }
    .btn-ghost:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); filter: none; }

    .btn-sm {
      padding: 0.38rem 0.85rem;
      font-size: 0.78rem;
    }

    /* ── Spinner ────────────────────────────────────────── */
    .spin {
      display: inline-block;
      width: 14px; height: 14px;
      border: 2px solid rgba(0,0,0,.25);
      border-top-color: currentColor;
      border-radius: 50%;
      animation: spin .65s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Error banner ───────────────────────────────────── */
    .error-banner {
      display: none;
      background: rgba(255,71,87,.09);
      border: 1px solid rgba(255,71,87,.3);
      border-radius: var(--radius-sm);
      color: var(--danger);
      padding: 0.7rem 1rem;
      font-size: 0.83rem;
      margin-top: 1rem;
    }

    /* ── Result card ────────────────────────────────────── */
    #result-card {
      display: none;
      background: linear-gradient(135deg, #0f1f1c 0%, var(--surface2) 100%);
      border-color: rgba(0,212,170,.25);
      margin-top: 1.1rem;
    }

    .result-layout {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .result-left .ticker-label {
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .07em;
      margin-bottom: 0.3rem;
    }

    .result-price {
      font-size: 3.2rem;
      font-weight: 800;
      font-family: var(--mono);
      color: var(--accent);
      line-height: 1;
      text-shadow: 0 0 32px var(--accent-glow);
    }

    .result-ts {
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 0.4rem;
    }

    .result-meta {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      text-align: right;
    }
    .result-meta span { font-size: 0.78rem; color: var(--muted); }
    .result-meta strong { font-family: var(--mono); color: var(--text); }

    /* ── Table ──────────────────────────────────────────── */
    .table-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .table-wrap { overflow-x: auto; }

    table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }

    th {
      text-align: left;
      padding: 0.5rem 0.8rem;
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .07em;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }

    td {
      padding: 0.62rem 0.8rem;
      border-bottom: 1px solid rgba(37,37,53,.7);
      font-family: var(--mono);
      font-size: 0.81rem;
      white-space: nowrap;
    }
    tr:last-child td { border-bottom: none; }
    tbody tr:hover td { background: var(--surface2); }

    .tk-pill {
      display: inline-block;
      background: var(--accent-dim);
      color: var(--accent);
      border-radius: 4px;
      padding: 0.12rem 0.48rem;
      font-size: 0.75rem;
      font-weight: 700;
    }

    .delta-up   { color: var(--accent); font-weight: 700; }
    .delta-down { color: var(--danger);  font-weight: 700; }

    .empty { text-align: center; padding: 2.5rem 1rem; color: var(--muted); font-size: 0.88rem; font-family: var(--font); }

    /* ── Animations ─────────────────────────────────────── */
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .fade-up { animation: fadeUp .3s ease forwards; }

    /* ── Footer ─────────────────────────────────────────── */
    footer {
      text-align: center;
      padding: 1.75rem 1rem;
      border-top: 1px solid var(--border);
      font-size: 0.76rem;
      color: var(--muted);
      margin-top: 0.5rem;
    }
    footer a { color: var(--accent); text-decoration: none; }
    footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>

<!-- ── Header ───────────────────────────────────────────── -->
<header>
  <div class="logo">
    <div class="logo-mark">📈</div>
    <div class="logo-text">
      <h1>LSTM Stock Predictor</h1>
      <span>Multi-asset closing price forecast</span>
    </div>
  </div>
  <div class="header-right">
    <div id="health-badge" class="badge loading">
      <span class="dot pulse"></span>
      <span id="health-text">Connecting…</span>
    </div>
  </div>
</header>

<!-- ── Main ─────────────────────────────────────────────── -->
<main>

  <!-- Model info -->
  <div class="card" id="model-card">
    <div class="card-title">Champion Model</div>
    <div class="info-grid">
      <div class="info-tile">
        <div class="info-tile-label">Version</div>
        <div class="info-tile-value accent" id="m-version">—</div>
      </div>
      <div class="info-tile">
        <div class="info-tile-label">Architecture</div>
        <div class="info-tile-value sm" id="m-type">—</div>
      </div>
      <div class="info-tile">
        <div class="info-tile-label">Test RMSE</div>
        <div class="info-tile-value" id="m-rmse">—</div>
      </div>
      <div class="info-tile">
        <div class="info-tile-label">Registry</div>
        <div class="info-tile-value sm" id="m-name">—</div>
      </div>
    </div>
  </div>

  <!-- Prediction form -->
  <div class="card">
    <div class="card-title">Make a Prediction</div>

    <form id="predict-form">
      <div class="form-cols">

        <!-- Left column -->
        <div style="display:flex;flex-direction:column;gap:1rem;">
          <div class="form-group">
            <label for="ticker-input">Ticker symbol</label>
            <input id="ticker-input" type="text" placeholder="AAPL, MSFT, PETR4.SA…" autocomplete="off" />
            <span class="hint">The model is asset-agnostic — any ticker works.</span>
          </div>
          <div class="form-group">
            <label>Quick select</label>
            <div class="chips" id="chips"></div>
          </div>
        </div>

        <!-- Right column -->
        <div class="form-group">
          <label for="prices-input">Closing prices — at least 60 values (one per line or comma-separated)</label>
          <textarea id="prices-input" placeholder="176.38&#10;177.31&#10;175.73&#10;…"></textarea>
          <span class="hint" id="price-hint">0 values entered</span>
          <div id="chart-wrap">
            <div id="chart-meta"></div>
            <svg id="sparkline" viewBox="0 0 600 72" preserveAspectRatio="none"></svg>
          </div>
        </div>

      </div><!-- /.form-cols -->

      <div class="btn-row">
        <button type="submit" class="btn" id="submit-btn">Predict closing price</button>
        <button type="button" class="btn btn-ghost" id="example-btn">Fill example (AAPL)</button>
      </div>
    </form>

    <div class="error-banner" id="error-banner"></div>

    <!-- Result -->
    <div class="card fade-up" id="result-card">
      <div class="result-layout">
        <div class="result-left">
          <div class="ticker-label" id="res-ticker">—</div>
          <div class="result-price" id="res-price">$0.00</div>
          <div class="result-ts" id="res-ts">—</div>
        </div>
        <div class="result-meta">
          <span>Version: <strong id="res-version">—</strong></span>
          <span>Alias: <strong id="res-alias">—</strong></span>
        </div>
      </div>
    </div>

  </div><!-- /.card form -->

  <!-- Recent predictions -->
  <div class="card">
    <div class="table-bar">
      <div class="card-title" style="margin-bottom:0">Recent Predictions</div>
      <button class="btn btn-ghost btn-sm" id="refresh-btn">↺ Refresh</button>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Predicted close</th>
            <th>Last input close</th>
            <th>Δ</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody id="pred-tbody">
          <tr><td colspan="5" class="empty">No predictions logged yet.</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</main>

<footer>
  LSTM Stock Predictor &nbsp;·&nbsp;
  <a href="/docs" target="_blank">Swagger UI</a> &nbsp;·&nbsp;
  <a href="/redoc" target="_blank">ReDoc</a> &nbsp;·&nbsp;
  <a href="/metrics" target="_blank">Prometheus metrics</a>
</footer>

<script>
(function () {
  'use strict';

  const TICKERS  = ['AAPL','MSFT','GOOGL','AMZN','PETR4.SA','VALE3.SA'];
  const EXAMPLE  = Array.from({length:60},(_,i)=>(150+i*0.5+Math.sin(i*.35)*4).toFixed(2));

  // ── Helpers ────────────────────────────────────────────
  async function apiFetch(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      const body = await res.json().catch(() => ({detail: res.statusText}));
      throw new Error(body.detail || res.statusText);
    }
    return res.json();
  }

  function parsePrices(raw) {
    return raw.split(/[,\\n\\r\\s]+/)
              .map(s => s.trim())
              .filter(Boolean)
              .map(Number)
              .filter(n => !isNaN(n) && isFinite(n));
  }

  function fmt(n, d=4) { return '$' + n.toFixed(d); }
  function fmtTs(iso)   { return new Date(iso).toLocaleString(); }

  // ── Sparkline ──────────────────────────────────────────
  function drawSparkline(prices) {
    const svg  = document.getElementById('sparkline');
    const wrap = document.getElementById('chart-wrap');
    const meta = document.getElementById('chart-meta');
    if (prices.length < 2) { wrap.style.display='none'; return; }

    const W=600, H=72, P=6;
    const min=Math.min(...prices), max=Math.max(...prices), rng=max-min||1;
    const pts = prices.map((p,i)=>{
      const x = P + (i/(prices.length-1))*(W-P*2);
      const y = H - P - ((p-min)/rng)*(H-P*2);
      return `${x},${y}`;
    }).join(' ');
    const [lx,ly] = pts.split(' ').pop().split(',');

    svg.innerHTML = `
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#00d4aa" stop-opacity=".22"/>
          <stop offset="100%" stop-color="#00d4aa" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <polygon points="${pts} ${W-P},${H} ${P},${H}" fill="url(#lg)"/>
      <polyline points="${pts}" fill="none" stroke="#00d4aa" stroke-width="1.8"
                stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="${lx}" cy="${ly}" r="3.5" fill="#00d4aa"/>`;

    meta.textContent =
      `Window (${prices.length} days) · last ${fmt(prices[prices.length-1],2)} · `+
      `min ${fmt(min,2)} · max ${fmt(max,2)}`;
    wrap.style.display = 'block';
  }

  // ── Health ─────────────────────────────────────────────
  async function checkHealth() {
    const badge = document.getElementById('health-badge');
    const text  = document.getElementById('health-text');
    try {
      const h = await apiFetch('/health');
      if (h.status==='ok' && h.model_loaded) {
        badge.className='badge ok';
        text.textContent='API online · model loaded';
        badge.querySelector('.dot').classList.add('pulse');
      } else {
        badge.className='badge error';
        text.textContent='Model not loaded';
      }
    } catch {
      badge.className='badge error';
      text.textContent='API unreachable';
    }
  }

  // ── Model info ─────────────────────────────────────────
  async function loadModelInfo() {
    try {
      const info = await apiFetch('/model/info');
      document.getElementById('m-version').textContent = 'v' + info.version;
      document.getElementById('m-type').textContent    = info.model_type.replace(/_/g,' ');
      document.getElementById('m-rmse').textContent    = info.rmse.toFixed(4);
      document.getElementById('m-name').textContent    = info.name;
    } catch { /* badge already reflects failure */ }
  }

  // ── Predictions table ──────────────────────────────────
  async function loadPredictions() {
    try {
      const rows = await apiFetch('/monitoring/predictions?n=25');
      renderTable(rows.slice().reverse());
    } catch { /* log file may not exist yet */ }
  }

  function renderTable(rows) {
    const tbody = document.getElementById('pred-tbody');
    if (!rows || rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty">No predictions logged yet.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r => {
      const delta = r.predicted_close - r.last_close;
      const cls   = delta >= 0 ? 'delta-up' : 'delta-down';
      const sign  = delta >= 0 ? '+' : '';
      return `<tr>
        <td><span class="tk-pill">${r.ticker}</span></td>
        <td>${fmt(r.predicted_close)}</td>
        <td>${fmt(r.last_close)}</td>
        <td class="${cls}">${sign}${delta.toFixed(2)}</td>
        <td style="color:var(--muted)">${fmtTs(r.timestamp)}</td>
      </tr>`;
    }).join('');
  }

  // ── Price input listener ────────────────────────────────
  document.getElementById('prices-input').addEventListener('input', function () {
    const prices = parsePrices(this.value);
    const hint   = document.getElementById('price-hint');
    const ok     = prices.length >= 60;
    hint.textContent = `${prices.length} value${prices.length!==1?'s':''} entered`
      + (ok ? ' ✓' : ` — need at least 60`);
    hint.className = ok ? 'hint ok' : 'hint';
    drawSparkline(prices);
  });

  // ── Ticker chips ───────────────────────────────────────
  const chipsEl = document.getElementById('chips');
  TICKERS.forEach(t => {
    const c = document.createElement('button');
    c.type = 'button';
    c.className = 'chip';
    c.textContent = t;
    c.addEventListener('click', () => {
      document.getElementById('ticker-input').value = t;
    });
    chipsEl.appendChild(c);
  });

  // ── Example fill ───────────────────────────────────────
  document.getElementById('example-btn').addEventListener('click', () => {
    document.getElementById('ticker-input').value = 'AAPL';
    document.getElementById('prices-input').value = EXAMPLE.join('\\n');
    document.getElementById('prices-input').dispatchEvent(new Event('input'));
  });

  // ── Submit ─────────────────────────────────────────────
  document.getElementById('predict-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
    const prices = parsePrices(document.getElementById('prices-input').value);
    const errEl  = document.getElementById('error-banner');
    const resEl  = document.getElementById('result-card');
    const btn    = document.getElementById('submit-btn');

    errEl.style.display = 'none';
    resEl.style.display  = 'none';

    if (!ticker)           { showError('Enter a ticker symbol.'); return; }
    if (prices.length < 60){ showError(`Need ≥ 60 prices — got ${prices.length}.`); return; }

    btn.disabled = true;
    btn.innerHTML = '<span class="spin"></span> Predicting…';

    try {
      const res = await apiFetch('/predict', {
        method : 'POST',
        headers: {'Content-Type':'application/json'},
        body   : JSON.stringify({ticker, close_prices: prices}),
      });

      document.getElementById('res-ticker').textContent  = res.ticker;
      document.getElementById('res-price').textContent   = fmt(res.predicted_close);
      document.getElementById('res-ts').textContent      = fmtTs(res.prediction_timestamp);
      document.getElementById('res-version').textContent = 'v' + res.model_version;
      document.getElementById('res-alias').textContent   = res.model_alias;

      resEl.style.display = 'block';
      resEl.classList.remove('fade-up');
      void resEl.offsetWidth;
      resEl.classList.add('fade-up');

      loadPredictions();
    } catch (err) {
      showError(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Predict closing price';
    }
  });

  function showError(msg) {
    const el = document.getElementById('error-banner');
    el.textContent  = msg;
    el.style.display = 'block';
  }

  // ── Refresh button ─────────────────────────────────────
  document.getElementById('refresh-btn').addEventListener('click', loadPredictions);

  // ── Auto-refresh ───────────────────────────────────────
  setInterval(loadPredictions, 30_000);

  // ── Boot ───────────────────────────────────────────────
  checkHealth();
  loadModelInfo();
  loadPredictions();
})();
</script>
</body>
</html>"""
