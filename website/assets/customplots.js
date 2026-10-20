/* Custom Plots: the Company series plot of the live training page (Forecaster v4 tab), and the projection of a company's price on N Legendre polynomials.
   Reads through the ops server (scripts/tools/ops_server.jl): /api/symbols, /api/series and /api/legendre. */
(() => {
  const GRANS = [['15 min', '15min'], ['1 min', '1min'], ['5 min', '5min'], ['hourly', 'hourly'], ['daily', 'daily']];
  const RUN_DIR = { '1min': 'data/forecaster', '15min': 'data/forecaster_15min', '5min': 'data/forecaster_5min', hourly: 'data/forecaster_hourly', daily: 'data/forecaster_daily' };
  const PRESETS = [5, 10, 20, 50, 100, 200];
  const SERIES_CACHE_SIZE = 3, SERIES_MAX_POINTS = 1500, LEGENDRE_DELAY_MS = 350;
  const BAR_MINUTES = { 375: 1, 75: 5, 25: 15, 7: 60, 1: 375 };
  const COL = { val: '#4f8ef7', train: '#f5c842', dim: '#6b7599', fit: '#f76f6f', actual: '#9aa3c7', good: '#3dd68c' };

  const $ = id => document.getElementById(id);
  const store = {
    get: k => { try { return localStorage.getItem(k); } catch { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } },
  };
  const num = (x, d = 3) => (x === null || x === undefined || !isFinite(x)) ? '—' : Number(x).toFixed(d);
  const charts = {};
  let gran = store.get('cp_gran') || '15min', info = null, config = null;
  let seriesToken = 0, legToken = 0, legTimer = null;
  const seriesData = new Map();

  async function getJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    const d = await r.json().catch(() => null);
    if (!r.ok || !d) throw new Error((d && d.error) || 'the cache is read by the ops server: start it with ./serve.sh (scripts/tools/ops_server.jl)');
    return d;
  }

  // ── company series (copy of the live training page's card) ──
  async function loadSeries(sym, g) {
    const key = `${g}/${sym}`;
    if (seriesData.has(key)) return seriesData.get(key);
    const d = await getJSON(`/api/series?granularity=${encodeURIComponent(g)}&symbol=${encodeURIComponent(sym)}`);
    const bpd = d.bars_per_day, step = BAR_MINUTES[bpd] || 1, t = [];
    d.dates.forEach(day => {
      for (let b = 0; b < bpd; b++) {
        const m = 555 + b * step;
        t.push(bpd === 1 ? day : `${day} ${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`);
      }
    });
    const out = { t, bpd, logret: d.logret, traded: d.traded, price: d.price || null };
    seriesData.set(key, out);
    if (seriesData.size > SERIES_CACHE_SIZE) seriesData.delete(seriesData.keys().next().value);
    return out;
  }

  const periodOf = label => {
    const d = label.slice(0, 10), p = config?.dates || {};
    return d >= p.train_start && d <= p.train_end ? 'train' : d >= p.val_start && d <= p.val_end ? 'val' : 'other';
  };
  const PERIOD_COL = { train: COL.train, val: COL.val, other: COL.dim };

  function darkScales(extra = {}) {
    return { x: { ticks: { maxTicksLimit: 10, color: '#9aa3c7', callback(v) { return String(this.getLabelForValue(v)).slice(0, 10); } }, grid: { color: '#2a314f' } },
             y: { ticks: { color: '#9aa3c7' }, grid: { color: '#2a314f' }, ...extra } };
  }
  function drawChart(id, cfg) {
    if (charts[id]) { charts[id].destroy(); delete charts[id]; }
    charts[id] = new Chart($(id), cfg);
  }

  function setSeriesSpan(span) {
    const p = config?.dates || {}, last = info?.last_date, first = info?.first_date;
    let a = '', b = '';
    if (span === 'train') [a, b] = [p.train_start, p.train_end];
    else if (span === 'val') [a, b] = [p.val_start, p.val_end];
    else if (span === '1y' || span === '2y') {
      if (last) { const d = new Date(last + 'T00:00:00Z'); d.setUTCFullYear(d.getUTCFullYear() - (span === '2y' ? 2 : 1)); a = d.toISOString().slice(0, 10); if (first && a < first) a = first; b = last; }
    }
    $('cp-s-from').value = a || ''; $('cp-s-to').value = b || '';
    renderAll();
  }

  async function renderSeries() {
    const sym = $('cp-s-sym').value, note = $('cp-s-note'), token = ++seriesToken;
    if (charts.series) { charts.series.destroy(); delete charts.series; }
    if (!sym) { note.textContent = ''; return; }
    const kind = $('cp-s-kind').value;
    note.textContent = 'loading…';
    let s;
    try { s = await loadSeries(sym, gran); } catch (e) { if (token === seriesToken) note.textContent = e.message; return; }
    if (token !== seriesToken) return;
    const from = $('cp-s-from').value, to = $('cp-s-to').value;
    let lo = from ? s.t.findIndex(x => x.slice(0, 10) >= from) : 0, hi = s.t.length - 1;
    if (to) while (hi >= 0 && s.t[hi].slice(0, 10) > to) hi--;
    if (lo < 0 || hi < lo) { note.textContent = 'no bars in this range'; return; }
    if (kind === 'price' && !s.price) { note.textContent = `this ${gran} cache holds returns only, no prices: rebuild it (build_forecaster_cache.jl --granularity ${gran}) or pick another series`; return; }
    let cum = 0;
    for (let i = 0; i < lo; i++) cum += s.logret[i];
    const n = hi - lo + 1, step = Math.ceil(n / SERIES_MAX_POINTS), labels = [], data = [];
    for (let i = lo; i <= hi; i += step) {
      const end = Math.min(i + step - 1, hi);
      let ret = 0, traded = 0;
      for (let j = i; j <= end; j++) {
        cum += s.logret[j]; traded += s.traded[j];
        if (j % s.bpd !== 0 || s.bpd === 1) ret += s.logret[j];
      }
      labels.push(s.t[end]);
      data.push(kind === 'price' ? s.price[end] : kind === 'cum' ? cum : kind === 'ret' ? ret : traded / (end - i + 1));
    }
    const role = $('cp-s-sym').selectedOptions[0]?.dataset.role || '';
    const how = { price: 'close of the last bar of each point, as in the OHLCV file (not adjusted here; forward-filled over bars with no trade)', cum: 'cumulative from the start of the range', ret: 'overnight bars left out, as in the targets', traded: 'share of bars traded' }[kind];
    note.textContent = `${role ? role + ' company · ' : ''}${gran} · ${n.toLocaleString()} bars${step > 1 ? `, ${step} per point` : ''} · ${how}`;
    const colors = labels.map(l => PERIOD_COL[periodOf(l)]), bar = kind !== 'cum' && kind !== 'price';
    charts.series = new Chart($('cp-c-series'), {
      type: bar ? 'bar' : 'line',
      data: { labels, datasets: [{
        label: { price: 'close price ₹', cum: 'cumulative return %', ret: 'return %', traded: 'traded share' }[kind], data, borderWidth: bar ? 0 : 1.5, pointRadius: 0, tension: 0,
        backgroundColor: colors, borderColor: COL.dim, segment: { borderColor: c => colors[c.p1DataIndex] } }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false },
                 plugins: { legend: { display: false } }, scales: darkScales() },
    });
  }

  // ── Legendre projection (computed by the ops server from legendre_projection.jl) ──
  const tile = (label, value, sub, cls = '') => `<div class="fcl-tile ${cls}"><div class="fcl-tl">${label}</div><div class="fcl-tv">${value}</div><div class="fcl-ts">${sub}</div></div>`;

  function legendreError(msg) {
    $('cp-leg-note').innerHTML = `<span class="cp-err">${msg}</span>`;
    $('cp-tiles').innerHTML = '';
    ['fit', 'res', 'curve', 'coef'].forEach(k => { const id = 'cp-c-' + k; if (charts[id]) { charts[id].destroy(); delete charts[id]; } });
  }

  async function renderLegendre() {
    const sym = $('cp-s-sym').value, token = ++legToken, n = parseInt($('cp-n').value, 10);
    if (!sym) return;
    if (!(n >= 1)) return legendreError('N must be a whole number of at least 1');
    const kind = $('cp-leg-kind').value;
    $('cp-leg-note').textContent = 'fitting…';
    const q = new URLSearchParams({ granularity: gran, symbol: sym, n, kind, from: $('cp-s-from').value, to: $('cp-s-to').value });
    let d;
    try { d = await getJSON('/api/legendre?' + q); } catch (e) { if (token === legToken) legendreError(e.message); return; }
    if (token !== legToken) return;
    const m = d.metrics, thinned = m.stride > 1;
    $('cp-leg-note').textContent = `${sym} · ${gran} · N = ${d.n} · ${m.points.toLocaleString()} bars${thinned ? ` (every ${m.stride}th used for the fit)` : ''}`;
    const r2 = m.r2;
    $('cp-tiles').innerHTML =
      tile('R²', num(r2, 4), 'share of the series variance kept', r2 !== null && r2 > 0.99 ? 'good' : '') +
      tile('RMS error', '₹' + num(m.rmse, 2), num(m.rmse_pct, 2) + '% of the mean price') +
      tile('largest error', '₹' + num(m.max_abs, 2), num(m.max_abs_pct, 2) + '% of the mean price') +
      tile('return variance kept', m.return_variance_kept === null ? '—' : num(100 * m.return_variance_kept, 2) + '%', `of the ${thinned ? 'sampled-step' : 'bar-to-bar'} log-return variance`, m.return_variance_kept !== null && m.return_variance_kept < 0.2 ? 'warn' : '') +
      tile('step correlation', num(m.step_corr, 3), 'fit returns against actual returns');

    const labels = d.t;
    drawChart('cp-c-fit', { type: 'line', data: { labels, datasets: [
      { label: 'close price ₹', data: d.actual, borderColor: COL.actual, borderWidth: 1.2, pointRadius: 0, tension: 0 },
      { label: `Legendre fit, N = ${d.n}`, data: d.fit, borderColor: COL.fit, borderWidth: 2, pointRadius: 0, tension: 0 } ] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false },
                 plugins: { legend: { labels: { boxWidth: 12, color: '#9aa3c7' } } }, scales: darkScales() } });
    const res = d.actual.map((a, i) => +(a - d.fit[i]).toFixed(2));
    drawChart('cp-c-res', { type: 'line', data: { labels, datasets: [{ label: 'price − fit ₹', data: res, borderColor: COL.val, borderWidth: 1, pointRadius: 0, tension: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: false } }, scales: darkScales() } });

    const ks = d.error_curve.map(p => p[0]), es = d.error_curve.map(p => p[1]);
    drawChart('cp-c-curve', { type: 'line', data: { labels: ks, datasets: [
      { label: 'RMS error, % of mean price', data: es, borderColor: COL.val, borderWidth: 1.5, pointRadius: 0, tension: 0 },
      { label: `N = ${d.n}`, data: ks.map(k => k === d.n ? es[k - 1] : null), borderColor: COL.fit, backgroundColor: COL.fit, pointRadius: 5, showLine: false } ] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false },
                 plugins: { legend: { labels: { boxWidth: 12, color: '#9aa3c7' } } },
                 scales: { x: { title: { display: true, text: 'number of terms', color: '#6b7280' }, ticks: { maxTicksLimit: 10, color: '#9aa3c7' }, grid: { color: '#2a314f' } },
                           y: { type: 'logarithmic', ticks: { color: '#9aa3c7' }, grid: { color: '#2a314f' } } } } });

    const contrib = d.coefficients.map((c, k) => Math.abs(c) / Math.sqrt(2 * k + 1));
    drawChart('cp-c-coef', { type: 'bar', data: { labels: contrib.map((_, k) => k), datasets: [{ label: 'RMS contribution ₹', data: contrib,
      backgroundColor: d.coefficients.map(c => c >= 0 ? COL.val : COL.train), borderWidth: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false },
                   tooltip: { callbacks: { label: c => `k=${c.dataIndex}: c=${d.coefficients[c.dataIndex].toFixed(3)} · contribution ₹${c.parsed.y.toFixed(3)}` } } },
                 scales: { x: { title: { display: true, text: 'degree k (blue: c ≥ 0, yellow: c < 0)', color: '#6b7280' }, ticks: { maxTicksLimit: 12, color: '#9aa3c7' }, grid: { display: false } },
                           y: { type: 'logarithmic', ticks: { color: '#9aa3c7' }, grid: { color: '#2a314f' } } } } });
  }

  function renderAll() {
    renderSeries();
    clearTimeout(legTimer);
    legTimer = setTimeout(renderLegendre, LEGENDRE_DELAY_MS);
  }

  // ── company list, bar size, run info ──
  async function loadBarSize() {
    $('cp-gran').innerHTML = GRANS.map(([label, g]) => `<button class="mode-tab${g === gran ? ' active' : ''}" data-g="${g}">${label}</button>`).join('');
    $('cp-gran').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { gran = b.dataset.g; store.set('cp_gran', gran); loadBarSize(); }));
    $('cp-s-note').textContent = 'loading the company list…';
    try { info = await getJSON(`/api/symbols?granularity=${gran}`); }
    catch (e) { info = null; $('cp-s-sym').innerHTML = ''; $('cp-s-note').textContent = e.message; legendreError(e.message); return; }
    config = null;
    try { const r = await fetch(`${RUN_DIR[gran]}/run_config.json?t=${Date.now()}`, { cache: 'no-store' }); if (r.ok) config = await r.json(); } catch { /* no run of this bar size */ }
    const train = new Set(config?.train_companies || []), val = new Set(config?.val_companies || []);
    const sorted = [...info.symbols].sort((a, b) => a.localeCompare(b));
    const opt = s => `<option value="${s}" data-role="${val.has(s) ? 'val' : train.has(s) ? 'train' : ''}">${s}</option>`;
    const v = sorted.filter(s => val.has(s)), t = sorted.filter(s => train.has(s)), o = sorted.filter(s => !val.has(s) && !train.has(s));
    $('cp-s-sym').innerHTML = (v.length ? `<optgroup label="Val companies (${v.length})">${v.map(opt).join('')}</optgroup>` : '')
      + (t.length ? `<optgroup label="Train companies (${t.length})">${t.map(opt).join('')}</optgroup>` : '')
      + `<optgroup label="Other (${o.length})">${o.map(opt).join('')}</optgroup>`;
    const keep = store.get('cp_sym');
    if (keep && sorted.includes(keep)) $('cp-s-sym').value = keep;
    ['cp-span-train', 'cp-span-val'].forEach(id => { $(id).style.display = config?.dates ? '' : 'none'; });
    setSeriesSpan('2y');
  }

  function wire() {
    $('cp-n-presets').innerHTML = PRESETS.map(n => `<button class="mode-tab" data-n="${n}">${n}</button>`).join('');
    $('cp-n-presets').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { $('cp-n').value = b.dataset.n; store.set('cp_n', b.dataset.n); renderLegendre(); }));
    $('cp-n').addEventListener('input', () => { store.set('cp_n', $('cp-n').value); clearTimeout(legTimer); legTimer = setTimeout(renderLegendre, LEGENDRE_DELAY_MS); });
    $('cp-leg-kind').addEventListener('change', () => { store.set('cp_leg_kind', $('cp-leg-kind').value); renderLegendre(); });
    $('cp-s-sym').addEventListener('change', () => { store.set('cp_sym', $('cp-s-sym').value); renderAll(); });
    ['cp-s-kind', 'cp-s-from', 'cp-s-to'].forEach(id => $(id).addEventListener('change', renderAll));
    $('cp-series').querySelectorAll('[data-span]').forEach(b => b.addEventListener('click', () => setSeriesSpan(b.dataset.span)));
    if (store.get('cp_n')) $('cp-n').value = store.get('cp_n');
    if (store.get('cp_leg_kind')) $('cp-leg-kind').value = store.get('cp_leg_kind');
  }

  wire();
  loadBarSize();
})();
