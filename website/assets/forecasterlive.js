/* Live view of a v4 forecaster run (scripts/forecaster/train_minute_forecaster.jl), used by tradinggamelive.html.
   Reads, from one run directory (default data/forecaster):
     live_status.json  rewritten every 10 steps: state, epoch, step, running loss, ETA
     epoch_log.jsonl   one row per epoch: train objective and every evaluation set's metrics
     run_config.json   model, schedule, dates, companies
     timing.json       seconds per phase for each epoch
     val_report.json   written at the end: per-stock / per-slot results of the best checkpoint
     memory_profile.json  with memory_profile = true: GPU memory per epoch and its summary, rewritten after every epoch */
(() => {
  const DEFAULT_RUN = 'data/forecaster';
  const RUN_TABS = [['1 min', 'data/forecaster', ''], ['15 min', 'data/forecaster_15min', ' --granularity 15min']];
  const STOP_ACTIONS = { 'data/forecaster': ['stop_forecaster_1min', 'stop_forecaster_1min_now'], 'data/forecaster_15min': ['stop_forecaster_15min', 'stop_forecaster_15min_now'] };
  const COL = { val: '#4f8ef7', val_train_companies: '#3dd68c', train: '#f5c842', objective: '#f76f6f', dim: '#6b7599', white: '#e6e9f5' };
  const SET_LABEL = { val: 'val companies, val dates', val_train_companies: 'train companies, val dates', train: 'train companies, train dates' };
  const charts = {};
  let runDir = DEFAULT_RUN, built = false, mode = 'game';
  let epochs = [], timing = [], live = null, config = null, report = null, reportKey = '', memory = null, memoryKey = '', nllUnits = 'raw';

  const $ = id => document.getElementById(id);
  const store = {
    get: k => { try { return localStorage.getItem(k); } catch { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } },
  };
  const num = (x, d = 4) => (x === null || x === undefined || !isFinite(x)) ? '—' : Number(x).toFixed(d);
  const pct = (x, d = 1) => (x === null || x === undefined || !isFinite(x)) ? '—' : (100 * x).toFixed(d) + '%';
  const dur = s => {
    if (s === null || s === undefined || !isFinite(s)) return '—';
    s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), r = s % 60;
    return h ? `${h}h ${String(m).padStart(2, '0')}m` : m ? `${m}m ${String(r).padStart(2, '0')}s` : `${r}s`;
  };

  async function getJSON(name) {
    const r = await fetch(`${runDir}/${name}?t=${Date.now()}`, { cache: 'no-store' });
    if (!r.ok) throw new Error(name);
    return r.json();
  }
  async function getJSONL(name) {
    const r = await fetch(`${runDir}/${name}?t=${Date.now()}`, { cache: 'no-store' });
    if (!r.ok) throw new Error(name);
    return (await r.text()).split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
  }

  // ── chart helpers ────────────────────────────────────────────────────────
  // Series hidden through the legend, by chart id then label: each poll rebuilds the datasets, which would show them again.
  const hiddenSeries = {};
  function legendToggle(id) {
    return (e, item, legend) => {
      const hide = new Set(hiddenSeries[id] || []);
      hide.has(item.text) ? hide.delete(item.text) : hide.add(item.text);
      hiddenSeries[id] = hide;
      Chart.defaults.plugins.legend.onClick(e, item, legend);
    };
  }
  function lineChart(id, labels, datasets, o = {}) {
    const cv = $(id); if (!cv) return;
    const data = { labels, datasets: datasets.map(d => ({ pointRadius: 2.5, borderWidth: 2, tension: 0.15, spanGaps: false, ...d, hidden: (hiddenSeries[id] || new Set()).has(d.label) })) };
    if (charts[id] && charts[id].canvas === cv) { charts[id].data = data; charts[id].$rounds = o.rounds || []; charts[id].update('none'); return; }
    if (charts[id]) { charts[id].destroy(); delete charts[id]; }
    charts[id] = new Chart(cv, {
      type: o.type || 'line', data, plugins: [roundMarks],
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: datasets.length > 1, onClick: legendToggle(id), labels: { boxWidth: 12, color: '#9aa3c7' } },
                   tooltip: { callbacks: o.tooltip ? { label: o.tooltip } : {} } },
        scales: { x: { title: { display: !!o.xTitle, text: o.xTitle || '' }, stacked: !!o.stacked, grid: { color: '#2a314f' } },
                  y: { title: { display: !!o.yTitle, text: o.yTitle || '' }, stacked: !!o.stacked, grid: { color: '#2a314f' }, ...(o.y || {}) } },
      },
    });
  }

  // ── epoch window ─────────────────────────────────────────────────────────
  const WIN_PRESETS = [10, 25, 50, 100];
  let win = { from: null, to: null, last: null };

  function winBounds() {
    const nums = epochs.map(r => r.epoch);
    const lo = Math.min(...nums), hi = Math.max(...nums);
    let from = win.from ?? lo, to = win.to ?? hi;
    if (win.last) { to = hi; from = Math.max(lo, hi - win.last + 1); }
    return { lo, hi, from, to };
  }
  function windowed(rows, key) {
    if (!epochs.length) return rows;
    const { from, to } = winBounds();
    return rows.filter(r => key(r) >= from && key(r) <= to);
  }
  function renderWindowUi() {
    const box = $('fcl-win'); if (!box || !epochs.length) return;
    const { lo, hi, from, to } = winBounds();
    for (const [id, v] of [['fcl-win-from', from], ['fcl-win-to', to]]) {
      const el = $(id); el.min = lo; el.max = hi;
      if (document.activeElement !== el) el.value = v;
    }
    box.querySelectorAll('[data-last]').forEach(b => b.classList.toggle('active', win.last === (+b.dataset.last || null) && (win.last !== null || (win.from === null && win.to === null))));
    $('fcl-win-note').textContent = `showing epochs ${from}–${to} of ${lo}–${hi}`;
  }
  function wireWindowUi() {
    const apply = () => {
      const a = parseInt($('fcl-win-from').value, 10), b = parseInt($('fcl-win-to').value, 10);
      win = { from: isFinite(a) ? a : null, to: isFinite(b) ? b : null, last: null };
      if (win.from !== null && win.to !== null && win.from > win.to) [win.from, win.to] = [win.to, win.from];
      renderCharts();
    };
    $('fcl-win-from').addEventListener('change', apply);
    $('fcl-win-to').addEventListener('change', apply);
    $('fcl-win').querySelectorAll('[data-last]').forEach(b => b.addEventListener('click', () => {
      const n = +b.dataset.last;
      win = { from: null, to: null, last: n || null };
      renderCharts();
    }));
  }

  // ── skeleton ─────────────────────────────────────────────────────────────
  function build() {
    if (built) return; built = true;
    $('fc-root').innerHTML = `
      <div class="chart-card" style="padding:12px 16px">
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
          <div class="mode-tabs" id="fcl-run-tabs" style="margin:0"></div>
          <span id="fcl-stop" style="display:none;margin-left:auto;gap:8px;align-items:center">
  <span id="fcl-run-note" style="font-size:12px;color:var(--text-dim)"></span>
  <button class="mode-tab" id="fcl-stop-epoch">Stop after this epoch</button>
  <button class="mode-tab" id="fcl-stop-now" style="color:var(--red)">Stop now</button>
</span>
        </div>
      </div>
      <div class="chart-card" id="fcl-win" style="padding:10px 16px">
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px">
          <span class="chart-title" style="font-size:14px">Epoch window</span>
          <label>from <input type="number" id="fcl-win-from" step="1" style="width:76px;background:#12162a;color:#e6e9f5;border:1px solid #2a314f;border-radius:6px;padding:3px 6px"></label>
          <label>to <input type="number" id="fcl-win-to" step="1" style="width:76px;background:#12162a;color:#e6e9f5;border:1px solid #2a314f;border-radius:6px;padding:3px 6px"></label>
          <span class="mode-tabs" style="margin:0">
            <button class="mode-tab" data-last="0">All</button>
            ${WIN_PRESETS.map(n => `<button class="mode-tab" data-last="${n}">Last ${n}</button>`).join('')}
          </span>
          <span id="fcl-win-note" style="font-size:12px;color:var(--text-dim)"></span>
        </div>
      </div>
      <div id="fcl-placeholder" class="placeholder" style="display:none">
        No v4 run found in <b id="fcl-ph-dir"></b>. Start one:
        <code>julia --project=packages/TradingGame scripts/forecaster/train_minute_forecaster.jl --config configs/bse_v3.toml --device gpu<span id="fcl-ph-flag"></span></code>
      </div>
      <div id="fcl-body">
        <div class="chart-card" id="fcl-status"></div>
        <div class="chart-card" id="fcl-config"></div>
        <div class="section-title">Latest evaluation <span id="fcl-latest-epoch" style="font-weight:400;color:var(--text-dim);font-size:12px"></span></div>
        <div class="fcl-tiles" id="fcl-tiles"></div>
        <div id="fcl-rounds"></div>
        <div class="section-title">Training progress</div>
        <div class="fcl-grid">
          <div class="chart-card" style="grid-column:1/-1"><div class="chart-header"><div class="chart-title">Likelihood gain of the predicted scale over a baseline σ</div><div class="chart-sub">nats per target · independent of the return scale, so comparable between rounds · above 0 = the scale head adds information · dashed vertical lines = a new round</div></div><div class="canvas-wrap" style="height:300px"><canvas id="fcl-c-gain"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Negative log-likelihood per epoch</div><div class="chart-sub"><select id="fcl-nll-units" class="fcl-input" style="padding:1px 4px"><option value="raw">raw return units (NLL + ln scale)</option><option value="logged">as logged (in units of the cache scale)</option></select> · <span id="fcl-nll-note">lower is better</span></div></div><div class="canvas-wrap"><canvas id="fcl-c-nll"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Forecast skill over predicting zero</div><div class="chart-sub">1 − MSE(μ)/MSE(0) · 0 = no better than no move</div></div><div class="canvas-wrap"><canvas id="fcl-c-skill"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Calibration of the predicted intervals (val)</div><div class="chart-sub">share of targets inside the model's own 68% / 95% interval · dashed = nominal</div></div><div class="canvas-wrap"><canvas id="fcl-c-cov"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Sign hit rate</div><div class="chart-sub">direction of μ against the next return, over targets that moved · 50% = coin</div></div><div class="canvas-wrap"><canvas id="fcl-c-hit"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Where the time goes</div><div class="chart-sub">seconds per epoch by phase</div></div><div class="canvas-wrap"><canvas id="fcl-c-time"></canvas></div></div>
        </div>
        <div id="fcl-report"></div>
        <div class="chart-card" id="fcl-series">
          <div class="chart-header"><div class="chart-title">Company series</div>
            <div class="chart-sub">one train or val company of this run, read from the forecaster cache (the log-returns the model sees) through the ops server · line colour = period (train / val)</div></div>
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px;margin-bottom:8px">
            <select id="fcl-s-sym" class="fcl-input" style="min-width:190px"></select>
            <select id="fcl-s-kind" class="fcl-input">
              <option value="price">Close price, ₹</option><option value="cum">Cumulative return, 100·Σ ln(c/c₋₁) %</option><option value="ret">Return per bar, 100·ln(c/c₋₁) %</option><option value="traded">Traded share of bars</option>
            </select>
            <label>from <input type="date" id="fcl-s-from" class="fcl-input"></label>
            <label>to <input type="date" id="fcl-s-to" class="fcl-input"></label>
            <span class="mode-tabs" style="margin:0">
              <button class="mode-tab" data-span="train">Train period</button>
              <button class="mode-tab" data-span="val">Val period</button>
              <button class="mode-tab" data-span="all">All</button>
            </span>
            <span id="fcl-s-note" style="font-size:12px;color:var(--text-dim)"></span>
          </div>
          <div class="canvas-wrap" style="height:300px"><canvas id="fcl-c-series"></canvas></div>
        </div>
        <details class="fcl-gloss"><summary>What every plotted variable means</summary>
<div class="section-note" style="margin-top:0">
        These are the quantities on this tab, in the order it shows them.
        All of them come from <code>epoch_log.jsonl</code>, <code>live_status.json</code>, <code>run_config.json</code>, <code>timing.json</code>, <code>val_report.json</code> and <code>memory_profile.json</code> in the run directory.
      </div>
<div class="met-h">Evaluation sets (every metric below is computed on each of these after every epoch)</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>val</td><td>The validation companies on the validation dates. <b>This is the headline set</b>: it decides which checkpoint is kept (lowest val NLL). Both the companies and the dates were never seen in training, with a gap of one window between the train and val periods.</td></tr><tr><td>val_train_companies</td><td>The <em>training</em> companies on the <em>validation</em> dates. Compared with <code>val</code> it isolates the cost of new companies; compared with <code>train</code> it isolates the cost of new dates.</td></tr><tr><td>train</td><td>The training companies on the training dates, scored with dropout off. The gap to <code>val</code> is the overfitting gap.</td></tr><tr><td>target</td><td>One (stock, minute) next-minute return. Only <em>traded</em> targets count (the stock had volume in that minute) and the first bar of a day is skipped, because the overnight gap is not a one-minute move. Every ratio below is over these targets.</td></tr><tr><td>return units</td><td>The model works on returns divided by the <em>return scale</em> (the RMS of traded one-minute returns in %, shown in the run configuration). Absolute NLL and MSE therefore depend on the cache; compare <em>gains</em> and <em>skill</em>, which are ratios and differences, across runs.</td></tr></table>
<div class="met-h">Latest evaluation tiles and the charts under "Training progress"</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>val NLL / "Negative log-likelihood per epoch"</td><td>The mean negative log-probability, in nats, that the predicted Student-t (location μ, scale σ, ν degrees of freedom) assigns to the realised next return: <code>NLL = −ln p(y | μ, σ, ν)</code>, averaged over targets. Lower is better. It scores the location and the scale together, so it is the number that picks the best epoch. The dashed "training objective" line on the chart is a different quantity: the same NLL multiplied by σ<sup>2β</sup> (β-NLL, held constant), which changes as σ moves, so it is not comparable across epochs.</td></tr><tr><td>gain over const σ</td><td><code>NLL(const σ) − NLL(model)</code>, in nats per target. The baseline predicts a location of zero and one constant σ per stock (its RMS return). A positive gain means the model's predicted location and, mainly, its time-varying scale add information over "this stock is always this volatile". The baseline σ is clamped to the range the model's own σ can take, so it is not handicapped.</td></tr><tr><td>gain over EWMA σ</td><td>The same difference against a harder baseline: zero location and a σ from an exponentially weighted average of recent squared returns (decay 0.94 per minute, the classic RiskMetrics volatility). Beating it means the model knows more about the next minute's volatility than simple volatility clustering.</td></tr><tr><td>forecast skill / "Forecast skill over predicting zero"</td><td><code>1 − MSE(μ) / MSE(0)</code>, where MSE(μ) is the squared error of the predicted location and MSE(0) is the error of predicting no move. 0 means no better than zero; positive means the location carries real information; negative means worse than saying nothing. One-minute returns are close to a coin flip in the mean, so a few percent is already meaningful.</td></tr><tr><td>repeat-last</td><td>The same skill formula for the baseline "the next return equals the last return". It is strongly negative (bid-ask bounce makes a minute tend to partly reverse the previous one), so repeating the last move is worse than predicting zero.</td></tr><tr><td>sign hit rate</td><td>Among targets where the price actually moved (<code>y ≠ 0</code>), the share where <code>sign(μ) = sign(y)</code>. 50% is a coin. Targets that did not move are excluded, because there is no direction to get right.</td></tr><tr><td>68% / 95% interval ("Calibration")</td><td>The share of targets that fall inside the model's own central 68% (95%) interval around μ, computed from the Student-t with its predicted σ and ν. A calibrated model scores about 68% and 95% (the dashed lines). Above nominal means the intervals are too wide (under-confident), below means too narrow (over-confident).</td></tr><tr><td>timestamp</td><td>The accuracy of the timestamp head: <code>minute</code> is the share of next-minute-of-day predictions within a tolerance of the truth, <code>gap</code> the same for the log day-gap, <code>weekday</code> for the weekday. It is a calendar rule plus a small learned correction, so about 100% is expected and says the head is working, not that the market is predictable.</td></tr></table>
<div class="met-h">Run status and configuration cards</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>state, epoch, step</td><td><code>starting</code>, <code>training</code>, <code>evaluating</code> or <code>finished</code>. An "epoch" is <code>steps_per_epoch</code> random batches, not a pass over the data. "not updating" means <code>live_status.json</code> has not been rewritten for two minutes.</td></tr><tr><td>time left / speed</td><td>Speed is seconds per optimiser step over recent steps (batch × window × N tokens per step). Time left is the remaining steps times that speed.</td></tr><tr><td>running train loss</td><td>The mean training objective over the steps of the current epoch (β-weighted NLL plus 0.1 × the timestamp MSE). Use it to see whether training is alive; use the evaluation tiles to judge the model.</td></tr><tr><td>learning rate</td><td>AdamW with a linear warm-up for <code>warmup_steps</code>, then a cosine decay to zero at the last step.</td></tr><tr><td>best so far</td><td>The epoch with the lowest val NLL. Its weights are what <code>forecaster.bson</code> holds and what the final report scores.</td></tr><tr><td>skipped steps / loss scale</td><td>Only used with <code>dtype = "float16"</code>: a step whose gradients overflowed is skipped and the loss scale lowered. Always 0 / 1 for float32.</td></tr><tr><td>window M, width d, heads, time blocks, dropout</td><td>The model: minutes of history per window, token width, attention heads, causal time-attention blocks L, dropout. See the layer cards on the <a href="architecture.html">Architecture</a> page (Forecaster_v4 tab).</td></tr><tr><td>batch, steps / epoch, epochs, peak lr, weight decay, warm-up, seed</td><td>The optimiser settings. The parameter count at the top of the card matches the parameter breakdown on the <a href="architecture.html">Architecture</a> page.</td></tr><tr><td>Student-t ν, β-NLL</td><td>The degrees of freedom of the output distribution (smaller = heavier tails) and the exponent β of the σ<sup>2β</sup> weight on each NLL term. β = 0 is the plain NLL; β = 0.5 stops the model lowering the loss just by inflating σ where its mean is wrong.</td></tr><tr><td>return scale</td><td>The traded-minute RMS return, in %, that every return is divided by before the model sees it.</td></tr><tr><td>train / val window, companies</td><td>The date ranges and the number of companies on each side. Their symbol lists are in <code>run_config.json</code>.</td></tr></table>
<div class="met-h">Timing chart ("Where the time goes")</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>batch</td><td>Building a batch of windows from the cache (indexing and feature assembly on the CPU).</td></tr><tr><td>forward_backward</td><td>The model forward pass and gradient computation. It dominates, because attention scores grow as M² per stock and N² per minute.</td></tr><tr><td>optimizer</td><td>The AdamW update, including gradient clipping.</td></tr><tr><td>eval</td><td>Scoring the three evaluation sets once at the end of the epoch.</td></tr></table>
<div class="met-h">Final report (appears when the run ends; best checkpoint, val set)</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>Predicted vs realised volatility by time of day</td><td>For each 15-minute slot of the session, the mean predicted standard deviation (<code>σ·√(ν/(ν−2))</code>, the t's own std) against the realised RMS of the returns. The shape (high at the open, low mid-session, a rise into the close) should match. The predicted line sits below the realised one because heavy tails put a few very large moves into the RMS, so a typical-scale σ is smaller than an RMS.</td></tr><tr><td>Forecast skill by time of day</td><td>The skill formula above, computed separately within each 15-minute slot. It shows whether the edge is concentrated at the open or close or spread through the session.</td></tr><tr><td>Forecast skill by stock</td><td>The same skill per validation company, sorted. Bars below zero are stocks where the model does worse than predicting no move. A stock with too few traded targets has no value. High skill on thinly traded names usually reflects microstructure effects (bounce, stale quotes), so check the traded fraction before trusting it.</td></tr></table>
<div class="met-h">Rounds and the NLL units</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>round</td><td>One run of <code>train_minute_forecaster.jl</code> (a <code>--repeat</code> round, or a restart). It redraws the training companies, rebuilds the cache and restarts the learning-rate schedule, so numbers from different rounds are not on the same footing. Dashed vertical lines on the epoch charts mark where one starts; the <b>Rounds</b> table lists each round's epochs, date windows, return scale, company set (a short hash) and averages. Rounds logged before the run information was recorded are told apart by the step counter restarting and show — for the rest.</td></tr><tr><td>NLL in raw return units</td><td>The logged NLL is measured on returns divided by the cache's return scale, and that scale changes with the training companies. Adding ln(scale) gives the NLL of the return itself (in %), which does not depend on the scale. Rows without a recorded scale are gaps in that view; "as logged" shows every row.</td></tr><tr><td>Likelihood gain (top chart)</td><td>The NLL of the baseline σ (a constant per stock, or an EWMA of recent returns) minus the model's NLL, in nats per target. Both are in the same units, so the gain does not depend on the return scale and can be compared between rounds.</td></tr></table>
<div class="met-h">Memory usage (bottom of the tab; needs <code>memory_profile = true</code>; updated after every epoch)</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>peak in training / typical training peak</td><td>The highest number of bytes of the GPU memory pool in use during one training step (one extra step is run per epoch from a collected pool, same batch size, no update), and the median of that over the epochs. Training memory is set by the batch (or its piece, with micro-batches), the window, the number of stocks and the width, so it is almost flat from epoch to epoch.</td></tr><tr><td>peak in evaluation</td><td>The same during the epoch's evaluation of the three sets (no gradients are kept, so it is usually lower).</td></tr><tr><td>held from the device</td><td>The size of the memory pool: what is in use plus what has been freed but kept for reuse. This is the figure <code>nvidia-smi</code> shows, and it is always at least the peak.</td></tr><tr><td>speed</td><td>Seconds per training step over the epoch's training phase (batches, forward and backward passes, update).</td></tr><tr><td>change over the run</td><td>The training peak averaged over the last five epochs minus the first five. A steady rise would point to memory that is not released; it is highlighted when it exceeds 5% of the typical peak.</td></tr><tr><td>pool high-water mark (chart)</td><td>The highest pool use over the epoch's real training steps. It sits near the card's size because garbage from earlier steps is freed only when Julia's collector runs, and a collection takes time, which is why real steps can be slower than the one-step profile shows.</td></tr><tr><td>host memory</td><td>The process's peak resident set in RAM (the cache, the batches, the Julia runtime), not GPU memory.</td></tr></table>
        </details>
        <div id="fcl-memory"></div>
       </div>`;
    wireWindowUi();
    wireSeriesUi();
    $('fcl-nll-units').addEventListener('change', e => { nllUnits = e.target.value; store.set('fcl_nll_units', nllUnits); renderCharts(); });
    nllUnits = store.get('fcl_nll_units') || 'raw'; $('fcl-nll-units').value = nllUnits;
  }

  // ── company series (read from the forecaster cache by the ops server, /api/series) ──
  const SERIES_CACHE_SIZE = 3, SERIES_MAX_POINTS = 1500;
  const BAR_MINUTES = { 375: 1, 75: 5, 25: 15, 7: 60, 1: 375 };
  const seriesData = new Map();
  let seriesCompanies = '', seriesToken = 0;

  async function loadSeries(sym, gran) {
    const key = `${gran}/${sym}`;
    if (seriesData.has(key)) return seriesData.get(key);
    const r = await fetch(`/api/series?granularity=${encodeURIComponent(gran)}&symbol=${encodeURIComponent(sym)}`);
    const d = await r.json().catch(() => null);
    if (!r.ok || !d) throw new Error((d && d.error) || 'the cache is read by the ops server: start scripts/tools/ops_server.jl');
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

  function fillSeriesCompanies() {
    const train = config?.train_companies || [], val = config?.val_companies || [];
    const sig = train.join() + '|' + val.join();
    if (sig === seriesCompanies) return;
    seriesCompanies = sig;
    const sel = $('fcl-s-sym'), keep = sel.value;
    const opts = (list, role) => [...list].sort((a, b) => a.localeCompare(b)).map(s => `<option value="${s}" data-role="${role}">${s}</option>`).join('');
    sel.innerHTML = `<optgroup label="Val companies (${val.length})">${opts(val, 'val')}</optgroup><optgroup label="Train companies (${train.length})">${opts(train, 'train')}</optgroup>`;
    if (keep && [...sel.options].some(o => o.value === keep)) sel.value = keep;
    setSeriesSpan('val');
  }

  function setSeriesSpan(span) {
    const p = config?.dates || {};
    const [a, b] = span === 'train' ? [p.train_start, p.train_end] : span === 'val' ? [p.val_start, p.val_end] : ['', ''];
    $('fcl-s-from').value = a || ''; $('fcl-s-to').value = b || '';
    renderSeries();
  }

  async function renderSeries() {
    const sym = $('fcl-s-sym').value, note = $('fcl-s-note'), token = ++seriesToken;
    if (charts.series) { charts.series.destroy(); delete charts.series; }
    if (!sym || !config) { note.textContent = ''; return; }
    const kind = $('fcl-s-kind').value, gran = config.model?.granularity || '1min';
    note.textContent = 'loading…';
    let s;
    try { s = await loadSeries(sym, gran); } catch (e) { if (token === seriesToken) note.textContent = e.message; return; }
    if (token !== seriesToken) return;
    const from = $('fcl-s-from').value, to = $('fcl-s-to').value;
    let lo = from ? s.t.findIndex(x => x.slice(0, 10) >= from) : 0, hi = s.t.length - 1;
    if (to) while (hi >= 0 && s.t[hi].slice(0, 10) > to) hi--;
    if (lo < 0 || hi < lo) { note.textContent = 'no bars in this range'; return; }
    if (kind === 'price' && !s.price) { note.textContent = `this ${gran} cache holds returns only, no prices: rebuild it (build_forecaster_cache.jl --granularity ${gran}, or the next pipeline round) or pick another series`; return; }
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
    const role = $('fcl-s-sym').selectedOptions[0]?.dataset.role || '';
    const how = { price: 'close of the last bar of each point, as in the OHLCV file (not adjusted here; forward-filled over bars with no trade)', cum: 'cumulative from the start of the range', ret: 'overnight bars left out, as in the targets', traded: 'share of bars traded' }[kind];
    note.textContent = `${role} company · ${gran} · ${n.toLocaleString()} bars${step > 1 ? `, ${step} per point` : ''} · ${how}`;
    const colors = labels.map(l => PERIOD_COL[periodOf(l)]), bar = kind !== 'cum' && kind !== 'price';
    charts.series = new Chart($('fcl-c-series'), {
      type: bar ? 'bar' : 'line',
      data: { labels, datasets: [{
        label: { price: 'close price ₹', cum: 'cumulative return %', ret: 'return %', traded: 'traded share' }[kind], data, borderWidth: bar ? 0 : 1.5, pointRadius: 0, tension: 0,
        backgroundColor: colors, borderColor: COL.dim, segment: { borderColor: c => colors[c.p1DataIndex] } }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: false } },
        scales: { x: { ticks: { maxTicksLimit: 10, callback(v) { return this.getLabelForValue(v).slice(0, 10); } }, grid: { color: '#2a314f' } },
                  y: { grid: { color: '#2a314f' } } },
      },
    });
  }

  function wireSeriesUi() {
    ['fcl-s-sym', 'fcl-s-kind', 'fcl-s-from', 'fcl-s-to'].forEach(id => $(id).addEventListener('change', renderSeries));
    $('fcl-series').querySelectorAll('[data-span]').forEach(b => b.addEventListener('click', () => setSeriesSpan(b.dataset.span)));
  }

  function renderRunTabs() {
    const tabs = RUN_TABS.some(t => t[1] === runDir) ? RUN_TABS : [...RUN_TABS, [runDir.replace(/^data\//, ''), runDir, '']];
    $('fcl-run-tabs').innerHTML = tabs.map(t => `<button class="mode-tab${t[1] === runDir ? ' active' : ''}" data-run="${t[1]}" title="${t[1]}">${t[0]}</button>`).join('');
    $('fcl-run-tabs').querySelectorAll('button').forEach(b => b.addEventListener('click', () => setRun(b.dataset.run)));
  renderStopButtons();
}

async function renderStopButtons() {
  const box = $('fcl-stop'), names = STOP_ACTIONS[runDir];
  const ready = names && window.Ops && await Ops.available();
  box.style.display = ready ? 'flex' : 'none';
  if (!ready) return;
  const note = $('fcl-run-note');
  const pending = (await Ops.refresh() || {});
  note.textContent = pending[names[1]]?.exists ? 'STOP_NOW file is set' : pending[names[0]]?.exists ? 'STOP file is set: stops after this epoch' : '';
  [[$('fcl-stop-epoch'), names[0], 'Stop this run after the current epoch? The best checkpoint is kept.'],
   [$('fcl-stop-now'), names[1], 'Stop this run immediately? The best checkpoint is already saved, the current epoch is lost.']].forEach(([btn, name, msg]) => {
    btn.onclick = async () => {
      if (!confirm(msg)) return;
      const r = await Ops.run(name);
      note.textContent = r.ok ? (name.endsWith('_now') ? 'STOP_NOW file written' : 'STOP file written: stops after this epoch') : 'failed: ' + (r.body.error || r.status);
    };
  });
}

  function setRun(dir) {
    runDir = dir; store.set('fcl_run', dir);
    Object.values(charts).forEach(c => c.destroy()); for (const k in charts) delete charts[k];
    seriesCompanies = '';
    epochs = []; timing = []; live = null; config = null; report = null; memory = null; memoryKey = ''; roundsKey = ''; win = { from: null, to: null, last: null };
    renderRunTabs(); renderAll(); pollSlow(); pollLive();
  }

  // ── renderers ────────────────────────────────────────────────────────────
  function renderStatus() {
    const el = $('fcl-status');
    if (!live) { el.innerHTML = '<div class="chart-title">Run status</div><div class="chart-sub" style="margin-top:6px">no live_status.json yet</div>'; return; }
    const age = (Date.now() - new Date(live.updated).getTime()) / 1000;
    const active = ['training', 'evaluating', 'starting'].includes(live.state);
    const stalled = active && age > 120;
    const label = stalled ? 'not updating' : live.state;
    const colour = stalled ? '#f76f6f' : live.state === 'training' ? '#3dd68c' : live.state === 'evaluating' ? '#4f8ef7' : live.state === 'finished' ? '#9aa3c7' : '#f5c842';
    const pe = live.steps_per_epoch ? live.step_in_epoch / live.steps_per_epoch : 0;
    const pt = live.steps_total ? live.step / live.steps_total : 0;
    const bar = (p, c) => `<div style="height:8px;background:var(--surface-2,#1d2036);border-radius:4px;overflow:hidden"><div style="height:100%;width:${Math.min(100, 100 * p).toFixed(1)}%;background:${c}"></div></div>`;
    el.innerHTML = `
      <div class="chart-header"><div class="chart-title">Run status</div>
        <div class="chart-sub"><span style="color:${colour};font-weight:600">● ${label}</span> · epoch ${live.epoch} of ${live.epochs_total} · step ${live.step.toLocaleString()} of ${live.steps_total.toLocaleString()} · updated ${dur(age)} ago</div></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:8px 0 12px">
        <div><div style="font-size:11px;color:var(--text-dim);margin-bottom:4px">this epoch · ${live.step_in_epoch} / ${live.steps_per_epoch} steps</div>${bar(pe, '#4f8ef7')}</div>
        <div><div style="font-size:11px;color:var(--text-dim);margin-bottom:4px">whole run · ${(100 * pt).toFixed(1)}%</div>${bar(pt, '#3dd68c')}</div>
      </div>
      <div class="fcl-tiles">
        ${tile('time left', dur(live.eta_secs), `elapsed ${dur(live.elapsed_secs)}`)}
        ${tile('speed', live.secs_per_step ? num(live.secs_per_step, 2) + ' s/step' : '—', live.secs_per_step ? num(1 / live.secs_per_step, 1) + ' steps/s' : '')}
        ${tile('running train loss', num(live.running_train_loss, 4), 'mean this epoch')}
        ${tile('learning rate', live.lr ? live.lr.toExponential(2) : '—', 'warm-up then cosine')}
        ${tile('best so far', live.best_epoch ? `epoch ${live.best_epoch}` : '—', live.best_loss !== null && isFinite(live.best_loss) ? 'val NLL ' + num(live.best_loss, 4) : '')}
        ${tile('skipped steps', live.skipped_steps, `loss scale ${live.loss_scale} · ${live.dtype}`, live.skipped_steps > 0 ? 'warn' : '')}
      </div>`;
  }

  function tile(label, value, sub = '', cls = '') {
    return `<div class="fcl-tile ${cls}"><div class="fcl-tl">${label}</div><div class="fcl-tv">${value}</div><div class="fcl-ts">${sub}</div></div>`;
  }

  function renderConfig() {
    const el = $('fcl-config');
    if (!config) { el.innerHTML = ''; return; }
    const m = config.model || {}, t = config.training || {}, d = config.dates || {};
    const item = (k, v) => `<div class="fcl-item"><span>${k}</span><b>${v}</b></div>`;
    el.innerHTML = `
      <div class="chart-header"><div class="chart-title">Run configuration</div>
        <div class="chart-sub">${(m.parameters || 0).toLocaleString()} parameters · ${config.loss || 'huber'} objective · ${config.dtype || 'float32'} on ${config.resolved_device || '—'}${config.resumed ? ' · resumed' : ''}</div></div>
      <div class="fcl-items">
        ${item('window M', m.window)}${item('width d', m.d)}${item('heads', m.heads)}${item('time blocks', m.blocks)}${item('dropout', m.dropout)}
        ${item('batch', t.batch)}${item('steps / epoch', t.steps_per_epoch)}${item('epochs', t.epochs)}${item('peak lr', t.lr)}${item('weight decay', t.weight_decay)}${item('warm-up', t.warmup_steps)}
        ${item('Student-t ν', config.nu ?? '—')}${item('β-NLL', config.nll_beta ?? '—')}${item('seed', t.seed ?? '—')}${item('return scale', config.return_scale_percent ? num(config.return_scale_percent, 3) + '%' : '—')}
        ${item('train window', `${d.train_start} → ${d.train_end}`)}${item('val window', `${d.val_start} → ${d.val_end}`)}
        ${item('train companies', (config.train_companies || []).length)}${item('val companies', (config.val_companies || []).length)}
      </div>`;
  }

  function renderTiles() {
    const el = $('fcl-tiles');
    const last = epochs[epochs.length - 1];
    $('fcl-latest-epoch').textContent = last ? `· epoch ${last.epoch}${last.best ? ' (best so far)' : ''}` : '';
    if (!last) { el.innerHTML = ''; return; }
    const v = (last.eval || {}).val || {}, huber = (last.objective || 'student_t') === 'huber';
    const g1 = v.nll_const != null && v.nll != null ? v.nll_const - v.nll : null, g2 = v.nll_ewma != null && v.nll != null ? v.nll_ewma - v.nll : null;
    el.innerHTML = [
      tile('val NLL', num(v.nll, 4), huber ? 'huber run: no likelihood' : 'nats per target'),
      tile('gain over const σ', num(g1, 4), 'one σ per stock', g1 > 0 ? 'good' : ''),
      tile('gain over EWMA σ', num(g2, 4), 'recent-returns σ', g2 > 0 ? 'good' : ''),
      tile('forecast skill', pct(v.skill, 2), `repeat-last ${pct(v.skill_last, 0)}`, v.skill > 0 ? 'good' : 'warn'),
      tile('sign hit rate', pct(v.hit_rate, 1), '50% = coin'),
      tile('68% interval', pct(v.cov68, 1), 'nominal 68%'),
      tile('95% interval', pct(v.cov95, 1), 'nominal 95%'),
      tile('timestamp', `${pct(v.minute_acc, 1)}`, `gap ${pct(v.gap_acc, 1)} · weekday ${pct(v.weekday_acc, 1)}`),
    ].join('');
  }

  function renderCharts() {
    if (!epochs.length) return;
    const E = windowed(epochs, r => r.epoch), T = windowed(timing, r => r.iteration);
    renderWindowUi();
    const labels = E.map(r => r.epoch);
    const sets = ['val', 'val_train_companies', 'train'];
    const ev = (r, s, k) => ((r.eval || {})[s] || {})[k] ?? null;
    const setLine = (s, k, extra = {}) => ({ label: SET_LABEL[s], data: E.map(r => ev(r, s, k)), borderColor: COL[s], backgroundColor: COL[s], ...extra });
    const marks = roundStarts(epochs).filter(e => labels.includes(e));
    const rawMode = nllUnits === 'raw' && E.some(r => r.run?.scale);
    const nllNote = $('fcl-nll-note');
    if (nllNote) nllNote.textContent = rawMode ? 'lower is better · rows logged before the scale was recorded are gaps' : 'lower is better · the unit changes with the training companies, so rounds are not comparable · the training objective is weighted, so it sits apart';
    lineChart('fcl-c-nll', labels, [
      ...sets.map(s => rawMode ? { ...setLine(s, 'nll'), data: E.map(r => nllRawOf(r, ev(r, s, 'nll'))) } : setLine(s, 'nll')),
      ...(rawMode ? [] : [{ label: 'training objective (β-weighted)', data: E.map(r => r.train_return_loss ?? null), borderColor: COL.objective, backgroundColor: COL.objective, borderDash: [5, 4] }]),
    ], { rounds: marks, xTitle: 'epoch', y: { beginAtZero: false } });
    lineChart('fcl-c-skill', labels, sets.map(s => setLine(s, 'skill')), { rounds: marks, xTitle: 'epoch', tooltip: c => `${c.dataset.label}: ${(100 * c.parsed.y).toFixed(2)}%`, y: { ticks: { callback: v => (100 * v).toFixed(1) + '%' } } });
    const gain = (s, base) => E.map(r => { const v = (r.eval || {})[s] || {}; return v.nll != null && v[base] != null ? v[base] - v.nll : null; });
    lineChart('fcl-c-gain', labels, [
      { label: 'val companies, over constant σ per stock', data: gain('val', 'nll_const'), borderColor: COL.val, backgroundColor: COL.val },
      { label: 'val companies, over EWMA σ', data: gain('val', 'nll_ewma'), borderColor: COL.val, backgroundColor: COL.val, borderDash: [5, 4] },
      { label: 'train companies on val dates, over constant σ', data: gain('val_train_companies', 'nll_const'), borderColor: COL.val_train_companies, backgroundColor: COL.val_train_companies },
      { label: 'train companies on train dates, over constant σ', data: gain('train', 'nll_const'), borderColor: COL.train, backgroundColor: COL.train, borderDash: [2, 3] },
    ], { rounds: marks, xTitle: 'epoch', yTitle: 'nats per target', y: { suggestedMin: 0 } });
    const nominal = (v, c) => ({ label: `nominal ${v}`, data: E.map(() => v), borderColor: c, borderDash: [4, 4], pointRadius: 0, borderWidth: 1 });
    lineChart('fcl-c-cov', labels, [
      { label: '68% interval', data: E.map(r => ev(r, 'val', 'cov68')), borderColor: COL.val, backgroundColor: COL.val },
      nominal(0.68, COL.dim),
      { label: '95% interval', data: E.map(r => ev(r, 'val', 'cov95')), borderColor: COL.train, backgroundColor: COL.train },
      nominal(0.95, COL.dim),
    ], { rounds: marks, xTitle: 'epoch', y: { suggestedMin: 0.6, suggestedMax: 1 } });
    lineChart('fcl-c-hit', labels, [...sets.map(s => setLine(s, 'hit_rate')), nominal(0.5, COL.dim)], { rounds: marks, xTitle: 'epoch', y: { ticks: { callback: v => (100 * v).toFixed(0) + '%' } } });
    if (T.length) {
      const names = [...new Set(T.flatMap(r => (r.phases || []).map(p => p.name)))];
      const pal = { batch: '#f5c842', forward_backward: '#4f8ef7', optimizer: '#3dd68c', eval: '#f76f6f' };
      lineChart('fcl-c-time', T.map(r => r.iteration), names.map(n => ({
        label: n, data: T.map(r => ((r.phases || []).find(p => p.name === n) || {}).secs ?? 0), backgroundColor: pal[n] || '#9aa3c7', borderColor: pal[n] || '#9aa3c7', borderWidth: 0,
      })), { type: 'bar', stacked: true, xTitle: 'epoch', yTitle: 'seconds' });
    }
  }

  function renderReport() {
    const el = $('fcl-report');
    if (!report) { el.innerHTML = ''; reportKey = ''; return; }
    const key = JSON.stringify(report.val ? [report.val.n_targets, report.val.nll, report.val.skill_vs_zero] : []);
    if (key === reportKey && el.childElementCount) return;
    reportKey = key;
    const names = Object.keys(report), huber = names.every(n => report[n].distribution === 'huber');
    const row = n => { const r = report[n]; return `<tr><td>${SET_LABEL[n] || n}</td><td>${r.n_targets.toLocaleString()}</td><td>${num(r.nll, 4)}</td><td>${num(r.nll_gain_vs_const_sigma, 4)}</td><td>${num(r.nll_gain_vs_ewma_sigma, 4)}</td><td>${pct(r.coverage_68, 1)}</td><td>${pct(r.coverage_95, 1)}</td><td>${pct(r.skill_vs_zero, 2)}</td><td>${pct(r.skill_repeat_last_vs_zero, 0)}</td><td>${pct(r.sign_hit_rate, 1)}</td></tr>`; };
    el.innerHTML = `
      <div class="section-title">Final report <span style="font-weight:400;color:var(--text-dim);font-size:12px">· best checkpoint, written when the run ends</span></div>
      <div class="chart-card" style="overflow-x:auto"><table class="fcl-table"><tr><th>evaluation set</th><th>targets</th><th>NLL</th><th>gain const σ</th><th>gain EWMA σ</th><th>68%</th><th>95%</th><th>skill</th><th>repeat-last</th><th>hit</th></tr>${names.map(row).join('')}</table></div>
      <div class="fcl-grid">
        <div class="chart-card"><div class="chart-header"><div class="chart-title">Predicted vs realised volatility by time of day (val)</div><div class="chart-sub">standard deviation per 15-minute slot · heavy tails put realised RMS above the t's own std</div></div><div class="canvas-wrap"><canvas id="fcl-c-slotstd"></canvas></div></div>
        <div class="chart-card"><div class="chart-header"><div class="chart-title">Forecast skill by time of day (val)</div><div class="chart-sub">1 − MSE/MSE(0) per slot</div></div><div class="canvas-wrap"><canvas id="fcl-c-slotskill"></canvas></div></div>
        <div class="chart-card" style="grid-column:1/-1"><div class="chart-header"><div class="chart-title">Forecast skill by stock (val companies)</div><div class="chart-sub">sorted · how consistent the edge is across names</div></div><div class="canvas-wrap" style="height:260px"><canvas id="fcl-c-stock"></canvas></div></div>
      </div>`;
    const v = report.val; if (!v) return;
    const slots = (v.std_by_slot || []).map(s => `${String(Math.floor((555 + s.from_minute) / 60)).padStart(2, '0')}:${String((555 + s.from_minute) % 60).padStart(2, '0')}`);
    if (!huber) lineChart('fcl-c-slotstd', slots, [
      { label: 'predicted std', data: (v.std_by_slot || []).map(s => s.predicted_std), borderColor: COL.val, backgroundColor: COL.val },
      { label: 'realised RMS', data: (v.std_by_slot || []).map(s => s.realised_rms), borderColor: COL.train, backgroundColor: COL.train },
    ], { yTitle: 'return units' });
    lineChart('fcl-c-slotskill', (v.per_slot_skill || []).map(s => slots[Math.round(s.from_minute / 15)] ?? s.from_minute), [
      { label: 'skill', data: (v.per_slot_skill || []).map(s => s.skill), backgroundColor: COL.val, borderWidth: 0 }], { type: 'bar', y: { ticks: { callback: x => (100 * x).toFixed(1) + '%' } } });
    const stocks = Object.entries(v.per_stock_skill || {}).filter(([, s]) => s !== null).sort((a, b) => b[1] - a[1]);
    lineChart('fcl-c-stock', stocks.map(s => s[0]), [{ label: 'skill', data: stocks.map(s => s[1]), backgroundColor: stocks.map(s => s[1] >= 0 ? COL.val_train_companies : COL.objective), borderWidth: 0 }],
      { type: 'bar', y: { ticks: { callback: x => (100 * x).toFixed(1) + '%' } } });
  }

  // ── memory usage (memory_profile.json, written when a run with memory_profile = true ends) ──
  const mib = x => (x === null || x === undefined || !isFinite(x)) ? '—' : x >= 1024 ? (x / 1024).toFixed(2) + ' GiB' : Math.round(x) + ' MiB';

  function renderMemory() {
    const el = $('fcl-memory'); if (!el) return;
    const title = '<div class="section-title">Memory usage <span style="font-weight:400;color:var(--text-dim);font-size:12px">· per epoch, updated after every epoch</span></div>';
    if (!memory) {
      memoryKey = '';
      el.innerHTML = title + `<div class="chart-card"><div class="chart-sub">No <code>memory_profile.json</code> in this run directory yet. Set <code>memory_profile = true</code> in the <code>[forecaster]</code> section of the config
        (or pass <code>--memory-profile</code>); the file appears after the first epoch and is updated after every epoch.</div></div>`;
      return;
    }
    const s = memory.summary || {}, c = memory.config || {}, total = memory.gpu_total_mib, gpu = memory.device === 'gpu';
    const key = JSON.stringify([(memory.epochs || []).length, win.from, win.to, win.last]);
    const tp = s.train_peak_mib, ep = s.eval_peak_mib, st = s.secs_per_step;
    const pctOf = x => total && x != null ? ` · ${(100 * x / total).toFixed(0)}% of the card` : '';
    const drift = s.train_peak_drift_mib, driftWarn = drift != null && tp && drift > 0.05 * tp.median;
    if (key !== memoryKey || !el.childElementCount) {
      memoryKey = key;
      const item = (k, v) => `<div class="fcl-item"><span>${k}</span><b>${v}</b></div>`;
      el.innerHTML = title + `
        <div class="chart-card">
          <div class="chart-header"><div class="chart-title">GPU memory</div>
            <div class="chart-sub">${gpu ? `${memory.gpu} · ${mib(total)} · ` : 'CPU run: no GPU figures · '}${s.epochs} epochs profiled · updated ${String(memory.written).replace('T', ' ').slice(0, 19)}</div></div>
          <div class="fcl-tiles">
            ${tile('peak in training', gpu && tp ? mib(tp.max) : '—', gpu && tp ? 'one step, highest epoch' + pctOf(tp.max) : 'needs --device gpu', tp && total && tp.max > 0.95 * total ? 'warn' : '')}
            ${tile('typical training peak', gpu && tp ? mib(tp.median) : '—', gpu && tp ? `min ${mib(tp.min)} · mean ${mib(tp.mean)}` : '')}
            ${tile('peak in evaluation', gpu && ep ? mib(ep.max) : '—', gpu && ep ? `median ${mib(ep.median)}` : '')}
            ${tile('held from the device', gpu && s.reserved_mib ? mib(s.reserved_mib.max) : '—', gpu && s.reserved_mib ? 'pool incl. freed blocks kept for reuse' : '')}
            ${tile('speed', st ? num(st.median, 3) + ' s/step' : '—', st ? `fastest ${num(st.min, 3)} · the first epoch includes compiling` : '')}
            ${tile('change over the run', gpu && drift != null ? (drift >= 0 ? '+' : '') + mib(drift) : '—', gpu && drift != null ? 'last 5 epochs vs first 5 (training peak)' : 'needs 10+ epochs', driftWarn ? 'warn' : '')}
            ${tile('host memory', s.host_rss_peak_mib != null ? mib(s.host_rss_peak_mib) : '—', 'peak resident set of the process')}
          </div>
          <div class="fcl-items" style="margin-bottom:10px">
            ${item('batch × pieces', `${c.batch} × ${c.micro_batches}`)}${item('window M', c.window)}${item('width d', c.d)}${item('time blocks', c.blocks)}
            ${item('stocks per window (train / val)', `${c.train_stocks} / ${c.val_stocks}`)}${item('dtype', c.dtype)}${item('bar size', c.granularity)}${item('steps / epoch', c.steps_per_epoch)}
          </div>
          ${gpu ? '<div class="canvas-wrap" style="height:260px"><canvas id="fcl-c-mem"></canvas></div>' : ''}
          <div class="chart-sub" style="margin-top:8px">The training peak is measured on one extra step per epoch (same batch size, from a collected pool, no update), so it is what a step needs. The pool high-water mark over the
            epoch's real steps is higher: Julia frees GPU arrays only when its collector runs, so garbage from earlier steps piles up until the card is nearly full, and every collection costs time.
            "Held from the device" is the pool's size, which is what <code>nvidia-smi</code> shows.</div>
        </div>`;
    }
    if (!gpu) return;
    const rows = windowed(memory.epochs || [], r => r.epoch);
    const labels = rows.map(r => r.epoch);
    const line = (label, k, color, extra = {}) => ({ label, data: rows.map(r => r[k] ?? null), borderColor: color, backgroundColor: color, ...extra });
    lineChart('fcl-c-mem', labels, [
      line('training peak (one step)', 'train_peak_mib', COL.val),
      line('pool high-water mark over the epoch', 'train_pool_peak_mib', COL.objective, { borderDash: [5, 4] }),
      line('evaluation peak', 'eval_peak_mib', COL.train),
      line('held from the device', 'reserved_mib', COL.dim, { borderDash: [2, 3] }),
      ...(total ? [{ label: 'card total', data: rows.map(() => total), borderColor: COL.white, borderDash: [2, 4], pointRadius: 0, borderWidth: 1 }] : []),
    ], { xTitle: 'epoch', yTitle: 'MiB', tooltip: c => `${c.dataset.label}: ${mib(c.parsed.y)}`, y: { beginAtZero: true } });
  }

  // ── rounds: one train_minute_forecaster.jl invocation each (a --repeat round) ─────────────────
  // New rows carry run.session_start_epoch; rows written before that are told apart by the step counter restarting.
  const roundMarks = {
    id: 'roundMarks',
    afterDatasetsDraw(chart) {
      const marks = chart.$rounds; if (!marks || !marks.length) return;
      const x = chart.scales.x, a = chart.chartArea, ctx = chart.ctx, labels = chart.data.labels;
      ctx.save(); ctx.strokeStyle = 'rgba(154,163,199,.35)'; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      marks.forEach(ep => {
        const i = labels.indexOf(ep); if (i < 0) return;
        const px = x.getPixelForValue(i); ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
      });
      ctx.restore();
    },
  };

  function roundStarts(rows) {
    const out = [];
    rows.forEach((r, i) => {
      if (!i) return;
      const p = rows[i - 1];
      if ((r.run?.session_start_epoch ?? null) !== (p.run?.session_start_epoch ?? null) || r.step <= p.step) out.push(r.epoch);
    });
    return out;
  }
  function splitRounds(rows) {
    const starts = new Set(roundStarts(rows)), groups = [];
    rows.forEach(r => { if (!groups.length || starts.has(r.epoch)) groups.push([]); groups[groups.length - 1].push(r); });
    return groups;
  }
  /** NLL in the units of the raw return (percent): the logged NLL is in units of the cache's return scale, which changes with the training companies. */
  const nllRawOf = (r, nll) => (nll != null && r.run?.scale > 0) ? nll + Math.log(r.run.scale) : null;
  const mean = a => { const v = a.filter(x => x != null && isFinite(x)); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null; };

  let roundsKey = '';
  function renderRounds() {
    const el = $('fcl-rounds'); if (!el) return;
    const groups = splitRounds(epochs);
    if (!groups.length) { el.innerHTML = ''; roundsKey = ''; return; }
    const key = `${epochs.length}:${epochs[epochs.length - 1].epoch}`;
    if (key === roundsKey && el.childElementCount) return;
    roundsKey = key;
    const wasOpen = el.querySelector('details')?.open ?? (groups.length > 1);
    const rows = groups.map((g, i) => {
      const first = g[0], last = g[g.length - 1], run = first.run || {}, d = run.dates || {};
      const v = r => (r.eval || {}).val || {};
      const best = Math.min(...g.map(r => nllRawOf(r, v(r).nll) ?? Infinity));
      const win = d.train_start ? `${d.train_start.slice(2)} → ${d.train_end.slice(2)}<br>val ${d.val_start.slice(2)} → ${d.val_end.slice(2)}` : '—';
      return `<tr${i === groups.length - 1 ? ' style="color:var(--text)"' : ''}><td>${i + 1}</td><td>${first.epoch}–${last.epoch}</td><td>${win}</td><td>${run.scale ? num(run.scale, 3) : '—'}</td>`
        + `<td>${run.train_set ? run.train_set.slice(0, 6) : '—'}</td><td>${isFinite(best) ? num(best, 3) : '—'}</td>`
        + `<td>${num(mean(g.map(r => v(r).nll_const != null && v(r).nll != null ? v(r).nll_const - v(r).nll : null)), 3)}</td>`
        + `<td>${num(mean(g.map(r => v(r).nll_ewma != null && v(r).nll != null ? v(r).nll_ewma - v(r).nll : null)), 3)}</td>`
        + `<td>${pct(mean(g.map(r => v(r).skill)), 2)}</td><td>${pct(mean(g.map(r => v(r).hit_rate)), 1)}</td></tr>`;
    }).join('');
    el.innerHTML = `<details class="fcl-gloss"${wasOpen ? ' open' : ''}><summary>Rounds (${groups.length}): the runs this log is made of</summary>
      <div class="section-note" style="margin-top:0">Each <code>--repeat</code> round (or restart) redraws the training companies and rebuilds the cache, so the numbers of different rounds are not measured on the same footing:
        compare rounds by the likelihood gains, which do not depend on the return scale, not by the raw logged NLL. Dashed lines on the charts below mark where a round starts. Rows logged before this
        table existed have no dates, scale or company set recorded (—).</div>
      <div style="overflow:auto;max-height:340px"><table class="fcl-table" style="font-size:12px"><tr><th>round</th><th>epochs</th><th>train window<br>val window</th><th>scale %</th><th>train set</th><th>best val NLL (raw)</th><th>gain const σ</th><th>gain EWMA σ</th><th>skill</th><th>hit</th></tr>${rows}</table></div>
      <div class="chart-sub" style="margin-top:6px">gains, skill and hit rate are means over the round's epochs on the val companies and val dates; NLL (raw) is the logged NLL + ln(return scale), the best epoch of the round.</div></details>`;
  }

  function renderAll() {
    const none = !live && !epochs.length && !config;
    $('fcl-placeholder').style.display = none ? '' : 'none'; $('fcl-ph-dir').textContent = runDir; $('fcl-ph-flag').textContent = (RUN_TABS.find(t => t[1] === runDir) || [0, 0, ''])[2];
    $('fcl-body').style.display = none ? 'none' : '';
    renderStatus(); renderConfig(); fillSeriesCompanies(); renderTiles(); renderCharts(); renderReport(); renderMemory(); renderRounds();
  }

  // ── polling ──────────────────────────────────────────────────────────────
  async function pollLive() {
    if (mode !== 'fc') return;
    try { live = await getJSON('live_status.json'); } catch { live = null; }
    renderAll();
  }
  async function pollSlow() {
    if (mode !== 'fc') return;
    try { epochs = await getJSONL('epoch_log.jsonl'); } catch { epochs = []; }
    try { timing = (await getJSON('timing.json')).runs || []; } catch { timing = []; }
    try { config = await getJSON('run_config.json'); } catch { config = null; }
    try { report = await getJSON('val_report.json'); } catch { report = null; }
    try { memory = await getJSON('memory_profile.json'); } catch { memory = null; }
    renderAll();
  }

  // ── page mode (game v1–v3 / forecaster v4) ───────────────────────────────
  const TEXT = {
    game: ['Live Training', 'Watch a PPO training run play out — portfolio value, current holdings, and every buy/sell as they happen'],
    fc: ['Live Training — Forecaster v4', 'Watch a next-minute forecaster train: likelihood, forecast skill, interval calibration and the predicted volatility, updated as the run goes'],
  };
  function setMode(m) {
    mode = m; store.set('live_mode', m);
    $('content').style.display = m === 'game' ? '' : 'none';
    $('fc-root').style.display = m === 'fc' ? '' : 'none';
    document.querySelectorAll('.mode-tab').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
    document.querySelector('.page-title').textContent = TEXT[m][0];
    document.querySelector('.page-subtitle').textContent = TEXT[m][1];
    if (m === 'fc') { build(); setRun(runDir); }
  }

  function init() {
    const q = new URLSearchParams(location.search);
    runDir = q.get('run') || store.get('fcl_run') || DEFAULT_RUN;
    document.querySelectorAll('.mode-tab').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
    setMode(q.get('mode') || store.get('live_mode') || 'game');
    const every = (fn, ms) => {
      setInterval(() => { if (!document.hidden) fn(); }, ms);
      document.addEventListener('visibilitychange', () => { if (!document.hidden) fn(); });
    };
    every(pollLive, 10000);
    every(pollSlow, 20000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
