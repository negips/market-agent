/* Live view of a v4 forecaster run (scripts/train_minute_forecaster.jl), used by tradinggamelive.html.
   Reads, from one run directory (default data/forecaster):
     live_status.json  rewritten every 10 steps: state, epoch, step, running loss, ETA
     epoch_log.jsonl   one row per epoch: train objective and every evaluation set's metrics
     run_config.json   model, schedule, dates, companies
     timing.json       seconds per phase for each epoch
     val_report.json   written at the end: per-stock / per-slot results of the best checkpoint */
(() => {
  const DEFAULT_RUN = 'data/forecaster';
  const RUN_CHOICES = ['data/forecaster', 'data/forecaster_sweep/d_32', 'data/forecaster_sweep/d_64', 'data/forecaster_sweep/d_128'];
  const COL = { val: '#4f8ef7', val_train_companies: '#3dd68c', train: '#f5c842', objective: '#f76f6f', dim: '#6b7599', white: '#e6e9f5' };
  const SET_LABEL = { val: 'val companies, val dates', val_train_companies: 'train companies, val dates', train: 'train companies, train dates' };
  const charts = {};
  let runDir = DEFAULT_RUN, built = false, mode = 'game';
  let epochs = [], timing = [], live = null, config = null, report = null, reportKey = '';

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
  function lineChart(id, labels, datasets, o = {}) {
    const cv = $(id); if (!cv) return;
    const data = { labels, datasets: datasets.map(d => ({ pointRadius: 2.5, borderWidth: 2, tension: 0.15, spanGaps: false, ...d })) };
    if (charts[id] && charts[id].canvas === cv) { charts[id].data = data; charts[id].update('none'); return; }
    if (charts[id]) { charts[id].destroy(); delete charts[id]; }
    charts[id] = new Chart(cv, {
      type: o.type || 'line', data,
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: datasets.length > 1, labels: { boxWidth: 12, color: '#9aa3c7' } },
                   tooltip: { callbacks: o.tooltip ? { label: o.tooltip } : {} } },
        scales: { x: { title: { display: !!o.xTitle, text: o.xTitle || '' }, stacked: !!o.stacked, grid: { color: '#2a314f' } },
                  y: { title: { display: !!o.yTitle, text: o.yTitle || '' }, stacked: !!o.stacked, grid: { color: '#2a314f' }, ...(o.y || {}) } },
      },
    });
  }

  // ── skeleton ─────────────────────────────────────────────────────────────
  function build() {
    if (built) return; built = true;
    $('fc-root').innerHTML = `
      <div class="chart-card" style="padding:12px 16px">
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
          <label style="font-size:12px;color:var(--text-dim)">Run directory</label>
          <input id="fcl-run" list="fcl-runs" style="min-width:300px;background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:6px 10px;font-family:var(--mono);font-size:12.5px">
          <datalist id="fcl-runs">${RUN_CHOICES.map(r => `<option value="${r}">`).join('')}</datalist>
          <span id="fcl-run-note" style="font-size:12px;color:var(--text-dim)"></span>
        </div>
      </div>
      <div id="fcl-placeholder" class="placeholder" style="display:none">
        No v4 run found in <b id="fcl-ph-dir"></b>. Start one:
        <code>julia --project=packages/TradingGame scripts/train_minute_forecaster.jl --config configs/bse_v3.toml --device gpu</code>
      </div>
      <div id="fcl-body">
        <div class="chart-card" id="fcl-status"></div>
        <div class="chart-card" id="fcl-config"></div>
        <div class="section-title">Latest evaluation <span id="fcl-latest-epoch" style="font-weight:400;color:var(--text-dim);font-size:12px"></span></div>
        <div class="fcl-tiles" id="fcl-tiles"></div>
        <div class="section-title">Training progress</div>
        <div class="fcl-grid">
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Negative log-likelihood per epoch</div><div class="chart-sub">lower is better · the training objective is weighted, so it sits apart</div></div><div class="canvas-wrap"><canvas id="fcl-c-nll"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Forecast skill over predicting zero</div><div class="chart-sub">1 − MSE(μ)/MSE(0) · 0 = no better than no move</div></div><div class="canvas-wrap"><canvas id="fcl-c-skill"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Likelihood gain of the predicted scale (val)</div><div class="chart-sub">nats per target over each baseline σ · above 0 = the scale head adds information</div></div><div class="canvas-wrap"><canvas id="fcl-c-gain"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Calibration of the predicted intervals (val)</div><div class="chart-sub">share of targets inside the model's own 68% / 95% interval · dashed = nominal</div></div><div class="canvas-wrap"><canvas id="fcl-c-cov"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Sign hit rate</div><div class="chart-sub">direction of μ against the next return, over targets that moved · 50% = coin</div></div><div class="canvas-wrap"><canvas id="fcl-c-hit"></canvas></div></div>
          <div class="chart-card"><div class="chart-header"><div class="chart-title">Where the time goes</div><div class="chart-sub">seconds per epoch by phase</div></div><div class="canvas-wrap"><canvas id="fcl-c-time"></canvas></div></div>
        </div>
        <div id="fcl-report"></div>
        <details class="fcl-gloss"><summary>What every plotted variable means</summary>
<div class="section-note" style="margin-top:0">
        These are the quantities on this tab, in the order it shows them.
        All of them come from <code>epoch_log.jsonl</code>, <code>live_status.json</code>, <code>run_config.json</code>, <code>timing.json</code> and <code>val_report.json</code> in the run directory.
      </div>
<div class="met-h">Evaluation sets (every metric below is computed on each of these after every epoch)</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>val</td><td>The validation companies on the validation dates. <b>This is the headline set</b>: it decides which checkpoint is kept (lowest val NLL). Both the companies and the dates were never seen in training, with a gap of one window between the train and val periods.</td></tr><tr><td>val_train_companies</td><td>The <em>training</em> companies on the <em>validation</em> dates. Compared with <code>val</code> it isolates the cost of new companies; compared with <code>train</code> it isolates the cost of new dates.</td></tr><tr><td>train</td><td>The training companies on the training dates, scored with dropout off. The gap to <code>val</code> is the overfitting gap.</td></tr><tr><td>target</td><td>One (stock, minute) next-minute return. Only <em>traded</em> targets count (the stock had volume in that minute) and the first bar of a day is skipped, because the overnight gap is not a one-minute move. Every ratio below is over these targets.</td></tr><tr><td>return units</td><td>The model works on returns divided by the <em>return scale</em> (the RMS of traded one-minute returns in %, shown in the run configuration). Absolute NLL and MSE therefore depend on the cache; compare <em>gains</em> and <em>skill</em>, which are ratios and differences, across runs.</td></tr></table>
<div class="met-h">Latest evaluation tiles and the charts under "Training progress"</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>val NLL / "Negative log-likelihood per epoch"</td><td>The mean negative log-probability, in nats, that the predicted Student-t (location μ, scale σ, ν degrees of freedom) assigns to the realised next return: <code>NLL = −ln p(y | μ, σ, ν)</code>, averaged over targets. Lower is better. It scores the location and the scale together, so it is the number that picks the best epoch. The dashed "training objective" line on the chart is a different quantity: the same NLL multiplied by σ<sup>2β</sup> (β-NLL, held constant), which changes as σ moves, so it is not comparable across epochs.</td></tr><tr><td>gain over const σ</td><td><code>NLL(const σ) − NLL(model)</code>, in nats per target. The baseline predicts a location of zero and one constant σ per stock (its RMS return). A positive gain means the model's predicted location and, mainly, its time-varying scale add information over "this stock is always this volatile". The baseline σ is clamped to the range the model's own σ can take, so it is not handicapped.</td></tr><tr><td>gain over EWMA σ</td><td>The same difference against a harder baseline: zero location and a σ from an exponentially weighted average of recent squared returns (decay 0.94 per minute, the classic RiskMetrics volatility). Beating it means the model knows more about the next minute's volatility than simple volatility clustering.</td></tr><tr><td>forecast skill / "Forecast skill over predicting zero"</td><td><code>1 − MSE(μ) / MSE(0)</code>, where MSE(μ) is the squared error of the predicted location and MSE(0) is the error of predicting no move. 0 means no better than zero; positive means the location carries real information; negative means worse than saying nothing. One-minute returns are close to a coin flip in the mean, so a few percent is already meaningful.</td></tr><tr><td>repeat-last</td><td>The same skill formula for the baseline "the next return equals the last return". It is strongly negative (bid-ask bounce makes a minute tend to partly reverse the previous one), so repeating the last move is worse than predicting zero.</td></tr><tr><td>sign hit rate</td><td>Among targets where the price actually moved (<code>y ≠ 0</code>), the share where <code>sign(μ) = sign(y)</code>. 50% is a coin. Targets that did not move are excluded, because there is no direction to get right.</td></tr><tr><td>68% / 95% interval ("Calibration")</td><td>The share of targets that fall inside the model's own central 68% (95%) interval around μ, computed from the Student-t with its predicted σ and ν. A calibrated model scores about 68% and 95% (the dashed lines). Above nominal means the intervals are too wide (under-confident), below means too narrow (over-confident).</td></tr><tr><td>timestamp</td><td>The accuracy of the timestamp head: <code>minute</code> is the share of next-minute-of-day predictions within a tolerance of the truth, <code>gap</code> the same for the log day-gap, <code>weekday</code> for the weekday. It is a calendar rule plus a small learned correction, so about 100% is expected and says the head is working, not that the market is predictable.</td></tr></table>
<div class="met-h">Run status and configuration cards</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>state, epoch, step</td><td><code>starting</code>, <code>training</code>, <code>evaluating</code> or <code>finished</code>. An "epoch" is <code>steps_per_epoch</code> random batches, not a pass over the data. "not updating" means <code>live_status.json</code> has not been rewritten for two minutes.</td></tr><tr><td>time left / speed</td><td>Speed is seconds per optimiser step over recent steps (batch × window × N tokens per step). Time left is the remaining steps times that speed.</td></tr><tr><td>running train loss</td><td>The mean training objective over the steps of the current epoch (β-weighted NLL plus 0.1 × the timestamp MSE). Use it to see whether training is alive; use the evaluation tiles to judge the model.</td></tr><tr><td>learning rate</td><td>AdamW with a linear warm-up for <code>warmup_steps</code>, then a cosine decay to zero at the last step.</td></tr><tr><td>best so far</td><td>The epoch with the lowest val NLL. Its weights are what <code>forecaster.bson</code> holds and what the final report scores.</td></tr><tr><td>skipped steps / loss scale</td><td>Only used with <code>dtype = "float16"</code>: a step whose gradients overflowed is skipped and the loss scale lowered. Always 0 / 1 for float32.</td></tr><tr><td>window M, width d, heads, time blocks, dropout</td><td>The model: minutes of history per window, token width, attention heads, causal time-attention blocks L, dropout. See the layer cards on the <a href="architecture.html">Architecture</a> page (Forecaster_v4 tab).</td></tr><tr><td>batch, steps / epoch, epochs, peak lr, weight decay, warm-up, seed</td><td>The optimiser settings. The parameter count at the top of the card matches the parameter breakdown on the <a href="architecture.html">Architecture</a> page.</td></tr><tr><td>Student-t ν, β-NLL</td><td>The degrees of freedom of the output distribution (smaller = heavier tails) and the exponent β of the σ<sup>2β</sup> weight on each NLL term. β = 0 is the plain NLL; β = 0.5 stops the model lowering the loss just by inflating σ where its mean is wrong.</td></tr><tr><td>return scale</td><td>The traded-minute RMS return, in %, that every return is divided by before the model sees it.</td></tr><tr><td>train / val window, companies</td><td>The date ranges and the number of companies on each side. Their symbol lists are in <code>run_config.json</code>.</td></tr></table>
<div class="met-h">Timing chart ("Where the time goes")</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>batch</td><td>Building a batch of windows from the cache (indexing and feature assembly on the CPU).</td></tr><tr><td>forward_backward</td><td>The model forward pass and gradient computation. It dominates, because attention scores grow as M² per stock and N² per minute.</td></tr><tr><td>optimizer</td><td>The AdamW update, including gradient clipping.</td></tr><tr><td>eval</td><td>Scoring the three evaluation sets once at the end of the epoch.</td></tr></table>
<div class="met-h">Final report (appears when the run ends; best checkpoint, val set)</div><table class="met-tbl"><tr><th>Shown as</th><th>Definition and how to read it</th></tr><tr><td>Predicted vs realised volatility by time of day</td><td>For each 15-minute slot of the session, the mean predicted standard deviation (<code>σ·√(ν/(ν−2))</code>, the t's own std) against the realised RMS of the returns. The shape (high at the open, low mid-session, a rise into the close) should match. The predicted line sits below the realised one because heavy tails put a few very large moves into the RMS, so a typical-scale σ is smaller than an RMS.</td></tr><tr><td>Forecast skill by time of day</td><td>The skill formula above, computed separately within each 15-minute slot. It shows whether the edge is concentrated at the open or close or spread through the session.</td></tr><tr><td>Forecast skill by stock</td><td>The same skill per validation company, sorted. Bars below zero are stocks where the model does worse than predicting no move. A stock with too few traded targets has no value. High skill on thinly traded names usually reflects microstructure effects (bounce, stale quotes), so check the traded fraction before trusting it.</td></tr></table>
        </details>
      </div>`;
    const inp = $('fcl-run'); inp.value = runDir;
    inp.addEventListener('change', () => { setRun(inp.value.trim().replace(/\/+$/, '') || DEFAULT_RUN); });
  }

  function setRun(dir) {
    runDir = dir; store.set('fcl_run', dir);
    Object.values(charts).forEach(c => c.destroy()); for (const k in charts) delete charts[k];
    epochs = []; timing = []; live = null; config = null; report = null;
    $('fcl-run').value = dir; renderAll(); pollSlow(); pollLive();
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
    const labels = epochs.map(r => r.epoch);
    const sets = ['val', 'val_train_companies', 'train'];
    const ev = (r, s, k) => ((r.eval || {})[s] || {})[k] ?? null;
    const setLine = (s, k, extra = {}) => ({ label: SET_LABEL[s], data: epochs.map(r => ev(r, s, k)), borderColor: COL[s], backgroundColor: COL[s], ...extra });
    lineChart('fcl-c-nll', labels, [
      ...sets.map(s => setLine(s, 'nll')),
      { label: 'training objective (β-weighted)', data: epochs.map(r => r.train_return_loss ?? null), borderColor: COL.objective, backgroundColor: COL.objective, borderDash: [5, 4] },
    ], { xTitle: 'epoch', y: { beginAtZero: false } });
    lineChart('fcl-c-skill', labels, sets.map(s => setLine(s, 'skill')), { xTitle: 'epoch', tooltip: c => `${c.dataset.label}: ${(100 * c.parsed.y).toFixed(2)}%`, y: { ticks: { callback: v => (100 * v).toFixed(1) + '%' } } });
    lineChart('fcl-c-gain', labels, [
      { label: 'over constant σ per stock', data: epochs.map(r => { const v = (r.eval || {}).val || {}; return v.nll != null && v.nll_const != null ? v.nll_const - v.nll : null; }), borderColor: COL.val, backgroundColor: COL.val },
      { label: 'over EWMA σ', data: epochs.map(r => { const v = (r.eval || {}).val || {}; return v.nll != null && v.nll_ewma != null ? v.nll_ewma - v.nll : null; }), borderColor: COL.val_train_companies, backgroundColor: COL.val_train_companies },
    ], { xTitle: 'epoch' });
    const nominal = (v, c) => ({ label: `nominal ${v}`, data: epochs.map(() => v), borderColor: c, borderDash: [4, 4], pointRadius: 0, borderWidth: 1 });
    lineChart('fcl-c-cov', labels, [
      { label: '68% interval', data: epochs.map(r => ev(r, 'val', 'cov68')), borderColor: COL.val, backgroundColor: COL.val },
      nominal(0.68, COL.dim),
      { label: '95% interval', data: epochs.map(r => ev(r, 'val', 'cov95')), borderColor: COL.train, backgroundColor: COL.train },
      nominal(0.95, COL.dim),
    ], { xTitle: 'epoch', y: { suggestedMin: 0.6, suggestedMax: 1 } });
    lineChart('fcl-c-hit', labels, [...sets.map(s => setLine(s, 'hit_rate')), nominal(0.5, COL.dim)], { xTitle: 'epoch', y: { ticks: { callback: v => (100 * v).toFixed(0) + '%' } } });
    if (timing.length) {
      const names = [...new Set(timing.flatMap(r => (r.phases || []).map(p => p.name)))];
      const pal = { batch: '#f5c842', forward_backward: '#4f8ef7', optimizer: '#3dd68c', eval: '#f76f6f' };
      lineChart('fcl-c-time', timing.map(r => r.iteration), names.map(n => ({
        label: n, data: timing.map(r => ((r.phases || []).find(p => p.name === n) || {}).secs ?? 0), backgroundColor: pal[n] || '#9aa3c7', borderColor: pal[n] || '#9aa3c7', borderWidth: 0,
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

  function renderAll() {
    const none = !live && !epochs.length && !config;
    $('fcl-placeholder').style.display = none ? '' : 'none'; $('fcl-ph-dir').textContent = runDir;
    $('fcl-body').style.display = none ? 'none' : '';
    renderStatus(); renderConfig(); renderTiles(); renderCharts(); renderReport();
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
    setInterval(pollLive, 3000);
    setInterval(pollSlow, 10000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
