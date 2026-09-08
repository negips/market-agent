// Shared formatters and helpers used across all pages.

const _inr = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 });
const _vol = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 });

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtPrice(v) {
  if (v == null) return '<span class="na">—</span>';
  return `₹${_inr.format(v)}`;
}

function fmtMcap(v) {
  if (v == null) return '<span class="na">—</span>';
  if (v >= 100000) return `₹${(v / 100000).toFixed(2)} L Cr`;
  if (v >= 1000)   return `₹${_inr.format(Math.round(v))} Cr`;
  return `₹${_inr.format(v)} Cr`;
}

function fmtVol(v) {
  if (v == null) return '<span class="na">—</span>';
  if (v >= 1e7) return `${(v / 1e7).toFixed(1)} Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(1)} L`;
  return _vol.format(v);
}

// Returns number of calendar days from today to dateStr (YYYY-MM-DD or Date)
function daysUntil(dateStr) {
  const d    = new Date(dateStr);
  const now  = new Date();
  now.setHours(0, 0, 0, 0);
  d.setHours(0, 0, 0, 0);
  return Math.round((d - now) / 86400000);
}

function fmtDaysPill(n) {
  let label, cls;
  if (n === 0)      { label = 'today';         cls = 'days-today'; }
  else if (n === 1) { label = 'tomorrow';       cls = 'days-hot'; }
  else if (n <= 3)  { label = `in ${n} days`;  cls = 'days-hot'; }
  else if (n <= 7)  { label = `in ${n} days`;  cls = 'days-soon'; }
  else              { label = `in ${n} days`;  cls = 'days-normal'; }
  return `<span class="days-pill ${cls}">${label}</span>`;
}

function fmtDate(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}

// Render a confidence badge (score + pass/fail colouring)
// with 5 signal dots below. Tooltip has full breakdown.
function fmtConf(conf) {
  if (!conf) return '<span class="na">—</span>';
  const score = conf.score;
  const cls   = score >= 70 ? 'conf-pass' : score >= 40 ? 'conf-warn' : 'conf-fail';

  const sigDot = (flagged, na) => {
    const c = na ? 'sig-na' : flagged ? 'sig-bad' : 'sig-ok';
    return `<span class="sig-dot ${c}"></span>`;
  };

  const b   = conf.beneish   || {};
  const cf  = conf.cashflow  || {};
  const pl  = conf.pledging  || {};
  const fo  = conf.forensics || {};
  const sv  = conf.surveillance || {};

  const bDot = !b.applicable ? sigDot(false, true) :
                b.is_flagged ? sigDot(true, false)  : sigDot(false, false);

  const signals = [
    bDot,
    sigDot(cf.is_flagged, false),
    sigDot(pl.is_flagged, pl.quarters_used === 0),
    sigDot(fo.is_flagged, fo.flag_count == null),
    sigDot(sv.is_flagged, !sv.checked),
  ].join('');

  const tooltip = [
    `Score: ${score}/100 (${conf.pass ? 'PASS' : 'FAIL'})`,
    `Beneish M-Score: ${b.applicable ? (b.m_score != null ? b.m_score.toFixed(2) : 'N/A') : 'N/A (bank/NBFC)'}`,
    `CFO < NI: ${cf.years_cfo_lt_ni ?? '?'}/${cf.years_checked ?? '?'} years`,
    `Pledging: ${pl.latest_pct != null ? pl.latest_pct.toFixed(1) + '%' : 'N/R'} (${pl.trend ?? '—'})`,
    `Forensics: ${fo.flag_count ?? 0} red flag(s)`,
    `Surveillance: ${sv.checked ? ((sv.on_asm ? 'ASM ' : '') + (sv.on_gsm ? 'GSM' : '') || 'clean') : 'unchecked'}`,
  ].join('\n');

  return `<span class="conf-badge ${cls}" title="${esc(tooltip)}">
    <span class="conf-dot"></span>${score}
    <div class="signals">${signals}</div>
  </span>`;
}

function showError(container, msg, hint = '') {
  container.innerHTML = `
    <div class="state-overlay">
      <div class="error-icon">⚠️</div>
      <div class="error-msg">${msg}</div>
      ${hint ? `<div class="error-hint">${hint}</div>` : ''}
    </div>`;
}

function visiblePages(cur, total) {
  const s = new Set([1, total, cur - 1, cur, cur + 1, cur - 2, cur + 2]
    .filter(p => p >= 1 && p <= total));
  return [...s].sort((a, b) => a - b);
}

function renderPagination(el, cur, total, onPage) {
  if (total <= 1) { el.innerHTML = ''; return; }
  const pages = visiblePages(cur, total);
  let html = `<button class="page-btn" ${cur === 1 ? 'disabled' : ''} data-p="${cur - 1}">‹ Prev</button>`;
  let prev  = 0;
  for (const p of pages) {
    if (p - prev > 1) html += `<span class="page-ellipsis">…</span>`;
    html += `<button class="page-btn ${p === cur ? 'active' : ''}" data-p="${p}">${p}</button>`;
    prev = p;
  }
  html += `<button class="page-btn" ${cur === total ? 'disabled' : ''} data-p="${cur + 1}">Next ›</button>`;
  el.innerHTML = html;
  el.querySelectorAll('.page-btn:not(:disabled)').forEach(btn => {
    btn.addEventListener('click', () => onPage(Number(btn.dataset.p)));
  });
}
