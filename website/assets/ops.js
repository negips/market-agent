/* Client for scripts/ops_server.jl. Only works when the page is served by that server
   (julia --project=packages/StockSwingPredictor scripts/ops_server.jl); under any other static server
   Ops.available() resolves false and callers hide their buttons. */
window.Ops = (() => {
  let actions = null;
  async function load() {
    try {
      const r = await fetch('/api/actions', { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      const list = await r.json();
      actions = Object.fromEntries(list.map(a => [a.name, a]));
    } catch { actions = null; }
    return actions;
  }
  const available = async () => (actions || await load()) !== null;
  async function run(name) {
    const r = await fetch('/api/run/' + encodeURIComponent(name), { method: 'POST' });
    const body = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, body };
  }
  async function job(name) {
    const r = await fetch('/api/job/' + encodeURIComponent(name), { cache: 'no-store' });
    return r.ok ? r.json() : null;
  }
  /* Start a job and call onUpdate(state) every 1.5 s until it stops running; resolves with the final state. */
  async function runJob(name, onUpdate) {
    const res = await run(name);
    if (!res.ok && res.status !== 409) throw new Error(res.body.error || ('HTTP ' + res.status));
    for (;;) {
      const s = await job(name);
      if (onUpdate && s) onUpdate(s);
      if (!s || !s.running) return s;
      await new Promise(r => setTimeout(r, 1500));
    }
  }
  return { available, run, job, runJob, refresh: load };
})();
