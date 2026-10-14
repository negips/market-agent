/* Forecaster v4 diagrams: one SVG string per layer, shared by architecture.html and mathematics.html.
   window.FC_VIS[name]() -> '<svg …>…</svg>'. Colours follow the v3 diagrams. */
(() => {
  const C = { blue: '#4f8ef7', blueF: '#122455', green: '#3dd68c', greenF: '#0b2e20', yellow: '#f5c842', yellowF: '#2a2210',
              red: '#f76f6f', ink: '#17192b', edge: '#2a314f', mute: '#6b7599', txt: '#e6e9f5', dim: '#9aa3c7' };
  const FONT = "font-family:system-ui,sans-serif";
  const MONO = "font-family:ui-monospace,Menlo,monospace";

  const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" style="width:100%;min-width:${Math.min(w,560)}px;height:auto;display:block" role="img">` +
    `<defs><marker id="fa" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="${C.mute}"/></marker></defs>${body}</svg>`;
  const rect = (x, y, w, h, fill, stroke, r = 4, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.2" ${extra}/>`;
  const text = (x, y, s, o = {}) => `<text x="${x}" y="${y}" fill="${o.c || C.dim}" font-size="${o.s || 11}" text-anchor="${o.a || 'start'}" style="${o.m ? MONO : FONT}${o.b ? ';font-weight:600' : ''}">${s}</text>`;
  const line = (x1, y1, x2, y2, o = {}) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${o.c || C.edge}" stroke-width="${o.w || 1}" ${o.d ? `stroke-dasharray="${o.d}"` : ''} ${o.arrow ? 'marker-end="url(#fa)"' : ''} opacity="${o.o || 1}"/>`;
  const dot = (x, y, r, fill, stroke) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.2"/>`;
  const val = (i, j) => Math.sin(i * 1.7 + j * 2.3) * Math.cos(i * 0.6 - j);
  const heat = v => v >= 0 ? `rgba(79,142,247,${0.15 + 0.6 * Math.min(1, v)})` : `rgba(247,111,111,${0.15 + 0.6 * Math.min(1, -v)})`;
  const grid = (x, y, rows, cols, s, fn) => {
    let o = '';
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) o += rect(x + j * s, y + i * s, s - 1, s - 1, fn(i, j), 'none', 2);
    return o;
  };

  const V = {};

  V.flow = () => {
    const steps = [['Inputs', 'M×N, M×K, N×2', C.mute], ['1 Embed', 'M×N×d', C.green], ['2 Mix', 'over N stocks', C.green], ['3 Fuse', '+ s_t + P_t', C.green],
                   ['4 Time ×L', 'over M, causal', C.blue], ['5 LN', 'M×N×d', C.blue]];
    let o = '', x = 10;
    steps.forEach(([a, b, col], i) => {
      o += rect(x, 30, 108, 50, C.ink, col) + text(x + 54, 52, a, { c: C.txt, a: 'middle', b: 1, m: 1, s: 12 }) + text(x + 54, 68, b, { a: 'middle', s: 10 });
      if (i < steps.length - 1) o += line(x + 108, 55, x + 124, 55, { c: C.mute, arrow: 1 });
      x += 124;
    });
    o += line(x - 16, 55, x + 4, 28, { c: C.mute, arrow: 1 }) + line(x - 16, 55, x + 4, 82, { c: C.mute, arrow: 1 });
    o += rect(x + 6, 8, 118, 40, C.yellowF, C.yellow) + text(x + 65, 25, '6 Return head', { c: C.txt, a: 'middle', b: 1, m: 1, s: 11 }) + text(x + 65, 39, 'μ, σ: M×N', { a: 'middle', s: 10 });
    o += rect(x + 6, 62, 118, 40, C.yellowF, C.yellow) + text(x + 65, 79, '7 Time head', { c: C.txt, a: 'middle', b: 1, m: 1, s: 11 }) + text(x + 65, 93, 'τ̂: M×3', { a: 'middle', s: 10 });
    o += text(10, 112, 'green = shared per-stock weights · blue = causal over time · yellow = predictions', { s: 10.5, c: C.mute });
    return svg(x + 134, 122, o);
  };

  V.axes = () => {
    const cols = 12, rows = 6, s = 26, x0 = 70, y0 = 36;
    let o = '';
    for (let k = 2; k >= 1; k--) o += rect(x0 + k * 5, y0 - k * 5, cols * s, rows * s, 'none', C.edge, 3);
    o += rect(x0, y0, cols * s, rows * s, C.ink, C.edge, 3);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) o += rect(x0 + j * s + 2, y0 + i * s + 2, s - 4, s - 4, heat(val(i, j)), 'none', 2);
    const cj = 7, ri = 3;
    o += rect(x0 + cj * s, y0, s, rows * s, 'rgba(61,214,140,.16)', C.green, 3);
    o += rect(x0, y0 + ri * s, cols * s, s, 'rgba(79,142,247,.16)', C.blue, 3);
    for (let i = 0; i < rows; i++) o += text(x0 - 8, y0 + i * s + 17, 'stock ' + (i + 1), { a: 'end', s: 10 });
    for (let j = 0; j < cols; j += 1) if ([0, 1, cj, cols - 1].includes(j)) o += text(x0 + j * s + s / 2, y0 + rows * s + 14, 't=' + (j + 1), { a: 'middle', s: 10 });
    o += text(x0 + cols * s + 16, y0 + 6, 'depth d', { s: 10.5, c: C.mute }) + line(x0 + cols * s + 12, y0 + 12, x0 + cols * s + 2, y0 + 4, { c: C.mute, arrow: 1 });
    o += text(x0, 18, 'one window: M minutes × N stocks (each cell is a d-vector)', { c: C.txt, s: 11.5 });
    o += rect(x0 + cols * s + 70, 44, 310, 40, C.greenF, C.green) + text(x0 + cols * s + 82, 60, 'Stock mixing (layer 2)', { c: C.green, b: 1, s: 11.5 }) + text(x0 + cols * s + 82, 76, "one column: a minute's N stocks attend to each other", { s: 10 });
    o += rect(x0 + cols * s + 70, 108, 310, 40, C.blueF, C.blue) + text(x0 + cols * s + 82, 124, 'Time attention (layer 4)', { c: C.blue, b: 1, s: 11.5 }) + text(x0 + cols * s + 82, 140, "one row: a stock's minutes attend causally →", { s: 10 });
    o += line(x0 + cj * s + s / 2, y0 + 4, x0 + cols * s + 68, 62, { c: C.green, d: '3 3' });
    o += line(x0 + cols * s, y0 + ri * s + s / 2, x0 + cols * s + 68, 128, { c: C.blue, d: '3 3' });
    return svg(x0 + cols * s + 390, y0 + rows * s + 34, o);
  };

  V.in = () => {
    let o = '';
    o += grid(20, 28, 6, 8, 20, (i, j) => heat(val(i, j))) + text(100, 18, 'R', { c: C.txt, a: 'middle', b: 1, m: 1 }) + text(100, 172, 'returns · M×N', { a: 'middle', s: 10 });
    o += text(8, 94, 'M', { s: 10 }) + text(100, 156, 'N stocks →', { a: 'middle', s: 10 });
    o += grid(220, 28, 6, 5, 20, (i, j) => `rgba(245,200,66,${0.12 + 0.45 * Math.abs(Math.sin(i * 0.7 + j * 1.9))})`) + text(270, 18, 'T', { c: C.txt, a: 'middle', b: 1, m: 1 }) + text(270, 172, 'clock · M×K', { a: 'middle', s: 10 });
    o += grid(360, 28, 8, 2, 20, (i, j) => `rgba(61,214,140,${0.12 + 0.45 * Math.abs(Math.sin(i * 1.3 + j))})`) + text(380, 18, 'C', { c: C.txt, a: 'middle', b: 1, m: 1 }) + text(380, 208, 'context · N×2', { a: 'middle', s: 10 });
    o += text(430, 60, 'each stock token will be', { s: 10.5 }) + text(430, 76, '[ r , τ (5) , c (2) ]  = 8 numbers', { c: C.txt, m: 1, s: 11 });
    o += text(430, 100, 'r: its return · τ: the clock (shared)', { s: 10 }) + text(430, 114, 'c: mean |r| and traded share', { s: 10 });
    return svg(720, 220, o);
  };

  V.embed = () => {
    let o = '';
    const ins = [['r', C.blue], ['τ₁', C.yellow], ['τ₂', C.yellow], ['τ₃', C.yellow], ['τ₄', C.yellow], ['τ₅', C.yellow], ['c₁', C.green], ['c₂', C.green]];
    const xi = 90, xo = 330;
    const yi = k => 20 + k * 20, yo = k => 28 + k * 24;
    for (let a = 0; a < 8; a++) for (let b = 0; b < 6; b++) o += line(xi, yi(a), xo, yo(b), { c: C.mute, o: .28 });
    ins.forEach(([l, c], k) => { o += dot(xi, yi(k), 6, C.ink, c) + text(xi - 12, yi(k) + 4, l, { a: 'end', m: 1, s: 10.5 }); });
    for (let b = 0; b < 6; b++) o += dot(xo, yo(b), 7, C.greenF, C.green);
    o += text(xo, 28 + 6 * 24 + 4, '⋮ d = 64 numbers', { a: 'middle', s: 10.5, c: C.green });
    o += text(xi, 190, '8 inputs (F)', { a: 'middle', s: 10.5 });
    o += text(440, 60, 'E = X Wₑᵀ + b', { c: C.txt, m: 1, s: 12 }) + text(440, 82, 'one matrix Wₑ (d×8),', { s: 10.5 }) + text(440, 96, 'the same for every stock', { s: 10.5 }) + text(440, 110, 'and every minute', { s: 10.5 });
    o += text(440, 134, 'M·N tokens, each → d numbers', { s: 10.5, c: C.green });
    return svg(640, 200, o);
  };

  V.mha = () => {
    let o = '';
    const b = (x, y, w, t, f = C.ink, s = C.edge) => rect(x, y, w, 30, f, s) + text(x + w / 2, y + 19, t, { c: C.txt, a: 'middle', m: 1, s: 11 });
    o += b(10, 60, 50, 'A') + line(60, 75, 84, 40, { c: C.mute, arrow: 1 }) + line(60, 75, 84, 75, { c: C.mute, arrow: 1 }) + line(60, 75, 84, 110, { c: C.mute, arrow: 1 });
    o += b(86, 25, 62, 'Q=AW_Q') + b(86, 60, 62, 'K=AW_K') + b(86, 95, 62, 'V=AW_V');
    o += line(148, 40, 176, 66, { c: C.mute, arrow: 1 }) + line(148, 75, 176, 75, { c: C.mute, arrow: 1 });
    o += rect(178, 52, 120, 46, C.blueF, C.blue) + text(238, 71, 'softmax(QKᵀ/√d_h + B)', { c: C.txt, a: 'middle', m: 1, s: 10 }) + text(238, 88, 'n×n weights, per head', { a: 'middle', s: 10 });
    o += line(298, 75, 322, 75, { c: C.mute, arrow: 1 }) + line(148, 110, 340, 110, { c: C.mute }) + line(340, 110, 340, 92, { c: C.mute, arrow: 1 });
    o += b(324, 60, 32, '·V') + line(356, 75, 378, 75, { c: C.mute, arrow: 1 });
    o += b(380, 60, 74, 'concat heads') + line(454, 75, 474, 75, { c: C.mute, arrow: 1 }) + b(476, 60, 46, 'W_O');
    o += text(10, 150, 'H heads in parallel, each on d_h = d/H columns · B = 0 (stock mixing) or causal (time)', { s: 10.5, c: C.mute });
    return svg(540, 160, o);
  };

  V.mix = () => {
    let o = '';
    const n = 7, cx = 120, cy = 82, r = 56, pts = [];
    for (let i = 0; i < n; i++) { const a = i / n * 2 * Math.PI - Math.PI / 2; pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) o += line(pts[i][0], pts[i][1], pts[j][0], pts[j][1], { c: C.green, o: .3 });
    pts.forEach(p => o += dot(p[0], p[1], 8, C.greenF, C.green));
    o += text(cx, 168, 'minute t: every stock attends to every other', { a: 'middle', s: 10.5 });
    o += line(190, 82, 262, 82, { c: C.mute, arrow: 1 }) + text(226, 74, 'mean over i', { a: 'middle', s: 10 });
    o += dot(290, 82, 14, C.yellowF, C.yellow) + text(290, 108, 's_t', { a: 'middle', c: C.yellow, m: 1, s: 12 });
    o += text(330, 62, 'E′ = E + MHA(LN(E))', { c: C.txt, m: 1, s: 11.5 }) + text(330, 82, 'no positional code: stock order', { s: 10.5 }) + text(330, 96, 'does not matter', { s: 10.5 });
    o += text(330, 120, 's_t: market summary of the minute', { s: 10.5, c: C.yellow });
    return svg(600, 180, o);
  };

  V.fuse = () => {
    let o = '';
    const ys = [28, 58, 88, 118], xl = 40, xr = 400;
    ys.forEach((y, i) => {
      o += rect(xl, y - 10, 70, 20, C.greenF, C.green) + text(xl + 35, y + 4, `E′_{t,${i + 1}}`, { c: C.txt, a: 'middle', m: 1, s: 10 });
      o += line(110, y, 178, y, { c: C.mute, arrow: 1 }) + dot(190, y, 11, C.ink, C.mute) + text(190, y + 4, '+', { c: C.txt, a: 'middle', s: 14 });
      o += line(201, y, 396, y, { c: C.mute, arrow: 1 }) + rect(xr, y - 10, 70, 20, C.blueF, C.blue) + text(xr + 35, y + 4, `z_{t,${i + 1}}`, { c: C.txt, a: 'middle', m: 1, s: 10 });
      o += line(262, 23, 190, y - 11, { c: C.yellow, d: '3 3', o: .6 }) + line(290, 160, 190, y + 11, { c: C.blue, d: '3 3', o: .6 });
    });
    o += rect(232, 3, 100, 20, C.yellowF, C.yellow) + text(282, 17, 'W_s s_t + b_s', { c: C.txt, a: 'middle', m: 1, s: 10 });
    o += rect(234, 160, 100, 20, C.blueF, C.blue) + text(284, 174, 'P_t (position)', { c: C.txt, a: 'middle', m: 1, s: 10 });
    o += text(40, 150, 'N stocks, one minute', { s: 10 }) + text(400, 150, 'same s_t and P_t for all', { s: 10 });
    return svg(540, 190, o);
  };

  V.time = () => {
    let o = '', n = 9, s = 22, x0 = 40, y0 = 26;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const on = j <= i;
      o += rect(x0 + j * s, y0 + i * s, s - 2, s - 2, on ? `rgba(79,142,247,${0.2 + 0.55 * Math.abs(Math.sin(i * 1.1 + j * 2.1))})` : C.ink, on ? 'none' : C.edge, 2);
    }
    o += text(x0 + n * s / 2, 16, 'key minute s →', { a: 'middle', s: 10.5 }) + text(8, y0 + n * s / 2, 't', { s: 11, c: C.txt, m: 1 });
    o += text(x0 + n * s / 2, y0 + n * s + 16, 'attention weights, one stock (M×M)', { a: 'middle', s: 10.5 });
    o += text(x0 + 150, y0 + 40, 'future', { c: C.mute, s: 10.5 });
    const bx = 300, by = 90, gap = 30;
    for (let k = 0; k < 7; k++) o += dot(bx + k * gap, by, 8, k <= 5 ? C.blueF : C.ink, k <= 5 ? C.blue : C.edge);
    for (let k = 0; k < 5; k++) o += `<path d="M${bx + 5 * gap},${by - 9} Q${bx + (5 + k) * gap / 2 + 0},${by - 28 - (5 - k) * 6} ${bx + k * gap},${by - 9}" stroke="${C.blue}" fill="none" opacity="${0.35 + 0.1 * k}"/>`;
    o += text(bx + 5 * gap, by + 24, 't', { a: 'middle', c: C.txt, m: 1, s: 11 }) + text(bx + 6 * gap, by + 24, 't+1', { a: 'middle', c: C.mute, m: 1, s: 10 });
    o += text(bx - 10, by + 56, 'minute t sees only minutes ≤ t', { s: 10.5, c: C.blue }) + text(bx - 10, by + 72, 'so every position is a valid prediction', { s: 10.5 });
    o += text(bx - 10, by + 100, 'Z ← Z + MHA(LN Z)  ;  Z ← Z + FFN(LN Z)', { c: C.txt, m: 1, s: 10.5 });
    return svg(560, 250, o);
  };

  V.ret = () => {
    let o = '';
    const ys = [26, 62, 98, 134];
    ys.forEach((y, i) => {
      o += rect(30, y - 12, 14, 26, C.blueF, C.blue) + rect(46, y - 12, 14, 26, C.blueF, C.blue) + rect(62, y - 12, 14, 26, C.blueF, C.blue) + text(88, y + 4, `h_{t,${i + 1}} (d)`, { m: 1, s: 10 });
      o += line(150, y, 224, 80, { c: C.mute, o: .5 }) + dot(330, y, 11, C.yellowF, C.yellow) + text(330, y + 4, 'μ,σ', { c: C.txt, a: 'middle', s: 10 });
      o += line(266, 80, 318, y, { c: C.yellow, o: .6, arrow: 1 });
    });
    o += rect(224, 62, 44, 36, C.ink, C.yellow) + text(246, 84, 'W_r', { c: C.txt, a: 'middle', m: 1, s: 12 });
    o += text(360, 70, '(μ, s) = W_r h_{t,i} + b_r', { c: C.txt, m: 1, s: 11.5 }) + text(360, 90, 'one weight matrix for all stocks;', { s: 10.5 }) + text(360, 104, 'μ estimates r_{t+1,i}, σ its spread', { s: 10.5 });
    return svg(600, 164, o);
  };

  V.timehead = () => {
    let o = '';
    const ys = [26, 56, 86, 116];
    ys.forEach((y, i) => { o += rect(30, y - 10, 60, 20, C.blueF, C.blue) + text(60, y + 4, `H_{t,${i + 1}}`, { c: C.txt, a: 'middle', m: 1, s: 10 }) + line(90, y, 140, 71, { c: C.mute, o: .5 }); });
    o += rect(140, 51, 56, 40, C.ink, C.yellow) + text(168, 68, 'mean', { c: C.txt, a: 'middle', m: 1, s: 11 }) + text(168, 82, 'over i', { a: 'middle', s: 10 });
    o += line(196, 71, 232, 71, { c: C.mute, arrow: 1 }) + rect(234, 51, 46, 40, C.ink, C.yellow) + text(257, 76, 'W_τ', { c: C.txt, a: 'middle', m: 1, s: 12 });
    const outs = ['minute of day', 'log day-gap', 'weekday'];
    outs.forEach((t, k) => { const y = 36 + k * 34; o += line(280, 71, 316, y, { c: C.yellow, o: .6, arrow: 1 }) + dot(328, y, 9, C.yellowF, C.yellow) + text(346, y + 4, t, { s: 10.5 }); });
    o += text(30, 150, 'τ̂_{t+1} = W_τ h̄_t + b_τ  (next bar\'s clock; near-deterministic)', { c: C.txt, m: 1, s: 10.5 });
    return svg(560, 160, o);
  };

  V.loss = () => {
    let o = '', rows = 5, cols = 10, s = 24, x0 = 70, y0 = 28;
    const traded = (i, j) => Math.sin(i * 3.1 + j * 1.3) > -0.25;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const t = traded(i, j);
      o += rect(x0 + j * s, y0 + i * s, s - 2, s - 2, t ? `rgba(61,214,140,.35)` : C.ink, t ? C.green : C.edge, 2);
      if (!t) o += line(x0 + j * s + 3, y0 + i * s + s - 5, x0 + j * s + s - 5, y0 + i * s + 3, { c: C.mute, o: .7 });
    }
    for (let i = 0; i < rows; i++) o += text(x0 - 8, y0 + i * s + 15, 'stock ' + (i + 1), { a: 'end', s: 10 });
    o += text(x0 + cols * s / 2, 18, 'target minutes t+1 →', { a: 'middle', s: 10.5 });
    o += rect(x0 + cols * s + 24, 40, 14, 14, 'rgba(61,214,140,.35)', C.green, 2) + text(x0 + cols * s + 46, 51, 'traded: counted in L_ret', { s: 10.5 });
    o += rect(x0 + cols * s + 24, 64, 14, 14, C.ink, C.edge, 2) + line(x0 + cols * s + 26, 76, x0 + cols * s + 36, 66, { c: C.mute }) + text(x0 + cols * s + 46, 75, 'forward-filled: ignored', { s: 10.5 });
    o += text(x0 + cols * s + 24, 104, 'Student-t NLL of r given (μ, σ), mean over counted cells', { s: 10.5, c: C.txt });
    return svg(640, 170, o);
  };

  window.FC_VIS = V;
  window.fcVisHtml = name => (V[name] ? V[name]() : '');
})();
