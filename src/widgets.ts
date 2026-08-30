// Интерактивные демо внутри карточек разбора. Каждый виджет — маленькая
// самодостаточная модель механизма, не картинка.
import { t } from './i18n';
import { el, svgEl } from './world';
import { kindOf } from './color';

export function buildWidget(kind: string): HTMLElement | null {
  switch (kind) {
    case 'delta': return deltaWidget();
    case 'rope': return ropeWidget();
    case 'gqa': return gqaWidget();
    case 'conv': return convWidget();
    case 'gate': return gateWidget();
    case 'norm': return normWidget();
    case 'posembed': return posembedWidget();
    case 'merge': return mergeWidget();
    case 'mtp': return mtpWidget();
    case 'cost': return costWidget();
    default: return null;
  }
}

const box = () => el('div', '', `border:1px solid var(--line);background:var(--glass-soft);border-radius:12px;padding:16px 18px;display:flex;flex-direction:column;gap:14px`);
const lbl = (s: string) => el('div', 'mono wlabel', '', s);
function slider(min: number, max: number, val: number, step: number, on: (v: number) => void): HTMLInputElement {
  const s = document.createElement('input');
  s.type = 'range'; s.min = String(min); s.max = String(max); s.step = String(step); s.value = String(val);
  s.addEventListener('input', () => on(+s.value));
  return s;
}
function ctl(label: string, input: HTMLElement, out?: HTMLElement): HTMLElement {
  const row = el('div', 'wctl');
  row.appendChild(lbl(label));
  input.style.flex = '1';
  row.appendChild(input);
  if (out) row.appendChild(out);
  return row;
}
function btn(label: string, on: () => void): HTMLElement {
  const b = el('div', 'chip mini', '', label);
  b.addEventListener('click', on);
  return b;
}
const rnd = (() => { let s = 42; return () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; }; })();
const randVec = (n: number) => { const v = Array.from({ length: n }, () => rnd() * 2 - 1); const m = Math.hypot(...v); return v.map(x => x / m); };
const cellColor = (v: number) => Math.abs(v) < 0.03
  ? 'var(--line)'
  : v >= 0
    ? `rgba(176,73,42,${Math.min(1, 0.12 + Math.abs(v)).toFixed(2)})`
    : `rgba(78,106,134,${Math.min(1, 0.12 + Math.abs(v)).toFixed(2)})`;

// ── Дельта-правило: состояние, которое стирает и переписывает ──
function deltaWidget(): HTMLElement {
  const N = 8;
  const root = box();
  let S: number[][] = Array.from({ length: N }, () => Array(N).fill(0));
  let alpha = 0.95, beta = 0.8, step = 0, timer = 0;
  const keys = Array.from({ length: 4 }, () => randVec(N));

  const grid = el('div', '', `display:grid;grid-template-columns:repeat(${N},22px);gap:2px;align-self:center`);
  const cells: HTMLElement[] = [];
  for (let i = 0; i < N * N; i++) {
    const c = el('div', '', 'width:22px;height:22px;border-radius:3px;background:var(--line)');
    cells.push(c); grid.appendChild(c);
  }
  const barsWrap = el('div', '', 'display:flex;gap:22px;justify-content:center');
  const mkBars = (title: string) => {
    const w = el('div', '', 'display:flex;flex-direction:column;gap:5px;align-items:center');
    const row = el('div', '', 'display:flex;gap:3px;align-items:center;height:44px');
    const bars: HTMLElement[] = [];
    for (let i = 0; i < N; i++) {
      const holder = el('div', '', 'width:9px;height:44px;position:relative');
      const b = el('div', '', 'position:absolute;left:0;width:9px;border-radius:2px;background:var(--warn)');
      holder.appendChild(b); row.appendChild(holder); bars.push(b);
    }
    w.appendChild(row); w.appendChild(lbl(title));
    barsWrap.appendChild(w);
    return bars;
  };
  const bV = mkBars(t('w.target')), bP = mkBars(t('w.pred')), bE = mkBars(t('w.err'));
  const setBars = (bars: HTMLElement[], v: number[], color: string) => bars.forEach((b, i) => {
    const h = Math.min(21, Math.abs(v[i]) * 26);
    b.style.height = h + 'px'; b.style.background = color;
    if (v[i] >= 0) { b.style.top = (22 - h) + 'px'; } else { b.style.top = '22px'; }
  });

  const stat = el('div', 'mono wlabel', 'align-self:center');
  const paint = () => {
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
      cells[i * N + j].style.background = cellColor(S[i][j] * 1.4);
    stat.textContent = `${t('w.step')} ${step} · α ${alpha.toFixed(2)} · β ${beta.toFixed(2)}`;
  };
  const doStep = () => {
    step++;
    const k = keys[step % keys.length];
    const v = randVec(N).map(x => x * 0.9);
    const pred = S.map(row => row.reduce((a, x, j) => a + x * k[j], 0));
    const err = v.map((x, i) => x - alpha * pred[i]);
    S = S.map((row, i) => row.map((x, j) => alpha * x + beta * err[i] * k[j]));
    setBars(bV, v, 'var(--good)'); setBars(bP, pred, 'var(--warn)'); setBars(bE, err, 'var(--bad)');
    paint();
  };
  const reset = () => { S = Array.from({ length: N }, () => Array(N).fill(0)); step = 0; paint(); };

  const controls = el('div', '', 'display:flex;gap:8px;align-items:center;flex-wrap:wrap');
  controls.append(btn(t('w.step'), doStep), btn(t('w.play'), () => {
    if (timer) { clearInterval(timer); timer = 0; return; }
    timer = window.setInterval(doStep, 450);
  }), btn(t('w.reset'), reset));
  root.append(
    lbl(t('w.state')),
    grid, stat, barsWrap,
    ctl('α', slider(0.5, 1, alpha, 0.01, v => { alpha = v; paint(); })),
    ctl('β', slider(0, 1, beta, 0.01, v => { beta = v; paint(); })),
    controls,
    el('div', 'small-note', '', t('w.delta.note')),
  );
  (root as any).cleanup = () => timer && clearInterval(timer);
  paint();
  return root;
}

// ── RoPE: циферблаты с разной скоростью ──
function ropeWidget(): HTMLElement {
  const root = box();
  const dials = el('div', '', 'display:flex;gap:16px;justify-content:center;flex-wrap:wrap');
  const freqs = [1 / 2, 1 / 8, 1 / 32, 1 / 128];
  const needles: SVGElement[] = [];
  freqs.forEach((_, i) => {
    const w = el('div', '', 'display:flex;flex-direction:column;align-items:center;gap:4px');
    const svg = svgEl('svg', { width: 74, height: 74 });
    svg.appendChild(svgEl('circle', { cx: 37, cy: 37, r: 33, fill: 'var(--glass)', stroke: 'var(--line-strong)' }));
    const n = svgEl('line', { x1: 37, y1: 37, x2: 37, y2: 8, stroke: 'var(--accent-deep)', 'stroke-width': 2.5, 'stroke-linecap': 'round' });
    svg.appendChild(n); needles.push(n);
    w.appendChild(svg as any);
    w.appendChild(lbl(i === 0 ? t('w.fast') : i === freqs.length - 1 ? t('w.slow') : `θ${i}`));
    dials.appendChild(w);
  });
  // замороженная пара (partial rotary)
  const fw = el('div', '', 'display:flex;flex-direction:column;align-items:center;gap:4px');
  const fsvg = svgEl('svg', { width: 74, height: 74 });
  fsvg.appendChild(svgEl('circle', { cx: 37, cy: 37, r: 33, fill: 'var(--line)', stroke: 'var(--line-strong)', 'stroke-dasharray': '4 4' }));
  fsvg.appendChild(svgEl('line', { x1: 37, y1: 37, x2: 37, y2: 8, stroke: 'var(--line-strong)', 'stroke-width': 2.5 }));
  fw.appendChild(fsvg as any); fw.appendChild(lbl(t('w.frozen')));
  dials.appendChild(fw);

  const out = el('div', 'mono wlabel');
  const setPos = (p: number) => {
    needles.forEach((n, i) => {
      const a = p * freqs[i];
      n.setAttribute('x2', String(37 + 29 * Math.sin(a)));
      n.setAttribute('y2', String(37 - 29 * Math.cos(a)));
    });
    out.textContent = String(p);
  };
  root.append(dials, ctl(t('w.pos'), slider(0, 512, 0, 1, setPos), out), el('div', 'small-note', '', t('w.rope.note')));
  setPos(0);
  return root;
}

// ── GQA: 24 головы на 4 KV + калькулятор кеша ──
function gqaWidget(): HTMLElement {
  const root = box();
  const qRow = el('div', '', 'display:flex;gap:3px;justify-content:center;flex-wrap:wrap');
  const kvRow = el('div', '', 'display:flex;gap:10px;justify-content:center;margin-top:10px');
  const kvEls: HTMLElement[] = [];
  const attn = kindOf('attn');
  for (let g = 0; g < 4; g++) {
    const kv = el('div', 'mono', `width:120px;height:34px;border-radius:6px;background:${attn.bg};border:1.5px solid ${attn.bd};
      display:flex;align-items:center;justify-content:center;font-size:11px;color:${attn.fg};transition:all 120ms`, `KV ${g}`);
    kvEls.push(kv); kvRow.appendChild(kv);
  }
  for (let q = 0; q < 24; q++) {
    const c = el('div', 'mono', `width:19px;height:26px;border-radius:3px;background:${attn.solid};opacity:0.65;
      display:flex;align-items:center;justify-content:center;font-size:8px;color:#fff;cursor:default`, String(q));
    c.addEventListener('mouseenter', () => {
      kvEls.forEach((k, g) => {
        const on = g === Math.floor(q / 6);
        k.style.boxShadow = on ? '0 0 0 2.5px var(--accent)' : '';
        k.style.transform = on ? 'scale(1.06)' : '';
      });
      c.style.opacity = '1';
    });
    c.addEventListener('mouseleave', () => {
      kvEls.forEach(k => { k.style.boxShadow = ''; k.style.transform = ''; });
      c.style.opacity = '0.65';
    });
    qRow.appendChild(c);
  }
  const out = el('div', 'mono', 'font-size:13px;color:var(--ink-soft);text-align:center;line-height:1.6');
  // KV-кеш: только 16 полных слоёв, 2(K,V) × головы × 256 × 2 байта (bf16)
  const calc = (n: number) => {
    const gqa = 16 * 2 * 4 * 256 * 2 * n / 1e9;
    const mha = 16 * 2 * 24 * 256 * 2 * n / 1e9;
    out.innerHTML = `${t('w.kv.gqa')}: <b>${gqa.toFixed(2)} GB</b> · ${t('w.kv.mha')}: <b>${mha.toFixed(2)} GB</b>`;
  };
  const sl = slider(1, 262, 32, 1, v => calc(v * 1000));
  const ctxOut = el('div', 'mono wlabel');
  sl.addEventListener('input', () => { ctxOut.textContent = sl.value + 'K'; });
  ctxOut.textContent = '32K';
  root.append(lbl(t('w.gqa.hover')), qRow, kvRow, ctl(t('w.ctx'), sl, ctxOut), out, el('div', 'small-note', '', t('w.gqa.note')));
  calc(32000);
  return root;
}

// ── conv1d: окно из 4 токенов ──
function convWidget(): HTMLElement {
  const root = box();
  const N = 14;
  const row = el('div', '', 'display:flex;gap:4px;justify-content:center');
  const toks: HTMLElement[] = [];
  for (let i = 0; i < N; i++) {
    const c = el('div', 'mono', `width:30px;height:30px;border-radius:5px;background:rgba(143,178,189,0.35);
      border:1px solid rgba(130,175,170,0.4);display:flex;align-items:center;justify-content:center;font-size:10px;color:#3f6f6a;transition:all 130ms`, `t${i}`);
    toks.push(c); row.appendChild(c);
  }
  const cap = el('div', 'mono wlabel', 'align-self:center');
  const setP = (p: number) => {
    toks.forEach((c, i) => {
      const inWin = i <= p && i >= p - 3;
      c.style.background = i === p ? 'var(--accent-deep)' : inWin ? 'rgba(224,164,92,0.55)' : 'rgba(143,178,189,0.35)';
      c.style.color = i === p ? '#fff' : inWin ? '#7a4a12' : '#3f6f6a';
      c.style.transform = inWin ? 'translateY(-3px)' : '';
    });
    cap.textContent = `${t('w.out')} t${p} = f(t${Math.max(0, p - 3)} … t${p})`;
  };
  root.append(row, cap, ctl(t('w.pos'), slider(3, N - 1, 6, 1, setP)), el('div', 'small-note', '', t('w.conv.note')));
  setP(6);
  return root;
}

// ── swish/silu-гейт ──
function gateWidget(): HTMLElement {
  const root = box();
  const W = 420, H = 150;
  const svg = svgEl('svg', { width: W, height: H, style: 'align-self:center' });
  const X = (x: number) => (x + 6) / 12 * (W - 20) + 10;
  const Y = (y: number) => H - 24 - (y + 1) / 7 * (H - 40);
  const silu = (x: number) => x / (1 + Math.exp(-x));
  let d = '';
  for (let x = -6; x <= 6; x += 0.15) d += (d ? ' L ' : 'M ') + X(x).toFixed(1) + ' ' + Y(silu(x)).toFixed(1);
  svg.appendChild(svgEl('line', { x1: 10, y1: Y(0), x2: W - 10, y2: Y(0), stroke: 'var(--line-strong)' }));
  svg.appendChild(svgEl('line', { x1: X(0), y1: 8, x2: X(0), y2: H - 16, stroke: 'var(--line-strong)' }));
  svg.appendChild(svgEl('path', { d, fill: 'none', stroke: 'var(--accent-deep)', 'stroke-width': 2.4 }));
  const dot = svgEl('circle', { r: 6, fill: 'var(--bad)', stroke: '#fff', 'stroke-width': 1.5 });
  svg.appendChild(dot);
  const out = el('div', 'mono wlabel');
  const set = (x: number) => {
    dot.setAttribute('cx', String(X(x))); dot.setAttribute('cy', String(Y(silu(x))));
    out.textContent = `silu(${x.toFixed(1)}) = ${silu(x).toFixed(2)}`;
  };
  root.append(svg as any, ctl(t('w.x'), slider(-6, 6, 1.5, 0.1, set), out), el('div', 'small-note', '', t('w.gate.note')));
  set(1.5);
  return root;
}

// ── RMSNorm: инвариантность к масштабу ──
function normWidget(): HTMLElement {
  const root = box();
  const N = 8;
  let base = randVec(N).map(x => x * 2);
  const mk = (title: string) => {
    const w = el('div', '', 'display:flex;flex-direction:column;gap:6px;align-items:center;flex:1');
    const row = el('div', '', 'display:flex;gap:5px;align-items:flex-end;height:70px');
    const bars: HTMLElement[] = [];
    for (let i = 0; i < N; i++) {
      const b = el('div', '', 'width:16px;border-radius:3px 3px 0 0;background:#8fb2bd;transition:height 120ms');
      bars.push(b); row.appendChild(b);
    }
    w.append(row, lbl(title));
    return { w, bars };
  };
  const L = mk(t('w.before')), R = mk(t('w.after'));
  const wrap = el('div', '', 'display:flex;gap:30px');
  wrap.append(L.w, R.w);
  const out = el('div', 'mono wlabel');
  const set = (s: number) => {
    const scale = Math.pow(10, s);
    const v = base.map(x => x * scale);
    const rms = Math.sqrt(v.reduce((a, x) => a + x * x, 0) / N) || 1;
    L.bars.forEach((b, i) => b.style.height = Math.min(70, Math.abs(v[i]) * 12) + 'px');
    R.bars.forEach((b, i) => b.style.height = (Math.abs(v[i] / rms) * 26) + 'px');
    out.textContent = `×${scale < 1 ? scale.toFixed(2) : scale.toFixed(0)}`;
  };
  root.append(wrap, ctl(t('w.scale'), slider(-1, 2, 0, 0.05, set), out),
    btn(t('w.newvec'), () => { base = randVec(N).map(x => x * 2); set(0); }),
    el('div', 'small-note', '', t('w.norm.note')));
  set(0);
  return root;
}

// ── pos_embed: сетка 48×48 и интерполяция ──
function posembedWidget(): HTMLElement {
  const root = box();
  const area = el('div', '', `align-self:center;position:relative;width:240px;height:240px;border:1px solid var(--line-strong);border-radius:6px;
    background-image:repeating-linear-gradient(0deg,rgba(150,130,200,0.25) 0 1px,transparent 1px 5px),repeating-linear-gradient(90deg,rgba(150,130,200,0.25) 0 1px,transparent 1px 5px)`);
  const ov = el('div', '', 'position:absolute;left:0;top:0;border:2px solid var(--accent);border-radius:4px;background:rgba(255,138,61,0.08);transition:all 150ms');
  area.appendChild(ov);
  const out = el('div', 'mono wlabel');
  const set = (px: number) => {
    const patches = Math.round(px / 16 / 2); // патч 16 + merge 2×2
    const frac = Math.min(1, patches / 48);
    ov.style.width = (frac * 240) + 'px'; ov.style.height = (frac * 240) + 'px';
    out.textContent = `${px}px → ${patches}×${patches}`;
  };
  root.append(lbl(t('w.grid')), area, ctl(t('w.img'), slider(256, 1536, 640, 32, set), out),
    el('div', 'small-note', '', t('w.pos.note')));
  set(640);
  return root;
}

// ── spatial merge 2×2 ──
function mergeWidget(): HTMLElement {
  const root = box();
  const vis = kindOf('vision');
  const grid = el('div', '', 'display:grid;grid-template-columns:repeat(8,26px);gap:3px;align-self:center;transition:all 200ms');
  const cells: HTMLElement[] = [];
  for (let i = 0; i < 64; i++) {
    const r = Math.floor(i / 8), c = i % 8;
    const g = (Math.floor(r / 2) * 4 + Math.floor(c / 2));
    const cell = el('div', '', `width:26px;height:26px;border-radius:3px;background:${vis.solid};opacity:${0.35 + (g % 4) * 0.16};transition:all 260ms`);
    cells.push(cell); grid.appendChild(cell);
  }
  let merged = false;
  const out = el('div', 'mono wlabel', 'align-self:center', '64 → 64');
  const toggle = () => {
    merged = !merged;
    cells.forEach((cell, i) => {
      const r = Math.floor(i / 8), c = i % 8;
      if (merged) {
        const inCorner = r % 2 === 0 && c % 2 === 0;
        cell.style.transform = inCorner ? 'scale(1.35)' : 'scale(0.1)';
        cell.style.opacity = inCorner ? '0.9' : '0';
      } else {
        cell.style.transform = ''; cell.style.opacity = String(0.35 + ((Math.floor(r / 2) * 4 + Math.floor(c / 2)) % 4) * 0.16);
      }
    });
    out.textContent = merged ? '64 → 16' : '64 → 64';
  };
  root.append(grid, out, btn(t('w.merge.btn'), toggle), el('div', 'small-note', '', t('w.merge.note')));
  return root;
}

// ── MTP: сколько токенов за проход ──
function mtpWidget(): HTMLElement {
  const root = box();
  const bar = el('div', '', 'height:26px;border-radius:6px;background:var(--line);overflow:hidden;display:flex');
  const b1 = el('div', '', 'background:#e0955c;height:100%;width:50%');
  const b2 = el('div', '', 'background:var(--good);height:100%;width:30%;transition:width 120ms');
  bar.append(b1, b2);
  const out = el('div', 'mono', 'font-size:14px;color:var(--ink-soft);text-align:center');
  const set = (p: number) => {
    b2.style.width = (p / 2) + '%';
    out.innerHTML = `${t('w.tps')}: <b>${(1 + p / 100).toFixed(2)}</b>`;
  };
  root.append(bar, ctl(t('w.accept'), slider(0, 100, 60, 1, set)), out, el('div', 'small-note', '', t('w.mtp.note')));
  set(60);
  return root;
}

// ── память: KV-кеш растёт, состояние — нет ──
function costWidget(): HTMLElement {
  const root = box();
  const rows = el('div', '', 'display:flex;flex-direction:column;gap:10px');
  const mk = (color: string) => {
    const r = el('div', '', 'display:flex;align-items:center;gap:10px');
    const track = el('div', '', 'flex:1;height:14px;border-radius:7px;background:var(--line);overflow:hidden');
    const fill = el('div', '', `height:100%;border-radius:7px;background:${color};transition:width 120ms`);
    track.appendChild(fill);
    const val = el('div', 'mono', 'font-size:12px;width:88px;text-align:right;color:var(--ink-soft)');
    const name = el('div', 'mono wlabel', 'width:210px');
    r.append(name, track, val);
    rows.appendChild(r);
    return { fill, val, name };
  };
  const kv = mk(kindOf('attn').solid), st = mk(kindOf('lin').solid);
  kv.name.textContent = t('w.kvline');
  st.name.textContent = t('w.stateline');
  const ctxOut = el('div', 'mono wlabel');
  const set = (nK: number) => {
    const n = nK * 1000;
    const kvGB = 16 * 2 * 4 * 256 * 2 * n / 1e9;                   // 16 полных слоёв, bf16
    const stGB = 48 * 48 * 128 * 128 * 4 / 1e9;                    // 48 слоёв × состояние fp32, константа
    const mx = Math.max(kvGB, stGB);
    kv.fill.style.width = (kvGB / mx * 100) + '%';
    st.fill.style.width = Math.max(1.5, stGB / mx * 100) + '%';
    kv.val.textContent = kvGB >= 1 ? kvGB.toFixed(2) + ' GB' : (kvGB * 1000).toFixed(0) + ' MB';
    st.val.textContent = (stGB * 1000).toFixed(0) + ' MB';
    ctxOut.textContent = nK + 'K';
  };
  root.append(rows, ctl(t('w.ctx'), slider(1, 262, 64, 1, set), ctxOut), el('div', 'small-note', '', t('w.cost.note')));
  set(64);
  return root;
}
