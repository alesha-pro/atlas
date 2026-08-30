// Лаборатория связей: любая метрика против любой, точка = тензор.
import { fmtN, type MetricDef, type Tensor } from '../data';
import { colorForTensor, kindOf } from '../color';
import { t as tr } from '../i18n';
import { el, svgEl } from '../world';
import { tip, hideTip, type Store } from '../store';

const W = 1300, PLOT_W = 1180, PLOT_H = 780, ML = 74, MT = 16;

const PRESETS: [string, string][] = [
  ['kurt', 'int4'], ['hot', 'int4'], ['srank', 'int4'], ['size', 'int4'], ['kurt', 'int8'],
];

export function buildScatter(store: Store, X: number, Y: number): {
  root: HTMLElement; rect: { x: number; y: number; w: number; h: number };
} {
  const m = store.model;
  let xKey = 'kurt', yKey = 'int4';

  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `
    <div class="section-tag mono">${tr('sec.scatter.tag')}</div>
    <div class="section-title">${tr('sec.scatter.title')}</div>
    <div class="section-sub">${tr('sec.scatter.sub')}</div>`));

  const card = el('div', 'card', `position:relative;width:${W}px;padding:20px 24px 22px;display:flex;flex-direction:column;gap:14px`);
  root.appendChild(card);

  const controls = el('div', 'no-pan', 'display:flex;align-items:center;gap:9px;flex-wrap:wrap');
  const selX = mkSelect(m.metrics, xKey);
  const selY = mkSelect(m.metrics, yKey);
  controls.appendChild(el('div', 'mono', 'font-size:11.5px;color:var(--faint)', tr('scatter.x')));
  controls.appendChild(selX);
  controls.appendChild(el('div', 'mono', 'font-size:11.5px;color:var(--faint)', tr('scatter.y')));
  controls.appendChild(selY);
  const presetWrap = el('div', '', 'display:flex;gap:6px;flex-wrap:wrap;margin-left:10px');
  controls.appendChild(presetWrap);
  card.appendChild(controls);

  const svgWrap = el('div', '', `position:relative;width:${PLOT_W + ML + 10}px;height:${PLOT_H + 60}px`);
  card.appendChild(svgWrap);
  const svg = svgEl('svg', { width: PLOT_W + ML + 10, height: PLOT_H + 60, class: 'no-pan' });
  svgWrap.appendChild(svg as any);

  const foot = el('div', 'small-note', 'max-width:1100px');
  card.appendChild(foot);

  const pts = m.tensors.filter(t => t.is2d);

  function draw() {
    const dx = m.metrics[xKey], dy = m.metrics[yKey];
    svg.innerHTML = '';

    // оси и сетка
    const gx0 = ML, gy0 = MT, gw = PLOT_W, gh = PLOT_H;
    svg.appendChild(svgEl('rect', { x: gx0, y: gy0, width: gw, height: gh, fill: 'rgba(255,253,248,0.55)', stroke: 'rgba(120,106,84,0.18)', rx: 6 }));

    const px = (t: Tensor) => { const v = dx.get(t); return v == null ? null : gx0 + norm(v, dx) * gw; };
    const py = (t: Tensor) => { const v = dy.get(t); return v == null ? null : gy0 + (1 - norm(v, dy)) * gh; };

    for (const [axis, d] of [['x', dx], ['y', dy]] as const) {
      for (let i = 0; i <= 4; i++) {
        const f = i / 4;
        const val = d.lo + f * (d.hi - d.lo);
        const raw = d.log ? Math.pow(10, val) : val;
        const label = d.fmt(raw) + (d.unit && !d.log ? '' : '');
        if (axis === 'x') {
          const x = gx0 + f * gw;
          svg.appendChild(svgEl('line', { x1: x, y1: gy0, x2: x, y2: gy0 + gh, stroke: 'rgba(120,106,84,0.10)' }));
          svg.appendChild(text(label, x, gy0 + gh + 18, 'middle'));
        } else {
          const y = gy0 + (1 - f) * gh;
          svg.appendChild(svgEl('line', { x1: gx0, y1: y, x2: gx0 + gw, y2: y, stroke: 'rgba(120,106,84,0.10)' }));
          svg.appendChild(text(label, gx0 - 8, y + 4, 'end'));
        }
      }
    }
    svg.appendChild(text(`${dx.label}${dx.log ? ' · log' : ''}`, gx0 + gw / 2, gy0 + gh + 40, 'middle', 12.5));
    const yl = text(`${dy.label}${dy.log ? ' · log' : ''}`, 0, 0, 'middle', 12.5);
    yl.setAttribute('transform', `translate(16,${gy0 + gh / 2}) rotate(-90)`);
    svg.appendChild(yl);

    // точки
    for (const t of pts) {
      const x = px(t), y = py(t);
      if (x == null || y == null) continue;
      const r = 2 + Math.max(0, Math.log10(t.numel) - 4) * 1.5;
      const c = svgEl('circle', {
        cx: x.toFixed(1), cy: y.toFixed(1), r: r.toFixed(1),
        fill: colorForTensor(t, store.md),
        stroke: 'rgba(60,50,35,0.35)', 'stroke-width': 0.6,
        style: 'cursor:pointer',
      });
      c.addEventListener('click', () => store.select({ type: 'tensor', tensor: t }));
      c.addEventListener('mouseenter', e => {
        const vx = dx.get(t), vy = dy.get(t);
        tip(`<div class="mono" style="font-size:12px;margin-bottom:3px">${t.short}</div>
          ${dx.label}: <span class="mono">${vx == null ? tr('na') : dx.fmt(vx)}</span> ·
          ${dy.label}: <span class="mono">${vy == null ? tr('na') : dy.fmt(vy)}</span> · ${fmtN(t.numel)}`,
          (e as MouseEvent).clientX, (e as MouseEvent).clientY);
      });
      c.addEventListener('mouseleave', hideTip);
      if (store.sel.type === 'tensor' && store.sel.tensor.idx === t.idx) {
        c.setAttribute('stroke', '#ff8a3d');
        c.setAttribute('stroke-width', '2.5');
      }
      svg.appendChild(c);
    }

    foot.innerHTML = footNote(xKey, yKey);
  }

  function norm(v: number, d: MetricDef) {
    return Math.max(0, Math.min(1, (d.transform(v) - d.lo) / (d.hi - d.lo)));
  }

  for (const [kx, ky] of PRESETS) {
    const c = el('div', 'chip mini no-pan', '', tr(`preset.${kx}.${ky}`));
    c.addEventListener('click', () => {
      xKey = kx; yKey = ky;
      (selX as HTMLSelectElement).value = kx;
      (selY as HTMLSelectElement).value = ky;
      draw();
    });
    presetWrap.appendChild(c);
  }
  selX.addEventListener('change', () => { xKey = (selX as HTMLSelectElement).value; draw(); });
  selY.addEventListener('change', () => { yKey = (selY as HTMLSelectElement).value; draw(); });
  store.addEventListener('metric', draw);
  store.addEventListener('sel', draw);
  draw();

  return { root, rect: { x: X, y: Y, w: W, h: PLOT_H + 220 } };
}

function mkSelect(metrics: Record<string, MetricDef>, value: string): HTMLSelectElement {
  const s = document.createElement('select');
  s.className = 'plain no-pan';
  for (const d of Object.values(metrics)) {
    const o = document.createElement('option');
    o.value = d.key; o.textContent = d.label;
    s.appendChild(o);
  }
  s.value = value;
  return s;
}

function text(str: string, x: number, y: number, anchor = 'start', size = 10.5): SVGElement {
  const t = svgEl('text', { x, y, 'text-anchor': anchor, fill: '#8c8578', 'font-size': size, 'font-family': "'IBM Plex Mono',monospace" });
  t.textContent = str;
  return t;
}

function footNote(x: string, y: string): string {
  if (x === 'kurt' && (y === 'int4' || y === 'int8')) return tr('scatter.note.kurt');
  if (x === 'hot') return tr('scatter.note.hot');
  if (x === 'srank') return tr('scatter.note.srank');
  if (x === 'size') return tr('scatter.note.size');
  return tr('scatter.note.default');
}
