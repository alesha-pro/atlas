// Живая модель: внутренности, снятые с реальных прямых проходов.
// Данные — public/models/<slug>/live.json (atlas_live.py + reduce_live.py).
// Каждая карточка самодостаточна: нет данных — карточки нет.
import { kindOf, ramp } from '../color';
import { t as tr } from '../i18n';
import { el, svgEl } from '../world';
import { tip, hideTip, type Store } from '../store';

// ── контракт live.json (schema 1) ──
type Series = Record<string, (number | null)[]>;

export interface LiveActqSite { int8: number; fp8: number; n?: number }
export interface LiveFlow {
  h_rms: Series; delta_rms: Series; out_ratio: (number | null)[];
  n_out_dims: (number | null)[]; h_rms0?: Series;
  actq?: Record<string, LiveActqSite>;
  io_cos?: (number | null)[];
}
export interface LiveAttn {
  layers: number[]; ent: number[]; ent_std?: number[];
  first: number[]; diag: number[]; gate: number[];
  prof: number[][]; decay_edges?: number[];
}
export interface LiveMap { mean: number[][]; star: number[][]; star_head: number }
export interface LiveAttnMaps { prompt: string; tokens: string[]; maps: Record<string, LiveMap> }
export interface LiveLinAttn {
  layers: number[]; beta: Series; g_mean: Series; state_rms: Series;
  half_life?: (number | null)[]; layers_lambda?: Record<string, number[]>;
}
export interface LiveNeuronEx { layer: number; neuron: number; en: number; code: number; agent: number; pol: number }
export interface LiveNeurons {
  n: number; fire_mean: number[]; fire_p99: number[];
  dead_frac: number[]; conc: number[]; spec_frac: number[];
  heat: number[][]; examples: LiveNeuronEx[];
}
export interface LiveFragility { kl: number[]; logit_cos: number[] }
export interface LiveVision { n_img_tokens: number; img_share: Record<string, number> }
export interface Live {
  schema: number;
  meta: Record<string, unknown>;
  flow?: LiveFlow;
  attn?: LiveAttn;
  attn_maps?: LiveAttnMaps;
  linattn?: LiveLinAttn;
  neurons?: LiveNeurons;
  fragility?: LiveFragility;
  vision?: LiveVision;
}

const DOMAINS = ['en', 'code', 'agent'];
const DOM_COLOR: Record<string, string> = {
  en: kindOf('attn').solid, code: kindOf('mlp').solid, agent: kindOf('lin').solid,
  all: '#8a7a5f',
};

// высокое значение — терракота (обратная рампа атласа), низкое — волна
function heatColor(t: number): string {
  return ramp(1 - Math.max(0, Math.min(1, t)));
}

// лог-шкала 1e-4..1: бумага → янтарь → терракота → тёмный кирпич
function attnColor(v: number): [number, number, number] {
  const a = Math.max(0, Math.min(1, (Math.log10(Math.max(v, 1e-4)) + 4) / 4));
  const stops: [number, number, number][] = [
    [248, 246, 238], [243, 213, 160], [222, 143, 88], [130, 52, 34],
  ];
  const k = a * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(k));
  const f = k - i;
  return [
    Math.round(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f),
    Math.round(stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f),
    Math.round(stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f),
  ];
}

function linePath(vals: (number | null)[], px: (i: number) => number, py: (v: number) => number): string {
  let d = '', started = false;
  vals.forEach((v, i) => {
    if (v == null || !isFinite(v)) { started = false; return; }
    d += (started ? ' L ' : 'M ') + px(i).toFixed(1) + ' ' + py(v).toFixed(1);
    started = true;
  });
  return d;
}

function cardShell(w: number, title: string, sub?: string, hint = false): { card: HTMLElement; body: HTMLElement } {
  const card = el('div', 'card', `width:${w}px;padding:20px 24px;display:flex;flex-direction:column;gap:14px`);
  card.innerHTML = `<div style="display:flex;align-items:baseline;justify-content:space-between;gap:14px">
      <div class="eyebrow mono">${title}</div>
      ${hint ? `<div class="mono" style="font-size:10px;color:var(--accent-ink);white-space:nowrap">✛ ${tr('live.hint')}</div>` : ''}
    </div>` +
    (sub ? `<div class="note" style="max-width:${w - 60}px">${sub}</div>` : '');
  const body = el('div', '', 'display:flex;flex-direction:column;gap:12px');
  card.appendChild(body);
  return { card, body };
}

function yLabel(svg: SVGElement, x: number, y: number, text: string): void {
  const e = svgEl('text', { x, y, 'font-size': 10, fill: 'var(--ghost)' });
  e.textContent = text;
  svg.appendChild(e);
}

// ─────────────────────────── flow ───────────────────────────

function flowCard(live: Live, store: Store): HTMLElement | null {
  const f = live.flow;
  const hasStream = !!f?.h_rms?.all?.length;
  if (!hasStream && !f?.io_cos?.length) return null;
  const n = (hasStream ? f.h_rms.all.length : (f.io_cos?.length || 1));
  const W = 1460, H = hasStream ? 400 : 240, W2 = W - 70, H2 = 180;
  const { card, body } = cardShell(W, tr('live.flow.title'), tr('live.flow.sub'), true);
  let domSel = 'all', keySel: 'h_rms' | 'delta_rms' = 'h_rms';

  const ctl = el('div', 'no-pan', 'display:flex;gap:8px;align-items:center;flex-wrap:wrap');
  if (hasStream) for (const d of ['all', ...DOMAINS]) {
    const b = el('div', 'mono no-pan',
      `font-size:10px;padding:3px 9px;border-radius:10px;cursor:pointer;border:1px solid var(--line-strong);
       background:${d === 'all' ? 'var(--line)' : 'transparent'};color:${DOM_COLOR[d]}`, tr('live.dom.' + d));
    b.onclick = (e: MouseEvent) => {
      e.stopPropagation();
      domSel = d;
      ctl.querySelectorAll('[data-dom]').forEach(x => {
        const elx = x as HTMLElement;
        elx.style.background = elx.dataset.dom === d ? 'var(--line)' : 'transparent';
      });
      draw();
    };
    b.dataset.dom = d;
    ctl.appendChild(b);
  }
  if (hasStream) for (const [k, lbl] of [['h_rms', 'live.flow.h'], ['delta_rms', 'live.flow.delta']] as const) {
    const b = el('div', 'mono no-pan',
      `font-size:10px;padding:3px 9px;border-radius:10px;cursor:pointer;border:1px solid var(--line-strong);
       background:${k === 'h_rms' ? 'var(--line)' : 'transparent'}`, tr(lbl));
    b.onclick = (e: MouseEvent) => {
      e.stopPropagation();
      keySel = k;
      ctl.querySelectorAll('[data-key]').forEach(x => {
        const elx = x as HTMLElement;
        elx.style.background = elx.dataset.key === k ? 'var(--line)' : 'transparent';
      });
      draw();
    };
    b.dataset.key = k;
    ctl.appendChild(b);
  }
  if (!hasStream) ctl.appendChild(el('div', 'mono',
    'font-size:10px;padding:3px 9px;border-radius:10px;border:1px dashed var(--line-strong);color:var(--faint)',
    tr('live.pending')));
  body.appendChild(ctl);

  const chartWrap = el('div', 'no-pan', `position:relative;width:${W}px;height:${H}px;cursor:crosshair`);
  body.appendChild(chartWrap);
  body.appendChild(el('div', 'small-note', `max-width:${W - 60}px`, tr('live.flow.foot')));

  function draw() {
    chartWrap.innerHTML = '';
    const svg = svgEl('svg', { width: W, height: H });
    chartWrap.appendChild(svg);
    const px = (i: number) => 46 + (i / Math.max(1, n - 1)) * (W2 - 14);
    const series = hasStream ? (f[keySel][domSel] || []) : [];
    const vals = series.filter((v): v is number => v != null);
    if (hasStream && !vals.length) return;
    let lo = 0, hi = 1;
    const py = (v: number) => 12 + (1 - (v - lo) / (hi - lo)) * (H2 - 24);
    if (hasStream) {
      lo = Math.min(...vals); hi = Math.max(...vals);
      const pad = (hi - lo) * 0.15 || 1;
      lo -= pad; hi += pad;
    }

    if (hasStream)
      svg.appendChild(svgEl('rect', { x: 38, y: 4, width: W2, height: H2, fill: 'var(--chart-bg)', stroke: 'var(--line)', rx: 6 }));

    if (hasStream) {
      // вклад слоя — мягкая подложка при просмотре ||h||
      if (keySel === 'h_rms' && f.delta_rms?.[domSel]) {
        const dv = f.delta_rms[domSel];
        const dvals = dv.filter((v): v is number => v != null);
        let dlo = Math.min(...dvals), dhi = Math.max(...dvals);
        const dp = (dhi - dlo) * 0.15 || 1; dlo -= dp; dhi += dp;
        const dpy = (v: number) => 12 + (1 - (v - dlo) / (dhi - dlo)) * (H2 - 24);
        let area = '';
        dv.forEach((v, i) => {
          if (v == null) return;
          area += (area ? ' L ' : 'M ') + px(i).toFixed(1) + ' ' + dpy(v).toFixed(1);
        });
        if (area) {
          area += ` L ${px(n - 1).toFixed(1)} ${H2 - 10} L ${px(0).toFixed(1)} ${H2 - 10} Z`;
          svg.appendChild(svgEl('path', { d: area, fill: 'rgba(143,178,189,0.16)' }));
          svg.appendChild(svgEl('path', { d: linePath(dv, px, dpy), fill: 'none', stroke: 'rgba(63,111,106,0.45)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
        }
      }

      svg.appendChild(svgEl('path', { d: linePath(series, px, py), fill: 'none', stroke: DOM_COLOR[domSel], 'stroke-width': 2 }));
      series.forEach((v, i) => {
        if (v == null) return;
        svg.appendChild(svgEl('circle', {
          cx: px(i), cy: py(v), r: i % 4 === 3 ? 3.4 : 2,
          fill: keySel === 'h_rms' ? kindOf(i % 4 === 3 ? 'attn' : 'lin').solid : DOM_COLOR[domSel],
          opacity: 0.9,
        }));
      });
      for (const frac of [0, 0.5, 1])
        yLabel(svg, 6, py(hi - frac * (hi - lo)) + 4, (hi - frac * (hi - lo)).toFixed(1));
    }

    // клик по графику → выбрать слой на стене
    const layerAt = (e: MouseEvent) => {
      const r = svg.getBoundingClientRect();
      const i = Math.round(((e.clientX - r.left) / r.width * W - 46) / ((W2 - 14) / Math.max(1, n - 1)));
      return Math.max(0, Math.min(n - 1, i));
    };
    svg.addEventListener('click', (e: MouseEvent) => {
      const L = store.model.langLayers[layerAt(e)];
      if (L) store.select({ type: 'layer', layer: L });
    });
    const guide = svgEl('line', { y1: 6, y2: H - 6, stroke: 'var(--accent)', 'stroke-width': 1, 'stroke-dasharray': '3 4', opacity: 0 });
    svg.appendChild(guide);
    svg.addEventListener('mousemove', (e: MouseEvent) => {
      const i = layerAt(e);
      const L = store.model.langLayers[i];
      if (!L) return;
      guide.setAttribute('x1', String(px(i)));
      guide.setAttribute('x2', String(px(i)));
      guide.setAttribute('opacity', '0.7');
      const parts = [`<b>${tr('live.layer')} ${L.label}</b> · ${L.kind === 'full' ? tr('kind.full') : tr('kind.linear')}`];
      if (hasStream) {
        const hv = f[keySel]?.[domSel]?.[i];
        if (hv != null) parts.push(`${tr(keySel === 'h_rms' ? 'live.flow.h' : 'live.flow.delta')} ${Number(hv).toFixed(2)}`);
      }
      const cv = f.io_cos?.[i];
      if (cv != null) parts.push(`${tr('live.flow.io')} ${Number(cv).toFixed(3)}`);
      tip(parts.join('<br>'), e.clientX, e.clientY);
    });
    svg.addEventListener('mouseleave', () => { hideTip(); guide.setAttribute('opacity', '0'); });

    // нижние полосы: io_cos + каналы-выбросы
    let y0 = hasStream ? H2 + 26 : 12;
    if (f.io_cos?.length) {
      const bandH = hasStream ? 52 : 150;
      yLabel(svg, 6, y0 + 4, tr('live.flow.io'));
      const cio = f.io_cos.filter((v): v is number => v != null);
      let clo = Math.min(...cio), chi = Math.max(...cio);
      const cp = (chi - clo) * 0.2 || 0.01; clo -= cp; chi += cp;
      const py2 = (v: number) => y0 + 12 + (1 - (v - clo) / (chi - clo)) * bandH;
      svg.appendChild(svgEl('rect', { x: 38, y: y0 + 6, width: W2, height: bandH + 12, fill: 'var(--chart-bg)', stroke: 'var(--line)', rx: 6 }));
      svg.appendChild(svgEl('path', { d: linePath(f.io_cos, px, py2), fill: 'none', stroke: 'rgba(122,86,140,0.65)', 'stroke-width': 1.6 }));
      f.io_cos.forEach((v, i) => {
        if (v == null) return;
        svg.appendChild(svgEl('circle', { cx: px(i), cy: py2(v), r: i % 4 === 3 ? 3 : 1.8, fill: i % 4 === 3 ? kindOf('attn').solid : 'rgba(122,86,140,0.75)' }));
      });
      yLabel(svg, 8, py2(chi - cp) + 4, (chi - cp).toFixed(2));
      yLabel(svg, 8, py2(clo + cp) + 4, (clo + cp).toFixed(2));
      if (!hasStream) for (let i = 0; i < n; i += 8) yLabel(svg, px(i) - 6, y0 + bandH + 34, String(i));
      y0 += bandH + 16;
    }
    if (f.n_out_dims?.some(v => v != null)) {
      yLabel(svg, 6, y0 + 4, tr('live.flow.outdims'));
      const mx = Math.max(...f.n_out_dims.filter((v): v is number => v != null)) || 1;
      f.n_out_dims.forEach((v, i) => {
        if (v == null) return;
        svg.appendChild(svgEl('rect', {
          x: px(i) - 3, y: y0 + 56 - (v / mx) * 50, width: 6, height: (v / mx) * 50,
          fill: heatColor(v / mx), rx: 1,
        }));
      });
    }
  }
  draw();
  return card;
}

// ─────────────────────────── actq ───────────────────────────

function actqCard(live: Live): HTMLElement | null {
  const sites = live.flow?.actq;
  if (!sites || !Object.keys(sites).length) return null;
  const entries = Object.entries(sites).sort((a, b) => b[1].int8 - a[1].int8);
  const W = 640, rowH = 26, chartH = entries.length * rowH + 30;
  const { card, body } = cardShell(W, tr('live.actq.title'), tr('live.actq.sub'));
  const wrap = el('div', 'no-pan', `position:relative;width:${W - 48}px;height:${chartH}px`);
  body.appendChild(wrap);
  const svg = svgEl('svg', { width: W - 48, height: chartH });
  wrap.appendChild(svg);
  const lblW = 128, barW = W - 48 - lblW - 72;
  const lo = 20, hi = 50;
  const bx = (v: number) => lblW + ((v - lo) / (hi - lo)) * barW;
  for (const g of [20, 30, 40, 50]) {
    svg.appendChild(svgEl('line', { x1: bx(g), x2: bx(g), y1: 4, y2: chartH - 18, stroke: 'var(--line)' }));
    yLabel(svg, bx(g) - 6, chartH - 6, String(g));
  }
  entries.forEach(([site, s], i) => {
    const y = 8 + i * rowH;
    const dot = site.indexOf('.');
    const grp = dot > 0 ? site.slice(0, dot) : 'x';
    const kind = grp === 'attn' ? 'attn' : grp === 'mlp' ? 'mlp' : 'lin';
    yLabel(svg, 0, y + 12, site.slice(dot + 1) || site);
    (svg.lastChild as SVGElement).setAttribute('fill', kindOf(kind).fg);
    svg.appendChild(svgEl('rect', { x: lblW, y: y + 1, width: Math.max(2, bx(s.int8) - lblW), height: 8.5, fill: kindOf(kind).solid, rx: 2 }));
    svg.appendChild(svgEl('rect', { x: lblW, y: y + 11, width: Math.max(2, bx(s.fp8) - lblW), height: 8.5, fill: 'var(--line-strong)', rx: 2 }));
    yLabel(svg, Math.max(bx(s.int8), bx(s.fp8)) + 6, y + 12, s.int8.toFixed(1));
  });
  body.appendChild(el('div', 'mono', 'font-size:10px;color:var(--faint);display:flex;gap:16px', `
    <span>■ INT8 <span style="opacity:.75">${tr('live.actq.dyn')}</span></span>
    <span style="opacity:.55">■ FP8 e4m3</span>`));
  return card;
}

// ─────────────────────────── attention aggregate ───────────────────────────

function attnCard(live: Live, store: Store): HTMLElement | null {
  const a = live.attn;
  if (!a?.layers?.length) return null;
  const W = 980, H = 360;
  const { card, body } = cardShell(W, tr('live.attn.title'), tr('live.attn.sub'), true);
  const wrap = el('div', 'no-pan', `position:relative;width:${W}px;height:${H}px`);
  body.appendChild(wrap);
  const svg = svgEl('svg', { width: W, height: H });
  wrap.appendChild(svg);

  const n = a.layers.length;
  const chartW = 560, entH = 190, x0 = 46;
  const px = (i: number) => x0 + (i / Math.max(1, n - 1)) * (chartW - 10);
  svg.appendChild(svgEl('rect', { x: x0 - 8, y: 4, width: chartW, height: entH + 16, fill: 'var(--chart-bg)', stroke: 'var(--line)', rx: 6 }));
  yLabel(svg, 2, 16, tr('live.attn.ent'));
  a.ent.forEach((v, i) => {
    const h = v * entH;
    svg.appendChild(svgEl('rect', {
      x: px(i) - 5, y: 12 + entH - h, width: 10, height: h,
      fill: ramp(1 - v), rx: 2,
    }));
  });
  for (const f of [0, 0.5, 1])
    yLabel(svg, x0 - 34, 12 + entH - f * entH + 4, f.toFixed(1));

  // точечные ряды: sink / near / gate, каждый в своей нормировке
  const rows: [string, number[], string][] = [
    [tr('live.attn.sink'), a.first, 'rgba(122,86,140,0.75)'],
    [tr('live.attn.near'), a.diag, kindOf('lin').solid],
    [tr('live.attn.gate'), a.gate, kindOf('out').solid],
  ];
  rows.forEach(([lbl, vals, col], r) => {
    const y0 = entH + 40 + r * 26;
    yLabel(svg, 2, y0 + 4, lbl);
    (svg.lastChild as SVGElement).setAttribute('fill', col);
    const vv = vals.filter(v => isFinite(v));
    const lo = Math.min(...vv), hi = Math.max(...vv);
    vals.forEach((v, i) => {
      if (!isFinite(v)) return;
      const t = (v - lo) / (hi - lo || 1);
      svg.appendChild(svgEl('circle', { cx: px(i), cy: y0 - t * 14, r: 3, fill: col, opacity: 0.9 }));
    });
  });
  a.layers.forEach((l, i) => {
    if (i % 2 === 0) yLabel(svg, px(i) - 8, entH + 40 + rows.length * 26 + 4, String(l));
  });

  // профиль затухания: слои × дистанция
  if (a.prof?.length) {
    const hx = x0 + chartW + 44, cellW = 21, cellH = 13;
    yLabel(svg, hx, 14, tr('live.attn.decay'));
    a.prof.forEach((row, li) => {
      row.forEach((v, di) => {
        svg.appendChild(svgEl('rect', {
          x: hx + di * (cellW + 1), y: 22 + li * (cellH + 1), width: cellW, height: cellH,
          fill: heatColor(Math.min(1, v / 0.45)), rx: 2,
        }));
      });
    });
    (a.decay_edges || [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]).forEach((e, di) => {
      if (di % 2 === 0)
        yLabel(svg, hx + di * (cellW + 1), 30 + a.prof.length * (cellH + 1), '≥' + e);
    });
  }

  svg.addEventListener('mousemove', (e: MouseEvent) => {
    const r = svg.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width * W - x0) / ((chartW - 10) / Math.max(1, n - 1)));
    if (i < 0 || i >= n) return hideTip();
    tip(`<b>${tr('live.layer')} ${a.layers[i]}</b><br>
      ${tr('live.attn.ent')} ${a.ent[i]?.toFixed(3)}<br>
      ${tr('live.attn.sink')} ${(a.first[i] * 100).toFixed(1)}%<br>
      ${tr('live.attn.near')} ${(a.diag[i] * 100).toFixed(1)}%<br>
      ${tr('live.attn.gate')} ${a.gate[i]?.toFixed(3)}`, e.clientX, e.clientY);
  });
  svg.addEventListener('mouseleave', hideTip);
  svg.addEventListener('click', (e: MouseEvent) => {
    const r = svg.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width * W - x0) / ((chartW - 10) / Math.max(1, n - 1)));
    const L = store.model.langLayers[a.layers[i]];
    if (L) store.select({ type: 'layer', layer: L });
  });
  body.appendChild(el('div', 'small-note', `max-width:${W - 60}px`, tr('live.attn.foot')));
  return card;
}

// ─────────────────────────── attention map viewer ───────────────────────────

function mapCard(live: Live): HTMLElement | null {
  const am = live.attn_maps;
  if (!am?.maps) return null;
  const layers = Object.keys(am.maps).map(Number).sort((x, y) => x - y);
  if (!layers.length) return null;
  const W = 1560, N = am.tokens.length, SIZE = 760;
  const { card, body } = cardShell(W, tr('live.map.title'), tr('live.map.sub', String(N)));
  const scene = el('div', 'no-pan', 'display:flex;gap:26px;align-items:flex-start');
  body.appendChild(scene);

  let selLayer = layers[Math.floor(layers.length / 2)];
  let selView: 'mean' | 'star' = 'mean';

  const ctl = el('div', '', 'display:flex;flex-direction:column;gap:12px;min-width:300px;max-width:340px');
  scene.appendChild(ctl);
  ctl.appendChild(el('div', 'mono', 'font-size:10px;color:var(--faint)', tr('live.map.layer')));
  const layerRow = el('div', '', 'display:flex;flex-wrap:wrap;gap:6px');
  ctl.appendChild(layerRow);
  ctl.appendChild(el('div', 'mono', 'font-size:10px;color:var(--faint)', tr('live.map.view')));
  const viewRow = el('div', '', 'display:flex;gap:6px');
  ctl.appendChild(viewRow);
  const info = el('div', 'note', 'max-width:320px');
  ctl.appendChild(info);
  ctl.appendChild(el('div', 'mono', 'font-size:10px;color:var(--faint)', tr('live.map.prompt')));
  ctl.appendChild(el('div', 'small-note', 'max-width:330px;max-height:170px;overflow:auto;white-space:normal;line-height:1.5', am.prompt));

  const plotWrap = el('div', '', `position:relative;width:${SIZE + 150}px;height:${SIZE + 130}px`);
  scene.appendChild(plotWrap);
  // верхний треугольник не заливаем — просвечивает фон карточки, поэтому
  // причинная пустота остаётся по теме и в светлой, и в тёмной
  const canvas = el('canvas', 'no-pan', `border-radius:6px;position:absolute;left:120px;top:0;
    box-shadow:0 0 0 1px var(--line-strong);cursor:crosshair;background:var(--card)`);
  // буфер держим в физических пикселях: при canvas.width = N браузер растягивал
  // матрицу 89×89 на 760 css-px билинейной интерполяцией и всё расплывалось
  const DPR = Math.min(2, window.devicePixelRatio || 1);
  const PX = Math.round(SIZE * DPR);
  canvas.width = PX; canvas.height = PX;
  canvas.style.width = SIZE + 'px'; canvas.style.height = SIZE + 'px';
  plotWrap.appendChild(canvas);
  const xLabels = el('div', 'mono', `position:absolute;left:120px;top:${SIZE + 8}px;width:${SIZE}px;height:110px;overflow:hidden`);
  const yLabels = el('div', 'mono', `position:absolute;left:0;top:0;width:112px;height:${SIZE}px`);
  plotWrap.appendChild(xLabels); plotWrap.appendChild(yLabels);
  const hoverI = el('div', '', `position:absolute;width:${SIZE}px;pointer-events:none;border-top:1px solid rgba(176,73,42,0.5);display:none`);
  const hoverJ = el('div', '', `position:absolute;height:${SIZE}px;pointer-events:none;border-left:1px solid rgba(176,73,42,0.5);display:none`);
  plotWrap.appendChild(hoverI); plotWrap.appendChild(hoverJ);

  const tokStr = (s: string) => s.replace('Ġ', ' ').replace('Ċ', '¶').replace('▁', ' ');

  function drawLabels() {
    xLabels.innerHTML = ''; yLabels.innerHTML = '';
    const step = Math.max(1, Math.round(N / 24));
    for (let i = 0; i < N; i += step) {
      const frac = i / Math.max(1, N - 1);
      xLabels.appendChild(el('div', '', `position:absolute;left:${(frac * (SIZE - 24)).toFixed(0)}px;font-size:9px;color:var(--ghost);
        transform:rotate(60deg);transform-origin:left top;white-space:nowrap`, tokStr(am.tokens[i] || '')));
      yLabels.appendChild(el('div', '', `position:absolute;right:8px;top:${(frac * (SIZE - 8)).toFixed(0)}px;font-size:9px;color:var(--ghost);white-space:nowrap`, tokStr(am.tokens[i] || '')));
    }
  }

  function render() {
    const map = am.maps[String(selLayer)];
    const mat = map[selView];
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let vmax = 0;
    for (const row of mat) for (const v of row) if (v > vmax) vmax = v;
    ctx.clearRect(0, 0, PX, PX);
    // ячейку кладём прямоугольником с округлением границ: соседние клетки
    // делят пиксель, поэтому нет ни щелей, ни размытия
    const step = PX / N;
    for (let j = 0; j < N; j++) {
      const y0 = Math.round(j * step), y1 = Math.round((j + 1) * step);
      for (let i = 0; i <= j; i++) {
        const x0 = Math.round(i * step), x1 = Math.round((i + 1) * step);
        const [r, g, b] = attnColor((mat[j][i] / (vmax || 1)) * 0.999 + 1e-6);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
    drawLabels();
    info.innerHTML = tr('live.map.info', String(selLayer), String(map.star_head + 1),
      selView === 'star' ? tr('live.map.star') : tr('live.map.mean'));
    layerRow.querySelectorAll('div').forEach(d => {
      const on = Number((d as HTMLElement).dataset.l) === selLayer;
      (d as HTMLElement).style.background = on ? kindOf('attn').solid : 'transparent';
      (d as HTMLElement).style.color = on ? '#fff' : 'inherit';
    });
    viewRow.querySelectorAll('div').forEach(d => {
      (d as HTMLElement).style.background =
        (d as HTMLElement).dataset.v === selView ? 'var(--line)' : 'transparent';
    });
  }

  for (const l of layers) {
    const b = el('div', 'mono no-pan', `font-size:10px;padding:3px 8px;border-radius:9px;cursor:pointer;
      border:1px solid var(--line-strong)`, String(l));
    b.dataset.l = String(l);
    b.onclick = (e: MouseEvent) => { e.stopPropagation(); selLayer = l; render(); };
    layerRow.appendChild(b);
  }
  for (const [v, lbl] of [['mean', 'live.map.mean'], ['star', 'live.map.star']] as const) {
    const b = el('div', 'mono no-pan', `font-size:10px;padding:3px 9px;border-radius:9px;cursor:pointer;
      border:1px solid var(--line-strong)`, tr(lbl));
    b.dataset.v = v;
    b.onclick = (e: MouseEvent) => { e.stopPropagation(); selView = v; render(); };
    viewRow.appendChild(b);
  }

  canvas.addEventListener('mousemove', (e: MouseEvent) => {
    const r = canvas.getBoundingClientRect();
    const i = Math.floor((e.clientX - r.left) / r.width * N);
    const j = Math.floor((e.clientY - r.top) / r.height * N);
    if (i < 0 || j < 0 || i >= N || j >= N) return;
    const v = am.maps[String(selLayer)][selView]?.[j]?.[i];
    hoverI.style.display = 'block'; hoverJ.style.display = 'block';
    hoverI.style.left = '120px'; hoverI.style.width = SIZE + 'px';
    hoverI.style.top = (j / N * SIZE) + 'px';
    hoverJ.style.top = '0px'; hoverJ.style.height = SIZE + 'px';
    hoverJ.style.left = (120 + i / N * SIZE) + 'px';
    if (v != null)
      tip(`<b>${tokStr(am.tokens[i] || '')}</b> ← <b>${tokStr(am.tokens[j] || '')}</b><br>
        ${tr('live.layer')} ${selLayer} · ${tr('live.map.w')} ${(v * 100).toFixed(2)}%`, e.clientX, e.clientY);
  });
  canvas.addEventListener('mouseleave', () => { hideTip(); hoverI.style.display = 'none'; hoverJ.style.display = 'none'; });

  render();
  body.appendChild(el('div', 'small-note', `max-width:${W - 60}px`, tr('live.map.foot')));
  return card;
}

// ─────────────────────────── linear attention ───────────────────────────

function linattnCard(live: Live, store: Store): HTMLElement | null {
  const la = live.linattn;
  if (!la) return null;
  const hasBars = !!la.layers?.length && !!(la.beta?.en?.length);
  if (!hasBars && !la.layers_lambda) return null;
  const W = hasBars ? 1180 : 720, H = 400;
  const { card, body } = cardShell(W, tr('live.la.title'), tr('live.la.sub'), true);
  if (!hasBars) body.appendChild(el('div', 'mono no-pan',
    'align-self:flex-start;font-size:10px;padding:3px 9px;border-radius:10px;border:1px dashed var(--line-strong);color:var(--faint)',
    tr('live.pending')));
  const wrap = el('div', 'no-pan', `position:relative;width:${W}px;height:${H}px;cursor:crosshair`);
  body.appendChild(wrap);
  const svg = svgEl('svg', { width: W, height: H });
  wrap.appendChild(svg);

  const n = la.layers?.length || 1;
  const chartW = 540, chartH = 330, x0 = 40;
  const px = (i: number) => x0 + (i / Math.max(1, n - 1)) * (chartW - 10);
  if (hasBars)
    svg.appendChild(svgEl('rect', { x: x0 - 8, y: 4, width: chartW, height: chartH, fill: 'var(--chart-bg)', stroke: 'var(--line)', rx: 6 }));

  if (hasBars) {
    yLabel(svg, 2, 16, tr('live.la.beta'));
    const beta = la.beta?.en || [];
    const bmax = Math.max(...beta.filter((v): v is number => v != null), 0.01);
    beta.forEach((v, i) => {
      if (v == null) return;
      const h = (v / bmax) * (chartH - 90);
      svg.appendChild(svgEl('rect', {
        x: px(i) - 3.5, y: chartH - 24 - h, width: 7, height: h,
        fill: kindOf('lin').solid, opacity: 0.85, rx: 1.5,
      }));
    });
    yLabel(svg, 2, 30, tr('live.la.hl'));
    if (la.half_life) {
      const hl = la.half_life.map(v => v == null ? null : Math.log10(Math.max(v, 1)));
      const hv = hl.filter((v): v is number => v != null);
      if (hv.length) {
        let lo = Math.min(...hv), hi = Math.max(...hv);
        const p = (hi - lo) * 0.15 || 0.1; lo -= p; hi += p;
        const py = (v: number) => 42 + (1 - (v - lo) / (hi - lo)) * (chartH - 110);
        svg.appendChild(svgEl('path', { d: linePath(hl, px, py), fill: 'none', stroke: 'var(--bad)', 'stroke-width': 1.6 }));
        for (const f of [0, 0.5, 1]) {
          const val = Math.pow(10, hi - f * (hi - lo));
          yLabel(svg, x0 + chartW - 68, py(Math.log10(val)) + 4, val >= 100 ? String(Math.round(val)) : val.toFixed(1));
        }
      }
    }
    (la.layers || []).forEach((l, i) => {
      if (i % 8 === 0) yLabel(svg, px(i) - 6, chartH - 6, String(l));
    });
  }

  // λ: слои × головы, из весов A_log/dt_bias
  const lam = la.layers_lambda;
  if (lam && Object.keys(lam).length) {
    const hx = hasBars ? x0 + chartW + 56 : x0;
    const keys = Object.keys(lam).map(Number).sort((x, y) => x - y);
    const heads = lam[keys[0]]?.length || 48;
    const availW = hasBars ? 330 : W - hx - 60;
    const availH = hasBars ? 300 : H - 90;
    const cellW = Math.min(13, availW / heads), cellH = Math.min(7.2, availH / keys.length);
    yLabel(svg, hx, 14, tr('live.la.lambda'));
    keys.forEach((k, ki) => {
      (lam[k] || []).forEach((v, h) => {
        svg.appendChild(svgEl('rect', {
          x: hx + h * (cellW + 0.5), y: 22 + ki * (cellH + 0.5), width: cellW, height: cellH,
          fill: heatColor(Math.min(1, Math.max(0, (v - 0.98) / 0.0199))), rx: 0.5,
        }));
      });
    });
    // шкала λ
    const sy = 30 + keys.length * (cellH + 0.5);
    for (let s = 0; s < 60; s++)
      svg.appendChild(svgEl('rect', {
        x: hx + s, y: sy, width: 1.5, height: 8,
        fill: heatColor(s / 59),
      }));
    yLabel(svg, hx + 64, sy + 8, 'λ 0.98 → 1.0');
    yLabel(svg, hx, sy + 24, tr('live.la.axes', String(heads - 1), String(keys[keys.length - 1])));

    svg.addEventListener('mousemove', (e: MouseEvent) => {
      const r = svg.getBoundingClientRect();
      const wx = (e.clientX - r.left) / r.width * W, wy = (e.clientY - r.top) / r.height * H;
      if (wx < hx || wy < 22) return;
      const h = Math.floor((wx - hx) / (cellW + 0.5));
      const ki = Math.floor((wy - 22) / (cellH + 0.5));
      if (h < 0 || h >= heads || ki < 0 || ki >= keys.length) return;
      const layer = keys[ki];
      const v = lam[layer]?.[h];
      if (v == null) return;
      tip(`<b>${tr('live.layer')} ${layer} · ${tr('live.head')} ${h}</b><br>λ ${v.toFixed(4)}<br>` +
        (v > 0.999 ? tr('live.la.long') : tr('live.la.short')), e.clientX, e.clientY);
    });
    svg.addEventListener('mouseleave', hideTip);
    svg.addEventListener('click', (e: MouseEvent) => {
      const r = svg.getBoundingClientRect();
      const wx = (e.clientX - r.left) / r.width * W, wy = (e.clientY - r.top) / r.height * H;
      const ki = Math.floor((wy - 22) / (cellH + 0.5));
      if (wx < hx || ki < 0 || ki >= keys.length) return;
      const L = store.model.langLayers[keys[ki]];
      if (L) store.select({ type: 'layer', layer: L });
    });
  }

  if (hasBars) {
    svg.addEventListener('mousemove', (e: MouseEvent) => {
      const r = svg.getBoundingClientRect();
      const wx = (e.clientX - r.left) / r.width * W;
      if (wx < x0 || wx > x0 + chartW) return;
      const i = Math.round((wx - x0) / ((chartW - 10) / Math.max(1, n - 1)));
      if (i < 0 || i >= n || la.layers?.[i] == null) return;
      tip(`<b>${tr('live.layer')} ${la.layers[i]}</b><br>
        β ${la.beta?.en?.[i]?.toFixed(3) ?? '—'}<br>
        ${tr('live.la.hl')} ${la.half_life?.[i] ?? '—'}`, e.clientX, e.clientY);
    });
    svg.addEventListener('mouseleave', hideTip);
    svg.addEventListener('click', (e: MouseEvent) => {
      const r = svg.getBoundingClientRect();
      const i = Math.round(((e.clientX - r.left) / r.width * W - x0) / ((chartW - 10) / Math.max(1, n - 1)));
      if (la.layers?.[i] == null) return;
      const L = store.model.langLayers[la.layers[i]];
      if (L) store.select({ type: 'layer', layer: L });
    });
  }
  body.appendChild(el('div', 'small-note', `max-width:${W - 60}px`, tr('live.la.foot')));
  return card;
}

// ─────────────────────────── neurons ───────────────────────────

const NEU_Q = ['0', '1', '5', '10', '25', '50', '75', '90', '95', '99', '99.9', '100'];

function neuronsCard(live: Live, store: Store): HTMLElement | null {
  const nb = live.neurons;
  if (!nb?.heat?.length) return null;
  const W = 1130, H = 360;
  const { card, body } = cardShell(W, tr('live.neu.title'), tr('live.neu.sub'), true);
  const wrap = el('div', 'no-pan', `position:relative;width:${W}px;height:${H}px;cursor:crosshair`);
  body.appendChild(wrap);
  const svg = svgEl('svg', { width: W, height: H });
  wrap.appendChild(svg);

  const n = nb.heat.length;
  const chartW = 880, x0 = 54, stripY = 26, cellW = (chartW - 60) / n;
  const rows = nb.heat[0]?.length || NEU_Q.length, cellH = 13;
  yLabel(svg, x0, 12, tr('live.neu.heat'));
  nb.heat.forEach((row, li) => {
    row.forEach((v, qi) => {
      const t = Math.min(1, Math.max(0, (Math.log10(Math.max(v, 1e-6)) + 5) / 4));
      svg.appendChild(svgEl('rect', {
        x: x0 + 46 + li * cellW, y: stripY + qi * (cellH + 1),
        width: Math.max(2, cellW - 1), height: cellH,
        fill: ramp(1 - t), rx: 1,
      }));
    });
  });
  NEU_Q.forEach((q, qi) => {
    if (qi % 2 === 0)
      yLabel(svg, x0, stripY + qi * (cellH + 1) + 10, 'p' + q);
  });
  for (let li = 0; li < n; li += 8)
    yLabel(svg, x0 + 46 + li * cellW - 4, stripY + rows * (cellH + 1) + 12, String(li));

  // шкала частоты стрельбы (лог): 1e-5 → 1e-1
  const scx = x0 + 46 + n * cellW + 14;
  for (let s = 0; s < 48; s++) {
    const t = s / 47;
    svg.appendChild(svgEl('rect', {
      x: scx + s * 1.6, y: stripY, width: 1.6, height: rows * (cellH + 1) - 1,
      fill: ramp(1 - t),
    }));
  }
  yLabel(svg, scx, stripY + rows * (cellH + 1) + 10, '1e-5 … 1e-1');

  svg.addEventListener('mousemove', (e: MouseEvent) => {
    const r = svg.getBoundingClientRect();
    const wx = (e.clientX - r.left) / r.width * W, wy = (e.clientY - r.top) / r.height * H;
    if (wx < x0 + 46 || wy < stripY) return hideTip();
    const li = Math.floor((wx - x0 - 46) / cellW);
    const qi = Math.floor((wy - stripY) / (cellH + 1));
    if (li < 0 || li >= n || qi < 0 || qi >= rows) return hideTip();
    const v = nb.heat[li]?.[qi];
    if (v == null) return;
    tip(`<b>${tr('live.layer')} ${li}</b> · p${NEU_Q[qi]}<br>${tr('live.neu.fire')} ${(v * 100).toFixed(3)}%`, e.clientX, e.clientY);
  });
  svg.addEventListener('mouseleave', hideTip);
  svg.addEventListener('click', (e: MouseEvent) => {
    const r = svg.getBoundingClientRect();
    const wx = (e.clientX - r.left) / r.width * W;
    const li = Math.floor((wx - x0 - 46) / cellW);
    if (li < 0 || li >= n) return;
    const L = store.model.langLayers[li];
    if (L) store.select({ type: 'layer', layer: L });
  });

  // кривые: мёртвые / концентрация / специализация
  const cy = stripY + rows * (cellH + 1) + 32;
  const curves: [string, (number | null)[], string][] = [
    [tr('live.neu.dead'), nb.dead_frac, 'var(--bad)'],
    [tr('live.neu.conc'), nb.conc, kindOf('mlp').solid],
    [tr('live.neu.spec'), nb.spec_frac, kindOf('lin').solid],
  ];
  curves.forEach(([lbl, vals, col]) => {
    const vv = vals.filter((v): v is number => v != null);
    if (!vv.length) return;
    let lo = Math.min(...vv), hi = Math.max(...vv);
    if (hi - lo < 1e-6) hi = lo + 1e-6;
    const py = (v: number) => cy + (1 - (v - lo) / (hi - lo)) * 70;
    svg.appendChild(svgEl('path', { d: linePath(vals, i => x0 + 46 + i * cellW, py), fill: 'none', stroke: col, 'stroke-width': 1.5 }));
    yLabel(svg, x0 + 46 + n * cellW + 6, py(vv[vv.length - 1]) + 4, lbl);
    (svg.lastChild as SVGElement).setAttribute('fill', col);
  });

  if (nb.examples?.length) {
    const ex = el('div', 'no-pan', 'display:flex;flex-direction:column;gap:4px');
    ex.appendChild(el('div', 'mono', 'font-size:10px;color:var(--faint)', tr('live.neu.ex')));
    for (const e of nb.examples.slice(0, 6)) {
      const total = e.en + e.code + 1e-9;
      const row = el('div', 'mono no-pan', 'font-size:10px;display:flex;gap:10px;align-items:center;cursor:pointer;border-radius:5px;padding:1px 4px');
      row.addEventListener('mouseenter', () => { row.style.background = 'var(--line)'; });
      row.addEventListener('mouseleave', () => { row.style.background = ''; });
      row.addEventListener('click', () => {
        const L = store.model.langLayers[e.layer];
        if (L) store.select({ type: 'layer', layer: L });
      });
      row.innerHTML = `
        <span style="width:130px;color:var(--faint)">${tr('live.layer')} ${e.layer} · №${e.neuron}</span>
        <span style="display:inline-flex;width:150px;height:8px;border-radius:4px;overflow:hidden">
          <span style="width:${(e.en / total * 100).toFixed(0)}%;background:${DOM_COLOR.en}"></span>
          <span style="width:${(e.code / total * 100).toFixed(0)}%;background:${DOM_COLOR.code}"></span>
        </span>
        <span>en ${(e.en * 100).toFixed(2)}% · code ${(e.code * 100).toFixed(2)}%</span>`;
      ex.appendChild(row);
    }
    body.appendChild(ex);
  }
  body.appendChild(el('div', 'small-note', `max-width:${W - 60}px`, tr('live.neu.foot')));
  return card;
}

export interface LiveSection { root: HTMLElement; rect: { x: number; y: number; w: number; h: number } }

// ─────────────────────────── что уже снято ───────────────────────────

function statusCard(live: Live): HTMLElement {
  const { card, body } = cardShell(560, tr('live.status.title'), tr('live.status.sub'));
  const rows: [string, 'ok' | 'part' | 'wait', string][] = [
    [tr('live.flow.title'), live.flow?.h_rms?.all?.length ? 'ok' : live.flow?.io_cos?.length ? 'part' : 'wait',
      live.flow?.h_rms?.all?.length ? tr('live.status.src.pass') : tr('live.status.src.carve')],
    [tr('live.actq.title'), live.flow?.actq ? 'ok' : 'wait', tr('live.status.src.pass')],
    [tr('live.attn.title'), live.attn?.layers?.length ? 'ok' : 'wait', tr('live.status.src.pass')],
    [tr('live.map.title'), live.attn_maps?.maps ? 'ok' : 'wait', tr('live.status.src.pass')],
    [tr('live.la.title'), live.linattn?.beta?.en?.length ? 'ok' : live.linattn?.layers_lambda ? 'part' : 'wait',
      live.linattn?.beta?.en?.length ? tr('live.status.src.pass') : tr('live.status.src.weights')],
    [tr('live.neu.title'), live.neurons?.heat?.length ? 'ok' : 'wait', tr('live.status.src.carve')],
    [tr('live.frag.title'), live.fragility?.kl?.length ? 'ok' : 'wait', tr('live.status.src.pass')],
    [tr('live.vis.title'), live.vision?.img_share ? 'ok' : 'wait', tr('live.status.src.pass')],
  ];
  for (const [name, st, src] of rows) {
    const color = st === 'ok' ? 'var(--good)' : st === 'part' ? 'var(--warn)' : 'var(--ghost)';
    const label = st === 'ok' ? src : st === 'part' ? `${tr('live.status.part')} · ${src}` : tr('live.status.wait');
    body.appendChild(el('div', '', 'display:flex;align-items:baseline;gap:10px;border-top:1px solid var(--line);padding:6px 0', `
      <span style="width:9px;height:9px;border-radius:5px;background:${color};flex-shrink:0;transform:translateY(1px);${st === 'wait' ? 'opacity:0.45' : ''}"></span>
      <span style="font-size:14px;flex:1">${name}</span>
      <span class="mono" style="font-size:10px;color:${st === 'wait' ? 'var(--ghost)' : color}">${label}</span>`));
  }
  const when = (live.meta as any)?.when;
  if (when) body.appendChild(el('div', 'small-note', '', `${tr('live.status.when')} ${when}`));
  return card;
}

// ─────────────────────────── fragility ───────────────────────────

function fragCard(live: Live, store: Store): HTMLElement | null {
  const fr = live.fragility;
  if (!fr?.kl?.length) return null;
  const W = 1180, H = 400;
  const { card, body } = cardShell(W, tr('live.frag.title'), tr('live.frag.sub'), true);
  const wrap = el('div', 'no-pan', `position:relative;width:${W}px;height:${H}px`);
  body.appendChild(wrap);
  const svg = svgEl('svg', { width: W, height: H });
  wrap.appendChild(svg);

  const n = fr.kl.length;
  const chartW = 520, chartH = 330, x0 = 40;
  const px = (i: number) => x0 + (i / Math.max(1, n - 1)) * (chartW - 12);
  const kmax = Math.max(...fr.kl) * 1.1 || 1;
  svg.appendChild(svgEl('rect', { x: x0 - 8, y: 4, width: chartW, height: chartH, fill: 'var(--chart-bg)', stroke: 'var(--line)', rx: 6 }));
  yLabel(svg, 2, 16, tr('live.frag.kl'));
  fr.kl.forEach((v, i) => {
    const h = (v / kmax) * (chartH - 40);
    svg.appendChild(svgEl('rect', {
      x: px(i) - 3.4, y: chartH - 24 - h, width: 6.8, height: h,
      fill: ramp(1 - Math.min(1, v / kmax)), rx: 1.5,
    }));
  });
  for (let i = 0; i < n; i += 8)
    yLabel(svg, px(i) - 6, chartH - 6, String(i));

  // мост: статическая квантуемость против живого ущерба
  const bx0 = x0 + chartW + 70, bw = 480;
  yLabel(svg, bx0, 16, tr('live.frag.bridge'));
  const pts: { x: number; y: number; i: number }[] = [];
  store.model.langLayers.forEach((L, i) => {
    const sq = L.tensors.map(x => x.sqnr_int4_g128).filter((v): v is number => v != null);
    if (!sq.length || fr.kl[i] == null) return;
    pts.push({ x: sq.reduce((p, c) => p + c, 0) / sq.length, y: fr.kl[i], i });
  });
  if (pts.length) {
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xlo = Math.min(...xs), xhi = Math.max(...xs), yhi = Math.max(...ys) * 1.1;
    const sx = (v: number) => bx0 + 30 + ((v - xlo) / (xhi - xlo || 1)) * (bw - 50);
    const sy = (v: number) => 44 + (1 - v / yhi) * (chartH - 100);
    svg.appendChild(svgEl('rect', { x: bx0 + 24, y: 32, width: bw - 34, height: chartH - 84, fill: 'var(--chart-bg)', stroke: 'var(--line)', rx: 6 }));
    for (const p of pts) {
      const L = store.model.langLayers[p.i];
      svg.appendChild(svgEl('circle', {
        cx: sx(p.x), cy: sy(p.y), r: L.kind === 'full' ? 4 : 2.6,
        fill: kindOf(L.kind === 'full' ? 'attn' : 'lin').solid, opacity: 0.9,
      }));
    }
    yLabel(svg, bx0 + 30, chartH - 36, tr('live.frag.x') + ' →');
    yLabel(svg, bx0, 40, '↑ ' + tr('live.frag.y'));
  }

  svg.addEventListener('mousemove', (e: MouseEvent) => {
    const r = svg.getBoundingClientRect();
    const wx = (e.clientX - r.left) / r.width * W;
    if (wx < x0 || wx > x0 + chartW) return hideTip();
    const i = Math.round((wx - x0) / ((chartW - 12) / Math.max(1, n - 1)));
    if (i < 0 || i >= n) return;
    tip(`<b>${tr('live.layer')} ${i}</b><br>KL ${fr.kl[i]?.toFixed(4)}<br>cos ${fr.logit_cos[i]?.toFixed(4)}`, e.clientX, e.clientY);
  });
  svg.addEventListener('mouseleave', hideTip);
  svg.addEventListener('click', (e: MouseEvent) => {
    const r = svg.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width * W - x0) / ((chartW - 12) / Math.max(1, n - 1)));
    const L = store.model.langLayers[i];
    if (L) store.select({ type: 'layer', layer: L });
  });
  body.appendChild(el('div', 'small-note', `max-width:${W - 60}px`, tr('live.frag.foot')));
  return card;
}

// ─────────────────────────── vision showcase ───────────────────────────

function visionCard(live: Live, store: Store): HTMLElement | null {
  const v = live.vision;
  if (!v?.img_share || !Object.keys(v.img_share).length) return null;
  const W = 640, H = 320;
  const { card, body } = cardShell(W, tr('live.vis.title'), tr('live.vis.sub'));
  const row = el('div', '', 'display:flex;gap:20px;align-items:flex-start');
  body.appendChild(row);
  const img = el('img', 'no-pan', 'width:250px;border-radius:6px;box-shadow:0 0 0 1px var(--line-strong)');
  img.src = `models/${store.model.slug}/atlas_shot.png`;
  img.alt = 'the screenshot the model is looking at';
  row.appendChild(img);
  const wrap = el('div', 'no-pan', `position:relative;width:330px;height:${H - 80}px`);
  row.appendChild(wrap);
  const svg = svgEl('svg', { width: 330, height: H - 80 });
  wrap.appendChild(svg);
  const entries = Object.entries(v.img_share).sort((a, b) => Number(a[0]) - Number(b[0]));
  const mx = Math.max(...entries.map(e => e[1])) || 1;
  entries.forEach(([l, val], i) => {
    const y = 8 + i * ((H - 100) / entries.length);
    const w = (val / mx) * 190;
    yLabel(svg, 0, y + 10, l);
    svg.appendChild(svgEl('rect', { x: 34, y, width: Math.max(2, w), height: 11, fill: kindOf('vision').solid, rx: 2 }));
    yLabel(svg, 38 + Math.max(2, w), y + 10, (val * 100).toFixed(1) + '%');
  });
  body.appendChild(el('div', 'small-note', `max-width:${W - 60}px`, tr('live.vis.foot')));
  return card;
}

// ─────────────────────────── сборка региона ───────────────────────────

export function buildLive(store: Store, live: Live, X: number, Y: number): LiveSection {
  const W = 6420;
  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `
    <div class="section-tag mono">${tr('sec.live.tag')}</div>
    <div class="section-title">${tr('sec.live.title')}</div>
    <div class="section-sub">${tr('sec.live.sub')}</div>`));

  const flow = el('div', '', 'display:flex;flex-wrap:wrap;gap:26px;align-items:flex-start');
  root.appendChild(flow);

  for (const c of [
    statusCard(live),
    flowCard(live, store), mapCard(live), attnCard(live, store), linattnCard(live, store),
    actqCard(live), fragCard(live, store), neuronsCard(live, store), visionCard(live, store),
  ].filter((c): c is HTMLElement => c != null))
    flow.appendChild(c);

  // высота до монтирования неизвестна: вызывающий меряет root.offsetHeight и обновляет rect.h
  return { root, rect: { x: X, y: Y, w: W, h: 900 } };
}
