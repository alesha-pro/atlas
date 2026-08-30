// Глубина: как текущая метрика меняется от первого слоя к последнему.
import { fmtN, type Layer } from '../data';
import { colorFor, ink, kindOf, ramp, tOf } from '../color';
import { t as tr } from '../i18n';
import { el, svgEl } from '../world';
import { tip, hideTip, type Store } from '../store';

const W = 1000, CH_W = 900, CH_H = 420;

export function buildDepth(store: Store, X: number, Y: number): {
  root: HTMLElement; rect: { x: number; y: number; w: number; h: number };
} {
  const m = store.model;
  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `
    <div class="section-tag mono">${tr('sec.depth.tag')}</div>
    <div class="section-title">${tr('sec.depth.title')}</div>
    <div class="section-sub">${tr('sec.depth.sub')}</div>`));

  const card = el('div', 'card', `position:relative;width:${W}px;padding:22px 26px;display:flex;flex-direction:column;gap:18px`);
  root.appendChild(card);

  const chartWrap = el('div', 'no-pan', `position:relative;width:${CH_W}px;height:${CH_H + 50}px`);
  card.appendChild(chartWrap);

  const qWrap = el('div', '', 'display:flex;gap:12px');
  card.appendChild(qWrap);
  const foot = el('div', 'small-note', 'max-width:900px');
  card.appendChild(foot);

  function layerAvg(L: Layer): number | null {
    const d = store.md;
    const vs = L.tensors.map(t => d.get(t)).filter((v): v is number => v != null);
    if (!vs.length) return null;
    return vs.reduce((a, b) => a + b, 0) / vs.length;
  }

  function draw() {
    const d = store.md;
    chartWrap.innerHTML = '';
    const svg = svgEl('svg', { width: CH_W, height: CH_H + 50 });
    chartWrap.appendChild(svg as any);

    const pts = m.langLayers.map(L => ({ L, v: layerAvg(L) })).filter(p => p.v != null) as { L: Layer; v: number }[];
    if (!pts.length) return;
    const vs = pts.map(p => d.transform(p.v));
    let lo = Math.min(...vs), hi = Math.max(...vs);
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = (hi - lo) * 0.12;
    lo -= pad; hi += pad;

    const px = (i: number) => 46 + (i / (pts.length - 1)) * (CH_W - 66);
    const py = (v: number) => 10 + (1 - (d.transform(v) - lo) / (hi - lo)) * (CH_H - 20);

    // фон
    svg.appendChild(svgEl('rect', { x: 36, y: 4, width: CH_W - 46, height: CH_H, fill: 'var(--chart-bg)', stroke: 'var(--line)', rx: 6 }));

    // линия
    let path = '';
    pts.forEach((p, i) => { path += (i ? ' L ' : 'M ') + px(i).toFixed(1) + ' ' + py(p.v).toFixed(1); });
    svg.appendChild(svgEl('path', { d: path, fill: 'none', stroke: 'var(--line-strong)', 'stroke-width': 1.6 }));

    // точки
    pts.forEach((p, i) => {
      const c = svgEl('circle', {
        cx: px(i).toFixed(1), cy: py(p.v).toFixed(1),
        r: p.L.kind === 'full' ? 6 : 4.2,
        fill: colorFor(p.v, d),
        stroke: p.L.kind === 'full' ? kindOf('attn').fg : 'rgba(60,50,35,0.3)',
        'stroke-width': p.L.kind === 'full' ? 1.6 : 0.8,
        style: 'cursor:pointer',
      });
      c.addEventListener('click', () => store.select({ type: 'layer', layer: p.L }));
      c.addEventListener('mouseenter', e => tip(
        `<span class="mono">${tr('depth.layer', p.L.label)}</span> · ${p.L.kind === 'full' ? tr('kind.full') : tr('kind.linear')}<br>
         ${tr('depth.avg')} ${d.label}: <span class="mono">${d.fmt(p.v)}${d.unit ? ' ' + d.unit : ''}</span>`,
        (e as MouseEvent).clientX, (e as MouseEvent).clientY));
      c.addEventListener('mouseleave', hideTip);
      svg.appendChild(c);
      if (p.L.idx % 8 === 0 || p.L.idx === pts.length - 1) {
        const t = svgEl('text', { x: px(i), y: CH_H + 26, 'text-anchor': 'middle', fill: 'var(--ghost)', 'font-size': 10, 'font-family': "'IBM Plex Mono',monospace" });
        t.textContent = p.L.label;
        svg.appendChild(t);
      }
    });

    // y-подписи
    for (const f of [0, 0.5, 1]) {
      const tv = lo + f * (hi - lo);
      const raw = d.log ? Math.pow(10, tv) : tv;
      const t = svgEl('text', { x: 30, y: 12 + (1 - f) * (CH_H - 20), 'text-anchor': 'end', fill: 'var(--ghost)', 'font-size': 10, 'font-family': "'IBM Plex Mono',monospace" });
      t.textContent = d.fmt(raw);
      svg.appendChild(t);
    }

    // четверти
    qWrap.innerHTML = '';
    const L = m.langLayers.length;
    [0, 1, 2, 3].forEach(qi => {
      const ls = m.langLayers.filter(l => Math.floor(l.idx * 4 / L) === qi);
      const vals = ls.map(layerAvg).filter((v): v is number => v != null);
      const avg = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
      const box = el('div', '', `flex:1;border:1px solid var(--line);background:var(--card-solid);border-radius:6px;padding:12px 14px;display:flex;flex-direction:column;gap:6px`);
      box.innerHTML = `
        <div class="mono" style="font-size:10.5px;color:var(--ghost)">${tr('layers.range', ls[0].label, ls[ls.length - 1].label)}</div>
        <div class="mono" style="font-size:21px;color:${ink(avg, d)}">${d.fmt(avg)}<span style="font-size:11px;color:var(--ghost)"> ${d.unit}</span></div>
        <div style="height:6px;border-radius:3px;background:${colorFor(avg, d)}"></div>`;
      qWrap.appendChild(box);
    });

    foot.textContent = store.metric === 'int4' ? tr('depth.foot.int4') : tr('depth.foot.other');
  }

  store.addEventListener('metric', draw);
  draw();

  return { root, rect: { x: X, y: Y, w: W, h: 760 } };
}
