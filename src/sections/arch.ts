// Архитектурная карта: группы-узлы со связями + мини-рельса слоёв.
import { fmtN, type GroupInfo, type Model } from '../data';
import { t } from '../i18n';
import { colorForTensor, ink, kindOf } from '../color';
import { el, svgEl } from '../world';
import { tip, hideTip, type Store } from '../store';

interface NodeGeom { x: number; y: number; w: number; h: number; }


export function buildArch(store: Store, X: number, Y: number): {
  root: HTMLElement; rect: { x: number; y: number; w: number; h: number };
} {
  const m = store.model;
  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:2360px;height:1620px`);
  root.appendChild(sectionHead(t('sec.arch.tag'), t('sec.arch.title'), t('sec.arch.sub')));

  const area = el('div', '', 'position:relative;flex:1');
  root.appendChild(area);

  const G: Record<string, NodeGeom> = {
    vision: { x: 850, y: 0, w: 400, h: 0 },
    embed: { x: 0, y: 690, w: 320, h: 0 },
    rail: { x: 400, y: 260, w: 380, h: 0 },
    attn: { x: 860, y: 300, w: 390, h: 0 },
    linattn: { x: 860, y: 760, w: 390, h: 0 },
    mlp: { x: 1380, y: 500, w: 390, h: 0 },
    norm: { x: 1380, y: 1080, w: 340, h: 0 },
    head: { x: 1920, y: 560, w: 340, h: 0 },
    mtp: { x: 1920, y: 1020, w: 340, h: 0 },
  };

  // ── связи ──
  const svg = svgEl('svg', { width: 2360, height: 1520, style: 'position:absolute;left:0;top:0;overflow:visible;pointer-events:none' });
  area.appendChild(svg as any);

  // ── узлы-группы ──
  const groupByKey = new Map(m.groups.map(g => [g.key, g]));
  for (const key of ['embed', 'vision', 'attn', 'linattn', 'mlp', 'norm', 'head', 'mtp']) {
    const g = groupByKey.get(key);
    if (!g) continue;
    area.appendChild(groupNode(store, g, G[key]));
  }

  // ── рельса слоёв ──
  area.appendChild(railNode(store, G.rail));

  // measure heights after mount
  requestAnimationFrame(() => {
    for (const [key, geom] of Object.entries(G)) {
      const n = area.querySelector<HTMLElement>(`[data-node="${key}"]`);
      if (n) geom.h = n.offsetHeight;
    }
    drawEdges(svg, G);
  });

  return { root, rect: { x: X, y: Y, w: 2360, h: 1620 } };
}

function sectionHead(tag: string, title: string, sub: string): HTMLElement {
  return el('div', 'section-head', '', `
    <div class="section-tag mono">${tag}</div>
    <div class="section-title">${title}</div>
    <div class="section-sub">${sub}</div>`);
}

function groupNode(store: Store, g: GroupInfo, geom: NodeGeom): HTMLElement {
  const m = store.model;
  const k = kindOf(g.kind);
  const node = el('div', 'fade-in', `position:absolute;left:${geom.x}px;top:${geom.y}px;width:${geom.w}px;
    background:rgba(255,253,248,0.94);border:1px solid ${k.bd};border-radius:18px;padding:20px 22px;
    box-shadow:0 20px 44px -32px rgba(60,50,35,0.55);cursor:pointer`);
  node.dataset.node = g.key;

  const d2 = g.tensors.filter(t => t.is2d);
  const int4s = d2.map(t => t.sqnr_int4_g128!);
  const avg = int4s.length ? int4s.reduce((a, b) => a + b, 0) / int4s.length : null;
  const worst = d2.length ? d2.reduce((a, b) => a.sqnr_int4_g128! < b.sqnr_int4_g128! ? a : b) : null;
  const md = m.metrics.int4;

  node.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px">
      <div class="tag-pill mono" style="background:${k.bg};color:${k.fg}">${g.plain}</div>
      <div class="mono" style="font-size:12px;color:var(--ghost)">${t('node.tensors', g.tensors.length)}</div>
    </div>
    <div class="mono" style="font-size:34px;font-weight:500;color:${k.fg};letter-spacing:-0.02em;margin-bottom:8px">${fmtN(g.params)}</div>
    <div style="font-size:24px;line-height:1.15;margin-bottom:9px">${g.label}</div>
    <div style="font-size:15px;line-height:1.5;color:var(--muted);text-wrap:pretty">${t('node.text.' + g.key)}</div>
    <div style="margin-top:13px;display:flex;flex-direction:column;gap:7px">
      <div style="display:flex;align-items:center;gap:8px">
        <div class="bar-track" style="flex:1"><div class="bar-fill" style="width:${g.share.toFixed(1)}%;background:${k.solid}"></div></div>
        <div class="mono" style="font-size:12.5px;color:var(--faint);white-space:nowrap">${g.share.toFixed(1)}% ${t('node.share')}</div>
      </div>
      ${avg != null ? `
      <div style="display:flex;align-items:center;gap:8px;justify-content:space-between">
        <div class="mono" style="font-size:12.5px;color:var(--faint)">${t('node.avg')} <span style="color:${ink(avg, md)}">${avg.toFixed(2)} ${t('unit.db')}</span></div>
        <div class="mono" style="font-size:12px;color:var(--faint)">${t('node.worst')} <span style="color:${ink(worst!.sqnr_int4_g128!, md)}">${worst!.sqnr_int4_g128!.toFixed(1)}</span></div>
      </div>` : `
      <div class="mono" style="font-size:12.5px;color:var(--ghost)">${t('node.1d')}</div>`}
    </div>`;

  node.addEventListener('click', () => store.select({ type: 'group', group: g }));
  const outline = () => {
    const on = store.sel.type === 'group' && store.sel.group.key === g.key;
    node.style.borderColor = on ? 'var(--accent)' : k.bd;
    node.style.boxShadow = on
      ? '0 0 0 3px rgba(255,138,61,0.22), 0 26px 50px -30px rgba(60,50,35,0.6)'
      : '0 20px 44px -32px rgba(60,50,35,0.55)';
  };
  store.addEventListener('sel', outline);
  return node;
}

function railNode(store: Store, geom: NodeGeom): HTMLElement {
  const m = store.model;
  const node = el('div', '', `position:absolute;left:${geom.x}px;top:${geom.y}px;width:${geom.w}px;
    background:rgba(255,253,248,0.94);border:1px solid var(--line-strong);border-radius:18px;padding:18px 20px;
    box-shadow:0 20px 44px -32px rgba(60,50,35,0.55)`);
  node.dataset.node = 'rail';

  node.innerHTML = `
    <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:4px">
      <div class="eyebrow mono">${t('rail.eyebrow')}</div>
      <div class="mono" style="font-size:12.5px;color:var(--ghost)">${fmtN(m.langLayers.reduce((a, l) => a + l.params, 0))}</div>
    </div>
    <div style="font-size:26px;line-height:1.1;margin-bottom:10px">${t('rail.title', m.langLayers.length)}</div>
    <div class="rail-rows" style="display:flex;flex-direction:column;gap:1px"></div>
    <div style="display:flex;align-items:center;gap:14px;margin-top:10px" class="mono">
      <div style="display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--faint)">
        <div style="width:6px;height:11px;border-radius:2px;background:${kindOf('attn').solid}"></div>${t('kind.full')}</div>
      <div style="display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--faint)">
        <div style="width:6px;height:11px;border-radius:2px;background:${kindOf('lin').solid}"></div>${t('kind.linear')}</div>
    </div>`;

  const rows = node.querySelector('.rail-rows')!;
  const cellRefs: { div: HTMLElement; t: any }[] = [];
  for (const L of m.langLayers) {
    const row = el('div', '', 'display:flex;align-items:center;gap:7px;padding:0 3px;border-radius:4px;cursor:pointer;height:9px');
    const num = el('div', 'mono', `width:20px;font-size:8.5px;color:var(--ghost);text-align:right;line-height:1`);
    num.textContent = L.idx % 4 === 0 ? L.label : '';
    row.appendChild(num);
    row.appendChild(el('div', '', `width:5px;height:7px;border-radius:1px;background:${L.kind === 'full' ? kindOf('attn').solid : kindOf('lin').solid}`));
    const strip = el('div', '', 'flex:1;display:flex;gap:1px');
    for (const t of L.tensors) {
      const c = el('div', '', `flex:${Math.max(0.05, t.numel / 2e7).toFixed(3)};height:7px;border-radius:1px;background:${colorForTensor(t, store.md)}`);
      strip.appendChild(c);
      cellRefs.push({ div: c, t });
    }
    row.appendChild(strip);
    row.addEventListener('click', () => store.select({ type: 'layer', layer: L }));
    row.addEventListener('mouseenter', (e) => tip(
      `<span class="mono">${t('depth.layer', L.label)}</span> · ${L.kind === 'full' ? t('kind.full') : t('kind.linear')} · ${fmtN(L.params)}`,
      (e as MouseEvent).clientX, (e as MouseEvent).clientY));
    row.addEventListener('mouseleave', hideTip);
    rows.appendChild(row);
  }
  store.addEventListener('metric', () => {
    for (const { div, t } of cellRefs) div.style.background = colorForTensor(t, store.md);
  });
  return node;
}

function drawEdges(svg: SVGElement, G: Record<string, NodeGeom>) {
  svg.innerHTML = '';
  const E: [string, string, number][] = [
    ['embed', 'rail', 1], ['vision', 'rail', 0], ['rail', 'attn', 1], ['rail', 'linattn', 1],
    ['attn', 'mlp', 1], ['linattn', 'mlp', 1], ['linattn', 'norm', 0], ['mlp', 'head', 1], ['head', 'mtp', 0],
  ];
  for (const [a, b, spine] of E) {
    const A = G[a], B = G[b];
    if (!A || !B || !A.h || !B.h) continue;
    const acx = A.x + A.w / 2, acy = A.y + A.h / 2;
    const bcx = B.x + B.w / 2, bcy = B.y + B.h / 2;
    let d: string;
    if (Math.abs(bcx - acx) >= Math.abs(bcy - acy)) {
      const ltr = bcx >= acx;
      const x1 = ltr ? A.x + A.w : A.x, y1 = acy;
      const x2 = ltr ? B.x : B.x + B.w, y2 = bcy;
      const dx = Math.max(70, Math.abs(x2 - x1) * 0.48) * (ltr ? 1 : -1);
      d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
    } else {
      const ttb = bcy >= acy;
      const x1 = acx, y1 = ttb ? A.y + A.h : A.y;
      const x2 = bcx, y2 = ttb ? B.y : B.y + B.h;
      const dy = Math.max(60, Math.abs(y2 - y1) * 0.48) * (ttb ? 1 : -1);
      d = `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`;
    }
    const p = svgEl('path', {
      d,
      fill: 'none',
      stroke: spine ? 'rgba(150,118,76,0.75)' : 'rgba(150,118,76,0.42)',
      'stroke-width': spine ? 3 : 1.8,
      'stroke-linecap': 'round',
    });
    if (spine) {
      p.setAttribute('stroke-dasharray', '10 9');
      (p as any).style.animation = 'flow 4.5s linear infinite';
    }
    svg.appendChild(p);
  }
}
