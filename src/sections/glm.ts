// GLM-5.3-Flash NVFP4: регион измеренной модели.
// Данные: public/models/glm-5.3-flash-nvfp4/insights.json (scripts/build_glm_atlas.py).
// Каждая карточка отвечает на один вопрос и держит свои числа при себе.
import type { Store } from '../store';
import { hideTip, tip } from '../store';
import { lang } from '../i18n';
import { kindOf, ramp } from '../color';
import { el, svgEl } from '../world';

type Rect = { x: number; y: number; w: number; h: number };
export interface GlmSection { root: HTMLElement; rect: Rect }

// ───────────────────────── утилиты ─────────────────────────

const l = (en: string, ru: string) => lang === 'ru' ? ru : en;
const loc = () => lang === 'ru' ? 'ru-RU' : 'en-US';
const fmt = (v: number, d = 1) => Number(v).toLocaleString(loc(), { maximumFractionDigits: d });
const fix = (v: number, d = 2) => Number(v).toFixed(d);
const pct = (v: number, d = 1) => `${fmt(v * 100, d)}%`;
const big = (v: number) => v >= 1e6 ? `${fmt(v / 1e6, 2)}M` : v >= 1e3 ? `${fmt(v / 1e3, 0)}K` : fmt(v, 0);
const labelDomain = (d: string) => d.replace('knowledge/multitopic', 'knowledge').replace('multilingual/', '').replace('vision/', 'vision · ');
const armName = (a: string) => a.replace('low_reap_', 'bottom ').replace('high_reap_', 'top ').replace('random_', 'random ').replace('pct', '%');

const DOMAIN_COLOR: Record<string, string> = {
  'agent': '#d3a05a', 'code': '#a891cf', 'general': '#8fb2bd', 'knowledge/multitopic': '#7f9cb8',
  'long-context': '#b7ac96', 'multilingual/es': '#e0955c', 'multilingual/ru': '#c46a5a', 'multilingual/zh': '#9a6a2c',
  'reasoning': '#6a4f8c', 'systems': '#3f7f7a', 'vision/chart': '#a98cc0', 'vision/document': '#7e6aa8',
  'vision/general': '#c9a7d8', 'vision/ocr': '#5c4a8f',
};
const dcol = (d: string) => DOMAIN_COLOR[d] || '#8a7a5f';
const AMBER = kindOf('attn').solid, TEAL = kindOf('lin').solid, PLUM = kindOf('mlp').solid, BLUE = kindOf('in').solid, ORANGE = kindOf('out').solid;
const TERRA = 'oklch(0.62 0.15 35)', DEEP_TEAL = 'oklch(0.55 0.08 200)';

// высокое значение = терракота, низкое = волна (обратная рампа атласа): для линий по глубине
const hot = (t: number) => ramp(1 - Math.max(0, Math.min(1, t)));
// последовательная рампа для теплокарт: интенсивность = контраст с фоном.
// светлая тема: бумага → янтарь → терракота → кирпич; тёмная: кирпич → терракота → янтарь → персик
const isDark = () => document.documentElement.dataset.theme === 'dark';
function seq(t: number): string {
  t = Math.max(0, Math.min(1, t));
  const L = isDark() ? 0.30 + 0.58 * t : 0.95 - 0.60 * t;
  const C = isDark() ? 0.06 + 0.09 * Math.sin(Math.PI * Math.min(1, t * 1.1)) + 0.03 * t : 0.02 + 0.13 * Math.min(1, t * 1.4);
  const H = isDark() ? 25 + 55 * t : 90 - 62 * t;
  return `oklch(${L.toFixed(3)} ${C.toFixed(3)} ${H.toFixed(1)})`;
}
// цвет подписи поверх ячейки seq(t)
const seqInk = (t: number) => (isDark() ? t < 0.5 : t > 0.5) ? '#fff6ea' : '#2f2b25';
// перерисовки канвасов при смене темы
const redraws: (() => void)[] = [];
// расходящаяся: волна (−) ↔ бумага ↔ терракота (+)
function diverge(t: number): string {
  t = Math.max(-1, Math.min(1, t));
  return t < 0 ? `oklch(${(0.93 + 0.35 * t).toFixed(3)} ${(0.02 - 0.08 * t).toFixed(3)} 210)`
    : `oklch(${(0.93 - 0.35 * t).toFixed(3)} ${(0.02 + 0.13 * t).toFixed(3)} 35)`;
}
function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}
function quantile(a: number[], q: number): number {
  const x = a.filter(Number.isFinite).sort((p, c) => p - c);
  return x[Math.max(0, Math.min(x.length - 1, Math.round(q * (x.length - 1))))] ?? 0;
}
const median = (a: number[]) => quantile(a, .5);

function cardShell(w: number, title: string, sub: string, hint = true): { root: HTMLElement; body: HTMLElement } {
  const root = el('div', 'card glm-card', `width:${w}px;padding:22px 26px 24px;display:flex;flex-direction:column;gap:14px`);
  root.innerHTML = `<div style="display:flex;align-items:baseline;justify-content:space-between;gap:14px">
      <div class="eyebrow accent mono">${title}</div>
      ${hint ? `<div class="mono" style="font-size:10px;color:var(--accent-ink);white-space:nowrap">✛ ${l('hover for exact values', 'наведите за точным значением')}</div>` : ''}
    </div>
    <div class="note" style="font-size:14.5px;max-width:${Math.min(w - 60, 1500)}px">${sub}</div>`;
  const body = el('div', '', 'display:flex;flex-direction:column;gap:14px');
  root.appendChild(body);
  return { root, body };
}
function chipRow(items: [string, string][], on: string, onPick: (k: string) => void): HTMLElement {
  const row = el('div', 'no-pan', 'display:flex;gap:6px;flex-wrap:wrap');
  for (const [k, name] of items) {
    const c = el('div', `chip mini no-pan${k === on ? ' on' : ''}`, '', name);
    c.dataset.k = k;
    c.onclick = (e) => { e.stopPropagation(); row.querySelectorAll('.chip').forEach(x => x.classList.toggle('on', (x as HTMLElement).dataset.k === k)); onPick(k); };
    row.appendChild(c);
  }
  return row;
}
function domainSelect(domains: string[], onPick: (d: string) => void, allLabel = l('all domains', 'все домены')): HTMLSelectElement {
  const s = document.createElement('select');
  s.className = 'plain no-pan';
  s.innerHTML = ['all', ...domains].map(d => `<option value="${d}">${d === 'all' ? allLabel : labelDomain(d)}</option>`).join('');
  s.onchange = () => onPick(s.value);
  s.onpointerdown = (e) => e.stopPropagation();
  return s;
}
function text(svg: SVGElement, x: number, y: number, s: string, o: { fill?: string; size?: number; anchor?: string; weight?: string; mono?: boolean; op?: number } = {}): SVGElement {
  const t = svgEl('text', { x, y, fill: o.fill || 'var(--faint)', 'font-size': o.size || 10, 'text-anchor': o.anchor || 'start', 'font-weight': o.weight || 'normal', opacity: o.op ?? 1 });
  if (o.mono !== false) t.setAttribute('font-family', 'var(--mono)');
  t.textContent = s;
  svg.appendChild(t);
  return t;
}
function frame(svg: SVGElement, x: number, y: number, w: number, h: number): void {
  svg.appendChild(svgEl('rect', { x, y, width: w, height: h, rx: 6, fill: 'var(--chart-bg)', stroke: 'var(--line)' }));
}
function path(vals: (number | null)[], px: (i: number) => number, py: (v: number) => number): string {
  let d = '', on = false;
  vals.forEach((v, i) => { if (v == null || !isFinite(v)) { on = false; return; } d += (on ? ' L' : 'M') + px(i).toFixed(1) + ',' + py(v).toFixed(1); on = true; });
  return d;
}
function line(svg: SVGElement, vals: (number | null)[], px: (i: number) => number, py: (v: number) => number, color: string, w = 2, dash = ''): void {
  svg.appendChild(svgEl('path', { d: path(vals, px, py), fill: 'none', stroke: color, 'stroke-width': w, 'stroke-linejoin': 'round', ...(dash ? { 'stroke-dasharray': dash } : {}) }));
}
function hoverDots(svg: SVGElement, vals: (number | null)[], px: (i: number) => number, py: (v: number) => number, color: string, html: (i: number) => string, onClick?: (i: number) => void, r = 3): void {
  vals.forEach((v, i) => {
    if (v == null || !isFinite(v)) return;
    const c = svgEl('circle', { cx: px(i), cy: py(v), r, fill: color, stroke: 'var(--card-solid)', 'stroke-width': 1 });
    c.addEventListener('mousemove', (e: MouseEvent) => tip(html(i), e.clientX, e.clientY));
    c.addEventListener('mouseleave', hideTip);
    if (onClick) { (c as any).style.cursor = 'pointer'; c.addEventListener('click', () => onClick(i)); }
    svg.appendChild(c);
  });
}
function gridY(svg: SVGElement, x0: number, x1: number, py: (v: number) => number, ticks: number[], f: (v: number) => string, lx = x0 - 6): void {
  for (const v of ticks) {
    svg.appendChild(svgEl('line', { x1: x0, x2: x1, y1: py(v), y2: py(v), stroke: 'var(--line)', 'stroke-dasharray': '2 5' }));
    text(svg, lx, py(v) + 3.5, f(v), { anchor: 'end', size: 9.5, fill: 'var(--ghost)' });
  }
}
function selectLayer(store: Store, layer: number): void {
  const L = store.model.langLayers.find(x => +x.label === layer);
  if (L) store.select({ type: 'layer', layer: L });
}
function legendRow(items: [string, string][], extra = ''): HTMLElement {
  return el('div', 'mono', 'font-size:10.5px;color:var(--faint);display:flex;gap:18px;flex-wrap:wrap;align-items:center',
    items.map(([c, n]) => `<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${c};vertical-align:-1px;margin-right:6px"></span>${n}</span>`).join('') + extra);
}
function rampBar(lo: string, hi: string, w = 220): string {
  const stops = [0, .25, .5, .75, 1].map(seq).join(',');
  return `<span class="mono" style="font-size:10px;color:var(--faint);display:inline-flex;align-items:center;gap:8px">${lo}<span style="display:inline-block;width:${w}px;height:9px;border-radius:3px;background:linear-gradient(90deg,${stops})"></span>${hi}</span>`;
}
function tile(value: string, name: string, note: string, accent = false): string {
  return `<div class="glm-tile"><div class="mono glm-tile-v" style="${accent ? 'color:var(--accent-deep)' : ''}">${value}</div>
    <div class="glm-tile-n">${name}</div><div class="mono glm-tile-s">${note}</div></div>`;
}

// canvas-теплокарта: рисуем в device pixels, наводим по CSS-координатам
interface Heat { wrap: HTMLElement; canvas: HTMLCanvasElement; draw: (fill: (c: number, r: number) => string | null, after?: (ctx: CanvasRenderingContext2D) => void) => void; pick: (e: MouseEvent) => { c: number; r: number } | null }
function heat(cols: number, rows: number, cw: number, ch: number, padL: number, padT: number): Heat {
  const W = padL + cols * cw, H = padT + rows * ch;
  const DPR = Math.min(2, window.devicePixelRatio || 1);
  const wrap = el('div', 'no-pan', `position:relative;width:${W}px;height:${H}px;cursor:crosshair`);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(W * DPR); canvas.height = Math.round(H * DPR);
  canvas.style.cssText = `width:${W}px;height:${H}px;display:block`;
  wrap.appendChild(canvas);
  const ctx = canvas.getContext('2d')!;
  return {
    wrap, canvas,
    draw(fill, after) {
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.clearRect(0, 0, W, H);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const f = fill(c, r);
        if (!f) continue;
        ctx.fillStyle = f;
        ctx.fillRect(padL + c * cw, padT + r * ch, cw - (cw > 4 ? 1 : 0), ch - (ch > 4 ? 1 : 0));
      }
      after?.(ctx);
    },
    pick(e) {
      const b = canvas.getBoundingClientRect();
      const x = (e.clientX - b.left) / b.width * W - padL, y = (e.clientY - b.top) / b.height * H - padT;
      const c = Math.floor(x / cw), r = Math.floor(y / ch);
      return (c < 0 || r < 0 || c >= cols || r >= rows) ? null : { c, r };
    },
  };
}

// ───────────────────────── 1. паспорт ─────────────────────────

function receiptCard(d: any, jump: (key: string) => void): HTMLElement {
  const m = d.meta;
  const { root, body } = cardShell(5800, l('MEASURED MODEL · CAPTURE RECEIPT', 'ИЗМЕРЕННАЯ МОДЕЛЬ · ПАСПОРТ CAPTURE'), l(
    'One deployed NVFP4 checkpoint on 4 RTX PRO 6000, 8 accepted captures, every number below from a real run. Click a finding to jump to its card.',
    'Один развёрнутый NVFP4-чекпоинт на 4 RTX PRO 6000, 8 принятых captures, каждое число ниже из реального прогона. Клик по находке переносит к её карточке.'), false);
  const reap = Object.values(m.reap_tokens_per_layer as Record<string, number>)[0] as number;
  const dyn = d.routing.dynamics.all, half = d.stability.halves.exact_reap_rho;
  const hl = d.memory.kda_heads.flat() as number[];
  const fc2 = median(d.quantization.fc2.map((r: any) => r.sqnr_db)), zero = median(d.quantization.fc2.map((r: any) => r.qdq_zero_fraction));
  const pr = Object.fromEntries(d.pruning.map((r: any) => [r.arm, r]));
  const vis = d.vision.arms;
  const findings: [string, string, string, string][] = [
    ['atlas', `${fmt(reap / 1e6, 2)}M`, l('tokens per routed layer, exact REAP', 'токенов на routed-слой, exact REAP'), l('42 layers × 288 experts, 14 domains kept apart', '42 слоя × 288 экспертов, 14 доменов раздельно')],
    ['trust', fix(half.median, 3), l('split-half Spearman of the ranking', 'split-half Spearman ранжирования'), l(`frequency alone scores ${fix(d.stability.controls.count.median, 2)}: wrong direction`, `одна частота даёт ${fix(d.stability.controls.count.median, 2)}: не туда`)],
    ['router', `${fix(dyn[0].effective, 1)} → ${fix(Math.max(...dyn.map((r: any) => r.effective)), 1)}`, l('effective experts of 8, layer 3 → 43', 'эффективных экспертов из 8, слой 3 → 43'), l(`top1 − top2 margin falls ${fix(dyn[0].margin, 3)} → ${fix(dyn[40].margin, 3)}`, `разрыв top1 − top2 падает ${fix(dyn[0].margin, 3)} → ${fix(dyn[40].margin, 3)}`)],
    ['kda', `${fix(Math.min(...hl), 2)} … ${big(Math.max(...hl))}`, l('KDA head half-life, tokens', 'half-life KDA-голов, токенов'), l('34 layers × 64 heads, no shared forget rate', '34 слоя × 64 головы, общей скорости забывания нет')],
    ['nvfp4', `${fix(fc2, 2)} dB`, l('median FC2-input SQNR, deployed scale', 'медианный SQNR на входе FC2, deployed scale'), l(`${pct(zero, 1)} of post-SwiGLU values become exact zeros`, `${pct(zero, 1)} значений после SwiGLU становятся нулями`)],
    ['vision', `${pct(vis.original.all.contains, 1)} vs ${pct(vis.blank_same_geometry.all.contains, 0)}`, l('reference contained: real image vs blank', 'reference contained: картинка vs blank'), l('same 40 questions, same pixel geometry', 'те же 40 вопросов, та же геометрия пикселей')],
    ['pruning', `${fix(pr.low_reap_2pct.normalized_edit_similarity, 3)} vs ${fix(pr.high_reap_2pct.normalized_edit_similarity, 3)}`, l('edit similarity: drop bottom 2% vs top 2%', 'edit similarity: убрать bottom 2% vs top 2%'), l('REAP extremes are not interchangeable', 'крайние по REAP не взаимозаменяемы')],
  ];
  const grid = el('div', '', 'display:grid;grid-template-columns:repeat(7,1fr);gap:12px');
  for (const [key, v, n, s] of findings) {
    const t = el('div', 'glm-tile glm-jump no-pan', '', `<div class="mono glm-tile-v">${v}</div><div class="glm-tile-n">${n}</div><div class="mono glm-tile-s">${s}</div><div class="mono glm-tile-go">→ ${l('open', 'открыть')}</div>`);
    t.onclick = (e) => { e.stopPropagation(); jump(key); };
    grid.appendChild(t);
  }
  body.appendChild(grid);

  const capNames: Record<string, [string, string]> = {
    inventory: ['checkpoint inventory', 'инвентарь чекпоинта'], scales: ['block-scale scan', 'скан block scales'], router: ['router dynamics', 'динамика роутера'],
    contrib: ['shared vs routed', 'shared против routed'], rich: ['KDA + indexer', 'KDA + indexer'], vision: ['paired vision', 'парный vision'],
    fc2: ['FC2 activation QDQ', 'QDQ активаций FC2'], pruning: ['causal REAP arms', 'causal REAP arms'],
  };
  const ledger = el('div', '', 'display:flex;gap:8px;align-items:stretch');
  for (const c of m.captures) {
    const [en, ru] = capNames[c.key] || [c.key, c.key];
    ledger.appendChild(el('div', 'glm-cap', '', `<div class="mono" style="font-size:9.5px;color:var(--ghost)">${c.id} · ${c.gpu ? 'GPU' : 'CPU'}</div>
      <div style="font-size:12.5px;color:var(--ink-soft);margin-top:3px">${l(en, ru)}</div>
      <div class="mono" style="font-size:10px;color:var(--faint);margin-top:4px">${c.wall_s ? `${fmt(c.wall_s, 0)} s ${l('wall', 'wall')}` : l('storage I/O only', 'только диск')}</div>`));
  }
  body.appendChild(ledger);
  body.appendChild(el('div', 'mono', 'font-size:10.5px;color:var(--faint);display:flex;gap:14px;flex-wrap:wrap;align-items:center', [
    `<b style="color:var(--ink-soft)">${m.model}</b>`, `${fmt(m.weight_bytes / 1e9, 2)} GB`, `${fmt(m.tensor_count, 0)} ${l('tensors', 'тензоров')}`,
    `vLLM ${m.runtime.vllm}`, `torch ${m.runtime.torch}`, `transformers ${m.runtime.transformers}`,
    `${Object.keys(m.checksums).length} SHA-256`, `${fmt(m.router_records, 0)} ${l('behavioural rows', 'behavioural-строк')} · ${big(m.router_tokens)} ${l('tokens', 'токенов')}`,
    `<span style="color:var(--good)">${l('no prompts, generations, images, activations or raw routes stored', 'prompts, generations, изображения, активации и raw routes не сохранялись')}</span>`,
  ].map(x => `<span>${x}</span>`).join('<span style="color:var(--line-strong)">·</span>')));
  return root;
}

// ───────────────────────── 2. атлас экспертов ─────────────────────────

function atlasCard(d: any, store: Store): HTMLElement {
  const R = d.routing, layers: number[] = R.layers, E = R.experts as number, N = layers.length;
  const { root, body } = cardShell(5800, l('THE EXPERT ATLAS · 42 × 288', 'АТЛАС ЭКСПЕРТОВ · 42 × 288'), l(
    `${fmt(N * E, 0)} cells, one per routed expert. Row = layer, column = expert. Switch what colour means, slice REAP by domain, outline a prune set, click a cell to open its dossier on the right.`,
    `${fmt(N * E, 0)} ячеек, по одной на routed-эксперта. Строка = слой, столбец = эксперт. Переключайте смысл цвета, режьте REAP по доменам, обводите prune-set, кликайте по ячейке за досье справа.`));
  let metric = 'reap', domain = 'all', relative = false, sorted = false, overlay = 'none';
  let pinned: { r: number; e: number } | null = null, hovered: { r: number; e: number } | null = null;
  const metricDefs: Record<string, { name: string; fmt: (v: number) => string; log: boolean }> = {
    reap: { name: 'exact REAP', fmt: v => fmt(v, 4), log: true },
    route_share: { name: l('route share', 'доля маршрутов'), fmt: v => pct(v, 3), log: true },
    top1_share: { name: l('top-1 share', 'доля top-1'), fmt: v => pct(v, 3), log: true },
    contribution: { name: l('output contribution', 'вклад в выход'), fmt: v => fmt(v, 3), log: false },
  };
  const ctl = el('div', 'no-pan', 'display:flex;gap:14px;align-items:center;flex-wrap:wrap');
  ctl.appendChild(chipRow(Object.entries(metricDefs).map(([k, v]) => [k, v.name]), metric, k => { metric = k; domWrap.style.opacity = k === 'reap' ? '1' : '.35'; domWrap.style.pointerEvents = k === 'reap' ? '' : 'none'; draw(); }));
  const domWrap = el('div', '', 'display:flex;gap:8px;align-items:center');
  domWrap.appendChild(domainSelect(R.domains, v => { domain = v; relChip.style.display = v === 'all' ? 'none' : ''; draw(); }));
  const relChip = el('div', 'chip mini no-pan', 'display:none', l('÷ all domains', '÷ все домены'));
  relChip.onclick = (e) => { e.stopPropagation(); relative = !relative; relChip.classList.toggle('on', relative); draw(); };
  domWrap.appendChild(relChip); ctl.appendChild(domWrap);
  ctl.appendChild(chipRow([['id', l('expert id order', 'по id эксперта')], ['rank', l('ranked within layer', 'ранжировано в слое')]], 'id', k => { sorted = k === 'rank'; draw(); }));
  const ovSel = document.createElement('select'); ovSel.className = 'plain no-pan';
  ovSel.innerHTML = `<option value="none">${l('no prune-set outline', 'без обводки prune-set')}</option>` + Object.keys(R.prune_sets).map(a => `<option value="${a}">${l('outline', 'обвести')}: ${armName(a)}</option>`).join('');
  ovSel.onchange = () => { overlay = ovSel.value; draw(); }; ovSel.onpointerdown = e => e.stopPropagation();
  ctl.appendChild(ovSel);
  body.appendChild(ctl);

  const cw = 17, ch = 13, padL = 54, padT = 22;
  const H = heat(E, N, cw, ch, padL, padT);
  const marginW = 90;
  const row = el('div', '', 'display:flex;gap:18px;align-items:flex-start');
  const left = el('div', '', 'display:flex;flex-direction:column;gap:10px');
  const canvasRow = el('div', '', 'display:flex;gap:6px;align-items:flex-start');
  canvasRow.appendChild(H.wrap);
  const marg = svgEl('svg', { width: marginW, height: padT + N * ch });
  canvasRow.appendChild(marg as any);
  left.appendChild(canvasRow);
  const legend = el('div', '', 'display:flex;gap:26px;align-items:center;flex-wrap:wrap');
  left.appendChild(legend);
  row.appendChild(left);
  const insp = el('div', 'glm-insp no-pan', `width:${5800 - 52 - padL - E * cw - marginW - 24 - 20 - 42}px;min-height:${padT + N * ch}px`);
  row.appendChild(insp);
  body.appendChild(row);

  // gini по слоям как правое поле
  const gini = R.dynamics.all.map((r: any) => r.selected_gini as number);
  const gmax = Math.max(...gini);
  text(marg, 0, 12, l('load gini', 'gini нагрузки'), { size: 9 });
  gini.forEach((g: number, i: number) => {
    const b = svgEl('rect', { x: 0, y: padT + i * ch + 1, width: Math.max(2, g / gmax * (marginW - 4)), height: ch - 3, rx: 1.5, fill: seq(.35 + .65 * g / gmax), opacity: .9 });
    b.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${layers[i]}</b><br>${l('selected-load Gini', 'Gini нагрузки')}: ${fix(g, 3)}<br>top-1 Gini: ${fix(R.dynamics.all[i].top1_gini, 3)}`, e.clientX, e.clientY));
    b.addEventListener('mouseleave', hideTip);
    marg.appendChild(b);
  });

  let order: number[][] = [];
  let matrix: number[][] = [], lo = 0, hi = 1, isRel = false;
  const rankIn = (r: number, e: number) => { const v = R.reap[r]; let k = 0; for (let i = 0; i < E; i++) if (v[i] > v[e]) k++; return k + 1; };
  function draw() {
    isRel = metric === 'reap' && domain !== 'all' && relative;
    const base = metric === 'reap' ? (domain === 'all' ? R.reap : R.reap_domains[domain]) : R[metric];
    matrix = isRel ? base.map((rowv: number[], r: number) => rowv.map((v, e) => Math.log2(Math.max(v, 1e-9) / Math.max(R.reap[r][e], 1e-9)))) : base;
    const flat = matrix.flat().filter((v: number) => Number.isFinite(v) && (isRel || v > 0));
    const log = metricDefs[metric].log && !isRel;
    const tx = (v: number) => log ? Math.log10(Math.max(v, 1e-12)) : v;
    if (isRel) { const m = Math.min(2.5, quantile(flat.map(Math.abs), .98)); lo = -m; hi = m; } else { lo = quantile(flat.map(tx), .02); hi = quantile(flat.map(tx), .98); }
    order = matrix.map((rowv: number[]) => { const ids = [...Array(E).keys()]; return sorted ? ids.sort((a, b) => rowv[b] - rowv[a]) : ids; });
    const setOf = overlay !== 'none' ? R.prune_sets[overlay] : null;
    H.draw((c, r) => {
      const e = order[r][c], v = matrix[r][e];
      if (!Number.isFinite(v) || (!isRel && v <= 0)) return cssVar('--hatch-a');
      return isRel ? diverge(v / hi) : seq((tx(v) - lo) / (hi - lo || 1));
    }, (ctx) => {
      ctx.font = '9.5px IBM Plex Mono, monospace';
      ctx.fillStyle = cssVar('--faint');
      layers.forEach((L, i) => { if (i % 3 === 0 || i === N - 1) ctx.fillText(`L${L}`, 22, padT + i * ch + 10); });
      ctx.fillStyle = cssVar('--ghost');
      for (let e = 0; e < E; e += 16) ctx.fillText(sorted ? `#${e + 1}` : String(e), padL + e * cw, 14);
      if (setOf) {
        ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 1.4;
        for (let r = 0; r < N; r++) { const inv = new Map(order[r].map((e, c) => [e, c])); for (const e of setOf[r]) { const c = inv.get(e)!; ctx.strokeRect(padL + c * cw + .7, padT + r * ch + .7, cw - 2.4, ch - 2.4); } }
      }
      const mark = (p: { r: number; e: number }, color: string, w: number) => {
        const c = order[p.r].indexOf(p.e);
        ctx.strokeStyle = color; ctx.lineWidth = w; ctx.strokeRect(padL + c * cw - 1, padT + p.r * ch - 1, cw + 1, ch + 1);
      };
      if (hovered) mark(hovered, cssVar('--ink'), 1.2);
      if (pinned) mark(pinned, cssVar('--accent'), 2);
    });
    legend.innerHTML = isRel
      ? `<span class="mono" style="font-size:10px;color:var(--faint);display:inline-flex;align-items:center;gap:8px">÷${fix(Math.pow(2, hi), 1)}<span style="display:inline-block;width:220px;height:9px;border-radius:3px;background:linear-gradient(90deg,${[-1, -.5, 0, .5, 1].map(diverge).join(',')})"></span>×${fix(Math.pow(2, hi), 1)}</span><span class="mono" style="font-size:10px;color:var(--ghost)">${labelDomain(domain)} REAP ÷ all-domain REAP, log₂, ${l('clipped at the 98th percentile', 'обрезано 98-м перцентилем')}</span>`
      : rampBar(l('low', 'ниже'), l('high', 'выше')) + `<span class="mono" style="font-size:10px;color:var(--ghost)">${metricDefs[metric].name}${domain !== 'all' && metric === 'reap' ? ' · ' + labelDomain(domain) : ''} · ${log ? 'log' : 'linear'} · ${l('clipped 2nd–98th pct', 'обрезано 2–98 перцентилем')} · ${l('hatched = no observation', 'штрих = нет наблюдений')}</span>`;
    inspector();
  }
  H.wrap.addEventListener('mousemove', (e: MouseEvent) => {
    const p = H.pick(e);
    if (!p) { hideTip(); return; }
    const ex = order[p.r][p.c], v = matrix[p.r][ex];
    hovered = { r: p.r, e: ex };
    tip(`<b>L${layers[p.r]} · ${l('expert', 'эксперт')} ${ex}</b> · ${l('rank', 'ранг')} ${rankIn(p.r, ex)}/${E}<br>${metricDefs[metric].name}${isRel ? ' ÷ all' : ''}: <span class="mono">${isRel ? `×${fix(Math.pow(2, v), 2)}` : metricDefs[metric].fmt(v)}</span><br>REAP ${fmt(R.reap[p.r][ex], 3)} · ${l('routes', 'маршруты')} ${pct(R.route_share[p.r][ex], 2)} · top-1 ${pct(R.top1_share[p.r][ex], 2)}`, e.clientX, e.clientY);
    if (!pinned) inspector();
  });
  H.wrap.addEventListener('mouseleave', () => { hideTip(); hovered = null; if (!pinned) inspector(); });
  H.wrap.addEventListener('click', (e: MouseEvent) => {
    const p = H.pick(e); if (!p) return;
    const ex = order[p.r][p.c];
    pinned = pinned && pinned.r === p.r && pinned.e === ex ? null : { r: p.r, e: ex };
    draw();
  });

  function inspector() {
    const p = pinned || hovered;
    if (!p) {
      insp.innerHTML = `<div class="eyebrow mono">${l('expert dossier', 'досье эксперта')}</div>
        <div class="note" style="margin-top:10px">${l('Hover a cell to preview, click to pin. The dossier shows the four measured views of that expert, its domain profile from exact REAP, its most frequent co-routing partners and every prune set it belongs to.', 'Наведите на ячейку для превью, кликните, чтобы закрепить. Досье показывает четыре измеренных вида эксперта, доменный профиль по exact REAP, самых частых партнёров по co-routing и все prune-set, куда он входит.')}</div>
        <div style="margin-top:18px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
          ${tile(fmt(N * E, 0), l('expert cells', 'ячеек экспертов'), l('all with observations', 'все с наблюдениями'))}
          ${tile(big(Object.values(d.meta.reap_tokens_per_layer)[0] as number), l('REAP tokens per layer', 'REAP-токенов на слой'), l('exact, not sampled', 'точно, не сэмпл'))}
          ${tile(big(d.meta.routed_tokens_per_layer), l('routed tokens per layer', 'routed-токенов на слой'), l('router dynamics pass', 'прогон динамики роутера'))}
          ${tile(String(R.domains.length), l('domains', 'доменов'), l('4 of them real images', '4 из них реальные изображения'))}
        </div>`;
      return;
    }
    const L = layers[p.r], ex = p.e, rk = rankIn(p.r, ex);
    const prof = R.domains.map((dm: string) => [dm, Math.log2(Math.max(R.reap_domains[dm][p.r][ex], 1e-9) / Math.max(R.reap[p.r][ex], 1e-9))] as [string, number]).sort((a: any, b: any) => b[1] - a[1]);
    const co = R.coroute[p.r];
    const partners = [...co.count.filter((q: number[]) => q[0] === ex || q[1] === ex).map((q: number[]) => ({ e: q[0] === ex ? q[1] : q[0], n: q[2], f: q[3], kind: 'count' })),
      ...co.lift.filter((q: number[]) => q[0] === ex || q[1] === ex).map((q: number[]) => ({ e: q[0] === ex ? q[1] : q[0], n: q[2], f: q[3], kind: 'lift' }))];
    const sets = Object.entries(R.prune_sets).filter(([, s]: any) => s[p.r].includes(ex)).map(([a]) => a);
    const aligned = (d.contributions.pairs_extreme[p.r].aligned as number[][]).filter(q => q[0] === ex || q[1] === ex);
    insp.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:baseline">
        <div class="eyebrow mono">${l('expert dossier', 'досье эксперта')}${pinned ? ` · <span style="color:var(--accent-deep)">${l('pinned', 'закреплено')}</span>` : ''}</div>
        <div class="mono" style="font-size:10px;color:var(--ghost)">${pinned ? l('click again to release', 'клик ещё раз отпускает') : l('click to pin', 'клик закрепляет')}</div></div>
      <div style="font-size:26px;margin-top:8px;letter-spacing:-.01em">${l('Layer', 'Слой')} ${L} · ${l('expert', 'эксперт')} ${ex}</div>
      <div class="mono" style="font-size:11px;color:var(--faint);margin-top:2px">REAP ${l('rank', 'ранг')} ${rk} ${l('of', 'из')} ${E} ${l('in this layer', 'в этом слое')} · ${rk <= 6 ? `<span style="color:var(--accent-deep)">top 2%</span>` : rk > E - 6 ? `<span style="color:${DEEP_TEAL}">bottom 2%</span>` : l('middle of the pack', 'середина')}</div>
      <div style="margin-top:14px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px">
        ${tile(fmt(R.reap[p.r][ex], 3), 'exact REAP', l('mean over 12.59M tokens', 'среднее по 12.59M токенов'))}
        ${tile(pct(R.route_share[p.r][ex], 2), l('route share', 'доля маршрутов'), l('uniform = 2.78%', 'равномерно = 2.78%'))}
        ${tile(pct(R.top1_share[p.r][ex], 2), l('top-1 share', 'доля top-1'), l('uniform = 0.35%', 'равномерно = 0.35%'))}
        ${tile(fmt(R.contribution[p.r][ex], 2), l('output norm', 'норма выхода'), l('weighted, sampled replay', 'взвешенно, sampled replay'))}
      </div>
      <div class="eyebrow mono" style="margin-top:18px">${l('domain profile · REAP ÷ all-domain REAP', 'доменный профиль · REAP ÷ REAP по всем доменам')}</div>
      <div style="margin-top:8px;display:flex;flex-direction:column;gap:3px">${prof.map(([dm, v]: any) => {
        const w = Math.min(1, Math.abs(v) / 3) * 50;
        return `<div style="display:flex;align-items:center;gap:8px;font-size:11.5px"><span style="width:118px;color:var(--muted);text-align:right">${labelDomain(dm)}</span>
          <span style="position:relative;flex:1;height:11px;background:var(--track);border-radius:3px"><span style="position:absolute;top:0;height:100%;left:${v < 0 ? 50 - w : 50}%;width:${w}%;background:${diverge(v / 3)};border-radius:2px"></span><span style="position:absolute;left:50%;top:0;height:100%;width:1px;background:var(--line-strong)"></span></span>
          <span class="mono" style="width:52px;color:${v > .6 ? 'var(--accent-deep)' : v < -.6 ? DEEP_TEAL : 'var(--faint)'}">×${fix(Math.pow(2, v), 2)}</span></div>`; }).join('')}</div>
      <div style="display:flex;gap:18px;margin-top:18px">
        <div style="flex:1"><div class="eyebrow mono">${l('co-routing partners', 'партнёры по co-routing')}</div>
          <div style="margin-top:8px;font-size:12px;color:var(--muted);line-height:1.6">${partners.length ? partners.slice(0, 8).map(q => `<div>E${q.e} <span class="mono" style="color:var(--faint)">${q.kind === 'count' ? `${fmt(q.n, 0)} ${l('tokens', 'токенов')} · ${pct(q.f, 2)}` : `lift ${fix(q.f, 1)} · ${fmt(q.n, 0)}`}</span></div>`).join('') : `<span style="color:var(--ghost)">${l('not in the top-16 pairs of this layer', 'не входит в top-16 пар этого слоя')}</span>`}</div></div>
        <div style="flex:1"><div class="eyebrow mono">prune sets</div>
          <div style="margin-top:8px;display:flex;gap:5px;flex-wrap:wrap">${sets.length ? sets.map(a => `<span class="chip mini" style="cursor:default;${a.startsWith('high') ? 'border-color:var(--accent);color:var(--accent-deep)' : ''}">${armName(a)}</span>`).join('') : `<span style="font-size:12px;color:var(--ghost)">${l('in no ablation arm', 'ни в одном arm')}</span>`}</div>
          ${aligned.length ? `<div class="eyebrow mono" style="margin-top:14px">${l('aligned output pair', 'согласованная пара выходов')}</div><div class="mono" style="font-size:11px;color:var(--muted);margin-top:6px">${aligned.map(q => `E${q[0]}+E${q[1]} cos ${fix(q[3], 3)} · n=${q[2]}`).join('<br>')}</div>` : ''}
        </div></div>
      <div class="mono glm-link" style="margin-top:16px">↗ ${l('open layer', 'открыть слой')} ${L} ${l('on the wall', 'на стене')}</div>`;
    (insp.querySelector('.glm-link') as HTMLElement).onclick = (e) => { e.stopPropagation(); selectLayer(store, L); };
  }
  draw(); redraws.push(draw);
  body.appendChild(el('div', 'small-note', '', l(
    'Colour is normalized inside the selected view: REAP, route frequency, top-1 frequency and output norm are different units and are never mixed on one scale. Ranked order sorts each row independently, so a column no longer means one expert; the tooltip always names the real one.',
    'Цвет нормируется внутри выбранного вида: REAP, частота маршрутов, частота top-1 и норма выхода это разные единицы и на одну шкалу не смешиваются. Ранжированный порядок сортирует каждую строку отдельно, столбец перестаёт значить одного эксперта; tooltip всегда называет настоящего.')));
  return root;
}

// ───────────────────────── 3. роутер под нагрузкой ─────────────────────────

function routerCard(d: any, store: Store): HTMLElement {
  const R = d.routing, layers: number[] = R.layers, N = layers.length;
  const { root, body } = cardShell(2850, l('ROUTER UNDER LOAD', 'РОУТЕР ПОД НАГРУЗКОЙ'), l(
    'Top-8 routing gets flatter with depth: the mixture widens, the top-1 lead shrinks, but corpus-wide load stays uneven. Pick a layer to see which experts travel together.',
    'Top-8 routing с глубиной становится ровнее: смесь шире, отрыв top-1 меньше, но нагрузка по корпусу остаётся неравномерной. Выберите слой, чтобы увидеть, какие эксперты ходят вместе.'));
  let domain = 'all', sel = N - 1, pairMode: 'count' | 'lift' = 'count';
  const ctl = el('div', 'no-pan', 'display:flex;gap:14px;align-items:center;flex-wrap:wrap');
  ctl.appendChild(domainSelect(R.domains, v => { domain = v; drawLines(); }));
  const layerLabel = el('div', 'mono', 'font-size:11px;color:var(--faint)');
  ctl.appendChild(layerLabel);
  body.appendChild(ctl);
  const wrap = el('div', '', 'display:flex;gap:20px;align-items:flex-start');
  const LW = 1720, LH = 560, lines = svgEl('svg', { width: LW, height: LH, class: 'no-pan' });
  const ringW = 1020, ringH = 560, ring = svgEl('svg', { width: ringW, height: ringH, class: 'no-pan' });
  wrap.appendChild(lines as any);
  const rightCol = el('div', '', 'display:flex;flex-direction:column;gap:8px');
  rightCol.appendChild(chipRow([['count', l('most frequent pairs', 'самые частые пары')], ['lift', l('strongest affinity (lift)', 'сильнейшее сродство (lift)')]], 'count', k => { pairMode = k as any; drawRing(); }));
  rightCol.appendChild(ring as any);
  wrap.appendChild(rightCol);
  body.appendChild(wrap);

  const x0 = 60, chartW = LW - x0 - 20, px = (i: number) => x0 + i / (N - 1) * chartW;
  const panels: [string, string[], string[]][] = [
    [l('effective experts of 8', 'эффективных экспертов из 8'), ['effective'], [TEAL]],
    [l('top1 − top2 margin', 'разрыв top1 − top2'), ['margin'], [TERRA]],
    [l('load Gini: selected · top-1', 'Gini нагрузки: selected · top-1'), ['selected_gini', 'top1_gini'], [PLUM, AMBER]],
  ];
  function drawLines() {
    lines.innerHTML = '';
    const rows = domain === 'all' ? R.dynamics.all : R.dynamics.domains[domain];
    const ph = 120, gap = 22;
    panels.forEach(([name, keys, colors], pi) => {
      const y0 = 8 + pi * (ph + gap);
      const all = keys.flatMap(k => rows.map((r: any) => r[k] as number));
      let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * .15 || .05; lo -= pad; hi += pad;
      const py = (v: number) => y0 + ph - 6 - (v - lo) / (hi - lo) * (ph - 24);
      frame(lines, x0 - 8, y0, chartW + 16, ph);
      text(lines, x0, y0 + 14, name, { fill: 'var(--ink-soft)', size: 10.5 });
      gridY(lines, x0, x0 + chartW, py, [lo + pad, hi - pad], v => fix(v, 2));
      keys.forEach((k, ki) => {
        const vals = rows.map((r: any) => r[k]);
        line(lines, vals, px, py, colors[ki], 2);
        hoverDots(lines, vals, px, py, colors[ki], i => `<b>L${rows[i].layer}</b> · ${domain === 'all' ? l('all domains', 'все домены') : labelDomain(domain)}<br>${name}: <span class="mono">${fix(vals[i], 4)}</span><br>${fmt(rows[i].tokens, 0)} ${l('routed tokens', 'routed-токенов')}`, i => { sel = i; drawRing(); selectLayer(store, rows[i].layer); }, 3.2);
      });
    });
    // позиционные бакеты: gini нагрузки по позиции промпта
    const y0 = 8 + 3 * (ph + gap), cellW = chartW / N, cellH = 15;
    text(lines, x0, y0 + 10, l('selected-load Gini by prompt position', 'Gini нагрузки по позиции в промпте'), { fill: 'var(--ink-soft)', size: 10.5 });
    const G: number[][] = R.position.gini, gflat = G.flat(), glo = Math.min(...gflat), ghi = Math.max(...gflat);
    const buckets: string[] = d.memory.position_buckets;
    G.forEach((rowv, i) => rowv.forEach((g, b) => {
      const r = svgEl('rect', { x: px(i) - cellW / 2 + .5, y: y0 + 18 + b * cellH, width: cellW - 1, height: cellH - 1, fill: seq((g - glo) / (ghi - glo || 1)), rx: 1 });
      r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${layers[i]} · ${l('positions', 'позиции')} ${buckets[b]}</b><br>Gini ${fix(g, 3)}<br>${fmt(R.position.token_count[b], 0)} ${l('tokens in this bucket', 'токенов в бакете')}`, e.clientX, e.clientY));
      r.addEventListener('mouseleave', hideTip);
      lines.appendChild(r);
    }));
    buckets.forEach((b, i) => text(lines, x0 - 8, y0 + 18 + i * cellH + 11, b, { anchor: 'end', size: 9 }));
    layers.forEach((L, i) => { if (i % 3 === 0 || i === N - 1) text(lines, px(i), LH - 4, `L${L}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }); });
    const guide = svgEl('line', { x1: px(sel), x2: px(sel), y1: 4, y2: LH - 14, stroke: 'var(--accent)', 'stroke-width': 1.2, 'stroke-dasharray': '3 4', opacity: .8 });
    guide.setAttribute('id', 'glm-router-guide');
    lines.appendChild(guide);
  }
  lines.addEventListener('click', (e: MouseEvent) => {
    const b = lines.getBoundingClientRect(); const i = Math.round(((e.clientX - b.left) / b.width * LW - x0) / (chartW / (N - 1)));
    if (i >= 0 && i < N) { sel = i; drawRing(); }
  });
  function drawRing() {
    ring.innerHTML = '';
    const guide = lines.querySelector('#glm-router-guide'); if (guide) { guide.setAttribute('x1', String(px(sel))); guide.setAttribute('x2', String(px(sel))); }
    const L = layers[sel], co = R.coroute[sel], E = R.experts, cx = 300, cy = 290, rad = 246;
    layerLabel.textContent = `${l('layer', 'слой')} L${L} · ${l('click anywhere on the charts to change', 'клик по графику меняет слой')}`;
    text(ring, cx, 22, `L${L} · ${pairMode === 'count' ? l('16 most frequent pairs', '16 самых частых пар') : l('16 pairs with the highest lift', '16 пар с наибольшим lift')}`, { anchor: 'middle', fill: 'var(--ink-soft)', size: 11.5 });
    const reap = R.reap[sel], rl = Math.log10(quantile(reap, .02)), rh = Math.log10(quantile(reap, .98));
    const ang = (e: number) => -Math.PI / 2 + e / E * 2 * Math.PI;
    for (let e = 0; e < E; e++) {
      const a = ang(e), t = (Math.log10(Math.max(reap[e], 1e-9)) - rl) / (rh - rl || 1);
      const s = svgEl('line', { x1: cx + Math.cos(a) * rad, y1: cy + Math.sin(a) * rad, x2: cx + Math.cos(a) * (rad + 9), y2: cy + Math.sin(a) * (rad + 9), stroke: seq(.2 + .8 * t), 'stroke-width': 2.2 });
      s.addEventListener('mousemove', (ev: MouseEvent) => tip(`<b>E${e}</b> · REAP ${fmt(reap[e], 3)}<br>${l('route share', 'доля маршрутов')} ${pct(R.route_share[sel][e], 2)}`, ev.clientX, ev.clientY));
      s.addEventListener('mouseleave', hideTip);
      ring.appendChild(s);
      if (e % 36 === 0) text(ring, cx + Math.cos(a) * (rad + 22), cy + Math.sin(a) * (rad + 22) + 3, String(e), { anchor: 'middle', size: 9, fill: 'var(--ghost)' });
    }
    const pairs: number[][] = co[pairMode];
    const wmax = Math.max(...pairs.map(p => p[pairMode === 'count' ? 2 : 3]));
    pairs.forEach((p, i) => {
      const a1 = ang(p[0]), a2 = ang(p[1]);
      const x1 = cx + Math.cos(a1) * rad, y1 = cy + Math.sin(a1) * rad, x2 = cx + Math.cos(a2) * rad, y2 = cy + Math.sin(a2) * rad;
      const w = 1 + 6 * (p[pairMode === 'count' ? 2 : 3] / wmax);
      const arc = svgEl('path', { d: `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`, fill: 'none', stroke: i === 0 ? 'var(--accent)' : pairMode === 'count' ? PLUM : DEEP_TEAL, 'stroke-width': w, opacity: .55, 'stroke-linecap': 'round' });
      arc.addEventListener('mousemove', (ev: MouseEvent) => { arc.setAttribute('opacity', '1'); tip(`<b>E${p[0]} + E${p[1]}</b><br>${fmt(p[2], 0)} ${l('tokens together', 'токенов вместе')}${pairMode === 'count' ? ` · ${pct(p[3], 2)} ${l('of tokens', 'токенов')}` : ` · lift ${fix(p[3], 1)}`}`, ev.clientX, ev.clientY); });
      arc.addEventListener('mouseleave', () => { arc.setAttribute('opacity', '.55'); hideTip(); });
      ring.appendChild(arc);
    });
    // список справа от кольца
    const lx = 620;
    text(ring, lx, 60, pairMode === 'count' ? l('pair · tokens · share', 'пара · токенов · доля') : l('pair · tokens · lift', 'пара · токенов · lift'), { fill: 'var(--ink-soft)', size: 10.5 });
    pairs.slice(0, 16).forEach((p, i) => {
      const y = 82 + i * 21;
      text(ring, lx, y, `E${p[0]} + E${p[1]}`, { fill: i === 0 ? 'var(--accent-deep)' : 'var(--muted)', size: 11 });
      text(ring, lx + 150, y, fmt(p[2], 0), { anchor: 'end', size: 10.5, fill: 'var(--faint)' });
      text(ring, lx + 240, y, pairMode === 'count' ? pct(p[3], 2) : `×${fix(p[3], 1)}`, { anchor: 'end', size: 10.5, fill: 'var(--faint)' });
      const bw = 130 * (p[pairMode === 'count' ? 2 : 3] / wmax);
      ring.appendChild(svgEl('rect', { x: lx + 256, y: y - 9, width: bw, height: 11, rx: 2, fill: i === 0 ? 'var(--accent)' : pairMode === 'count' ? PLUM : DEEP_TEAL, opacity: .7 }));
    });
    text(ring, lx, 82 + 16 * 21 + 8, `${l('lift needs ≥', 'lift считается от ≥')} ${co.min_count} ${l('co-occurrences', 'совпадений')} · ${l('tick colour = REAP', 'цвет штриха = REAP')}`, { size: 9.5, fill: 'var(--ghost)' });
  }
  drawLines(); drawRing(); redraws.push(() => { drawLines(); drawRing(); });
  body.appendChild(el('div', 'small-note', '', l(
    'Co-routing is association on this balanced corpus, not evidence that two experts are interchangeable. Lift = observed pair count ÷ the count expected if the two were routed independently.',
    'Co-routing это ассоциация на сбалансированном корпусе, а не доказательство взаимозаменяемости. Lift = наблюдаемое число пар ÷ ожидаемое при независимом выборе.')));
  return root;
}

// ───────────────────────── 4. доверие к ранжированию ─────────────────────────

function trustCard(d: any): HTMLElement {
  const S = d.stability, half = S.halves.exact_reap_rho, ctrl = S.controls;
  const { root, body } = cardShell(2900, l('CAN THE RANKING BE TRUSTED?', 'МОЖНО ЛИ ДОВЕРЯТЬ РАНЖИРОВАНИЮ?'), l(
    'The 12.8M-token REAP profile was split into two disjoint halves and re-ranked. Then three cheaper proxies were checked against the exact score, and every domain against the global list.',
    'REAP-профиль на 12.8M токенов разрезали на две непересекающиеся половины и переранжировали. Затем три дешёвых прокси сверили с точным score, а каждый домен со сводным списком.'));
  body.appendChild(el('div', '', 'display:grid;grid-template-columns:repeat(5,1fr);gap:10px', [
    tile(fix(half.median, 4), 'split-half Spearman ρ', `${l('min', 'мин')} ${fix(half.min, 4)} · 42 ${l('layers', 'слоя')}`, true),
    tile(fix(S.halves.exact_reap_jaccard['108'].median, 3), l('keep-set Jaccard @108', 'Jaccard keep-set @108'), `p10 ${fix(S.halves.exact_reap_jaccard['108'].p10, 3)} · ${l('min', 'мин')} ${fix(S.halves.exact_reap_jaccard['108'].min, 3)}`),
    tile(fix(ctrl.proxy.median, 3), l('input proxy vs exact', 'input proxy vs exact'), l('cheap and nearly right', 'дёшево и почти верно')),
    tile(fix(ctrl.mass.median, 3), l('routing mass vs exact', 'routing mass vs exact'), l('weak', 'слабо')),
    tile(fix(ctrl.count.median, 3), l('selection count vs exact', 'частота выбора vs exact'), l('negative: popular ≠ important', 'отрицательно: популярный ≠ важный')),
  ].join('')));
  const W = 2850, H = 470, svg = svgEl('svg', { width: W, height: H, class: 'no-pan' });
  body.appendChild(svg as any);
  // A: контроли на оси ρ
  const ax = 190, aw = 720, ay = 34;
  text(svg, ax, 18, l('Spearman ρ against exact REAP · dot = median, bar = p10 … min', 'Spearman ρ к exact REAP · точка = медиана, полоса = p10 … мин'), { fill: 'var(--ink-soft)', size: 10.5 });
  const rx = (v: number) => ax + (v + .7) / 1.7 * aw;
  frame(svg, ax - 6, ay, aw + 12, 190);
  for (const g of [-.5, 0, .5, 1]) { svg.appendChild(svgEl('line', { x1: rx(g), x2: rx(g), y1: ay + 4, y2: ay + 186, stroke: 'var(--line)', 'stroke-dasharray': '2 5' })); text(svg, rx(g), ay + 204, fix(g, 1), { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' }); }
  const rowsA: [string, any, string][] = [
    [l('split half A vs B', 'половина A vs B'), half, 'var(--accent)'],
    [l('input-norm proxy', 'прокси по норме входа'), ctrl.proxy, TEAL],
    ['routing mass', ctrl.mass, PLUM],
    [l('selection count', 'частота выбора'), ctrl.count, TERRA],
  ];
  rowsA.forEach(([name, s, color], i) => {
    const y = ay + 30 + i * 42;
    text(svg, ax - 14, y + 4, name, { anchor: 'end', size: 11, fill: 'var(--muted)' });
    svg.appendChild(svgEl('line', { x1: rx(s.min), x2: rx(s.p10), y1: y, y2: y, stroke: color, 'stroke-width': 3, opacity: .35, 'stroke-linecap': 'round' }));
    svg.appendChild(svgEl('line', { x1: rx(s.p10), x2: rx(s.median), y1: y, y2: y, stroke: color, 'stroke-width': 6, opacity: .55, 'stroke-linecap': 'round' }));
    const c = svgEl('circle', { cx: rx(s.median), cy: y, r: 7, fill: color, stroke: 'var(--card-solid)', 'stroke-width': 2 });
    c.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${name}</b><br>${l('median', 'медиана')} ρ ${fix(s.median, 4)}<br>p10 ${fix(s.p10, 4)} · ${l('min', 'мин')} ${fix(s.min, 4)}`, e.clientX, e.clientY));
    c.addEventListener('mouseleave', hideTip);
    svg.appendChild(c);
    text(svg, rx(s.median) + 12, y + 4, fix(s.median, 3), { size: 10.5, fill: color });
  });
  // B: домены против глобального списка
  const bx = 1170, bw = 720, by = 34;
  text(svg, bx, 18, l('domain ranking vs global · median ρ, tick = min', 'ранжирование домена vs сводное · медиана ρ, засечка = мин'), { fill: 'var(--ink-soft)', size: 10.5 });
  const doms = Object.entries(S.domain_vs_global_exact_reap).map(([k, v]: any) => ({ raw: k, rho: v.rho, zero: v.max_zero_sample_experts })).sort((a, b) => b.rho.median - a.rho.median);
  const bh = 24, bxv = (v: number) => bx + Math.max(0, v) * bw;
  doms.forEach((dm, i) => {
    const y = by + i * (bh + 5), key = Object.keys(DOMAIN_COLOR).find(x => x.replace('/', '_') === dm.raw) || dm.raw;
    text(svg, bx - 12, y + 16, labelDomain(key), { anchor: 'end', size: 11, fill: 'var(--muted)' });
    const r = svgEl('rect', { x: bx, y: y + 3, width: Math.max(2, bxv(dm.rho.median) - bx), height: bh - 6, rx: 3, fill: dcol(key), opacity: .85 });
    r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${labelDomain(key)}</b><br>${l('median', 'медиана')} ρ ${fix(dm.rho.median, 3)} · p10 ${fix(dm.rho.p10, 3)} · ${l('min', 'мин')} ${fix(dm.rho.min, 3)}<br>${l('experts unseen in a half', 'экспертов без выборки в половине')}: ${dm.zero}`, e.clientX, e.clientY));
    r.addEventListener('mouseleave', hideTip);
    svg.appendChild(r);
    svg.appendChild(svgEl('line', { x1: bxv(dm.rho.min), x2: bxv(dm.rho.min), y1: y + 1, y2: y + bh - 2, stroke: 'var(--ink)', 'stroke-width': 1.5, opacity: .6 }));
    text(svg, bxv(dm.rho.median) + 6, y + 16, fix(dm.rho.median, 2), { size: 10, fill: 'var(--faint)' });
  });
  for (const g of [0, .5, 1]) text(svg, bxv(g), by + doms.length * (bh + 5) + 10, fix(g, 1), { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' });
  // C: Jaccard по размеру keep-set
  const cx0 = 2090, cw = 700, cy0 = 34, chh = 190;
  text(svg, cx0, 18, l('keep-set overlap between halves by size', 'пересечение keep-set половин по размеру'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, cx0 - 6, cy0, cw + 12, chh);
  const sizes = ['72', '108', '144', '216'], jx = (i: number) => cx0 + 40 + i / 3 * (cw - 80), jy = (v: number) => cy0 + chh - 14 - (v - .8) / .2 * (chh - 40);
  gridY(svg, cx0, cx0 + cw, jy, [.8, .9, 1], v => fix(v, 2), cx0 - 8);
  const J = S.halves.exact_reap_jaccard;
  ['min', 'p10', 'median'].forEach((k, ki) => {
    const vals = sizes.map(s => J[s][k]);
    line(svg, vals, jx, jy, 'var(--accent)', ki === 2 ? 2.4 : 1.2, ki === 2 ? '' : '3 4');
    hoverDots(svg, vals, jx, jy, 'var(--accent)', i => `<b>keep ${sizes[i]} ${l('of', 'из')} 288</b><br>${k} Jaccard ${fix(J[sizes[i]][k], 3)}`, undefined, ki === 2 ? 3.5 : 2.4);
  });
  sizes.forEach((s, i) => text(svg, jx(i), cy0 + chh + 16, `keep ${s}`, { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' }));
  text(svg, cx0, cy0 + chh + 40, l('median · p10 · min across 42 layers, exact REAP halves', 'медиана · p10 · мин по 42 слоям, половины exact REAP'), { size: 9.5, fill: 'var(--ghost)' });
  body.appendChild(el('div', 'small-note', '', l(
    'Chinese, OCR and long-context domains disagree most with the global list, which is exactly why the atlas lets you slice REAP by domain before believing any global prune set. Sample rate for the halves is 2% per layer.',
    'Китайский, OCR и long-context сильнее всего расходятся со сводным списком, поэтому атлас даёт резать REAP по доменам до того, как верить сводному prune-set. Sample rate половин: 2% на слой.')));
  return root;
}

// ───────────────────────── 5. KDA-память ─────────────────────────

function kdaCard(d: any, store: Store): HTMLElement {
  const M = d.memory, layers: number[] = M.kda_layers, N = layers.length, HN = 64;
  const { root, body } = cardShell(2850, l('34 × 64 RECURRENT MEMORIES', '34 × 64 РЕКУРРЕНТНЫХ ПАМЯТИ'), l(
    'Every KDA head learned its own forget rate. Half-life = tokens until the recurrent state halves; β = how strongly a token is written in. Both are means over the balanced corpus.',
    'Каждая KDA-голова выучила свою скорость забывания. Half-life = токенов до уменьшения состояния вдвое; β = насколько сильно токен записывается. Оба значения средние по корпусу.'));
  let mode: 'half' | 'beta' = 'half', sortHeads = false;
  const ctl = el('div', 'no-pan', 'display:flex;gap:14px;align-items:center;flex-wrap:wrap');
  ctl.appendChild(chipRow([['half', l('half-life, log tokens', 'half-life, log токенов')], ['beta', l('β write gate', 'β гейт записи')]], 'half', k => { mode = k as any; draw(); }));
  ctl.appendChild(chipRow([['id', l('head id order', 'по id головы')], ['sort', l('sorted within layer', 'сортировка в слое')]], 'id', k => { sortHeads = k === 'sort'; draw(); }));
  body.appendChild(ctl);
  const cw = 30, ch = 14, padL = 46, padT = 20, H = heat(HN, N, cw, ch, padL, padT);
  const wrap = el('div', '', 'display:flex;gap:22px;align-items:flex-start');
  wrap.appendChild(H.wrap);
  const sw = 2850 - 52 - padL - HN * cw - 22, side = svgEl('svg', { width: sw, height: padT + N * ch + 24, class: 'no-pan' });
  wrap.appendChild(side as any);
  body.appendChild(wrap);
  const legend = el('div', ''); body.appendChild(legend);
  const HL: number[][] = M.kda_heads, BT: number[][] = M.kda_heads_beta;
  const lhl = HL.flat().map(v => Math.log10(Math.max(v, .1))), lo = quantile(lhl, .02), hi = quantile(lhl, .98);
  let order: number[][] = [];
  function draw() {
    const src = mode === 'half' ? HL : BT;
    order = src.map(rowv => { const ids = [...Array(HN).keys()]; return sortHeads ? ids.sort((a, b) => rowv[b] - rowv[a]) : ids; });
    H.draw((c, r) => {
      const h = order[r][c];
      return mode === 'half' ? seq((Math.log10(Math.max(HL[r][h], .1)) - lo) / (hi - lo || 1)) : seq(BT[r][h]);
    }, ctx => {
      ctx.font = '9.5px IBM Plex Mono, monospace'; ctx.fillStyle = cssVar('--faint');
      layers.forEach((L, i) => { if (i % 3 === 0 || i === N - 1) ctx.fillText(`L${L}`, 14, padT + i * ch + 10); });
      ctx.fillStyle = cssVar('--ghost');
      for (let h = 0; h < HN; h += 8) ctx.fillText(sortHeads ? `#${h + 1}` : `h${h}`, padL + h * cw, 13);
    });
    legend.innerHTML = mode === 'half'
      ? rampBar(`${fmt(Math.pow(10, lo), 1)} tok`, `${big(Math.pow(10, hi))} tok`) + `<span class="mono" style="font-size:10px;color:var(--ghost);margin-left:14px">${l('log colour, clipped 2nd–98th pct; hover keeps exact means · click a row opens the layer', 'log-цвет, обрезано 2–98 перцентилем; tooltip хранит точные средние · клик по строке открывает слой')}</span>`
      : rampBar('β 0', 'β 1') + `<span class="mono" style="font-size:10px;color:var(--ghost);margin-left:14px">${l('mean write gate, linear', 'средний гейт записи, линейно')}</span>`;
  }
  H.wrap.addEventListener('mousemove', (e: MouseEvent) => {
    const p = H.pick(e); if (!p) { hideTip(); return; }
    const h = order[p.r][p.c];
    tip(`<b>L${layers[p.r]} · ${l('head', 'голова')} ${h}</b><br>half-life <span class="mono">${HL[p.r][h] >= 1e5 ? big(HL[p.r][h]) : fmt(HL[p.r][h], 2)}</span> ${l('tokens', 'токенов')}<br>β ${fix(BT[p.r][h], 3)}<br>${l('layer mean half-life', 'средняя half-life слоя')} ${fmt(M.kda[p.r].half_life, 1)} · β ${fix(M.kda[p.r].beta_open, 3)}`, e.clientX, e.clientY);
  });
  H.wrap.addEventListener('mouseleave', hideTip);
  H.wrap.addEventListener('click', (e: MouseEvent) => { const p = H.pick(e); if (p) selectLayer(store, layers[p.r]); });
  // справа: гистограмма half-life по всем головам + позиционные бакеты
  const hx = 40, hw = sw - hx - 16, hy = 20, hh = 190;
  text(side, hx, 12, l('all 2,176 heads · half-life distribution', 'все 2176 голов · распределение half-life'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(side, hx - 6, hy, hw + 12, hh);
  const bins = 36, bmin = -1, bmax = 7, counts = new Array(bins).fill(0);
  for (const v of lhl) counts[Math.max(0, Math.min(bins - 1, Math.floor((v - bmin) / (bmax - bmin) * bins)))]++;
  const cmax = Math.max(...counts), bw = hw / bins;
  counts.forEach((c, i) => {
    const t = (bmin + (i + .5) / bins * (bmax - bmin) - lo) / (hi - lo || 1);
    const r = svgEl('rect', { x: hx + i * bw + .5, y: hy + hh - 16 - c / cmax * (hh - 30), width: bw - 1, height: c / cmax * (hh - 30), fill: seq(t), rx: 1 });
    r.addEventListener('mousemove', (e: MouseEvent) => tip(`${big(Math.pow(10, bmin + i / bins * (bmax - bmin)))} … ${big(Math.pow(10, bmin + (i + 1) / bins * (bmax - bmin)))} ${l('tokens', 'токенов')}<br><b>${c}</b> ${l('heads', 'голов')}`, e.clientX, e.clientY));
    r.addEventListener('mouseleave', hideTip);
    side.appendChild(r);
  });
  [['1', 0], ['10', 1], ['100', 2], ['1K', 3], ['10K', 4], ['100K', 5], ['1M', 6]].forEach(([n, v]) => text(side, hx + ((v as number) - bmin) / (bmax - bmin) * hw, hy + hh - 3, n as string, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }));
  // позиционные бакеты
  const py0 = hy + hh + 40, ph = padT + N * ch - py0 - 6;
  text(side, hx, py0 - 8, l('does memory change with prompt position? median over layers', 'меняется ли память с позицией? медиана по слоям'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(side, hx - 6, py0, hw + 12, ph);
  const buckets: string[] = M.position_buckets;
  const hlPos = [0, 1, 2, 3, 4, 5].map(b => median(M.kda_position.half_life.map((r: (number | null)[]) => r[b]).filter((v: any) => v != null).map((v: number) => Math.log10(Math.max(v, .1)))));
  const btPos = [0, 1, 2, 3, 4, 5].map(b => median(M.kda_position.beta_open.map((r: (number | null)[]) => r[b]).filter((v: any) => v != null)));
  const ppx = (i: number) => hx + 30 + i / 5 * (hw - 60);
  const hlo = Math.min(...hlPos) - .15, hhi = Math.max(...hlPos) + .15, blo = Math.min(...btPos) - .03, bhi = Math.max(...btPos) + .03;
  const pyH = (v: number) => py0 + ph - 22 - (v - hlo) / (hhi - hlo) * (ph - 50), pyB = (v: number) => py0 + ph - 22 - (v - blo) / (bhi - blo) * (ph - 50);
  line(side, hlPos, ppx, pyH, TERRA, 2.2); line(side, btPos, ppx, pyB, TEAL, 2.2, '4 3');
  hoverDots(side, hlPos, ppx, pyH, TERRA, i => `<b>${l('positions', 'позиции')} ${buckets[i]}</b><br>${l('median layer half-life', 'медианная half-life слоя')} ${big(Math.pow(10, hlPos[i]))} ${l('tokens', 'токенов')}`);
  hoverDots(side, btPos, ppx, pyB, TEAL, i => `<b>${l('positions', 'позиции')} ${buckets[i]}</b><br>${l('median layer β', 'медианная β слоя')} ${fix(btPos[i], 3)}`);
  buckets.forEach((b, i) => text(side, ppx(i), py0 + ph - 6, b, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }));
  text(side, hx, py0 + 16, `— half-life`, { size: 9.5, fill: TERRA }); text(side, hx + 90, py0 + 16, `- - β`, { size: 9.5, fill: TEAL });
  draw(); redraws.push(draw);
  body.appendChild(el('div', 'small-note', '', l(
    'Means above about 1M tokens come from heads whose learned log-decay is nearly zero; they are real but should be read on the log scale, never averaged arithmetically. Layer 36 head 33 holds the record at 2.7M tokens.',
    'Средние выше примерно 1M токенов дают головы с почти нулевым выученным log-decay; они настоящие, но читаются на log-шкале, а не усредняются арифметически. Рекорд у слоя 36, головы 33: 2.7M токенов.')));
  return root;
}

// ───────────────────────── 6. sparse indexer ─────────────────────────

function indexerCard(d: any, store: Store): HTMLElement {
  const M = d.memory, rows: any[] = M.indexer, N = rows.length, buckets: string[] = M.position_buckets, ranks: string[] = M.rank_buckets;
  const { root, body } = cardShell(2900, l('HOW FAR THE SPARSE INDEXER REACHES', 'КАК ДАЛЕКО ТЯНЕТСЯ SPARSE INDEXER'), l(
    'Eleven sparse-attention layers select 2,048 positions per query with a learned 32-head indexer. The selected distances grow with prompt position, so the indexer really reaches far context instead of a fixed local window.',
    'Одиннадцать sparse-attention слоёв выбирают 2048 позиций на запрос learned-индексером на 32 головы. Выбранные дистанции растут с позицией, то есть indexer реально тянется в дальний контекст, а не в фиксированное окно.'));
  const W = 2850, H = 720, svg = svgEl('svg', { width: W, height: H, class: 'no-pan' });
  body.appendChild(svg as any);
  const col = (i: number) => hot(.15 + .85 * i / (N - 1));
  // A: дистанция по позиции
  const ax = 70, aw = 1300, ay = 30, ah = 380;
  text(svg, ax, 18, l('mean selected distance by query position · tokens, log', 'средняя выбранная дистанция по позиции запроса · токены, log'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, ax - 8, ay, aw + 16, ah);
  const allv = rows.flatMap(r => r.position_distance).filter((v: number) => v > 0);
  const vmax = Math.log10(Math.max(...allv)) + .1, vmin = Math.log10(Math.min(...allv)) - .1;
  const apx = (b: number) => ax + 40 + b / 5 * (aw - 80), apy = (v: number) => ay + ah - 24 - (Math.log10(Math.max(1, v)) - vmin) / (vmax - vmin) * (ah - 44);
  gridY(svg, ax, ax + aw, apy, [10, 100, 1000, 10000], v => big(v), ax - 10);
  rows.forEach((r, i) => {
    line(svg, r.position_distance, apx, apy, col(i), i === N - 1 ? 3 : 1.6);
    hoverDots(svg, r.position_distance, apx, apy, col(i), b => `<b>L${r.layer} · ${l('positions', 'позиции')} ${buckets[b]}</b><br>${fmt(r.position_distance[b], 0)} ${l('tokens back on average', 'токенов назад в среднем')}<br>${l('within 1,024', 'в пределах 1024')}: ${pct(r.position_within_1024[b] ?? 0, 0)}`, () => selectLayer(store, r.layer), i === N - 1 ? 3.6 : 2.6);
    text(svg, apx(5) + 8, apy(r.position_distance[5]) + 3, `L${r.layer}`, { size: 9, fill: col(i) });
  });
  buckets.forEach((b, i) => text(svg, apx(i), ay + ah + 16, b, { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' }));
  // B: по рангу
  const bx = 1470, bw = 1300;
  text(svg, bx, 18, l('mean selected distance by returned rank · higher ranks are more local', 'средняя дистанция по рангу выдачи · верхние ранги локальнее'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, bx - 8, ay, bw + 16, ah);
  const rv = rows.flatMap(r => r.rank_distance).filter((v: number) => v != null && v > 0), rmin = Math.min(...rv) * .85, rmax = Math.max(...rv) * 1.1;
  const bpx = (k: number) => bx + 60 + k / 3 * (bw - 120), bpy = (v: number) => ay + ah - 24 - (v - rmin) / (rmax - rmin) * (ah - 44);
  gridY(svg, bx, bx + bw, bpy, [rmin, (rmin + rmax) / 2, rmax].map(v => Math.round(v / 100) * 100), v => fmt(v, 0), bx - 10);
  rows.forEach((r, i) => {
    line(svg, r.rank_distance, bpx, bpy, col(i), i === N - 1 ? 3 : 1.6);
    hoverDots(svg, r.rank_distance, bpx, bpy, col(i), k => `<b>L${r.layer} · ${l('rank', 'ранг')} ${ranks[k]}</b><br>${fmt(r.rank_distance[k], 0)} ${l('tokens back on average', 'токенов назад в среднем')}`, () => selectLayer(store, r.layer), i === N - 1 ? 3.6 : 2.6);
  });
  ranks.forEach((b, i) => text(svg, bpx(i), ay + ah + 16, `${l('rank', 'ранг')} ${b}`, { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' }));
  // C: локальность стеком
  const cy = ay + ah + 50, chh = 170, cxx = 70, cwid = 1300;
  text(svg, cxx, cy - 8, l('where selections land · share within 128 / 1,024 / 8,192 tokens / beyond', 'куда падает выбор · доля в пределах 128 / 1024 / 8192 токенов / дальше'), { fill: 'var(--ink-soft)', size: 10.5 });
  const bwid = (cwid - 40) / N, stackCol = [TERRA, AMBER, TEAL, BLUE];
  rows.forEach((r, i) => {
    const w = r.within as number[], parts = [w[0], w[1] - w[0], w[2] - w[1], 1 - w[2]];
    let y = cy + chh;
    parts.forEach((p, k) => {
      const hgt = p * (chh - 20);
      const rect = svgEl('rect', { x: cxx + 20 + i * bwid + 4, y: y - hgt, width: bwid - 8, height: hgt, fill: stackCol[k], rx: 1.5, opacity: .85 });
      rect.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${r.layer}</b><br>${['≤128', '129–1,024', '1,025–8,192', '>8,192'][k]} ${l('tokens back', 'токенов назад')}: <span class="mono">${pct(p, 1)}</span><br>${l('unique positions per query', 'уникальных позиций на запрос')} ${pct(r.unique_fraction, 1)} · ${fmt(r.selected_per_query, 0)} ${l('selected', 'выбрано')}`, e.clientX, e.clientY));
      rect.addEventListener('mouseleave', hideTip);
      rect.addEventListener('click', () => selectLayer(store, r.layer));
      svg.appendChild(rect);
      y -= hgt;
    });
    text(svg, cxx + 20 + i * bwid + bwid / 2, cy + chh + 14, `L${r.layer}`, { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' });
  });
  [['≤128', TERRA], ['≤1,024', AMBER], ['≤8,192', TEAL], [l('beyond', 'дальше'), BLUE]].forEach(([n, c], i) => { svg.appendChild(svgEl('rect', { x: cxx + 20 + i * 110, y: cy + chh + 26, width: 10, height: 10, rx: 2, fill: c })); text(svg, cxx + 36 + i * 110, cy + chh + 35, n, { size: 9.5 }); });
  // D: within 1024 по позиции × слою
  const dx = 1470, dw = 1300, cellW = (dw - 60) / 6, cellH = 24;
  text(svg, dx, cy - 8, l('share of selections within 1,024 tokens, by layer × position', 'доля выбора в пределах 1024 токенов, слой × позиция'), { fill: 'var(--ink-soft)', size: 10.5 });
  rows.forEach((r, i) => {
    text(svg, dx + 32, cy + i * (cellH + 1) + 16, `L${r.layer}`, { anchor: 'end', size: 9.5 });
    (r.position_within_1024 as (number | null)[]).forEach((v, b) => {
      const rect = svgEl('rect', { x: dx + 40 + b * cellW, y: cy + i * (cellH + 1), width: cellW - 2, height: cellH, rx: 2, fill: v == null ? cssVar('--hatch-a') : seq(v) });
      rect.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${r.layer} · ${buckets[b]}</b><br>${v == null ? l('no observation', 'нет наблюдений') : pct(v, 1) + ' ' + l('within 1,024 tokens', 'в пределах 1024 токенов')}`, e.clientX, e.clientY));
      rect.addEventListener('mouseleave', hideTip);
      svg.appendChild(rect);
      if (v != null) text(svg, dx + 40 + b * cellW + cellW / 2, cy + i * (cellH + 1) + 16, pct(v, 0), { anchor: 'middle', size: 9.5, fill: seqInk(v) });
    });
  });
  buckets.forEach((b, i) => text(svg, dx + 40 + i * cellW + cellW / 2, cy + N * (cellH + 1) + 14, b, { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' }));
  body.appendChild(el('div', 'small-note', '', l(
    'The fused sparse-MLA kernel exposes the selected token ids but not attention probabilities, so this describes what was retrieved, not how much mass each retrieved token received. Queries at 8K+ positions were captured within the 32K context of the capture config.',
    'Fused sparse-MLA кернел отдаёт id выбранных токенов, но не вероятности внимания, поэтому здесь видно, что было извлечено, а не сколько массы получил каждый токен. Запросы на позициях 8K+ сняты в пределах 32K контекста capture-конфига.')));
  return root;
}

// ───────────────────────── 7. поток сигнала ─────────────────────────

function flowCard(d: any, store: Store): HTMLElement {
  const F = d.flow, V = d.vision_tower, N = F.layers.length;
  const { root, body } = cardShell(2850, l('SIGNAL FLOW THROUGH 45 LAYERS', 'ПОТОК СИГНАЛА ЧЕРЕЗ 45 СЛОЁВ'), l(
    'How much each layer changes the residual stream (Δ RMS) and how much it rotates it (input/output cosine). Bars are coloured by mixer type; pick a domain to see whose text a layer works hardest on.',
    'Насколько каждый слой меняет residual stream (Δ RMS) и насколько его поворачивает (cosine вход/выход). Бары окрашены по типу микшера; выберите домен, чтобы увидеть, над чьим текстом слой работает сильнее.'));
  let domain = 'all';
  const ctl = el('div', 'no-pan', 'display:flex;gap:14px;align-items:center;flex-wrap:wrap');
  ctl.appendChild(domainSelect(d.routing.domains, v => { domain = v; draw(); }));
  ctl.appendChild(legendRow([[AMBER, l('sparse MLA layer', 'слой sparse MLA')], [TEAL, l('KDA layer', 'слой KDA')], [PLUM, 'io cosine']]));
  body.appendChild(ctl);
  const W = 2800, H = 560, svg = svgEl('svg', { width: W, height: H, class: 'no-pan' });
  body.appendChild(svg as any);
  const x0 = 60, cw = 1740, px = (i: number) => x0 + 12 + i / (N - 1) * (cw - 24);
  function draw() {
    svg.innerHTML = '';
    // A: Δ RMS + io cosine
    const ay = 26, ah = 300;
    text(svg, x0, 16, `Δ RMS ${l('per layer', 'на слой')}${domain !== 'all' ? ' · ' + labelDomain(domain) : ''} · ${l('bars', 'бары')}   |   io cosine · ${l('line', 'линия')}`, { fill: 'var(--ink-soft)', size: 10.5 });
    frame(svg, x0 - 8, ay, cw + 16, ah);
    const dv: number[] = domain === 'all' ? F.delta_rms : F.domains[domain];
    const dmax = Math.max(...dv) * 1.08, dpy = (v: number) => ay + ah - 20 - v / dmax * (ah - 60);
    const bw = (cw - 24) / (N - 1) * .72;
    dv.forEach((v, i) => {
      const r = svgEl('rect', { x: px(i) - bw / 2, y: dpy(v), width: bw, height: ay + ah - 20 - dpy(v), fill: F.kind[i] === 'mla' ? AMBER : TEAL, rx: 2, opacity: .9 });
      r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${i}</b> · ${F.kind[i] === 'mla' ? 'sparse MLA' : 'KDA'}${i < 3 ? ' · dense FFN' : ' · MoE'}<br>Δ RMS <span class="mono">${fix(v, 3)}</span> (${l('all domains', 'все домены')} ${fix(F.delta_rms[i], 3)})<br>${l('mixer Δ', 'Δ микшера')} ${fix(F.mixer_delta[i], 3)} · FFN Δ ${fix(F.ffn_delta[i], 3)}<br>io cosine ${fix(F.io_cosine[i], 3)}<br>${l('output RMS', 'RMS выхода')} ${fix(F.output_rms[i], 3)}`, e.clientX, e.clientY));
      r.addEventListener('mouseleave', hideTip);
      r.addEventListener('click', () => selectLayer(store, i));
      svg.appendChild(r);
    });
    gridY(svg, x0, x0 + cw, dpy, [0, dmax / 2, dmax * .95].map(v => +v.toFixed(2)), v => fix(v, 2), x0 - 10);
    const cpy = (v: number) => ay + ah - 20 - (v + 1) / 2 * (ah - 60);
    svg.appendChild(svgEl('line', { x1: x0, x2: x0 + cw, y1: cpy(0), y2: cpy(0), stroke: PLUM, 'stroke-dasharray': '2 5', opacity: .6 }));
    line(svg, F.io_cosine, px, cpy, PLUM, 2);
    hoverDots(svg, F.io_cosine, px, cpy, PLUM, i => `<b>L${i}</b><br>io cosine <span class="mono">${fix(F.io_cosine[i], 3)}</span><br>${l('mixer io', 'io микшера')} ${fix(F.mixer_io[i], 3)} · FFN io ${fix(F.ffn_io[i], 3)}`, i => selectLayer(store, i), 2.8);
    text(svg, x0 + cw + 6, cpy(1) + 3, '+1', { size: 9, fill: PLUM }); text(svg, x0 + cw + 6, cpy(0) + 3, '0', { size: 9, fill: PLUM }); text(svg, x0 + cw + 6, cpy(-1) + 3, '−1', { size: 9, fill: PLUM });
    for (let i = 0; i < N; i += 4) text(svg, px(i), ay + ah + 14, `L${i}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' });
    // B: микшер vs FFN
    const by = ay + ah + 50, bh = 160;
    text(svg, x0, by - 8, l('who does the work: mixer Δ vs FFN Δ', 'кто работает: Δ микшера vs Δ FFN'), { fill: 'var(--ink-soft)', size: 10.5 });
    frame(svg, x0 - 8, by, cw + 16, bh);
    const mmax = Math.max(...F.mixer_delta, ...F.ffn_delta) * 1.1, mpy = (v: number) => by + bh - 14 - v / mmax * (bh - 30);
    line(svg, F.mixer_delta, px, mpy, BLUE, 2); line(svg, F.ffn_delta, px, mpy, ORANGE, 2);
    hoverDots(svg, F.mixer_delta, px, mpy, BLUE, i => `<b>L${i}</b> ${F.kind[i] === 'mla' ? 'MLA' : 'KDA'} Δ ${fix(F.mixer_delta[i], 3)}`, i => selectLayer(store, i), 2.4);
    hoverDots(svg, F.ffn_delta, px, mpy, ORANGE, i => `<b>L${i}</b> ${i < 3 ? 'dense FFN' : 'MoE'} Δ ${fix(F.ffn_delta[i], 3)}`, i => selectLayer(store, i), 2.4);
    gridY(svg, x0, x0 + cw, mpy, [0, mmax / 2].map(v => +v.toFixed(2)), v => fix(v, 2), x0 - 10);
    text(svg, x0 + 8, by + 14, `— ${l('mixer', 'микшер')}`, { size: 9.5, fill: BLUE }); text(svg, x0 + 80, by + 14, `— FFN`, { size: 9.5, fill: ORANGE });
    // C: башня зрения
    const vx = x0 + cw + 70, vw = W - vx - 20, vN = V.blocks.length, vpx = (i: number) => vx + 12 + i / (vN - 1) * (vw - 24);
    text(svg, vx, 16, l('vision tower · 24 blocks · Δ RMS and io cosine', 'башня зрения · 24 блока · Δ RMS и io cosine'), { fill: 'var(--ink-soft)', size: 10.5 });
    frame(svg, vx - 8, ay, vw + 16, ah);
    const vmax = Math.max(...V.delta_rms) * 1.1, vpy = (v: number) => ay + ah - 20 - v / vmax * (ah - 60), vbw = (vw - 24) / (vN - 1) * .7;
    (V.delta_rms as number[]).forEach((v, i) => {
      const r = svgEl('rect', { x: vpx(i) - vbw / 2, y: vpy(v), width: vbw, height: ay + ah - 20 - vpy(v), fill: kindOf('vision').solid, rx: 2, opacity: .9 });
      r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${l('vision block', 'vision-блок')} ${i}</b><br>Δ RMS ${fix(v, 3)} · io cosine ${fix(V.io_cosine[i], 3)}<br>${l('input SQNR', 'SQNR входа')}: INT8 ${fix(V.int8[i], 1)} · FP8 ${fix(V.fp8[i], 1)} · NVFP4 ${fix(V.nvfp4_ideal[i], 1)} dB`, e.clientX, e.clientY));
      r.addEventListener('mouseleave', hideTip);
      svg.appendChild(r);
    });
    const vcpy = (v: number) => ay + ah - 20 - (v + 1) / 2 * (ah - 60);
    line(svg, V.io_cosine, vpx, vcpy, PLUM, 2);
    hoverDots(svg, V.io_cosine, vpx, vcpy, PLUM, i => `<b>${l('vision block', 'vision-блок')} ${i}</b><br>io cosine ${fix(V.io_cosine[i], 3)}`, undefined, 2.6);
    gridY(svg, vx, vx + vw, vpy, [0, vmax / 2].map(v => +v.toFixed(2)), v => fix(v, 2), vx - 10);
    for (let i = 0; i < vN; i += 4) text(svg, vpx(i), ay + ah + 14, `v${i}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' });
    frame(svg, vx - 8, by, vw + 16, bh);
    text(svg, vx, by + 18, 'patch merger', { fill: 'var(--ink-soft)', size: 10.5 });
    text(svg, vx, by + 40, `${l('input RMS', 'RMS входа')} ${fix(V.merger.input_rms, 3)} → ${l('output RMS', 'RMS выхода')} ${fix(V.merger.output_rms, 4)}`, { size: 10.5, fill: 'var(--muted)' });
    text(svg, vx, by + 58, `io cosine ${fix(V.merger.io_cosine, 4)}`, { size: 10.5, fill: 'var(--muted)' });
    text(svg, vx, by + 78, l('the merger rewrites the vision stream almost completely', 'merger почти полностью переписывает vision-поток'), { size: 10, fill: 'var(--faint)' });
    text(svg, vx, by + 92, l('before it enters the 4,096-wide language stream', 'перед входом в language-поток шириной 4096'), { size: 10, fill: 'var(--faint)' });
  }
  draw();
  body.appendChild(el('div', 'small-note', '', l(
    'Layer 3, the first MoE block, moves the stream more than any other layer (Δ 1.27, cosine 0.53). Negative cosines early in the stack are real: those KDA layers partly flip what the embedding wrote.',
    'Слой 3, первый MoE-блок, сдвигает поток сильнее любого другого (Δ 1.27, cosine 0.53). Отрицательные cosine в начале стека настоящие: те KDA-слои частично переворачивают то, что записал embedding.')));
  return root;
}

// ───────────────────────── 8. квантуемость активаций ─────────────────────────

function actqCard(d: any, store: Store): HTMLElement {
  const A = d.actq, V = d.vision_tower;
  const { root, body } = cardShell(2900, l('ACTIVATIONS UNDER INT8 · FP8 · NVFP4', 'АКТИВАЦИИ ПОД INT8 · FP8 · NVFP4'), l(
    'Hypothetical SQNR of the live activations at four boundaries, plus the real one: the MoE FC1 input quantized with the checkpoint’s own deployed scale. Higher is cleaner.',
    'Гипотетический SQNR живых активаций на четырёх границах плюс настоящий: вход MoE FC1, квантованный собственным deployed scale чекпоинта. Выше значит чище.'));
  body.appendChild(legendRow([[TEAL, 'INT8 ' + l('dynamic per-token', 'динамический per-token')], [AMBER, 'FP8 e4m3'], [PLUM, 'NVFP4 ' + l('ideal scale', 'идеальный scale')], ['var(--accent)', 'NVFP4 ' + l('deployed scale (measured)', 'deployed scale (замер)')]]));
  const W = 2850, PW = 1330, PH = 210, svg = svgEl('svg', { width: W, height: 2 * PH + 2 * 44 + 200, class: 'no-pan' });
  body.appendChild(svg as any);
  const sites: [string, string, string][] = [
    ['kda_input', l('KDA input · 34 layers', 'вход KDA · 34 слоя'), l('before the recurrent mixer', 'перед рекуррентным микшером')],
    ['mla_input', l('sparse MLA input · 11 layers', 'вход sparse MLA · 11 слоёв'), l('before the indexer and latent attention', 'перед indexer и latent attention')],
    ['dense_mlp_input', l('FFN input · 45 layers', 'вход FFN · 45 слоёв'), l('dense for 0–2, MoE for 3–44', 'dense для 0–2, MoE для 3–44')],
    ['moe_fc1_input', l('MoE FC1 input · 42 layers', 'вход MoE FC1 · 42 слоя'), l('the one boundary with a deployed NVFP4 scale', 'единственная граница с deployed NVFP4 scale')],
  ];
  sites.forEach(([key, name, sub], si) => {
    const s = A[key], N = s.layers.length, x0 = 70 + (si % 2) * (PW + 120), y0 = 10 + Math.floor(si / 2) * (PH + 44);
    text(svg, x0, y0 + 10, name, { fill: 'var(--ink-soft)', size: 11 }); text(svg, x0 + name.length * 7 + 14, y0 + 10, sub, { size: 9.5, fill: 'var(--ghost)' });
    frame(svg, x0 - 8, y0 + 18, PW + 16, PH - 18);
    const px = (i: number) => x0 + 10 + i / Math.max(1, N - 1) * (PW - 20), py = (v: number) => y0 + PH - 22 - (v - 10) / 40 * (PH - 60);
    gridY(svg, x0, x0 + PW, py, [20, 30, 40], v => `${v} dB`, x0 - 10);
    const series: [string, string, string][] = [['int8', TEAL, 'INT8'], ['fp8', AMBER, 'FP8'], ['nvfp4_ideal', PLUM, 'NVFP4 ideal']];
    if (s.nvfp4_deployed) series.push(['nvfp4_deployed', 'var(--accent)', 'NVFP4 deployed']);
    series.forEach(([k, c, n]) => {
      line(svg, s[k], px, py, c, k === 'nvfp4_deployed' ? 2.6 : 1.8, k === 'nvfp4_ideal' && s.nvfp4_deployed ? '5 4' : '');
      hoverDots(svg, s[k], px, py, c, i => `<b>L${s.layers[i]}</b> · ${name}<br>${n} <span class="mono">${fix(s[k][i], 2)} dB</span><br>INT8 ${fix(s.int8[i], 1)} · FP8 ${fix(s.fp8[i], 1)} · NVFP4 ${fix(s.nvfp4_ideal[i], 1)}${s.nvfp4_deployed ? ` · deployed ${fix(s.nvfp4_deployed[i], 2)}` : ''}<br>${l('hot channels ×5', 'горячих каналов ×5')}: ${pct(s.outlier_ratio[i], 2)} · max |x| ${fmt(s.max_abs[i], 1)}`, i => selectLayer(store, s.layers[i]), 2.2);
    });
    for (let i = 0; i < N; i += Math.max(1, Math.round(N / 8))) text(svg, px(i), y0 + PH + 8, `L${s.layers[i]}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' });
  });
  // выбросы по каналам + башня зрения
  const y2 = 2 * (PH + 44) + 8;
  const ox = 70, ow = PW;
  text(svg, ox, y2 + 10, l('hot channels: share of channels above 5× the RMS, FC1 input', 'горячие каналы: доля каналов выше 5× RMS, вход FC1'), { fill: 'var(--ink-soft)', size: 11 });
  frame(svg, ox - 8, y2 + 18, ow + 16, 150);
  const s4 = A.moe_fc1_input, N4 = s4.layers.length, opx = (i: number) => ox + 10 + i / (N4 - 1) * (ow - 20), omax = Math.max(...s4.outlier_ratio) * 1.15, opy = (v: number) => y2 + 158 - v / omax * 120;
  (s4.outlier_ratio as number[]).forEach((v, i) => {
    const r = svgEl('rect', { x: opx(i) - 12, y: opy(v), width: 24, height: y2 + 158 - opy(v), fill: seq(.15 + .85 * v / omax), rx: 2 });
    r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${s4.layers[i]}</b><br>${pct(v, 2)} ${l('of channels are hot', 'каналов горячие')}<br>RMS ${fix(s4.rms[i], 3)} · max |x| ${fmt(s4.max_abs[i], 1)}`, e.clientX, e.clientY));
    r.addEventListener('mouseleave', hideTip);
    svg.appendChild(r);
  });
  gridY(svg, ox, ox + ow, opy, [0, omax / 2].map(v => +v.toFixed(3)), v => pct(v, 1), ox - 10);
  for (let i = 0; i < N4; i += 5) text(svg, opx(i), y2 + 182, `L${s4.layers[i]}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' });
  const vx = 70 + PW + 120, vN = V.blocks.length, vpx = (i: number) => vx + 10 + i / (vN - 1) * (PW - 20), vpy = (v: number) => y2 + 158 - (v - 10) / 40 * 120;
  text(svg, vx, y2 + 10, l('vision tower block input · 24 blocks', 'вход vision-блока · 24 блока'), { fill: 'var(--ink-soft)', size: 11 });
  frame(svg, vx - 8, y2 + 18, PW + 16, 150);
  gridY(svg, vx, vx + PW, vpy, [20, 30, 40], v => `${v} dB`, vx - 10);
  ([['int8', TEAL], ['fp8', AMBER], ['nvfp4_ideal', PLUM]] as [string, string][]).forEach(([k, c]) => {
    line(svg, V[k], vpx, vpy, c, 1.8);
    hoverDots(svg, V[k], vpx, vpy, c, i => `<b>${l('vision block', 'vision-блок')} ${i}</b><br>INT8 ${fix(V.int8[i], 1)} · FP8 ${fix(V.fp8[i], 1)} · NVFP4 ${fix(V.nvfp4_ideal[i], 1)} dB<br>${l('hot channels', 'горячих каналов')} ${pct(V.outlier_ratio[i], 2)}`, undefined, 2.2);
  });
  for (let i = 0; i < vN; i += 4) text(svg, vpx(i), y2 + 182, `v${i}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' });
  const dep = A.moe_fc1_input.nvfp4_deployed as number[], ideal = A.moe_fc1_input.nvfp4_ideal as number[];
  body.appendChild(el('div', 'small-note', '', l(
    `The deployed FC1 scale tracks the ideal per-tensor scale within ${fix(Math.max(...dep.map((v, i) => Math.abs(v - ideal[i]))), 2)} dB on every layer: the quantizer left nothing on the table at that boundary. The FC2 side is a different story, see the NVFP4 card. All SQNR values except the deployed FC1 line are what-if computations on captured activations, not a BF16 comparison.`,
    `Deployed scale FC1 повторяет идеальный per-tensor scale в пределах ${fix(Math.max(...dep.map((v, i) => Math.abs(v - ideal[i]))), 2)} dB на каждом слое: квантайзер на этой границе ничего не оставил. Сторона FC2 отдельная история, см. карточку NVFP4. Все SQNR кроме deployed FC1 это what-if расчёты на снятых активациях, а не сравнение с BF16.`)));
  return root;
}

// ───────────────────────── 9. shared vs routed ─────────────────────────

function sharedCard(d: any, store: Store): HTMLElement {
  const C = d.contributions, rows: any[] = C.layers, N = rows.length, PQ: any[] = C.pair_quantiles, PX: any[] = C.pairs_extreme;
  const { root, body } = cardShell(2850, l('THE ALWAYS-ON EXPERT VS THE ROUTED EIGHT', 'ВСЕГДА-ВКЛЮЧЁННЫЙ ЭКСПЕРТ ПРОТИВ ВОСЬМИ ROUTED'), l(
    'Every MoE block adds one shared expert to the routed sum. Early on the routed branch carries the energy; by layer 44 the shared expert owns 86% of it. Expert outputs are close to orthogonal to each other throughout.',
    'Каждый MoE-блок добавляет к routed-сумме один shared expert. В начале энергию несёт routed-ветка; к слою 44 shared expert владеет 86%. Выходы экспертов почти ортогональны друг другу по всей глубине.'));
  const W = 2800, H = 520, svg = svgEl('svg', { width: W, height: H, class: 'no-pan' });
  body.appendChild(svg as any);
  const x0 = 60, cw = 1740, px = (i: number) => x0 + 12 + i / (N - 1) * (cw - 24);
  const ay = 26, ah = 250, apy = (v: number) => ay + ah - 20 - v * (ah - 40);
  text(svg, x0, 16, l('energy split of the MoE output · shared (terracotta) vs routed (wave), plus shared/routed cosine', 'разделение энергии выхода MoE · shared (терракота) vs routed (волна), плюс cosine shared/routed'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, x0 - 8, ay, cw + 16, ah);
  let area = `M${px(0)},${apy(0)}`; rows.forEach((r, i) => area += ` L${px(i)},${apy(r.shared_energy)}`); area += ` L${px(N - 1)},${apy(0)} Z`;
  svg.appendChild(svgEl('path', { d: area, fill: TERRA, opacity: .55 }));
  let area2 = `M${px(0)},${apy(1)}`; rows.forEach((r, i) => area2 += ` L${px(i)},${apy(r.shared_energy)}`); area2 += ` L${px(N - 1)},${apy(1)} Z`;
  svg.appendChild(svgEl('path', { d: area2, fill: TEAL, opacity: .45 }));
  gridY(svg, x0, x0 + cw, apy, [0, .25, .5, .75, 1], v => pct(v, 0), x0 - 10);
  line(svg, rows.map(r => r.shared_routed_cosine), px, (v: number) => apy((v + .2) / .6), PLUM, 2.2);
  hoverDots(svg, rows.map(r => r.shared_energy), px, apy, 'var(--ink)', i => `<b>L${rows[i].layer}</b><br>${l('shared energy', 'энергия shared')} <span class="mono">${pct(rows[i].shared_energy, 1)}</span> · routed ${pct(1 - rows[i].shared_energy, 1)}<br>${l('shared/routed norm ratio', 'отношение норм shared/routed')} ${fix(rows[i].shared_to_routed, 2)}<br>${l('shared·routed cosine', 'cosine shared·routed')} ${fix(rows[i].shared_routed_cosine, 3)}<br>constructive gain ${fix(rows[i].constructive_gain, 3)}`, i => selectLayer(store, rows[i].layer), 3.2);
  text(svg, x0 + cw + 6, apy((.4 + .2) / .6) + 3, 'cos .4', { size: 9, fill: PLUM }); text(svg, x0 + cw + 6, apy(.2 / .6) + 3, 'cos 0', { size: 9, fill: PLUM });
  text(svg, x0 + 14, ay + 18, 'shared', { fill: '#fff3e8', size: 11, weight: '600' }); text(svg, x0 + 14, ay + ah - 28, 'routed', { fill: '#eef7f7', size: 11, weight: '600' });
  // pair cosine band
  const by = ay + ah + 48, bh = 160;
  text(svg, x0, by - 8, l('pairwise cosine of selected expert outputs · p05 … p50 … p95 over sampled pairs', 'попарный cosine выходов выбранных экспертов · p05 … p50 … p95 по sampled-парам'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, x0 - 8, by, cw + 16, bh);
  const plo = Math.min(...PQ.map(p => p.p05)) - .02, phi = Math.max(...PQ.map(p => p.p95)) + .02, bpy = (v: number) => by + bh - 16 - (v - plo) / (phi - plo) * (bh - 32);
  let band = `M${px(0)},${bpy(PQ[0].p95)}`; PQ.forEach((p, i) => band += ` L${px(i)},${bpy(p.p95)}`); for (let i = N - 1; i >= 0; i--) band += ` L${px(i)},${bpy(PQ[i].p05)}`; band += ' Z';
  svg.appendChild(svgEl('path', { d: band, fill: PLUM, opacity: .22 }));
  svg.appendChild(svgEl('line', { x1: x0, x2: x0 + cw, y1: bpy(0), y2: bpy(0), stroke: 'var(--line-strong)', 'stroke-dasharray': '3 4' }));
  line(svg, PQ.map(p => p.p50), px, bpy, PLUM, 2.2);
  hoverDots(svg, PQ.map(p => p.p50), px, bpy, PLUM, i => `<b>L${PQ[i].layer}</b><br>p50 ${fix(PQ[i].p50, 3)} · p05 ${fix(PQ[i].p05, 3)} · p95 ${fix(PQ[i].p95, 3)}<br>${l('most aligned pair', 'самая согласованная пара')}: ${PX[i].aligned.length ? `E${PX[i].aligned[0][0]}+E${PX[i].aligned[0][1]} cos ${fix(PX[i].aligned[0][3], 3)} (n=${PX[i].aligned[0][2]})` : '–'}`, i => selectLayer(store, PQ[i].layer), 3);
  gridY(svg, x0, x0 + cw, bpy, [plo + .02, 0, phi - .02].map(v => +v.toFixed(2)), v => fix(v, 2), x0 - 10);
  rows.forEach((r, i) => { if (i % 3 === 0 || i === N - 1) text(svg, px(i), H - 6, `L${r.layer}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }); });
  // справа: сводка
  const sx = x0 + cw + 60;
  const first = rows[0], last = rows[N - 1], minE = rows.reduce((a, b) => a.shared_energy < b.shared_energy ? a : b);
  const stats = [
    [pct(minE.shared_energy, 1), `${l('minimum shared energy', 'минимум энергии shared')} · L${minE.layer}`],
    [pct(last.shared_energy, 1), `${l('shared energy at', 'энергия shared на')} L${last.layer}`],
    [fix(last.shared_to_routed, 2), `${l('shared/routed norm ratio at', 'отношение норм на')} L${last.layer}`],
    [fix(median(PQ.map(p => p.p50)), 3), l('median pair cosine, all layers', 'медианный cosine пар, все слои')],
    [`${fix(first.shared_routed_cosine, 3)} → ${fix(last.shared_routed_cosine, 3)}`, l('shared·routed cosine, L3 → L44', 'cosine shared·routed, L3 → L44')],
  ];
  stats.forEach(([v, n], i) => { text(svg, sx, 40 + i * 92, v, { fill: 'var(--ink)', size: 26 }); text(svg, sx, 62 + i * 92, n, { size: 11, fill: 'var(--muted)', mono: false }); });
  body.appendChild(el('div', 'small-note', '', l(
    'Component norms are observed after the runtime’s routed scaling; the diagnostic copies are all-reduced separately so the split reflects the full model, not one tensor-parallel shard. Pair cosines come from unit-weight replays of each selected expert through the deployed NVFP4 kernel, minimum 5 samples per pair.',
    'Нормы компонент сняты после routed-масштабирования рантайма; диагностические копии all-reduce отдельно, так что разделение отражает всю модель, а не один TP-шард. Cosine пар из replay каждого выбранного эксперта с единичным весом через deployed NVFP4-кернел, минимум 5 сэмплов на пару.')));
  return root;
}

// ───────────────────────── 10. NVFP4 как он развёрнут ─────────────────────────

function nvfp4Card(d: any, store: Store): HTMLElement {
  const Q = d.quantization, SH = Q.scale_hist, GS = Q.global_scales, FC: any[] = Q.fc2, inv = Q.inventory;
  const { root, body } = cardShell(2900, l('NVFP4, AS DEPLOYED', 'NVFP4 КАК ОН РАЗВЁРНУТ'), l(
    `${big(SH.value_count)} FP8 block scales read straight from the packed expert weights, the scalar global scales, and the one activation error that could be measured with the checkpoint’s own scale: quantize/dequantize at the FC2 input.`,
    `${big(SH.value_count)} FP8 block scales, прочитанных прямо из упакованных весов экспертов, скалярные global scales и единственная ошибка активаций, которую можно было измерить собственным scale чекпоинта: quantize/dequantize на входе FC2.`));
  const modeIdx = SH.all.reduce((best: number, v: number, i: number) => v > SH.all[best] ? i : best, 0);
  body.appendChild(el('div', '', 'display:grid;grid-template-columns:repeat(6,1fr);gap:10px', [
    tile(fix(median(FC.map(r => r.sqnr_db)), 2) + ' dB', l('median FC2-input SQNR', 'медианный SQNR входа FC2'), `${fix(Math.min(...FC.map(r => r.sqnr_db)), 1)} … ${fix(Math.max(...FC.map(r => r.sqnr_db)), 1)} dB`, true),
    tile(pct(median(FC.map(r => r.relative_l2)), 2), l('median relative L2 error', 'медианная относительная L2'), l('post-SwiGLU tensor vs its QDQ', 'тензор после SwiGLU vs его QDQ')),
    tile(pct(median(FC.map(r => r.qdq_zero_fraction)), 1), l('values mapped to exact 0', 'значений стали ровно 0'), l('inputs were essentially nonzero', 'входы были практически ненулевыми')),
    tile(pct(median(FC.map(r => r.exact_fraction)), 2), l('values survive bit-exact', 'значений выжили bit-exact'), l('after QDQ', 'после QDQ')),
    tile(String(SH.codebook[modeIdx]), l('modal block-scale value', 'модальное значение block scale'), `${pct(SH.all[modeIdx] / SH.value_count, 1)} ${l('of all scales', 'всех scales')} · max 448`),
    tile(fmt(GS.input.p50, 0), l('median input_global_scale', 'медианный input_global_scale'), `${fmt(GS.input.min, 0)} … ${fmt(GS.input.max, 0)} · 36,288 ${l('scalars', 'скаляров')}`),
  ].join('')));
  const W = 2850, H = 880, svg = svgEl('svg', { width: W, height: H, class: 'no-pan' });
  body.appendChild(svg as any);
  // A: гистограмма кодов
  const ax = 70, aw = 1300, ay = 30, ah = 300;
  text(svg, ax, 18, l('every block-scale value as an FP8 code · count, log · gate (amber), up (plum), down (wave)', 'каждый block scale как FP8-код · счёт, log · gate (янтарь), up (слива), down (волна)'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, ax - 8, ay, aw + 16, ah);
  const idx = SH.all.map((v: number, i: number) => v ? i : -1).filter((i: number) => i >= 0), i0 = idx[0] - 1, i1 = idx[idx.length - 1] + 1;
  const hpx = (i: number) => ax + 10 + (i - i0) / (i1 - i0) * (aw - 20), cmax = Math.log10(Math.max(...SH.all)), hpy = (c: number) => ay + ah - 20 - (c > 0 ? Math.log10(c) / cmax : 0) * (ah - 44);
  gridY(svg, ax, ax + aw, hpy, [1e3, 1e6, 1e9], v => big(v), ax - 10);
  ([['gate_proj', AMBER], ['up_proj', PLUM], ['down_proj', TEAL]] as [string, string][]).forEach(([p, c]) => {
    const vals: number[] = SH[p];
    let dpath = ''; for (let i = i0; i <= i1; i++) { const y = hpy(vals[i] || 0); dpath += `${dpath ? ' L' : 'M'}${hpx(i)},${y} L${hpx(i + 1)},${y}`; }
    svg.appendChild(svgEl('path', { d: dpath, fill: 'none', stroke: c, 'stroke-width': 1.8, opacity: .9 }));
  });
  for (let i = i0; i <= i1; i++) {
    const r = svgEl('rect', { x: hpx(i), y: ay + 2, width: hpx(i + 1) - hpx(i), height: ah - 4, fill: 'transparent' });
    r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${l('code', 'код')} ${i} = ${SH.codebook[i]}</b><br>gate ${big(SH.gate_proj[i] || 0)} · up ${big(SH.up_proj[i] || 0)} · down ${big(SH.down_proj[i] || 0)}<br>${l('all', 'всего')} ${big(SH.all[i] || 0)} (${pct((SH.all[i] || 0) / SH.value_count, 4)})`, e.clientX, e.clientY));
    r.addEventListener('mouseleave', hideTip);
    svg.appendChild(r);
  }
  for (const v of [1, 4, 16, 64, 160, 256, 448]) { const i = SH.codebook.indexOf(v); if (i >= i0 && i <= i1) text(svg, hpx(i), ay + ah + 14, String(v), { anchor: 'middle', size: 9, fill: 'var(--ghost)' }); }
  const i448 = SH.codebook.indexOf(448);
  text(svg, hpx(i448) - 6, hpy(SH.all[i448]) - 6, `448 × ${big(SH.all[i448])}`, { anchor: 'end', size: 9.5, fill: 'var(--accent-deep)' });
  // B: per-layer p01/p50/p99 по проекциям
  const bx = 1470, bw = 1300, N = Q.scales.length, bpx = (i: number) => bx + 10 + i / (N - 1) * (bw - 20), bpy = (v: number) => ay + ah - 20 - (v - 40) / 300 * (ah - 44);
  text(svg, bx, 18, l('block-scale p01 … p50 … p99 by layer and projection', 'block scale p01 … p50 … p99 по слоям и проекциям'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, bx - 8, ay, bw + 16, ah);
  gridY(svg, bx, bx + bw, bpy, [80, 160, 256], v => String(v), bx - 10);
  ([['gate_proj', AMBER], ['up_proj', PLUM], ['down_proj', TEAL]] as [string, string][]).forEach(([p, c]) => {
    let band = `M${bpx(0)},${bpy(Q.scales[0][p].p99)}`; Q.scales.forEach((r: any, i: number) => band += ` L${bpx(i)},${bpy(r[p].p99)}`); for (let i = N - 1; i >= 0; i--) band += ` L${bpx(i)},${bpy(Q.scales[i][p].p01)}`; band += ' Z';
    svg.appendChild(svgEl('path', { d: band, fill: c, opacity: .12 }));
    line(svg, Q.scales.map((r: any) => r[p].p50), bpx, bpy, c, 2);
    hoverDots(svg, Q.scales.map((r: any) => r[p].mean), bpx, bpy, c, i => `<b>L${Q.scales[i].layer} · ${p}</b><br>p01 ${Q.scales[i][p].p01} · p50 ${Q.scales[i][p].p50} · p99 ${Q.scales[i][p].p99}<br>${l('mean', 'среднее')} ${fix(Q.scales[i][p].mean, 1)} · max ${Q.scales[i][p].max}`, i => selectLayer(store, Q.scales[i].layer), 2.2);
  });
  Q.scales.forEach((r: any, i: number) => { if (i % 4 === 0 || i === N - 1) text(svg, bpx(i), ay + ah + 14, `L${r.layer}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }); });
  // C: FC2 SQNR + zero fraction
  const cy = ay + ah + 54, chh = 220, cx = 70, cw = 2700, cpx = (i: number) => cx + 12 + i / (N - 1) * (cw - 24);
  text(svg, cx, cy - 8, l('FC2-input QDQ with the deployed a2 scale · SQNR (line) and share of values mapped to zero (bars)', 'QDQ входа FC2 с deployed a2 scale · SQNR (линия) и доля значений, ушедших в ноль (бары)'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, cx - 8, cy, cw + 16, chh);
  const zmax = .3, zpy = (v: number) => cy + chh - 16 - v / zmax * (chh - 40), bwid = (cw - 24) / (N - 1) * .6;
  FC.forEach((r, i) => {
    const rect = svgEl('rect', { x: cpx(i) - bwid / 2, y: zpy(r.qdq_zero_fraction), width: bwid, height: cy + chh - 16 - zpy(r.qdq_zero_fraction), fill: 'var(--line)', rx: 2 });
    rect.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${r.layer}</b><br>${pct(r.qdq_zero_fraction, 2)} ${l('of values → 0', 'значений → 0')}<br>${pct(r.exact_fraction, 2)} bit-exact<br>max |x| ${r.input_abs_max} · max |err| ${r.error_abs_max}`, e.clientX, e.clientY));
    rect.addEventListener('mouseleave', hideTip);
    svg.appendChild(rect);
  });
  const slo = 19, shi = 30, spy = (v: number) => cy + chh - 16 - (v - slo) / (shi - slo) * (chh - 40);
  gridY(svg, cx, cx + cw, spy, [20, 24, 28], v => `${v} dB`, cx - 10);
  line(svg, FC.map(r => r.sqnr_db), cpx, spy, 'var(--accent)', 2.4);
  hoverDots(svg, FC.map(r => r.sqnr_db), cpx, spy, 'var(--accent)', i => `<b>L${FC[i].layer}</b><br>SQNR <span class="mono">${fix(FC[i].sqnr_db, 3)} dB</span><br>${l('relative L2', 'относительная L2')} ${pct(FC[i].relative_l2, 2)}<br>${pct(FC[i].qdq_zero_fraction, 2)} → 0 · ${pct(FC[i].exact_fraction, 2)} bit-exact`, i => selectLayer(store, FC[i].layer), 3.2);
  text(svg, cx + cw + 6, zpy(.3) + 3, '30%', { size: 9 }); text(svg, cx + cw + 6, zpy(0) + 3, '0%', { size: 9 });
  FC.forEach((r, i) => { if (i % 3 === 0 || i === N - 1) text(svg, cpx(i), cy + chh + 14, `L${r.layer}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }); });
  // D: input_global_scale by layer; E: inventory
  const dy = cy + chh + 50, dh = H - dy - 10, dx = 70, dw = 1300, dpx = (i: number) => dx + 10 + i / (N - 1) * (dw - 20);
  text(svg, dx, dy - 8, l('input_global_scale by layer · p05 … p50 … p95 across 288 experts × 3 projections', 'input_global_scale по слоям · p05 … p50 … p95 по 288 экспертам × 3 проекции'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, dx - 8, dy, dw + 16, dh);
  const gmax = GS.input.max * 1.05, dpy = (v: number) => dy + dh - 16 - v / gmax * (dh - 36);
  let gband = `M${dpx(0)},${dpy(GS.input_by_layer[0].p95)}`; GS.input_by_layer.forEach((r: any, i: number) => gband += ` L${dpx(i)},${dpy(r.p95)}`); for (let i = N - 1; i >= 0; i--) gband += ` L${dpx(i)},${dpy(GS.input_by_layer[i].p05)}`; gband += ' Z';
  svg.appendChild(svgEl('path', { d: gband, fill: ORANGE, opacity: .18 }));
  line(svg, GS.input_by_layer.map((r: any) => r.p50), dpx, dpy, ORANGE, 2);
  hoverDots(svg, GS.input_by_layer.map((r: any) => r.p50), dpx, dpy, ORANGE, i => `<b>L${GS.input_by_layer[i].layer}</b><br>p50 ${fmt(GS.input_by_layer[i].p50, 0)} · p05 ${fmt(GS.input_by_layer[i].p05, 0)} · p95 ${fmt(GS.input_by_layer[i].p95, 0)}<br>weight_global_scale p50 ${fmt(GS.weight_by_layer[i].p50, 0)}`, i => selectLayer(store, GS.input_by_layer[i].layer), 2.4);
  gridY(svg, dx, dx + dw, dpy, [500, 1000, 1500], v => fmt(v, 0), dx - 10);
  GS.input_by_layer.forEach((r: any, i: number) => { if (i % 4 === 0 || i === N - 1) text(svg, dpx(i), dy + dh + 14, `L${r.layer}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }); });
  const ex = 1470, ew = 1300;
  text(svg, ex, dy - 8, l(`checkpoint inventory · ${fmt(inv.tensor_count, 0)} tensors by family and dtype · log width`, `инвентарь чекпоинта · ${fmt(inv.tensor_count, 0)} тензоров по семейству и dtype · log-ширина`), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, ex - 8, dy, ew + 16, dh);
  const fams = Object.entries(inv.family_counts as Record<string, number>).sort((a, b) => b[1] - a[1]);
  const dtc: Record<string, string> = { U8: TERRA, F8_E4M3: AMBER, F32: TEAL, BF16: BLUE };
  const fmax = Math.log10(Math.max(...fams.map(f => f[1])) * 1.3), rowH = (dh - 30) / fams.length;
  fams.forEach(([fam, n], i) => {
    const y = dy + 12 + i * rowH, total = Math.log10(n + 1) / fmax * (ew - 280);
    text(svg, ex + 150, y + rowH / 2 + 3, fam.replace('mtp_or_extra_layer', 'MTP / aux layer 45').replace('_', ' '), { anchor: 'end', size: 10.5, fill: 'var(--muted)' });
    let x = ex + 160;
    for (const [dt, c] of Object.entries(inv.family_dtypes[fam] as Record<string, number>).sort((a, b) => b[1] - a[1])) {
      const w = Math.max(2, total * (c / n));
      const seg = svgEl('rect', { x, y: y + 4, width: w, height: rowH - 8, fill: dtc[dt] || 'var(--line-strong)', rx: 2, opacity: .85 });
      seg.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${fam}</b> · ${dt}<br>${fmt(c, 0)} ${l('tensors', 'тензоров')} · ${pct(c / n, 1)} ${l('of family', 'семейства')}`, e.clientX, e.clientY));
      seg.addEventListener('mouseleave', hideTip);
      svg.appendChild(seg);
      x += w + 2;
    }
    text(svg, x + 6, y + rowH / 2 + 3, fmt(n, 0), { size: 10, fill: 'var(--faint)' });
  });
  Object.entries(dtc).forEach(([dt, c], i) => { svg.appendChild(svgEl('rect', { x: ex + 160 + i * 110, y: dy + dh - 14, width: 10, height: 10, rx: 2, fill: c })); text(svg, ex + 176 + i * 110, dy + dh - 5, dt, { size: 9.5 }); });
  body.appendChild(el('div', 'small-note', '', l(
    'Block scales are the quantizer’s deployed scale field, not an error measurement: 448 is the largest finite E4M3 code and occurs 70,296 times, which is code incidence, not proof of clipping. The FC2 numbers come from vLLM’s NVFP4 emulation backend on the same packed weights, because the native fused SM120 kernel never exposes its FC2 input. No BF16 checkpoint exists, so nothing here is total quantization damage.',
    'Block scales это deployed scale-поле квантайзера, а не замер ошибки: 448 это наибольший конечный код E4M3, встречается 70 296 раз, это частота кода, а не доказательство клиппинга. Числа FC2 получены через NVFP4 emulation backend vLLM на тех же упакованных весах, потому что нативный fused SM120 кернел не отдаёт вход FC2. BF16-чекпоинта нет, поэтому ничего здесь не является полным уроном от квантования.')));
  return root;
}

// ───────────────────────── 11. causal vision ─────────────────────────

function visionCard(d: any): HTMLElement {
  const V = d.vision, arms = Object.keys(V.arms), domains: string[] = V.domains;
  const { root, body } = cardShell(2850, l('DOES THE IMAGE CAUSE THE ANSWER?', 'КАРТИНКА ДЕЙСТВИТЕЛЬНО МЕНЯЕТ ОТВЕТ?'), l(
    '40 real vision questions, 10 each from ChartQA, DocVQA, MME and OCRBench, each asked 4 ways in the same loaded model: with its image, with a blank of identical pixel size, with the next image from the same benchmark, and with no image at all.',
    '40 настоящих vision-вопросов, по 10 из ChartQA, DocVQA, MME и OCRBench, каждый задан 4 способами в одной загруженной модели: со своей картинкой, с blank того же размера в пикселях, со следующей картинкой того же бенчмарка и вовсе без картинки.'));
  const armColor: Record<string, string> = { original: 'var(--accent)', blank_same_geometry: 'var(--line-strong)', mismatched_same_domain: PLUM, text_only: TEAL };
  const armLabel: Record<string, string> = { original: l('matching image', 'своя картинка'), blank_same_geometry: l('blank, same geometry', 'blank той же геометрии'), mismatched_same_domain: l('mismatched, same domain', 'чужая, тот же домен'), text_only: l('text only', 'только текст') };
  const domLabel = (dm: string) => ({ 'vision/chart': 'ChartQA', 'vision/document': 'DocVQA', 'vision/general': 'MME', 'vision/ocr': 'OCRBench' } as Record<string, string>)[dm] || dm;
  body.appendChild(legendRow(arms.map(a => [armColor[a], armLabel[a]])));
  const W = 2800, H = 520, svg = svgEl('svg', { width: W, height: H, class: 'no-pan' });
  body.appendChild(svg as any);
  // A: containment by domain × arm
  const ax = 70, aw = 1500, ay = 30, ah = 380, groups = ['all', ...domains], gw = aw / groups.length, bw = (gw - 40) / 4;
  text(svg, ax, 18, l('reference contained in the generated answer · share of questions', 'reference содержится в сгенерированном ответе · доля вопросов'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, ax - 8, ay, aw + 16, ah);
  const apy = (v: number) => ay + ah - 30 - v * (ah - 60);
  gridY(svg, ax, ax + aw, apy, [0, .25, .5, .75, 1], v => pct(v, 0), ax - 10);
  groups.forEach((g, gi) => {
    arms.forEach((a, ai) => {
      const s = g === 'all' ? V.arms[a].all : V.arms[a].by_domain[g], x = ax + gi * gw + 20 + ai * bw, v = s.contains;
      const r = svgEl('rect', { x, y: apy(v), width: bw - 4, height: apy(0) - apy(v), fill: armColor[a], rx: 3, opacity: a === 'blank_same_geometry' ? .8 : .9 });
      r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${g === 'all' ? l('all 40', 'все 40') : domLabel(g)} · ${armLabel[a]}</b><br>${l('contained', 'содержится')}: <span class="mono">${pct(v, 0)}</span> (${Math.round(v * s.n)}/${s.n})<br>token F1 ${fix(s.token_f1, 3)} · exact ${pct(s.exact, 0)}`, e.clientX, e.clientY));
      r.addEventListener('mouseleave', hideTip);
      svg.appendChild(r);
      if (v > 0) text(svg, x + (bw - 4) / 2, apy(v) - 5, pct(v, 0), { anchor: 'middle', size: 9.5, fill: a === 'original' ? 'var(--accent-deep)' : 'var(--faint)' });
    });
    text(svg, ax + gi * gw + gw / 2, ay + ah - 10, g === 'all' ? l('all 40', 'все 40') : domLabel(g), { anchor: 'middle', size: 11, fill: 'var(--muted)', weight: g === 'all' ? '600' : 'normal' });
  });
  // B: парные исходы
  const bx = 1670, bw2 = 1110, controls = ['blank_same_geometry', 'mismatched_same_domain', 'text_only'];
  text(svg, bx, 18, l('paired outcome per question: matching image wins · ties · loses', 'парный исход по вопросу: своя картинка выигрывает · ничья · проигрывает'), { fill: 'var(--ink-soft)', size: 10.5 });
  const rowsB: [string, string, any][] = [];
  controls.forEach(c => { rowsB.push([c, 'all', V.deltas[c].all]); domains.forEach(dm => rowsB.push([c, dm, V.deltas[c].by_domain[dm]])); });
  const rh = 22, lblW = 250, barW = bw2 - lblW - 90;
  rowsB.forEach(([c, g, s], i) => {
    const y = ay + 8 + i * rh + controls.indexOf(c) * 14;
    const n = s.n, w = s.contains_wins / n * barW, t = s.contains_ties / n * barW, lo = s.contains_losses / n * barW;
    text(svg, bx + lblW - 10, y + 12, g === 'all' ? `vs ${armLabel[c]}` : domLabel(g), { anchor: 'end', size: g === 'all' ? 10.5 : 10, fill: g === 'all' ? 'var(--ink-soft)' : 'var(--faint)', weight: g === 'all' ? '600' : 'normal' });
    const segs: [number, string, string][] = [[w, 'var(--accent)', l('wins', 'выигрывает')], [t, 'var(--line-strong)', l('ties', 'ничья')], [lo, TEAL, l('loses', 'проигрывает')]];
    let x = bx + lblW;
    segs.forEach(([ww, col, name]) => {
      if (ww <= 0) return;
      const r = svgEl('rect', { x, y: y + 3, width: ww, height: rh - 7, fill: col, rx: 2, opacity: .85 });
      r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${g === 'all' ? l('all 40', 'все 40') : domLabel(g)} · vs ${armLabel[c]}</b><br>${name}: ${Math.round(ww / barW * n)} / ${n}<br>${l('mean containment delta', 'средняя дельта containment')} ${s.mean_contains_delta >= 0 ? '+' : ''}${pct(s.mean_contains_delta, 0)}<br>token F1 Δ ${s.mean_token_f1_delta >= 0 ? '+' : ''}${fix(s.mean_token_f1_delta, 3)}`, e.clientX, e.clientY));
      r.addEventListener('mouseleave', hideTip);
      svg.appendChild(r);
      x += ww;
    });
    text(svg, bx + lblW + barW + 8, y + 12, `${s.mean_contains_delta >= 0 ? '+' : ''}${pct(s.mean_contains_delta, 0)}`, { size: 10, fill: s.mean_contains_delta > 0 ? 'var(--accent-deep)' : 'var(--faint)' });
  });
  const gy = ay + 8 + rowsB.length * rh + 3 * 14 + 8;
  const gt = V.generated_tokens;
  text(svg, bx, gy + 10, l('generation length receipts · answers were capped at 64 tokens', 'квитанции длины генерации · ответы ограничены 64 токенами'), { fill: 'var(--ink-soft)', size: 10.5 });
  arms.forEach((a, i) => text(svg, bx, gy + 30 + i * 16, `${armLabel[a]}: ${l('mean', 'среднее')} ${fix(gt[a].mean, 1)} · ${gt[a].hit_limit}/${gt[a].n} ${l('hit the cap', 'упёрлись в лимит')}`, { size: 10, fill: 'var(--faint)' }));
  body.appendChild(el('div', 'small-note', '', l(
    'Exact match is 0% in every arm because most answers are explanatory and 36 of 40 ran into the 64-token cap; containment is the honest metric here and token F1 is diluted by the extra prose. DocVQA is the cleanest case: 10/10 with the real document, 0/10 with a blank or with no image. MME is tied and too small to conclude anything.',
    'Exact match равен 0% во всех arms, потому что большинство ответов объяснительные и 36 из 40 упёрлись в лимит 64 токена; честная метрика здесь containment, а token F1 размыт лишней прозой. DocVQA самый чистый случай: 10/10 с настоящим документом, 0/10 с blank или без картинки. MME ничья и слишком мал для выводов.')));
  return root;
}

// ───────────────────────── 12. causal pruning arms ─────────────────────────

function pruningCard(d: any, store: Store): HTMLElement {
  const P: any[] = d.pruning, layers: number[] = d.routing.layers, N = layers.length, domains: string[] = d.routing.domains;
  const { root, body } = cardShell(2900, l('CAUSAL REAP STRESS TEST · 5 ARMS', 'CAUSAL REAP STRESS TEST · 5 ARMS'), l(
    'For every routed layer the selected routes of a prune set were zeroed and the surviving router weights renormalized, in the live model, without touching a single weight. If the REAP ranking means something, removing its bottom should hurt less than removing its top.',
    'В каждом routed-слое выбранные маршруты из prune-set обнулялись, а веса оставшихся перенормировались, в живой модели, не трогая ни одного веса. Если ранжирование REAP что-то значит, удаление низа должно вредить меньше удаления верха.'));
  const armColor: Record<string, string> = { low_reap_2pct: DEEP_TEAL, low_reap_5pct: TEAL, low_reap_10pct: '#a8cdd0', random_10pct: 'var(--line-strong)', high_reap_2pct: 'var(--accent)' };
  const W = 2850, H = 900, svg = svgEl('svg', { width: W, height: H, class: 'no-pan' });
  body.appendChild(svg as any);
  // A: scatter
  const ax = 90, aw = 1180, ay = 30, ah = 400;
  text(svg, ax, 18, l('output similarity vs router mass removed · bubble = share of tokens touched', 'сходство выхода vs удалённая router mass · пузырь = доля затронутых токенов'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, ax - 8, ay, aw + 16, ah);
  const spx = (v: number) => ax + 30 + v / .11 * (aw - 60), spy = (v: number) => ay + ah - 34 - (v - .58) / .24 * (ah - 60);
  gridY(svg, ax, ax + aw, spy, [.6, .7, .8], v => fix(v, 2), ax - 10);
  for (const x of [.02, .04, .06, .08, .1]) text(svg, spx(x), ay + ah - 14, pct(x, 0), { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' });
  const lo2 = P.find(r => r.arm === 'low_reap_2pct'), hi2 = P.find(r => r.arm === 'high_reap_2pct');
  svg.appendChild(svgEl('line', { x1: spx(lo2.removed_mass_fraction), y1: spy(lo2.normalized_edit_similarity), x2: spx(hi2.removed_mass_fraction), y2: spy(hi2.normalized_edit_similarity), stroke: 'var(--accent)', 'stroke-width': 1.5, 'stroke-dasharray': '4 4', opacity: .8 }));
  P.forEach(r => {
    const x = spx(r.removed_mass_fraction), y = spy(r.normalized_edit_similarity), rad = 10 + r.affected_token_fraction * 30;
    const c = svgEl('circle', { cx: x, cy: y, r: rad, fill: armColor[r.arm], opacity: .85, stroke: 'var(--card-solid)', 'stroke-width': 2 });
    c.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${armName(r.arm)}</b> · ${r.pruned} ${l('experts per layer', 'экспертов на слой')}<br>${l('router mass removed', 'удалено router mass')} ${pct(r.removed_mass_fraction, 2)} · ${l('slots', 'слотов')} ${pct(r.removed_slot_fraction, 2)}<br>${l('tokens touched', 'токенов затронуто')} ${pct(r.affected_token_fraction, 1)}<br>edit similarity <span class="mono">${fix(r.normalized_edit_similarity, 3)}</span> · ${l('sequences identical', 'последовательностей без изменений')} ${pct(r.sequence_exact, 1)}<br>${l('first-step top-20 Jaccard', 'top-20 Jaccard первого шага')} ${fix(r.first_step_topk_jaccard, 3)}<br>${l('fallback events', 'событий fallback')} ${r.fallback_events}`, e.clientX, e.clientY));
    c.addEventListener('mouseleave', hideTip);
    svg.appendChild(c);
    text(svg, x + rad + 8, y + 4, armName(r.arm), { size: 11, fill: r.arm === 'high_reap_2pct' ? 'var(--accent-deep)' : 'var(--muted)', weight: r.arm.endsWith('2pct') ? '600' : 'normal' });
  });
  text(svg, ax + aw / 2, ay + ah + 16, l('router mass removed →', 'удалённая router mass →'), { anchor: 'middle', size: 10 });
  text(svg, ax - 60, ay + 12, l('↑ similar', '↑ похоже'), { size: 10 });
  // B: метрики по arm
  const bx = 1370, bw = 1400, metricsB: [string, string, string][] = [['normalized_edit_similarity', 'edit similarity', TERRA], ['sequence_exact', l('sequences identical', 'последовательности целы'), AMBER], ['first_step_topk_jaccard', l('top-20 Jaccard, step 1', 'top-20 Jaccard, шаг 1'), PLUM], ['first_token_same', l('first token same', 'первый токен тот же'), TEAL]];
  text(svg, bx, 18, l('four ways to say "the output changed" · per arm, 14 records × 32 tokens', 'четыре способа сказать «выход изменился» · по arm, 14 записей × 32 токена'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, bx - 8, ay, bw + 16, ah);
  const gw2 = (bw - 40) / P.length, mb = (gw2 - 30) / metricsB.length, bpy = (v: number) => ay + ah - 34 - v * (ah - 70);
  gridY(svg, bx, bx + bw, bpy, [0, .5, 1], v => fix(v, 1), bx - 10);
  P.forEach((r, pi) => {
    metricsB.forEach(([k, name, c], mi) => {
      const x = bx + 20 + pi * gw2 + 15 + mi * mb, v = r[k];
      const rect = svgEl('rect', { x, y: bpy(v), width: mb - 4, height: bpy(0) - bpy(v), fill: c, rx: 2, opacity: .85 });
      rect.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${armName(r.arm)}</b><br>${name}: <span class="mono">${fix(v, 3)}</span>`, e.clientX, e.clientY));
      rect.addEventListener('mouseleave', hideTip);
      svg.appendChild(rect);
    });
    text(svg, bx + 20 + pi * gw2 + gw2 / 2, ay + ah - 14, armName(r.arm), { anchor: 'middle', size: 10.5, fill: r.arm === 'high_reap_2pct' ? 'var(--accent-deep)' : 'var(--muted)' });
  });
  metricsB.forEach(([, n, c], i) => { svg.appendChild(svgEl('rect', { x: bx + 20 + i * 250, y: ay + ah + 10, width: 10, height: 10, rx: 2, fill: c })); text(svg, bx + 36 + i * 250, ay + ah + 19, n, { size: 9.5 }); });
  // C: домены × arms
  const cy = ay + ah + 60, cx = 90, cell = (2760 - 220) / domains.length, ch = 30;
  text(svg, cx, cy - 8, l('edit similarity by calibration domain · one record per domain, so read as a sensitivity map, not a score', 'edit similarity по домену калибровки · по одной записи на домен, читать как карту чувствительности, не как оценку'), { fill: 'var(--ink-soft)', size: 10.5 });
  P.forEach((r, pi) => {
    const y = cy + pi * (ch + 2);
    text(svg, cx + 200, y + 19, armName(r.arm), { anchor: 'end', size: 10.5, fill: r.arm === 'high_reap_2pct' ? 'var(--accent-deep)' : 'var(--muted)' });
    domains.forEach((dm, di) => {
      const s = r.by_domain[dm], v = s.edit, rect = svgEl('rect', { x: cx + 210 + di * cell, y, width: cell - 3, height: ch, rx: 3, fill: seq(1 - v) });
      rect.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${armName(r.arm)} · ${labelDomain(dm)}</b><br>edit similarity ${fix(v, 3)} · exact ${pct(s.exact, 0)} · Jaccard ${fix(s.jaccard, 3)}<br>n = ${s.n}`, e.clientX, e.clientY));
      rect.addEventListener('mouseleave', hideTip);
      svg.appendChild(rect);
      text(svg, cx + 210 + di * cell + (cell - 3) / 2, y + 19, fix(v, 2), { anchor: 'middle', size: 10, fill: seqInk(1 - v) });
    });
  });
  domains.forEach((dm, di) => text(svg, cx + 210 + di * cell + (cell - 3) / 2, cy + P.length * (ch + 2) + 14, labelDomain(dm), { anchor: 'middle', size: 9.5, fill: 'var(--ghost)' }));
  // D: per-layer affected
  const dy = cy + P.length * (ch + 2) + 50, dh = H - dy - 12, dx = 90, dw = 2680, dpx = (i: number) => dx + 12 + i / (N - 1) * (dw - 24), dpy = (v: number) => dy + dh - 16 - v * (dh - 34);
  text(svg, dx, dy - 8, l('share of routed tokens touched, by layer and arm', 'доля затронутых routed-токенов по слоям и arms'), { fill: 'var(--ink-soft)', size: 10.5 });
  frame(svg, dx - 8, dy, dw + 16, dh);
  gridY(svg, dx, dx + dw, dpy, [0, .25, .5, .75], v => pct(v, 0), dx - 10);
  P.forEach(r => {
    line(svg, r.per_layer_affected, dpx, dpy, armColor[r.arm], r.arm.endsWith('2pct') ? 2.4 : 1.6);
    hoverDots(svg, r.per_layer_affected, dpx, dpy, armColor[r.arm], i => `<b>${armName(r.arm)} · L${layers[i]}</b><br>${pct(r.per_layer_affected[i], 1)} ${l('of tokens touched', 'токенов затронуто')}<br>${pct(r.per_layer_mass[i], 2)} ${l('router mass removed', 'router mass удалено')}`, i => selectLayer(store, layers[i]), 2.2);
  });
  layers.forEach((L, i) => { if (i % 3 === 0 || i === N - 1) text(svg, dpx(i), dy + dh + 14, `L${L}`, { anchor: 'middle', size: 9, fill: 'var(--ghost)' }); });
  body.appendChild(el('div', 'small-note', '', l(
    'At 2%, dropping the top-REAP experts changes 79% of sequences and pulls edit similarity to 0.679 while dropping the bottom-REAP experts changes 64% and keeps 0.775, on similar removed mass. That is the causal signal. The 10% arms are mixed and fired the all-selected fallback 29 times; they say why a larger held-out evaluation is needed, not that bottom 10% is safe. Masking selected routes is not the same as physically removing experts and rerouting, so no speed claim follows either.',
    'На 2% удаление top-REAP экспертов меняет 79% последовательностей и роняет edit similarity до 0.679, а удаление bottom-REAP меняет 64% и держит 0.775, при близкой удалённой массе. Это и есть causal-сигнал. Arms на 10% смешанные и 29 раз сработал fallback all-selected; они объясняют, зачем нужна большая held-out оценка, а не что bottom 10% безопасен. Маскирование маршрутов не равно физическому удалению экспертов с перемаршрутизацией, поэтому и выводов о скорости отсюда нет.')));
  return root;
}

// ───────────────────────── сборка региона ─────────────────────────

export function buildGlmInsights(store: Store, data: any, X: number, Y: number, fly?: (r: Rect) => void): GlmSection {
  const W = 5800;
  const root = el('div', 'section glm-lab', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `<div class="section-tag mono">LIVE</div><div class="section-title">${l('Measured model', 'Измеренная модель')}</div><div class="section-sub">${l('weights · forward passes · causal controls, all from the deployed NVFP4 checkpoint', 'веса · forward passes · causal controls, всё с развёрнутого NVFP4-чекпоинта')}</div>`));
  redraws.length = 0;
  const cards = new Map<string, HTMLElement>();
  const jump = (key: string) => {
    const c = cards.get(key); if (!c || !fly) return;
    fly({ x: X + c.offsetLeft - 40, y: Y + c.offsetTop - 40, w: c.offsetWidth + 80, h: c.offsetHeight + 80 });
  };
  const grid = el('div', '', 'display:flex;flex-direction:column;gap:26px');
  const rowOf = (...items: [string, HTMLElement][]) => {
    const r = el('div', '', 'display:flex;gap:26px;align-items:flex-start');
    for (const [k, c] of items) { cards.set(k, c); r.appendChild(c); }
    grid.appendChild(r);
  };
  rowOf(['receipt', receiptCard(data, jump)]);
  rowOf(['atlas', atlasCard(data, store)]);
  rowOf(['router', routerCard(data, store)], ['trust', trustCard(data)]);
  rowOf(['kda', kdaCard(data, store)], ['indexer', indexerCard(data, store)]);
  rowOf(['flow', flowCard(data, store)], ['actq', actqCard(data, store)]);
  rowOf(['shared', sharedCard(data, store)], ['nvfp4', nvfp4Card(data, store)]);
  rowOf(['vision', visionCard(data)], ['pruning', pruningCard(data, store)]);
  root.appendChild(grid);
  new MutationObserver(() => redraws.forEach(f => f())).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  return { root, rect: { x: X, y: Y, w: W, h: 7200 } };
}
