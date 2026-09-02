import type { Store } from '../store';
import { hideTip, tip } from '../store';
import { lang } from '../i18n';
import { el, svgEl } from '../world';

type Rect = { x: number; y: number; w: number; h: number };
export interface GlmSection { root: HTMLElement; rect: Rect }

const l = (en: string, ru: string) => lang === 'ru' ? ru : en;
const fmt = (v: number, d = 1) => Number(v).toLocaleString(lang === 'ru' ? 'ru-RU' : 'en-US', { maximumFractionDigits: d });
const pct = (v: number, d = 1) => `${fmt(v * 100, d)}%`;
const labelDomain = (d: string) => d.replace('knowledge/multitopic', 'knowledge').replace('multilingual/', '').replace('vision/', 'vision · ');

function card(w: number, title: string, sub: string): { root: HTMLElement; body: HTMLElement } {
  const root = el('div', 'card glm-card', `width:${w}px;padding:22px 25px;display:flex;flex-direction:column;gap:14px`);
  root.innerHTML = `<div class="eyebrow accent mono">${title}</div><div class="note" style="font-size:14.5px;max-width:${w - 60}px">${sub}</div>`;
  const body = el('div', '', 'display:flex;flex-direction:column;gap:12px');
  root.appendChild(body);
  return { root, body };
}

function chip(text: string, on = false): HTMLElement {
  return el('div', `chip mini no-pan${on ? ' on' : ''}`, '', text);
}

function ramp(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  const stops = [[236, 230, 217], [108, 174, 174], [88, 104, 150], [139, 70, 119], [215, 103, 56]];
  const z = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(z)), f = z - i;
  return `rgb(${stops[i].map((x, j) => Math.round(x + (stops[i + 1][j] - x) * f)).join(',')})`;
}

function quantile(a: number[], q: number): number {
  const x = a.filter(Number.isFinite).sort((p, c) => p - c);
  return x[Math.max(0, Math.min(x.length - 1, Math.round(q * (x.length - 1))))] ?? 0;
}

function statTile(value: string, name: string, note: string): string {
  return `<div style="border:1px solid var(--line);background:var(--glass-soft);border-radius:10px;padding:14px 15px;min-height:96px">
    <div class="mono" style="font-size:25px;color:var(--ink);letter-spacing:-.03em">${value}</div>
    <div style="font-size:13px;color:var(--muted);margin-top:5px">${name}</div>
    <div class="mono" style="font-size:9.5px;color:var(--ghost);margin-top:7px">${note}</div></div>`;
}

function overviewCard(data: any): HTMLElement {
  const { root, body } = card(1130, l('CAPTURE RECEIPT', 'ПАСПОРТ CAPTURE'),
    l('One deployed checkpoint, multiple independent views. Counts below are kept separate so “real data” never means one undifferentiated run.',
      'Один развёрнутый чекпоинт, несколько независимых срезов. Счётчики разделены, чтобы «реальные данные» не превращались в один неразличимый прогон.'));
  const m = data.meta;
  const reap = Object.values(m.reap_tokens_per_layer as Record<string, number>)[0] as number;
  body.appendChild(el('div', '', 'display:grid;grid-template-columns:repeat(3,1fr);gap:10px', [
    statTile('320B / 18B', l('total / active', 'всего / активно'), l('official release + config', 'официальный релиз + config')),
    statTile('42 × 288', l('routed layer × expert', 'routed-слой × эксперт'), `${fmt(reap / 1e6, 2)}M REAP ${l('tokens/layer', 'токенов/слой')}`),
    statTile('724', l('behavioural rows', 'behavioral-строк'), `${fmt(m.estimated_tokens / 1e3, 0)}K ${l('estimated tokens', 'токенов')}`),
    statTile('100', l('real Vision rows', 'реальных Vision-строк'), l('40 paired causal questions', '40 парных causal-вопросов')),
    statTile('36,288', l('NVFP4 expert matrices', 'NVFP4-матриц экспертов'), l('all block scales finite', 'все block scales конечны')),
    statTile('14', l('capture domains', 'доменов capture'), l('text · code · multilingual · vision', 'text · code · multilingual · vision')),
  ].join('')));
  body.appendChild(el('div', 'small-note', '', l(
    'Hover charts for exact values. Click a layer in any depth plot to connect it back to the architecture wall. No prompts, generations, images, activations or raw routes are published here.',
    'Наведитесь на график за точным значением. Клик по слою связывает любой depth-график с архитектурной стеной. Prompts, generations, изображения, активации и raw routes здесь не публикуются.')));
  return root;
}

function routingAtlas(data: any, store: Store): HTMLElement {
  const { root, body } = card(3600, l('THE EXPERT ATLAS', 'АТЛАС ЭКСПЕРТОВ'), l(
    '12,096 expert cells. Switch between exact REAP importance, observed route share and sampled output contribution. REAP can be sliced by 14 domains.',
    '12 096 ячеек экспертов. Переключайтесь между exact REAP importance, наблюдаемой долей маршрутов и sampled output contribution. REAP разбивается на 14 доменов.'));
  const controls = el('div', '', 'display:flex;gap:8px;align-items:center;flex-wrap:wrap');
  const metricDefs: Record<string, { name: string; unit: string }> = {
    reap: { name: 'exact REAP', unit: '' }, route_share: { name: l('route share', 'доля routes'), unit: '%' }, contribution: { name: l('output contribution', 'вклад выхода'), unit: '' },
  };
  let metric = 'reap', domain = 'all';
  const metricRow = el('div', '', 'display:flex;gap:7px');
  Object.entries(metricDefs).forEach(([k, d], i) => {
    const b = chip(d.name, i === 0); b.dataset.metric = k;
    b.onclick = (e) => { e.stopPropagation(); metric = k; metricRow.querySelectorAll('.chip').forEach(x => x.classList.toggle('on', (x as HTMLElement).dataset.metric === k)); domainSel.style.display = k === 'reap' ? '' : 'none'; draw(); };
    metricRow.appendChild(b);
  });
  controls.appendChild(metricRow);
  const domainSel = document.createElement('select');
  domainSel.className = 'plain no-pan';
  domainSel.innerHTML = ['all', ...data.routing.domains].map((d: string) => `<option value="${d}">${d === 'all' ? l('all domains', 'все домены') : labelDomain(d)}</option>`).join('');
  domainSel.onchange = () => { domain = domainSel.value; draw(); };
  controls.appendChild(domainSel); body.appendChild(controls);
  const W = 3500, H = 545, x0 = 72, y0 = 26, cw = 11.55, ch = 11.2;
  const wrap = el('div', 'no-pan', `width:${W}px;height:${H}px;position:relative;cursor:crosshair`); body.appendChild(wrap);
  function draw() {
    wrap.innerHTML = '';
    const svg = svgEl('svg', { width: W, height: H }); wrap.appendChild(svg);
    let matrix = metric === 'reap' ? (domain === 'all' ? data.routing.reap : data.routing.reap_domains[domain]) : data.routing[metric];
    const values = matrix.flat().filter((v: number) => v > 0 && Number.isFinite(v));
    const log = metric !== 'contribution';
    const tx = (v: number) => log ? Math.log10(Math.max(v, 1e-14)) : Math.sqrt(Math.max(0, v));
    const lo = quantile(values.map(tx), .02), hi = quantile(values.map(tx), .98);
    matrix.forEach((row: number[], ri: number) => row.forEach((v, ei) => {
      const t = (tx(v) - lo) / (hi - lo || 1);
      const r = svgEl('rect', { x: x0 + ei * cw, y: y0 + ri * ch, width: cw - .8, height: ch - .8, rx: 1, fill: v > 0 ? ramp(t) : 'var(--hatch-a)' });
      r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${data.routing.layers[ri]} · E${ei}</b><br>${metricDefs[metric].name}: ${metric === 'route_share' ? pct(v, 4) : fmt(v, 6)}${domain !== 'all' ? `<br>${labelDomain(domain)}` : ''}`, e.clientX, e.clientY));
      r.addEventListener('mouseleave', hideTip);
      r.addEventListener('click', () => { const L = store.model.langLayers.find(x => +x.label === data.routing.layers[ri]); if (L) store.select({ type: 'layer', layer: L }); });
      svg.appendChild(r);
    }));
    data.routing.layers.forEach((v: number, i: number) => { if (i % 4 === 0 || i === 41) { const t = svgEl('text', { x: 34, y: y0 + i * ch + 9, fill: 'var(--faint)', 'font-size': 9 }); t.textContent = `L${v}`; svg.appendChild(t); } });
    for (let e = 0; e < 288; e += 16) { const t = svgEl('text', { x: x0 + e * cw, y: 12, fill: 'var(--ghost)', 'font-size': 9 }); t.textContent = String(e); svg.appendChild(t); }
    const legend = svgEl('g', {}); for (let i = 0; i < 100; i++) legend.appendChild(svgEl('rect', { x: x0 + i * 3, y: H - 32, width: 3.2, height: 10, fill: ramp(i / 99) }));
    const lt = svgEl('text', { x: x0 + 315, y: H - 23, fill: 'var(--faint)', 'font-size': 10 }); lt.textContent = `${l('low', 'ниже')} → ${l('high', 'выше')} · ${log ? 'log' : 'sqrt'} scale`; legend.appendChild(lt); svg.appendChild(legend);
  }
  draw();
  body.appendChild(el('div', 'small-note', '', l(
    'Colour is normalized within the selected view, so it reveals structure rather than pretending that REAP, route frequency and output norm share one unit.',
    'Цвет нормируется внутри выбранного среза: так видна структура, но REAP, частота маршрута и output norm не притворяются одной физической величиной.')));
  return root;
}

function dynamicsCard(data: any, store: Store): HTMLElement {
  const { root, body } = card(1780, l('ROUTER UNDER LOAD', 'РОУТЕР ПОД НАГРУЗКОЙ'), l(
    'How decisive is top-8 routing, how broad is the effective mixture, and how uneven is expert use as depth changes?',
    'Насколько решителен top-8 routing, какова эффективная ширина смеси и насколько неравномерно используются эксперты по глубине?'));
  let domain = 'all';
  const ctl = document.createElement('select'); ctl.className = 'plain no-pan';
  ctl.innerHTML = ['all', ...data.routing.domains].map((d: string) => `<option value="${d}">${d === 'all' ? l('all domains', 'все домены') : labelDomain(d)}</option>`).join('');
  ctl.onchange = () => { domain = ctl.value; draw(); }; body.appendChild(ctl);
  const W = 1715, H = 390, wrap = el('div', 'no-pan', `width:${W}px;height:${H}px`); body.appendChild(wrap);
  const defs = [
    ['effective', l('effective experts', 'эффективных экспертов'), '#3f7f7a'],
    ['margin', l('top1 − top2 margin', 'разрыв top1 − top2'), '#d76a38'],
    ['selected_gini', l('selected-load Gini', 'Gini нагрузки'), '#8a5791'],
  ];
  function draw() {
    wrap.innerHTML = ''; const svg = svgEl('svg', { width: W, height: H }); wrap.appendChild(svg);
    const rows = domain === 'all' ? data.routing.dynamics.all : data.routing.dynamics.domains[domain];
    const chartW = 1510, x0 = 72, px = (i: number) => x0 + i / (rows.length - 1) * chartW;
    defs.forEach(([key, name, color], di) => {
      const y0 = 24 + di * 116, vals = rows.map((x: any) => x[key]);
      let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * .12 || 1; lo -= pad; hi += pad;
      const py = (v: number) => y0 + 76 - (v - lo) / (hi - lo) * 70;
      svg.appendChild(svgEl('rect', { x: x0 - 8, y: y0, width: chartW + 16, height: 84, rx: 5, fill: 'var(--chart-bg)', stroke: 'var(--line)' }));
      let d = ''; vals.forEach((v: number, i: number) => d += `${i ? 'L' : 'M'}${px(i)},${py(v)} `);
      svg.appendChild(svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': 2 }));
      const tx = svgEl('text', { x: 2, y: y0 + 13, fill: color, 'font-size': 10 }); tx.textContent = name as string; svg.appendChild(tx);
      rows.forEach((row: any, i: number) => { const c = svgEl('circle', { cx: px(i), cy: py(vals[i]), r: 3.2, fill: color }); c.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${row.layer}</b><br>${name}: ${fmt(vals[i], 4)}<br>${fmt(row.tokens, 0)} tokens`, e.clientX, e.clientY)); c.addEventListener('mouseleave', hideTip); c.addEventListener('click', () => { const L = store.model.langLayers.find(x => +x.label === row.layer); if (L) store.select({ type: 'layer', layer: L }); }); svg.appendChild(c); });
    });
    rows.forEach((x: any, i: number) => { if (i % 4 === 0) { const t = svgEl('text', { x: px(i) - 8, y: H - 4, fill: 'var(--ghost)', 'font-size': 9 }); t.textContent = String(x.layer); svg.appendChild(t); } });
  } draw(); return root;
}

function memoryCard(data: any, store: Store): HTMLElement {
  const { root, body } = card(2500, l('TWO KINDS OF LONG MEMORY', 'ДВА ВИДА ДЛИННОЙ ПАМЯТИ'), l(
    'Left: KDA half-life for every recurrent head. Right: how far the learned sparse indexer reaches as sequence position grows.',
    'Слева: half-life каждой рекуррентной KDA-головы. Справа: как далеко тянется learned sparse indexer с ростом позиции.'));
  const W = 2430, H = 590, svg = svgEl('svg', { width: W, height: H }); body.appendChild(svg as any);
  const x0 = 70, y0 = 50, cw = 17, ch = 13.5;
  const flat = data.memory.kda_heads.flat().map((x: number) => Math.log10(Math.max(.1, x))), lo = quantile(flat, .02), hi = quantile(flat, .98);
  let title = svgEl('text', { x: x0, y: 20, fill: 'var(--ink)', 'font-size': 13 }); title.textContent = l('KDA head half-life · tokens · log colour', 'KDA half-life голов · токены · log-цвет'); svg.appendChild(title);
  data.memory.kda_heads.forEach((row: number[], ri: number) => row.forEach((v, h) => {
    const t = (Math.log10(Math.max(.1, v)) - lo) / (hi - lo || 1), r = svgEl('rect', { x: x0 + h * cw, y: y0 + ri * ch, width: cw - 1, height: ch - 1, rx: 1, fill: ramp(t) });
    r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${data.memory.kda_layers[ri]} · head ${h}</b><br>half-life: ${fmt(v, 2)} tokens`, e.clientX, e.clientY)); r.addEventListener('mouseleave', hideTip); r.addEventListener('click', () => { const L = store.model.langLayers.find(x => +x.label === data.memory.kda_layers[ri]); if (L) store.select({ type: 'layer', layer: L }); }); svg.appendChild(r);
  }));
  for (let h = 0; h < 64; h += 8) { const t = svgEl('text', { x: x0 + h * cw, y: 40, fill: 'var(--ghost)', 'font-size': 9 }); t.textContent = String(h); svg.appendChild(t); }
  data.memory.kda_layers.forEach((v: number, i: number) => { if (i % 4 === 0) { const t = svgEl('text', { x: 32, y: y0 + i * ch + 10, fill: 'var(--faint)', 'font-size': 9 }); t.textContent = `L${v}`; svg.appendChild(t); } });
  const ix = 1290, iy = 50, iw = 1040, ih = 445;
  title = svgEl('text', { x: ix, y: 20, fill: 'var(--ink)', 'font-size': 13 }); title.textContent = l('Sparse indexer · mean selected distance', 'Sparse indexer · средняя выбранная дистанция'); svg.appendChild(title);
  svg.appendChild(svgEl('rect', { x: ix, y: iy, width: iw, height: ih, rx: 6, fill: 'var(--chart-bg)', stroke: 'var(--line)' }));
  const vals = data.memory.indexer.flatMap((x: any) => x.position_distance).filter((x: number) => x > 0), vmax = Math.max(...vals), py = (v: number) => iy + ih - Math.log10(Math.max(1, v)) / Math.log10(vmax) * (ih - 24), px = (b: number) => ix + 48 + b * ((iw - 95) / 5);
  data.memory.indexer.forEach((row: any, ri: number) => {
    let d = ''; row.position_distance.forEach((v: number, b: number) => d += `${b ? 'L' : 'M'}${px(b)},${py(v)} `);
    const color = ramp(ri / (data.memory.indexer.length - 1)); svg.appendChild(svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': ri === data.memory.indexer.length - 1 ? 3 : 1.5, opacity: .8 }));
    row.position_distance.forEach((v: number, b: number) => { const c = svgEl('circle', { cx: px(b), cy: py(v), r: 3, fill: color }); c.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${row.layer} · ${data.memory.position_buckets[b]}</b><br>${fmt(v, 0)} tokens away`, e.clientX, e.clientY)); c.addEventListener('mouseleave', hideTip); svg.appendChild(c); });
  });
  data.memory.position_buckets.forEach((v: string, i: number) => { const t = svgEl('text', { x: px(i) - 18, y: iy + ih + 20, fill: 'var(--faint)', 'font-size': 9 }); t.textContent = v; svg.appendChild(t); });
  for (const v of [10, 100, 1000, 10000]) { const y = py(v); svg.appendChild(svgEl('line', { x1: ix, x2: ix + iw, y1: y, y2: y, stroke: 'var(--line)', 'stroke-dasharray': '3 4' })); const t = svgEl('text', { x: ix + 5, y: y - 3, fill: 'var(--ghost)', 'font-size': 9 }); t.textContent = String(v); svg.appendChild(t); }
  body.appendChild(el('div', 'small-note', '', l('The heatmap is clipped at the 2nd/98th percentiles for readability; tooltips retain exact means. Long-lived heads form a heavy tail rather than one shared decay rate.', 'Heatmap ограничен 2-м/98-м перцентилями для читаемости; tooltip сохраняет точные средние. Долгоживущие головы образуют heavy tail, а не одну общую скорость забывания.')));
  return root;
}

function contributionCard(data: any, store: Store): HTMLElement {
  const { root, body } = card(1390, l('SHARED VS ROUTED', 'SHARED ПРОТИВ ROUTED'), l('The always-on expert and the routed branch do not keep the same balance through depth.', 'Баланс always-on эксперта и routed-ветки меняется по глубине.'));
  const W = 1320, H = 410, svg = svgEl('svg', { width: W, height: H }); body.appendChild(svg as any);
  const rows = data.contributions.layers, x0 = 55, y0 = 35, cw = 28, ch = 275;
  svg.appendChild(svgEl('line', { x1: x0, x2: x0 + cw * rows.length, y1: y0 + ch / 2, y2: y0 + ch / 2, stroke: 'var(--line-strong)', 'stroke-dasharray': '4 4' }));
  rows.forEach((r: any, i: number) => {
    const x = x0 + i * cw, h = r.shared_energy * ch;
    const bar = svgEl('rect', { x, y: y0 + ch - h, width: cw - 4, height: h, rx: 2, fill: ramp(r.shared_energy) });
    bar.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${r.layer}</b><br>${l('shared energy', 'энергия shared')}: ${pct(r.shared_energy)}<br>${l('shared/routed cosine', 'cosine shared/routed')}: ${fmt(r.shared_routed_cosine, 3)}<br>${l('constructive gain', 'constructive gain')}: ${fmt(r.constructive_gain, 3)}`, e.clientX, e.clientY)); bar.addEventListener('mouseleave', hideTip); bar.addEventListener('click', () => { const L = store.model.langLayers.find(x => +x.label === r.layer); if (L) store.select({ type: 'layer', layer: L }); }); svg.appendChild(bar);
    if (i % 4 === 0) { const t = svgEl('text', { x, y: H - 55, fill: 'var(--ghost)', 'font-size': 9 }); t.textContent = String(r.layer); svg.appendChild(t); }
  });
  const p = data.contributions.pair_quantiles, py = (v: number) => H - 18 - (v + .15) / .4 * 72;
  let d = ''; p.forEach((r: any, i: number) => d += `${i ? 'L' : 'M'}${x0 + i * cw + 11},${py(r.p50)} `); svg.appendChild(svgEl('path', { d, fill: 'none', stroke: '#8a5791', 'stroke-width': 2 }));
  const tx = svgEl('text', { x: 4, y: 18, fill: 'var(--faint)', 'font-size': 10 }); tx.textContent = l('shared energy fraction', 'доля shared energy'); svg.appendChild(tx);
  const tx2 = svgEl('text', { x: 4, y: H - 10, fill: '#8a5791', 'font-size': 10 }); tx2.textContent = l('median expert-pair cosine', 'median cosine пар экспертов'); svg.appendChild(tx2);
  return root;
}

function quantCard(data: any, store: Store): HTMLElement {
  const { root, body } = card(2070, l('NVFP4, AS DEPLOYED', 'NVFP4 КАК ОН РАБОТАЛ'), l(
    'Block-scale structure from packed weights, then FC2-input quantize/dequantize error with the checkpoint’s own deployed activation scale.',
    'Структура block scales из packed weights, затем ошибка quantize/dequantize на входе FC2 с собственным deployed activation scale чекпоинта.'));
  const W = 2000, H = 420, svg = svgEl('svg', { width: W, height: H }); body.appendChild(svg as any);
  const scales = data.quantization.scales, fc = data.quantization.fc2, x0 = 70, plotW = 850, y0 = 45, h = 285, px = (i: number) => x0 + i / 41 * plotW;
  const colors: Record<string, string> = { gate_proj: '#d76a38', up_proj: '#8a5791', down_proj: '#3f7f7a' };
  let title = svgEl('text', { x: x0, y: 18, fill: 'var(--ink)', 'font-size': 13 }); title.textContent = l('weight block-scale median · FP8 scale code', 'median block scale весов · FP8 scale code'); svg.appendChild(title);
  for (const p of Object.keys(colors)) {
    const vals = scales.map((r: any) => r[p].p50), lo = 60, hi = 240, py = (v: number) => y0 + h - (v - lo) / (hi - lo) * h;
    let d = ''; vals.forEach((v: number, i: number) => d += `${i ? 'L' : 'M'}${px(i)},${py(v)} `); svg.appendChild(svgEl('path', { d, fill: 'none', stroke: colors[p], 'stroke-width': 2 }));
    const t = svgEl('text', { x: x0 + Object.keys(colors).indexOf(p) * 180, y: H - 18, fill: colors[p], 'font-size': 10 }); t.textContent = p; svg.appendChild(t);
  }
  const fx = 1080, fw = 840, fpx = (i: number) => fx + i / 41 * fw, sq = fc.map((r: any) => r.sqnr_db), lo = Math.min(...sq) - .5, hi = Math.max(...sq) + .5, fpy = (v: number) => y0 + h - (v - lo) / (hi - lo) * h;
  title = svgEl('text', { x: fx, y: 18, fill: 'var(--ink)', 'font-size': 13 }); title.textContent = l('FC2-input deployed-scale SQNR · dB', 'FC2-input deployed-scale SQNR · dB'); svg.appendChild(title);
  svg.appendChild(svgEl('rect', { x: fx - 10, y: y0, width: fw + 20, height: h, rx: 6, fill: 'var(--chart-bg)', stroke: 'var(--line)' }));
  let d = ''; fc.forEach((r: any, i: number) => d += `${i ? 'L' : 'M'}${fpx(i)},${fpy(r.sqnr_db)} `); svg.appendChild(svgEl('path', { d, fill: 'none', stroke: '#d76a38', 'stroke-width': 2.3 }));
  fc.forEach((r: any, i: number) => { const c = svgEl('circle', { cx: fpx(i), cy: fpy(r.sqnr_db), r: 3.4, fill: '#d76a38' }); c.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>L${r.layer}</b><br>SQNR ${fmt(r.sqnr_db, 3)} dB<br>relative L2 ${pct(r.relative_l2, 2)}<br>QDQ zero ${pct(r.qdq_zero_fraction, 2)}<br>exact ${pct(r.exact_fraction, 2)}`, e.clientX, e.clientY)); c.addEventListener('mouseleave', hideTip); c.addEventListener('click', () => { const L = store.model.langLayers.find(x => +x.label === r.layer); if (L) store.select({ type: 'layer', layer: L }); }); svg.appendChild(c); });
  body.appendChild(el('div', 'small-note', '', l('Median SQNR is 21.05 dB, measured on FC2 input quantize/dequantize with the checkpoint’s deployed activation scale.', 'Median SQNR равен 21,05 dB: это замер FC2 input quantize/dequantize с deployed activation scale самого чекпоинта.')));
  return root;
}

function visionCard(data: any): HTMLElement {
  const { root, body } = card(1510, l('DOES THE IMAGE CAUSE THE ANSWER?', 'ИЗОБРАЖЕНИЕ ДЕЙСТВИТЕЛЬНО МЕНЯЕТ ОТВЕТ?'), l(
    'Same question, four causal arms: matching image, blank image with the same geometry, wrong image from the same domain, and text only.',
    'Один вопрос, четыре causal arms: правильная картинка, blank той же геометрии, неправильная картинка того же домена и только текст.'));
  const W = 1440, H = 400, svg = svgEl('svg', { width: W, height: H }); body.appendChild(svg as any);
  const arms = Object.entries(data.vision.arms), domains = data.vision.domains, colors = ['#d76a38', '#b5aa96', '#8a5791', '#3f7f7a'], x0 = 70, groupW = 330, barW = 54, maxH = 280;
  domains.forEach((d: string, di: number) => {
    arms.forEach(([name, a]: any, ai) => {
      const v = a.by_domain[d].contains, h = v * maxH, x = x0 + di * groupW + ai * (barW + 10), y = 305 - h;
      const r = svgEl('rect', { x, y, width: barW, height: h, rx: 3, fill: colors[ai] }); r.addEventListener('mousemove', (e: MouseEvent) => tip(`<b>${labelDomain(d)}</b><br>${name.replaceAll('_', ' ')}<br>${l('reference contained', 'reference contained')}: ${pct(v, 0)}<br>token F1: ${fmt(a.by_domain[d].token_f1, 4)}`, e.clientX, e.clientY)); r.addEventListener('mouseleave', hideTip); svg.appendChild(r);
    });
    const t = svgEl('text', { x: x0 + di * groupW, y: 328, fill: 'var(--faint)', 'font-size': 10 }); t.textContent = labelDomain(d); svg.appendChild(t);
  });
  arms.forEach(([name], i) => { const t = svgEl('text', { x: 25 + i * 350, y: 382, fill: colors[i], 'font-size': 10 }); t.textContent = name.replace('blank_same_geometry', 'blank same geometry').replace('mismatched_same_domain', 'mismatched same domain').replaceAll('_', ' '); svg.appendChild(t); });
  body.appendChild(el('div', 'small-note', '', l('Reference containment: 57.5% with the original image, 5% blank, 10% mismatched, 10% text-only. Exact match is intentionally not presented as quality evidence because explanatory outputs and a 64-token cap drove it to zero.', 'Reference containment: 57,5% с оригиналом, 5% blank, 10% mismatch, 10% text-only. Exact match не выдаётся за оценку качества: explanatory outputs и лимит 64 токена обнулили эту метрику.')));
  return root;
}

function pruningCard(data: any): HTMLElement {
  const { root, body } = card(1510, l('CAUSAL REAP STRESS TEST', 'CAUSAL REAP STRESS TEST'), l(
    'Selected routes were zeroed and surviving weights renormalized. This tests whether the ranking matters without modifying a single checkpoint weight.',
    'Выбранные routes занулялись, а веса оставшихся перенормировались. Так проверяется смысл ranking без изменения весов чекпоинта.'));
  const W = 1440, H = 410, svg = svgEl('svg', { width: W, height: H }); body.appendChild(svg as any);
  const x0 = 90, y0 = 25, pw = 1120, ph = 300, colors: Record<string, string> = { low_reap_2pct:'#3f7f7a', low_reap_5pct:'#62a39d', low_reap_10pct:'#84bbb1', random_10pct:'#b5aa96', high_reap_2pct:'#d76a38' };
  const px = (v: number) => x0 + v / .11 * pw, py = (v: number) => y0 + ph - (v - .55) / .25 * ph;
  svg.appendChild(svgEl('rect', { x: x0, y: y0, width: pw, height: ph, rx: 6, fill: 'var(--chart-bg)', stroke: 'var(--line)' }));
  for (const x of [.02,.04,.06,.08,.10]) { const xx = px(x); const t = svgEl('text', { x: xx - 10, y: y0 + ph + 18, fill:'var(--ghost)','font-size':9 }); t.textContent = pct(x,0); svg.appendChild(t); }
  data.pruning.forEach((r: any) => {
    const x = px(r.removed_mass_fraction), y = py(r.normalized_edit_similarity), rad = 9 + r.affected_token_fraction * 22;
    const c = svgEl('circle', { cx:x, cy:y, r:rad, fill:colors[r.arm], opacity:.86, stroke:'var(--card-solid)','stroke-width':2 }); c.addEventListener('mousemove',(e:MouseEvent)=>tip(`<b>${r.arm.replaceAll('_',' ')}</b><br>${l('mass removed','масса удалена')}: ${pct(r.removed_mass_fraction,2)}<br>${l('tokens affected','токенов затронуто')}: ${pct(r.affected_token_fraction,1)}<br>${l('sequence exact','sequence exact')}: ${pct(r.sequence_exact,1)}<br>${l('edit similarity','edit similarity')}: ${fmt(r.normalized_edit_similarity,3)}<br>top20 Jaccard: ${fmt(r.first_step_topk_jaccard,3)}`,e.clientX,e.clientY)); c.addEventListener('mouseleave',hideTip); svg.appendChild(c);
    const t = svgEl('text',{x:x+rad+6,y:y+4,fill:colors[r.arm],'font-size':10}); t.textContent=r.arm.replace('reap_','').replace('pct','%').replaceAll('_',' '); svg.appendChild(t);
  });
  let tx=svgEl('text',{x:x0+pw/2-80,y:H-42,fill:'var(--faint)','font-size':10}); tx.textContent=l('router mass removed →','удалённая router mass →'); svg.appendChild(tx);
  tx=svgEl('text',{x:4,y:18,fill:'var(--faint)','font-size':10}); tx.textContent=l('↑ output similarity','↑ сходство выхода'); svg.appendChild(tx);
  body.appendChild(el('div','small-note','',l('At 2%, removing high-REAP experts changes sequences more than removing low-REAP experts even though removed route mass is similar. The 10% arms are mixed and do not justify a production pruning claim.', 'При 2% удаление high-REAP экспертов меняет sequences сильнее, чем удаление low-REAP при близкой удалённой route mass. Результаты 10% смешанные и не дают основания заявлять production pruning.')));
  return root;
}

function stabilityCard(data: any): HTMLElement {
  const { root, body } = card(1190, l('CAN THE RANKING BE TRUSTED?', 'МОЖНО ЛИ ДОВЕРЯТЬ RANKING?'), l('Split-half agreement is high; frequency alone is not an importance proxy.', 'Split-half agreement высокий; одна частота не заменяет importance.'));
  const W=1120,H=390,svg=svgEl('svg',{width:W,height:H});body.appendChild(svg as any);
  const half=data.stability.halves.exact_reap_rho, controls=data.stability.controls_vs_exact_reap, domains=Object.entries(data.stability.domain_vs_global_exact_reap).map(([k,v]:any)=>[k,v.rho.median]).sort((a:any,b:any)=>b[1]-a[1]);
  body.insertBefore(el('div','', 'display:grid;grid-template-columns:repeat(3,1fr);gap:10px', [
    statTile(fmt(half.median,4),l('split-half Spearman','split-half Spearman'),`min ${fmt(half.min,4)}`),
    statTile(fmt(controls.proxy.rho.median,3),l('proxy vs exact REAP','proxy vs exact REAP'),l('strong control','сильный control')),
    statTile(fmt(controls.count.rho.median,3),l('frequency vs exact','frequency vs exact'),l('wrong direction','неверное направление')),
  ].join('')),body.firstChild);
  const x0=190,y0=20,bh=18,maxW=780;
  domains.forEach(([name,value]:any,i:number)=>{const y=y0+i*(bh+6);const t=svgEl('text',{x:0,y:y+12,fill:'var(--faint)','font-size':9});t.textContent=name.replaceAll('_',' · ');svg.appendChild(t);const r=svgEl('rect',{x:x0,y,width:Math.max(1,value*maxW),height:bh,rx:2,fill:ramp(value)});r.addEventListener('mousemove',(e:MouseEvent)=>tip(`<b>${name.replaceAll('_',' ')}</b><br>median ρ ${fmt(value,4)}`,e.clientX,e.clientY));r.addEventListener('mouseleave',hideTip);svg.appendChild(r);});
  return root;
}

export function buildGlmInsights(store: Store, data: any, X: number, Y: number): GlmSection {
  const W = 5800;
  const root = el('div', 'section glm-lab', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `<div class="section-tag mono">LIVE</div><div class="section-title">${l('Measured model', 'Измеренная модель')}</div><div class="section-sub">${l('weights · forward passes · causal controls', 'веса · forward passes · causal controls')}</div>`));
  const grid = el('div', '', 'display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start'); root.appendChild(grid);
  [overviewCard(data), dynamicsCard(data, store), stabilityCard(data), routingAtlas(data, store), memoryCard(data, store), contributionCard(data, store), quantCard(data, store), visionCard(data), pruningCard(data)].forEach(x => grid.appendChild(x));
  return { root, rect: { x: X, y: Y, w: W, h: 3200 } };
}
