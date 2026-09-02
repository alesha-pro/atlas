// Инспектор: разбор тензора / слоя / группы простым языком, все числа настоящие.
import { fmtN, plainName, type Layer, type MetricDef, type Model, type Tensor } from './data';
import { t as tr } from './i18n';
import { colorFor, colorForTensor, ink, kindOf, ramp, tOf, NA_INK } from './color';
import { el } from './world';
import type { Store } from './store';

export function buildPanel(store: Store, onFly: (t: Tensor) => void): HTMLElement {
  const panel = el('div', 'panel no-pan');
  const head = el('div', 'panel-head');
  const body = el('div', 'panel-body');
  panel.append(head, body);

  const render = () => {
    const s = store.sel;
    if (s.type === 'model') { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    if (s.type === 'tensor') renderTensor(store, head, body, s.tensor, onFly);
    else if (s.type === 'layer') renderLayer(store, head, body, s.layer);
    else if (s.type === 'group') renderGroup(store, head, body, s.group.key);
  };
  store.addEventListener('sel', render);
  store.addEventListener('metric', render);
  render();
  return panel;
}

function headHTML(tag: string, tagKind: string, title: string, sub: string): string {
  const k = kindOf(tagKind);
  return `
    <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
      <div class="tag-pill mono" style="background:${k.bg};color:${k.fg}">${tag}</div>
      <div class="mono close-btn" style="font-size:15px;color:var(--ghost);cursor:pointer;padding:0 4px">✕</div>
    </div>
    <div style="font-size:24px;line-height:1.15;margin-top:12px;overflow-wrap:anywhere">${title}</div>
    <div class="mono" style="font-size:12px;color:var(--ghost);margin-top:6px;overflow-wrap:anywhere">${sub}</div>`;
}

function wireClose(store: Store, head: HTMLElement) {
  head.querySelector('.close-btn')?.addEventListener('click', () => store.select({ type: 'model' }));
}

function kindOfTensor(x: Tensor): string {
  return x.group === 'attn' ? 'attn' : x.group === 'linattn' ? 'lin' : x.group === 'mlp' ? 'mlp'
    : x.group === 'vision' ? 'vision' : x.group === 'embed' ? 'in'
    : x.group === 'head' || x.group === 'mtp' ? 'out' : 'norm';
}

// ── вердикт: правило-based текст из настоящих чисел ──
function verdict(m: Model, x: Tensor): string {
  if (!x.is2d) return tr('verdict.1d', fmtN(x.numel));
  const v4 = x.sqnr_int4_g128!;
  const arr = m.int4Sorted;
  const pos = arr.findIndex(v => v >= v4) / arr.length;
  const posKey = pos < 0.1 ? 'pos.bottom10' : pos < 0.33 ? 'pos.bottom3' : pos > 0.75 ? 'pos.top' : 'pos.mid';
  const parts: string[] = [tr('verdict.int4', v4.toFixed(1), tr(posKey))];
  if (x.kurtosis > 50) parts.push(tr('verdict.wild', x.kurtosis.toFixed(0)));
  else if (x.kurtosis > 6) parts.push(tr('verdict.tail', x.kurtosis.toFixed(1)));
  if ((x.hot ?? 0) > 10) parts.push(tr('verdict.hot', x.hot!.toFixed(1)));
  if (x.stable_rank != null && x.stable_rank < 12) parts.push(tr('verdict.lowrank', x.stable_rank.toFixed(1)));
  if (v4 >= 18 && x.kurtosis < 3 && (x.hot ?? 0) < 8) parts.push(tr('verdict.calm'));
  return parts.join('; ') + '.';
}

function sqnrCards(x: Tensor, m: Model): string {
  const items: [string, number | undefined, MetricDef, string][] = [
    [tr('sqnr.int8'), x.sqnr_int8_ch, m.metrics.int8, tr('sqnr.int8.note')],
    [tr('sqnr.int4'), x.sqnr_int4_g128, m.metrics.int4, tr('sqnr.int4.note')],
    [tr('sqnr.fp8'), x.sqnr_fp8_e4m3, m.metrics.fp8, tr('sqnr.fp8.note')],
  ];
  return `<div style="display:flex;gap:9px">${items.map(([label, v, d, note]) => `
    <div class="stat-card">
      <div class="mono" style="font-size:10px;letter-spacing:0.08em;color:var(--faint)">${label}</div>
      <div class="mono" style="font-size:21px;line-height:1;color:${v == null ? NA_INK : ink(v, d)}">${v == null ? tr('na') : v.toFixed(2)}</div>
      <div class="small-note" style="font-size:11.5px">${note}</div>
    </div>`).join('')}</div>`;
}

function histBlock(x: Tensor): string {
  const bins = x.hist_log2 || [];
  const mx = Math.max(...bins, 1e-9);
  const bars = bins.map((v, i) => {
    const h = 4 + (v / mx) * 96;
    const tt = i / (bins.length - 1);
    return `<div style="flex:1;height:${h.toFixed(1)}px;background:${ramp(0.25 + tt * 0.5)};border-radius:2px 2px 0 0" title="${(v * 100).toFixed(2)}%"></div>`;
  }).join('');
  return `
    <div style="display:flex;flex-direction:column;gap:8px">
      <div style="display:flex;align-items:baseline;justify-content:space-between">
        <div class="eyebrow mono">${tr('hist.title')}</div>
        <div class="mono" style="font-size:10.5px;color:var(--ghost)">${tr('hist.sub', bins.length)}</div>
      </div>
      <div style="height:112px;display:flex;align-items:flex-end;gap:1.5px;padding:8px 10px;background:var(--chart-bg);border:1px solid var(--line);border-radius:10px">${bars}</div>
      <div class="mono" style="display:flex;justify-content:space-between;font-size:10px;color:var(--ghost)">
        <div>${tr('hist.x0')}</div><div>2⁻¹⁰</div><div>2⁰</div><div>2⁴</div>
      </div>
      <div class="small-note">${x.kurtosis > 5 || (x.hot ?? 0) > 10 ? tr('hist.note.tail') : tr('hist.note.bell')}</div>
    </div>`;
}

function spectrumBlock(x: Tensor): string {
  if (!x.sv_top?.length) return '';
  const mx = x.sv_top[0] || 1;
  const bars = x.sv_top.map(v => `
    <div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px">
      <div class="mono" style="font-size:8.5px;color:var(--ghost)">${v.toFixed(1)}</div>
      <div style="width:70%;height:${(6 + (v / mx) * 74).toFixed(1)}px;background:oklch(0.82 0.06 62);border-radius:2px 2px 0 0"></div>
    </div>`).join('');
  return `
    <div style="display:flex;flex-direction:column;gap:8px">
      <div style="display:flex;align-items:baseline;justify-content:space-between">
        <div class="eyebrow mono">${tr('spec.title')}</div>
        <div class="mono" style="font-size:10.5px;color:var(--ghost)">${tr('spec.rank')} ${x.stable_rank?.toFixed(1)}</div>
      </div>
      <div style="height:104px;display:flex;align-items:flex-end;gap:2px;padding:6px 10px;background:var(--chart-bg);border:1px solid var(--line);border-radius:10px">${bars}</div>
      <div class="small-note">${x.stable_rank != null && x.stable_rank < 15 ? tr('spec.note.low') : tr('spec.note.dense')}</div>
    </div>`;
}

function rowsBlock(store: Store, x: Tensor): string {
  const m = store.model;
  const rows: [string, string, number | null | undefined, MetricDef | null, string][] = [
    [tr('row.kurt'), 'kurtosis', x.kurtosis, m.metrics.kurt, tr('row.kurt.plain')],
    [tr('row.skew'), 'skew', x.skew, null, tr('row.skew.plain')],
    [tr('row.hot'), 'row/col amax ratio', x.hot, m.metrics.hot, tr('row.hot.plain')],
    [tr('row.dyn'), 'dyn_range', x.dyn_range, m.metrics.dyn, tr('row.dyn.plain')],
    [tr('row.sparsity'), 'sparsity', x.sparsity * 100, m.metrics.sparsity, tr('row.sparsity.plain')],
    [tr('row.out3s'), 'outlier_3s', x.outlier_3s * 100, m.metrics.out3s, tr('row.out3s.plain')],
  ];
  return `
    <div style="display:flex;flex-direction:column;gap:2px">
      <div class="eyebrow mono" style="margin-bottom:6px">${tr('rows.title')}</div>
      ${rows.map(([label, key, v, d, plain]) => {
        const has = v != null && isFinite(v as number);
        const w = has && d ? Math.max(3, Math.min(100, (tOf(v as number, d) ?? 0) * 100)) : 0;
        return `
        <div class="metric-row">
          <div style="display:flex;align-items:baseline;justify-content:space-between;gap:10px">
            <div style="font-size:14px">${label} <span class="mono" style="font-size:9.5px;color:var(--ghost)">${key}</span></div>
            <div class="mono" style="font-size:13.5px;color:var(--ink-soft)">${has ? (v as number).toFixed(key === 'sparsity' || key === 'outlier_3s' ? 3 : 2) : tr('na')}</div>
          </div>
          ${d ? `<div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${has ? colorFor(v as number, d) : 'var(--na-fill)'}"></div></div>` : ''}
          <div class="small-note" style="font-size:12px">${plain}</div>
        </div>`;
      }).join('')}
    </div>`;
}

function percentileBlock(x: Tensor): string {
  const items: [string, number][] = [
    ['p50', x.p50], ['p90', x.p90], ['p99', x.p99], ['p99.9', x.p999], ['p99.99', x.p9999], ['absmax', x.absmax],
  ];
  return `
    <div style="display:flex;flex-direction:column;gap:7px">
      <div class="eyebrow mono">${tr('pct.title')}</div>
      <div style="display:flex;gap:5px">${items.map(([l, v]) => `
        <div style="flex:1;border:1px solid var(--line);border-radius:8px;padding:7px 4px;text-align:center;background:var(--glass-soft)">
          <div class="mono" style="font-size:9px;color:var(--ghost)">${l}</div>
          <div class="mono" style="font-size:10.5px;color:var(--ink-soft);margin-top:2px">${fmtVal(v)}</div>
        </div>`).join('')}</div>
      <div class="small-note" style="font-size:12px">${tr('pct.note', (x.absmax / Math.max(x.p9999, 1e-12)).toFixed(1))}</div>
    </div>`;
}

function fmtVal(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toExponential(1).replace('e-', 'e−');
}

function renderTensor(store: Store, head: HTMLElement, body: HTMLElement, x: Tensor, onFly: (t: Tensor) => void) {
  const m = store.model;
  const kind = kindOfTensor(x);
  head.innerHTML = headHTML(plainName(x.slot) || x.group, kind, x.short,
    `[${x.shape.join('×')}] · ${x.dtype} · ${fmtN(x.numel)} · ${x.shard.replace('model-', '').replace('.safetensors', '')}`);
  wireClose(store, head);

  if (m.entry.kind === 'glm-live') {
    const qdq = x.sqnr_int4_g128;
    body.innerHTML = `
      <div style="font-size:16px;line-height:1.55;color:var(--ink-soft);text-wrap:pretty">${tr('glm.panel.tensor.note')}</div>
      <div class="stat-card">
        <div class="mono" style="font-size:10px;letter-spacing:0.08em;color:var(--faint)">${qdq == null ? tr('glm.panel.provenance') : tr('glm.metric.qdq')}</div>
        <div class="mono" style="font-size:${qdq == null ? 15 : 24}px;line-height:1.15;color:var(--ink)">${qdq == null ? tr('glm.panel.config') : `${qdq.toFixed(2)} ${tr('unit.db')}`}</div>
        <div class="small-note">${qdq == null ? tr('glm.panel.config.note') : tr('glm.panel.qdq.note')}</div>
      </div>
      <div class="chip mini" style="align-self:flex-start" data-fly>${tr('panel.fly')}</div>`;
    body.querySelector('[data-fly]')?.addEventListener('click', () => onFly(x));
    body.scrollTop = 0;
    return;
  }

  body.innerHTML = `
    <div style="font-size:16px;line-height:1.55;color:var(--ink-soft);text-wrap:pretty">${verdict(m, x)}</div>
    ${x.is2d || x.sqnr_int8_ch != null ? sqnrCards(x, m) : ''}
    ${histBlock(x)}
    ${spectrumBlock(x)}
    ${rowsBlock(store, x)}
    ${percentileBlock(x)}
    <div class="chip mini" style="align-self:flex-start" data-fly>${tr('panel.fly')}</div>`;
  body.querySelector('[data-fly]')?.addEventListener('click', () => onFly(x));
  body.scrollTop = 0;
}

function renderLayer(store: Store, head: HTMLElement, body: HTMLElement, L: Layer) {
  const m = store.model;
  const d = store.md;
  const kind = L.kind === 'full' ? 'attn' : L.kind === 'linear' ? 'lin' : L.kind === 'vision' ? 'vision' : 'out';
  const label = L.kind === 'full' ? tr(m.entry.kind === 'glm-live' ? 'glm.kind.sparse' : 'kind.full') : L.kind === 'linear' ? tr(m.entry.kind === 'glm-live' ? 'glm.kind.kda' : 'kind.linear.long')
    : L.kind === 'vision' ? tr('kind.visblock') : tr('kind.top');
  const title = L.kind === 'vision' ? tr('layer.title.vis', L.label.replace('v', ''))
    : L.kind === 'top' ? tr('layer.title.top') : tr('layer.title', L.label);
  head.innerHTML = headHTML(label, kind, title, tr('layer.sub', L.tensors.length, fmtN(L.params)));
  wireClose(store, head);

  const int4s = L.tensors.map(x => x.sqnr_int4_g128).filter((v): v is number => v != null);
  const avg = int4s.length ? int4s.reduce((a, b) => a + b, 0) / int4s.length : null;
  const textKey = m.entry.kind === 'glm-live'
    ? (L.kind === 'full' ? 'glm.layer.text.sparse' : L.kind === 'linear' ? 'glm.layer.text.kda'
      : L.kind === 'vision' ? 'glm.layer.text.vision' : 'glm.layer.text.top')
    : L.kind === 'full' ? 'layer.text.full' : L.kind === 'linear' ? 'layer.text.linear'
    : L.kind === 'vision' ? 'layer.text.vision' : 'layer.text.top';

  body.innerHTML = `
    <div style="font-size:15.5px;line-height:1.55;color:var(--ink-soft);text-wrap:pretty">${tr(textKey)}${
      avg != null ? (m.entry.kind === 'glm-live' ? tr('glm.layer.avg', avg.toFixed(2)) : tr('layer.avg', avg.toFixed(2))) : ''}</div>
    <div style="display:flex;flex-direction:column;gap:6px">
      <div class="eyebrow mono">${tr('layer.list')}</div>
      <div class="layer-list" style="display:flex;flex-direction:column;gap:5px"></div>
    </div>`;

  const list = body.querySelector('.layer-list')!;
  for (const x of L.tensors) {
    const v = d.get(x);
    const row = el('div', 't-row');
    row.innerHTML = `
      <div style="width:9px;height:24px;border-radius:2px;background:${colorForTensor(x, d)}"></div>
      <div style="flex:1;min-width:0">
        <div class="mono" style="font-size:12.5px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${x.slot.replace('vis.', '')}</div>
        <div style="font-size:12.5px;color:var(--faint)">${plainName(x.slot) || ''} · ${fmtN(x.numel)}</div>
      </div>
      <div class="mono" style="font-size:13.5px;color:${v == null ? NA_INK : ink(v, d)}">${v == null ? tr('na') : d.fmt(v)}</div>`;
    row.addEventListener('click', () => store.select({ type: 'tensor', tensor: x }));
    list.appendChild(row);
  }
  body.scrollTop = 0;
}

function renderGroup(store: Store, head: HTMLElement, body: HTMLElement, key: string) {
  const m = store.model;
  const d = store.md;
  const g = m.groups.find(x => x.key === key)!;
  head.innerHTML = headHTML(g.plain, g.kind, g.label,
    `${tr('layer.sub', g.tensors.length, fmtN(g.params))} · ${g.share.toFixed(1)}%`);
  wireClose(store, head);

  if (m.entry.kind === 'glm-live') {
    const measured = g.tensors.filter(x => x.sqnr_int4_g128 != null);
    body.innerHTML = `
      <div style="font-size:15.5px;line-height:1.55;color:var(--ink-soft);text-wrap:pretty">${tr('glm.panel.group.note', g.tensors.length, measured.length)}</div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <div class="eyebrow mono">${tr('layer.list')}</div>
        ${g.tensors.slice(0, 18).map(x => `<div class="t-row" data-idx="${x.idx}">
          <div style="width:9px;height:24px;border-radius:2px;background:${colorForTensor(x, d)}"></div>
          <div style="flex:1;min-width:0"><div class="mono" style="font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${x.short}</div><div style="font-size:12px;color:var(--faint)">${fmtN(x.numel)} · ${x.dtype}</div></div>
          <div class="mono" style="font-size:13px;color:var(--ink-soft)">${x.sqnr_int4_g128 == null ? tr('na') : `${x.sqnr_int4_g128.toFixed(2)} ${tr('unit.db')}`}</div>
        </div>`).join('')}
      </div>`;
    body.querySelectorAll<HTMLElement>('[data-idx]').forEach(row => row.addEventListener('click', () => store.select({ type: 'tensor', tensor: m.tensors[+row.dataset.idx!] })));
    body.scrollTop = 0;
    return;
  }

  const d2 = g.tensors.filter(x => x.is2d);
  const worst = [...d2].sort((a, b) => a.sqnr_int4_g128! - b.sqnr_int4_g128!).slice(0, 7);
  const best = [...d2].sort((a, b) => b.sqnr_int4_g128! - a.sqnr_int4_g128!).slice(0, 3);
  const avg = d2.length ? d2.reduce((a, x) => a + x.sqnr_int4_g128!, 0) / d2.length : null;

  const mkRow = (x: Tensor) => {
    const v = d.get(x);
    return `<div class="t-row" data-idx="${x.idx}">
      <div style="width:9px;height:24px;border-radius:2px;background:${colorForTensor(x, d)}"></div>
      <div style="flex:1;min-width:0">
        <div class="mono" style="font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${x.short}</div>
        <div style="font-size:12px;color:var(--faint)">${tr('group.row.sub', x.sqnr_int4_g128?.toFixed(2), x.kurtosis.toFixed(1))}</div>
      </div>
      <div class="mono" style="font-size:13px;color:${v == null ? NA_INK : ink(v, d)}">${v == null ? tr('na') : d.fmt(v)}</div>
    </div>`;
  };

  body.innerHTML = `
    <div style="font-size:15.5px;line-height:1.55;color:var(--ink-soft);text-wrap:pretty">
      ${avg != null ? tr('group.avg', avg.toFixed(2)) : tr('group.1d')}
    </div>
    ${d2.length ? `
      <div style="display:flex;flex-direction:column;gap:6px">
        <div class="eyebrow mono" style="color:var(--bad)">${tr('group.fragile')}</div>
        ${worst.map(mkRow).join('')}
      </div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <div class="eyebrow mono" style="color:var(--good)">${tr('group.robust')}</div>
        ${best.map(mkRow).join('')}
      </div>` : ''}`;

  body.querySelectorAll<HTMLElement>('[data-idx]').forEach(row => {
    row.addEventListener('click', () => {
      const x = m.tensors[+row.dataset.idx!];
      store.select({ type: 'tensor', tensor: x });
    });
  });
  body.scrollTop = 0;
}
