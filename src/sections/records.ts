// Доска рекордов: крайние точки модели, посчитанные из данных.
import { fmtN, type Model, type Tensor } from '../data';
import { ink } from '../color';
import { t as tr } from '../i18n';
import { el } from '../world';
import type { Store } from '../store';

const W = 1200;

export function buildRecords(store: Store, X: number, Y: number): {
  root: HTMLElement; rect: { x: number; y: number; w: number; h: number };
} {
  const m = store.model;
  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `
    <div class="section-tag mono">${tr('sec.records.tag')}</div>
    <div class="section-title">${tr('sec.records.title')}</div>
    <div class="section-sub">${tr('sec.records.sub')}</div>`));

  const grid = el('div', '', 'display:grid;grid-template-columns:1fr 1fr;gap:12px');
  root.appendChild(grid);

  const d2 = m.tensors.filter(t => t.is2d);
  const by = (f: (t: Tensor) => number | null | undefined, min = true) => {
    let best: Tensor | null = null, bv = min ? Infinity : -Infinity;
    for (const t of m.tensors) {
      const v = f(t);
      if (v == null || !isFinite(v)) continue;
      if (min ? v < bv : v > bv) { bv = v; best = t; }
    }
    return best;
  };

  const md4 = m.metrics.int4, md8 = m.metrics.int8;
  const db = tr('unit.db');
  const recs: { label: string; t: Tensor | null; val: (t: Tensor) => string; note: string; inkc?: string }[] = [
    { label: tr('rec.worst4.label'), t: by(t => t.sqnr_int4_g128), val: t => t.sqnr_int4_g128!.toFixed(2) + ' ' + db, note: tr('rec.worst4.note'), inkc: 'var(--bad)' },
    { label: tr('rec.best4.label'), t: by(t => t.sqnr_int4_g128, false), val: t => t.sqnr_int4_g128!.toFixed(2) + ' ' + db, note: tr('rec.best4.note'), inkc: 'var(--good)' },
    { label: tr('rec.worst8.label'), t: by(t => t.sqnr_int8_ch), val: t => t.sqnr_int8_ch!.toFixed(1) + ' ' + db, note: tr('rec.worst8.note'), inkc: 'var(--bad)' },
    { label: tr('rec.kurt.label'), t: by(t => t.kurtosis, false), val: t => tr('rec.kurt.val', t.kurtosis.toFixed(0)), note: tr('rec.kurt.note') },
    { label: tr('rec.hot.label'), t: by(t => t.hot, false), val: t => '×' + t.hot!.toFixed(1), note: tr('rec.hot.note') },
    { label: tr('rec.dyn.label'), t: by(t => t.dyn_range, false), val: t => '×' + t.dyn_range.toFixed(0), note: tr('rec.dyn.note') },
    { label: tr('rec.lowrank.label'), t: by(t => t.stable_rank), val: t => tr('rec.lowrank.val', t.stable_rank!.toFixed(1)), note: tr('rec.lowrank.note') },
    { label: tr('rec.dense.label'), t: by(t => t.stable_rank, false), val: t => tr('rec.lowrank.val', String(Math.round(t.stable_rank!))), note: tr('rec.dense.note') },
    { label: tr('rec.sparse.label'), t: by(t => t.sparsity, false), val: t => tr('rec.sparse.val', (t.sparsity * 100).toFixed(2)), note: tr('rec.sparse.note') },
    { label: tr('rec.big.label'), t: by(t => t.numel, false), val: t => fmtN(t.numel), note: tr('rec.big.note') },
  ];

  for (const r of recs) {
    if (!r.t) continue;
    const t = r.t;
    const c = el('div', 'card no-pan', 'padding:16px 18px;display:flex;flex-direction:column;gap:6px;cursor:pointer');
    c.innerHTML = `
      <div class="eyebrow mono">${r.label}</div>
      <div class="mono" style="font-size:26px;color:${r.inkc || 'var(--ink)'};letter-spacing:-0.01em">${r.val(t)}</div>
      <div class="mono" style="font-size:11.5px;color:var(--faint);overflow-wrap:anywhere">${t.short}</div>
      <div class="small-note">${r.note}</div>`;
    c.addEventListener('click', () => store.select({ type: 'tensor', tensor: t }));
    c.addEventListener('mouseenter', () => c.style.borderColor = 'var(--accent)');
    c.addEventListener('mouseleave', () => c.style.borderColor = 'var(--card-bd)');
    grid.appendChild(c);
  }

  return { root, rect: { x: X, y: Y, w: W, h: 1500 } };
}
