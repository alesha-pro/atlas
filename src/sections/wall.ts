// Стена: все тензоры модели одной таблицей. Колонка = слой, ряд = роль тензора.
// Ряды выровнены между слоями — ритм полное/линейное внимание читается сразу.
import { fmtN, plainName, slotRank, type Tensor } from '../data';
import { t as tr } from '../i18n';
import { colorForTensor, kindOf } from '../color';
import { el } from '../world';
import { tip, hideTip, type Store } from '../store';

const PAD = 26, HEAD_H = 120, LBL_W = 190;
const COL_W = 34, COL_G = 3, ROW_H = 21, ROW_G = 3, TICK_H = 18, NUM_H = 20;
const VCOL_W = 24, VCOL_G = 2;
const VLBL_W = 150;               // отдельная полоса под подписи рядов башни

export function buildWall(store: Store, X: number, Y: number): {
  root: HTMLElement; rect: { x: number; y: number; w: number; h: number };
} {
  const m = store.model;

  // ряды языковой части — объединение слотов всех слоёв
  const slotSet = new Map<string, number>();
  for (const L of m.langLayers) for (const x of L.tensors)
    if (!slotSet.has(x.slot)) slotSet.set(x.slot, slotRank(x.slot));
  const slots = [...slotSet.keys()].sort((a, b) => slotSet.get(a)! - slotSet.get(b)!);

  // ряды визуальной башни
  const vslotSet = new Set<string>();
  for (const B of m.visBlocks) for (const x of B.tensors) vslotSet.add(x.slot);
  const vslots = [...vslotSet];

  const nLang = m.langLayers.length, nVis = m.visBlocks.length;
  const langW = nLang * (COL_W + COL_G);
  const visW = nVis * (VCOL_W + VCOL_G);
  const gridH = slots.length * (ROW_H + ROW_G);
  const gapLV = 40;
  const specialsW = 250;

  const cardW = PAD * 2 + LBL_W + langW + gapLV + VLBL_W + visW + gapLV + specialsW;
  const cardH = HEAD_H + TICK_H + gridH + NUM_H + PAD + 14;

  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:${cardW}px`);
  root.appendChild(el('div', 'section-head', '', `
    <div class="section-tag mono">${tr('sec.wall.tag')}</div>
    <div class="section-title">${tr('sec.wall.title', m.tensors.length)}</div>
    <div class="section-sub">${tr('sec.wall.sub')}</div>`));

  const card = el('div', 'card', `position:relative;width:${cardW}px;height:${cardH}px`);
  root.appendChild(card);

  card.appendChild(el('div', '', `position:absolute;left:${PAD}px;top:22px;display:flex;flex-direction:column;gap:5px`, `
    <div class="eyebrow mono">${tr('wall.head', nLang, nVis)}</div>
    <div class="note" style="max-width:900px">${tr('wall.note').replace('◆', `<span class="mono" style="color:${kindOf('attn').fg}">◆</span>`)}</div>`));

  const gx = PAD + LBL_W, gy = HEAD_H + TICK_H;

  // подписи рядов
  for (let r = 0; r < slots.length; r++) {
    const s = slots[r];
    const lbl = el('div', 'mono', `position:absolute;left:${PAD}px;top:${gy + r * (ROW_H + ROW_G)}px;width:${LBL_W - 12}px;height:${ROW_H}px;
      display:flex;align-items:center;justify-content:flex-end;gap:7px;font-size:10.5px;color:var(--faint)`);
    const plain = plainName(s);
    lbl.innerHTML = `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s}</span>${plain ? `<span style="color:var(--ghost);font-style:italic;font-family:var(--serif);font-size:11.5px;white-space:nowrap">${plain}</span>` : ''}`;
    card.appendChild(lbl);
  }

  const cellRefs: { div: HTMLElement; t: Tensor }[] = [];
  const addCell = (x0: Tensor, x: number, y: number, w: number, h: number) => {
    const c = el('div', 'wall-cell no-pan', `position:absolute;left:${x}px;top:${y}px;width:${w}px;height:${h}px;border-radius:2px;background:${colorForTensor(x0, store.md)}`);
    c.addEventListener('click', () => store.select({ type: 'tensor', tensor: x0 }));
    c.addEventListener('mouseenter', e => showTensorTip(store, x0, e as MouseEvent));
    c.addEventListener('mousemove', e => showTensorTip(store, x0, e as MouseEvent));
    c.addEventListener('mouseleave', hideTip);
    card.appendChild(c);
    cellRefs.push({ div: c, t: x0 });
    store.cellRect.set(x0.idx, { x: X + x, y: Y + 38 + y, w, h }); // 38 ≈ высота section-head
    return c;
  };

  // ── языковые колонки ──
  m.langLayers.forEach((L, li) => {
    const cx = gx + li * (COL_W + COL_G);
    // засечка
    const tick = el('div', 'mono', `position:absolute;left:${cx}px;top:${HEAD_H}px;width:${COL_W}px;height:${TICK_H - 4}px;
      display:flex;align-items:flex-end;justify-content:center;font-size:9px;color:${kindOf('attn').fg}`);
    if (L.kind === 'full') tick.textContent = '◆';
    card.appendChild(tick);
    // клетки
    for (const x0 of L.tensors) {
      const r = slots.indexOf(x0.slot);
      if (r < 0) continue;
      addCell(x0, cx, gy + r * (ROW_H + ROW_G), COL_W, ROW_H);
    }
    // номер
    const num = el('div', 'mono', `position:absolute;left:${cx}px;top:${gy + gridH + 2}px;width:${COL_W}px;text-align:center;font-size:9.5px;color:var(--ghost)`);
    if (li % 4 === 0 || li === nLang - 1) num.textContent = L.label;
    card.appendChild(num);
  });

  // ── визуальная башня (своя полоса подписей — ничего не накладывается) ──
  const vlx = gx + langW + gapLV;         // начало полосы подписей
  const vx = vlx + VLBL_W;                // начало сетки башни
  if (nVis) {
    card.appendChild(el('div', 'eyebrow mono', `position:absolute;left:${vlx}px;top:${HEAD_H - 4}px;color:${kindOf('vision').fg}`, tr('wall.vision')));
    for (let r = 0; r < vslots.length; r++) {
      const lblTxt = vslots[r].replace('vis.', '');
      const lbl = el('div', 'mono', `position:absolute;left:${vlx}px;top:${gy + r * (ROW_H + ROW_G)}px;width:${VLBL_W - 10}px;height:${ROW_H}px;
        display:flex;align-items:center;justify-content:flex-end;font-size:9.5px;color:var(--ghost);white-space:nowrap;overflow:hidden`);
      lbl.textContent = lblTxt;
      card.appendChild(lbl);
    }
    m.visBlocks.forEach((B, bi) => {
      const cx = vx + bi * (VCOL_W + VCOL_G);
      for (const x0 of B.tensors) {
        const r = vslots.indexOf(x0.slot);
        if (r < 0) continue;
        addCell(x0, cx, gy + r * (ROW_H + ROW_G), VCOL_W, ROW_H);
      }
      const num = el('div', 'mono', `position:absolute;left:${cx}px;top:${gy + gridH + 2}px;width:${VCOL_W}px;text-align:center;font-size:9px;color:var(--ghost)`);
      if (bi % 4 === 0 || bi === nVis - 1) num.textContent = String(bi);
      card.appendChild(num);
    });
  }

  // ── особые тензоры: вход/выход/mtp + внеблочные визуальные ──
  const sx = vx + visW + gapLV;
  card.appendChild(el('div', 'eyebrow mono', `position:absolute;left:${sx}px;top:${HEAD_H - 4}px`, tr('wall.outside')));
  const specials = [...m.topLayer.tensors, ...m.visExtra];
  specials.forEach((x0, i) => {
    const y = gy + i * (ROW_H + ROW_G);
    if (y + ROW_H > gy + gridH + 40) return;
    addCell(x0, sx, y, 26, ROW_H);
    const lbl = el('div', 'mono no-pan', `position:absolute;left:${sx + 32}px;top:${y}px;height:${ROW_H}px;display:flex;align-items:center;
      font-size:10px;color:var(--faint);white-space:nowrap;overflow:hidden;max-width:${specialsW - 40}px;cursor:pointer`);
    lbl.textContent = x0.short;
    lbl.addEventListener('click', () => store.select({ type: 'tensor', tensor: x0 }));
    card.appendChild(lbl);
  });

  // ── реакции ──
  store.addEventListener('metric', () => {
    for (const { div, t } of cellRefs) div.style.background = colorForTensor(t, store.md);
  });
  let lastSel: HTMLElement | null = null;
  store.addEventListener('sel', () => {
    if (lastSel) { lastSel.style.boxShadow = ''; lastSel.style.zIndex = ''; }
    if (store.sel.type === 'tensor') {
      const sel = store.sel;
      const ref = cellRefs.find(c => c.t.idx === sel.tensor.idx);
      if (ref) {
        ref.div.style.boxShadow = '0 0 0 2.5px var(--accent), 0 0 0 5px rgba(255,138,61,0.25)';
        ref.div.style.zIndex = '3';
        lastSel = ref.div;
      }
    }
  });

  return { root, rect: { x: X, y: Y, w: cardW, h: cardH + 40 } };
}

export function showTensorTip(store: Store, x0: Tensor, e: MouseEvent) {
  const d = store.md;
  const v = d.get(x0);
  tip(`<div class="mono" style="font-size:12px;margin-bottom:3px">${x0.short}</div>
    <div style="display:flex;gap:10px;align-items:baseline">
      <span class="mono">${v == null ? tr('na') : d.fmt(v) + (d.unit ? ' ' + d.unit : '')}</span>
      <span style="opacity:0.75">${d.label}</span>
      <span style="opacity:0.6">· ${fmtN(x0.numel)}</span>
    </div>`, e.clientX, e.clientY);
}
