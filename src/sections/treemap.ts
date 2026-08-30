// Гербарий: площадь = число параметров. Видно, что 212 нормировок легче одного down_proj.
import { fmtN, type Tensor } from '../data';
import { colorForTensor, kindOf, HATCH } from '../color';
import { t as tr } from '../i18n';
import { el } from '../world';
import { hideTip, type Store } from '../store';
import { showTensorTip } from './wall';

const W = 1560, H = 1330;

interface Item { t?: Tensor; label: string; value: number; tiny?: Tensor[]; }
interface Placed extends Item { x: number; y: number; w: number; h: number; }

// классический squarify
function squarify(items: Item[], x: number, y: number, w: number, h: number): Placed[] {
  const out: Placed[] = [];
  let list = [...items].sort((a, b) => b.value - a.value);
  const total = list.reduce((a, i) => a + i.value, 0);
  let scale = (w * h) / total;

  let cx = x, cy = y, cw = w, ch = h;
  while (list.length) {
    const horiz = cw >= ch; // укладываем колонку вдоль короткой стороны
    const side = horiz ? ch : cw;
    let row: Item[] = [];
    let best = Infinity;
    let i = 0;
    for (; i < list.length; i++) {
      const cand = [...row, list[i]];
      const s = cand.reduce((a, it) => a + it.value * scale, 0);
      const thick = s / side;
      let worst = 0;
      for (const it of cand) {
        const len = (it.value * scale) / thick;
        worst = Math.max(worst, thick / len, len / thick);
      }
      if (worst > best && row.length) break;
      best = worst;
      row = cand;
    }
    const s = row.reduce((a, it) => a + it.value * scale, 0);
    const thick = s / side;
    let off = 0;
    for (const it of row) {
      const len = (it.value * scale) / thick;
      out.push(horiz
        ? { ...it, x: cx, y: cy + off, w: thick, h: len }
        : { ...it, x: cx + off, y: cy, w: len, h: thick });
      off += len;
    }
    if (horiz) { cx += thick; cw -= thick; } else { cy += thick; ch -= thick; }
    list = list.slice(row.length);
  }
  return out;
}

export function buildTreemap(store: Store, X: number, Y: number): {
  root: HTMLElement; rect: { x: number; y: number; w: number; h: number };
} {
  const m = store.model;
  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `
    <div class="section-tag mono">${tr('sec.treemap.tag')}</div>
    <div class="section-title">${tr('sec.treemap.title')}</div>
    <div class="section-sub">${tr('sec.treemap.sub')}</div>`));

  const card = el('div', 'card', `position:relative;width:${W}px;height:${H}px;padding:18px`);
  root.appendChild(card);

  const inner = el('div', 'no-pan', `position:relative;width:${W - 36}px;height:${H - 76}px`);
  card.appendChild(inner);

  // группы → крупные тензоры + «мелочь» одной плиткой
  const groupItems: Item[] = m.groups.map(g => ({ label: g.label, value: g.params, t: undefined, tiny: undefined }));
  const placedGroups = squarify(groupItems, 0, 0, W - 36, H - 76);

  const cellRefs: { div: HTMLElement; t: Tensor }[] = [];

  for (const pg of placedGroups) {
    const g = m.groups.find(x => x.label === pg.label)!;
    const k = kindOf(g.kind);
    const box = el('div', '', `position:absolute;left:${pg.x}px;top:${pg.y}px;width:${pg.w - 4}px;height:${pg.h - 4}px;
      border:1.5px solid ${k.bd};border-radius:6px;overflow:hidden;background:rgba(255,253,248,0.4)`);
    inner.appendChild(box);

    const title = el('div', 'mono', `position:absolute;left:7px;top:5px;z-index:4;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
      color:${k.fg};background:rgba(253,250,244,0.85);padding:2px 7px;border-radius:4px;pointer-events:none;white-space:nowrap`);
    title.textContent = `${g.label} · ${fmtN(g.params)}`;
    box.appendChild(title);

    // внутри группы: тензоры крупнее порога — отдельные плитки, остальное — одна
    const MIN = g.params / 3000;
    const bigs = g.tensors.filter(t => t.numel >= MIN && t.is2d);
    const rest = g.tensors.filter(t => !(t.numel >= MIN && t.is2d));
    const items: Item[] = bigs.map(t => ({ t, label: t.short, value: t.numel }));
    const restSum = rest.reduce((a, t) => a + t.numel, 0);
    if (restSum > 0) items.push({ label: tr('treemap.tiny', rest.length), value: Math.max(restSum, g.params / 400), tiny: rest });

    const placed = squarify(items, 0, 0, pg.w - 7, pg.h - 7);
    for (const p of placed) {
      if (p.w < 2 || p.h < 2) continue;
      const isTiny = !!p.tiny;
      const cell = el('div', 'wall-cell', `position:absolute;left:${p.x + 1.5}px;top:${p.y + 1.5}px;width:${Math.max(1, p.w - 2.4)}px;height:${Math.max(1, p.h - 2.4)}px;
        border-radius:2px;background:${isTiny ? HATCH : colorForTensor(p.t!, store.md)};border:1px solid rgba(80,70,55,0.12)`);
      if (p.w > 70 && p.h > 15) {
        const lbl = el('div', 'mono', `position:absolute;left:4px;top:2px;right:4px;font-size:9.5px;color:rgba(50,44,36,0.6);
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none`);
        lbl.textContent = isTiny ? p.label : p.t!.short.split('.').slice(-2).join('.');
        cell.appendChild(lbl);
      }
      if (!isTiny) {
        cell.addEventListener('click', () => store.select({ type: 'tensor', tensor: p.t! }));
        cell.addEventListener('mousemove', e => showTensorTip(store, p.t!, e as MouseEvent));
        cell.addEventListener('mouseleave', hideTip);
        cellRefs.push({ div: cell, t: p.t! });
      } else {
        cell.title = `${p.label} · ${tr('treemap.together')} ${fmtN(p.tiny!.reduce((a, t) => a + t.numel, 0))}`;
      }
      box.appendChild(cell);
    }
  }

  store.addEventListener('metric', () => {
    for (const { div, t } of cellRefs) div.style.background = colorForTensor(t, store.md);
  });

  return { root, rect: { x: X, y: Y, w: W, h: H + 40 } };
}
