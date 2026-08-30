// Разбор модели: паспорт, схемы блоков, статьи. Контент приходит из
// public/models/<slug>/dossier.json — код рендерит любые блоки, ничего не зная
// о конкретной архитектуре.
import type { Tensor } from '../data';
import { t as tr, lang } from '../i18n';
import { kindOf } from '../color';
import { el, svgEl } from '../world';
import { buildWidget } from '../widgets';
import type { Store } from '../store';

const W = 2480, CARD_W = 2480, INNER = CARD_W - 70;

type L10n = string | { en: string; ru: string };
const pick = (s: L10n | undefined): string =>
  s == null ? '' : typeof s === 'string' ? s : (s[lang] ?? s.en);

export function buildDossier(
  store: Store, dossier: any, X: number, Y: number,
  flyToTensor: (t: Tensor) => void,
): { root: HTMLElement; rect: { x: number; y: number; w: number; h: number } } {
  const root = el('div', 'section', `left:${X}px;top:${Y}px;width:${W}px`);
  root.appendChild(el('div', 'section-head', '', `
    <div class="section-tag mono">${tr('sec.dossier.tag')}</div>
    <div class="section-title">${tr('sec.dossier.title')}</div>
    <div class="section-sub">${tr('sec.dossier.sub')}</div>`));

  const col = el('div', '', 'display:flex;flex-direction:column;gap:22px');
  root.appendChild(col);

  for (const b of dossier.blocks || []) {
    if (b.type === 'intro') col.appendChild(introBlock(b));
    else if (b.type === 'facts') col.appendChild(factsBlock(b));
    else if (b.type === 'pattern') col.appendChild(patternBlock(store, b));
    else if (b.type === 'diagram') col.appendChild(diagramBlock(store, b, flyToTensor));
    else if (b.type === 'papers') col.appendChild(papersBlock(b));
  }

  return { root, rect: { x: X, y: Y, w: W, h: 4650 } };
}

function blockHead(title: string, extra = ''): string {
  return `<div style="display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:10px">
    <div style="font-size:23px">${title}</div>${extra}</div>`;
}

function introBlock(b: any): HTMLElement {
  const card = el('div', 'card', `width:${CARD_W}px;padding:24px 32px`);
  card.innerHTML = blockHead(pick(b.title)) +
    `<div class="note" style="font-size:16px;max-width:2100px">${pick(b.text)}</div>`;
  return card;
}

function factsBlock(b: any): HTMLElement {
  const card = el('div', 'card', `width:${CARD_W}px;padding:24px 32px`);
  const rows = (b.rows as [L10n, L10n][]).map(([k, v]) => `
    <div style="display:flex;gap:18px;padding:8px 0;border-top:1px solid var(--line);align-items:baseline">
      <div class="mono" style="width:260px;flex-shrink:0;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;color:var(--faint)">${pick(k)}</div>
      <div class="mono" style="font-size:14.5px;color:var(--ink-soft)">${pick(v)}</div>
    </div>`).join('');
  card.innerHTML = blockHead(pick(b.title)) + `<div style="columns:2;column-gap:48px">${rows}</div>`;
  return card;
}

// ── всплывающая карточка «как это работает» ──
function openDetail(title: string, subtitle: string, detail: any) {
  const app = document.getElementById('app')!;
  const veil = el('div', 'veil no-pan');
  const modal = el('div', 'modal');
  modal.innerHTML = `
    <div class="modal-head">
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
        <div>
          <div class="eyebrow accent mono">${subtitle}</div>
          <div class="mono" style="font-size:24px;line-height:1.2;margin-top:7px">${title}</div>
        </div>
        <div class="mono close-x" style="cursor:pointer;color:var(--ghost);font-size:16px;padding:2px 6px">✕</div>
      </div>
    </div>`;
  const body = el('div', 'modal-body');
  body.appendChild(el('div', 'note', 'font-size:15.5px;line-height:1.6', pick(detail.text)));
  let widget: HTMLElement | null = null;
  if (detail.widget) {
    widget = buildWidget(detail.widget);
    if (widget) body.appendChild(widget);
  }
  modal.appendChild(body);
  veil.appendChild(modal);
  app.appendChild(veil);

  const close = () => {
    (widget as any)?.cleanup?.();
    veil.remove();
    window.removeEventListener('keydown', onKey, true);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') { e.stopPropagation(); close(); }
  };
  window.addEventListener('keydown', onKey, true);
  veil.addEventListener('click', e => { if (e.target === veil) close(); });
  veil.addEventListener('mousedown', e => e.stopPropagation());
  modal.querySelector('.close-x')!.addEventListener('click', close);
}

function patternBlock(store: Store, b: any): HTMLElement {
  const card = el('div', 'card', `width:${CARD_W}px;padding:24px 32px`);
  const lin = kindOf('lin'), attn = kindOf('attn');
  // 16 групп по 3+1, живые цвета пород
  let strip = `<div style="display:flex;align-items:flex-end;gap:5px;margin:14px 0 6px;flex-wrap:nowrap">`;
  strip += `<div class="mono" style="font-size:10px;color:var(--ghost);margin-right:8px">embed</div>`;
  for (let g = 0; g < 16; g++) {
    strip += `<div style="display:flex;gap:2.5px;padding:4px;border:1px solid var(--line);border-radius:4px;background:var(--glass-soft)">`;
    for (let i = 0; i < 3; i++) strip += `<div style="width:26px;height:40px;border-radius:2px;background:${lin.solid};opacity:0.75"></div>`;
    strip += `<div style="width:26px;height:52px;border-radius:2px;background:${attn.solid}"></div></div>`;
  }
  strip += `<div class="mono" style="font-size:10px;color:var(--ghost);margin-left:8px">lm_head</div></div>`;
  strip += `<div class="mono" style="display:flex;gap:22px;font-size:11.5px;color:var(--faint);margin-bottom:12px">
    <div style="display:flex;align-items:center;gap:6px"><div style="width:9px;height:14px;border-radius:2px;background:${lin.solid}"></div>Gated DeltaNet → FFN · ×48</div>
    <div style="display:flex;align-items:center;gap:6px"><div style="width:9px;height:18px;border-radius:2px;background:${attn.solid}"></div>Gated Attention → FFN · ×16</div>
  </div>`;
  const howChip = b.detail ? `<div class="chip mini no-pan" data-how>${tr('dossier.how')}</div>` : '';
  card.innerHTML = blockHead(pick(b.title), howChip) + strip +
    `<div class="note" style="max-width:2100px">${pick(b.note)}</div>`;
  card.querySelector('[data-how]')?.addEventListener('click', () =>
    openDetail(pick(b.title), tr('sec.dossier.title'), b.detail));
  return card;
}

function diagramBlock(store: Store, b: any, flyToTensor: (t: Tensor) => void): HTMLElement {
  const card = el('div', 'card', `width:${CARD_W}px;padding:24px 32px 26px`);
  const H = b.h || 500;

  const jumpChip = b.jump ? `<div class="chip mini no-pan" data-jump>${tr('dossier.jump')}</div>` : '';
  card.innerHTML = blockHead(pick(b.title), jumpChip);

  const area = el('div', '', `position:relative;width:${INNER}px;height:${H}px`);
  card.appendChild(area);
  const svg = svgEl('svg', { width: INNER, height: H, style: 'position:absolute;left:0;top:0;pointer-events:none;overflow:visible' });
  area.appendChild(svg as any);

  // узлы
  const geom = new Map<string, { x: number; y: number; w: number; h: number }>();
  for (const n of b.nodes) {
    const k = kindOf(n.kind || 'norm');
    const node = el('div', n.detail ? 'node-click no-pan' : '', `position:absolute;left:${n.x}px;top:${n.y}px;width:${n.w}px;
      background:var(--glass);border:1px solid ${k.bd};border-radius:12px;padding:12px 14px;
      box-shadow:0 10px 24px -20px rgba(60,50,35,0.5)`);
    node.innerHTML = `
      <div class="mono" style="font-size:13.5px;color:${k.fg};line-height:1.3">${pick(n.label)}</div>
      ${n.sub ? `<div style="font-size:12.5px;color:var(--faint);line-height:1.35;margin-top:4px;text-wrap:pretty">${pick(n.sub)}</div>` : ''}`;
    if (n.detail) {
      node.appendChild(el('div', 'node-more', '', '+'));
      node.addEventListener('click', () => openDetail(pick(n.label), pick(b.title), n.detail));
    }
    area.appendChild(node);
    geom.set(n.id, { x: n.x, y: n.y, w: n.w, h: 0 });
  }

  // рёбра после измерения высот
  requestAnimationFrame(() => {
    const kids = area.querySelectorAll<HTMLElement>(':scope > div');
    b.nodes.forEach((n: any, i: number) => { geom.get(n.id)!.h = kids[i].offsetHeight; });
    for (const e of b.edges || []) {
      const [a, bId, dashed] = e;
      const A = geom.get(a), B = geom.get(bId);
      if (!A || !B) continue;
      const acx = A.x + A.w / 2, acy = A.y + A.h / 2;
      const bcx = B.x + B.w / 2, bcy = B.y + B.h / 2;
      let d: string;
      if (Math.abs(bcx - acx) >= Math.abs(bcy - acy)) {
        const ltr = bcx >= acx;
        const x1 = ltr ? A.x + A.w : A.x, x2 = ltr ? B.x : B.x + B.w;
        const dx = Math.max(30, Math.abs(x2 - x1) * 0.45) * (ltr ? 1 : -1);
        d = `M ${x1} ${acy} C ${x1 + dx} ${acy}, ${x2 - dx} ${bcy}, ${x2} ${bcy}`;
      } else {
        const ttb = bcy >= acy;
        const y1 = ttb ? A.y + A.h : A.y, y2 = ttb ? B.y : B.y + B.h;
        const dy = Math.max(24, Math.abs(y2 - y1) * 0.45) * (ttb ? 1 : -1);
        d = `M ${acx} ${y1} C ${acx} ${y1 + dy}, ${bcx} ${y2 - dy}, ${bcx} ${y2}`;
      }
      const p = svgEl('path', {
        d, fill: 'none',
        stroke: dashed ? 'rgba(150,118,76,0.38)' : 'rgba(150,118,76,0.6)',
        'stroke-width': dashed ? 1.6 : 2.2, 'stroke-linecap': 'round',
      });
      if (dashed) p.setAttribute('stroke-dasharray', '7 7');
      svg.appendChild(p);
    }
  });

  const note = el('div', 'note', 'max-width:2100px;margin-top:14px');
  note.innerHTML = pick(b.note);
  card.appendChild(note);

  card.querySelector('[data-jump]')?.addEventListener('click', () => {
    const t0 = store.model.tensors.find(x => x.name.includes(b.jump));
    if (t0) { store.select({ type: 'tensor', tensor: t0 }); flyToTensor(t0); }
  });
  return card;
}

function papersBlock(b: any): HTMLElement {
  const card = el('div', 'card', `width:${CARD_W}px;padding:24px 32px`);
  const items = (b.items || []).map((p: any) => `
    <a href="${p.url}" target="_blank" rel="noopener" class="no-pan" style="display:flex;flex-direction:column;gap:5px;
      border:1px solid var(--line);background:var(--glass-soft);border-radius:10px;padding:14px 16px;
      text-decoration:none;color:inherit">
      <div class="mono" style="font-size:10.5px;letter-spacing:0.1em;color:var(--accent-deep)">${p.id}</div>
      <div style="font-size:16.5px;line-height:1.25;color:var(--ink)">${p.title}</div>
      <div class="small-note" style="font-size:12.5px">${pick(p.why)}</div>
    </a>`).join('');
  card.innerHTML = blockHead(pick(b.title)) +
    `<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">${items}</div>`;
  return card;
}
