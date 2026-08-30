import type { GroupInfo, Layer, Model, Tensor } from './data';

export type Sel =
  | { type: 'model' }
  | { type: 'tensor'; tensor: Tensor }
  | { type: 'layer'; layer: Layer }
  | { type: 'group'; group: GroupInfo };

export class Store extends EventTarget {
  model: Model;
  metric = 'int4';
  scale: 'rel' | 'abs' = 'rel';
  sel: Sel = { type: 'model' };
  // мировые координаты клетки каждого тензора (заполняет стена)
  cellRect = new Map<number, { x: number; y: number; w: number; h: number }>();

  constructor(model: Model) {
    super();
    this.model = model;
  }

  get md() {
    const d = this.model.metrics[this.metric];
    if (this.scale === 'abs' && d.abs) {
      return { ...d, lo: d.abs[0], hi: d.abs[1], loT: d.abs[0] + ' ' + d.unit, hiT: d.abs[1] + ' ' + d.unit };
    }
    return d;
  }

  setMetric(k: string) {
    if (this.metric === k) return;
    this.metric = k;
    this.dispatchEvent(new Event('metric'));
  }

  setScale(s: 'rel' | 'abs') {
    if (this.scale === s) return;
    this.scale = s;
    this.dispatchEvent(new Event('metric'));
  }

  select(sel: Sel) {
    this.sel = sel;
    this.dispatchEvent(new Event('sel'));
  }
}

// ── общий тултип ──
let tipEl: HTMLDivElement | null = null;
export function tip(html: string, x: number, y: number) {
  if (!tipEl) {
    tipEl = document.createElement('div');
    tipEl.className = 'tip';
    document.body.appendChild(tipEl);
  }
  tipEl.innerHTML = html;
  tipEl.classList.add('show');
  const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  const px = Math.min(x + 16, window.innerWidth - w - 12);
  const py = y + 18 + h > window.innerHeight - 10 ? y - h - 12 : y + 18;
  tipEl.style.left = px + 'px';
  tipEl.style.top = py + 'px';
}
export function hideTip() { tipEl?.classList.remove('show'); }
document.addEventListener('click', hideTip, true);
