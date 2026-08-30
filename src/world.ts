// Движок полотна: панорамирование, зум к курсору, плавный перелёт, миникарта.

export interface Rect { x: number; y: number; w: number; h: number; }

export class World {
  board: HTMLDivElement;
  world: HTMLDivElement;
  pan = { x: 0, y: 0 };
  zoom = 0.5;
  size: { w: number; h: number };
  private drag: { mx: number; my: number; px: number; py: number } | null = null;
  private anim: { x: number; y: number; z: number } | null = null;
  private raf = 0;
  moved = false;
  onChange: (() => void) | null = null;

  constructor(parent: HTMLElement, size: { w: number; h: number }) {
    this.size = size;
    this.board = document.createElement('div');
    this.board.className = 'board';
    this.world = document.createElement('div');
    this.world.className = 'world';
    this.world.style.width = size.w + 'px';
    this.world.style.height = size.h + 'px';
    this.board.appendChild(this.world);
    parent.appendChild(this.board);

    this.board.style.touchAction = 'none';

    // клик после реального пана не должен ничего выбирать
    this.board.addEventListener('click', e => {
      if (this.moved) { e.stopPropagation(); e.preventDefault(); }
    }, true);

    // ── тач: один палец пан, два пальца пинч-зум ──
    let ts: { mode: 'pan' | 'zoom'; sx: number; sy: number; px: number; py: number;
      dist: number; midX: number; midY: number; zoom0: number } | null = null;
    const dist2 = (e: TouchEvent) => Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY);
    const mid2 = (e: TouchEvent) => {
      const r = this.board.getBoundingClientRect();
      return {
        x: (e.touches[0].clientX + e.touches[1].clientX) / 2 - r.left,
        y: (e.touches[0].clientY + e.touches[1].clientY) / 2 - r.top,
      };
    };
    this.board.addEventListener('touchstart', e => {
      this.anim = null;
      if (e.touches.length === 1) {
        const t0 = e.touches[0];
        ts = { mode: 'pan', sx: t0.clientX, sy: t0.clientY, px: this.pan.x, py: this.pan.y,
          dist: 0, midX: 0, midY: 0, zoom0: this.zoom };
        this.moved = false;
      } else if (e.touches.length >= 2) {
        const m = mid2(e);
        ts = { mode: 'zoom', sx: 0, sy: 0, px: this.pan.x, py: this.pan.y,
          dist: dist2(e), midX: m.x, midY: m.y, zoom0: this.zoom };
        this.moved = true;
      }
    }, { passive: true });
    this.board.addEventListener('touchmove', e => {
      if (!ts) return;
      e.preventDefault();
      if (ts.mode === 'pan' && e.touches.length === 1) {
        const t0 = e.touches[0];
        const dx = t0.clientX - ts.sx, dy = t0.clientY - ts.sy;
        if (Math.abs(dx) + Math.abs(dy) > 8) this.moved = true;
        this.pan.x = ts.px + dx;
        this.pan.y = ts.py + dy;
        this.apply();
      } else if (e.touches.length >= 2) {
        if (ts.mode !== 'zoom') {
          const m0 = mid2(e);
          ts = { mode: 'zoom', sx: 0, sy: 0, px: this.pan.x, py: this.pan.y,
            dist: dist2(e), midX: m0.x, midY: m0.y, zoom0: this.zoom };
        }
        const m = mid2(e);
        const z2 = Math.max(0.05, Math.min(2.6, ts.zoom0 * dist2(e) / ts.dist));
        this.pan.x = m.x - (ts.midX - ts.px) * (z2 / ts.zoom0);
        this.pan.y = m.y - (ts.midY - ts.py) * (z2 / ts.zoom0);
        this.zoom = z2;
        this.moved = true;
        this.apply();
      }
    }, { passive: false });
    this.board.addEventListener('touchend', e => {
      if (e.touches.length === 0) {
        ts = null;
        setTimeout(() => { this.moved = false; }, 0);
      } else if (e.touches.length === 1 && ts) {
        const t0 = e.touches[0];
        ts = { mode: 'pan', sx: t0.clientX, sy: t0.clientY, px: this.pan.x, py: this.pan.y,
          dist: 0, midX: 0, midY: 0, zoom0: this.zoom };
      }
    });

    this.board.addEventListener('mousedown', e => {
      if ((e.target as HTMLElement).closest('.no-pan')) return;
      this.drag = { mx: e.clientX, my: e.clientY, px: this.pan.x, py: this.pan.y };
      this.moved = false;
      this.anim = null;
      this.board.classList.add('dragging');
    });
    window.addEventListener('mousemove', e => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.mx, dy = e.clientY - this.drag.my;
      if (Math.abs(dx) + Math.abs(dy) > 4) this.moved = true;
      this.pan.x = this.drag.px + dx;
      this.pan.y = this.drag.py + dy;
      this.apply();
    });
    window.addEventListener('mouseup', () => {
      if (this.drag) { this.drag = null; this.board.classList.remove('dragging'); }
      setTimeout(() => { this.moved = false; }, 0);
    });
    this.board.addEventListener('wheel', e => {
      e.preventDefault();
      this.anim = null;
      const r = this.board.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      if (e.ctrlKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        const z2 = Math.max(0.08, Math.min(2.6, this.zoom * Math.exp(-e.deltaY * (e.ctrlKey ? 0.008 : 0.0016))));
        this.pan.x = mx - (mx - this.pan.x) * (z2 / this.zoom);
        this.pan.y = my - (my - this.pan.y) * (z2 / this.zoom);
        this.zoom = z2;
      } else {
        this.pan.x -= e.deltaX;
      }
      this.apply();
    }, { passive: false });
  }

  apply() {
    this.world.style.transform = `translate(${this.pan.x.toFixed(1)}px,${this.pan.y.toFixed(1)}px) scale(${this.zoom.toFixed(4)})`;
    const s = (26 * this.zoom).toFixed(2);
    this.board.style.backgroundSize = `${s}px ${s}px`;
    this.board.style.backgroundPosition = `${this.pan.x.toFixed(1)}px ${this.pan.y.toFixed(1)}px`;
    this.onChange?.();
  }

  viewport(): Rect {
    const W = this.board.clientWidth, H = this.board.clientHeight;
    return { x: -this.pan.x / this.zoom, y: -this.pan.y / this.zoom, w: W / this.zoom, h: H / this.zoom };
  }

  flyTo(rect: Rect, opts: { padRight?: number; maxZoom?: number } = {}) {
    const W = this.board.clientWidth, H = this.board.clientHeight;
    const padR = opts.padRight ?? 0;
    const availW = W - padR - 60, availH = H - 140;
    const z = Math.min(opts.maxZoom ?? 1.1, availW / (rect.w + 60), availH / (rect.h + 60));
    const target = {
      x: 30 + (availW - rect.w * z) / 2 - rect.x * z,
      y: 100 + (availH - rect.h * z) / 2 - rect.y * z,
      z,
    };
    this.animateTo(target);
  }

  fitAll(padRight = 0) {
    this.flyTo({ x: 0, y: 0, w: this.size.w, h: this.size.h }, { padRight, maxZoom: 1 });
  }

  private animateTo(target: { x: number; y: number; z: number }) {
    this.anim = target;
    cancelAnimationFrame(this.raf);
    const step = () => {
      if (!this.anim) return;
      const t = this.anim;
      const nx = this.pan.x + (t.x - this.pan.x) * 0.17;
      const ny = this.pan.y + (t.y - this.pan.y) * 0.17;
      const nz = this.zoom + (t.z - this.zoom) * 0.17;
      const done = Math.abs(t.x - nx) < 0.5 && Math.abs(t.y - ny) < 0.5 && Math.abs(t.z - nz) < 0.0015;
      this.pan.x = done ? t.x : nx;
      this.pan.y = done ? t.y : ny;
      this.zoom = done ? t.z : nz;
      this.apply();
      if (done) { this.anim = null; return; }
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  zoomBy(f: number) {
    this.anim = null;
    const W = this.board.clientWidth / 2, H = this.board.clientHeight / 2;
    const z2 = Math.max(0.08, Math.min(2.6, this.zoom * f));
    this.pan.x = W - (W - this.pan.x) * (z2 / this.zoom);
    this.pan.y = H - (H - this.pan.y) * (z2 / this.zoom);
    this.zoom = z2;
    this.apply();
  }
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K, cls = '', style = '', html = '',
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (style) e.style.cssText = style;
  if (html) e.innerHTML = html;
  return e;
}

export function svgEl(tag: string, attrs: Record<string, string | number> = {}): SVGElement {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e as SVGElement;
}
