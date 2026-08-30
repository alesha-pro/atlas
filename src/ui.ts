// Оверлеи поверх полотна: шапка с метриками и поиском, туры, зум, миникарта.
import { fmtN, type ManifestEntry, type Tensor } from './data';
import { rampCSS, HATCH } from './color';
import { lang, t, type Lang } from './i18n';
import { el, type Rect, type World } from './world';
import type { Store } from './store';

export interface Tour { label: string; rect: Rect | null; } // null = всё полотно

export function buildTopbar(
  store: Store, world: World, manifest: ManifestEntry[], current: string,
  onModelChange: (slug: string) => void, onFlyTensor: (t: Tensor) => void,
  onLangChange: (l: Lang) => void,
): HTMLElement {
  const bar = el('div', 'overlay no-pan topbar', 'left:22px;top:18px;right:22px;display:flex;align-items:center;gap:12px;row-gap:6px;padding:10px 16px;z-index:20;flex-wrap:wrap');

  // титул
  const title = el('div', '', 'display:flex;flex-direction:column;gap:1px;margin-right:6px');
  title.innerHTML = `
    <div class="mono" style="font-size:10px;letter-spacing:0.2em;text-transform:uppercase;color:var(--accent-deep)">${t('topbar.eyebrow')}</div>
    <div style="font-size:17px;line-height:1">${store.model.name}</div>`;
  bar.appendChild(title);

  // выбор модели — виден всегда, даже с одной моделью
  const sel = document.createElement('select');
  sel.className = 'plain';
  for (const e of manifest) {
    const o = document.createElement('option');
    o.value = e.slug; o.textContent = e.name;
    sel.appendChild(o);
  }
  sel.value = current;
  sel.addEventListener('change', () => onModelChange(sel.value));
  bar.appendChild(sel);

  bar.appendChild(el('div', '', 'width:1px;height:26px;background:var(--line-strong)'));

  // метрики
  const tabs = el('div', '', 'display:flex;gap:6px;flex-wrap:wrap');
  const chipEls = new Map<string, HTMLElement>();
  const shown = ['int4', 'int8', 'fp8', 'kurt', 'hot', 'srank', 'size'];
  for (const key of shown) {
    const d = store.model.metrics[key];
    if (!d) continue;
    const c = el('div', 'chip mini' + (store.metric === key ? ' on' : ''), '', d.label);
    c.addEventListener('click', () => store.setMetric(key));
    tabs.appendChild(c);
    chipEls.set(key, c);
  }
  bar.appendChild(tabs);
  store.addEventListener('metric', () => {
    for (const [k, c] of chipEls) c.classList.toggle('on', store.metric === k);
    legendLo.textContent = store.md.loT;
    legendHi.textContent = store.md.hiT;
  });

  // шкала: относительная / абсолютная (для SQNR-метрик)
  const scaleWrap = el('div', '', 'display:flex;gap:4px;margin-left:2px');
  const scaleChips = new Map<string, HTMLElement>();
  for (const mode of ['rel', 'abs'] as const) {
    const c = el('div', 'chip mini' + (store.scale === mode ? ' on' : ''), '', t('scale.' + mode));
    c.title = t('scale.' + mode + '.tip');
    c.addEventListener('click', () => store.setScale(mode));
    scaleWrap.appendChild(c);
    scaleChips.set(mode, c);
  }
  bar.appendChild(scaleWrap);
  const syncScale = () => {
    for (const [m, c] of scaleChips) c.classList.toggle('on', store.scale === m);
    const has = !!store.model.metrics[store.metric].abs;
    scaleWrap.style.opacity = has ? '1' : '0.35';
    scaleWrap.style.pointerEvents = has ? 'auto' : 'none';
  };
  store.addEventListener('metric', syncScale);
  syncScale();

  // легенда
  const legend = el('div', 'tb-legend', 'display:flex;align-items:center;gap:8px;margin-left:4px');
  const legendLo = el('div', 'mono', 'font-size:11px;color:var(--faint);white-space:nowrap', store.md.loT);
  const legendHi = el('div', 'mono', 'font-size:11px;color:var(--faint);white-space:nowrap', store.md.hiT);
  legend.appendChild(legendLo);
  legend.appendChild(el('div', '', `width:130px;height:9px;border-radius:5px;background:${rampCSS()};border:1px solid rgba(0,0,0,0.06)`));
  legend.appendChild(legendHi);
  const na = el('div', '', 'display:flex;align-items:center;gap:5px;margin-left:6px');
  na.appendChild(el('div', '', `width:12px;height:12px;border-radius:2px;background:${HATCH};border:1px solid var(--card-bd)`));
  na.appendChild(el('div', 'mono', 'font-size:10.5px;color:var(--faint)', t('na')));
  legend.appendChild(na);
  bar.appendChild(legend);

  bar.appendChild(el('div', '', 'flex:1'));

  // тема: переключение живое, все цвета на CSS-переменных
  const themeChip = el('div', 'chip mini', 'margin-right:2px');
  const syncTheme = () => {
    themeChip.textContent = document.documentElement.dataset.theme === 'dark' ? '☀' : '☾';
  };
  themeChip.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('atlas.theme', next);
    syncTheme();
  });
  syncTheme();
  bar.appendChild(themeChip);

  // язык
  const langWrap = el('div', '', 'display:flex;gap:4px;margin-right:4px');
  for (const l of ['en', 'ru'] as Lang[]) {
    const c = el('div', 'chip mini' + (lang === l ? ' on' : ''), '', l.toUpperCase());
    c.addEventListener('click', () => { if (lang !== l) onLangChange(l); });
    langWrap.appendChild(c);
  }
  bar.appendChild(langWrap);

  // поиск
  bar.appendChild(buildSearch(store, onFlyTensor));

  return bar;
}

function buildSearch(store: Store, onFlyTensor: (t: Tensor) => void): HTMLElement {
  const wrap = el('div', 'search-box', 'position:relative;display:flex;align-items:center;gap:6px;border:1px solid var(--line-strong);border-radius:10px;padding:0 10px;background:var(--glass)');
  wrap.innerHTML = `<span class="mono" style="color:var(--ghost);font-size:13px">⌕</span>`;
  const input = document.createElement('input');
  input.placeholder = t('search.placeholder');
  input.spellcheck = false;
  wrap.appendChild(input);
  const drop = el('div', 'search-drop', 'display:none');
  wrap.appendChild(drop);

  const close = () => { drop.style.display = 'none'; };
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    drop.innerHTML = '';
    if (q.length < 2) { close(); return; }
    const hits = store.model.tensors.filter(t => t.name.toLowerCase().includes(q)).slice(0, 14);
    if (!hits.length) { close(); return; }
    for (const x of hits) {
      const item = el('div', 'search-item');
      const v = store.md.get(x);
      item.innerHTML = `
        <div class="mono" style="flex:1;font-size:11.5px;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${x.short}</div>
        <div class="mono" style="font-size:11px;color:var(--ghost)">${v == null ? t('na') : store.md.fmt(v)}</div>`;
      item.addEventListener('click', () => {
        store.select({ type: 'tensor', tensor: x });
        onFlyTensor(x);
        close();
        input.value = '';
      });
      drop.appendChild(item);
    }
    drop.style.display = 'flex';
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') (drop.firstElementChild as HTMLElement)?.click();
    if (e.key === 'Escape') { close(); input.blur(); }
  });
  document.addEventListener('click', e => { if (!wrap.contains(e.target as Node)) close(); });
  return wrap;
}

export function buildBottombar(world: World, tours: Tour[], panelPad: () => number): HTMLElement {
  const bar = el('div', 'overlay no-pan bottombar', 'left:22px;bottom:22px;display:flex;align-items:center;gap:10px;padding:9px 13px;z-index:20');
  const zo = el('div', 'zbtn', '', '−');
  const zl = el('div', 'mono', 'font-size:12.5px;color:var(--muted);width:46px;text-align:center', '100%');
  const zi = el('div', 'zbtn', '', '+');
  zo.addEventListener('click', () => world.zoomBy(1 / 1.3));
  zi.addEventListener('click', () => world.zoomBy(1.3));
  bar.append(zo, zl, zi);
  bar.appendChild(el('div', '', 'width:1px;height:22px;background:var(--line)'));
  for (const t of tours) {
    const c = el('div', 'chip mini', '', t.label);
    c.addEventListener('click', () => {
      if (t.rect) world.flyTo(t.rect, { padRight: panelPad() });
      else world.fitAll(panelPad());
    });
    bar.appendChild(c);
  }
  const prev = world.onChange;
  world.onChange = () => { prev?.(); zl.textContent = Math.round(world.zoom * 100) + '%'; };
  return bar;
}

export function buildMinimap(world: World, sections: { rect: Rect; color: string }[]): HTMLElement {
  const MW = 236;
  const scale = MW / world.size.w;
  const MH = Math.round(world.size.h * scale);
  const mm = el('div', 'overlay minimap no-pan', `width:${MW}px;height:${MH}px;z-index:20`);
  for (const s of sections) {
    mm.appendChild(el('div', 'mm-sec', `left:${s.rect.x * scale}px;top:${s.rect.y * scale}px;width:${s.rect.w * scale}px;height:${s.rect.h * scale}px;background:${s.color}`));
  }
  const view = el('div', 'mm-view');
  mm.appendChild(view);

  const update = () => {
    const v = world.viewport();
    view.style.left = Math.max(0, v.x * scale) + 'px';
    view.style.top = Math.max(0, v.y * scale) + 'px';
    view.style.width = Math.min(MW, v.w * scale) + 'px';
    view.style.height = Math.min(MH, v.h * scale) + 'px';
  };
  const prev = world.onChange;
  world.onChange = () => { prev?.(); update(); };
  update();

  const jump = (e: MouseEvent) => {
    const r = mm.getBoundingClientRect();
    const wx = (e.clientX - r.left) / scale, wy = (e.clientY - r.top) / scale;
    world.pan.x = world.board.clientWidth / 2 - wx * world.zoom;
    world.pan.y = world.board.clientHeight / 2 - wy * world.zoom;
    world.apply();
  };
  let down = false;
  mm.addEventListener('mousedown', e => { down = true; jump(e); e.stopPropagation(); });
  window.addEventListener('mousemove', e => { if (down) jump(e); });
  window.addEventListener('mouseup', () => { down = false; });
  return mm;
}
