import './style.css';
import { loadDossier, loadInsights, loadLive, loadManifest, loadModel, type Tensor } from './data';
import { buildLive, type LiveSection } from './sections/live';
import { Store } from './store';
import { World, el } from './world';
import { buildIntro } from './sections/intro';
import { buildArch } from './sections/arch';
import { buildWall } from './sections/wall';
import { buildScatter } from './sections/scatter';
import { buildTreemap } from './sections/treemap';
import { buildRecords } from './sections/records';
import { buildDepth } from './sections/depth';
import { buildDossier } from './sections/dossier';
import { buildGlmInsights, type GlmSection } from './sections/glm';
import { buildPanel } from './panel';
import { buildTopbar, buildBottombar, buildMinimap, type Tour } from './ui';
import { kindOf } from './color';
import { setLang, t, type Lang } from './i18n';

const app = document.getElementById('app')!;

interface Keep { metric: string; pan: { x: number; y: number }; zoom: number; }

// модель в адресе: ?model=<slug>, чтобы ссылку можно было кидать напрямую
const urlSlug = () => new URLSearchParams(location.search).get('model') || undefined;
function writeUrl(slug: string, push = false) {
  const u = new URL(location.href);
  if (u.searchParams.get('model') === slug) return;
  u.searchParams.set('model', slug);
  push ? history.pushState(null, '', u) : history.replaceState(null, '', u);
}

async function boot(slug?: string, keep?: Keep) {
  slug = slug || urlSlug();
  app.innerHTML = '';
  document.title = t('title');
  document.documentElement.lang = localStorage.getItem('atlas.lang') === 'ru' ? 'ru' : 'en';
  const loading = el('div', 'mono', `position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    font-size:13px;color:var(--faint);letter-spacing:0.15em`, t('loading'));
  app.appendChild(loading);

  const manifest = await loadManifest();
  const entry = manifest.find(e => e.slug === slug) || manifest[0];
  writeUrl(entry.slug);
  const [model, dossier, live, insights] = await Promise.all([
    loadModel(entry), loadDossier(entry.slug), loadLive(entry.slug), loadInsights(entry.slug),
  ]);
  app.innerHTML = '';
  const isGlm = entry.kind === 'glm-live';

  const store = new Store(model);
  if (keep) store.metric = keep.metric;
  // scrollIntoView/focus умеют скроллить даже overflow:hidden — гасим
  app.addEventListener('scroll', () => { app.scrollTop = 0; app.scrollLeft = 0; });
  const world = new World(app, isGlm ? { w: 6000, h: 8200 } : dossier ? { w: 9260, h: 4950 } : { w: 6400, h: 3300 });
  (window as any).__world = world; // отладочный хук для скриншотов

  // цветовые пятна фона
  const washes: [number, number, number, number, string][] = [
    [200, 100, 1400, 900, 'rgba(206,228,224,0.5)'],
    [2400, 900, 1800, 1100, 'rgba(255,214,178,0.45)'],
    [3600, 100, 1500, 1000, 'rgba(219,208,236,0.42)'],
    [700, 2000, 1600, 1100, 'rgba(246,226,201,0.5)'],
    [4600, 1900, 1500, 1100, 'rgba(206,228,224,0.4)'],
  ];
  for (const [x, y, w, h, c] of washes) {
    world.world.appendChild(el('div', 'wash', `left:${x}px;top:${y}px;width:${w}px;height:${h}px;
      background:radial-gradient(ellipse at 50% 50%,${c},transparent 66%)`));
  }

  // ── секции ──
  world.world.appendChild(buildIntro(store, 80, 170));
  const arch = buildArch(store, 790, 150);
  world.world.appendChild(arch.root);
  let treemap: ReturnType<typeof buildTreemap> | null = null;
  let records: ReturnType<typeof buildRecords> | null = null;
  if (!isGlm) {
    treemap = buildTreemap(store, 3280, 150);
    world.world.appendChild(treemap.root);
    records = buildRecords(store, 4990, 150);
    world.world.appendChild(records.root);
  }
  const wall = buildWall(store, 80, 1890);
  world.world.appendChild(wall.root);
  let scatter: ReturnType<typeof buildScatter> | null = null;
  let depth: ReturnType<typeof buildDepth> | null = null;
  if (!isGlm) {
    scatter = buildScatter(store, wall.rect.x + wall.rect.w + 90, 1890);
    world.world.appendChild(scatter.root);
    depth = buildDepth(store, scatter.rect.x + scatter.rect.w + 90, 1890);
    world.world.appendChild(depth.root);
  }

  // ── панель и оверлеи ──
  const mobile = () => window.innerWidth < 760;
  const panelOpen = () => store.sel.type !== 'model';
  const panelPad = () => (panelOpen() && !mobile()) ? Math.min(430, window.innerWidth * 0.36) + 30 : 0;

  const flyToTensor = (t: Tensor) => {
    const r = store.cellRect.get(t.idx);
    if (r) world.flyTo({ x: r.x - 260, y: r.y - 200, w: r.w + 520, h: r.h + 400 }, { padRight: panelPad(), maxZoom: 1.6 });
  };

  // ── разбор модели (если у модели есть dossier.json) ──
  let dossierSec: ReturnType<typeof buildDossier> | null = null;
  if (dossier) {
    dossierSec = buildDossier(store, dossier, isGlm ? 3280 : 6560, 150, (x) => flyToTensor(x));
    world.world.appendChild(dossierSec.root);
    dossierSec.rect.h = dossierSec.root.offsetHeight + 40;
    world.world.appendChild(el('div', 'wash', `left:${isGlm ? 3420 : 6700}px;top:600px;width:2200px;height:1600px;
      background:radial-gradient(ellipse at 50% 50%,rgba(219,208,236,0.4),transparent 66%)`));
  }

  let glmSec: GlmSection | null = null;
  if (isGlm && insights) {
    const startY = Math.max(wall.rect.y + wall.rect.h, dossierSec ? dossierSec.rect.y + dossierSec.rect.h : 0) + 150;
    glmSec = buildGlmInsights(store, insights, 80, startY, (r) => world.flyTo(r, { padRight: panelPad(), maxZoom: 1 }));
    world.world.appendChild(glmSec.root);
    glmSec.rect.h = glmSec.root.offsetHeight + 40;
    const needH = glmSec.rect.y + glmSec.rect.h + 260;
    world.size.h = Math.max(world.size.h, needH);
    world.world.style.height = world.size.h + 'px';
    const needW = glmSec.rect.x + glmSec.rect.w + 120;
    if (needW > world.size.w) { world.size.w = needW; world.world.style.width = needW + 'px'; }
  }

  // ── живая модель (если есть live.json) ──
  let liveSec: LiveSection | null = null;
  if (live && !isGlm) {
    // строго ниже второго ряда, чтобы не лечь на стену и скаттер
    const row2Bottom = Math.max(
      wall.rect.y + wall.rect.h,
      scatter!.rect.y + scatter!.rect.h,
      depth!.rect.y + depth!.rect.h);
    liveSec = buildLive(store, live, 80, row2Bottom + 140);
    world.world.appendChild(liveSec.root);
    // регион уже в DOM: настоящие размеры известны до туров и миникарты
    liveSec.rect.h = liveSec.root.offsetHeight + 40;
    const liveCards = [...liveSec.root.querySelectorAll('.card')] as HTMLElement[];
    if (liveCards.length)
      liveSec.rect.w = Math.max(...liveCards.map(c => c.offsetLeft + c.offsetWidth)) + 60;
    world.world.appendChild(el('div', 'wash', `left:600px;top:${liveSec.rect.y + 200}px;width:3400px;height:${Math.max(600, liveSec.rect.h - 200)}px;
      background:radial-gradient(ellipse at 50% 50%,rgba(206,228,224,0.42),transparent 66%)`));
    const needH = liveSec.rect.y + liveSec.rect.h + 240;
    if (needH > world.size.h) {
      world.size.h = needH;
      world.world.style.height = needH + 'px';
    }
  }

  const keepNow = (): Keep => ({ metric: store.metric, pan: { ...world.pan }, zoom: world.zoom });
  const onLang = (l: Lang) => { setLang(l); boot(entry.slug, keepNow()); };
  app.appendChild(buildPanel(store, flyToTensor));
  app.appendChild(buildTopbar(store, world, manifest, entry.slug, s => { writeUrl(s, true); boot(s, keepNow()); }, flyToTensor, onLang));

  const introRect = { x: 60, y: 120, w: 700, h: 900 };
  const tours: Tour[] = [
    { label: t('tour.overview'), rect: null },
    { label: t('tour.intro'), rect: introRect },
    { label: t('tour.arch'), rect: arch.rect },
    { label: t('tour.wall'), rect: wall.rect },
  ];
  if (scatter) tours.push({ label: t('tour.scatter'), rect: scatter.rect });
  if (depth) tours.push({ label: t('tour.depth'), rect: depth.rect });
  if (treemap) tours.push({ label: t('tour.treemap'), rect: treemap.rect });
  if (records) tours.push({ label: t('tour.records'), rect: records.rect });
  if (dossierSec) tours.push({ label: t('tour.dossier'), rect: { ...dossierSec.rect, h: 1500 } });
  if (liveSec) tours.push({ label: t('tour.live'), rect: liveSec.rect });
  if (glmSec) tours.push({ label: t('tour.live'), rect: glmSec.rect });
  app.appendChild(buildBottombar(world, tours, panelPad));
  const minimap = buildMinimap(world, [
    { rect: introRect, color: 'var(--line-strong)' },
    { rect: arch.rect, color: kindOf('attn').solid },
    { rect: wall.rect, color: kindOf('lin').solid },
    ...(treemap ? [{ rect: treemap.rect, color: kindOf('mlp').solid }] : []),
    ...(records ? [{ rect: records.rect, color: kindOf('out').solid }] : []),
    ...(scatter ? [{ rect: scatter.rect, color: kindOf('vision').solid }] : []),
    ...(depth ? [{ rect: depth.rect, color: kindOf('norm').solid }] : []),
    ...(liveSec ? [{ rect: liveSec.rect, color: kindOf('attn').solid }] : []),
    ...(glmSec ? [{ rect: glmSec.rect, color: kindOf('attn').solid }] : []),
    ...(dossierSec ? [{ rect: dossierSec.rect, color: kindOf('in').bg }] : []),
  ]);
  app.appendChild(minimap);
  const placeMinimap = () => {
    minimap.style.right = (panelOpen() && !mobile()) ? `${Math.min(430, window.innerWidth * 0.36) + 44}px` : '22px';
  };
  store.addEventListener('sel', placeMinimap);
  placeMinimap();

  // стартовый вид
  if (keep) {
    world.pan = { ...keep.pan };
    world.zoom = keep.zoom;
    world.apply();
  } else {
    world.pan = { x: 40, y: 60 };
    world.zoom = 0.2;
    world.apply();
    // на телефоне общий план нечитаем: стартуем со вводной карточки
    requestAnimationFrame(() => mobile()
      ? world.flyTo({ x: 60, y: 130, w: 720, h: 980 }, { maxZoom: 1 })
      : world.fitAll(0));
  }

  // клавиатура
  window.addEventListener('keydown', e => {
    if ((e.target as HTMLElement).tagName === 'INPUT') return;
    if (e.key === 'Escape') store.select({ type: 'model' });
    if (e.key === '0') world.fitAll(panelPad());
    if (e.key === '+' || e.key === '=') world.zoomBy(1.3);
    if (e.key === '-') world.zoomBy(1 / 1.3);
  });
}

window.addEventListener('popstate', () => boot(urlSlug()));
boot();
