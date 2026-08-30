import { t } from './i18n';

// Загрузка atlas.jsonl и все производные структуры.
// Ничего модельно-специфичного не хардкодим: слои, компоненты и диапазоны
// выводятся из самих данных, чтобы любой следующий чекпойнт лёг без правок.

export interface Tensor {
  name: string;
  shape: number[];
  dtype: string;
  numel: number;
  mean: number; std: number; absmax: number; absmean: number;
  p50: number; p90: number; p99: number; p999: number; p9999: number;
  kurtosis: number; skew: number; sparsity: number;
  outlier_3s: number; outlier_4s: number; outlier_6s: number;
  dyn_range: number;
  hist_log2: number[];
  row_amax_ratio?: number; col_amax_ratio?: number; row_amax_p99?: number;
  sqnr_int8_ch?: number; sqnr_int4_g128?: number; sqnr_fp8_e4m3?: number;
  sv_top?: number[]; stable_rank?: number; sv_decay?: number;
  component: string;
  layer: number | null;
  shard: string;
  // производные
  idx: number;
  stack: 'lang' | 'vision' | 'top';
  langLayer: number | null;
  visBlock: number | null;
  group: string;
  slot: string;
  short: string;
  is2d: boolean;
  hot: number | null;
}

export interface Layer {
  key: string;            // 'L12' | 'V3' | 'top'
  label: string;
  kind: 'full' | 'linear' | 'vision' | 'top';
  idx: number;            // порядковый номер в своей стопке
  tensors: Tensor[];
  params: number;
}

export interface GroupInfo {
  key: string; label: string; plain: string;
  tensors: Tensor[]; params: number; share: number;
  kind: string;
}

export interface MetricDef {
  key: string;
  label: string;
  unit: string;
  get: (t: Tensor) => number | null;
  log: boolean;
  invert: boolean;        // true = большое значение это плохо (красный)
  lo: number; hi: number; // домен (после transform)
  loT: string; hiT: string;
  transform: (v: number) => number;
  fmt: (v: number) => string;
}

export interface Model {
  slug: string;
  name: string;
  tensors: Tensor[];
  langLayers: Layer[];
  visBlocks: Layer[];
  topLayer: Layer;
  visExtra: Tensor[];     // pos_embed / patch_embed / merger — вне блоков
  groups: GroupInfo[];
  metrics: Record<string, MetricDef>;
  totalParams: number;
  langParams: number;
  visParams: number;
  quarters: { label: string; avg: number }[];
  int4Sorted: number[];   // для перцентилей вердикта
}

export interface ManifestEntry { slug: string; name: string; note?: string; }

const SLOT_ORDER = [
  'attn.q', 'attn.k', 'attn.v', 'attn.o', 'attn.q_norm', 'attn.k_norm',
  'linattn.in_qkv', 'linattn.in_a', 'linattn.in_b', 'linattn.in_z',
  'linattn.conv', 'linattn.A_log', 'linattn.dt_bias', 'linattn.norm', 'linattn.out',
  'mlp.gate', 'mlp.up', 'mlp.down',
  'norm.input', 'norm.post',
];

function slotOf(t: { component: string; name: string }): string {
  const n = t.name;
  let c = t.component;
  if (c === 'other' && n.includes('in_proj_qkv')) c = 'linattn.in_qkv';
  if (c === 'linattn.dt') c = n.includes('A_log') ? 'linattn.A_log' : 'linattn.dt_bias';
  if (c === 'norm') {
    if (n.includes('q_norm')) return 'attn.q_norm';
    if (n.includes('k_norm')) return 'attn.k_norm';
    if (n.includes('linear_attn')) return 'linattn.norm';
    if (n.includes('input_layernorm')) return 'norm.input';
    if (n.includes('post_attention_layernorm')) return 'norm.post';
    return 'norm.' + n.split('.').slice(-2).join('.').replace('.weight', '');
  }
  if (c === 'linattn.out') return 'linattn.out';
  return c;
}

export function slotRank(slot: string): number {
  const i = SLOT_ORDER.indexOf(slot);
  return i >= 0 ? i : SLOT_ORDER.length + slot.charCodeAt(0) / 200;
}

// Человеческое имя компонента (локализовано)
const PLAIN_KEYS = new Set([
  'attn.q', 'attn.k', 'attn.v', 'attn.o', 'attn.q_norm', 'attn.k_norm',
  'linattn.in_qkv', 'linattn.in_a', 'linattn.in_b', 'linattn.in_z', 'linattn.conv',
  'linattn.A_log', 'linattn.dt_bias', 'linattn.norm', 'linattn.out',
  'mlp.gate', 'mlp.up', 'mlp.down', 'norm.input', 'norm.post', 'embed', 'lm_head',
]);
export function plainName(slot: string): string | null {
  return PLAIN_KEYS.has(slot) ? t('plain.' + slot) : null;
}

export function fmtN(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + t('fmt.b');
  if (n >= 1e6) return (n / 1e6).toFixed(1) + t('fmt.m');
  if (n >= 1e3) return (n / 1e3).toFixed(1) + t('fmt.k');
  return String(n);
}

function q(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const i = Math.max(0, Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1))));
  return sorted[i];
}

function makeMetric(
  tensors: Tensor[], key: string, label: string, unit: string,
  get: (t: Tensor) => number | null | undefined,
  opts: { log?: boolean; shift?: number; invert?: boolean; loT?: string; hiT?: string; digits?: number; fixed?: [number, number] },
): MetricDef {
  const g = (t: Tensor) => { const v = get(t); return v == null ? null : v; };
  const shift = opts.shift ?? 0;
  const transform = (v: number) => opts.log ? Math.log10(Math.max(v + shift, 1e-4)) : v;
  const vals = tensors.map(g).filter((v): v is number => v != null && isFinite(v))
    .map(transform)
    .sort((a, b) => a - b);
  let lo = q(vals, 0.02), hi = q(vals, 0.98);
  if (opts.fixed) { lo = opts.fixed[0]; hi = opts.fixed[1]; }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const d = opts.digits ?? 1;
  return {
    key, label, unit, log: !!opts.log, invert: !!opts.invert, lo, hi,
    loT: opts.loT ?? '', hiT: opts.hiT ?? '',
    get: g, transform,
    fmt: (v: number) => opts.log && v >= 1000 ? fmtN(v) : v.toFixed(d),
  };
}

export async function loadManifest(): Promise<ManifestEntry[]> {
  const r = await fetch(new URL('models/manifest.json', document.baseURI));
  return r.json();
}

export async function loadModel(entry: ManifestEntry): Promise<Model> {
  const r = await fetch(new URL(`models/${entry.slug}/atlas.jsonl`, document.baseURI));
  const text = await r.text();
  const tensors: Tensor[] = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const t = JSON.parse(line) as Tensor;
    t.idx = tensors.length;
    const m = t.name.match(/\.layers\.(\d+)\./);
    const vm = t.name.match(/visual\.blocks\.(\d+)\./);
    const isVis = t.name.includes('visual.');
    t.stack = isVis ? 'vision' : m ? 'lang' : 'top';
    t.langLayer = !isVis && m ? +m[1] : null;
    t.visBlock = vm ? +vm[1] : null;
    t.slot = isVis ? visSlot(t) : slotOf(t);
    t.is2d = t.shape.length >= 2 && t.sqnr_int4_g128 != null;
    t.hot = t.row_amax_ratio != null && t.col_amax_ratio != null
      ? Math.max(t.row_amax_ratio, t.col_amax_ratio) : null;
    t.group = groupOf(t);
    t.short = t.name.replace(/^model\./, '').replace(/language_model\./, '').replace(/\.weight$/, '');
    tensors.push(t);
  }

  // языковые слои
  const langMap = new Map<number, Tensor[]>();
  for (const t of tensors) if (t.langLayer != null) {
    if (!langMap.has(t.langLayer)) langMap.set(t.langLayer, []);
    langMap.get(t.langLayer)!.push(t);
  }
  const langLayers: Layer[] = [...langMap.keys()].sort((a, b) => a - b).map((l, i) => {
    const ts = langMap.get(l)!.sort((a, b) => slotRank(a.slot) - slotRank(b.slot));
    const kind = ts.some(t => t.component === 'attn.q') ? 'full' : 'linear';
    return { key: 'L' + l, label: String(l), kind, idx: i, tensors: ts, params: sum(ts) };
  });

  // блоки визуальной башни
  const visMap = new Map<number, Tensor[]>();
  const visExtra: Tensor[] = [];
  for (const t of tensors) if (t.stack === 'vision') {
    if (t.visBlock != null) {
      if (!visMap.has(t.visBlock)) visMap.set(t.visBlock, []);
      visMap.get(t.visBlock)!.push(t);
    } else visExtra.push(t);
  }
  const visBlocks: Layer[] = [...visMap.keys()].sort((a, b) => a - b).map((l, i) => {
    const ts = visMap.get(l)!.sort((a, b) => slotRank2(a.slot) - slotRank2(b.slot));
    return { key: 'V' + l, label: 'v' + l, kind: 'vision', idx: i, tensors: ts, params: sum(ts) };
  });

  const topTs = tensors.filter(t => t.stack === 'top')
    .sort((a, b) => b.numel - a.numel);
  const topLayer: Layer = { key: 'top', label: 'вход/выход', kind: 'top', idx: 0, tensors: topTs, params: sum(topTs) };

  const totalParams = sum(tensors);
  const langParams = sum(tensors.filter(t => t.stack === 'lang')) + sum(topTs.filter(t => !t.name.startsWith('mtp')));
  const visParams = sum(tensors.filter(t => t.stack === 'vision'));

  // группы для архитектурной карты
  const groups = buildGroups(tensors, totalParams);

  // четверти стека по INT4
  const L = langLayers.length;
  const quarters = [0, 1, 2, 3].map(qi => {
    const ls = langLayers.filter(l => Math.floor(l.idx * 4 / L) === qi);
    const vals = ls.flatMap(l => l.tensors).map(t => t.sqnr_int4_g128).filter((v): v is number => v != null);
    return {
      label: t('layers.range', ls[0]?.label, ls[ls.length - 1]?.label),
      avg: vals.reduce((a, b) => a + b, 0) / (vals.length || 1),
    };
  });

  const metrics: Record<string, MetricDef> = {};
  const M = (key: string, unit: string, get: (x: Tensor) => number | null | undefined,
    opts: Parameters<typeof makeMetric>[5]) =>
    makeMetric(tensors, key, t('metric.' + key), unit, get,
      { ...opts, loT: t(`sc.${key}.lo`), hiT: t(`sc.${key}.hi`) });
  const db = t('unit.db');
  const defs: MetricDef[] = [
    M('int4', db, x => x.sqnr_int4_g128, {}),
    M('int8', db, x => x.sqnr_int8_ch, {}),
    M('fp8', db, x => x.sqnr_fp8_e4m3, { digits: 2 }),
    M('kurt', '', x => x.kurtosis, { log: true, shift: 3, invert: true }),
    M('hot', '×', x => x.hot, { log: true, invert: true }),
    M('dyn', '×', x => x.dyn_range, { log: true, invert: true }),
    M('srank', '', x => x.stable_rank, { log: true, digits: 0 }),
    M('size', '', x => x.numel, { log: true, digits: 0 }),
    M('sparsity', '%', x => x.sparsity * 100, { log: false, invert: true, digits: 3 }),
    M('out3s', '%', x => x.outlier_3s * 100, { invert: true, digits: 2 }),
  ];
  for (const d of defs) metrics[d.key] = d;

  const int4Sorted = tensors.map(t => t.sqnr_int4_g128).filter((v): v is number => v != null).sort((a, b) => a - b);

  return {
    slug: entry.slug, name: entry.name, tensors, langLayers, visBlocks, topLayer, visExtra,
    groups, metrics, totalParams, langParams, visParams, quarters, int4Sorted,
  };
}

function visSlot(t: { name: string }): string {
  const tail = t.name.replace(/^model\.visual\./, '').replace(/blocks\.\d+\./, '');
  return 'vis.' + tail.replace(/\.(weight|bias)$/, (m) => m === '.bias' ? '.b' : '');
}
function slotRank2(slot: string): number {
  const order = ['vis.attn.qkv', 'vis.attn.qkv.b', 'vis.attn.proj', 'vis.attn.proj.b',
    'vis.mlp.linear_fc1', 'vis.mlp.linear_fc1.b', 'vis.mlp.linear_fc2', 'vis.mlp.linear_fc2.b',
    'vis.norm1', 'vis.norm1.b', 'vis.norm2', 'vis.norm2.b'];
  const i = order.indexOf(slot);
  return i >= 0 ? i : 90;
}

function sum(ts: { numel: number }[]): number { return ts.reduce((a, t) => a + t.numel, 0); }

function groupOf(t: Tensor): string {
  const n = t.name;
  if (n.startsWith('mtp.')) return 'mtp';
  if (t.stack === 'vision') return 'vision';
  if (t.component === 'embed') return 'embed';
  if (t.component === 'lm_head') return 'head';
  const s = t.slot;
  if (s.startsWith('attn.') && !s.includes('norm')) return 'attn';
  if (s.startsWith('linattn.')) return 'linattn';
  if (s.startsWith('mlp.')) return 'mlp';
  return 'norm';
}

function buildGroups(tensors: Tensor[], total: number): GroupInfo[] {
  const defs: [string, string][] = [
    ['embed', 'in'], ['attn', 'attn'], ['linattn', 'lin'], ['mlp', 'mlp'],
    ['norm', 'norm'], ['vision', 'vision'], ['head', 'out'], ['mtp', 'out'],
  ];
  return defs.map(([key, kind]) => {
    const ts = tensors.filter(x => x.group === key);
    return {
      key, kind, label: t(`group.${key}.label`), plain: t(`group.${key}.plain`),
      tensors: ts, params: sum(ts), share: sum(ts) / total * 100,
    };
  }).filter(g => g.tensors.length > 0);
}
