import type { MetricDef, Tensor } from './data';

// Пятиступенчатая тёплая шкала: терракота → янтарь → солома → шалфей → волна
const STOPS: [number, number, number, number][] = [
  [0.00, 0.78, 0.105, 24],
  [0.35, 0.85, 0.090, 55],
  [0.60, 0.90, 0.075, 95],
  [0.80, 0.88, 0.062, 155],
  [1.00, 0.86, 0.058, 215],
];

export const HATCH = 'repeating-linear-gradient(45deg,#e6e0d3,#e6e0d3 3px,#f2eee4 3px,#f2eee4 6px)';
export const NA_INK = '#a49c8d';

export function ramp(t: number): string {
  t = Math.max(0, Math.min(1, t));
  let a = STOPS[0], b = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++)
    if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { a = STOPS[i]; b = STOPS[i + 1]; }
  const k = (t - a[0]) / ((b[0] - a[0]) || 1);
  const m = (i: number) => (a[i] + (b[i] - a[i]) * k).toFixed(3);
  return `oklch(${m(1)} ${m(2)} ${m(3)})`;
}

export function rampCSS(): string {
  return `linear-gradient(90deg,${STOPS.map(s => `oklch(${s[1]} ${s[2]} ${s[3]})`).join(',')})`;
}

export function tOf(v: number | null, d: MetricDef): number | null {
  if (v == null || !isFinite(v)) return null;
  let t = (d.transform(v) - d.lo) / (d.hi - d.lo);
  t = Math.max(0, Math.min(1, t));
  return d.invert ? 1 - t : t;
}

export function colorFor(v: number | null, d: MetricDef): string {
  const t = tOf(v, d);
  return t == null ? HATCH : ramp(t);
}

export function colorForTensor(t: Tensor, d: MetricDef): string {
  return colorFor(d.get(t), d);
}

// цвет текста для значения: красный/янтарный/зелёный по позиции на шкале
export function ink(v: number | null, d: MetricDef): string {
  const t = tOf(v, d);
  if (t == null) return NA_INK;
  return t < 0.34 ? '#b0492a' : t < 0.62 ? '#a2701f' : '#4d7a63';
}

// стабильные цвета «пород» (виды блоков)
export const KIND: Record<string, { bg: string; fg: string; solid: string; bd: string }> = {
  in:     { bg: 'rgba(214,228,242,0.6)', fg: '#4e6a86', solid: '#7f9cb8', bd: 'rgba(120,150,180,0.32)' },
  attn:   { bg: 'rgba(247,225,201,0.7)', fg: '#9a6a2c', solid: '#d3a05a', bd: 'rgba(200,160,105,0.36)' },
  lin:    { bg: 'rgba(210,231,228,0.7)', fg: '#3f6f6a', solid: '#8fb2bd', bd: 'rgba(130,175,170,0.36)' },
  mlp:    { bg: 'rgba(226,214,240,0.7)', fg: '#6a4f8c', solid: '#a891cf', bd: 'rgba(160,135,195,0.34)' },
  norm:   { bg: 'rgba(226,220,205,0.8)', fg: '#6b6459', solid: '#b7ac96', bd: 'rgba(120,106,84,0.26)' },
  out:    { bg: 'rgba(255,222,190,0.7)', fg: '#a35c1c', solid: '#e0955c', bd: 'rgba(220,160,100,0.36)' },
  vision: { bg: 'rgba(224,216,243,0.7)', fg: '#5c4a8f', solid: '#a98cc0', bd: 'rgba(150,130,200,0.34)' },
};

export function kindOf(key: string) { return KIND[key] || KIND.norm; }
