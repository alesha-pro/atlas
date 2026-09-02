import { fmtN } from '../data';
import { t } from '../i18n';
import { el } from '../world';
import type { Store } from '../store';

export function buildIntro(store: Store, x: number, y: number): HTMLElement {
  const m = store.model;
  const root = el('div', 'section', `left:${x}px;top:${y}px;width:660px`);

  const full = m.langLayers.filter(l => l.kind === 'full').length;
  const lin = m.langLayers.filter(l => l.kind === 'linear').length;
  const n2d = m.tensors.filter(x => x.is2d).length;

  const isGlm = m.entry.kind === 'glm-live';
  root.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:13px">
      <div class="eyebrow accent mono">${t(isGlm ? 'glm.intro.eyebrow' : 'intro.eyebrow')}</div>
      <div style="font-size:52px;font-weight:400;letter-spacing:-0.015em;line-height:1.03">
        ${m.name}<br><span style="font-size:26px;color:var(--muted)">${t(isGlm ? 'glm.intro.sub' : 'intro.sub')}</span>
      </div>
      <div class="note" style="font-size:16.5px;max-width:600px">
        ${isGlm ? t('glm.intro.note') : t('intro.note', fmtN(m.totalParams), m.tensors.length)}
      </div>
      <div style="display:flex;gap:10px;margin-top:4px">
        ${[
          [isGlm ? fmtN(m.entry.active_params ?? 0) : fmtN(m.langParams), isGlm ? t('glm.intro.stat.active') : t('intro.stat.lang')],
          [t('intro.stat.layers', m.langLayers.length), isGlm ? t('glm.intro.stat.layers.sub', full, lin) : t('intro.stat.layers.sub', full, lin)],
          [isGlm ? '24 blocks' : fmtN(m.visParams), isGlm ? t('glm.intro.stat.vision') : t('intro.stat.vision')],
          [isGlm ? '288 → 8' : `${n2d}`, isGlm ? t('glm.intro.stat.experts') : t('intro.stat.sqnr')],
        ].map(([v, l]) => `
          <div style="flex:1;border:1px solid var(--line);background:var(--glass-soft);border-radius:4px;padding:12px 13px;display:flex;flex-direction:column;gap:4px">
            <div class="mono" style="font-size:19px;color:var(--ink)">${v}</div>
            <div style="font-size:12.5px;color:var(--faint);line-height:1.3">${l}</div>
          </div>`).join('')}
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:2px" class="mono">
        <div style="width:6px;height:6px;border-radius:3px;background:var(--accent)"></div>
        <div style="font-size:11.5px;color:var(--accent-ink)">${t('intro.hint')}</div>
      </div>
      <div class="small-note" style="max-width:580px">${isGlm ? t('glm.intro.rhythm') : t('intro.rhythm')}</div>
    </div>`;
  return root;
}
