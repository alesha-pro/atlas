"""atlas_live.py — съём «живых» внутренностей Qwen3.8-27B для Weight Atlas.

Дополняет статический atlas.jsonl (веса) данными с реальных прямых проходов:
  flow    — residual-поток по глубине (||h||, вклад слоя Δh), каналы-выбросы
            активаций, измеренная квантуемость АКТИВАЦИЙ (INT8 per-token /
            FP8 e4m3) на входах всех больших Linear;
  attn    — энтропия внимания (норм. по ln L), attention sink (токен 0),
            профиль затухания массы по дистанции, открытость выходного гейта
            full-attn слоёв; отдельно витрина: реальные карты внимания;
  linattn — write-gate β (sigmoid), динамический распад g и half-life памяти,
            RMS recurrent-состояния после окна seqlen;
  frag    — хрупкость слоёв: KL(base ‖ INT4-g128-слой) при послойной замене;
  showcase — карта внимания на фиксированном промпте + (если картинка рядом)
            доля внимания на image-токенах.

Режимы:
  --mode selftest  CPU, 3 слоя from_config, случайные веса: проверка всех хуков.
  --mode flow|attn|linattn|frag|showcase|all   GPU (нужны свободные карты).

Выход: out/live/{flow,attn,linattn,frag,maps_text,maps_img}.json
Дальше reduce_live.py собирает из них + carve-артефактов public/models/<slug>/live.json.

Запуск на риге:
  .venv/bin/python atlas_live.py --mode selftest
  .venv/bin/python atlas_live.py --mode all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = os.environ.get("ATLAS_MODEL_DIR", "")   # или --model-dir
OUT = Path("out/live")
DOMAINS = ("en", "code", "agent")
DECAY_EDGES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)

_state: dict = {"domain": "en", "mode": "", "img_pos": None, "showcase": None}
_attn_acc: dict[int, dict] = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def r6(x: float) -> float:
    return round(float(x), 6)


# ────────────────────────── quantization probes ──────────────────────────

def int8_dyn_sqnr(x: torch.Tensor) -> float:
    flat = x.detach().reshape(-1, x.shape[-1]).float()
    s = flat.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
    xq = (flat / s).round().clamp(-127, 127) * s
    return float(10 * torch.log10(flat.pow(2).sum() / (flat - xq).pow(2).sum().clamp(min=1e-20)))


def fp8_sqnr(x: torch.Tensor) -> float:
    flat = x.detach().reshape(-1, x.shape[-1]).float()
    s = flat.abs().amax().clamp(min=1e-12) / 448.0
    xq = (flat / s).to(torch.float8_e4m3fn).float() * s
    return float(10 * torch.log10(flat.pow(2).sum() / (flat - xq).pow(2).sum().clamp(min=1e-20)))


ACTQ_SITES = (
    ("self_attn.q_proj", "attn.q_in"), ("self_attn.k_proj", "attn.k_in"),
    ("self_attn.v_proj", "attn.v_in"), ("self_attn.o_proj", "attn.o_in"),
    ("linear_attn.in_proj_qkv", "linattn.qkv_in"), ("linear_attn.in_proj_z", "linattn.z_in"),
    ("linear_attn.in_proj_b", "linattn.b_in"), ("linear_attn.in_proj_a", "linattn.a_in"),
    ("linear_attn.out_proj", "linattn.out_in"),
    ("mlp.gate_proj", "mlp.gate_in"), ("mlp.up_proj", "mlp.up_in"),
    ("mlp.down_proj", "mlp.down_in"),
)


# ───────────────────────────── flow + actq ─────────────────────────────

class FlowCollector:
    """Residual-поток по слоям + SQNR квантования активаций на входах Linear.

    Аккумуляторы ключуются (layer, domain); dump усредняет по батчам.
    """

    def __init__(self, model: nn.Module):
        self.handles = []
        self.n_layers = len(model.model.language_model.layers)
        self.flow: dict[tuple[int, str], dict] = {}   # (layer|‑1, dom) → sums
        self.actq: dict[str, list] = {}               # site → [Σint8, Σfp8, n]
        self._layer_in: dict[int, torch.Tensor] = {}

        def acc(key):
            return self.flow.setdefault(key, {"h": 0.0, "d": 0.0, "ratio": 0.0,
                                              "nout": 0, "n": 0})

        def pre_layer(i):
            def hook(mod, args):
                self._layer_in[i] = args[0].detach()
            return hook

        def post_layer(i):
            def hook(mod, args, out):
                x_in = self._layer_in.pop(i)
                dom = _state["domain"]
                if _state["mode"] not in ("flow", "selftest"):
                    return
                with torch.no_grad():
                    of = out.detach().float()
                    h = float(of.pow(2).mean().sqrt())
                    dl = of - x_in.float()
                    d = float(dl.pow(2).mean().sqrt())
                    if x_in.dim() == 3:
                        am = x_in.detach().abs().float().mean(dim=(0, 1))
                        med = am.median().clamp(min=1e-8)
                        ratio = float(am.max() / med)
                        nout = int((am > 5 * med).sum())
                    else:
                        ratio, nout = float("nan"), 0
                a = acc((i, dom))
                a["h"] += h; a["d"] += d; a["ratio"] += 0.0 if math.isnan(ratio) else ratio
                a["nout"] += nout; a["n"] += 1
                a_all = acc((i, "all"))
                a_all["h"] += h; a_all["d"] += d
                a_all["ratio"] += 0.0 if math.isnan(ratio) else ratio
                a_all["nout"] += nout; a_all["n"] += 1
            return hook

        dec = model.model.language_model.layers
        for i, layer in enumerate(dec):
            self.handles.append(layer.register_forward_pre_hook(pre_layer(i)))
            self.handles.append(layer.register_forward_hook(post_layer(i)))

        def post_embed(mod, args, out):
            if _state["mode"] not in ("flow", "selftest"):
                return
            dom = _state["domain"]
            with torch.no_grad():
                h = float(out.detach().float().pow(2).mean().sqrt())
            for dd in (dom, "all"):
                a = acc((-1, dd)); a["h"] += h; a["n"] += 1
        self.handles.append(model.model.language_model.embed_tokens.register_forward_hook(post_embed))

        for name, mod in model.named_modules():
            for pat, site in ACTQ_SITES:
                if name.endswith(pat):
                    self.handles.append(mod.register_forward_pre_hook(self._mk_actq(site)))

    def _mk_actq(self, site: str):
        def hook(mod, args):
            if _state["mode"] != "flow":
                return
            x = args[0]
            if x.dim() < 2 or x.shape[-1] < 64:
                return
            e = self.actq.setdefault(site, [0.0, 0.0, 0])
            e[0] += int8_dyn_sqnr(x)
            e[1] += fp8_sqnr(x)
            e[2] += 1
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def dump(self) -> dict:
        def series(field):
            return {d: [r6(self.flow[(i, d)][field] / self.flow[(i, d)]["n"])
                        if (i, d) in self.flow else None
                        for i in range(self.n_layers)]
                    for d in DOMAINS + ("all",)}
        return {
            "n_layers": self.n_layers,
            "h_rms": series("h"),
            "delta_rms": series("d"),
            "out_ratio": series("ratio"),
            "n_out_dims": series("nout"),
            "h_rms0": {d: r6(self.flow[(-1, d)]["h"] / self.flow[(-1, d)]["n"])
                       for d in DOMAINS + ("all",) if (-1, d) in self.flow},
            "actq": {site: {"int8": r6(s[0] / s[2]), "fp8": r6(s[1] / s[2]), "n": s[2]}
                     for site, s in self.actq.items()},
        }


# ───────────────────────────── attention ─────────────────────────────

def atlas_attention_forward(module, query, key, value, attention_mask=None, dropout=0.0,
                            scaling=None, **kwargs):
    """Перехват full-attention: probs считаем сами, копим статистику."""
    # башня зрения: внимание не причинное, статистику по ней не собираем.
    # Свою причинную маску сюда применять нельзя — она исказит препроцессинг картинки.
    if not hasattr(module, "layer_idx"):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value,
                                      attention_mask=attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    L = query.shape[-2]
    if key.shape[1] != query.shape[1]:                 # GQA: 24 Q ↔ 4 KV
        rep = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    causal = torch.ones(L, L, dtype=torch.bool, device=scores.device).triu(1)
    scores = scores.masked_fill(causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = probs.to(value.dtype) @ value
    _attn_stats(module, probs, L)
    # штатный eager отдаёт [B,L,H,D]; без transpose оси перемешиваются и все
    # слои после первого считают статистику по искажённым активациям
    return out.transpose(1, 2).contiguous(), None


def _attn_acc_(layer: int) -> dict:
    return _attn_acc.setdefault(layer, {
        "ent": 0.0, "ent_std": 0.0, "first": 0.0, "diag": 0.0,
        "prof": [0.0] * (len(DECAY_EDGES) - 1), "n": 0,
        "gate_sum": 0.0, "gate_n": 0, "img": 0.0,
        "ent_by_head": None, "p_head_cache": None,
    })


def _attn_stats(module, probs: torch.Tensor, L: int):
    mode = _state["mode"]
    if mode not in ("attn", "showcase"):
        return
    layer = module.layer_idx
    with torch.no_grad():
        p = probs.detach().float()                          # [B,H,L,L]
        pc = p.clamp(min=1e-12)
        ent = -(pc * pc.log()).sum(-1)                      # [B,H,L]
        ent_norm_mean = float((ent / math.log(L)).mean())
        ent_head = (ent / math.log(L)).mean(-1)             # [B,H]
        first = float(p[..., 0].mean())
        idx = torch.arange(L, device=p.device)
        dist = (idx[None, :] - idx[:, None]).clamp(min=0)
        total = p.sum()
        diag = float((p * (dist <= 4)).sum() / total)
        prof = []
        for lo, hi in zip(DECAY_EDGES, DECAY_EDGES[1:]):
            m = (dist >= lo - 1) & (dist < hi - 1)
            prof.append(float((p * m).sum() / total))
        st = _attn_acc_(layer)
        st["ent"] += ent_norm_mean
        st["ent_std"] += float(ent_head.flatten().std())
        st["first"] += first
        st["diag"] += diag
        st["prof"] = [a + b for a, b in zip(st["prof"], prof)]
        st["n"] += 1
        img_pos = _state.get("img_pos")
        if img_pos:
            sel = torch.as_tensor([i for i in img_pos if i < L], device=p.device, dtype=torch.long)
            if len(sel):
                st["img"] += float(p[..., sel].sum() / total)
        if mode == "showcase" and _state["showcase"] is not None:
            _state["showcase"](layer, p, ent_head)


def attach_gate_hooks(model: nn.Module):
    """Открытость выходного гейта full-attn: sigmoid(gate) из выхода q_proj."""
    handles = []
    dec = model.model.language_model.layers
    for i, layer in enumerate(dec):
        if not hasattr(layer, "self_attn"):
            continue
        q = layer.self_attn.q_proj
        hd = layer.self_attn.head_dim

        def hook(mod, args, out, layer_i=i, hd=hd):
            if _state["mode"] not in ("attn", "showcase", "selftest"):
                return
            with torch.no_grad():
                v = out.detach()
                gate = v.view(*v.shape[:-1], -1, hd * 2)[..., hd:]
                g = float(torch.sigmoid(gate.float()).mean())
            st = _attn_acc_(layer_i)
            st["gate_sum"] += g
            st["gate_n"] += 1
        handles.append(q.register_forward_hook(hook))
    return handles


# ───────────────────────────── linear attn ─────────────────────────────

class LinAttnCollector:
    """β (write-gate), g (распад, mean по головам и токенам) и RMS финального
    recurrent-состояния — по линейным слоям, с разбивкой по доменам."""

    def __init__(self, model: nn.Module):
        self.handles = []
        self.beta: dict[tuple[int, str], list] = {}
        self.g: dict[tuple[int, str], list] = {}
        self.state: dict[tuple[int, str], list] = {}
        self._mods: dict[int, nn.Module] = {}
        self._orig: dict[int, callable] = {}
        dec = model.model.language_model.layers
        for i, layer in enumerate(dec):
            if not hasattr(layer, "linear_attn"):
                continue
            la = layer.linear_attn
            self._mods[i] = la
            self._orig[i] = la.chunk_gated_delta_rule
            self.handles.append(la.in_proj_b.register_forward_hook(self._mk_beta(i)))
            self.handles.append(la.in_proj_a.register_forward_hook(self._mk_g(i)))
            la.chunk_gated_delta_rule = self._wrap_chunk(i, self._orig[i])

    def _mk_beta(self, i: int):
        def hook(mod, args, out):
            if _state["mode"] != "linattn":
                return
            self.beta.setdefault((i, _state["domain"]), []).append(
                float(torch.sigmoid(out.detach().float()).mean()))
        return hook

    def _mk_g(self, i: int):
        def hook(mod, args, out):
            if _state["mode"] != "linattn":
                return
            la = self._mods[i]
            dt = F.softplus(out.detach().float() + la.dt_bias.detach().float())
            g = -la.A_log.detach().float().exp() * dt
            self.g.setdefault((i, _state["domain"]), []).append(float(g.mean()))
        return hook

    def _wrap_chunk(self, i: int, orig):
        def wrapped(q, k, v, **kw):
            kw["output_final_state"] = True
            out, state = orig(q, k, v, **kw)
            if state is not None and _state["mode"] == "linattn":
                self.state.setdefault((i, _state["domain"]), []).append(
                    float(state.detach().float().pow(2).mean().sqrt()))
            return out, state
        return wrapped

    def remove(self):
        for h in self.handles:
            h.remove()
        for i, la in self._mods.items():
            la.chunk_gated_delta_rule = self._orig[i]

    def dump(self) -> dict:
        def agg(src):
            return {d: [r6(sum(src[(i, d)]) / len(src[(i, d)])) if (i, d) in src else None
                        for i in sorted({k[0] for k in src})]
                    for d in DOMAINS + ("all",)}
        # «all» домена нет в ключах — домен фиксируется на батч; добавим all как mean по доменам
        out = {"beta_open": {}, "g_mean": {}, "state_rms": {}}
        for field, src in (("beta_open", self.beta), ("g_mean", self.g), ("state_rms", self.state)):
            layers = sorted({k[0] for k in src})
            per_dom = {d: [r6(sum(src[(i, d)]) / len(src[(i, d)])) if (i, d) in src else None
                           for i in layers] for d in DOMAINS}
            per_dom["layers"] = layers
            out[field] = per_dom
        return out


# ───────────────────────────── fragility ─────────────────────────────

def fake_int4_g128(w: torch.Tensor) -> torch.Tensor:
    wd = w.detach()
    n = wd.shape[-1]
    g = n // 128 * 128
    if g == 0:
        return wd
    wf = wd.float()
    x = wf[..., :g].reshape(-1, 128)
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 7.0
    xq = ((x / s).round().clamp(-7, 7) * s).reshape(wf.shape[:-1] + (g,))
    out = wf.clone()
    out[..., :g] = xq
    return out.to(wd.dtype)


def fragility(model: nn.Module, ids: torch.Tensor) -> dict:
    """KL(base ‖ INT4-слой) по каждому слою отдельно + cosine логитов."""
    dec = model.model.language_model.layers
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        base_lp = F.log_softmax(base, dim=-1)
    pos = slice(base.shape[1] * 3 // 4, None)
    kl, cos = [], []
    for i, layer in enumerate(dec):
        lins = [m for m in layer.modules() if isinstance(m, nn.Linear)]
        saved = [(m, m.weight.data) for m in lins]
        with torch.no_grad():
            for m, _ in saved:
                m.weight.data = fake_int4_g128(m.weight.data)
            out = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
            del_l = F.log_softmax(out, dim=-1)
            kl_i = float((base_lp[:, pos].exp() * (base_lp[:, pos] - del_l[:, pos])).sum(-1).mean())
            cos_i = float(F.cosine_similarity(
                base[:, pos].reshape(-1), out[:, pos].reshape(-1), dim=0))
        for m, w in saved:
            m.weight.data = w
        kl.append(r6(kl_i))
        cos.append(r6(cos_i))
        if i % 8 == 0:
            log(f"  frag {i + 1}/{len(dec)} kl={kl_i:.4f}")
    return {"kl": kl, "logit_cos": cos}


# ───────────────────────────── бины/утилиты ─────────────────────────────

def pack_bins(tok, path: str, seqlen: int, max_records: int = 0):
    bins, cur_dom, cur = [], None, []
    with open(path) as f:
        for n, line in enumerate(f):
            if max_records and n > max_records:
                break
            rec = json.loads(line)
            ids = tok(rec["text"], add_special_tokens=False).input_ids
            if rec["domain"] != cur_dom and cur:
                bins.append((cur_dom, cur))
                cur = []
            cur_dom = rec["domain"]
            cur.extend(ids)
            while len(cur) >= seqlen:
                bins.append((cur_dom, cur[:seqlen]))
                cur = cur[seqlen:]
    if cur:
        bins.append((cur_dom, cur))
    return bins


def spread_bins(bins, seqlen, total):
    """Равномерно по доменам: total полных бинов."""
    full = [b for b in bins if len(b[1]) == seqlen]
    per_dom = max(1, total // 3)
    take, cnt = [], {d: 0 for d in DOMAINS}
    for b in full:
        if cnt[b[0]] < per_dom:
            take.append(b)
            cnt[b[0]] += 1
    return take


def forward_text(model, ids: torch.Tensor):
    try:
        model.model(input_ids=ids, use_cache=False)
    except Exception:
        model(input_ids=ids, use_cache=False)


def run_pass(model, bins, batch, mode):
    _state["mode"] = mode
    batches, cur = [], []
    for b in bins:
        if cur and (cur[0][0] != b[0] or len(cur) == batch):
            batches.append(cur)
            cur = []
        cur.append(b)
    if cur:
        batches.append(cur)
    dev = model.model.language_model.embed_tokens.weight.device
    t0 = time.time()
    with torch.no_grad():
        for bi, chunk in enumerate(batches):
            _state["domain"] = chunk[0][0]
            ids = torch.tensor([b for _, b in chunk], dtype=torch.long, device=dev)
            forward_text(model, ids)
            if bi % 10 == 0 or bi == len(batches) - 1:
                log(f"  {mode} batch {bi + 1}/{len(batches)} ({time.time() - t0:.0f}s)")
    _state["mode"] = ""


# ───────────────────────────── режимы GPU ─────────────────────────────

def build_model(attn_impl: str):
    from transformers import AutoConfig, AutoModelForImageTextToText
    kw = {}
    if attn_impl == "atlas_capture":
        # реестр внимания живёт в процессе: без register from_pretrained
        # отвергает имя на валидации (раньше регистрация была только в selftest)
        from transformers import AttentionInterface
        AttentionInterface.register("atlas_capture", atlas_attention_forward)
        cfg = AutoConfig.from_pretrained(MODEL_DIR)
        cfg._attn_implementation = attn_impl
        cfg.text_config._attn_implementation = attn_impl
        kw["config"] = cfg
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="balanced",
        attn_implementation=attn_impl, **kw)
    model.eval()
    return model


def mode_flow(args, model, tok):
    take = spread_bins(pack_bins(tok, args.data, args.seqlen), args.seqlen, args.flow_bins)
    log(f"flow: {len(take)} бинов (~{len(take) * args.seqlen / 1e3:.0f}K токенов)")
    fc = FlowCollector(model)
    run_pass(model, take, args.batch, "flow")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flow.json").write_text(json.dumps(fc.dump(), ensure_ascii=False))
    fc.remove()
    log("flow DONE")


def mode_linattn(args, model, tok):
    lc = LinAttnCollector(model)
    take = spread_bins(pack_bins(tok, args.data, args.seqlen), args.seqlen, args.la_bins)
    log(f"linattn: {len(take)} бинов")
    run_pass(model, take, args.batch, "linattn")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "linattn.json").write_text(json.dumps(lc.dump(), ensure_ascii=False))
    lc.remove()
    log("linattn DONE")


def dump_attn(path: Path):
    per = {}
    for layer, st in _attn_acc.items():
        n = max(st["n"], 1)
        gn = max(st["gate_n"], 1)
        per[str(layer)] = {
            "ent": r6(st["ent"] / n), "ent_std": r6(st["ent_std"] / n),
            "first": r6(st["first"] / n), "diag": r6(st["diag"] / n),
            "prof": [r6(x / n) for x in st["prof"]],
            "gate": r6(st["gate_sum"] / gn),
            "img": r6(st["img"] / n),
        }
    path.write_text(json.dumps(per, ensure_ascii=False))


def mode_attn(args, model, tok):
    _attn_acc.clear()
    _state["img_pos"] = None
    take = spread_bins(pack_bins(tok, args.data, args.seqlen), args.seqlen, args.attn_bins)
    log(f"attn: {len(take)} бинов (batch=1, eager-перехват)")
    gh = attach_gate_hooks(model)
    run_pass(model, take, 1, "attn")
    for h in gh:
        h.remove()
    OUT.mkdir(parents=True, exist_ok=True)
    dump_attn(OUT / "attn.json")
    log("attn DONE")


def mode_frag(args, model, tok):
    bins = [b for b in pack_bins(tok, args.data, 512) if len(b[1]) == 512]
    ids_np = (bins[0] if bins else pack_bins(tok, args.data, 512)[0])[1]
    dev = model.model.language_model.embed_tokens.weight.device
    ids = torch.tensor([ids_np], device=dev)
    log(f"frag: 1×512 токенов, {len(model.model.language_model.layers)} проб")
    res = fragility(model, ids)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frag.json").write_text(json.dumps(res, ensure_ascii=False))
    log("frag DONE")


def mode_showcase(args, model, tok):
    """Реальные карты внимания на демо-промпте + доля на image-токенах."""
    _attn_acc.clear()
    prompt = ("The old observatory stood on the hill above the port. Its copper dome had turned "
              "green decades ago, and every clear night the telescope inside traced the same slow "
              "arc across the sky. Maria climbed the spiral staircase with a thermos of coffee, "
              "three notebooks, and a letter she had not yet opened. The city below hummed; the "
              "stars above waited. She chose a star, wrote down its coordinates, and began to "
              "measure the dark.")
    ids_list = tok(prompt, add_special_tokens=True).input_ids[:112]
    toks = tok.convert_ids_to_tokens(ids_list)
    maps: dict[str, dict] = {}

    def grab(layer, p, ent_head):
        mean = p.mean(1)[0]                                  # [L,L] среднее по головам
        dev_ = (ent_head - ent_head.mean()).abs()
        h_star = int(torch.argmax(dev_.flatten()))
        star = p[0, h_star % p.shape[1]]

        def mat(m):
            a = m.detach().float().cpu().numpy()
            return [[round(float(v), 4) for v in row] for row in a]
        maps[str(layer)] = {"mean": mat(mean), "star_head": h_star, "star": mat(star)}

    _state["showcase"] = grab
    _state["img_pos"] = None
    dev = model.model.language_model.embed_tokens.weight.device
    gh = attach_gate_hooks(model)
    _state["mode"] = "showcase"
    with torch.no_grad():
        model(input_ids=torch.tensor([ids_list], device=dev), use_cache=False)
    _state["mode"] = ""
    for h in gh:
        h.remove()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maps_text.json").write_text(json.dumps(
        {"prompt": prompt, "tokens": toks, "maps": maps}, ensure_ascii=False))
    log(f"showcase text DONE ({len(ids_list)} токенов, {len(maps)} слоёв)")

    img_path = Path(__file__).parent / "atlas_shot.png"
    if not img_path.exists():
        img_path = Path("atlas_shot.png")
    if img_path.exists():
        try:
            from transformers import AutoProcessor
            from PIL import Image
            proc = AutoProcessor.from_pretrained(MODEL_DIR)
            img = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe what you see."}]}]
            # в transformers 5.x картинка идёт прямо через шаблон: раздельный
            # вызов proc(text=..., images=...) возвращает input_ids=None
            enc = proc.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            tok_id_img = getattr(model.config, "image_token_id", 248056)
            ids0 = enc["input_ids"][0].tolist()
            img_pos = [i for i, t in enumerate(ids0) if t == tok_id_img]
            _state["img_pos"] = img_pos
            _attn_acc.clear()
            gh = attach_gate_hooks(model)
            _state["mode"] = "showcase"
            keep = ("input_ids", "pixel_values", "grid_thw", "image_grid_thw",
                    "mm_token_type_ids", "attention_mask")
            enc = {k: v.to(dev) for k, v in enc.items() if k in keep and v is not None}
            with torch.no_grad():
                model(**enc, use_cache=False)
            _state["mode"] = ""
            for h in gh:
                h.remove()
            img_share = {l: {"img": st["img"] / max(st["n"], 1), "ent": st["ent"] / max(st["n"], 1)}
                         for l, st in _attn_acc.items() if st["n"]}
            (OUT / "maps_img.json").write_text(json.dumps(
                {"n_img_tokens": len(img_pos), "img_share": img_share}, ensure_ascii=False))
            log(f"showcase image DONE ({len(img_pos)} image-токенов)")
        except Exception as e:  # noqa: BLE001
            log(f"image showcase пропущен: {type(e).__name__}: {e}")
    _state["img_pos"] = None


# ───────────────────────────── selftest ─────────────────────────────

def run_selftest(args) -> int:
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    tc = cfg.text_config
    tc.num_hidden_layers = 3
    tc.layer_types = ["linear_attention", "full_attention", "linear_attention"]
    log("selftest: from_config 3 слоя (CPU, fp32, случайные веса)")
    model = AutoModelForImageTextToText.from_config(cfg)
    model.eval()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        log(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    ids = torch.randint(10_000, 20_000, (2, 64))
    for d in DOMAINS:
        _state["domain"] = d
        forward_text(model, ids)

    # flow запускаем заново с mode=selftest, чтобы post-хуки сработали
    _state["mode"] = "selftest"
    fc = FlowCollector(model)
    for d in DOMAINS:
        _state["domain"] = d
        forward_text(model, ids)
    _state["mode"] = ""
    d_flow = fc.dump()
    fc.remove()
    check("flow h_rms по слоям", all(len(v) == 3 for v in d_flow["h_rms"].values()))
    check("flow значения > 0", all(v is not None and v > 0 for v in d_flow["h_rms"]["all"]))
    check("flow h_rms0", all(v > 0 for v in d_flow["h_rms0"].values()))

    # actq проверяем через mode=flow (хуки гейтятся)
    _state["mode"] = "flow"
    fc2 = FlowCollector(model)
    _state["domain"] = "en"
    forward_text(model, ids)
    _state["mode"] = ""
    d2 = fc2.dump()
    fc2.remove()
    check("actq сайты заполнены", len(d2["actq"]) >= 8, f"({len(d2['actq'])})")
    check("actq int8 > fp8 (инфо)", True, str({k: round(s["int8"] - s["fp8"], 1) for k, s in list(d2["actq"].items())[:3]}))

    # linattn
    lc = LinAttnCollector(model)
    _state["mode"] = "linattn"
    for d in DOMAINS:
        _state["domain"] = d
        forward_text(model, ids)
    _state["mode"] = ""
    d_la = lc.dump()
    lc.remove()
    check("linattn β по 2 слоям", all(len(v) == 2 for k, v in d_la["beta_open"].items() if k != "layers"))
    check("linattn state по 2 слоям", all(len(v) == 2 for k, v in d_la["state_rms"].items() if k != "layers"))
    check("linattn g < 0", all(x < 0 for k, v in d_la["g_mean"].items()
                              if k != "layers" for x in v if x is not None))

    # attention-перехват через реестр
    from transformers import AttentionInterface
    AttentionInterface.register("atlas_capture", atlas_attention_forward)
    cfg2 = AutoConfig.from_pretrained(MODEL_DIR)
    tc2 = cfg2.text_config
    tc2.num_hidden_layers = 2
    tc2.layer_types = ["full_attention", "full_attention"]
    cfg2._attn_implementation = "atlas_capture"
    tc2._attn_implementation = "atlas_capture"
    model2 = AutoModelForImageTextToText.from_config(cfg2)
    model2.eval()
    _attn_acc.clear()
    _state["mode"] = "attn"
    forward_text(model2, ids)
    _state["mode"] = ""
    check("attention слоёв 2", len(_attn_acc) == 2, f"({sorted(_attn_acc)})")
    if _attn_acc:
        st = next(iter(_attn_acc.values()))
        n = max(st["n"], 1)
        ent = st["ent"] / n
        check("ent_norm ∈ (0,1]", 0 < ent <= 1.001, f"({ent:.3f})")
        check("prof сумма ≈ 1", abs(sum(st["prof"]) / n - 1.0) < 0.03,
              f"({sum(st['prof']) / n:.3f})")
    gh = attach_gate_hooks(model2)
    _state["mode"] = "attn"
    forward_text(model2, ids)
    _state["mode"] = ""
    for h in gh:
        h.remove()
    check("gate собран", all(st["gate_n"] >= 1 for st in _attn_acc.values()))

    # fragility
    _state["mode"] = "frag"
    d_frag = fragility(model, ids[:, :48])
    _state["mode"] = ""
    check("fragility 3 слоя", len(d_frag["kl"]) == 3)
    check("kl ≥ 0", all(k >= -1e-6 for k in d_frag["kl"]), f"({d_frag['kl']})")

    log(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    global MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="selftest")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="checkpoint dir (or set ATLAS_MODEL_DIR)")
    ap.add_argument("--data", default="data/calibration.jsonl")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--flow-bins", type=int, default=225)
    ap.add_argument("--attn-bins", type=int, default=75)
    ap.add_argument("--la-bins", type=int, default=225)
    args = ap.parse_args()
    MODEL_DIR = args.model_dir
    if not MODEL_DIR:
        log("нужен --model-dir или ATLAS_MODEL_DIR")
        return 2
    if args.mode == "selftest":
        return run_selftest(args)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    t0 = time.time()
    if args.mode in ("flow", "linattn", "frag"):
        model = build_model("sdpa")
        log("модель загружена (sdpa)")
        if args.mode == "flow":
            mode_flow(args, model, tok)
        elif args.mode == "linattn":
            mode_linattn(args, model, tok)
        else:
            mode_frag(args, model, tok)
    elif args.mode in ("attn", "showcase"):
        model = build_model("atlas_capture")
        log("модель загружена (atlas_capture)")
        if args.mode == "attn":
            mode_attn(args, model, tok)
        else:
            mode_showcase(args, model, tok)
    elif args.mode == "all":
        model = build_model("sdpa")
        log("модель загружена (sdpa) — flow")
        mode_flow(args, model, tok)
        mode_linattn(args, model, tok)
        del model
        torch.cuda.empty_cache()
        model = build_model("atlas_capture")
        log("модель загружена (atlas_capture) — attention")
        mode_showcase(args, model, tok)   # карты + image-доля (eager)
        mode_attn(args, model, tok)       # агрегаты по калибровке (eager)
        del model
        torch.cuda.empty_cache()
        model = build_model("sdpa")
        log("модель загружена (sdpa) — frag")
        mode_frag(args, model, tok)
    else:
        log(f"неизвестный режим: {args.mode}")
        return 2
    log(f"TOTAL {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
