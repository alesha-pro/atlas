"""reduce_live.py — сборка public/models/qwen3.8-27b/live.json из артефактов.

Источники (что есть — то берём, остальное пропускаем):
  out/live/{flow,attn,linattn,frag,maps_text,maps_img}.json   — atlas_live.py
  out/neuron_stats.parquet, out/layer_io_sim.parquet          — carve-захват (FFN)
  MODEL_DIR/*.safetensors                                     — A_log/dt_bias → λ по головам

Выход: live.json рядом с atlas.jsonl (одна строка UTF-8, компактные округления).
Запуск на риге:
  cd /mnt/nvme2/projects/qwen38-carve && .venv/bin/python reduce_live.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import argparse

_ap = argparse.ArgumentParser(description="fold atlas_live/carve artifacts into live.json")
_ap.add_argument("--artifacts", default="out/live", help="atlas_live.py output dir")
_ap.add_argument("--carve", default="out", help="dir with neuron_stats.parquet / layer_io_sim.parquet")
_ap.add_argument("--model-dir", default="/mnt/ssd/models/Qwen3.8-27B", help="checkpoint dir (A_log/dt_bias for lambda)")
_ap.add_argument("--dest", default="live.json", help="output live.json path")
_args = _ap.parse_args()

OUT = Path(_args.artifacts)
CARVE = Path(_args.carve)
MODEL_DIR = Path(_args.model_dir)
DEST = Path(_args.dest)
DOMAINS = ("en", "code", "agent")


def r(x, nd=3):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd)


def load_json(name: str):
    p = OUT / name
    if p.exists():
        return json.loads(p.read_text())
    print(f"  (пропущен {name})")
    return None


# ───────────────────────── статические λ из весов ─────────────────────────

def lambda_static(n_layers: int = 64):
    """λ по головам линейных слоёв: exp(−exp(A_log)·softplus(dt_bias))."""
    try:
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open

        index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
        wm = index["weight_map"]
        la, dt = {}, {}
        for name, shard in wm.items():
            if ".linear_attn.A_log" in name or ".linear_attn.dt_bias" in name:
                layer = int(name.split(".layers.")[1].split(".")[0])
                with safe_open(MODEL_DIR / shard, framework="pt") as f:
                    t = f.get_tensor(name)
                (la if "A_log" in name else dt)[layer] = t.float()
        if not la:
            return None
        rows = {}
        for layer, a in sorted(la.items()):
            d = dt[layer]
            g = -a.exp() * F.softplus(d)
            lam = g.exp()                       # доля памяти за один шаг
            rows[layer] = [r(v, 4) for v in lam.tolist()]
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"  (λ_static пропущен: {type(e).__name__}: {e})")
        return None


# ───────────────────────────── нейроны FFN ─────────────────────────────

FIRE_Q = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100)


def neurons_block() -> dict | None:
    p = CARVE / "neuron_stats.parquet"
    if not p.exists():
        print("  (пропущен neuron_stats.parquet)")
        return None
    df = pd.read_parquet(p)
    g = df.groupby("layer")
    out = {
        "n": int(df.neuron.max()) + 1,
        "fire_mean": [], "fire_p99": [], "dead_frac": [], "conc": [], "spec_frac": [],
        "heat": [], "examples": [],
    }
    for layer, d in g:
        out["fire_mean"].append(r(d.fire_rate.mean(), 5))
        out["fire_p99"].append(r(d.fire_rate.quantile(0.99), 5))
        out["dead_frac"].append(r((d.fire_rate < 0.0002).mean(), 4))
        out["conc"].append(r(d.top1pct_share.mean(), 4))
        s = (d.fire_rate_code - d.fire_rate_en).abs() / (
            d.fire_rate_code + d.fire_rate_en + 1e-9)
        out["spec_frac"].append(r((s > 0.7).mean(), 4))
        out["heat"].append([r(v, 5) for v in d.fire_rate.quantile([q / 100 for q in FIRE_Q]).tolist()])
    # самые поляризованные en↔code нейроны всей модели
    d = df.copy()
    d["pol"] = (d.fire_rate_code - d.fire_rate_en).abs() / (
        d.fire_rate_code + d.fire_rate_en + 1e-9)
    d = d[(d.fire_rate_code + d.fire_rate_en) > 0.01]
    top = d.nlargest(10, "pol")
    for _, row in top.iterrows():
        out["examples"].append({
            "layer": int(row.layer), "neuron": int(row.neuron),
            "en": r(row.fire_rate_en, 5), "code": r(row.fire_rate_code, 5),
            "agent": r(row.fire_rate_agent, 5), "pol": r(row.pol, 3)})
    return out


# ───────────────────────────── сборка ─────────────────────────────

def main() -> int:
    live: dict = {"schema": 1}
    flow = load_json("flow.json")
    attn = load_json("attn.json")
    lin = load_json("linattn.json")
    frag = load_json("frag.json")
    maps = load_json("maps_text.json")
    img = load_json("maps_img.json")
    cap_meta_p = CARVE / "capture_meta.json"
    cap = json.loads(cap_meta_p.read_text()) if cap_meta_p.exists() else {}

    n_layers = 64
    meta = {
        "when": time.strftime("%F %T"),
        "model": "Qwen3.8-27B", "dtype": "bfloat16",
        "seqlen": 2048, "batch": 4,
        "domains": list(DOMAINS),
        "calibration": "en: wikitext-2 + C4 · code: the-stack (py/js/ts/go/c/cpp/java/sh/rb) · agent: SWE-agent + agent-traces",
    }
    if cap:
        meta["carve_tokens"] = cap.get("tokens_per_domain")
        meta["carve_k_fire"] = cap.get("k_fire")
    live["meta"] = meta

    # flow
    if flow:
        live["flow"] = {
            "h_rms": {d: [r(v, 2) for v in flow["h_rms"][d]] for d in DOMAINS + ("all",)},
            "delta_rms": {d: [r(v, 2) for v in flow["delta_rms"][d]] for d in DOMAINS + ("all",)},
            "out_ratio": [r(v, 2) for v in flow["out_ratio"]["all"]],
            "n_out_dims": flow["n_out_dims"]["all"],
            "h_rms0": {d: r(flow["h_rms0"].get(d), 2) for d in DOMAINS + ("all",)},
            "actq": {site: {"int8": r(s["int8"], 1), "fp8": r(s["fp8"], 1)}
                     for site, s in flow["actq"].items()},
        }
        meta["flow_bins"] = len(flow["h_rms"]["all"])
    io_p = CARVE / "layer_io_sim.parquet"
    if io_p.exists():
        io = pd.read_parquet(io_p).sort_values("layer")
        live.setdefault("flow", {})["io_cos"] = [r(v, 4) for v in io.io_cosine.tolist()]
        live["flow"]["io_cos_source"] = "carve 4M tokens"

    # attn
    if attn:
        layers = sorted(int(k) for k in attn)
        live["attn"] = {
            "layers": layers,
            "ent": [r(attn[str(l)]["ent"], 3) for l in layers],
            "ent_std": [r(attn[str(l)]["ent_std"], 3) for l in layers],
            "first": [r(attn[str(l)]["first"], 4) for l in layers],
            "diag": [r(attn[str(l)]["diag"], 4) for l in layers],
            "gate": [r(attn[str(l)]["gate"], 3) for l in layers],
            "prof": [[r(v, 4) for v in attn[str(l)]["prof"]] for l in layers],
            "decay_edges": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        }
    if maps:
        live["attn_maps"] = {
            "prompt": maps["prompt"],
            "tokens": maps["tokens"],
            "maps": {k: {"star_head": v["star_head"], "mean": v["mean"], "star": v["star"]}
                     for k, v in maps["maps"].items()},
        }

    # linattn
    lam = lambda_static()
    if lin:
        def strip_none(vals):
            return [r(v, 4) for v in vals]
        live["linattn"] = {
            "layers": lin["beta_open"]["layers"],
            "beta": {d: strip_none(lin["beta_open"][d]) for d in DOMAINS},
            "g_mean": {d: strip_none(lin["g_mean"][d]) for d in DOMAINS},
            "state_rms": {d: strip_none(lin["state_rms"][d]) for d in DOMAINS},
        }
        gm = lin["g_mean"]["en"]
        gm_all = [(a + b + c) / 3 for a, b, c in zip(
            lin["g_mean"]["en"], lin["g_mean"]["code"], lin["g_mean"]["agent"])]
        live["linattn"]["half_life"] = [
            r(math.log(2) / abs(v), 1) if v and v < -1e-6 else None for v in gm_all]
    if lam:
        live.setdefault("linattn", {})["layers_lambda"] = lam

    # fragility
    if frag:
        live["fragility"] = {
            "kl": [r(v, 4) for v in frag["kl"]],
            "logit_cos": [r(v, 5) for v in frag["logit_cos"]],
        }

    # vision showcase
    if img:
        live["vision"] = {
            "n_img_tokens": img["n_img_tokens"],
            "img_share": {k: r(v["img"], 4) for k, v in img["img_share"].items()},
        }

    # neurons
    nb = neurons_block()
    if nb:
        live["neurons"] = nb

    DEST.write_text(json.dumps(live, ensure_ascii=False, separators=(",", ":")))
    sz = DEST.stat().st_size / 1e6
    print(f"OK → {DEST} ({sz:.1f} MB), ключи: {sorted(live)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
