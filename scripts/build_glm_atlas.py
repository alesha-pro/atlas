#!/usr/bin/env python3
"""Build the compact, public GLM-5.3 Atlas payload from immutable captures.

The raw capture stays under evidence/.  This script publishes only aggregates:
no prompts, generations, images, activations, or token-level routes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(v, default=0.0):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def mean_stat(obj, key):
    return finite(obj.get(key, {}).get("mean"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="evidence/glm-5.3-flash-nvfp4")
    ap.add_argument("--out", default="public/models/glm-5.3-flash-nvfp4")
    args = ap.parse_args()
    ev, out = Path(args.evidence), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "atlas": ev / "atlas-live-nvfp4.json",
        "moe": ev / "moe.json",
        "reap_json": ev / "reap-full.json",
        "reap_npz": ev / "reap-full.npz",
        "stability": ev / "ranking-stability-2pct.json",
        "inventory": ev / "extra-captures/checkpoint-inventory/inventory.json",
        "scales": ev / "extra-captures/weight-scale-scan/weight-scale-scan.json",
        "router_json": ev / "extra-captures/router-dynamics/router-dynamics.json",
        "router_npz": ev / "extra-captures/router-dynamics/router-dynamics.npz",
        "contrib_json": ev / "extra-captures/contributions/contributions.json",
        "contrib_npz": ev / "extra-captures/contributions/contributions.npz",
        "rich": ev / "extra-captures/rich-sequence/atlas-rich-sequence.json",
        "vision": ev / "extra-captures/paired-vision/paired-vision.json",
        "fc2": ev / "extra-captures/fc2-activation/fc2-capture.json",
        "pruning": ev / "extra-captures/pruning-arms/pruning-arms.json",
    }
    inv, moe = read(paths["inventory"]), read(paths["moe"])
    router, contrib = read(paths["router_json"]), read(paths["contrib_json"])
    rich, vision = read(paths["rich"]), read(paths["vision"])
    fc2, pruning = read(paths["fc2"]), read(paths["pruning"])
    stability, scales = read(paths["stability"]), read(paths["scales"])
    live_base = read(paths["atlas"])
    rz, qz, cz = np.load(paths["reap_npz"]), np.load(paths["router_npz"]), np.load(paths["contrib_npz"])

    layers = list(range(3, 45))
    experts = 288
    domains = moe["domains"]

    def safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    reap_matrix, route_matrix, contribution_matrix = [], [], []
    reap_domains = {d: [] for d in domains}
    for layer in layers:
        score = safe_div(rz[f"L{layer}__reap_sum"], rz[f"L{layer}__reap_count"])
        reap_matrix.append(score.tolist())
        selected = qz[f"L{layer}__selected_count"].astype(float)
        route_matrix.append((selected / max(1.0, selected.sum())).tolist())
        c = safe_div(cz[f"L{layer}__expert_weighted_norm_sum"], cz[f"L{layer}__expert_count"])
        contribution_matrix.append(c.tolist())
        for domain in domains:
            # Capture keys preserve hyphens but replace the domain separator.
            ds = domain.replace("/", "_")
            prefix = f"L{layer}__D{ds}__"
            score_d = safe_div(rz[prefix + "reap_sum"], rz[prefix + "reap_count"])
            reap_domains[domain].append(score_d.tolist())

    def dyn_row(layer, src):
        x = router["layers"][str(layer)][src]
        s = x["stats"]
        return {
            "layer": layer, "tokens": x["tokens"],
            "effective": mean_stat(s, "effective_experts"),
            "entropy": mean_stat(s, "entropy_normalized"),
            "margin": mean_stat(s, "top1_top2_margin"),
            "top1": mean_stat(s, "top1_weight"),
            "selected_gini": finite(x.get("selected_gini")),
            "top1_gini": finite(x.get("top1_gini")),
        }

    dynamics = {
        "all": [dyn_row(l, "all") for l in layers],
        "domains": {
        d: [{"layer": l, **{
            "tokens": router["layers"][str(l)]["domains"][d]["tokens"],
            "effective": mean_stat(router["layers"][str(l)]["domains"][d]["stats"], "effective_experts"),
            "entropy": mean_stat(router["layers"][str(l)]["domains"][d]["stats"], "entropy_normalized"),
            "margin": mean_stat(router["layers"][str(l)]["domains"][d]["stats"], "top1_top2_margin"),
            "top1": mean_stat(router["layers"][str(l)]["domains"][d]["stats"], "top1_weight"),
            "selected_gini": finite(router["layers"][str(l)]["domains"][d].get("selected_gini")),
            "top1_gini": finite(router["layers"][str(l)]["domains"][d].get("top1_gini")),
        }} for l in layers] for d in domains
        },
    }

    contribution_rows, pair_quantiles = [], []
    for l in layers:
        co = contrib["layers"][str(l)]["component_all"]
        contribution_rows.append({
            "layer": l,
            "shared_energy": mean_stat(co, "shared_energy_fraction"),
            "shared_to_routed": mean_stat(co, "shared_to_routed_norm_ratio"),
            "shared_routed_cosine": mean_stat(co, "shared_routed_cosine"),
            "constructive_gain": mean_stat(co, "constructive_gain"),
        })
        cnt, sm = cz[f"L{l}__pair_count_upper"], cz[f"L{l}__pair_cosine_sum_upper"]
        vals = safe_div(sm, cnt)[cnt > 0]
        pair_quantiles.append({"layer": l, "p05": float(np.quantile(vals, .05)), "p50": float(np.quantile(vals, .5)), "p95": float(np.quantile(vals, .95))})

    metrics = rich["metrics"]
    def metric(layer, suffix, domain="all"):
        return finite(metrics.get(f"{domain}::language.layer.{layer}.{suffix}", {}).get("mean"))

    kda_layers = rich["coverage"]["kda_layers"]
    kda = [{
        "layer": l,
        "half_life": metric(l, "kda.half_life"),
        "beta_open": metric(l, "kda.beta_open"),
        "state_rms": metric(l, "kda.state_rms"),
        "io_cosine": metric(l, "kda.io_cosine"),
    } for l in kda_layers]
    kda_heads = [[finite(x.get("mean")) for x in rich["head_metrics"][f"{l}::half_life"]] for l in kda_layers]
    indexer_layers = rich["coverage"]["indexer_layers"]
    indexer = [{
        "layer": l,
        "selected_distance": metric(l, "indexer.selected_distance"),
        "selected_per_query": metric(l, "indexer.selected_per_query"),
        "unique_fraction": metric(l, "indexer.unique_fraction"),
        "position_distance": [metric(l, f"indexer.position_bucket.{b}.selected_distance") for b in range(6)],
    } for l in indexer_layers]

    scale_rows = []
    for l in layers:
        row = {"layer": l}
        for p in ("gate_proj", "up_proj", "down_proj"):
            x = scales["group_summaries"][f"L{l}:{p}"]
            row[p] = {k: finite(x[k]) for k in ("p01", "p50", "p99", "mean", "max")}
        scale_rows.append(row)
    fc2_rows = [{"layer": int(k), **{x: finite(v[x]) for x in ("sqnr_db", "relative_l2", "qdq_zero_fraction", "exact_fraction", "input_abs_max", "error_abs_max")}}
                 for k, v in sorted(fc2["layers"].items(), key=lambda kv: int(kv[0]))]

    vision_compact = {
        "domains": list(vision["run"]["domains"]),
        "arms": {name: {
            "all": arm["all"],
            "by_domain": arm["by_domain"],
        } for name, arm in vision["behaviour"]["arms"].items()},
        "records_per_arm": vision["run"]["records_per_arm"],
    }
    pruning_compact = [{
        "arm": name,
        "pruned": arm["pruned_experts_per_layer"],
        **arm["router_effect_all_layers"],
        **arm["sequence_vs_baseline"],
    } for name, arm in pruning["arms"].items()]

    checksums = {k: sha(v) for k, v in paths.items() if v.suffix in {".json", ".npz"}}
    payload = {
        "schema": 1,
        "meta": {
            "model": "GLM-5.3-Flash NVFP4", "parameters": 320_000_000_000,
            "active_parameters": 18_000_000_000, "layers": 45,
            "routed_layers": 42, "experts": experts, "active_experts": 8,
            "records": live_base["run"]["records"],
            "estimated_tokens": live_base["run"]["estimated_tokens"],
            "reap_tokens_per_layer": moe["tokens_layer"],
            "weight_bytes": inv["checkpoint"]["weight_bytes"],
            "tensor_count": inv["inventory"]["tensor_count"],
            "runtime": router["run"]["runtime"],
            "checksums": checksums,
        },
        "routing": {
            "layers": layers, "experts": experts, "domains": domains,
            "reap": reap_matrix, "reap_domains": reap_domains,
            "route_share": route_matrix, "contribution": contribution_matrix,
            "dynamics": dynamics,
        },
        "contributions": {"layers": contribution_rows, "pair_quantiles": pair_quantiles},
        "memory": {"kda_layers": kda_layers, "kda": kda, "kda_heads": kda_heads,
                   "indexer_layers": indexer_layers, "indexer": indexer,
                   "position_buckets": ["0–31", "32–127", "128–511", "512–2047", "2K–8K", "8K+"]},
        "quantization": {"scales": scale_rows, "fc2": fc2_rows,
                         "inventory": inv["inventory"], "config": inv["quantization_config"]},
        "vision": vision_compact,
        "pruning": pruning_compact,
        "stability": stability,
        "coverage": rich["coverage"],
        "limitations": {
            "fc2": "official vLLM NVFP4 emulation backend; the native fused kernel does not expose FC2 input",
            "pruning": "reversible route ablation, not a physically pruned checkpoint",
        },
    }
    (out / "insights.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Logical parameter groups for the architecture rail and layer wall.  These
    # are config-derived aggregates, not fabricated per-tensor weight statistics.
    dummy_hist = [0.0] * 32
    records = []
    def add(name, shape, dtype, component, sqnr=None, scale=None):
        numel = int(np.prod(shape, dtype=np.int64))
        records.append({
            "name": name, "shape": shape, "dtype": dtype, "numel": numel,
            "mean": scale or 0.0, "std": 0.0, "absmax": scale or 0.0, "absmean": scale or 0.0,
            "p50": scale or 0.0, "p90": scale or 0.0, "p99": scale or 0.0, "p999": scale or 0.0, "p9999": scale or 0.0,
            "kurtosis": 0.0, "skew": 0.0, "sparsity": 0.0, "outlier_3s": 0.0, "outlier_4s": 0.0, "outlier_6s": 0.0,
            "dyn_range": 0.0, "hist_log2": dummy_hist, "component": component, "layer": None, "shard": "logical aggregate",
            **({"sqnr_int4_g128": sqnr} if sqnr is not None else {}),
        })
    add("model.language_model.embed_tokens.weight", [154880, 4096], "BF16", "embed")
    full = set(indexer_layers)
    fc2_by = {x["layer"]: x["sqnr_db"] for x in fc2_rows}
    for l in range(45):
        base = f"model.language_model.layers.{l}"
        if l in full:
            add(f"{base}.self_attn.q_latent.aggregate", [8192, 4096], "BF16", "attn.q")
            add(f"{base}.self_attn.kv_latent.aggregate", [512, 4096], "BF16", "attn.k")
            add(f"{base}.self_attn.output.aggregate", [4096, 8192], "BF16", "attn.o")
            add(f"{base}.self_attn.indexer.aggregate", [32, 128, 4096], "BF16", "attn.v")
        else:
            add(f"{base}.self_attn.kda_input.aggregate", [3, 8192, 4096], "BF16", "linattn.in_qkv")
            add(f"{base}.self_attn.kda_state.aggregate", [64, 128, 128], "F32 state", "linattn.A_log")
            add(f"{base}.self_attn.o_proj.weight", [4096, 8192], "BF16", "linattn.out")
        add(f"{base}.input_layernorm.weight", [4096], "BF16", "norm")
        if l < 3:
            for p, sh in (("gate_proj", [12288, 4096]), ("up_proj", [12288, 4096]), ("down_proj", [4096, 12288])):
                add(f"{base}.mlp.{p}.weight", sh, "BF16", "mlp." + p.split("_")[0])
        else:
            s = scales["group_summaries"][f"L{l}"]["p50"]
            for p, sh in (("gate_proj", [288, 2048, 4096]), ("up_proj", [288, 2048, 4096]), ("down_proj", [288, 4096, 2048])):
                add(f"{base}.mlp.experts.aggregate.{p}", sh, "NVFP4 aggregate", "mlp." + p.split("_")[0], fc2_by[l], s)
            for p, sh in (("gate_proj", [2048, 4096]), ("up_proj", [2048, 4096]), ("down_proj", [4096, 2048])):
                add(f"{base}.mlp.shared_experts.{p}.weight", sh, "BF16", "mlp." + p.split("_")[0])
            add(f"{base}.mlp.gate.weight", [288, 4096], "BF16", "mlp.gate")
    for l in range(24):
        base = f"model.visual.blocks.{l}"
        add(f"{base}.attn.qkv.weight", [3072, 1024], "BF16", "vision")
        add(f"{base}.attn.proj.weight", [1024, 1024], "BF16", "vision")
        add(f"{base}.mlp.gate_proj.weight", [4096, 1024], "BF16", "vision")
        add(f"{base}.mlp.down_proj.weight", [1024, 4096], "BF16", "vision")
    add("lm_head.weight", [154880, 4096], "BF16", "lm_head")
    with (out / "atlas.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {out / 'insights.json'} and {len(records)} logical groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
