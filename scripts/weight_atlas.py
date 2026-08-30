#!/usr/bin/env python3
"""Weight Atlas: полная карта весов чекпойнта + измеренная квантуемость.

Стримит шарды safetensors, каждый тензор считает на GPU:
  - распределение: mean/std/absmax/percentiles/kurtosis/skew/sparsity
  - выбросы: доли за 3/4/6 сигм, dynamic range
  - гистограмма log2|w| (для визуализации)
  - канальные выбросы: разброс amax по строкам и столбцам (решает судьбу
    per-channel квантования)
  - спектр: топ сингулярных значений (lowrank SVD), stable rank
  - СИМУЛЯЦИЯ КВАНТОВАНИЯ: INT8 per-channel, INT4 group-128, FP8 e4m3 —
    реальный SQNR в дБ, то есть измеренный ответ «что можно квантовать»

Выход: JSONL по тензору + сводка.
    python3 weight_atlas.py <checkpoint_dir> <out.jsonl> [--device cuda]
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

HIST_BINS = list(range(-24, 5))  # log2|w| от 2^-24 до 2^4


def q_int8_per_channel(w: torch.Tensor) -> torch.Tensor:
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s = amax / 127.0
    return torch.round(w / s).clamp(-127, 127) * s


def q_int4_group(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    *lead, k = w.shape
    if k % group:
        return None
    g = w.reshape(*lead, k // group, group)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s = amax / 7.0
    return (torch.round(g / s).clamp(-7, 7) * s).reshape(*lead, k)


def q_fp8_e4m3(w: torch.Tensor) -> torch.Tensor:
    amax = w.abs().amax().clamp(min=1e-12)
    s = amax / 448.0
    return (w / s).to(torch.float8_e4m3fn).float() * s


def sqnr_db(w: torch.Tensor, wq: torch.Tensor) -> float:
    err = (w - wq).pow(2).sum()
    sig = w.pow(2).sum()
    if err.item() == 0:
        return 999.0
    return float(10.0 * torch.log10(sig / err).item())


def classify(name: str) -> tuple[str, int | None, int | None]:
    """component, layer_idx, expert_idx"""
    layer = None
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        layer = int(m.group(1))
    expert = None
    m = re.search(r"experts\.(\d+)\.", name)
    if m:
        expert = int(m.group(1))

    patterns = [
        ("embed", r"embed_tokens"), ("lm_head", r"lm_head"),
        ("norm", r"norm|layernorm|ln_"),
        ("attn.q", r"q_proj"), ("attn.k", r"k_proj"),
        ("attn.v", r"v_proj"), ("attn.o", r"o_proj"),
        ("linattn.in_a", r"in_proj_a"), ("linattn.in_b", r"in_proj_b"),
        ("linattn.in_z", r"in_proj_z"), ("linattn.out", r"out_proj"),
        ("linattn.conv", r"conv1d"), ("linattn.dt", r"dt_bias|A_log"),
        ("moe.router", r"gate\.weight|router"),
        ("moe.shared", r"shared_expert"),
        ("mlp.gate", r"gate_proj"), ("mlp.up", r"up_proj"),
        ("mlp.down", r"down_proj"),
        ("vision", r"visual\."),
    ]
    for label, pat in patterns:
        if re.search(pat, name):
            return label, layer, expert
    return "other", layer, expert


def analyze(name: str, t: torch.Tensor, device: str, spectral: bool) -> dict:
    w = t.to(device=device, dtype=torch.float32)
    n = w.numel()
    if n == 0:
        return {}
    flat = w.reshape(-1)
    absf = flat.abs()
    mean = flat.mean()
    std = flat.std() if n > 1 else torch.zeros((), device=w.device)
    centered = flat - mean
    m2 = centered.pow(2).mean()
    kurt = float((centered.pow(4).mean() / m2.pow(2) - 3).item()) if m2 > 0 else 0.0
    skew = float((centered.pow(3).mean() / m2.pow(1.5)).item()) if m2 > 0 else 0.0
    amax = float(absf.max().item())

    qs = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999], device=w.device)
    # quantile ограничен размером входа — считаем по подвыборке для больших
    sample = absf if n <= 16_000_000 else absf[torch.randint(0, n, (16_000_000,), device=w.device)]
    pcts = torch.quantile(sample.float(), qs).tolist()

    s = float(std.item())
    out3 = float((centered.abs() > 3 * s).float().mean().item()) if s > 0 else 0.0
    out4 = float((centered.abs() > 4 * s).float().mean().item()) if s > 0 else 0.0
    out6 = float((centered.abs() > 6 * s).float().mean().item()) if s > 0 else 0.0

    logs = torch.log2(absf.clamp(min=1e-30))
    hist = torch.histc(logs, bins=len(HIST_BINS), min=HIST_BINS[0], max=HIST_BINS[-1] + 1)
    hist = (hist / n).tolist()

    rec = {
        "name": name, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
        "numel": n,
        "mean": round(float(mean.item()), 8), "std": round(s, 8),
        "absmax": round(amax, 6), "absmean": round(float(absf.mean().item()), 8),
        "p50": round(pcts[0], 8), "p90": round(pcts[1], 8), "p99": round(pcts[2], 8),
        "p999": round(pcts[3], 8), "p9999": round(pcts[4], 8),
        "kurtosis": round(kurt, 3), "skew": round(skew, 3),
        "sparsity": round(float((absf < 1e-6).float().mean().item()), 6),
        "outlier_3s": round(out3, 8), "outlier_4s": round(out4, 8), "outlier_6s": round(out6, 8),
        "dyn_range": round(amax / max(pcts[0], 1e-12), 2),
        "hist_log2": [round(x, 6) for x in hist],
    }

    if w.dim() == 2:
        row_amax = w.abs().amax(dim=1)
        col_amax = w.abs().amax(dim=0)
        rec["row_amax_ratio"] = round(float((row_amax.max() / row_amax.median().clamp(min=1e-12)).item()), 3)
        rec["col_amax_ratio"] = round(float((col_amax.max() / col_amax.median().clamp(min=1e-12)).item()), 3)
        rec["row_amax_p99"] = round(float(torch.quantile(row_amax.float(), 0.99).item()), 6)

        # квантование: измеряем, а не гадаем
        rec["sqnr_int8_ch"] = round(sqnr_db(w, q_int8_per_channel(w)), 2)
        q4 = q_int4_group(w, 128)
        if q4 is not None:
            rec["sqnr_int4_g128"] = round(sqnr_db(w, q4), 2)
        try:
            rec["sqnr_fp8_e4m3"] = round(sqnr_db(w, q_fp8_e4m3(w)), 2)
        except (RuntimeError, TypeError):
            pass

        if spectral and min(w.shape) >= 16:
            try:
                q = min(32, min(w.shape) - 1)
                _, sv, _ = torch.svd_lowrank(w, q=q, niter=2)
                sv = sv.float()
                fro2 = w.pow(2).sum()
                rec["sv_top"] = [round(float(x), 4) for x in sv[:16].tolist()]
                rec["stable_rank"] = round(float((fro2 / sv[0].pow(2)).item()), 2)
                rec["sv_decay"] = round(float((sv[-1] / sv[0]).item()), 5)
            except RuntimeError:
                pass

    comp, layer, expert = classify(name)
    rec["component"], rec["layer"], rec["expert"] = comp, layer, expert
    del w
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-spectral", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    shards = sorted(ckpt.glob("*.safetensors"))
    out = Path(args.out)
    n_done = 0
    t0 = time.time()
    with out.open("w") as f:
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    rec = analyze(name, sf.get_tensor(name), args.device, not args.no_spectral)
                    if rec:
                        rec["shard"] = shard.name
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        n_done += 1
                        if n_done % 25 == 0:
                            print(f"{n_done} тензоров, {time.time()-t0:.0f}s", flush=True)
                    if args.limit and n_done >= args.limit:
                        print("LIMIT")
                        return
    print(f"DONE {n_done} тензоров за {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
