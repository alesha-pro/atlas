"""run_capture.py — этап 1: съём FFN-активаций Qwen3.8-27B.

Режимы:
- `--mode selftest` (CPU, без весей): 2-слойная модель from_config со
  случайными весами (linear_attention + full_attention), текст-only forward,
  полный путь хуков → dump → ассерты. Проверяет сигнатуру forward и пайплайн
  до GPU-окна.
- `--mode gpu`: полный чекпоинт BF16, device_map=balanced на 4 карты,
  калибровка data/calibration.jsonl пакуется в бины seqlen (бины не смешивают
  домены), прогон через модель, dump в out/.

Выход: out/{neuron_stats.parquet, coact_sketch.npz, layer_io_sim.parquet,
capture_meta.json}.

Запуск на риге:
  .venv/bin/python run_capture.py --mode selftest
  .venv/bin/python run_capture.py --mode gpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, ".")
from carve_hooks import DOMAINS, CarveCollector  # noqa: E402

MODEL_DIR = os.environ.get("ATLAS_MODEL_DIR", "")   # или --model-dir


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pack_bins(tok, path: str, seqlen: int, max_records: int = 0):
    """jsonl (домены блоками) → список бинов (domain, ids[<=seqlen]).

    Внутри домена документы склеиваются подряд и режутся на бины seqlen;
    хвост домена < seqlen идёт последним коротким бином. Домены не смешиваются.
    """
    bins: list[tuple[str, list[int]]] = []
    cur_dom, cur = None, []
    tok_dom: dict[str, int] = {}
    n = 0
    with open(path) as f:
        for line in f:
            n += 1
            if max_records and n > max_records:
                break
            rec = json.loads(line)
            d, ids = rec["domain"], None
            ids = tok(rec["text"], add_special_tokens=False).input_ids
            tok_dom[d] = tok_dom.get(d, 0) + len(ids)
            if d != cur_dom and cur:
                bins.append((cur_dom, cur))
                cur = []
            cur_dom = d
            cur.extend(ids)
            while len(cur) >= seqlen:
                bins.append((cur_dom, cur[:seqlen]))
                cur = cur[seqlen:]
    if cur:
        bins.append((cur_dom, cur))
    return bins, tok_dom


def forward_text(model, ids: torch.Tensor) -> str:
    """Текст-only forward без lm_head: сначала model.model, фолбэк — полный."""
    inner = model.model
    try:
        inner(input_ids=ids, use_cache=False)
        return "model.model(input_ids=...)"
    except Exception as e:  # noqa: BLE001
        log(f"model.model.forward не принял вызов ({type(e).__name__}: {e}); фолбэк на полный forward")
        model(input_ids=ids, use_cache=False)
        return "model(input_ids=...)"


def run_selftest(args) -> int:
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    tc = cfg.text_config
    tc.num_hidden_layers = 2
    tc.layer_types = ["linear_attention", "full_attention"]
    log("from_config (2 слоя, случайные веса, CPU fp32)…")
    model = AutoModelForImageTextToText.from_config(cfg)
    model.eval()

    coll = CarveCollector(model, k_fire=args.k_fire, sketch_n=512)
    log(f"hooked layers: {sorted(coll.neurons)}")

    torch.manual_seed(0)
    how = ""
    for d in DOMAINS:
        coll.set_domain(d)
        ids = torch.randint(10_000, 20_000, (1, 64))
        how = forward_text(model, ids)
        log(f"  {d}: forward ok ({how})")

    out = Path(args.out or "out/capture_selftest")
    paths = coll.dump(out)

    # -- ассерты ---------------------------------------------------------
    df = pd.read_parquet(paths["neuron_stats"])
    inter = tc.intermediate_size
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        log(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    check("rows == 2*intermediate", len(df) == 2 * inter, f"({len(df)})")
    check("both layer types hooked", set(df.layer) == {0, 1})
    check("все домены покрыты", all(coll.neurons[i].tok_dom[d] > 0 for i in (0, 1) for d in DOMAINS))
    check("layer_io по 2 слоям", set(coll.layer_io) == {0, 1} and len(coll.layer_io[0]) == 3)
    log(f"SELFTEST {'PASSED' if ok else 'FAILED'} | forward: {how}")

    # -- упаковка бинов на реальных данных (юнит-проверка) ----------------
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    bins, tok_dom = pack_bins(tok, args.data, args.seqlen, max_records=60)
    doms_in_order = [d for d, _ in bins[:3]] + [d for d, _ in bins[-3:]]
    check("бины не смешивают домены", all(len({d for d, _ in bins}) >= 1 for _ in [0]) and
          all(len(b) <= args.seqlen for _, b in bins))
    log(f"packing: {len(bins)} бинов из 60 записей, хвосты доменов: {tok_dom}")

    sys.exit(0 if ok else 1)


def run_gpu(args) -> int:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    log("паковка калибровки в бины…")
    bins, tok_dom = pack_bins(tok, args.data, args.seqlen)
    if args.max_bins:
        bins = bins[: args.max_bins]
    full = [b for b in bins if len(b[1]) == args.seqlen]
    tails = [b for b in bins if len(b[1]) < args.seqlen]
    log(f"bins={len(bins)} (full={len(full)}, tails={len(tails)}), "
        f"tokens: { {k: round(v / 1e6, 2) for k, v in tok_dom.items()} }M")

    log("загрузка чекпоинта BF16, device_map=balanced…")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="balanced"
    )
    model.eval()
    dev0 = model.model.language_model.embed_tokens.weight.device
    log(f"модель на картах за {time.time() - t0:.0f}s, embed на {dev0}")

    coll = CarveCollector(model, k_fire=args.k_fire, sketch_n=args.sketch)

    # бины доменно-непрерывны по построению; батчи режем, не смешивая домены
    batches, cur = [], []
    for b in full:
        if cur and (cur[0][0] != b[0] or len(cur) == args.batch):
            batches.append(cur)
            cur = []
        cur.append(b)
    if cur:
        batches.append(cur)
    log(f"hooked {len(coll.neurons)} слоёв, k_fire={args.k_fire}, sketch={args.sketch}")

    def run_batch(chunk):
        doms = {d for d, _ in chunk}
        assert len(doms) == 1, f"бины смешали домены: {doms}"
        coll.set_domain(next(iter(doms)))
        ids = torch.tensor([b for _, b in chunk], dtype=torch.long, device=dev0)
        forward_text(model, ids)

    with torch.no_grad():
        for bi, chunk in enumerate(batches):
            run_batch(chunk)
            if bi % 25 == 0 or bi == len(batches) - 1:
                done_tok = (bi + 1) * args.batch * args.seqlen
                log(f"batch {bi + 1}/{len(batches)} (~{done_tok / 1e6:.2f}M ток, "
                    f"{time.time() - t0:.0f}s)")
        for tb in tails:
            run_batch([tb])

    log("finalize + dump…")
    paths = coll.dump(args.out or "out")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak = 0.0
    meta = {
        "finished_at": time.strftime("%F %T"),
        "wall_s": round(time.time() - t0, 1),
        "tokens_per_domain": tok_dom,
        "bins_full": len(full), "bins_tail": len(tails), "batch": args.batch,
        "seqlen": args.seqlen, "k_fire": args.k_fire, "sketch_n": args.sketch,
        "peak_gpu_mem_gb_sum": round(peak, 1),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "layer_io_batches": len(coll.layer_io.get(0, [])),
    }
    (Path(args.out or "out") / "capture_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"DONE {meta}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("selftest", "gpu"), required=True)
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="checkpoint dir (or set ATLAS_MODEL_DIR)")
    ap.add_argument("--data", default="data/calibration.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--k-fire", type=int, default=64)
    ap.add_argument("--sketch", type=int, default=2048)
    ap.add_argument("--max-bins", type=int, default=0, help="0 = все (отладка)")
    args = ap.parse_args()
    if args.mode == "selftest":
        return run_selftest(args)
    return run_gpu(args)


if __name__ == "__main__":
    sys.exit(main())
