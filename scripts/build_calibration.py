"""build_calibration.py — калибровочный микс ~4M токенов → data/calibration.jsonl.

Домены (поле `domain` в каждой записи; ru убран по решению Алексея 22.08):
- en    ~1.6M ток: wikitext-2 train (25% бюджета домена) + C4 en;
- code  ~1.2M ток: codeparrot/github-code (the-stack* gated), фильтр по
        расширениям: py/js/ts/go/c/h/cpp/java/sh/rb;
- agent ~1.2M ток: nebius/SWE-agent-trajectories (реальные SWE-agent прогоны,
        75% бюджета, рендер trajectory как диалог) + trace-commons/agent-traces
        (реальные сессии coding-агентов claude-code/codex/cursor/opencode).

Запись = документ/чанк, обрезанный до MAX_RECORD_CHARS (diversity: без капа
agent-домен съезжал в ~60 гигантских траекторий). Нарезка на seqlen — этап 1.
Токены считаются токенизатором Qwen3.8-27B, без спец-токенов.

Запуск на риге:
  python3 build_calibration.py --model-dir <ckpt> --out data/calibration.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_DIR = os.environ.get("ATLAS_MODEL_DIR", "")   # или --model-dir
WIKITEXT_FRAC = 0.25  # доля wikitext внутри en-домена
MAX_RECORD_CHARS = 24_000


def ntok(tok, texts: list[str]) -> int:
    return sum(len(x) for x in tok(texts, add_special_tokens=False).input_ids)


def iter_en(tok, budget: int, seed: int):
    got = 0
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train").shuffle(seed=seed)
    wt_budget = int(budget * WIKITEXT_FRAC)
    for rec in wt:
        t = rec["text"].strip()
        if not t:
            continue
        if got >= wt_budget:
            break
        t = t[:MAX_RECORD_CHARS]
        got += ntok(tok, [t])
        yield "en", t
    # C4 — остальное
    c4 = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                      split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    for rec in c4:
        t = rec["text"].strip()
        if not t:
            continue
        if got >= budget:
            break
        t = t[:MAX_RECORD_CHARS]
        got += ntok(tok, [t])
        yield "en", t


CODE_EXTS = {".py", ".js", ".ts", ".go", ".c", ".h", ".cpp", ".java", ".sh", ".rb"}


def iter_code(tok, budget: int, seed: int):
    """github-code (codeparrot): открытый, языки вперемешку — фильтр по
    расширению. the-stack* gated; скрипт-лоадер datasets 4.x не принимает,
    поэтому паркеты напрямую (первые 2 шарда, ~200k файлов, хватает)."""
    import os as _os

    got = 0
    files = [f"hf://datasets/codeparrot/github-code/data/train-{i:05d}-of-01126.parquet"
             for i in range(2)]
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    for rec in ds:
        t = rec.get("content") or ""
        ext = _os.path.splitext(rec.get("path") or "")[1]
        if ext not in CODE_EXTS or len(t.strip()) < 100:
            continue
        if got >= budget:
            break
        t = t[:MAX_RECORD_CHARS]
        got += ntok(tok, [t])
        yield "code", t


def render_nebius(traj) -> str:
    parts = []
    for m in traj:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"[{m.get('role', '?')}]\n{text}")
    return "\n\n".join(parts)


def iter_agent(tok, budget: int, seed: int, nebius_share: float = 0.75):
    got = 0
    nb_budget = int(budget * nebius_share)
    sw = load_dataset("nebius/SWE-agent-trajectories", split="train",
                      streaming=True).shuffle(seed=seed, buffer_size=2_000)
    for rec in sw:
        if got >= nb_budget:
            break
        t = render_nebius(rec["trajectory"])[:MAX_RECORD_CHARS]
        if len(t) < 200:
            continue
        got += ntok(tok, [t])
        yield "agent", t

    tc = load_dataset("trace-commons/agent-traces", split="train",
                      streaming=True).shuffle(seed=seed, buffer_size=2_000)
    for rec in tc:
        if got >= budget:
            break
        msgs = rec.get("messages") or []
        parts = []
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)[:4000] if c else ""
            if c.strip():
                parts.append(f"[{m.get('role', '?')}]\n{c.strip()}")
        t = ("\n\n".join(parts))[:MAX_RECORD_CHARS]
        if len(t) < 200:
            continue
        got += ntok(tok, [t])
        yield "agent", t


def main():
    global MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="checkpoint dir for the tokenizer (or set ATLAS_MODEL_DIR)")
    ap.add_argument("--out", default="data/calibration.jsonl")
    ap.add_argument("--budget-en", type=int, default=1_600_000)
    ap.add_argument("--budget-code", type=int, default=1_200_000)
    ap.add_argument("--budget-agent", type=int, default=1_200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    MODEL_DIR = args.model_dir
    if not MODEL_DIR:
        raise SystemExit("нужен --model-dir или ATLAS_MODEL_DIR")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {"en": 0, "code": 0, "agent": 0}
    rows = 0
    with out.open("w") as f:
        for it, budget in ((iter_en, args.budget_en),
                           (iter_code, args.budget_code),
                           (iter_agent, args.budget_agent)):
            for domain, text in it(tok, budget, args.seed):
                f.write(json.dumps({"domain": domain, "text": text}, ensure_ascii=False) + "\n")
                stats[domain] += 1
                rows += 1
                if rows % 500 == 0:
                    print(f"  rows={rows} domains={stats}", flush=True)

    print(f"done: {rows} rows -> {out}")
    print(f"per-domain rows: {stats}")


if __name__ == "__main__":
    main()
