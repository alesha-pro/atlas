"""carve_hooks.py — съём FFN-активаций Qwen3.8-27B (qwen3_5), адаптация CMoE.

Что взято из CMoE (JarvisPei/CMoE, CMoE_utils.construct_moe):
- «активация нейрона» = элемент h = act_fn(x @ Wg^T) * (x @ Wu^T) — это ровно
  ВХОД down_proj, ловится forward_pre_hook'ом без пересчёта;
- fire-определение: нейрон «стреляет» на токене, если |h| входит в top-K
  нейронов этого токена (CMoE K_a=10; здесь k_fire настраивается, def 64).

Отличия под наш план (PLAN.md, этап 1):
- стриминговая статистика на ВСЕХ токенах калибровки, а не один батч;
- fire_rate считается отдельно по доменам (en/code/agent);
- **счётчики живут на устройстве h** (GPU-резидентно): на 4M токенов
  перекладка на CPU была бы ~18 ТБ через PCIe — вместо этого на GPU копим
  fire-счётчики/суммы/sketch, на CPU уходим только при finalize();
- raw-статистики |h| (mean_abs, p50/p99, top1pct_share) из sketch-выборки;
- выход — parquet/npz под pandas, а не MoE-модуль на лету.

Не трогаем: vision tower, MTP-голову (MTP-веса в чекпоинте есть, но
HF-класс Qwen3_5ForConditionalGeneration её вообще не строит — хуки на неё
не попадут; туда хуки и не вешаем: цепляем только `*.layers.N.mlp.down_proj`
text-декодера).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DOMAINS = ("en", "code", "agent")  # ru убран по решению Алексея 22.08
LAYER_RE = re.compile(r"^(?P<stem>.*layers)\.(?P<idx>\d+)\.mlp\.down_proj$")


class NeuronStats:
    """Стриминговые per-neuron статистики одного слоя + reservoir sketch.

    Все тензоры автоматически переезжают на устройство первого update()
    (GPU у раннера, CPU у smoke) и возвращаются на CPU в finalize().
    """

    def __init__(self, layer: int, n_neurons: int, k_fire: int = 64,
                 sketch_n: int = 2048, seed: int = 0):
        self.layer, self.n, self.k_fire = layer, n_neurons, min(k_fire, n_neurons)
        self.sketch_n = sketch_n
        self._seed = seed + layer
        self._dev = None
        self._gen = None
        self.abs_sum = torch.zeros(n_neurons, dtype=torch.float64)
        self.fire = {d: torch.zeros(n_neurons, dtype=torch.float64) for d in DOMAINS}
        self.tok_total = 0
        self.tok_dom = {d: 0 for d in DOMAINS}
        # reservoir sketch по токенам: [n_neurons, sketch_n]
        self.sketch = torch.zeros(n_neurons, sketch_n, dtype=torch.float16)
        self.sketch_filled = 0
        self._seen = 0  # всего токенов, предложенных reservoir'у

    def _ensure_device(self, dev: torch.device) -> None:
        if self._dev == dev:
            return
        self.abs_sum = self.abs_sum.to(dev)
        self.fire = {d: t.to(dev) for d, t in self.fire.items()}
        self.sketch = self.sketch.to(dev)
        self._gen = torch.Generator(device=dev).manual_seed(self._seed)
        self._dev = dev

    @torch.no_grad()
    def update(self, h: torch.Tensor, domain: str) -> None:
        """h: [.., n_neurons] — вход down_proj (h = act(gate)*up), любой dtype."""
        self._ensure_device(h.device)
        toks = h.reshape(-1, self.n)
        n_tok = toks.shape[0]
        a = toks.abs()  # bf16/fp32 на устройстве

        # --- fire-маркеры: top-K по |h| в каждом токене
        markers = torch.zeros_like(a, dtype=torch.bool)
        markers.scatter_(1, a.topk(self.k_fire, dim=1).indices, True)
        self.fire[domain] += markers.sum(0, dtype=torch.float64)
        self.tok_dom[domain] += n_tok

        # --- общие
        self.abs_sum += a.sum(0, dtype=torch.float64)
        self.tok_total += n_tok

        # --- reservoir sketch (take = сколько ушло в первичное заполнение;
        # rest-токены начинаются с индекса take — без смещения брались бы не те)
        if self.sketch_filled < self.sketch_n:
            take = min(self.sketch_n - self.sketch_filled, n_tok)
            self.sketch[:, self.sketch_filled:self.sketch_filled + take] = \
                a[:take].to(torch.float16).T
            self.sketch_filled += take
        else:
            take = 0
        rest = n_tok - take
        if rest > 0:
            total_after = self._seen + n_tok
            pos = torch.randint(0, total_after, (rest,), device=self._dev,
                                generator=self._gen)
            replace = pos < self.sketch_n  # P(замены) = sketch_n / total_after
            rows = torch.nonzero(replace).flatten()
            if rows.numel():
                self.sketch[:, pos[replace].long()] = a[take + rows].to(torch.float16).T
        self._seen += n_tok

    def finalize(self) -> dict:
        sk = self.sketch[:, : self.sketch_filled].to("cpu", torch.float32)  # [N, S]
        q = torch.tensor([0.5, 0.99], dtype=torch.float32)
        p50, p99 = torch.quantile(sk, q, dim=1)  # [N] каждый
        topm = max(1, int(math.ceil(0.01 * self.sketch_filled)))
        top_sum = sk.topk(topm, dim=1).values.sum(1)
        all_sum = sk.sum(1).clamp_min(1e-12)
        fire_all = sum(t.cpu() for t in self.fire.values()).numpy()
        out = {
            "layer": self.layer,
            "fire_rate": fire_all / max(self.tok_total, 1),
        }
        for d in DOMAINS:
            out[f"fire_rate_{d}"] = (self.fire[d].cpu().numpy() / max(self.tok_dom[d], 1))
        out.update({
            "mean_abs_act": (self.abs_sum.cpu().numpy() / max(self.tok_total, 1)),
            "p50_act": p50.numpy(),
            "p99_act": p99.numpy(),
            "top1pct_share": (top_sum / all_sum).numpy(),
        })
        return out

    def sketch_cpu(self) -> torch.Tensor:
        return self.sketch[:, : self.sketch_filled].cpu()


class CarveCollector:
    """Хуки на mlp.down_proj text-слоёв (активации) + forward hooks слоёв (IO-cos)."""

    def __init__(self, model: nn.Module, k_fire: int = 64, sketch_n: int = 2048,
                 layer_indices: set[int] | None = None, seed: int = 0):
        self.model = model
        self.current_domain = "en"
        self.neurons: dict[int, NeuronStats] = {}
        self.layer_io: dict[int, list[float]] = {}
        self._io_warned = False
        self._hooks = []
        self._layer_modules: dict[int, nn.Module] = {}

        modules = dict(model.named_modules())
        for name, mod in modules.items():
            m = LAYER_RE.match(name)
            if not m:
                continue
            idx = int(m["idx"])
            if layer_indices is not None and idx not in layer_indices:
                continue
            layer_mod = modules.get(f"{m['stem']}.{idx}")
            if layer_mod is None:
                continue
            self.neurons[idx] = NeuronStats(idx, mod.in_features, k_fire, sketch_n, seed)
            self._layer_modules[idx] = layer_mod
            self._hooks.append(mod.register_forward_pre_hook(self._make_pre(idx), with_kwargs=True))
            self._hooks.append(layer_mod.register_forward_hook(self._make_post(idx)))
        if not self.neurons:
            raise RuntimeError("не нашёл ни одного *.layers.N.mlp.down_proj — проверь структуру")

    # -- хуки ------------------------------------------------------------
    def _make_pre(self, idx: int):
        def pre(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is not None:
                self.neurons[idx].update(h, self.current_domain)
        return pre

    def _make_post(self, idx: int):
        def post(module, args, output):
            # layer-IO-косинус — бонус-метрика: считаем на устройстве входа
            # (accelerate переносит выход слоя на устройство следующего слоя
            # своим post-хуком до нашего) и не даём ей ронять прогон.
            try:
                inp = args[0] if args else None
                out = output[0] if isinstance(output, tuple) else output
                if inp is None or out is None:
                    return
                dev = inp.device
                inp_f = inp.detach().to(dev, torch.float32).flatten(0, -2)
                out_f = out.detach().to(dev, torch.float32).flatten(0, -2)
                cos = torch.nn.functional.cosine_similarity(inp_f, out_f, dim=-1)
                self.layer_io.setdefault(idx, []).append(cos.mean().item())
            except Exception as e:  # noqa: BLE001
                if not self._io_warned:
                    self._io_warned = True
                    print(f"[layer_io] hook error (заглушено): {e}", flush=True)
        return post

    def set_domain(self, domain: str) -> None:
        assert domain in DOMAINS, domain
        self.current_domain = domain

    # -- вывод -----------------------------------------------------------
    def dump(self, outdir: str | Path) -> dict[str, Path]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        stat_cols = [f"fire_rate_{d}" for d in DOMAINS]
        rows = {k: [] for k in ("layer", "neuron", "fire_rate", *stat_cols,
                                "mean_abs_act", "p50_act", "p99_act", "top1pct_share")}
        sketch = {}
        for idx in sorted(self.neurons):
            st = self.neurons[idx].finalize()
            nn_ = len(st["fire_rate"])
            rows["layer"].extend([idx] * nn_)
            rows["neuron"].extend(range(nn_))
            for k in rows:
                if k not in ("layer", "neuron"):
                    rows[k].extend(st[k].tolist())
            # sketch: активационные векторы + fire-маркеры тех же токенов
            ns = self.neurons[idx]
            sk = ns.sketch_cpu()
            sketch[f"act_{idx}"] = sk.numpy()
            a = sk.to(torch.float32)
            markers = torch.zeros_like(a, dtype=torch.bool)
            # top-K НЕЙРОНОВ на каждый sketch-токен: topk по dim=0 (строки = нейроны)
            top_n = a.topk(ns.k_fire, dim=0).indices  # [k, S]
            markers.scatter_(0, top_n, True)
            sketch[f"markers_{idx}"] = markers.numpy()

        pq = out / "neuron_stats.parquet"
        pd.DataFrame(rows).to_parquet(pq, index=False)

        npz = out / "coact_sketch.npz"
        np.savez_compressed(npz, **sketch)

        iopq = out / "layer_io_sim.parquet"
        io = pd.DataFrame([{"layer": i, "io_cosine": float(np.mean(v))}
                           for i, v in sorted(self.layer_io.items())])
        io.to_parquet(iopq, index=False)
        return {"neuron_stats": pq, "coact_sketch": npz, "layer_io_sim": iopq}

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- helper для CPU-smoke (без полного forward слоя) ------------------
    @torch.no_grad()
    def note_layer_io(self, idx: int, inp: torch.Tensor, out: torch.Tensor) -> None:
        self._make_post(idx)(None, (inp,), out)
