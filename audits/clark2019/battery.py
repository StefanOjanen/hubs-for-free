# Target 1 (Clark et al. 2019), step T2.3: the null battery for the frozen
# statistic. COMMITTED BEFORE ANY RESULT EXISTS; it must not be run for real
# until audits/PREREGISTRATION4_DRAFT.md is frozen and publicly registered
# (plan WS2). `--dry-run` exercises the code path on random maps with tiny
# draw counts and writes nothing under audits/.
#
# Statistic T_1: D = mean JS(different layer) - mean JS(same layer) over all
# 144 x 144 head pairs, averaged over inputs and query positions; secondary:
# the fraction of heads whose nearest neighbor is in their own layer.
# Nulls (PREREGISTRATION4_DRAFT.md, Target 1): (a) random bidirectional
# softmax maps, 144 heads with the 12 x 12 layer labels, 100 ensembles;
# (b) 100 per-row marginal-matched draws of the real maps, per window;
# (c) 100 column-preserving draws keeping columns 0 ([CLS]) and T - 1
# ([SEP]); (d) untrained bert-base architecture, five random initializations,
# same inputs. Outcome labels per T2.4: survives (real beyond the 95th
# percentile of every null and shrinkage under 0.5), shrinks (survives at
# least one null, shrinkage above 0.5 against (b) or (c)), matched.
#   python audits/clark2019/battery.py [--dry-run] [--draws N] [--windows N]
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from hubsfree.nulls import random_causal_softmax, surrogate_colfix, surrogate_plain

NAME, T = "bert-base-uncased", 64
DRY = "--dry-run" in sys.argv
DRAWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 3 if DRY else 100))
NWIN = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--windows=")), 2 if DRY else 38))
N_INIT = 1 if DRY else 5
OUT = "/tmp/clark2019_battery_dryrun.json" if DRY else "audits/clark2019/battery_result.json"


def js_matrix(A):
    """Mean pairwise Jensen-Shannon divergence between heads over query rows; A is (LH, T, T)."""
    P = np.clip(A, 1e-12, 1); n, Tq, _ = A.shape; acc = np.zeros((n, n))
    for i in range(Tq):
        p = P[:, i, :]; M = (p[:, None, :] + p[None, :, :]) / 2
        acc += 0.5 * ((p[:, None, :] * (np.log(p[:, None, :]) - np.log(M))).sum(-1) + (p[None, :, :] * (np.log(p[None, :, :]) - np.log(M))).sum(-1))
    return acc / Tq


def stats(JS, layer):
    n = len(layer); same = (layer[:, None] == layer[None, :]) & ~np.eye(n, dtype=bool); diff = layer[:, None] != layer[None, :]
    nn = float(np.mean([layer[np.argsort(JS[k] + np.eye(n)[k] * 1e9)[0]] == layer[k] for k in range(n)]))
    return {"D": float(JS[diff].mean() - JS[same].mean()), "nn_same_layer": nn}


def percentile(real, draws):
    v = np.asarray(draws); return float((v < real).mean() * 100)


t0 = time.time()
if DRY:
    L, H = 3, 4
    rng = np.random.default_rng(0)
    windows = [random_causal_softmax(rng, L * H, T, causal=False) for _ in range(NWIN)]   # stand-ins for real maps
    untrained = [[random_causal_softmax(rng, L * H, T, causal=False) for _ in range(NWIN)] for _ in range(N_INIT)]
else:
    from hubsfree.adapters import attentions, load_model, wikitext_texts
    import torch
    from transformers import AutoConfig, AutoModel
    tok, model = load_model(NAME)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    texts = wikitext_texts(NWIN)
    windows = [np.concatenate(attentions(tok, model, t, T)[0], 0) for t in texts]
    untrained = []
    cfg = AutoConfig.from_pretrained(NAME)
    for seed in range(N_INIT):
        torch.manual_seed(seed)
        um = AutoModel.from_config(cfg, attn_implementation="eager").eval()
        untrained.append([np.concatenate([a.squeeze(0).float().numpy().astype(np.float64) for a in um(**tok(t, return_tensors="pt", truncation=True, max_length=T), output_attentions=True).attentions], 0) for t in texts])
        del um
layer = np.repeat(np.arange(L), H); n = L * H
rng = np.random.default_rng(1)

# real
JS_real = np.mean([js_matrix(A) for A in windows], 0); real = stats(JS_real, layer)
print("real", json.dumps(real), f"({time.time()-t0:.0f}s)", flush=True)
# (a) random bidirectional maps, arbitrary labels
null_a = [stats(np.mean([js_matrix(random_causal_softmax(rng, n, T, causal=False)) for _ in range(NWIN)], 0), layer) for _ in range(DRAWS)]
print("null (a) done", f"({time.time()-t0:.0f}s)", flush=True)
# (b) marginal-matched and (c) column-preserving, per window, draw k uses the k-th surrogate of every window
null_b, null_c = [], []
sur_b = [surrogate_plain(A, rng, DRAWS, causal=False) for A in windows]
sur_c = [surrogate_colfix(A, rng, cols=(0, T - 1), draws=DRAWS, causal=False) for A in windows]
for k in range(DRAWS):
    null_b.append(stats(np.mean([js_matrix(S[k]) for S in sur_b], 0), layer))
    null_c.append(stats(np.mean([js_matrix(S[k]) for S in sur_c], 0), layer))
    if k % 10 == 0:
        print("draw", k, f"({time.time()-t0:.0f}s)", flush=True)
# (d) untrained
null_d = [stats(np.mean([js_matrix(A) for A in ws], 0), layer) for ws in untrained]

res = {"target": "Clark et al. 2019", "model": NAME, "n_windows": NWIN, "T": T, "draws": DRAWS, "dry_run": DRY, "real": real, "nulls": {}}
for fam, vals in (("a_random", null_a), ("b_marginal", null_b), ("c_colfix", null_c), ("d_untrained", null_d)):
    Ds = [v["D"] for v in vals]; med = float(np.median(Ds))
    res["nulls"][fam] = {"D_median": med, "D_p5_p95": [float(np.percentile(Ds, 5)), float(np.percentile(Ds, 95))],
                         "percentile_of_real": percentile(real["D"], Ds), "shrinkage": med / real["D"] if real["D"] else None,
                         "nn_median": float(np.median([v["nn_same_layer"] for v in vals])), "n": len(vals)}
p95 = all(res["nulls"][f]["percentile_of_real"] > 95 for f in res["nulls"])
shrink_bc = max(res["nulls"]["b_marginal"]["shrinkage"] or 0, res["nulls"]["c_colfix"]["shrinkage"] or 0)
res["outcome"] = ("survives" if p95 and shrink_bc < 0.5 else "shrinks" if any(res["nulls"][f]["percentile_of_real"] > 95 for f in res["nulls"]) and shrink_bc >= 0.5 else "matched")
res["registered_expectation"] = "survives (a) and (d); against (b) and (c) shrinkage between 0.2 and 0.7"
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "nulls"}, indent=1)); print(json.dumps(res["nulls"], indent=1)); print("BATTERY_DONE", OUT)
