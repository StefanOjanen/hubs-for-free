# Target 3 (CHAI 2024), step T2.3: the null battery for the frozen statistic.
# COMMITTED BEFORE ANY RESULT EXISTS; not to be run for real until
# audits/PREREGISTRATION4_DRAFT.md is publicly registered. `--dry-run` uses
# random maps and writes to /tmp.
#
# Statistic T_3 per layer (reproduce.py): mean off-diagonal Pearson
# correlation of the heads' last-token attention rows and the share of heads
# in the largest complete-linkage cluster at correlation 0.95 (secondary
# 0.90), averaged over documents. Nulls on the same documents
# (PREREGISTRATION4_DRAFT.md, Target 3): (a) random causal softmax maps
# matched in n and T; (b) per-row marginal-matched surrogates of the full
# maps; (c) column-set-preserving surrogates (sink set fixed); (d)
# config-initialized architecture, five seeds. The surrogate families act on
# the full (n, T, T) maps and the statistic reads their last row, so the
# real run needs the maps saved by `reproduce.py --save-rows` or a fresh
# extraction; this script re-extracts them.
#   python audits/chai/battery.py [--dry-run] [--draws=N] [--samples=N]
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "audits/chai")
from reproduce import cluster_stats
from hubsfree.nulls import random_causal_softmax, surrogate_colfix, surrogate_plain
from hubsfree.stats import sink_columns

NAME, T = "mistralai/Mistral-7B-v0.1", 1024
DRY = "--dry-run" in sys.argv
DRAWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 3 if DRY else 200))
NSAMP = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--samples=")), 2 if DRY else 32))
N_INIT = 1 if DRY else 5
OUT = "/tmp/chai_battery_dryrun.json" if DRY else "audits/chai/battery_result.json"
rng = np.random.default_rng(0)


def layer_stat(rows):
    """rows: (n, T) last-token attention per head."""
    R = np.corrcoef(rows); off = R[~np.eye(R.shape[0], dtype=bool)]
    return {"mean_corr": float(off.mean()), "largest95": cluster_stats(R, 0.95)["largest_frac"], "largest90": cluster_stats(R, 0.90)["largest_frac"]}


def maps_real_and_untrained():
    if DRY:
        L, n, Tq = 2, 6, 48
        real = [[random_causal_softmax(rng, n, Tq, sink_frac=1.0, sink_bias=3.0) for _ in range(L)] for _ in range(NSAMP)]
        unt = [[[random_causal_softmax(rng, n, Tq) for _ in range(L)] for _ in range(NSAMP)] for _ in range(N_INIT)]
        return real, unt, L, n, Tq
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from reproduce import c4_windows
    from hubsfree.adapters import pick_device, pick_dtype, release_memory
    dev = pick_device("auto"); tok = AutoTokenizer.from_pretrained(NAME)
    wins = c4_windows(tok, NSAMP, T)
    def extract(model):
        out = []
        for ids in wins:
            o = model(input_ids=ids.to(dev), output_attentions=True)
            out.append([a.squeeze(0).float().cpu().numpy().astype(np.float64) for a in o.attentions]); del o
        return out
    model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager", dtype=getattr(torch, pick_dtype(NAME, dev, "auto"))).eval().to(dev)
    L, n = model.config.num_hidden_layers, model.config.num_attention_heads
    real = extract(model); del model; release_memory(dev)
    cfg = AutoConfig.from_pretrained(NAME); unt = []
    for seed in range(N_INIT):
        torch.manual_seed(seed); um = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
        unt.append(extract(um)); del um; release_memory(dev)
    return real, unt, L, n, T


t0 = time.time()
real, unt, L, n, Tq = maps_real_and_untrained()
res = {"target": "CHAI 2024", "model": NAME, "T": Tq, "samples": NSAMP, "draws": DRAWS, "dry_run": DRY, "layers": []}
for l in range(L):
    A_docs = [real[k][l] for k in range(NSAMP)]
    st_real = {k: float(np.mean([layer_stat(A[:, -1, :])[k] for A in A_docs])) for k in ("mean_corr", "largest95", "largest90")}
    nulls = {}
    a_draws = [{k: float(np.mean([layer_stat(random_causal_softmax(rng, n, Tq)[:, -1, :])[k] for _ in range(NSAMP)])) for k in st_real} for _ in range(DRAWS)]
    sur_b = [surrogate_plain(A, rng, DRAWS) for A in A_docs]
    sur_c = [surrogate_colfix(A, rng, cols=tuple(sink_columns(A)), draws=DRAWS) for A in A_docs]
    b_draws = [{k: float(np.mean([layer_stat(S[d][:, -1, :])[k] for S in sur_b])) for k in st_real} for d in range(DRAWS)]
    c_draws = [{k: float(np.mean([layer_stat(S[d][:, -1, :])[k] for S in sur_c])) for k in st_real} for d in range(DRAWS)]
    d_draws = [{k: float(np.mean([layer_stat(unt[s][doc][l][:, -1, :])[k] for doc in range(NSAMP)])) for k in st_real} for s in range(N_INIT)]
    base = float(np.median([d["mean_corr"] for d in a_draws]))
    for fam, dr in (("a_random", a_draws), ("b_marginal", b_draws), ("c_colset", c_draws), ("d_untrained", d_draws)):
        ent = {}
        for k in st_real:
            v = np.array([d[k] for d in dr]); ent[k] = {"median": float(np.median(v)), "percentile_of_real": float((v < st_real[k]).mean() * 100)}
        rb = float(np.median([d["mean_corr"] for d in a_draws]))
        ent["reproduced_mean_corr"] = (ent["mean_corr"]["median"] - rb) / (st_real["mean_corr"] - rb) if abs(st_real["mean_corr"] - rb) > 1e-9 else None
        nulls[fam] = ent
    res["layers"].append({"layer": l, "real": st_real, "nulls": nulls})
    print(f"  L{l:2d} real corr {st_real['mean_corr']:.3f} largest95 {st_real['largest95']:.2f} | colset corr median {nulls['c_colset']['mean_corr']['median']:.3f} reproduced {nulls['c_colset']['reproduced_mean_corr']} | marginal corr median {nulls['b_marginal']['mean_corr']['median']:.3f} ({time.time()-t0:.0f}s)", flush=True)
res["summary"] = {"layers_where_colset_matches_corr": int(sum(5 <= r["nulls"]["c_colset"]["mean_corr"]["percentile_of_real"] <= 95 for r in res["layers"])),
                  "layers_where_marginal_below_p5": int(sum(r["nulls"]["b_marginal"]["mean_corr"]["percentile_of_real"] > 95 for r in res["layers"])), "n_layers": L}
res["registered_expectation"] = "matched by (c) at 2/3 or more of layers; survives (a) and (d) at the 95th percentile; shrinks by more than 50 percent against (b)"
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["summary"])); print("BATTERY_DONE", OUT)
