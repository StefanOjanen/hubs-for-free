# Target 3 (CHAI 2024), step T2.3: the null battery for the frozen statistic.
# COMMITTED BEFORE ANY RESULT EXISTS; not to be run for real until
# audits/PREREGISTRATION4.md is publicly registered. `--dry-run` uses random
# rows and writes to /tmp.
#
# Statistic T_3 per layer (reproduce.py): mean off-diagonal Pearson
# correlation of the heads' last-token attention rows and the share of heads
# in the largest complete-linkage cluster at correlation 0.95 (secondary
# 0.90), averaged over documents. The statistic reads only each head's last
# row, so the surrogates act on that row (amendment 2 of the frozen file;
# identical in distribution to permuting the full map): (a) random last rows,
# Gaussian logits with lognormal per-head temperature, softmax over T; (b)
# per-head permutation of the row with the diagonal entry fixed; (c) the
# layer's sink set (from the full map at extraction) and the diagonal fixed,
# the rest permuted; (d) config-initialized architecture, five seeds, in
# bfloat16, same documents. 200 draws per random family (dry run: 3).
#   python audits/chai/battery.py [--dry-run] [--draws=N] [--samples=N] [--model=NAME]
# The frozen file runs this battery on both mistralai/Mistral-7B-v0.1 (default)
# and facebook/opt-6.7b; the result file is named after the model.
import json
import os
import sys
import time
import numpy as np

# Registration guard (2026-09-15): a real run requires --registered=<URL>,
# the public record of the frozen criteria (the OSF registration, or the
# permalink of audits/PREREGISTRATION4.md at the public freeze commit), so
# that the battery cannot start by accident before the criteria are public.
# The value is written into the result file. Dry runs never touch real data
# or write under audits/.
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if "--dry-run" not in sys.argv and not REGISTERED:
    sys.exit("refusing to run the battery on real data without --registered=<URL of the public registration or freeze permalink>; use --dry-run to exercise the code")

sys.path.insert(0, ".")
sys.path.insert(0, "audits/chai")
_argv = sys.argv; sys.argv = [_argv[0]]        # reproduce.py parses positional arguments at import (addendum 8)
from reproduce import cluster_stats
sys.argv = _argv
from hubsfree.stats import sink_columns

DRY = "--dry-run" in sys.argv
NAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "mistralai/Mistral-7B-v0.1"); T = 1024
SHORT = NAME.split("/")[-1]
DRAWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 3 if DRY else 200))
NSAMP = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--samples=")), 2 if DRY else 32))
N_INIT = 1 if DRY else 5
OUT = "/tmp/chai_battery_dryrun.json" if DRY else f"audits/chai/battery_result_{SHORT}.json"
rng = np.random.default_rng(0)


def layer_stat(rows):
    """rows: (n, T) last-token attention per head."""
    R = np.corrcoef(rows); off = R[~np.eye(R.shape[0], dtype=bool)]
    return {"mean_corr": float(off.mean()), "largest95": cluster_stats(R, 0.95)["largest_frac"], "largest90": cluster_stats(R, 0.90)["largest_frac"]}


def random_rows(n, Tq):
    tau = np.exp(rng.normal(0, 0.5, n)); lo = rng.normal(size=(n, Tq)) / tau[:, None]
    ex = np.exp(lo - lo.max(1, keepdims=True)); return ex / ex.sum(1, keepdims=True)


def permute_rows(rows, keep):
    """Permute each head's row over the positions not in `keep` (the diagonal,
    i.e. the last position, is always kept)."""
    n, Tq = rows.shape
    free = np.array([c for c in range(Tq - 1) if c not in keep])
    S = rows.copy(); S[:, free] = rng.permuted(rows[:, free], axis=1); return S


def extract(model, wins, dev):
    """Per document and layer: the last attention row of every head (n, T)
    and the layer's sink set computed from the full map; nothing else kept."""
    import torch
    mods = [m for nme, m in model.named_modules() if nme.split(".")[-1] in ("self_attn", "self_attention", "attn", "attention") and "Attention" in type(m).__name__]
    assert len(mods) == model.config.num_hidden_layers
    cap = {}

    def hook(l):
        def f(module, args, output):
            w = output[1]
            if w is not None:
                A = w[0].float().cpu().numpy().astype(np.float64)
                cap[l] = (A[:, -1, :].copy(), list(sink_columns(A)))
        return f

    handles = [m.register_forward_hook(hook(l)) for l, m in enumerate(mods)]
    rows, cols = [], []
    for ids in wins:
        cap.clear(); model(input_ids=ids.to(dev), output_attentions=True)
        rows.append([cap[l][0] for l in range(len(mods))]); cols.append([cap[l][1] for l in range(len(mods))])
    for h in handles:
        h.remove()
    return rows, cols


def data():
    if DRY:
        L, n, Tq = 2, 6, 48
        def fake():
            r = random_rows(n, Tq); r[:, 0] += 2.0; return r / r.sum(1, keepdims=True)
        real = [[fake() for _ in range(L)] for _ in range(NSAMP)]; cols = [[[0] for _ in range(L)] for _ in range(NSAMP)]
        unt = [[[random_rows(n, Tq) for _ in range(L)] for _ in range(NSAMP)] for _ in range(N_INIT)]
        return real, cols, unt, L, n, Tq
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from reproduce import c4_windows
    from hubsfree.adapters import pick_device, pick_dtype, release_memory
    dev = pick_device("auto"); tok = AutoTokenizer.from_pretrained(NAME)
    wins = c4_windows(tok, NSAMP, T)
    model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager", dtype=getattr(torch, pick_dtype(NAME, dev, "auto"))).eval().to(dev)
    L, n = model.config.num_hidden_layers, model.config.num_attention_heads
    real, cols = extract(model, wins, dev); del model; release_memory(dev)
    cfg = AutoConfig.from_pretrained(NAME); unt = []
    for seed in range(N_INIT):
        torch.manual_seed(seed); um = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
        unt.append(extract(um, wins, dev)[0]); del um; release_memory(dev)
    return real, cols, unt, L, n, T


t0 = time.time()
real, cols, unt, L, n, Tq = data()
res = {"target": "CHAI 2024", "registered": REGISTERED, "model": NAME, "T": Tq, "samples": NSAMP, "draws": DRAWS, "dry_run": DRY, "layers": []}
KEYS = ("mean_corr", "largest95", "largest90")
for l in range(L):
    docs = [real[k][l] for k in range(NSAMP)]; keeps = [cols[k][l] for k in range(NSAMP)]
    st_real = {k: float(np.mean([layer_stat(r)[k] for r in docs])) for k in KEYS}
    fam = {
        "a_random": [{k: float(np.mean([layer_stat(random_rows(n, Tq))[k] for _ in range(NSAMP)])) for k in KEYS} for _ in range(DRAWS)],
        "b_marginal": [{k: float(np.mean([layer_stat(permute_rows(r, ()))[k] for r in docs])) for k in KEYS} for _ in range(DRAWS)],
        "c_colset": [{k: float(np.mean([layer_stat(permute_rows(r, kp))[k] for r, kp in zip(docs, keeps)])) for k in KEYS} for _ in range(DRAWS)],
        "d_untrained": [{k: float(np.mean([layer_stat(unt[s][doc][l])[k] for doc in range(NSAMP)])) for k in KEYS} for s in range(N_INIT)],
    }
    base = float(np.median([d["mean_corr"] for d in fam["a_random"]]))
    nulls = {}
    for name, dr in fam.items():
        ent = {}
        for k in KEYS:   # percentile_mid is the mid-rank percentile (ties count half), for the cluster share which ties often
            vals = np.array([d[k] for d in dr])
            ent[k] = {"median": float(np.median(vals)), "percentile_of_real": float((vals < st_real[k]).mean() * 100),
                      "percentile_mid": float(((vals < st_real[k]).mean() + 0.5 * (vals == st_real[k]).mean()) * 100),
                      "p5": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95))}
        ent["reproduced_mean_corr"] = (ent["mean_corr"]["median"] - base) / (st_real["mean_corr"] - base) if abs(st_real["mean_corr"] - base) > 1e-9 else None
        nulls[name] = ent
    res["layers"].append({"layer": l, "sink_sets": sorted({str(kp) for kp in keeps}), "real": st_real, "nulls": nulls})
    print(f"  L{l:2d} real corr {st_real['mean_corr']:.3f} largest95 {st_real['largest95']:.2f} | colset median {nulls['c_colset']['mean_corr']['median']:.3f} reproduced {nulls['c_colset']['reproduced_mean_corr']} | marginal median {nulls['b_marginal']['mean_corr']['median']:.3f} | untrained median {nulls['d_untrained']['mean_corr']['median']:.3f} ({time.time()-t0:.0f}s)", flush=True)
res["summary"] = {"layers_where_colset_matches_corr": int(sum(5 <= r["nulls"]["c_colset"]["mean_corr"]["percentile_of_real"] <= 95 for r in res["layers"])),
                  "layers_where_real_above_marginal_p95": int(sum(r["nulls"]["b_marginal"]["mean_corr"]["percentile_of_real"] > 95 for r in res["layers"])), "n_layers": L}
res["registered_expectation"] = "matched by (c) at 2/3 or more of layers; survives (a) and (d) at the 95th percentile; shrinks by more than 50 percent against (b)"
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["summary"])); print("BATTERY_DONE", OUT)
