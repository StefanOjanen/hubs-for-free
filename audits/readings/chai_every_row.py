# PREREGISTRATION17, Part C: Target 3 (CHAI 2024) under the every-row reading.
# Target 3 (preregistration 4) read the cross-head correlation from each
# head's attention row of the last token. The paper does not say which rows;
# a maintainer's reply (facebookresearch/chai issue 6) confirms Pearson
# correlation, and the maintainer's dump code (issue 4), which feeds the
# released clustering script, writes every query position's rows of every
# head. This script computes, per layer, the Pearson correlation across heads
# of each query row over its causal support, averaged over 64 rows evenly
# spaced from position 16 to T-1 and over documents (readings_common.py),
# then the registered summaries: mean off-diagonal correlation, and the share
# of heads in the largest complete-linkage cluster at 0.95 and 0.90. Same
# models, documents (32 C4 documents of 1,024 tokens), precision and null
# families as Target 3, applied to every row read:
#   (a) random rows: Gaussian logits, lognormal per-head temperature, softmax
#       over the row's support; (b) each row permuted over its support with the
#       diagonal fixed; (c) the same with the layer's sink set fixed as well
#       (computed from the full map, as in Target 3); (d) five config-initialized
#       models in bfloat16. DRAWS per random family. Cosine similarity, the
#       released clustering's distance, is reported alongside.
# The last-row statistic of Target 3 is recomputed on the same pass and
# checked against the registered battery's real values.
#   python audits/readings/chai_every_row.py --model=NAME --registered=<freeze permalink>
#   python audits/readings/chai_every_row.py --dev   (Qwen2.5-0.5B, 8 documents; labeled development)
import json
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "audits/readings")
sys.path.insert(0, "audits/chai")
_argv = sys.argv; sys.argv = [_argv[0]]        # reproduce.py parses positional arguments at import (addendum 8)
from reproduce import cluster_stats, c4_windows
sys.argv = _argv
from readings_common import permute_free, free_order, random_maps, row_pearson
from hubsfree.stats import sink_columns
from hubsfree.adapters import pick_device, pick_dtype, release_memory

DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run on the audited models without --registered=<freeze permalink of audits/PREREGISTRATION17.md>; use --dev")
NAME = "Qwen/Qwen2.5-0.5B" if DEV_RUN else next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "mistralai/Mistral-7B-v0.1")
SHORT = NAME.split("/")[-1]
T = 1024
NSAMP = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--samples=")), 8 if DEV_RUN else 32))
DRAWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 10 if DEV_RUN else 30))
N_INIT = 2 if DEV_RUN else 5
ROWS = np.unique(np.round(np.linspace(16, T - 1, 64)).astype(int))
OUT = f"audits/readings/dev/chai_every_row_{SHORT}.json" if DEV_RUN else f"audits/readings/chai_every_row_{SHORT}.json"
DEV = pick_device("auto")
FAMS = ("a_random", "b_marginal", "c_colset")
torch.set_grad_enabled(False)


def attention_modules(model):
    mods = [m for nme, m in model.named_modules() if nme.split(".")[-1] in ("self_attn", "self_attention", "attn", "attention") and "Attention" in type(m).__name__]
    assert len(mods) == model.config.num_hidden_layers
    return mods


KEYS = ("mean_corr", "largest95", "largest90")


def summary(R):
    n = R.shape[0]; off = R[~np.eye(n, dtype=bool)]
    return {"mean_corr": float(off.mean()), "largest95": cluster_stats(R, 0.95)["largest_frac"], "largest90": cluster_stats(R, 0.90)["largest_frac"]}


def vec(R):
    d = summary(R); return np.array([d[k] for k in KEYS])


def run(model, wins, trained, gen, log):
    """Per layer, summed over documents: the per-document summaries (mean
    off-diagonal correlation, largest-cluster shares at 0.95 and 0.90) of the
    every-row Pearson matrix, of the every-row cosine matrix, and, for the
    trained model, of the last-row Pearson matrix (Target 3's statistic) and
    of every null draw; the document-averaged every-row matrix; sink sets."""
    mods = attention_modules(model); L = len(mods); n = model.config.num_attention_heads
    rows_t = torch.tensor(ROWS, device=DEV)
    support = torch.arange(T, device=DEV)[None, :] <= rows_t[:, None]                      # (R, T): c <= r
    offdiag = support & (torch.arange(T, device=DEV)[None, :] != rows_t[:, None])          # off the diagonal
    dst_b = free_order(offdiag, n)
    acc = {"real": np.zeros((L, 3)), "real_cos": np.zeros((L, 3)), "matrix": np.zeros((L, n, n))}
    if trained:
        acc["last"] = np.zeros((L, 3))
        for f in FAMS:
            acc[f] = np.zeros((L, DRAWS, 3)); acc[f + "_cos"] = np.zeros((L, DRAWS))
    sinksets = [[] for _ in range(L)]
    cap = {}

    def hook(l):
        def f(module, args, output):
            w = output[1]
            if w is not None:
                cap[l] = w[0]
        return f

    handles = [m.register_forward_hook(hook(l)) for l, m in enumerate(mods)]
    for d, ids in enumerate(wins):
        cap.clear(); model(input_ids=ids.to(DEV), output_attentions=True)
        for l in range(L):
            Afull = cap[l]
            X = Afull[:, ROWS, :].float()                                                    # (n, R, T)
            c, cs = row_pearson(X, support)
            acc["real"][l] += vec(c); acc["real_cos"][l] += vec(cs); acc["matrix"][l] += c
            if not trained:
                continue
            A64 = Afull.float().cpu().numpy().astype(np.float64)
            S = list(sink_columns(A64)); sinksets[l].append(S)
            acc["last"][l] += vec(np.corrcoef(A64[:, -1, :]))
            free_c = offdiag.clone(); free_c[:, S] = False
            dst_c = free_order(free_c, n)
            for k in range(DRAWS):
                for fam, Y in (("a_random", random_maps(n, len(ROWS), T, support, gen, DEV)),
                               ("b_marginal", permute_free(X, offdiag, gen, dst_b)),
                               ("c_colset", permute_free(X, free_c, gen, dst_c))):
                    c, cs = row_pearson(Y, support); acc[fam][l, k] += vec(c)
                    acc[fam + "_cos"][l, k] += summary(cs)["mean_corr"]
        cap.clear()
        log(f"  document {d + 1}/{len(wins)}")
    for h in handles:
        h.remove()
    return acc, sinksets


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    wins = c4_windows(tok, NSAMP, T)
    dt = pick_dtype(NAME, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L, n = model.config.num_hidden_layers, model.config.num_attention_heads
    log(f"{NAME} DTYPE {dt} layers {L} heads {n} documents {NSAMP} rows {len(ROWS)} draws {DRAWS}")
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    acc, sinksets = run(model, wins, True, gen, log)
    del model; release_memory(DEV)
    cfg = AutoConfig.from_pretrained(NAME); unt = []
    for s in range(N_INIT):
        torch.manual_seed(s); um = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", dtype=torch.bfloat16).eval().to(DEV)
        a, _ = run(um, wins, False, gen, lambda s: None); unt.append(a["real"] / NSAMP); del um; release_memory(DEV)
        log(f"untrained {s + 1}/{N_INIT}")
    res = {"part": "C", "registered": REGISTERED, "dev": DEV_RUN, "model": NAME, "dtype": dt, "T": T, "samples": NSAMP, "rows": ROWS.tolist(),
           "draws": DRAWS, "n_untrained": N_INIT, "device": DEV, "layers": []}
    reg = None if DEV_RUN else json.load(open(f"audits/chai/battery_result_{SHORT}.json"))
    todict = lambda v: dict(zip(KEYS, [float(x) for x in v]))
    for l in range(L):
        real = todict(acc["real"][l] / NSAMP); last = todict(acc["last"][l] / NSAMP)
        ent = {"layer": l, "sink_sets": sorted({str(s) for s in sinksets[l]}), "real": real, "real_cosine": todict(acc["real_cos"][l] / NSAMP),
               "real_matrix_summary": summary(acc["matrix"][l] / NSAMP), "last_row": last, "nulls": {}}
        if reg:
            ent["last_row_check_abs_diff"] = {k: abs(last[k] - reg["layers"][l]["real"][k]) for k in KEYS}
        dist = {f: acc[f][l] / NSAMP for f in FAMS}
        dist["d_untrained"] = np.array([u[l] for u in unt])
        base = float(np.median(dist["a_random"][:, 0]))
        for f, dr in dist.items():
            vals = dr[:, 0]; med = float(np.median(vals))
            ent["nulls"][f] = {"mean_corr_median": med, "p5_p95": [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))],
                               "percentile_of_real": float((vals < real["mean_corr"]).mean() * 100),
                               "largest95_median": float(np.median(dr[:, 1])), "largest90_median": float(np.median(dr[:, 2])),
                               "reproduced": (med - base) / (real["mean_corr"] - base) if abs(real["mean_corr"] - base) > 1e-9 else None}
            if f in FAMS:
                ent["nulls"][f]["cosine_median"] = float(np.median(acc[f + "_cos"][l] / NSAMP))
        res["layers"].append(ent)
        print(f"  L{l:2d} every-row corr {real['mean_corr']:.3f} (last row {last['mean_corr']:.3f}) | colset reproduced {ent['nulls']['c_colset']['reproduced']:.2f} | marginal {ent['nulls']['b_marginal']['reproduced']:.2f} | untrained {ent['nulls']['d_untrained']['reproduced']:.2f}", flush=True)
    rep = lambda f: [e["nulls"][f]["reproduced"] for e in res["layers"]]
    res["summary"] = {f: {"median": float(np.median(rep(f))), "min": float(np.min(rep(f))), "max": float(np.max(rep(f)))} for f in ("b_marginal", "c_colset", "d_untrained")}
    res["summary"]["depth_spearman_every_row"] = float(spearmanr(range(L), [e["real"]["mean_corr"] for e in res["layers"]])[0])
    res["summary"]["depth_spearman_last_row"] = float(spearmanr(range(L), [e["last_row"]["mean_corr"] for e in res["layers"]])[0])
    res["summary"]["layers_majority_95_every_row"] = int(sum(e["real"]["largest95"] > 0.5 for e in res["layers"]))
    res["summary"]["layers_majority_90_every_row"] = int(sum(e["real"]["largest90"] > 0.5 for e in res["layers"]))
    if reg:
        res["summary"]["last_row_check_max_abs_diff"] = {k: float(max(e["last_row_check_abs_diff"][k] for e in res["layers"])) for k in KEYS}
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1)); print("CHAI_READING_DONE", OUT)
