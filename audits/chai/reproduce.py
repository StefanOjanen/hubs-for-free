# Target 3 (CHAI, arXiv:2403.08058), step T2.1: reproduce the base result only.
# Claim: attention heads within a layer are redundant in which tokens they
# attend to during decoding; pairwise correlations of the heads' attention
# scores over the context form one or two large clusters per layer with
# within-cluster correlation above 0.95, and correlation increases with
# depth (their Figures 2, 6 and 13, LLaMa-7B, C4 samples, sequence length
# 2048). Their code is unpublished; the object is reimplemented from the
# paper: for each sample and layer, each head's attention row of the last
# token over the whole context (a T-vector), the Pearson correlation matrix
# across heads, averaged over samples; complete-linkage clusters at
# correlation 0.95 (and 0.90 as a secondary threshold); the largest cluster's
# share of heads; the mean off-diagonal correlation per layer.
#   python audits/chai/reproduce.py <model> <n_samples> <T> [out_prefix] [--save-rows]
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from hubsfree.adapters import pick_device, pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
T = int(sys.argv[3]) if len(sys.argv) > 3 else 512
PREFIX = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("-") else "audits/chai/" + NAME.split("/")[-1]
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)


def c4_windows(tok, n, T):
    """First n C4 English validation documents that tokenize to at least T
    tokens, each truncated to T."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    out = []
    for ex in ds:
        ids = tok(ex["text"], return_tensors="pt", truncation=True, max_length=T)["input_ids"]
        if ids.shape[1] >= T:
            out.append(ids)
        if len(out) == n:
            break
    return out


def cluster_stats(R, thresh):
    n = R.shape[0]
    D = 1.0 - R; np.fill_diagonal(D, 0.0); D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method="complete")
    lab = fcluster(Z, 1.0 - thresh, criterion="distance")
    sizes = sorted(np.bincount(lab)[1:].tolist(), reverse=True)
    return {"n_clusters": len(sizes), "largest_frac": sizes[0] / n, "sizes": sizes[:5],
            "frac_in_clusters_of_2plus": sum(s for s in sizes if s >= 2) / n}


tok = AutoTokenizer.from_pretrained(NAME)
dt = pick_dtype(NAME, DEV, "auto")
model = AutoModelForCausalLM.from_pretrained(NAME, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
L, n = model.config.num_hidden_layers, model.config.num_attention_heads
print(NAME, "DEVICE", device_label(DEV), "DTYPE", dt, f"layers {L} heads {n} T {T} samples {NSAMP}", flush=True)
wins = c4_windows(tok, NSAMP, T)
SAVE_ROWS = "--save-rows" in sys.argv
acc = np.zeros((L, n, n)); sink_last = np.zeros((L, NSAMP)); rows_store = np.zeros((L, NSAMP, n, T), dtype=np.float32) if SAVE_ROWS else None
captured = {}


def make_hook(l):
    def hook(module, args, output):
        # eager attention modules return (attn_output, attn_weights); keep only the last query row
        w = output[1]
        if w is not None:
            captured[l] = w[0, :, -1, :].detach().float().cpu().numpy().astype(np.float64)
    return hook


handles = [model.model.layers[l].self_attn.register_forward_hook(make_hook(l)) for l in range(L)]
for k, ids in enumerate(wins):
    captured.clear()
    model(input_ids=ids.to(DEV))
    for l in range(L):
        row = captured[l]                                           # (n, T): last token's attention over the context
        acc[l] += np.corrcoef(row); sink_last[l, k] = row[:, 0].mean()
        if SAVE_ROWS:
            rows_store[l, k] = row
    print("sample", k + 1, "/", NSAMP, flush=True)
for h in handles:
    h.remove()
del model; release_memory(DEV)
R = acc / NSAMP
np.save(PREFIX + "_corr.npy", R)
if SAVE_ROWS:
    np.save(PREFIX + "_lastrows.npy", rows_store)
rows = []
for l in range(L):
    off = R[l][~np.eye(n, dtype=bool)]
    r = {"layer": l, "mean_offdiag_corr": round(float(off.mean()), 4), "median_offdiag_corr": round(float(np.median(off)), 4),
         "sink_mass_last_row": round(float(sink_last[l].mean()), 4)}
    for th in (0.95, 0.90):
        cs = cluster_stats(R[l], th)
        r[f"t{int(th*100)}_n_clusters"] = cs["n_clusters"]; r[f"t{int(th*100)}_largest_frac"] = round(cs["largest_frac"], 3)
        r[f"t{int(th*100)}_frac_in_2plus"] = round(cs["frac_in_clusters_of_2plus"], 3); r[f"t{int(th*100)}_sizes"] = cs["sizes"]
    rows.append(r)
    print(f"  L{l:2d} mean corr {r['mean_offdiag_corr']:.3f} sink(last row) {r['sink_mass_last_row']:.2f} | 0.95: clusters {r['t95_n_clusters']:2d} largest {r['t95_largest_frac']:.2f} in2+ {r['t95_frac_in_2plus']:.2f} | 0.90: largest {r['t90_largest_frac']:.2f} in2+ {r['t90_frac_in_2plus']:.2f}", flush=True)
summary = {"depth_spearman_mean_corr": round(float(spearmanr(range(L), [r["mean_offdiag_corr"] for r in rows])[0]), 3),
           "layers_with_largest_cluster_majority_t95": int(sum(r["t95_largest_frac"] > 0.5 for r in rows)),
           "layers_with_largest_cluster_majority_t90": int(sum(r["t90_largest_frac"] > 0.5 for r in rows)),
           "mean_corr_first_layer": rows[0]["mean_offdiag_corr"], "mean_corr_last_quarter": round(float(np.mean([r["mean_offdiag_corr"] for r in rows[-(L // 4):]])), 4)}
json.dump({"model": NAME, "n_samples": NSAMP, "T": T, "dtype": dt, "rows": rows, "summary": summary}, open(PREFIX + "_base_result.json", "w"), indent=1)
print("SUMMARY", json.dumps(summary)); print("CHAI_DONE", PREFIX)
