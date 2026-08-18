"""Qwen2.5-0.5B all-layer sweep: real vs plain/sinkfix/altsink surrogates.

Preregistered confirmatory run for predictions A1, A2 and the r1 clause of
A7 in alignment_study/PREREGISTRATION.md. Protocol: wikitext-103-raw-v1
validation, texts with >40 whitespace words, first 12, truncation at 64;
float32 eager attention; per (layer, window) G_h = (A_h - A_h^T)/2,
C_ij = ||[G_i, G_j]||_F, r1 = Pearson(upper-tri C, outer(gn, gn)),
gap = ev[-1]/|ev[-2]|. Surrogates (8 draws each, default_rng(0)) preserve
row sums, row IPR, diagonal, hence every gn exactly.

Output: alignment_study/qwen_multilayer.json
"""
import json
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B"
SEQ = 64
NWIN = 12
NDRAW = 8
OUT = ("/Users/stefanojanen/Documents/The Emergent Coordinator Hypothesis/"
       "alignment_study/qwen_multilayer.json")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, output_attentions=True,
    attn_implementation="eager", dtype=torch.float32).eval()
NL = model.config.num_hidden_layers
NH = model.config.num_attention_heads
print(f"MODEL LOADED layers={NL} heads={NH}", flush=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                  split="validation")
texts = [t for t in ds["text"] if len(t.split()) > 40][:NWIN]
print("TEXTS", len(texts), flush=True)

# One forward pass per window; keep all layers' attentions in float64.
ATT = []  # ATT[w][l] -> (NH, T, T)
for w, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
    out = model(**ids)
    ATT.append([a.squeeze(0).to(torch.float64).numpy()
                for a in out.attentions])
    print("FORWARD win", w, "T =", ATT[-1][0].shape[-1], flush=True)


def stats(A):
    """A: (n, T, T) float64 -> (gn, r1, gap), fully vectorized."""
    G = 0.5 * (A - np.transpose(A, (0, 2, 1)))
    gn = np.linalg.norm(G, axis=(1, 2))
    P = np.matmul(G[:, None], G[None, :])          # P[i, j] = G_i @ G_j
    C = np.linalg.norm(P - np.transpose(P, (1, 0, 2, 3)), axis=(2, 3))
    tr = np.triu_indices(A.shape[0], 1)
    r1 = float(np.corrcoef(C[tr], np.outer(gn, gn)[tr])[0, 1])
    ev = np.linalg.eigvalsh(C)
    gap = float(ev[-1] / abs(ev[-2]))
    return gn, r1, gap


def sur_plain(A, rg):
    """Permute causal off-diagonal columns 0..i-1 per head, per row."""
    S = A.copy()
    n, T = A.shape[0], A.shape[1]
    hr = np.arange(n)[:, None]
    for i in range(1, T):
        perms = np.argsort(rg.random((n, i)), axis=1)
        S[:, i, :i] = A[:, i, :i][hr, perms]
    return S


def sur_sinkfix(A, rg):
    """Permute columns 1..i-1; column 0 and diagonal untouched."""
    S = A.copy()
    n, T = A.shape[0], A.shape[1]
    hr = np.arange(n)[:, None]
    for i in range(2, T):
        perms = np.argsort(rg.random((n, i - 1)), axis=1)
        S[:, i, 1:i] = A[:, i, 1:i][hr, perms]
    return S


def sur_altsink(A, rg):
    """Head h targets column t_h = h mod 4: swap the row max among causal
    off-diagonal columns 0..i-1 into t_h (when t_h <= i-1), permute the
    remaining off-diagonal columns; rows with t_h > i-1 plain-permuted."""
    S = A.copy()
    n, T = A.shape[0], A.shape[1]
    for i in range(1, T):
        for g in range(4):
            hs = np.arange(g, n, 4)
            m = len(hs)
            if m == 0:
                continue
            r = np.arange(m)
            vals = A[hs, i, :i].copy()          # (m, i)
            if g <= i - 1:
                jmax = np.argmax(vals, axis=1)
                mx = vals[r, jmax].copy()
                tmp = vals[r, g].copy()
                vals[r, g] = mx
                vals[r, jmax] = tmp
                if i > 1:
                    cols = np.array([c for c in range(i) if c != g])
                    perms = np.argsort(rg.random((m, i - 1)), axis=1)
                    sub = vals[:, cols]
                    vals[:, cols] = sub[r[:, None], perms]
            else:
                perms = np.argsort(rg.random((m, i)), axis=1)
                vals = vals[r[:, None], perms]
            S[hs, i, :i] = vals
    return S


FAMS = {"plain": sur_plain, "sinkfix": sur_sinkfix, "altsink": sur_altsink}
rg = np.random.default_rng(0)

inv = {"rowsum": 0.0, "gnorm": 0.0, "col0_sinkfix": 0.0,
       "rowperm_altsink": 0.0}
per_layer = []
t0 = time.time()
for l in range(NL):
    real_r1s, real_gaps, col0s = [], [], []
    fam_r1 = {f: [] for f in FAMS}
    fam_gap = {f: [] for f in FAMS}
    for w in range(NWIN):
        A = ATT[w][l]
        gn, r1, gap = stats(A)
        real_r1s.append(r1)
        real_gaps.append(gap)
        col0s.append(float(A[:, :, 0].mean()))
        rowsum = A.sum(-1)
        Asort = np.sort(A, axis=-1)
        for fam, fn in FAMS.items():
            for d in range(NDRAW):
                Ssur = fn(A, rg)
                gns, r1s, gaps = stats(Ssur)
                fam_r1[fam].append(r1s)
                fam_gap[fam].append(gaps)
                inv["rowsum"] = max(
                    inv["rowsum"],
                    float(np.abs(Ssur.sum(-1) - rowsum).max()))
                inv["gnorm"] = max(inv["gnorm"],
                                   float(np.abs(gns - gn).max()))
                if fam == "sinkfix":
                    inv["col0_sinkfix"] = max(
                        inv["col0_sinkfix"],
                        float(np.abs(Ssur[:, :, 0] - A[:, :, 0]).max()))
                if fam == "altsink":
                    inv["rowperm_altsink"] = max(
                        inv["rowperm_altsink"],
                        float(np.abs(np.sort(Ssur, axis=-1) - Asort).max()))
    row = {
        "layer": l,
        "mean_col0": float(np.median(col0s)),
        "real_r1": float(np.median(real_r1s)),
        "real_gap": float(np.median(real_gaps)),
        "per_window": {"real_r1": real_r1s, "real_gap": real_gaps,
                       "mean_col0": col0s},
    }
    for f in FAMS:
        row[f + "_r1"] = float(np.median(fam_r1[f]))
        row[f + "_gap"] = float(np.median(fam_gap[f]))
    per_layer.append(row)
    print(f"LAYER {l:2d} {int(time.time() - t0):4d}s "
          f"col0={row['mean_col0']:.3f} real={row['real_r1']:+.3f} "
          f"plain={row['plain_r1']:.3f} sinkfix={row['sinkfix_r1']:+.3f} "
          f"altsink={row['altsink_r1']:+.3f}", flush=True)

invariants_ok = all(v < 1e-9 for v in inv.values())

# Anchor sanity: attentions index 20 (1-indexed layer 21).
anchor = per_layer[20]
anchor_ok = (abs(anchor["real_r1"] - (-0.04)) < 0.15
             and anchor["plain_r1"] > 0.9)
print("ANCHOR layer-index 20: real", round(anchor["real_r1"], 4),
      "plain", round(anchor["plain_r1"], 4), "ok:", anchor_ok, flush=True)

# Preregistered predictions evaluable from this run.
D = [p["plain_r1"] - p["real_r1"] for p in per_layer]
big = [l for l in range(NL) if D[l] > 0.3]
a1_pass = len(big) >= 16
a2_hits = [l for l in big
           if abs(per_layer[l]["sinkfix_r1"] - per_layer[l]["real_r1"]) < 0.2]
a2_pass = len(big) > 0 and len(a2_hits) >= (2 / 3) * len(big)
a7_hits = [l for l in big if per_layer[l]["altsink_r1"] > 0.7]
a7_r1_pass = len(big) > 0 and len(a7_hits) >= (2 / 3) * len(big)

results = {
    "model": MODEL,
    "n_layers": NL,
    "n_heads": NH,
    "n_windows": NWIN,
    "n_draws": NDRAW,
    "seq_len": SEQ,
    "per_layer": per_layer,
    "invariants_max_abs_dev": inv,
    "invariants_ok": invariants_ok,
    "anchor_layer_index_20": {"real_r1": anchor["real_r1"],
                              "plain_r1": anchor["plain_r1"],
                              "anchor_ok": anchor_ok},
    "prereg_eval": {
        "D_per_layer": D,
        "layers_D_gt_0.3": big,
        "A1_pass": a1_pass,
        "A1_detail": f"{len(big)}/24 layers with D>0.3 (need >=16)",
        "A2_layers_within_0.2": a2_hits,
        "A2_pass": a2_pass,
        "A2_detail": f"{len(a2_hits)}/{len(big)} qualifying layers "
                     "with |sinkfix-real|<0.2 (need >=2/3)",
        "A7_r1_layers_gt_0.7": a7_hits,
        "A7_r1_clause_pass": a7_r1_pass,
        "A7_detail": f"{len(a7_hits)}/{len(big)} qualifying layers with "
                     "altsink r1>0.7 (need >=2/3); rho clause is computed "
                     "in the shared-mode script, not here",
    },
}
with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("WROTE", OUT, flush=True)
print(json.dumps({k: results[k] for k in
                  ("invariants_max_abs_dev", "invariants_ok",
                   "anchor_layer_index_20", "prereg_eval")}, indent=1))
