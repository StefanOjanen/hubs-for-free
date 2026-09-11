# T4.2 development calibration (labeled): can a layer's shared-energy fraction
# be derived from per-head quantities alone, with no cross-head information?
#
# Derivation. For causal attention with sink column c, the ideal unit sink
# generator is S_c = (A_sink - A_sink^T) / 2 / sqrt((T - 1 - c) / 2), where
# A_sink puts all mass of rows i > c on column c. Its inner product with a
# head's generator G_h = (A_h - A_h^T) / 2 uses only column c of A_h
# (entries A_h[c, i] with i > c vanish by causality):
#     <G_h, S_c> = sum_{i > c} A_h[i, c] / sqrt(2 (T - 1 - c)),
# so the energy of G_h along the ideal sink direction is
#     e_h = (sum_{i > c} A_h[i, c])^2 / (2 (T - 1 - c)) = m_h^2 (T - 1 - c) / 2
# with m_h the head's mean column-c mass over rows i > c, while
# ||G_h||^2 = (1/2) sum_i (IPR_i - A_ii^2) (Proposition 2). The derived
# per-head fraction is e_h / ||G_h||^2 and the derived layer value is its
# mean over heads: a function of each head's sink mass and row sharpness.
# Secondary predictor (column energy, an upper bound on any single
# direction inside column c): (1/2) sum_{i > c} A_h[i, c]^2 / ||G_h||^2.
# Two-operator predictor: energy along S_c plus energy along the uniform
# causal operator U (A_ij = 1/(i+1) for j <= i) orthogonalized against S_c;
# the k-mode calibration found U to be the shared operator of untrained
# and low-sink layers, so this predictor should close the low-sink gap.
# Measured: the shared-energy fraction from the top principal component of
# the stacked generators (common.shared_mode), which uses all heads jointly.
# Output: derivation_dev.json (dev model only, T = 64, 12 windows).
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, shared_mode, sink_column, col_mass
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr, pearsonr

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
OUT = sys.argv[2] if len(sys.argv) > 2 else "alignment_study/derivation_dev.json"
NWIN, STRIDE, SEQ = 12, 40, 64
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)


def uniform_generator(T):
    U = np.tril(np.ones((T, T))); U = U / U.sum(-1, keepdims=True)
    G = (U - U.T) / 2.0
    return G / np.sqrt((G ** 2).sum())


def ideal_sink_generator(T, c):
    A = np.zeros((T, T))
    for i in range(T):
        if i > c: A[i, c] = 1.0
        else: A[i, i] = 1.0
    G = (A - A.T) / 2.0
    return G / np.sqrt((G ** 2).sum())


def derived(A):
    n, T, _ = A.shape
    c = sink_column(A)
    G = generators(A); g2 = gnorms(G) ** 2
    Sc = ideal_sink_generator(T, c); U = uniform_generator(T)
    U2 = U - (U * Sc).sum() * Sc; U2 = U2 / np.sqrt((U2 ** 2).sum())
    V = G.reshape(n, -1)
    e_dict = (V @ Sc.ravel()) ** 2 + (V @ U2.ravel()) ** 2
    frac_dict = e_dict / g2
    col = A[:, c + 1:, c]                                  # (n, T-1-c)
    e_ideal = col.sum(1) ** 2 / (2.0 * (T - 1 - c))
    e_col = 0.5 * (col ** 2).sum(1)
    frac_ideal = e_ideal / g2; frac_col = e_col / g2
    S, a, e, measured = shared_mode(G)
    per_head_measured = a ** 2 / g2
    return {"s": c, "smass": col_mass(A, c), "measured": float(measured), "derived_ideal": float(frac_ideal.mean()),
            "derived_col": float(frac_col.mean()), "derived_dict": float(frac_dict.mean()), "per_head_measured": per_head_measured.tolist(),
            "per_head_ideal": frac_ideal.tolist(), "per_head_col": frac_col.tolist()}


def run(name):
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    acc = {l: [] for l in range(L)}
    for ids in wins:
        out = model(**{k: v.to(DEV) for k, v in ids.items()})
        for l in range(L):
            acc[l].append(derived(out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)))
        del out
    rows = []
    for l in range(L):
        ds = acc[l]
        rows.append({"layer": l, "smass": round(float(np.mean([d["smass"] for d in ds])), 4),
                     "measured": round(float(np.mean([d["measured"] for d in ds])), 4),
                     "derived_ideal": round(float(np.mean([d["derived_ideal"] for d in ds])), 4),
                     "derived_col": round(float(np.mean([d["derived_col"] for d in ds])), 4),
                     "derived_dict": round(float(np.mean([d["derived_dict"] for d in ds])), 4),
                     "per_head_spearman_ideal": round(float(np.median([spearmanr(d["per_head_ideal"], d["per_head_measured"])[0] for d in ds])), 3)})
    del model; release_memory(DEV)
    return rows


def summarize(rows, hs_only=False):
    r = [x for x in rows if (x["smass"] > 0.4 or not hs_only)]
    m = np.array([x["measured"] for x in r]); out = {"n_layers": len(r)}
    if len(r) < 3:
        return out
    for k in ("derived_ideal", "derived_col", "derived_dict"):
        p = np.array([x[k] for x in r])
        lin = np.polyfit(p, m, 1); resid = m - np.polyval(lin, p)
        out[k] = {"pearson_r2_linear": round(float(pearsonr(p, m)[0] ** 2), 3), "r2_identity": round(float(1 - ((m - p) ** 2).sum() / ((m - m.mean()) ** 2).sum()), 3),
                  "spearman": round(float(spearmanr(p, m)[0]), 3), "median_abs_err": round(float(np.median(np.abs(m - p))), 4),
                  "median_signed_err": round(float(np.median(m - p)), 4), "slope": round(float(lin[0]), 3), "intercept": round(float(lin[1]), 3)}
    return out


if __name__ == "__main__":
    rows = run(NAME)
    res = {"model": NAME, "protocol": {"n_windows": NWIN, "stride": STRIDE, "seq": SEQ, "device": DEV}, "rows": rows,
           "summary_all_layers": summarize(rows), "summary_high_sink": summarize(rows, True)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(NAME)
    for r in rows:
        print(f"  L{r['layer']:2d} smass {r['smass']:.2f} measured {r['measured']:.3f} ideal {r['derived_ideal']:.3f} col {r['derived_col']:.3f} dict {r['derived_dict']:.3f} | per-head Spearman {r['per_head_spearman_ideal']:+.2f}")
    print("ALL   ", json.dumps(res["summary_all_layers"]))
    print("HIGH  ", json.dumps(res["summary_high_sink"]))
    print("DERIVATION_DONE", OUT)
