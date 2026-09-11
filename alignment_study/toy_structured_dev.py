# T4.3 development calibration (labeled): does a structured deviation term
# recover the per-pair z profile that the generic-deviation construction of
# tier2_toy.py (part b) missed? There, each head was rebuilt as
# a_h S + e_h K_h with K_h a random skew matrix orthogonal to the shared
# operator S; the rebuilt layers reproduced the r1 depth profile (Spearman
# 0.81) but not z (Spearman -0.34). The k-mode calibration found the
# previous-token (recency) operator to be the second principal operator of
# most layers and the uniform causal operator the third. Here each head is
# rebuilt with its real coefficients on S, on the recency operator R and on
# the uniform operator U (each orthogonalized against the previous ones),
# plus a random skew remainder of the right norm:
#   generic   : a_h S + e_h K_h
#   recency   : a_h S + b_h R' + e'_h K_h            (K_h orthogonal to S, R')
#   recency+U : a_h S + b_h R' + c_h U' + e''_h K_h  (K_h orthogonal to all)
# z is measured against the real layer's plain-surrogate reference, as in
# tier2_toy.py. Development model, 8 windows, T = 64, 6 rebuilds per window.
# Output: toy_structured_dev.json.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, attn_layer, generators, gnorms, coup, r1, shared_mode, sur_plain, zstats, zmed_of
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

NAME, NWIN, STRIDE, SEQ, REPS, KPLAIN = "Qwen/Qwen2.5-0.5B", 8, 40, 64, 6, 8
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)
rg = np.random.default_rng(0)


def unit(M):
    return M / np.sqrt((M ** 2).sum())


def gen_of(A):
    return unit((A - A.T) / 2.0)


def recency_generator(T):
    A = np.zeros((T, T)); A[0, 0] = 1.0
    for i in range(1, T):
        A[i, i - 1] = 1.0
    return gen_of(A)


def uniform_generator(T):
    U = np.tril(np.ones((T, T))); U = U / U.sum(-1, keepdims=True)
    return gen_of(U)


def orthonormalize(dirs):
    out = []
    for D in dirs:
        for B in out:
            D = D - (D * B).sum() * B
        out.append(unit(D))
    return out


def random_skew_perp(basis, T):
    M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2.0
    for B in basis:
        K -= (K * B).sum() * B
    return unit(K)


tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, pick_dtype(NAME, DEV, "auto"))).eval().to(DEV)
L = model.config.num_hidden_layers
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
VARIANTS = ("generic", "recency", "recency+uniform")
pred = {l: {v: {"r1": [], "z": []} for v in VARIANTS} | {"r1_real": [], "z_real": [], "energy": []} for l in range(L)}
for w, ids in enumerate(wins):
    out = model(**{k: v.to(DEV) for k, v in ids.items()})
    for l in range(L):
        A = out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)
        n, T, _ = A.shape; iu = np.triu_indices(n, 1)
        G = generators(A); gn = gnorms(G); C = coup(G)
        plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
        zr, mu, sd = zstats(C, plains, iu)
        S = unit(shared_mode(G)[0])
        basis_all = orthonormalize([S, recency_generator(T), uniform_generator(T)])
        V = G.reshape(n, -1)
        coef = np.stack([V @ B.ravel() for B in basis_all], 1)          # (n, 3)
        pred[l]["r1_real"].append(r1(C, gn, iu)); pred[l]["z_real"].append(zr)
        pred[l]["energy"].append(((coef ** 2).sum(0) / (V ** 2).sum()).tolist())
        for v, k in zip(VARIANTS, (1, 2, 3)):
            basis = basis_all[:k]
            for _ in range(REPS):
                Gt = np.zeros_like(G)
                for h in range(n):
                    struct = sum(coef[h, m] * basis[m] for m in range(k))
                    rem = np.sqrt(max(gn[h] ** 2 - (coef[h, :k] ** 2).sum(), 0.0))
                    Gt[h] = struct + rem * random_skew_perp(basis, T)
                Ct = coup(Gt)
                pred[l][v]["r1"].append(r1(Ct, gnorms(Gt), iu)); pred[l][v]["z"].append(zmed_of(Ct, mu, sd, iu))
    print("WIN", w, flush=True)
del model; release_memory(DEV)

rows = []
for l in range(L):
    row = {"layer": l, "r1_real": float(np.median(pred[l]["r1_real"])), "z_real": float(np.median(pred[l]["z_real"])),
           "energy_S_R_U": [round(float(x), 3) for x in np.mean(pred[l]["energy"], 0)]}
    for v in VARIANTS:
        row[f"r1_{v}"] = float(np.median(pred[l][v]["r1"])); row[f"z_{v}"] = float(np.median(pred[l][v]["z"]))
    rows.append(row)
summ = {}
for v in VARIANTS:
    summ[v] = {"spearman_r1": round(float(spearmanr([r["r1_real"] for r in rows], [r[f"r1_{v}"] for r in rows])[0]), 3),
               "spearman_z": round(float(spearmanr([r["z_real"] for r in rows], [r[f"z_{v}"] for r in rows])[0]), 3),
               "median_abs_err_r1": round(float(np.median([abs(r["r1_real"] - r[f"r1_{v}"]) for r in rows])), 3),
               "median_abs_err_z": round(float(np.median([abs(r["z_real"] - r[f"z_{v}"]) for r in rows])), 2)}
json.dump({"model": NAME, "protocol": {"n_windows": NWIN, "seq": SEQ, "reps": REPS, "k_plain": KPLAIN}, "rows": rows, "summary": summ},
          open("alignment_study/toy_structured_dev.json", "w"), indent=1)
for r in rows:
    print(f"  L{r['layer']:2d} energy S/R/U {r['energy_S_R_U']} | r1 real {r['r1_real']:+.2f} gen {r['r1_generic']:+.2f} rec {r['r1_recency']:+.2f} recU {r['r1_recency+uniform']:+.2f} | z real {r['z_real']:6.1f} gen {r['z_generic']:6.1f} rec {r['z_recency']:6.1f} recU {r['z_recency+uniform']:6.1f}")
print("SUMMARY", json.dumps(summ))
print("STRUCTURED_DONE")
