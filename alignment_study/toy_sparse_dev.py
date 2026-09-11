# T4.3 development calibration, third construction (labeled): rebuild each
# head's deviation as a SPARSE random skew matrix whose concentration matches
# the real deviation's, instead of a dense random one. deviation_structure_dev
# found real deviations 9 to 25 times more concentrated than dense random
# ones and commuting with S about three times more strongly. Construction per
# head: measure the normalized inverse participation ratio q of the real
# deviation's causal entries (q = 1 for one entry, 1/N for uniform over the N
# causal entries), draw a random causal support of m = round(N / q) entries
# with Gaussian values, skew-symmetrize, orthogonalize against S, scale to the
# real deviation norm. Everything else as in toy_structured_dev.py: real
# coefficients a_h on S, z against the real layer's plain-surrogate
# reference, 8 windows, T = 64, 6 rebuilds per window. Also reports the
# dense rebuild for comparison. Output: toy_sparse_dev.json.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, coup, r1, shared_mode, sur_plain, zstats, zmed_of
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

NAME, NWIN, STRIDE, SEQ, REPS, KPLAIN = "Qwen/Qwen2.5-0.5B", 8, 40, 64, 6, 8
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)
rg = np.random.default_rng(0)


def unit(M):
    return M / np.sqrt((M ** 2).sum())


def dense_perp(S, T):
    M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2.0; K -= (K * S).sum() * S
    return unit(K)


def sparse_perp(S, T, q, low):
    N = len(low[0]); m = int(np.clip(round(N * q), 1, N)) if q <= 1 else N   # q here is 1/IPR_norm = support fraction
    idx = rg.choice(N, m, replace=False)
    K = np.zeros((T, T)); K[low[0][idx], low[1][idx]] = rg.normal(0, 1, m)
    K = (K - K.T) / 2.0; K -= (K * S).sum() * S
    return unit(K)


tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, pick_dtype(NAME, DEV, "auto"))).eval().to(DEV)
L = model.config.num_hidden_layers
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
pred = {l: {"dense": {"r1": [], "z": []}, "sparse": {"r1": [], "z": []}, "r1_real": [], "z_real": [], "support": []} for l in range(L)}
for w, ids in enumerate(wins):
    out = model(**{k: v.to(DEV) for k, v in ids.items()})
    for l in range(L):
        A = out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)
        n, T, _ = A.shape; iu = np.triu_indices(n, 1); low = np.tril_indices(T, -1)
        G = generators(A); gn = gnorms(G); C = coup(G)
        plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
        zr, mu, sd = zstats(C, plains, iu)
        S = unit(shared_mode(G)[0])
        a = np.array([(G[h] * S).sum() for h in range(n)])
        E = np.stack([G[h] - a[h] * S for h in range(n)]); en = np.sqrt((E ** 2).sum((1, 2)))
        # support fraction from the real deviation's concentration: 1 / (IPR_norm), IPR_norm = N * sum p^2
        frac = []
        for h in range(n):
            e2 = E[h][low] ** 2; p = e2 / max(e2.sum(), 1e-30); frac.append(1.0 / (len(p) * (p ** 2).sum()))
        pred[l]["support"].append(float(np.median(frac)))
        pred[l]["r1_real"].append(r1(C, gn, iu)); pred[l]["z_real"].append(zr)
        for _ in range(REPS):
            Gd = np.stack([a[h] * S + en[h] * dense_perp(S, T) for h in range(n)])
            Gs = np.stack([a[h] * S + en[h] * sparse_perp(S, T, frac[h], low) for h in range(n)])
            for tag, Gt in (("dense", Gd), ("sparse", Gs)):
                Ct = coup(Gt); pred[l][tag]["r1"].append(r1(Ct, gnorms(Gt), iu)); pred[l][tag]["z"].append(zmed_of(Ct, mu, sd, iu))
    print("WIN", w, flush=True)
del model; release_memory(DEV)
rows = []
for l in range(L):
    rows.append({"layer": l, "support_frac": round(float(np.median(pred[l]["support"])), 4), "r1_real": float(np.median(pred[l]["r1_real"])), "z_real": float(np.median(pred[l]["z_real"])),
                 **{f"{k}_{t}": float(np.median(pred[l][t][k])) for t in ("dense", "sparse") for k in ("r1", "z")}})
summ = {t: {"spearman_r1": round(float(spearmanr([r["r1_real"] for r in rows], [r[f"r1_{t}"] for r in rows])[0]), 3),
            "spearman_z": round(float(spearmanr([r["z_real"] for r in rows], [r[f"z_{t}"] for r in rows])[0]), 3),
            "median_abs_err_z": round(float(np.median([abs(r["z_real"] - r[f"z_{t}"]) for r in rows])), 2),
            "median_abs_err_r1": round(float(np.median([abs(r["r1_real"] - r[f"r1_{t}"]) for r in rows])), 3)} for t in ("dense", "sparse")}
json.dump({"model": NAME, "protocol": {"n_windows": NWIN, "seq": SEQ, "reps": REPS, "k_plain": KPLAIN}, "rows": rows, "summary": summ}, open("alignment_study/toy_sparse_dev.json", "w"), indent=1)
for r in rows:
    print(f"  L{r['layer']:2d} support {r['support_frac']:.3f} | r1 real {r['r1_real']:+.2f} dense {r['r1_dense']:+.2f} sparse {r['r1_sparse']:+.2f} | z real {r['z_real']:6.1f} dense {r['z_dense']:6.1f} sparse {r['z_sparse']:6.1f}")
print("SUMMARY", json.dumps(summ)); print("SPARSE_DONE")
