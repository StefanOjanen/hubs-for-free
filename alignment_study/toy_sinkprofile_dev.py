# T4.3 development calibration, fourth construction (labeled): keep each
# head's REAL sink-column profile and randomize everything else. The
# deviation-column diagnostic found that at high-sink layers about half of
# each head's deviation energy lies in the sink column, so this rebuild tests
# whether the per-head sink profile is the structured deviation that sets the
# per-pair z. Per head: P_h = the entries of G_h in the sink column c and its
# skew mirror row (everything the head does with the sink token, shared part
# included); the remainder ||G_h||^2 - ||P_h||^2 is rebuilt as a random skew
# matrix with zero entries in column and row c (so orthogonal to P_h),
# scaled to the remainder norm. z against the real layer's plain-surrogate
# reference, as in the other toy constructions; the dense rebuild
# (a_h S + random remainder) is reported alongside. Development model, 8
# windows, T = 64, 6 rebuilds per window. Output: toy_sinkprofile_dev.json.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, coup, r1, shared_mode, sink_column, sur_plain, zstats, zmed_of
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

NAME, NWIN, STRIDE, SEQ, REPS, KPLAIN = "Qwen/Qwen2.5-0.5B", 8, 40, 64, 6, 8
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)
rg = np.random.default_rng(0)


def unit(M):
    return M / np.sqrt((M ** 2).sum())


def random_skew_off_column(T, c):
    M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2.0; K[:, c] = 0.0; K[c, :] = 0.0
    return unit(K)


def random_skew_perp(S, T):
    M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2.0; K -= (K * S).sum() * S
    return unit(K)


tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, pick_dtype(NAME, DEV, "auto"))).eval().to(DEV)
L = model.config.num_hidden_layers
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
pred = {l: {"r1_real": [], "z_real": [], "r1_dense": [], "z_dense": [], "r1_profile": [], "z_profile": [], "profile_share": []} for l in range(L)}
for w, ids in enumerate(wins):
    out = model(**{k: v.to(DEV) for k, v in ids.items()})
    for l in range(L):
        A = out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)
        n, T, _ = A.shape; iu = np.triu_indices(n, 1); c = sink_column(A)
        G = generators(A); gn = gnorms(G); C = coup(G)
        plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
        zr, mu, sd = zstats(C, plains, iu)
        S = unit(shared_mode(G)[0]); a = np.array([(G[h] * S).sum() for h in range(n)])
        pred[l]["r1_real"].append(r1(C, gn, iu)); pred[l]["z_real"].append(zr)
        P = np.zeros_like(G); P[:, :, c] = G[:, :, c]; P[:, c, :] = G[:, c, :]
        rem = np.sqrt(np.clip(gn ** 2 - (P ** 2).sum((1, 2)), 0, None))
        pred[l]["profile_share"].append(float(np.mean((P ** 2).sum((1, 2)) / gn ** 2)))
        for _ in range(REPS):
            Gd = np.stack([a[h] * S + np.sqrt(max(gn[h] ** 2 - a[h] ** 2, 0)) * random_skew_perp(S, T) for h in range(n)])
            Gp = np.stack([P[h] + rem[h] * random_skew_off_column(T, c) for h in range(n)])
            for tag, Gt in (("dense", Gd), ("profile", Gp)):
                Ct = coup(Gt); pred[l][f"r1_{tag}"].append(r1(Ct, gnorms(Gt), iu)); pred[l][f"z_{tag}"].append(zmed_of(Ct, mu, sd, iu))
    print("WIN", w, flush=True)
del model; release_memory(DEV)
rows = [{"layer": l, "profile_share": round(float(np.mean(pred[l]["profile_share"])), 3), **{k: float(np.median(pred[l][k])) for k in ("r1_real", "z_real", "r1_dense", "z_dense", "r1_profile", "z_profile")}} for l in range(L)]
summ = {t: {"spearman_r1": round(float(spearmanr([r["r1_real"] for r in rows], [r[f"r1_{t}"] for r in rows])[0]), 3),
            "spearman_z": round(float(spearmanr([r["z_real"] for r in rows], [r[f"z_{t}"] for r in rows])[0]), 3),
            "median_abs_err_z": round(float(np.median([abs(r["z_real"] - r[f"z_{t}"]) for r in rows])), 2),
            "median_abs_err_r1": round(float(np.median([abs(r["r1_real"] - r[f"r1_{t}"]) for r in rows])), 3)} for t in ("dense", "profile")}
json.dump({"model": NAME, "protocol": {"n_windows": NWIN, "seq": SEQ, "reps": REPS, "k_plain": KPLAIN}, "rows": rows, "summary": summ}, open("alignment_study/toy_sinkprofile_dev.json", "w"), indent=1)
for r in rows:
    print(f"  L{r['layer']:2d} profile share {r['profile_share']:.2f} | r1 real {r['r1_real']:+.2f} dense {r['r1_dense']:+.2f} profile {r['r1_profile']:+.2f} | z real {r['z_real']:6.1f} dense {r['z_dense']:6.1f} profile {r['z_profile']:6.1f}")
print("SUMMARY", json.dumps(summ)); print("SINKPROFILE_DONE")
