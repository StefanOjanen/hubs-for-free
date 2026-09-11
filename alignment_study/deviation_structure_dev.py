# Development diagnostic (labeled) for the open T4.3 question: what structure
# in the deviations E_h = G_h - a_h S sets the per-pair z that random
# deviations of the same norm miss (toy_structured_dev.py)? Three per-head
# descriptors on the development model, real against random skew matrices
# of the same norm orthogonal to S:
#   (i)   normalized commutator with the shared operator,
#         ||[S, E_h]||_F / (||S||_F ||E_h||_F);
#   (ii)  concentration of E_h, the inverse participation ratio of its
#         squared entries normalized by the number of causal entries
#         (1 = one entry carries everything, small = spread out);
#   (iii) the share of E_h's energy inside its single most energetic column
#         (excluding the sink column), the head-specific-column hypothesis.
# 8 windows, T = 64, 3 random draws per head. Output: deviation_structure_dev.json.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, shared_mode, sink_column
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, NWIN, STRIDE, SEQ, REPS = "Qwen/Qwen2.5-0.5B", 8, 40, 64, 3
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)
rg = np.random.default_rng(0)


def unit(M):
    return M / np.sqrt((M ** 2).sum())


def random_skew_perp(S, T):
    M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2.0; K -= (K * S).sum() * S
    return unit(K)


def descriptors(E, S, c):
    T = S.shape[0]
    com = np.linalg.norm(S @ E - E @ S) / (np.linalg.norm(S) * np.linalg.norm(E) + 1e-12)
    low = np.tril_indices(T, -1)
    e2 = E[low] ** 2; p = e2 / e2.sum()
    ipr = float((p ** 2).sum() * len(p))           # 1 for one entry, 0 spread
    colE = (E ** 2).sum(0) + (E ** 2).sum(1)      # energy touching each column index (skew: row and column parts)
    colE[c] = 0.0
    share = float(colE.max() / (2 * (E ** 2).sum()))
    return com, ipr, share


tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, pick_dtype(NAME, DEV, "auto"))).eval().to(DEV)
L = model.config.num_hidden_layers
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
acc = {l: {"real": [], "rand": []} for l in range(L)}
smass = {l: [] for l in range(L)}
for ids in wins:
    out = model(**{k: v.to(DEV) for k, v in ids.items()})
    for l in range(L):
        A = out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)
        n, T, _ = A.shape; c = sink_column(A)
        G = generators(A); S, a, e, frac = shared_mode(G); S = unit(S)
        smass[l].append(float(A[:, c + 1:, c].mean()))
        for h in range(n):
            E = G[h] - (G[h] * S).sum() * S
            acc[l]["real"].append(descriptors(E, S, c))
            for _ in range(REPS):
                acc[l]["rand"].append(descriptors(np.linalg.norm(E) * random_skew_perp(S, T), S, c))
    del out
del model; release_memory(DEV)
rows = []
for l in range(L):
    R = np.array(acc[l]["real"]); Q = np.array(acc[l]["rand"])
    rows.append({"layer": l, "smass": round(float(np.mean(smass[l])), 3),
                 "real": {"commutator": round(float(np.median(R[:, 0])), 4), "ipr": round(float(np.median(R[:, 1])), 4), "col_share": round(float(np.median(R[:, 2])), 3)},
                 "random": {"commutator": round(float(np.median(Q[:, 0])), 4), "ipr": round(float(np.median(Q[:, 1])), 4), "col_share": round(float(np.median(Q[:, 2])), 3)}})
    r, q = rows[-1]["real"], rows[-1]["random"]
    print(f"  L{l:2d} smass {rows[-1]['smass']:.2f} | [S,E] real {r['commutator']:.3f} rand {q['commutator']:.3f} (x{r['commutator']/max(q['commutator'],1e-9):.2f}) | IPR real {r['ipr']:.4f} rand {q['ipr']:.4f} (x{r['ipr']/max(q['ipr'],1e-9):.0f}) | top-column share real {r['col_share']:.2f} rand {q['col_share']:.2f}")
summ = {"median_commutator_ratio_real_over_random": round(float(np.median([r["real"]["commutator"] / r["random"]["commutator"] for r in rows])), 3),
        "median_ipr_ratio": round(float(np.median([r["real"]["ipr"] / r["random"]["ipr"] for r in rows])), 1),
        "median_col_share_real": round(float(np.median([r["real"]["col_share"] for r in rows])), 3),
        "median_col_share_random": round(float(np.median([r["random"]["col_share"] for r in rows])), 3)}
json.dump({"model": NAME, "protocol": {"n_windows": NWIN, "seq": SEQ, "reps": REPS}, "rows": rows, "summary": summ}, open("alignment_study/deviation_structure_dev.json", "w"), indent=1)
print("SUMMARY", json.dumps(summ)); print("DEVIATION_DONE")
