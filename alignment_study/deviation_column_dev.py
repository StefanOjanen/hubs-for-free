# Development diagnostic (labeled): where does the deviation energy sit
# relative to the sink geometry? For each head, E_h = G_h - a_h S is split
# into the entries in the sink column (and its mirrored row, since E_h is
# skew), the entries in the first sub-diagonal (previous token) and its
# mirror, and the rest. Reported as energy shares per layer on the
# development model, real deviations against random skew matrices of the
# same norm orthogonal to S. Output: deviation_column_dev.json.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, shared_mode, sink_column
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, NWIN, STRIDE, SEQ = "Qwen/Qwen2.5-0.5B", 8, 40, 64
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)
rg = np.random.default_rng(0)


def unit(M):
    return M / np.sqrt((M ** 2).sum())


def shares(E, c):
    T = E.shape[0]; tot = (E ** 2).sum()
    col = (E[:, c] ** 2).sum() + (E[c, :] ** 2).sum() - E[c, c] ** 2
    sub = sum(E[i, i - 1] ** 2 + E[i - 1, i] ** 2 for i in range(1, T))
    return col / tot, sub / tot


tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, pick_dtype(NAME, DEV, "auto"))).eval().to(DEV)
L = model.config.num_hidden_layers
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
acc = {l: {"real_col": [], "real_sub": [], "rand_col": [], "rand_sub": [], "smass": []} for l in range(L)}
for ids in wins:
    out = model(**{k: v.to(DEV) for k, v in ids.items()})
    for l in range(L):
        A = out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)
        n, T, _ = A.shape; c = sink_column(A)
        G = generators(A); S = unit(shared_mode(G)[0])
        acc[l]["smass"].append(float(A[:, c + 1:, c].mean()))
        for h in range(n):
            E = G[h] - (G[h] * S).sum() * S
            rc, rs = shares(E, c); acc[l]["real_col"].append(rc); acc[l]["real_sub"].append(rs)
            M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2; K -= (K * S).sum() * S
            qc, qs = shares(K, c); acc[l]["rand_col"].append(qc); acc[l]["rand_sub"].append(qs)
    del out
del model; release_memory(DEV)
rows = []
for l in range(L):
    r = {"layer": l, "smass": round(float(np.mean(acc[l]["smass"])), 3),
         "real_sink_col_share": round(float(np.median(acc[l]["real_col"])), 3), "real_prev_token_share": round(float(np.median(acc[l]["real_sub"])), 3),
         "random_sink_col_share": round(float(np.median(acc[l]["rand_col"])), 3), "random_prev_token_share": round(float(np.median(acc[l]["rand_sub"])), 3)}
    rows.append(r)
    print(f"  L{l:2d} smass {r['smass']:.2f} | deviation energy in sink column: real {r['real_sink_col_share']:.2f} random {r['random_sink_col_share']:.3f} | in previous-token band: real {r['real_prev_token_share']:.2f} random {r['random_prev_token_share']:.3f}")
hs = [r for r in rows if r["smass"] > 0.4]
summ = {"median_sink_col_share_high_sink": round(float(np.median([r["real_sink_col_share"] for r in hs])), 3),
        "median_prev_token_share_high_sink": round(float(np.median([r["real_prev_token_share"] for r in hs])), 3),
        "median_sink_col_share_random": round(float(np.median([r["random_sink_col_share"] for r in rows])), 3)}
json.dump({"model": NAME, "protocol": {"n_windows": NWIN, "seq": SEQ}, "rows": rows, "summary": summ}, open("alignment_study/deviation_column_dev.json", "w"), indent=1)
print("SUMMARY", json.dumps(summ)); print("COLUMN_DONE")
