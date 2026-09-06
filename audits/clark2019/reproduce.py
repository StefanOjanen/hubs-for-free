# Target 1 (Clark et al. 2019), step T2.1: reproduce the base result only.
# Claim: attention heads of BERT-base cluster by Jensen-Shannon distance
# between their attention distributions, with heads in the same layer more
# similar than heads in different layers (their Section 6, Figure 6).
# Statistic: JS(head_a, head_b) = mean over inputs and query positions of the
# Jensen-Shannon divergence between the two heads' attention rows.
# No surrogates are run here; criteria are frozen in PREREGISTRATION4.md
# after this reproduction and before any battery run.
import json
import sys
import numpy as np

sys.path.insert(0, ".")
from hubsfree.adapters import attentions, load_model, wikitext_texts

NAME, NWIN, T = "bert-base-uncased", 48, 64
tok, model = load_model(NAME)
texts = wikitext_texts(NWIN)
L, H = model.config.num_hidden_layers, model.config.num_attention_heads
acc = np.zeros((L * H, L * H)); count = 0
for w, text in enumerate(texts):
    mats, ids = attentions(tok, model, text, T)
    A = np.concatenate(mats, 0)                     # (L*H, T, T)
    P = np.clip(A, 1e-12, 1)
    # pairwise JS over all head pairs, vectorized per row block
    for i in range(A.shape[1]):
        p = P[:, i, :]                               # (LH, T)
        M = (p[:, None, :] + p[None, :, :]) / 2      # (LH, LH, T)
        kl_pm = (p[:, None, :] * (np.log(p[:, None, :]) - np.log(M))).sum(-1)
        kl_qm = (p[None, :, :] * (np.log(p[None, :, :]) - np.log(M))).sum(-1)
        acc += 0.5 * (kl_pm + kl_qm); count += 1
    if w % 8 == 0:
        print("WIN", w, flush=True)
JS = acc / count
layer = np.repeat(np.arange(L), H)
same = (layer[:, None] == layer[None, :]) & ~np.eye(L * H, dtype=bool)
diff = layer[:, None] != layer[None, :]
res = {"model": NAME, "n_windows": len(texts), "T": T,
       "mean_js_same_layer": float(JS[same].mean()),
       "mean_js_diff_layer": float(JS[diff].mean()),
       "contrast": float(JS[diff].mean() - JS[same].mean()),
       "nearest_neighbor_same_layer_frac": float(np.mean([
           layer[np.argsort(JS[k] + np.eye(L * H)[k] * 1e9)[0]] == layer[k] for k in range(L * H)])),
       "js_min": float(JS[~np.eye(L * H, dtype=bool)].min()),
       "js_max": float(JS.max())}
np.save("audits/clark2019/js_matrix.npy", JS)
json.dump(res, open("audits/clark2019/base_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
