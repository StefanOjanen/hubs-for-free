# POST-HOC EXPLORATORY ANALYSIS (not preregistered; registered predictions
# are A1-A7 in PREREGISTRATION.md). Motivated by the A1 failure pattern:
# some strong-sink layers (e.g. Qwen layer index 3) show no r1 breakdown.
# Hypothesis H-posthoc: with G_h = a_h S + E_h (S the shared component,
# e_h = ||E_h||_F), the coupling matrix concentrates on the rank-1 matrix
# of RESIDUAL norm products, C_ij ~ e_i e_j, so the breakdown of r1
# against TOTAL norm products gn_i gn_j occurs exactly where the
# alignment fraction a_h^2/gn_h^2 is heterogeneous across heads.
# Also tests the hub-identity consequence: the argmax row-sum of C should
# be the max-RESIDUAL head, not the max-gnorm head.
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

torch.set_grad_enabled(False)
MODEL, NWIN, SEQ = "Qwen/Qwen2.5-0.5B", 12, 64
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, output_attentions=True, attn_implementation="eager",
    dtype=torch.float32).eval()
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
texts = [t for t in ds["text"] if len(t.split()) > 40][:NWIN]
L = model.config.num_hidden_layers

def coup(G):
    n = G.shape[0]
    P = np.einsum('aij,bjk->abik', G, G)
    K = P - np.swapaxes(P, 0, 1).transpose(0, 1, 3, 2) * 0  # placeholder, fixed below
    # commutator [G_a, G_b] = G_a G_b - G_b G_a = P[a,b] - P[b,a]
    K = P - P.transpose(1, 0, 2, 3)
    return np.sqrt((K ** 2).sum((2, 3)))

rows = []
per_layer = {l: {"r1_total": [], "r1_resid": [], "hub_is_maxresid": [],
                 "hub_is_maxgn": [], "cv_alignfrac": []} for l in range(L)}
for w, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
    out = model(**ids)
    for l in range(L):
        A = out.attentions[l].squeeze(0).numpy().astype(np.float64)
        G = (A - A.transpose(0, 2, 1)) / 2.0
        n = G.shape[0]
        gn = np.sqrt((G ** 2).sum((1, 2)))
        V = G.reshape(n, -1)
        # top principal component of the generator set
        U, sv, Vh = np.linalg.svd(V, full_matrices=False)
        S = Vh[0]
        a = V @ S
        E = V - np.outer(a, S)
        e = np.sqrt((E ** 2).sum(1))
        C = coup(G)
        iu = np.triu_indices(n, 1)
        r1_total = np.corrcoef(C[iu], np.outer(gn, gn)[iu])[0, 1]
        r1_resid = np.corrcoef(C[iu], np.outer(e, e)[iu])[0, 1]
        rs = C.sum(1)
        frac = a ** 2 / gn ** 2
        per_layer[l]["r1_total"].append(float(r1_total))
        per_layer[l]["r1_resid"].append(float(r1_resid))
        per_layer[l]["hub_is_maxresid"].append(bool(rs.argmax() == e.argmax()))
        per_layer[l]["hub_is_maxgn"].append(bool(rs.argmax() == gn.argmax()))
        per_layer[l]["cv_alignfrac"].append(float(frac.std() / frac.mean()))
    if w % 4 == 0:
        print("WIN", w, flush=True)

res = {"model": MODEL, "n_windows": NWIN, "seq_len": SEQ, "per_layer": []}
for l in range(L):
    d = per_layer[l]
    res["per_layer"].append({
        "layer": l,
        "median_r1_total": float(np.median(d["r1_total"])),
        "median_r1_resid": float(np.median(d["r1_resid"])),
        "frac_hub_is_maxresid": float(np.mean(d["hub_is_maxresid"])),
        "frac_hub_is_maxgn": float(np.mean(d["hub_is_maxgn"])),
        "median_cv_alignfrac": float(np.median(d["cv_alignfrac"])),
    })
tt = np.array([r["median_r1_total"] for r in res["per_layer"]])
rr = np.array([r["median_r1_resid"] for r in res["per_layer"]])
cv = np.array([r["median_cv_alignfrac"] for r in res["per_layer"]])
try:
    from scipy import stats as _s
    sp = float(_s.spearmanr(cv, tt).statistic)
except Exception:
    def rank(x):
        idx = np.argsort(x); rk = np.empty_like(idx, float); rk[idx] = np.arange(len(x))
        return rk
    sp = float(np.corrcoef(rank(cv), rank(tt))[0, 1])
res["summary"] = {
    "median_r1_resid_all_layers": float(np.median(rr)),
    "min_r1_resid": float(rr.min()),
    "layers_r1_resid_gt_0.8": int((rr > 0.8).sum()),
    "spearman_cv_alignfrac_vs_r1_total": sp,
}
json.dump(res, open("alignment_study/posthoc_residual_law.json", "w"), indent=1)
print(json.dumps(res["summary"]))
print("layer | r1_total | r1_resid | hub=maxresid | hub=maxgn | cv_alignfrac")
for r in res["per_layer"]:
    print(f"{r['layer']:>3} | {r['median_r1_total']:+.3f} | {r['median_r1_resid']:+.3f} | "
          f"{r['frac_hub_is_maxresid']:.2f} | {r['frac_hub_is_maxgn']:.2f} | {r['median_cv_alignfrac']:.3f}")
