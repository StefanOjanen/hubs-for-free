# POST-REVIEW ROBUSTNESS ADDENDUM (not preregistered; run after the A1-A7
# scorecard was fixed). Addresses four issues raised in adversarial review:
# 1. GQA confound: Qwen2.5-0.5B has 14 query heads sharing 2 KV heads
#    (group = h // 7). If shared keys drive cross-head generator alignment,
#    within-KV-group cosines should exceed across-group cosines. Stated
#    expectation before running: sink sharing spans all heads, so within
#    and across should be comparable at high-sink layers.
# 2. Window dependence: the main runs used the first 12 wikitext texts
#    (adjacent). Here: 48 texts sampled with stride 20 across the filtered
#    validation list, plus percentile bootstrap CIs over windows.
# 3. Term decomposition (review demand): measure the cross terms
#    a_i[S,E_j] - a_j[S,E_i] and the deviation term [E_i,E_j] separately;
#    only the shared x shared term cancels identically, so the wording
#    "second order" is replaced by "shared-mode contribution cancels".
# 4. A7 joint criterion emitted by a committed script (review noted it was
#    never computed in one place): recomputed here from the two JSONs.
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

torch.set_grad_enabled(False)
MODEL, SEQ, STRIDE, NWIN, NDRAW, BOOT = "Qwen/Qwen2.5-0.5B", 64, 20, 48, 4, 1000
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, output_attentions=True, attn_implementation="eager",
    dtype=torch.float32).eval()
H = model.config.num_attention_heads
HKV = model.config.num_key_value_heads
GRP = H // HKV
L = model.config.num_hidden_layers
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
pool = [t for t in ds["text"] if len(t.split()) > 40]
texts = [pool[i * STRIDE] for i in range(NWIN) if i * STRIDE < len(pool)]
print("windows", len(texts), "| H", H, "| KV heads", HKV, "| group size", GRP, flush=True)

def coup(G):
    P = np.einsum('aij,bjk->abik', G, G)
    K = P - P.transpose(1, 0, 2, 3)
    return np.sqrt((K ** 2).sum((2, 3)))

def r1_of(C, gn, iu):
    return float(np.corrcoef(C[iu], np.outer(gn, gn)[iu])[0, 1])

rg = np.random.default_rng(0)
iu = None
res_layers = {l: {"real": [], "plain": [], "sinkfix": [],
                  "cos_within": [], "cos_across": [],
                  "c_within": [], "c_across": [], "col0": []} for l in range(L)}
TERM_LAYERS = [3, 4, 11, 16, 20, 21]
terms = {l: {"cross": [], "dev": [], "full": []} for l in TERM_LAYERS}
group = np.arange(H) // GRP
within = (group[:, None] == group[None, :])
for w, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
    out = model(**ids)
    for l in range(L):
        A = out.attentions[l].squeeze(0).numpy().astype(np.float64)
        n, T, _ = A.shape
        if iu is None or iu[0].max() >= n:
            iu = np.triu_indices(n, 1)
        G = (A - A.transpose(0, 2, 1)) / 2.0
        gn = np.sqrt((G ** 2).sum((1, 2)))
        C = coup(G)
        res_layers[l]["real"].append(r1_of(C, gn, iu))
        res_layers[l]["col0"].append(float(A[:, :, 0].mean()))
        # cosine Gram and normalized commutators, split by KV group
        V = G.reshape(n, -1)
        cos = (V @ V.T) / np.outer(gn, gn)
        cnorm = C / np.outer(gn, gn)
        m_off = ~np.eye(n, dtype=bool)
        res_layers[l]["cos_within"].append(float(cos[within & m_off].mean()))
        res_layers[l]["cos_across"].append(float(cos[~within].mean()))
        res_layers[l]["c_within"].append(float(cnorm[within & m_off].mean()))
        res_layers[l]["c_across"].append(float(cnorm[~within].mean()))
        # surrogates
        for fam in ("plain", "sinkfix"):
            vals = []
            for _ in range(NDRAW):
                S_ = A.copy()
                for h in range(n):
                    for i in range(1, T):
                        if fam == "plain":
                            S_[h, i, :i] = S_[h, i, rg.permutation(i)]
                        elif i >= 2:
                            S_[h, i, 1:i] = S_[h, i, 1 + rg.permutation(i - 1)]
                Gs = (S_ - S_.transpose(0, 2, 1)) / 2.0
                gns = np.sqrt((Gs ** 2).sum((1, 2)))
                vals.append(r1_of(coup(Gs), gns, iu))
            res_layers[l][fam].append(vals)
        # term decomposition at selected layers
        if l in TERM_LAYERS:
            U, sv, Vh = np.linalg.svd(V, full_matrices=False)
            Sm = Vh[0].reshape(T, T)
            a = V @ Vh[0]
            E = (V - np.outer(a, Vh[0])).reshape(n, T, T)
            SE = np.einsum('ij,bjk->bik', Sm, E) - np.einsum('bij,jk->bik', E, Sm)
            for i in range(n):
                for j in range(i + 1, n):
                    cross = a[i] * SE[j] - a[j] * SE[i]
                    dev = E[i] @ E[j] - E[j] @ E[i]
                    full = G[i] @ G[j] - G[j] @ G[i]
                    err = np.abs(cross + dev - full).max()
                    assert err < 1e-10, err
                    terms[l]["cross"].append(float(np.sqrt((cross ** 2).sum())))
                    terms[l]["dev"].append(float(np.sqrt((dev ** 2).sum())))
                    terms[l]["full"].append(float(np.sqrt((full ** 2).sum())))
    if w % 8 == 0:
        print("WIN", w, flush=True)

rgb = np.random.default_rng(1)
out_layers = []
for l in range(L):
    d = res_layers[l]
    real = np.array(d["real"])
    plain = np.array(d["plain"]).ravel()
    sink = np.array(d["sinkfix"]).ravel()
    nW = len(real)
    boots_real, boots_D = [], []
    plain_by_w = np.array(d["plain"])
    for _ in range(BOOT):
        idx = rgb.integers(0, nW, nW)
        boots_real.append(np.median(real[idx]))
        boots_D.append(np.median(plain_by_w[idx].ravel()) - np.median(real[idx]))
    lo_r, hi_r = np.percentile(boots_real, [2.5, 97.5])
    lo_D, hi_D = np.percentile(boots_D, [2.5, 97.5])
    out_layers.append({
        "layer": l, "mean_col0": float(np.mean(d["col0"])),
        "median_real_r1": float(np.median(real)),
        "real_r1_ci": [float(lo_r), float(hi_r)],
        "median_plain_r1": float(np.median(plain)),
        "median_sinkfix_r1": float(np.median(sink)),
        "D": float(np.median(plain) - np.median(real)),
        "D_ci": [float(lo_D), float(hi_D)],
        "cos_within": float(np.median(d["cos_within"])),
        "cos_across": float(np.median(d["cos_across"])),
        "cnorm_within": float(np.median(d["c_within"])),
        "cnorm_across": float(np.median(d["c_across"])),
    })
term_summary = {str(l): {
    "median_cross": float(np.median(terms[l]["cross"])),
    "median_dev": float(np.median(terms[l]["dev"])),
    "median_full": float(np.median(terms[l]["full"])),
    "frac_cross_dominant": float(np.mean(np.array(terms[l]["cross"]) > np.array(terms[l]["dev"]))),
} for l in TERM_LAYERS}

qm = json.load(open("alignment_study/qwen_multilayer.json"))
gt = json.load(open("alignment_study/gram_theorem.json"))
qrows = {r["layer"]: r for r in qm["per_layer"]}
grows = {r["layer"]: r for r in gt["per_layer"]}
qual = [l for l in range(L)
        if qrows[l]["plain_r1"] - qrows[l]["real_r1"] > 0.3]
joint = [l for l in qual
         if qrows[l]["altsink_r1"] > 0.7
         and grows[l]["rho_altsink_median"] < 0.5 * grows[l]["rho_real_median"]]
a7 = {"qualifying_layers": qual, "joint_pass_layers": joint,
      "joint_count": f"{len(joint)}/{len(qual)}",
      "verdict": "FAIL" if len(joint) < (2 / 3) * len(qual) else "PASS"}

res = {"model": MODEL, "n_windows": len(texts), "stride": STRIDE,
       "n_draws": NDRAW, "kv_heads": HKV, "group_size": GRP,
       "per_layer": out_layers, "term_decomposition": term_summary,
       "A7_joint_recomputed": a7}
json.dump(res, open("alignment_study/robustness_addendum.json", "w"), indent=1)
print(json.dumps(a7))
print("layer | col0 | real_r1 [CI] | plain | sinkfix | D [CI] | cosW/cosA | cW/cA")
for r in out_layers:
    print(f"{r['layer']:>3} | {r['mean_col0']:.2f} | {r['median_real_r1']:+.2f} "
          f"[{r['real_r1_ci'][0]:+.2f},{r['real_r1_ci'][1]:+.2f}] | {r['median_plain_r1']:+.2f} | "
          f"{r['median_sinkfix_r1']:+.2f} | {r['D']:+.2f} [{r['D_ci'][0]:+.2f},{r['D_ci'][1]:+.2f}] | "
          f"{r['cos_within']:.2f}/{r['cos_across']:.2f} | {r['cnorm_within']:.2f}/{r['cnorm_across']:.2f}")
print("TERMS:", json.dumps(term_summary))
