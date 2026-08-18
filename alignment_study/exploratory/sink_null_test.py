import json, time
from itertools import combinations
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

torch.set_grad_enabled(False)

MODEL, SEQ, LAYER, NWIN, NDRAW = "Qwen/Qwen2.5-0.5B", 64, 21, 10, 8
OUT = ("/private/tmp/claude-501/-Users-stefanojanen-Documents-The-Emergent-"
       "Coordinator-Hypothesis/29d2a60f-5423-4601-9108-e586ff2b2a8c/"
       "scratchpad/sink_null_results.json")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, output_attentions=True,
    attn_implementation="eager", dtype=torch.float32).eval()
H = model.config.num_attention_heads
print("MODEL LOADED", H, "heads", flush=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                  split="validation")
texts = [t for t in ds["text"] if len(t.split()) > 40][:NWIN]
print("TEXTS", len(texts), flush=True)


def stats(A):
    # A: (H, T, T) float64
    G = 0.5 * (A - np.transpose(A, (0, 2, 1)))
    gn = np.linalg.norm(G, axis=(1, 2))
    n = A.shape[0]
    C = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        C[i, j] = C[j, i] = np.linalg.norm(G[i] @ G[j] - G[j] @ G[i])
    tr = np.triu_indices(n, 1)
    r1 = float(np.corrcoef(C[tr], np.outer(gn, gn)[tr])[0, 1])
    ev = np.linalg.eigvalsh(C)
    gap = float(ev[-1] / abs(ev[-2]))
    return gn, r1, gap


def sur_plain(A, rg):
    S = A.copy()
    for h in range(A.shape[0]):
        for i in range(1, A.shape[1]):
            S[h, i, :i] = A[h, i, rg.permutation(i)]
    return S


def sur_sinkfix(A, rg):
    S = A.copy()
    for h in range(A.shape[0]):
        for i in range(2, A.shape[1]):
            S[h, i, 1:i] = A[h, i, 1 + rg.permutation(i - 1)]
    return S


def sur_localfix(A, rg):
    S = A.copy()
    for h in range(A.shape[0]):
        for i in range(3, A.shape[1]):
            S[h, i, 1:i - 1] = A[h, i, 1 + rg.permutation(i - 2)]
    return S


FAMS = {"plain": sur_plain, "sinkfix": sur_sinkfix, "localfix": sur_localfix}
rg = np.random.default_rng(0)

real_r1s, real_gaps = [], []
fam_r1 = {f: [] for f in FAMS}
fam_gap = {f: [] for f in FAMS}
inv = {"rowsum": 0.0, "gnorm": 0.0, "col0": 0.0, "prev": 0.0}

t0 = time.time()
for w, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
    out = model(**ids)
    A = out.attentions[LAYER - 1].squeeze(0).to(torch.float64).numpy()
    T = A.shape[1]
    gn, r1, gap = stats(A)
    real_r1s.append(r1)
    real_gaps.append(gap)
    rowsum = A.sum(-1)
    for fam, fn in FAMS.items():
        for d in range(NDRAW):
            S = fn(A, rg)
            gns, r1s, gaps = stats(S)
            fam_r1[fam].append(r1s)
            fam_gap[fam].append(gaps)
            inv["rowsum"] = max(inv["rowsum"],
                                float(np.abs(S.sum(-1) - rowsum).max()))
            inv["gnorm"] = max(inv["gnorm"], float(np.abs(gns - gn).max()))
            if fam in ("sinkfix", "localfix"):
                inv["col0"] = max(inv["col0"],
                                  float(np.abs(S[:, :, 0] - A[:, :, 0]).max()))
            if fam == "localfix":
                prev = np.arange(1, T)
                inv["prev"] = max(
                    inv["prev"],
                    float(np.abs(S[:, prev, prev - 1]
                                 - A[:, prev, prev - 1]).max()))
    print("WIN", w, int(time.time() - t0), "s", flush=True)

results = {
    "n_windows": len(texts),
    "n_draws": NDRAW,
    "layer": LAYER,
    "real": {"median_r1": float(np.median(real_r1s)),
             "median_gap": float(np.median(real_gaps)),
             "r1_per_window": real_r1s,
             "gap_per_window": real_gaps},
    "families": {f: {"median_r1": float(np.median(fam_r1[f])),
                     "median_gap": float(np.median(fam_gap[f]))}
                 for f in FAMS},
    "invariants_max_abs_dev": inv,
    "invariants_ok": all(v <= 1e-9 for v in inv.values()),
}
with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps({k: results[k] for k in
                  ("real", "families", "invariants_max_abs_dev",
                   "invariants_ok")}, indent=1))
