# T4.1 development calibration (labeled, not registered): k-mode decomposition
# of a layer's generators against a small dictionary of ideal operators.
# For each layer and window, the generators G_h (n heads) are stacked as
# vectors and decomposed by SVD; the top three right singular vectors are the
# layer's principal operators, with energy fractions s_k^2 / sum s^2. Each
# principal operator is matched to a dictionary of unit-norm generators:
#   sink(c)   : all rows below c attend to column c (the ideal sink operator)
#   uniform   : causal uniform attention, A_ij = 1/(i+1) for j <= i
#   recency-1 : each row attends to the previous token
#   recency-2 : each row attends two tokens back
# Also run on a random-init Pythia-160m from config, to identify the shared
# mode that an untrained network carries (dynamics round, PREREGISTRATION5).
# Output: kmode_dev_calibration.json.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, sink_column, col_mass, sink_generator, shared_mode
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

NWIN, STRIDE, SEQ = 12, 40, 64
torch.set_grad_enabled(False)


def unit(G):
    return G / np.sqrt((G ** 2).sum())


def gen_of(A):
    return unit((A - A.T) / 2.0)


def dictionary(T, sink_cols):
    D = {}
    for c in sink_cols:
        D[f"sink({c})"] = sink_generator(T, c)
    U = np.tril(np.ones((T, T))); U = U / U.sum(-1, keepdims=True); D["uniform"] = gen_of(U)
    for k in (1, 2):
        R = np.zeros((T, T))
        for i in range(T):
            R[i, max(i - k, 0)] = 1.0
        D[f"recency-{k}"] = gen_of(R)
    return D


def kmodes(A, kmax=3):
    n, T, _ = A.shape
    G = generators(A)
    V = G.reshape(n, -1)
    U, sv, Vh = np.linalg.svd(V, full_matrices=False)
    e = sv ** 2 / (sv ** 2).sum()
    s = sink_column(A)
    D = dictionary(T, sorted({0, s}))
    out = {"s": s, "smass": col_mass(A, s), "sharedE_perhead": shared_mode(G)[3], "e": [float(x) for x in e[:kmax]]}
    for k in range(min(kmax, len(sv))):
        M = Vh[k].reshape(T, T)
        cos = {name: float(abs((unit(M) * Dg).sum())) for name, Dg in D.items()}
        best = max(cos, key=cos.get)
        out[f"m{k+1}_best"] = best; out[f"m{k+1}_cos"] = cos[best]; out[f"m{k+1}_cos_all"] = cos
    return out


def summarize(acc, L):
    rows = []
    for l in range(L):
        ds = acc[l]
        r = {"layer": l, "s": sorted(set(int(d["s"]) for d in ds)), "smass": round(float(np.mean([d["smass"] for d in ds])), 4),
             "sharedE_perhead": round(float(np.mean([d["sharedE_perhead"] for d in ds])), 4),
             "e1": round(float(np.mean([d["e"][0] for d in ds])), 4), "e2": round(float(np.mean([d["e"][1] for d in ds])), 4),
             "e3": round(float(np.mean([d["e"][2] for d in ds])), 4)}
        for k in (1, 2, 3):
            names = [d[f"m{k}_best"] for d in ds]
            mode_name = max(set(names), key=names.count)
            r[f"m{k}_best"] = mode_name; r[f"m{k}_best_frac"] = round(names.count(mode_name) / len(names), 2)
            r[f"m{k}_cos"] = round(float(np.median([d[f"m{k}_cos"] for d in ds])), 3)
            r[f"m{k}_cos_uniform"] = round(float(np.median([d[f"m{k}_cos_all"]["uniform"] for d in ds])), 3)
        rows.append(r)
    return rows


def run_trained(name):
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    acc = {l: [] for l in range(L)}
    for ids in wins:
        out = model(**{k: v.to(DEV) for k, v in ids.items()})
        for l in range(L):
            acc[l].append(kmodes(out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)))
    del model; release_memory(DEV)
    return summarize(acc, L)


def run_random_init(name):
    cfg = AutoConfig.from_pretrained(name); torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval().to(DEV)
    tok = AutoTokenizer.from_pretrained(name)
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    L = cfg.num_hidden_layers; acc = {l: [] for l in range(L)}
    for ids in wins:
        out = model(**{k: v.to(DEV) for k, v in ids.items()}, output_attentions=True)
        for l in range(L):
            acc[l].append(kmodes(out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)))
    del model; release_memory(DEV)
    return summarize(acc, L)


def show(name, rows):
    print(name, flush=True)
    for r in rows:
        print(f"  L{r['layer']:2d} smass {r['smass']:.2f} sE(head) {r['sharedE_perhead']:.2f} | e1 {r['e1']:.2f} e2 {r['e2']:.2f} e3 {r['e3']:.2f} | "
              f"m1 {r['m1_best']:>10s} cos {r['m1_cos']:.2f} (uniform {r['m1_cos_uniform']:.2f}) | m2 {r['m2_best']:>10s} cos {r['m2_cos']:.2f} | m3 {r['m3_best']:>10s} cos {r['m3_cos']:.2f}", flush=True)


res = {"note": "development calibration, not registered", "protocol": {"n_windows": NWIN, "stride": STRIDE, "seq": SEQ}, "models": {}}
rows = run_trained("Qwen/Qwen2.5-0.5B"); res["models"]["Qwen/Qwen2.5-0.5B"] = rows; show("Qwen/Qwen2.5-0.5B (trained)", rows)
rows = run_random_init("EleutherAI/pythia-160m"); res["models"]["EleutherAI/pythia-160m random init"] = rows; show("EleutherAI/pythia-160m (random init from config)", rows)
json.dump(res, open("alignment_study/kmode_dev_calibration.json", "w"), indent=1)
print("KMODE_DONE")
