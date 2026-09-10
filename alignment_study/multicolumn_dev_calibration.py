# T3.3 development calibration (labeled, not a registered run): multi-column
# sink detection and column-preserving surrogates. Round 3 found that OLMo-2
# and the last five layers of Qwen2.5-3B sink on mid-sequence, window-varying
# columns, where the single-column surrogate loses its grip (S2 failure for
# OLMo). This script measures, on the dev model and on Qwen2.5-3B, how many
# columns a mass threshold selects per layer, how much of the real coupling
# statistic the multi-column surrogate recovers compared with the single
# column, and how the shared mode aligns with the multi-column sink operator.
# Output: multicolumn_dev_calibration.json. Informs PREREGISTRATION6.
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, r1, shared_mode, sink_column, col_mass, sink_generator
from scale_round_v2 import coup_batch, sur_plain_batch, DEV
from hubsfree.adapters import pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"]
NWIN, STRIDE, SEQ, K, KMAX, MIN_ROWS = 12, 40, 64, 8, 3, 16
THRESH = (0.10, 0.15, 0.20)
torch.set_grad_enabled(False)


def column_masses(A):
    """Layer-level mean attention mass per column over heads and the rows
    below the column (row 0 excluded), columns with >= MIN_ROWS rows only."""
    n, T, _ = A.shape
    cmax = T - 1 - MIN_ROWS
    cm = np.zeros(cmax)
    for c in range(cmax):
        rows = np.arange(max(1, c + 1), T)
        cm[c] = A[:, rows, c].mean()
    return cm


def sink_set(cm, thresh, modal):
    order = [int(c) for c in np.argsort(cm)[::-1]]
    cols = [modal] + [c for c in order if c != modal and cm[c] >= thresh][:KMAX - 1]
    return sorted(cols)


def sur_colfix_batch(A, rg, K, cols):
    n, T, _ = A.shape
    S = np.repeat(A[None], K, 0)
    keep = set(cols)
    for i in range(1, T):
        free = np.array([c for c in range(i) if c not in keep])
        if len(free) > 1:
            block = S[:, :, i, free].reshape(K * n, len(free))
            S[:, :, i, free] = rg.permuted(block, axis=1).reshape(K, n, len(free))
    return S


def multi_sink_generator(T, cols):
    G = sum(sink_generator(T, c) for c in cols)
    return G / np.sqrt((G ** 2).sum())


def layer_calib(A, rg):
    n, T, _ = A.shape
    iu = np.triu_indices(n, 1)
    G = generators(A); gn = gnorms(G); C = coup_batch(G[None])[0]
    s = sink_column(A, MIN_ROWS)
    cm = column_masses(A)
    Smat = shared_mode(G)[0]; Sn = Smat / np.sqrt((Smat ** 2).sum())
    Sp = sur_plain_batch(A, rg, K)
    Cp = coup_batch(np.stack([generators(x) for x in Sp]))
    r_real = r1(C, gn, iu)
    r_plain = float(np.median([r1(Cp[k], gnorms(generators(Sp[k])), iu) for k in range(K)]))
    d = {"s": s, "smass": col_mass(A, s), "r1_real": r_real, "r1_plain": r_plain,
         "cos_single": float(abs((Sn * sink_generator(T, s)).sum()))}
    for th in THRESH:
        cols = sink_set(cm, th, s)
        Sf = sur_colfix_batch(A, rg, K, cols)
        Cf = coup_batch(np.stack([generators(x) for x in Sf]))
        r_fix = float(np.median([r1(Cf[k], gnorms(generators(Sf[k])), iu) for k in range(K)]))
        tag = f"t{int(th * 100):02d}"
        d[f"{tag}_ncols"] = len(cols); d[f"{tag}_cols"] = cols
        d[f"{tag}_mass"] = float(sum(cm[c] for c in cols))
        d[f"{tag}_r1_fix"] = r_fix
        d[f"{tag}_R_r1"] = float(abs(r_real - r_fix) / max(abs(r_real - r_plain), 1e-9))
        d[f"{tag}_cos_multi"] = float(abs((Sn * multi_sink_generator(T, cols)).sum()))
    return d


def run(name):
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    rg = np.random.default_rng(0)
    acc = {l: [] for l in range(L)}
    t0 = time.time()
    for ids in wins:
        out = model(**{k: v.to(DEV) for k, v in ids.items()})
        for l in range(L):
            acc[l].append(layer_calib(out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64), rg))
        del out
    rows = []
    for l in range(L):
        ds = acc[l]; agg = {"layer": l}
        for k in ds[0]:
            vals = [d[k] for d in ds]
            if k == "s" or k.endswith("_cols"):
                agg[k] = sorted(set(str(v) for v in vals)) if k.endswith("_cols") else sorted(set(int(v) for v in vals))
            elif k.endswith("_ncols"):
                agg[k] = int(np.median(vals)); agg[k + "_max"] = int(max(vals))
            else:
                agg[k] = round(float(np.mean(vals)) if k in ("smass",) or k.endswith("_mass") else float(np.median(vals)), 4)
        rows.append(agg)
    del model
    release_memory(DEV)
    print(f"{name}: {L} layers, dtype {dt}, {time.time()-t0:.0f}s", flush=True)
    for r in rows:
        print(f"  L{r['layer']:2d} sink cols {r['s']} smass {r['smass']:.2f} r1 real {r['r1_real']:+.2f} plain {r['r1_plain']:+.2f} | "
              + " ".join(f"t{int(th*100):02d}: k={r[f't{int(th*100):02d}_ncols']} R={r[f't{int(th*100):02d}_R_r1']:.2f} cos {r[f't{int(th*100):02d}_cos_multi']:.2f}" for th in THRESH)
              + f" | single cos {r['cos_single']:.2f}", flush=True)
    return {"model": name, "dtype": dt, "n_windows": NWIN, "seq": SEQ, "k_draws": K, "thresholds": THRESH, "rows": rows}


if __name__ == "__main__":
    res = {"note": "development calibration, not registered", "models": {}}
    for m in MODELS:
        res["models"][m] = run(m)
        json.dump(res, open("alignment_study/multicolumn_dev_calibration.json", "w"), indent=1)
    print("MULTICOL_DONE", flush=True)
