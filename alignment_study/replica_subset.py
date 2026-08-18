"""Blind subset replication of the preregistered common protocol.

Independent implementation (no code shared with other scripts in this
directory). Reduced scale: Qwen/Qwen2.5-0.5B, attention layer indices
{2, 11, 20}, first 6 windows, 4 draws per surrogate family,
numpy default_rng(11), families: plain, sinkfix, altsink.

Outputs: replica_subset.json (same directory).
"""

import json
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "replica_subset.json")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYERS = [2, 11, 20]
N_WINDOWS = 6
N_DRAWS = 4
SEED = 11
MAX_LEN = 64
FAMILIES = ["plain", "sinkfix", "altsink"]


def layer_stats(A):
    """A: (n_heads, T, T) float64. Returns r1, gap, gn."""
    n = A.shape[0]
    G = (A - A.transpose(0, 2, 1)) / 2.0
    gn = np.linalg.norm(G, axis=(1, 2))
    # Vectorized pairwise commutator norms: P[i, j] = G_i @ G_j
    P = np.matmul(G[:, None, :, :], G[None, :, :, :])
    comm = P - P.transpose(1, 0, 2, 3)
    C = np.linalg.norm(comm, axis=(2, 3))
    iu = np.triu_indices(n, k=1)
    x = C[iu]
    y = np.outer(gn, gn)[iu]
    r1 = float(np.corrcoef(x, y)[0, 1])
    ev = np.linalg.eigvalsh(C)
    gap = float(ev[-1] / abs(ev[-2]))
    return r1, gap, gn


def make_surrogate(A, family, rng):
    """Build one surrogate draw. A: (n_heads, T, T) float64, causal."""
    n, T, _ = A.shape
    S = A.copy()
    for h in range(n):
        for i in range(1, T):
            if family == "plain":
                S[h, i, :i] = A[h, i, rng.permutation(i)]
            elif family == "sinkfix":
                if i >= 2:
                    idx = 1 + rng.permutation(i - 1)
                    S[h, i, 1:i] = A[h, i, idx]
            elif family == "altsink":
                t = h % 4
                row = A[h, i, :i].copy()
                if t <= i - 1:
                    j = int(np.argmax(row))
                    row[t], row[j] = row[j], row[t]
                    other = np.array([k for k in range(i) if k != t], dtype=int)
                    if other.size > 1:
                        row[other] = row[other][rng.permutation(other.size)]
                    S[h, i, :i] = row
                else:
                    S[h, i, :i] = row[rng.permutation(i)]
            else:
                raise ValueError(family)
    return S


def invariant_devs(A, S, family, gn_real):
    """Max abs deviations of the preserved invariants."""
    devs = {}
    devs["row_sums"] = float(np.max(np.abs(S.sum(axis=2) - A.sum(axis=2))))
    devs["diagonal"] = float(
        np.max(np.abs(np.diagonal(S, axis1=1, axis2=2) - np.diagonal(A, axis1=1, axis2=2)))
    )
    G_s = (S - S.transpose(0, 2, 1)) / 2.0
    gn_s = np.linalg.norm(G_s, axis=(1, 2))
    devs["gn"] = float(np.max(np.abs(gn_s - gn_real)))
    devs["row_perm"] = float(
        np.max(np.abs(np.sort(S, axis=2) - np.sort(A, axis=2)))
    )
    if family == "sinkfix":
        devs["col0"] = float(np.max(np.abs(S[:, :, 0] - A[:, :, 0])))
    return devs


def main():
    print("loading dataset...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.split()) > 40][:12]
    texts = texts[:N_WINDOWS]
    print(f"{len(texts)} windows", flush=True)

    print("loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager",
        output_attentions=True,
    )
    model.eval()

    # attn[layer][window] = (n_heads, T, T) float64
    attn = {l: [] for l in LAYERS}
    with torch.no_grad():
        for w, text in enumerate(texts):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            out = model(**enc)
            for l in LAYERS:
                A = out.attentions[l].squeeze(0).to(torch.float64).numpy()
                attn[l].append(A)
            print(f"window {w}: T={enc['input_ids'].shape[1]}", flush=True)

    rng = np.random.default_rng(SEED)
    results = {"model": MODEL_NAME, "layers": {}, "seed": SEED,
               "n_windows": N_WINDOWS, "n_draws": N_DRAWS}
    worst_dev = 0.0

    for l in LAYERS:
        real_r1s, real_gaps, col0 = [], [], []
        fam_r1s = {f: [] for f in FAMILIES}
        layer_devs = {f: {} for f in FAMILIES}
        for w in range(N_WINDOWS):
            A = attn[l][w]
            r1, gap, gn = layer_stats(A)
            real_r1s.append(r1)
            real_gaps.append(gap)
            col0.append(float(A[:, :, 0].mean()))
            for fam in FAMILIES:
                for d in range(N_DRAWS):
                    S = make_surrogate(A, fam, rng)
                    devs = invariant_devs(A, S, fam, gn)
                    for k, v in devs.items():
                        layer_devs[fam][k] = max(layer_devs[fam].get(k, 0.0), v)
                        worst_dev = max(worst_dev, v)
                    r1_s, _, _ = layer_stats(S)
                    fam_r1s[fam].append(r1_s)
            print(f"layer {l} window {w}: real_r1={r1:+.4f} col0={col0[-1]:.3f}",
                  flush=True)

        entry = {
            "mean_col0": float(np.mean(col0)),
            "real_r1_median": float(np.median(real_r1s)),
            "real_gap_median": float(np.median(real_gaps)),
            "invariant_max_devs": layer_devs,
        }
        for fam in FAMILIES:
            entry[f"{fam}_r1_median"] = float(np.median(fam_r1s[fam]))
        results["layers"][str(l)] = entry
        print(f"layer {l} done: real={entry['real_r1_median']:+.4f} "
              f"plain={entry['plain_r1_median']:+.4f} "
              f"sinkfix={entry['sinkfix_r1_median']:+.4f} "
              f"altsink={entry['altsink_r1_median']:+.4f}", flush=True)

    results["invariants_ok"] = bool(worst_dev < 1e-9)
    results["worst_invariant_dev"] = worst_dev
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {OUT_PATH}; worst invariant dev {worst_dev:.3e}", flush=True)


if __name__ == "__main__":
    main()
