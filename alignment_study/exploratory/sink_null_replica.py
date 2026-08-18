"""Blind replication: sink/local surrogate mechanism test on Qwen2.5-0.5B layer 21.

Reduced scale: first 4 windows, 4 surrogate draws per family, default_rng(7).
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-0.5B"
N_WINDOWS = 4
N_DRAWS = 4
SEED = 7
MAX_LEN = 64
FOCAL_ATT_IDX = 20  # out.attentions[20] == 1-indexed layer 21

OUT_JSON = "/private/tmp/claude-501/-Users-stefanojanen-Documents-The-Emergent-Coordinator-Hypothesis/29d2a60f-5423-4601-9108-e586ff2b2a8c/scratchpad/sink_null_replica.json"


def head_stats(A):
    """A: (H, T, T) float64. Returns r1, gap, gn (per-head Frobenius norm of antisym part)."""
    H = A.shape[0]
    G = (A - A.transpose(0, 2, 1)) / 2.0
    gn = np.linalg.norm(G, axis=(1, 2))
    C = np.zeros((H, H), dtype=np.float64)
    for i in range(H):
        for j in range(i + 1, H):
            comm = G[i] @ G[j] - G[j] @ G[i]
            v = np.linalg.norm(comm)
            C[i, j] = v
            C[j, i] = v
    iu = np.triu_indices(H, k=1)
    x = C[iu]
    y = np.outer(gn, gn)[iu]
    r1 = float(np.corrcoef(x, y)[0, 1])
    ev = np.linalg.eigvalsh(C)
    gap = float(ev[-1] / abs(ev[-2]))
    return r1, gap, gn


def make_surrogate(A, family, rng):
    """Row-wise permutation surrogates preserving row sums, row IPR, diagonal."""
    S = A.copy()
    H, T, _ = A.shape
    for h in range(H):
        for i in range(T):
            if family == "plain":
                if i >= 1:
                    perm = rng.permutation(i)
                    S[h, i, :i] = A[h, i, perm]
            elif family == "sinkfix":
                if i >= 2:
                    perm = rng.permutation(np.arange(1, i))
                    S[h, i, 1:i] = A[h, i, perm]
            elif family == "localfix":
                if i >= 3:
                    perm = rng.permutation(np.arange(1, i - 1))
                    S[h, i, 1:i - 1] = A[h, i, perm]
            else:
                raise ValueError(family)
    return S


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_attentions=True,
    )
    model.eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.split()) > 40]
    texts = texts[:N_WINDOWS]

    rng = np.random.default_rng(SEED)
    families = ["plain", "sinkfix", "localfix"]

    real_r1s, real_gaps = [], []
    fam_r1s = {f: [] for f in families}
    fam_gaps = {f: [] for f in families}

    inv = {
        "max_rowsum_dev": 0.0,
        "max_gnorm_dev": 0.0,
        "max_col0_dev": 0.0,
        "max_prevtok_dev": 0.0,
    }

    for w, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            out = model(**enc)
        A = out.attentions[FOCAL_ATT_IDX][0].to(torch.float64).numpy()  # (14, T, T)
        r1, gap, gn_real = head_stats(A)
        real_r1s.append(r1)
        real_gaps.append(gap)

        row_sums_real = A.sum(axis=2)
        col0_real = A[:, :, 0]
        H, T, _ = A.shape
        prev_idx = np.arange(1, T)

        for fam in families:
            for d in range(N_DRAWS):
                S = make_surrogate(A, fam, rng)
                sr1, sgap, gn_s = head_stats(S)
                fam_r1s[fam].append(sr1)
                fam_gaps[fam].append(sgap)

                dev_rowsum = float(np.max(np.abs(S.sum(axis=2) - row_sums_real)))
                inv["max_rowsum_dev"] = max(inv["max_rowsum_dev"], dev_rowsum)

                dev_gn = float(np.max(np.abs(gn_s - gn_real)))
                inv["max_gnorm_dev"] = max(inv["max_gnorm_dev"], dev_gn)

                if fam in ("sinkfix", "localfix"):
                    dev_c0 = float(np.max(np.abs(S[:, :, 0] - col0_real)))
                    inv["max_col0_dev"] = max(inv["max_col0_dev"], dev_c0)

                if fam == "localfix":
                    dev_prev = float(
                        np.max(np.abs(S[:, prev_idx, prev_idx - 1] - A[:, prev_idx, prev_idx - 1]))
                    )
                    inv["max_prevtok_dev"] = max(inv["max_prevtok_dev"], dev_prev)

        print(f"window {w}: T={T}, real r1={r1:.4f}, gap={gap:.3f}")

    results = {
        "n_windows": len(texts),
        "n_draws": N_DRAWS,
        "seed": SEED,
        "real": {
            "median_r1": float(np.median(real_r1s)),
            "median_gap": float(np.median(real_gaps)),
            "r1_per_window": real_r1s,
            "gap_per_window": real_gaps,
        },
        "surrogates": {
            f: {
                "median_r1": float(np.median(fam_r1s[f])),
                "median_gap": float(np.median(fam_gaps[f])),
                "r1_all": fam_r1s[f],
                "gap_all": fam_gaps[f],
            }
            for f in families
        },
        "invariants": inv,
        "invariants_ok": all(v < 1e-9 for v in inv.values()),
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)

    print(json.dumps({k: results[k] for k in ("invariants", "invariants_ok")}, indent=2))
    print("real median r1", results["real"]["median_r1"], "gap", results["real"]["median_gap"])
    for f in families:
        print(f, "median r1", results["surrogates"][f]["median_r1"],
              "median gap", results["surrogates"][f]["median_gap"])


if __name__ == "__main__":
    main()
