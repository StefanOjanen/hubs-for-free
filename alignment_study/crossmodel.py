"""Cross-model study (preregistered prediction A4, PREREGISTRATION.md).

Models: GPT-2 small and Pythia-160m (12 layers, 12 heads each).
Per (layer, window): real r1 and eigen-gap of the head-coupling matrix,
plain and sinkfix surrogate families (8 draws each), mean column-0 mass.
Per-layer medians written to crossmodel.json.

Common protocol: wikitext-103-raw-v1 validation, texts with > 40 whitespace
words, first 12, tokenized with truncation at max_length=64; models float32,
eager attention, output_attentions; per-window stats in float64.

Run:  python crossmodel.py            (main study)
      python crossmodel.py --anchor   (Qwen2.5-0.5B layer-21 sanity anchor)
"""
import json
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

SEQ, NWIN, NDRAW = 64, 12, 8
MODELS = ["gpt2", "EleutherAI/pythia-160m"]
OUT = ("/Users/stefanojanen/Documents/The Emergent Coordinator Hypothesis/"
       "alignment_study/crossmodel.json")

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                  split="validation")
texts = [t for t in ds["text"] if len(t.split()) > 40][:NWIN]
print("TEXTS", len(texts), flush=True)


def stats(A):
    """A: (n_heads, T, T) float64. Returns gn, r1, gap (common protocol)."""
    G = 0.5 * (A - np.transpose(A, (0, 2, 1)))
    gn = np.linalg.norm(G, axis=(1, 2))
    # Vectorized commutators: P[i, j] = G_i @ G_j, C_ij = ||P[i,j] - P[j,i]||_F
    P = np.einsum("aij,bjk->abik", G, G)
    C = np.linalg.norm(P - P.transpose(1, 0, 2, 3), axis=(2, 3))
    n = A.shape[0]
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


FAMS = {"plain": sur_plain, "sinkfix": sur_sinkfix}


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, output_attentions=True, attn_implementation="eager",
        dtype=torch.float32).eval()
    return tok, model


def attentions_all_layers(tok, model, text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
    out = model(**ids)
    return [a.squeeze(0).to(torch.float64).numpy() for a in out.attentions]


def run_model(name):
    tok, model = load_model(name)
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    print("MODEL LOADED", name, L, "layers", H, "heads", flush=True)

    # Forward passes once, cache float64 attentions per (window, layer).
    attn = []
    for w, text in enumerate(texts):
        attn.append(attentions_all_layers(tok, model, text))
        print("FWD", name, "win", w, flush=True)

    rg = np.random.default_rng(0)
    inv = {"rowsum": 0.0, "gnorm": 0.0, "col0": 0.0}
    layers = []
    t0 = time.time()
    for l in range(L):
        real_r1s, real_gaps, col0s = [], [], []
        fam_r1 = {f: [] for f in FAMS}
        fam_gap = {f: [] for f in FAMS}
        for w in range(len(texts)):
            A = attn[w][l]
            gn, r1, gap = stats(A)
            real_r1s.append(r1)
            real_gaps.append(gap)
            col0s.append(float(A[:, :, 0].mean()))
            rowsum = A.sum(-1)
            for fam, fn in FAMS.items():
                for _ in range(NDRAW):
                    S = fn(A, rg)
                    gns, r1s, gaps = stats(S)
                    fam_r1[fam].append(r1s)
                    fam_gap[fam].append(gaps)
                    inv["rowsum"] = max(inv["rowsum"], float(
                        np.abs(S.sum(-1) - rowsum).max()))
                    inv["gnorm"] = max(inv["gnorm"], float(
                        np.abs(gns - gn).max()))
                    if fam == "sinkfix":
                        inv["col0"] = max(inv["col0"], float(
                            np.abs(S[:, :, 0] - A[:, :, 0]).max()))
        row = {
            "layer": l + 1,
            "attn_index": l,
            "mean_col0": float(np.mean(col0s)),
            "median_real_r1": float(np.median(real_r1s)),
            "median_real_gap": float(np.median(real_gaps)),
            "median_plain_r1": float(np.median(fam_r1["plain"])),
            "median_plain_gap": float(np.median(fam_gap["plain"])),
            "median_sinkfix_r1": float(np.median(fam_r1["sinkfix"])),
            "median_sinkfix_gap": float(np.median(fam_gap["sinkfix"])),
        }
        row["D_l"] = row["median_plain_r1"] - row["median_real_r1"]
        row["abs_sinkfix_minus_real"] = abs(
            row["median_sinkfix_r1"] - row["median_real_r1"])
        layers.append(row)
        print("LAYER", name, l + 1, "of", L,
              "col0=%.3f real_r1=%.3f plain_r1=%.3f sinkfix_r1=%.3f (%ds)"
              % (row["mean_col0"], row["median_real_r1"],
                 row["median_plain_r1"], row["median_sinkfix_r1"],
                 int(time.time() - t0)), flush=True)

    # Preregistered A4 evaluation for this model.
    qual = [r for r in layers if r["mean_col0"] > 0.2]
    npass = sum(1 for r in qual
                if r["D_l"] > 0.3 and r["abs_sinkfix_minus_real"] < 0.2)
    testable = len(qual) >= 4
    a4 = {
        "qualifying_layers": [r["layer"] for r in qual],
        "n_qualifying": len(qual),
        "n_pass_both": npass,
        "frac_pass": (npass / len(qual)) if qual else None,
        "testable": testable,
        "passed": (npass / len(qual) >= 2 / 3) if testable else None,
    }
    return {
        "n_layers": L,
        "n_heads": H,
        "n_windows": len(texts),
        "n_draws": NDRAW,
        "invariants_max_abs_dev": inv,
        "invariants_ok": all(v <= 1e-9 for v in inv.values()),
        "layers": layers,
        "A4": a4,
    }


def run_anchor():
    """Sanity anchor: Qwen2.5-0.5B attentions index 20 (1-indexed layer 21).
    Expect real median r1 near -0.04 and plain near 0.98."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B",
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True, output_attentions=True,
        attn_implementation="eager", dtype=torch.float32).eval()
    print("ANCHOR MODEL LOADED", flush=True)
    rg = np.random.default_rng(0)
    real_r1s, plain_r1s = [], []
    for w, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
        A = model(**ids).attentions[20].squeeze(0).to(torch.float64).numpy()
        _, r1, _ = stats(A)
        real_r1s.append(r1)
        for _ in range(NDRAW):
            _, r1s, _ = stats(sur_plain(A, rg))
            plain_r1s.append(r1s)
        print("ANCHOR win", w, "real_r1=%.4f" % r1, flush=True)
    print("ANCHOR median real r1 = %.4f (expect near -0.04)"
          % np.median(real_r1s), flush=True)
    print("ANCHOR median plain r1 = %.4f (expect near 0.98)"
          % np.median(plain_r1s), flush=True)


if __name__ == "__main__":
    if "--anchor" in sys.argv:
        run_anchor()
        sys.exit(0)
    results = {
        "protocol": {
            "dataset": "Salesforce/wikitext wikitext-103-raw-v1 validation",
            "n_windows": NWIN, "max_length": SEQ, "n_draws": NDRAW,
            "families": list(FAMS), "rng": "numpy default_rng(0)",
            "preregistration": "PREREGISTRATION.md prediction A4",
        },
        "models": {},
    }
    for name in MODELS:
        results["models"][name] = run_model(name)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)
        print("SAVED", name, "->", OUT, flush=True)
    print("DONE", flush=True)
