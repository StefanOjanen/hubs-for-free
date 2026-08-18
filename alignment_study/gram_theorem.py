"""Decomposition and theorem study (preregistered A3, A5, A6) on Qwen/Qwen2.5-0.5B.

Per layer (attentions index 0..23) and window (12 wikitext validation texts):
  1. Shared-mode decomposition: stack vec(G_h) -> (n, T*T); S = top right
     singular vector (unit Frobenius norm); a_h = <G_h, S>; E_h = G_h - a_h S;
     shared-energy fraction a_h^2 / gn_h^2, mean over heads.
  2. rho = lambda_max / n of cosine Gram M_hk = <G_h,G_k>/(gn_h gn_k) for the
     real heads and for 8 plain-surrogate and 8 altsink-surrogate draws.
  3. |cos(S, G_sink)| with G_sink the ideal causal sink generator built from
     A_sink[i,0]=1 (i>=1), A_sink[0,0]=1, antisymmetrized and unit-normalized.
  4. Mean column-0 mass.
Theorem check (A5): [G_i,G_j] = a_i [S,E_j] - a_j [S,E_i] + [E_i,E_j] exactly
(max abs deviation, float64) and Frobenius bound
||[G_i,G_j]|| <= 2(|a_i| e_j + |a_j| e_i + e_i e_j); at the three layers with
highest mean column-0 mass, median of bound/actual over head pairs x windows.

Anchor sanity: attentions index 20 (1-indexed layer 21) real median r1 ~ -0.04,
plain ~ 0.98 (qwen_results.json / exploratory/sink_null_results.json).

Outputs: alignment_study/gram_theorem.json (this directory).
"""
import json
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B"
N_TEXTS = 12
SEQ = 64
N_DRAWS = 8
ANCHOR_LAYER = 20  # attentions index, 1-indexed layer 21
OUT_JSON = ("/Users/stefanojanen/Documents/The Emergent Coordinator Hypothesis/"
            "alignment_study/gram_theorem.json")

rng = np.random.default_rng(0)


def batched_comm_norms(G):
    """Pairwise commutator Frobenius norms, batched. G: (n, T, T) float64."""
    P = np.einsum("aij,bjk->abik", G, G)          # P[a,b] = G_a @ G_b
    D = P - P.transpose(1, 0, 2, 3)               # [G_a, G_b]
    return np.linalg.norm(D, axis=(2, 3)), D


def r1_of(C, gn):
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    return float(np.corrcoef(C[iu], np.outer(gn, gn)[iu])[0, 1])


def sink_generator(T):
    A = np.zeros((T, T))
    A[1:, 0] = 1.0
    A[0, 0] = 1.0
    G = 0.5 * (A - A.T)
    return G / np.linalg.norm(G)


def make_plain(A):
    S = A.copy()
    n, T, _ = A.shape
    for h in range(n):
        for i in range(1, T):
            S[h, i, :i] = A[h, i, rng.permutation(i)]
    return S


def make_altsink(A):
    S = A.copy()
    n, T, _ = A.shape
    for h in range(n):
        t = h % 4
        for i in range(1, T):
            row = A[h, i, :i].copy()
            if t <= i - 1:
                m = int(np.argmax(row))
                row[m], row[t] = row[t], row[m]
                others = np.array([c for c in range(i) if c != t], dtype=int)
                if len(others) > 1:
                    row[others] = row[others][rng.permutation(len(others))]
                S[h, i, :i] = row
            else:
                S[h, i, :i] = row[rng.permutation(i)]
    return S


def gen_stats(A):
    G = 0.5 * (A - A.transpose(0, 2, 1))
    gn = np.linalg.norm(G, axis=(1, 2))
    return G, gn


def rho_of(G, gn):
    n = G.shape[0]
    V = G.reshape(n, -1)
    M = (V @ V.T) / np.outer(gn, gn)
    return float(np.linalg.eigvalsh(M)[-1] / n)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, output_attentions=True,
        attn_implementation="eager", dtype=torch.float32).eval()
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    print(f"MODEL LOADED heads={H} layers={L} {time.time()-t0:.0f}s", flush=True)

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="validation")
    texts = [t for t in ds["text"] if len(t.split()) > 40][:N_TEXTS]
    print(f"TEXTS {len(texts)}", flush=True)

    # per layer accumulators
    acc = {l: {"col0": [], "shared": [], "rho_real": [], "rho_plain": [],
               "rho_altsink": [], "cos_sink": [], "r1_real": [],
               "id_dev": [], "ratios": [], "bound_viol": []}
           for l in range(L)}
    anchor_plain_r1 = []
    inv = {"rowsum": 0.0, "gnorm": 0.0, "perm_sorted": 0.0}

    for w, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
        out = model(**ids)
        T = ids["input_ids"].shape[1]
        gs_vec = sink_generator(T).ravel()
        for l in range(L):
            A = out.attentions[l].squeeze(0).numpy().astype(np.float64)
            n = A.shape[0]
            G, gn = gen_stats(A)
            V = G.reshape(n, -1)

            # 1. shared-mode decomposition
            _, sv, Vt = np.linalg.svd(V, full_matrices=False)
            Svec = Vt[0]                      # unit norm
            a = V @ Svec
            E = V - np.outer(a, Svec)
            e = np.linalg.norm(E, axis=1)
            acc[l]["shared"].append(float(np.mean(a ** 2 / gn ** 2)))

            # 2. rho real
            acc[l]["rho_real"].append(rho_of(G, gn))

            # 3. cosine against ideal sink generator
            acc[l]["cos_sink"].append(float(abs(Svec @ gs_vec)))

            # 4. mean column-0 mass
            acc[l]["col0"].append(float(A[:, :, 0].mean()))

            # real commutators, r1
            C, Dcomm = batched_comm_norms(G)
            acc[l]["r1_real"].append(r1_of(C, gn))

            # A5 theorem check
            Smat = Svec.reshape(T, T)
            Em = E.reshape(n, T, T)
            SE = np.einsum("ij,bjk->bik", Smat, Em) - \
                 np.einsum("bij,jk->bik", Em, Smat)       # [S, E_b]
            PE = np.einsum("aij,bjk->abik", Em, Em)
            EE = PE - PE.transpose(1, 0, 2, 3)            # [E_a, E_b]
            RHS = (a[:, None, None, None] * SE[None, :] -
                   a[None, :, None, None] * SE[:, None] + EE)
            acc[l]["id_dev"].append(float(np.abs(Dcomm - RHS).max()))
            bound = 2.0 * (np.abs(a)[:, None] * e[None, :] +
                           np.abs(a)[None, :] * e[:, None] +
                           np.outer(e, e))
            iu = np.triu_indices(n, 1)
            acc[l]["bound_viol"].append(float((C - bound)[iu].max()))
            acc[l]["ratios"].extend((bound[iu] / C[iu]).tolist())

            # surrogates
            for fam, maker, store in (("plain", make_plain, "rho_plain"),
                                      ("altsink", make_altsink, "rho_altsink")):
                for d in range(N_DRAWS):
                    Sa = maker(A)
                    inv["rowsum"] = max(inv["rowsum"], float(
                        np.abs(Sa.sum(-1) - A.sum(-1)).max()))
                    Gs, gns = gen_stats(Sa)
                    inv["gnorm"] = max(inv["gnorm"], float(
                        np.abs(gns - gn).max()))
                    inv["perm_sorted"] = max(inv["perm_sorted"], float(
                        np.abs(np.sort(Sa, -1) - np.sort(A, -1)).max()))
                    acc[l][store].append(rho_of(Gs, gns))
                    if fam == "plain" and l == ANCHOR_LAYER:
                        Cs, _ = batched_comm_norms(Gs)
                        anchor_plain_r1.append(r1_of(Cs, gns))
        print(f"WIN {w} T={T} {time.time()-t0:.0f}s", flush=True)

    # aggregate
    per_layer = []
    for l in range(L):
        a = acc[l]
        per_layer.append({
            "layer": l,
            "mean_col0": float(np.mean(a["col0"])),
            "col0_per_window": a["col0"],
            "mean_shared_energy": float(np.mean(a["shared"])),
            "shared_energy_per_window": a["shared"],
            "rho_real_median": float(np.median(a["rho_real"])),
            "rho_real_per_window": a["rho_real"],
            "rho_plain_median": float(np.median(a["rho_plain"])),
            "rho_altsink_median": float(np.median(a["rho_altsink"])),
            "cos_S_sink_median": float(np.median(a["cos_sink"])),
            "cos_S_sink_mean": float(np.mean(a["cos_sink"])),
            "cos_S_sink_per_window": a["cos_sink"],
            "r1_real_median": float(np.median(a["r1_real"])),
            "theorem_identity_max_dev": float(np.max(a["id_dev"])),
            "bound_max_violation": float(np.max(a["bound_viol"])),
            "bound_over_actual_median": float(np.median(a["ratios"])),
        })

    col0 = np.array([p["mean_col0"] for p in per_layer])
    rho_r = np.array([p["rho_real_median"] for p in per_layer])
    top3 = np.argsort(col0)[::-1][:3].tolist()
    top3_medians = {str(l): float(np.median(acc[l]["ratios"])) for l in top3}

    anchor = {
        "layer_attn_index": ANCHOR_LAYER,
        "real_r1_median": per_layer[ANCHOR_LAYER]["r1_real_median"],
        "plain_r1_median": float(np.median(anchor_plain_r1)),
        "expected": {"real": -0.04, "plain": 0.98},
    }
    print("ANCHOR", anchor, flush=True)

    invariants_ok = all(v < 1e-9 for v in inv.values())
    identity_max = float(max(p["theorem_identity_max_dev"] for p in per_layer))
    bound_ok = all(p["bound_max_violation"] <= 1e-12 for p in per_layer)

    result = {
        "model": MODEL,
        "n_windows": N_TEXTS,
        "n_draws": N_DRAWS,
        "seq_len": SEQ,
        "per_layer": per_layer,
        "A3": {
            "rho_real_ge_2x_plain_layers": int(sum(
                p["rho_real_median"] >= 2 * p["rho_plain_median"]
                for p in per_layer)),
            "n_layers": L,
            "spearman_rho_vs_col0": spearman(rho_r, col0),
        },
        "A5": {
            "identity_max_abs_dev_overall": identity_max,
            "bound_holds_everywhere": bound_ok,
            "top3_col0_layers": top3,
            "top3_bound_over_actual_median": top3_medians,
        },
        "A6": {
            "layers_col0_gt_0.3": [int(l) for l in range(L)
                                   if col0[l] > 0.3],
            "cos_gt_0.7_at_those": {
                str(l): bool(per_layer[l]["cos_S_sink_median"] > 0.7)
                for l in range(L) if col0[l] > 0.3},
        },
        "anchor_sanity": anchor,
        "invariants_max_abs_dev": inv,
        "invariants_ok": invariants_ok,
        "runtime_s": float(time.time() - t0),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1)
    print("SAVED", OUT_JSON, f"{time.time()-t0:.0f}s", flush=True)

    hdr = ("layer | mean_col0 | mean_shared_energy | rho_real | rho_plain | "
           "rho_altsink | cos_S_sink")
    print(hdr, flush=True)
    for p in per_layer:
        print(f"{p['layer']:5d} | {p['mean_col0']:9.4f} | "
              f"{p['mean_shared_energy']:18.4f} | {p['rho_real_median']:8.4f} | "
              f"{p['rho_plain_median']:9.4f} | {p['rho_altsink_median']:11.4f} | "
              f"{p['cos_S_sink_median']:10.4f}", flush=True)


if __name__ == "__main__":
    main()
