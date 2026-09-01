# SCALE ROUND v2. Same registered protocol as PREREGISTRATION3.md and the
# same statistics as scale_round.py; the implementation is vectorized so a
# free Colab session can finish (session 1 was reclaimed mid-run after
# model 1; deviation documented in RUNLOG.md). Differences from v1:
# 1. Surrogate rows are permuted in batched blocks (numpy Generator.permuted
#    across draws x heads per row index), so the surrogate DISTRIBUTIONS are
#    identical but the concrete random draws differ from v1's stream.
# 2. Commutator norms run on the GPU via torch when available.
# 3. Runs the four models session 1 did not finish; Qwen2.5-3B's completed
#    session-1 results are merged from the captured block at evaluation.
# Validate locally with: python alignment_study/scale_round_v2.py --validate
import json
import sys
import traceback
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
from common import (wikitext_windows, generators, gnorms, r1, rho,
                    shared_mode, sink_column, col_mass, sink_generator,
                    zstats, zmed_of)
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = ["Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1",
          "allenai/OLMo-2-1124-7B", "microsoft/Phi-3-mini-4k-instruct"]
NWIN, STRIDE, SEQ, KPLAIN, KFAM = 12, 40, 64, 12, 8
CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"
torch.set_grad_enabled(False)


def coup_batch(Gs):
    """Gs (m, n, T, T) float64 numpy -> (m, n, n) commutator Frobenius norms,
    on GPU in float32 when available (norm error vs float64 is far below
    draw noise), chunked to bound memory."""
    m, n, T, _ = Gs.shape
    out = np.zeros((m, n, n))
    step = max(1, int(2e8 / (n * n * T * T * 4)))
    for a in range(0, m, step):
        g = torch.as_tensor(Gs[a:a + step], dtype=torch.float32, device=DEV)
        P = torch.einsum('maij,mbjk->mabik', g, g)
        K = P - P.permute(0, 2, 1, 3, 4)
        out[a:a + step] = K.square().sum((-1, -2)).sqrt().cpu().numpy()
    return out


def sur_plain_batch(A, rg, K):
    n, T, _ = A.shape
    S = np.repeat(A[None], K, 0)
    for i in range(1, T):
        block = S[:, :, i, :i].reshape(K * n, i)
        S[:, :, i, :i] = rg.permuted(block, axis=1).reshape(K, n, i)
    return S


def sur_sinkfix_batch(A, rg, K, s):
    n, T, _ = A.shape
    S = np.repeat(A[None], K, 0)
    for i in range(1, T):
        cols = np.array([c for c in range(i) if c != s])
        if len(cols) > 1:
            block = S[:, :, i, cols].reshape(K * n, len(cols))
            S[:, :, i, cols] = rg.permuted(block, axis=1).reshape(K, n, len(cols))
    return S


def sur_alt2_batch(A, rg, K):
    n, T, _ = A.shape
    S = np.repeat(A[None], K, 0)
    t = np.arange(n) + 1
    for i in range(1, T):
        for h in range(n):
            block = S[:, h, i, :i]
            if t[h] < i:
                jmax = block.argmax(1)
                rows = np.arange(K)
                vmax = block[rows, jmax].copy()
                block[rows, jmax] = block[rows, t[h]]
                block[rows, t[h]] = vmax
                rest = np.array([c for c in range(i) if c != t[h]])
                block[:, rest] = rg.permuted(block[:, rest], axis=1)
            else:
                S[:, h, i, :i] = rg.permuted(block, axis=1)
    return S


def layer_stats(A, rg):
    n, T, _ = A.shape
    iu = np.triu_indices(n, 1)
    G = generators(A)
    gn = gnorms(G)
    C = coup_batch(G[None])[0]
    s = sink_column(A)
    Smat, a, e, frac = shared_mode(G)
    d = {"s": s, "smass": col_mass(A, s), "r1_real": r1(C, gn, iu),
         "rho_real": rho(G, gn), "sharedE": frac,
         "cv_gn": float(gn.std() / gn.mean()),
         "cosS": float(abs(np.sum(Smat / np.sqrt((Smat ** 2).sum())
                                  * sink_generator(T, s))))}
    Sp = sur_plain_batch(A, rg, KPLAIN)
    Gp = np.stack([generators(x) for x in Sp])
    Cp = coup_batch(Gp)
    gnp = np.stack([gnorms(g) for g in Gp])
    plains = [Cp[k] for k in range(KPLAIN)]
    zr, mu, sd = zstats(C, plains, iu)
    d.update(zmed_real=zr,
             r1_plain=float(np.median([r1(Cp[k], gnp[k], iu) for k in range(KPLAIN)])),
             rho_plain=float(np.median([rho(Gp[k], gnp[k]) for k in range(KPLAIN)])),
             zmed_plain=float(np.median([zmed_of(Cp[k], mu, sd, iu) for k in range(KPLAIN)])))
    Sf = sur_sinkfix_batch(A, rg, KFAM, s)
    Gf = np.stack([generators(x) for x in Sf])
    Cf = coup_batch(Gf)
    gnf = np.stack([gnorms(g) for g in Gf])
    d.update(zmed_sinkfix=float(np.median([zmed_of(Cf[k], mu, sd, iu) for k in range(KFAM)])),
             r1_sinkfix=float(np.median([r1(Cf[k], gnf[k], iu) for k in range(KFAM)])))
    Sa = sur_alt2_batch(A, rg, KFAM)
    Ga = np.stack([generators(x) for x in Sa])
    Ca = coup_batch(Ga)
    gna = np.stack([gnorms(g) for g in Ga])
    d.update(zmed_alt2=float(np.median([zmed_of(Ca[k], mu, sd, iu) for k in range(KFAM)])),
             r1_alt2=float(np.median([r1(Ca[k], gna[k], iu) for k in range(KFAM)])),
             rho_alt2=float(np.median([rho(Ga[k], gna[k]) for k in range(KFAM)])))
    # invariant spot-checks (first draw of each family)
    d["inv_dev"] = float(max(
        np.abs(np.sort(A, -1) - np.sort(Sp[0], -1)).max(),
        np.abs(np.sort(A, -1) - np.sort(Sf[0], -1)).max(),
        np.abs(np.sort(A, -1) - np.sort(Sa[0], -1)).max(),
        np.abs(A.sum(-1) - Sp[0].sum(-1)).max(),
        np.abs(A[:, min(s + 1, T - 1):, s] - Sf[0][:, min(s + 1, T - 1):, s]).max()))
    return d


def run_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, output_attentions=True, attn_implementation="eager",
        dtype=torch.bfloat16 if CUDA else torch.float32,
        device_map="auto" if CUDA else None).eval()
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    rg = np.random.default_rng(0)
    acc = {l: [] for l in range(L)}
    for w, ids in enumerate(wins):
        ids = {k: v.to(model.device) for k, v in ids.items()}
        out = model(**ids)
        for l in range(L):
            A = (out.attentions[l].squeeze(0).float().cpu().numpy()
                 .astype(np.float64))
            acc[l].append(layer_stats(A, rg))
        del out
        print(name, "WIN", w, flush=True)
    rows = []
    for l in range(L):
        ds = acc[l]
        agg = {"layer": l}
        for k in ds[0]:
            vals = [d[k] for d in ds]
            agg[k] = round(float(np.mean(vals)) if k in ("smass", "sharedE", "inv_dev")
                           else float(np.median(vals)), 6 if k == "inv_dev" else 4)
        rows.append(agg)
    del model
    import gc
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    return rows


if "--validate" in sys.argv:
    # anchor check on the dev model, layers 2/11/20, 6 windows
    name = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, output_attentions=True, attn_implementation="eager",
        dtype=torch.float32).eval()
    wins = wikitext_windows(6, STRIDE, SEQ, tok)
    rg = np.random.default_rng(0)
    for l in (2, 11, 20):
        vals = []
        for ids in wins:
            out = model(**ids)
            A = out.attentions[l].squeeze(0).numpy().astype(np.float64)
            vals.append(layer_stats(A, rg))
        med = {k: float(np.median([v[k] for v in vals]))
               for k in ("r1_real", "r1_plain", "r1_sinkfix", "r1_alt2",
                         "zmed_real", "inv_dev")}
        print("VALIDATE layer", l, json.dumps({k: round(v, 4) for k, v in med.items()}), flush=True)
    sys.exit(0)

if __name__ == "__main__":
    print("DEVICE", "cuda:" + torch.cuda.get_device_name(0) if CUDA else "cpu", flush=True)
    res = {"models": {}}
    for name in MODELS:
        try:
            rows = run_model(name)
            res["models"][name] = rows
            print("MODEL_RESULT_START " + name)
            print(json.dumps(rows))
            print("MODEL_RESULT_END", flush=True)
        except Exception:
            res["models"][name] = {"error": traceback.format_exc()[-400:]}
            print("MODEL_FAILED", name, flush=True)
    json.dump(res, open("alignment_study/scale_round_v2.json", "w"), indent=1)
    print("V2_DONE", flush=True)
