# PREREGISTRATION25: cross-layer KV cache merging under the convention null.
# MiniCache (Liu et al., NeurIPS 2024) merges the key and value caches of adjacent layers token by token,
# motivated by a high cosine similarity between adjacent layers' KV states. Every attention head's
# query/key space and value/output space can be rotated, and a layer's KV heads permuted (with their query
# heads), without changing the model; a per-token cosine between two layers' caches, and a merge that pairs
# head h of layer l-1 with head h of layer l, depend on that arbitrary choice.
#
# Part A (premise), on held-out documents: per-token cosine between adjacent layers (tokens 1..) of keys and
# of values (concatenated over KV heads), of the residual stream (layer outputs), and of the normalized layer
# input; keys and values again after per-head alignment fitted on calibration documents, and after the same
# fit against token-shuffled targets (the fitting null).
# Part B (merge): the context's cache (1,536 tokens) is merged, then the NLL of the next 256 tokens is
# measured with the merged cache. Merge as in KVCache-Factory's MiniCache implementation: per head and token,
# SLERP (t = 0.6) of the two unit directions, each layer's magnitudes kept, the 5% of tokens with the largest
# angular distance per head kept unmerged; layer pairs (S, S+1), (S+2, S+3), ... from S = L/2 (MiniCache's
# setting) and, as a second setting, from S = L/4. Bases:
#   trained      the published merge
#   random0..2   layer l-1's heads in random function-preserving bases (rotary-compatible rotations of keys,
#                any rotation of values, a random permutation of KV heads), undone after the merge
#   aligned      layer l-1's heads matched to layer l's (Hungarian on post-alignment similarity) and rotated
#                into their bases (per-head orthogonal Procrustes, rotary-compatible for keys), fitted on
#                calibration documents; equivalent to folding the rotations into W_K, W_Q, W_V, W_O
#   aligned_nomatch  the same rotations with heads paired by index
# Part C (XQuant, Yang et al., EMNLP 2025): KIVI 2-bit codes (keys per channel in 32-token groups, values per
# token in 32-channel groups, the last 128 context tokens in full precision). C-statistic: the distribution of
# |code difference| between adjacent layers (XQuant's Figure 3) against the independence prediction from the two
# layers' code frequencies, and after alignment. C-function: from layer S = L/4 (even S), layer l+1 dequantizes
# layer l's codes with its own scale and zero-point (XQuant's sharing, its 2-bit form without calibration);
# arms: kivi (no sharing), trained, random (layer l quantized in a random function-preserving basis), aligned
# (layer l quantized in the basis aligned to layer l+1), kivi_rot (aligned basis, no sharing).
#   python alignment_study/xlayer_round.py [--dev] --registered=<freeze permalink>   (XLAYER_DEVICE=cpu to override)
import json
import math
import os
import sys
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "alignment_study")
import gqa_common as C
from scale_round_v2 import DEV
DEV = os.environ.get("XLAYER_DEVICE", DEV)
from hubsfree.adapters import release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION25.md>; use --dev")
if DEV_RUN:
    MODELS, OUT = ["TinyLlama/TinyLlama_v1.1", "Qwen/Qwen2.5-0.5B", "EleutherAI/pythia-410m"], "alignment_study/xlayer_dev"
else:
    MODELS, OUT = ["yaofu/llama-2-7b-80k", "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen2.5-7B", "microsoft/Phi-3-mini-4k-instruct"], "alignment_study/xlayer"
N_CAL, T_CAL, N_DOCS, CTX, CONT, N_PREMISE = 4, 1024, 32, 1536, 256, 2
INTERP, KEEP = 0.6, 0.05
QB, QG, QRES = 2, 32, 128
ARMS = ("trained", "random0", "random1", "random2", "aligned", "aligned_nomatch")
torch.set_grad_enabled(False)


def rotary_dims(cfg, hd):
    rp = getattr(cfg, "rope_parameters", None) or {}
    frac = (rp.get("partial_rotary_factor") if isinstance(rp, dict) else None) or getattr(cfg, "partial_rotary_factor", None) \
        or getattr(cfg, "rotary_pct", None) or 1.0
    return int(hd * frac)


def wikitext_docs(tok, n, T):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))
    ids = tok("\n\n".join(t for t in df["text"] if t.strip()), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    k = ids.numel() // T
    picks = np.linspace(0, k - 1, n).round().astype(int)
    return torch.stack([ids[p * T:(p + 1) * T] for p in picks])


def kv(cache, l):
    return cache.layers[l].keys, cache.layers[l].values


def tokcos(A, B):
    """Mean per-token cosine (tokens 1..) of vectors concatenated over heads; A, B: (batch, heads, T, d)."""
    a = A.float().permute(0, 2, 1, 3).reshape(A.shape[0], A.shape[2], -1)[:, 1:]
    b = B.float().permute(0, 2, 1, 3).reshape(B.shape[0], B.shape[2], -1)[:, 1:]
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def fit_alignment(Kp, Kl, Vp, Vl, r, match=True):
    """Matched head b of layer l-1 for each head a of layer l, and rotations of b's basis onto a's.
    K*, V*: (batch, heads, T, hd) float64 on the host."""
    nkv, hd = Kl.shape[1], Kl.shape[-1]
    S = np.zeros((nkv, nkv)); RK, RV = {}, {}
    for a in range(nkv):
        Yk, Yv = Kl[:, a].reshape(-1, hd), Vl[:, a].reshape(-1, hd)
        for b in range(nkv):
            if not match and a != b:
                continue
            Xk, Xv = Kp[:, b].reshape(-1, hd), Vp[:, b].reshape(-1, hd)
            RK[(a, b)] = torch.tensor(C.rope_procrustes_C((Xk.T @ Yk).numpy(), r))
            RV[(a, b)] = torch.tensor(C.procrustes_C((Xv.T @ Yv).numpy()))
            S[a, b] = float(torch.nn.functional.cosine_similarity(Xk @ RK[(a, b)], Yk, dim=-1).mean()
                            + torch.nn.functional.cosine_similarity(Xv @ RV[(a, b)], Yv, dim=-1).mean())
    if match:
        rows, cols = linear_sum_assignment(-S)
        perm = {int(a): int(b) for a, b in zip(rows, cols)}
    else:
        perm = {a: a for a in range(nkv)}
    return perm, {a: (RK[(a, perm[a])], RV[(a, perm[a])]) for a in range(nkv)}


def random_alignment(nkv, hd, r, rng):
    def rk():
        R = np.zeros((hd, hd)); h = r // 2
        if r:
            i = np.arange(h); th = rng.uniform(0, 2 * np.pi, h); c, s = np.cos(th), np.sin(th)
            R[i, i] = c; R[i, i + h] = s; R[i + h, i] = -s; R[i + h, i + h] = c
        if r < hd:
            q, x = np.linalg.qr(rng.standard_normal((hd - r, hd - r))); R[r:, r:] = q * np.sign(np.diag(x))
        return torch.tensor(R)

    def rv():
        q, x = np.linalg.qr(rng.standard_normal((hd, hd))); return torch.tensor(q * np.sign(np.diag(x)))
    perm = {a: int(b) for a, b in enumerate(rng.permutation(nkv))}
    return perm, {a: (rk(), rv()) for a in range(nkv)}


def slerp_merge(cur, prev, t=INTERP, keep=KEEP, eps=1e-6):
    """KVCache-Factory's MiniCache pair merge for one head; cur, prev: (batch, T, hd), prev in cur's basis."""
    mc, mp = cur.norm(dim=-1, keepdim=True).clamp_min(eps), prev.norm(dim=-1, keepdim=True).clamp_min(eps)
    uc, up = cur / mc, prev / mp
    cos = (uc * up).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    ang = torch.acos(cos); s = torch.sin(ang)
    e = torch.where(s.abs() > eps, torch.sin((1 - t) * ang) / s * up + torch.sin(t * ang) / s * uc, (1 - t) * up + t * uc)
    e = e / e.norm(dim=-1, keepdim=True).clamp_min(eps)
    rc, rp = e * mc, e * mp
    k = math.ceil(cur.shape[1] * keep)
    if k:
        idx = torch.topk(ang[..., 0], k, dim=-1).indices
        for bi in range(cur.shape[0]):
            rc[bi, idx[bi]] = cur[bi, idx[bi]]; rp[bi, idx[bi]] = prev[bi, idx[bi]]
    return rc, rp


def merge_cache(cache, pairs, bases):
    """bases[l] = (perm, rotations) mapping layer l-1's heads into layer l's; merged in float32."""
    for (lp, l) in pairs:
        K1, V1 = kv(cache, l); K0, V0 = kv(cache, lp)
        dt = K1.dtype
        K1f, V1f, K0f, V0f = K1.float(), V1.float(), K0.float(), V0.float()
        nK1, nV1, nK0, nV0 = K1f.clone(), V1f.clone(), K0f.clone(), V0f.clone()
        perm, rot = bases[l]
        for a in range(K1.shape[1]):
            b = perm[a]
            Rk, Rv = (x.float().to(K1.device) for x in rot[a])
            ck, pk = slerp_merge(K1f[:, a], K0f[:, b] @ Rk)
            cv, pv = slerp_merge(V1f[:, a], V0f[:, b] @ Rv)
            nK1[:, a], nV1[:, a] = ck, cv
            nK0[:, b], nV0[:, b] = pk @ Rk.T, pv @ Rv.T
        cache.layers[l].keys, cache.layers[l].values = nK1.to(dt), nV1.to(dt)
        cache.layers[lp].keys, cache.layers[lp].values = nK0.to(dt), nV0.to(dt)


def cont_nll(model, x, pairs, bases):
    ctx, cont = x[:, :CTX].to(DEV), x[:, CTX:].to(DEV)
    cache = model(input_ids=ctx, use_cache=True).past_key_values
    if pairs:
        merge_cache(cache, pairs, bases)
    lg = model(input_ids=cont, past_key_values=cache, use_cache=True).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), cont[:, 1:].reshape(-1), reduction="sum")
    return float(nll), cont.shape[1] - 1


def kivi(A, which, codes=None):
    """KIVI-quantize all but the last QRES tokens of A (batch, heads, T, hd); with codes given, dequantize those
    codes with A's own scale and zero-point (XQuant's sharing). Returns the dequantized tensor and the codes."""
    T, hd = A.shape[2], A.shape[3]; Tq = (T - QRES) // QG * QG
    Aq = A[:, :, :Tq].double()
    if which == "k":
        Z = Aq.reshape(A.shape[0], A.shape[1], Tq // QG, QG, hd); mn, mx = Z.amin(3, keepdim=True), Z.amax(3, keepdim=True)
    else:
        Z = Aq.reshape(A.shape[0], A.shape[1], Tq, hd // QG, QG); mn, mx = Z.amin(4, keepdim=True), Z.amax(4, keepdim=True)
    sc = (mx - mn).clamp_min(1e-12) / (2 ** QB - 1)
    c = torch.round((Z - mn) / sc).clamp(0, 2 ** QB - 1) if codes is None else codes
    out = A.clone(); out[:, :, :Tq] = (c * sc + mn).reshape(Aq.shape).to(A.dtype)
    return out, c


def code_diff(c1, c2):
    d = (c1 - c2).abs().flatten()
    return [float((d == k).double().mean()) for k in range(2 ** QB)]


def code_indep(c1, c2):
    p1 = np.bincount(c1.long().flatten().numpy(), minlength=2 ** QB) / c1.numel()
    p2 = np.bincount(c2.long().flatten().numpy(), minlength=2 ** QB) / c2.numel()
    out = np.zeros(2 ** QB)
    for i in range(2 ** QB):
        for j in range(2 ** QB):
            out[abs(i - j)] += p1[i] * p2[j]
    return out.tolist()


def per_head(A, rots, inv=False):
    return torch.stack([A[:, h] @ (rots[h].T if inv else rots[h]).to(A.dtype).to(A.device) for h in range(A.shape[1])], 1)


def xquant_compress(cache, pairs, arm, rot):
    """rot[l] = list over heads of (Rk, Rv) mapping layer l's basis to layer l+1's (aligned or random)."""
    lower = {l for l, _ in pairs}; upper = {l1: l for l, l1 in pairs}
    codes = {}
    for l in range(len(cache.layers)):
        for j, which in ((0, "k"), (1, "v")):
            A = cache.layers[l].keys if which == "k" else cache.layers[l].values
            Af = A.float()
            if arm == "kivi" or (l not in lower and l not in upper):
                Aq, _ = kivi(Af, which)
            elif l in lower:
                if arm in ("random", "aligned", "kivi_rot"):
                    R = [x[j] for x in rot[l]]
                    Ar, c = kivi(per_head(Af, R), which); Aq = per_head(Ar, R, inv=True)
                else:
                    Aq, c = kivi(Af, which)
                codes[(l, which)] = c
            else:
                Aq = kivi(Af, which)[0] if arm == "kivi_rot" else kivi(Af, which, codes=codes[(upper[l], which)])[0]
            if which == "k":
                cache.layers[l].keys = Aq.to(A.dtype)
            else:
                cache.layers[l].values = Aq.to(A.dtype)


def xquant_nll(model, x, pairs, arm, rot):
    ctx, cont = x[:, :CTX].to(DEV), x[:, CTX:].to(DEV)
    cache = model(input_ids=ctx, use_cache=True).past_key_values
    if arm != "full":
        xquant_compress(cache, pairs, arm, rot)
    lg = model(input_ids=cont, past_key_values=cache, use_cache=True).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), cont[:, 1:].reshape(-1), reduction="sum")
    return float(nll), cont.shape[1] - 1


def run_model(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if any(s in name.lower() for s in ("7b", "8b")) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").eval().to(DEV)
    cfg = model.config; L = cfg.num_hidden_layers; hd = cfg.hidden_size // cfg.num_attention_heads
    r = rotary_dims(cfg, hd)
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    cc = model(input_ids=cal.to(DEV), use_cache=True).past_key_values
    Kc = [kv(cc, l)[0].float().cpu().double() for l in range(L)]; Vc = [kv(cc, l)[1].float().cpu().double() for l in range(L)]
    nkv = Kc[0].shape[1]
    del cc; release_memory(DEV)
    log(f"{name}: {dtype}, layers {L}, kv heads {nkv}, head dim {hd}, rotary dims {r}")
    X = wikitext_docs(tok, N_DOCS, CTX + CONT)
    # Part A: premise
    XA = X[:N_PREMISE, :CTX].to(DEV)
    normed = {}
    norm_mod = lambda l: getattr(model.model.layers[l], "input_layernorm") if hasattr(model, "model") else model.gpt_neox.layers[l].input_layernorm
    hooks = [norm_mod(l).register_forward_hook(lambda m_, i_, o_, l=l: normed.__setitem__(l, o_.float().cpu())) for l in range(L)]
    out = model(input_ids=XA, use_cache=True, output_hidden_states=True)
    for h in hooks:
        h.remove()
    H = [h_.float().cpu() for h_ in out.hidden_states]
    ca = out.past_key_values
    rng = np.random.default_rng(0)
    premise = []
    for l in range(1, L):
        Kp, Kl = kv(ca, l - 1)[0].float().cpu().double(), kv(ca, l)[0].float().cpu().double()
        Vp, Vl = kv(ca, l - 1)[1].float().cpu().double(), kv(ca, l)[1].float().cpu().double()
        cosh = float(torch.nn.functional.cosine_similarity(H[l][:, 1:], H[l + 1][:, 1:], dim=-1).mean())
        cosn = float(torch.nn.functional.cosine_similarity(normed[l - 1][:, 1:], normed[l][:, 1:], dim=-1).mean())
        perm, rot = fit_alignment(Kc[l - 1], Kc[l], Vc[l - 1], Vc[l], r, match=False)
        rotate = lambda A, j: torch.stack([A[:, h] @ rot[h][j] for h in range(nkv)], 1)
        sh = torch.tensor(rng.permutation(Kc[l].shape[2] - 1) + 1)
        shuf = lambda A: torch.cat([A[:, :, :1], A[:, :, sh]], 2)
        _, rotn = fit_alignment(Kc[l - 1], shuf(Kc[l]), Vc[l - 1], shuf(Vc[l]), r, match=False)
        rotate_n = lambda A, j: torch.stack([A[:, h] @ rotn[h][j] for h in range(nkv)], 1)
        cstat = {}
        for j, which in ((0, "k"), (1, "v")):
            A0, A1 = (Kp, Kl) if which == "k" else (Vp, Vl)
            c0, c1 = kivi(A0, which)[1], kivi(A1, which)[1]
            ca_ = kivi(rotate(A0, j), which)[1]
            cstat[which] = {"adjacent": code_diff(c0, c1), "independence": code_indep(c0, c1), "aligned": code_diff(ca_, c1)}
        premise.append({"pair": [l - 1, l], "hidden": cosh, "normed_input": cosn, "codes": cstat,
                        "k": tokcos(Kp, Kl), "v": tokcos(Vp, Vl),
                        "k_aligned": tokcos(rotate(Kp, 0), Kl), "v_aligned": tokcos(rotate(Vp, 1), Vl),
                        "k_null": tokcos(rotate_n(Kp, 0), Kl), "v_null": tokcos(rotate_n(Vp, 1), Vl)})
    del ca, out, H, normed; release_memory(DEV)
    log("  premise: median k {:+.3f}, v {:+.3f}, hidden {:+.3f}, v aligned {:+.3f} vs null {:+.3f}".format(
        *[float(np.median([p[k] for p in premise])) for k in ("k", "v", "hidden", "v_aligned", "v_null")]))
    # Part B: merges
    res = {"model": name, "dtype": str(dtype), "layers": L, "kv_heads": nkv, "head_dim": hd, "rotary_dims": r,
           "premise": premise, "merge": {}}
    for start in (L // 2, L // 4):
        pairs = [(l - 1, l) for l in range(start + 1, L, 2)]
        bases = {"trained": {l: ({a: a for a in range(nkv)}, {a: (torch.eye(hd, dtype=torch.float64),) * 2 for a in range(nkv)}) for _, l in pairs},
                 "aligned": {l: fit_alignment(Kc[l - 1], Kc[l], Vc[l - 1], Vc[l], r, match=True) for _, l in pairs},
                 "aligned_nomatch": {l: fit_alignment(Kc[l - 1], Kc[l], Vc[l - 1], Vc[l], r, match=False) for _, l in pairs}}
        for s in range(3):
            g = np.random.default_rng(100 + s)
            bases[f"random{s}"] = {l: random_alignment(nkv, hd, r, g) for _, l in pairs}
        out_s = {"pairs": pairs}
        for arm in ("full",) + ARMS:
            tot, cnt = 0.0, 0
            for i in range(X.shape[0]):
                nll, c = cont_nll(model, X[i:i + 1], [] if arm == "full" else pairs, None if arm == "full" else bases[arm])
                tot += nll; cnt += c
            out_s[arm] = tot / cnt
            log(f"  start {start}: {arm:16s} continuation NLL {tot / cnt:.4f} (ppl {math.exp(tot / cnt):.3f})")
        res["merge"][str(start)] = out_s
    # Part C: XQuant's cross-layer code sharing
    S = L // 4 + (L // 4) % 2
    qpairs = [(l, l + 1) for l in range(S, L - 1, 2)]
    al = {l: fit_alignment(Kc[l], Kc[l + 1], Vc[l], Vc[l + 1], r, match=False)[1] for l, _ in qpairs}
    al = {l: [al[l][h] for h in range(nkv)] for l in al}
    g = np.random.default_rng(300)
    rnd = {l: [random_alignment(nkv, hd, r, g)[1][h] for h in range(nkv)] for l, _ in qpairs}
    outc = {"pairs": qpairs}
    for arm in ("full", "kivi", "trained", "random", "aligned", "kivi_rot"):
        rot = rnd if arm == "random" else al
        tot, cnt = 0.0, 0
        for i in range(X.shape[0]):
            nll, c = xquant_nll(model, X[i:i + 1], qpairs, arm, rot)
            tot += nll; cnt += c
        outc[arm] = tot / cnt
        log(f"  xquant {arm:10s} continuation NLL {tot / cnt:.4f} (ppl {math.exp(tot / cnt):.3f})")
    res["xquant"] = outc
    res["runtime_s"] = round(time.time() - t0, 1)
    del model; release_memory(DEV)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    for name in MODELS:
        if only and name not in only:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        res = run_model(name, log); res["registered"] = REGISTERED; res["dev"] = DEV_RUN
        json.dump(res, open(path, "w"), indent=1)
        log(f"MODEL_DONE {name}")
    print("XLAYER_DONE", flush=True)
