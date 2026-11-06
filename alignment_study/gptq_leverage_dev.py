# Development (exploratory, labelled, not preregistered; CPU only, no forward pass): why does GPTQ calibrated without the
# beginning-of-sequence token damage that token (gptq_bos_dev/: Mistral-7B-Instruct per-channel 3-bit GPTQ of q, k, v, o gives
# 6.19 as the windows are and 569,348 with the token, the collapse sitting in layer 0's v_proj alone)? Layer 0's linear inputs
# depend on the token alone, x_t = RMSNorm(E[t]) * g, so the calibration Hessian of layer 0's q, k and v is a count-weighted sum
# over the vocabulary, H = sum_t c_t x_t x_t^T, computable from the embedding matrix and the calibration windows' token counts.
# For every vocabulary token the script reports the leverage x^T (H + d I)^-1 x (d: GPTQ's damping, 1 percent of the mean
# diagonal) relative to the calibration tokens' count-weighted mean, and for layer 0's q, k and v the relative output error
# ||dW x|| / ||W x|| of 3-bit GPTQ (gptq_ref_check.fasterquant, the reference port: per output channel; groups of 128 with
# act-order) against round-to-nearest with the reference quantizer's scales. Calibration: the reference check's windows
# (gqa_common.windows_c4, the first 2,048 tokens of 128 C4 validation documents, token-less) and the token-first form (bos=True).
# Predictions, stated 2026-11-06 before any run:
#   P1 under token-less calibration the start token's relative leverage is above the 99th percentile of the vocabulary on both
#      Mistral models;
#   P2 per channel, GPTQ's v_proj error on the start token is at least 10 times round-to-nearest's on both Mistral models, while
#      GPTQ's count-weighted mean error over the calibration tokens is below round-to-nearest's;
#   P3 on Llama-2-7B (no collapse with the token) the factor in P2 is below 3;
#   P4 token-first calibration lowers the factor in P2 at least tenfold on both Mistral models.
#   python alignment_study/gptq_leverage_dev.py --model=mistralai/Mistral-7B-v0.1 [--revision=SHA] [--nsamples=128]
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
from gptq_ref_check import fasterquant, params
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


PINS = {"mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da", "mistralai/Mistral-7B-Instruct-v0.2": "63a8b081895390a26e140280378bc85ec8bce07a",
        "NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"}
NAME = arg("model", "mistralai/Mistral-7B-v0.1"); REV = arg("revision", PINS.get(NAME)); NS = int(arg("nsamples", 128)); T = int(arg("seqlen", 2048)); BITS = 3
OUT = f"alignment_study/gptq_leverage_dev/{NAME.split('/')[-1]}.json"
CHUNK = 4096
torch.set_grad_enabled(False)


def tensors(names):
    """Read only the named tensors from the cached checkpoint (sharded or single-file safetensors)."""
    get = lambda f: hf_hub_download(NAME, f, revision=REV, local_files_only=True)
    try:
        wmap = json.load(open(get("model.safetensors.index.json")))["weight_map"]
    except Exception:
        try:
            get("model.safetensors"); wmap = {n: "model.safetensors" for n in names}
        except Exception:                                # a checkpoint cached as pytorch_model.bin only (TinyLlama v1.1): weights-only load
            try:
                sd = torch.load(get("pytorch_model.bin"), map_location="cpu", weights_only=True, mmap=True)
            except RuntimeError:
                sd = torch.load(get("pytorch_model.bin"), map_location="cpu", weights_only=True)
            out = {n: sd[n].float() for n in names}; del sd
            return out, json.load(open(get("config.json")))
    out = {}
    for shard in sorted({wmap[n] for n in names}):
        with safe_open(get(shard), "pt") as f:
            for n in names:
                if wmap[n] == shard:
                    out[n] = f.get_tensor(n).float()
    return out, json.load(open(get("config.json")))


def rtn(W, group=-1):
    """Round-to-nearest with the reference quantizer's asymmetric min-max scales, per row or per group of columns."""
    maxq = 2 ** BITS - 1; Q = torch.empty_like(W)
    for c0 in range(0, W.shape[1], W.shape[1] if group == -1 else group):
        w = W[:, c0:c0 + (W.shape[1] if group == -1 else group)]
        scale, zero = params(w, BITS)
        Q[:, c0:c0 + w.shape[1]] = scale[:, None] * (torch.clamp(torch.round(w / scale[:, None]) + zero[:, None], 0, maxq) - zero[:, None])
    return Q


def rel_err(W, dW, X):
    """||dW x|| / ||W x|| for every row x of X, in vocabulary chunks."""
    out = torch.empty(X.shape[0])
    for i in range(0, X.shape[0], CHUNK):
        x = X[i:i + CHUNK].T
        out[i:i + CHUNK] = (dW @ x).norm(dim=0) / (W @ x).norm(dim=0).clamp_min(1e-12)
    return out


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
    names = ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"] + [f"model.layers.0.self_attn.{p}_proj.weight" for p in "qkv"]
    W, cfg = tensors(names)
    E, g = W["model.embed_tokens.weight"], W["model.layers.0.input_layernorm.weight"]; V = E.shape[0]
    X = E / torch.sqrt(E.pow(2).mean(1, keepdim=True) + cfg.get("rms_norm_eps", 1e-6)) * g          # layer 0's linear input for every token
    bos = tok.bos_token_id
    log(f"{NAME}: vocabulary {V}, width {E.shape[1]}, start token {bos} ({tok.convert_ids_to_tokens(bos)})")
    res = {"model": NAME, "revision": REV, "exploratory": True, "bits": BITS, "calibration": [NS, T], "start_token": bos, "forms": {}}
    for form, use_bos in (("tokenless", False), ("token_first", True)):
        ids = C.windows_c4(tok, NS, T, bos=use_bos)
        c = torch.bincount(ids.flatten(), minlength=V).double(); seen = c > 0
        Xs = X[seen].double()
        H = (Xs * c[seen, None]).T @ Xs * (2 / c.sum())
        Hd = H + 0.01 * H.diagonal().mean() * torch.eye(H.shape[0], dtype=H.dtype)
        Lc = torch.linalg.cholesky(Hd)
        lev = torch.empty(V, dtype=torch.float64)
        for i in range(0, V, CHUNK):
            lev[i:i + CHUNK] = torch.linalg.solve_triangular(Lc, X[i:i + CHUNK].double().T, upper=False).pow(2).sum(0)
        lev_rel = lev / ((lev * c).sum() / c.sum())
        pct = lambda v, ref: float((ref < v).double().mean() * 100)
        F = {"distinct_tokens": int(seen.sum()), "start_token_count": int(c[bos]),
             "start_leverage_rel": float(lev_rel[bos]), "start_leverage_percentile_vocab": pct(lev_rel[bos], lev_rel),
             "top_leverage": [[int(i), tok.convert_ids_to_tokens(int(i)), float(lev_rel[i]), int(c[i])] for i in lev_rel.argsort(descending=True)[:25]],
             "projections": {}}
        log(f"{form}: {F['distinct_tokens']} distinct tokens, start token count {F['start_token_count']}, leverage {F['start_leverage_rel']:.3g} times the "
            f"calibration mean, percentile {F['start_leverage_percentile_vocab']:.2f} of the vocabulary")
        for p in "vkq":
            Wp = W[f"model.layers.0.self_attn.{p}_proj.weight"]
            for lay, group, actorder in (("channel", -1, False), ("g128_actorder", 128, True)):
                if p == "q" and lay != "channel":
                    continue
                t1 = time.time()
                dG = fasterquant(Wp, H.float(), BITS, group=group, actorder=actorder).float() - Wp
                dR = rtn(Wp, group) - Wp
                rG, rR = rel_err(Wp, dG, X), rel_err(Wp, dR, X)
                ratio = rG / rR.clamp_min(1e-12)
                wmean = lambda r: float((r.double() * c).sum() / c.sum())
                F["projections"][f"{p}_{lay}"] = {"start_gptq": float(rG[bos]), "start_rtn": float(rR[bos]), "start_factor": float(ratio[bos]),
                                                  "calib_mean_gptq": wmean(rG), "calib_mean_rtn": wmean(rR),
                                                  "start_factor_percentile_vocab": pct(ratio[bos], ratio),
                                                  "top_factor": [[int(i), tok.convert_ids_to_tokens(int(i)), float(ratio[i]), int(c[i])] for i in ratio.argsort(descending=True)[:15]],
                                                  "seconds": round(time.time() - t1, 1)}
                q = F["projections"][f"{p}_{lay}"]
                log(f"{form} {p}_proj {lay}: start token error GPTQ {q['start_gptq']:.4f}, round-to-nearest {q['start_rtn']:.4f} (factor {q['start_factor']:.2f}, "
                    f"percentile {q['start_factor_percentile_vocab']:.2f}); calibration mean {q['calib_mean_gptq']:.4f} against {q['calib_mean_rtn']:.4f} ({q['seconds']}s)")
        res["forms"][form] = F
    tl, tf = res["forms"]["tokenless"], res["forms"]["token_first"]
    res["factors"] = {"v_channel_tokenless": tl["projections"]["v_channel"]["start_factor"], "v_channel_token_first": tf["projections"]["v_channel"]["start_factor"]}
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(res, open(OUT, "w"), indent=1)
    log(f"wrote {OUT}"); print("GPTQ_LEVERAGE_DONE", flush=True)
