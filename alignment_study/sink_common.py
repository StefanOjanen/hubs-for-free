# Instruments for preregistration 13: what the sink does. A custom attention
# function, identical to the eager implementation of Qwen2, Mistral and Llama
# except in the layers listed in INTERVENE, where the attention weights are
# edited after the softmax, for query rows 1 to T-1 (row 0 attends only to
# itself and is left alone):
#   zero      the first token's weight set to 0, no renormalization: each
#             head loses its sink term a_r0 v_0 W_O
#   mean      the first token's weight replaced by its mean over rows (per
#             head and window): the sink term becomes a constant vector
#   renorm    the first token's weight set to 0 and the row renormalized: the
#             head has to spread the sink's share over the other tokens
#   zero_rand the weight of one random ordinary position (seeded, the same
#             for every head and row of a window, at least 16) set to 0:
#             removing an ordinary token's contribution, for scale
# The outcome is round 11's KL of the next-token distribution.
import sys
import numpy as np
import torch
from torch import nn

sys.path.insert(0, "alignment_study")
import functional_common as F
from transformers import AttentionInterface

INTERVENE = {"layers": set(), "mode": None, "rand_col": None}
MODES = ("zero", "mean", "renorm", "zero_rand")
torch.set_grad_enabled(False)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def sink_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    k = _repeat_kv(key, module.num_key_value_groups); v = _repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    q_len, k_len = query.shape[-2], k.shape[-2]
    if attention_mask is None:            # custom attention receives no mask; apply the causal mask here
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=w.device).tril(diagonal=k_len - q_len)
        w = w.masked_fill(~causal, float("-inf"))
    elif attention_mask.dtype == torch.bool:
        w = w.masked_fill(~attention_mask[:, :, :, :k_len], float("-inf"))
    else:
        w = w + attention_mask[:, :, :, :k_len]
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    if getattr(module, "layer_idx", None) in INTERVENE["layers"]:
        mode = INTERVENE["mode"]; w = w.clone()
        if mode == "zero":
            w[..., 1:, 0] = 0
        elif mode == "mean":
            w[..., 1:, 0] = w[..., 1:, 0].mean(dim=-1, keepdim=True)
        elif mode == "renorm":
            w[..., 1:, 0] = 0
            w[..., 1:, :] = w[..., 1:, :] / w[..., 1:, :].sum(-1, keepdim=True).clamp_min(1e-30)
        elif mode == "zero_rand":
            c = INTERVENE["rand_col"]; w[..., c + 1:, c] = 0
    out = torch.matmul(w, v).transpose(1, 2).contiguous()
    return out, w


AttentionInterface.register("sinkint", sink_attention)


def set_intervention(layers, mode, rand_col=None):
    INTERVENE["layers"] = set(layers); INTERVENE["mode"] = mode; INTERVENE["rand_col"] = rand_col


def run_interventions(model, X, base, layers, all_layers, log=print, seed=0):
    """Per-layer and all-layer KL for every mode. Returns nested dicts."""
    rg = np.random.default_rng(seed)
    T = X.shape[1]
    rand_col = int(rg.integers(F.ROW0, T - 2))
    res = {"rand_col": rand_col, "per_layer": {}, "all_layers": {}}
    for l in layers:
        res["per_layer"][str(l)] = {}
        for mode in MODES:
            set_intervention([l], mode, rand_col)
            kls, dn = F.kl_eval(model, X, base)
            res["per_layer"][str(l)][mode] = {"kl": float(np.mean(kls)), "dnll": float(np.mean(dn))}
        set_intervention([], None)
        log(f"  layer {l}: " + " ".join(f"{m} {res['per_layer'][str(l)][m]['kl']:.4f}" for m in MODES))
    for mode in MODES:
        set_intervention(all_layers, mode, rand_col)
        kls, dn = F.kl_eval(model, X, base)
        res["all_layers"][mode] = {"kl": float(np.mean(kls)), "kl_windows": kls, "dnll": float(np.mean(dn))}
    set_intervention([], None)
    log("  all layers: " + " ".join(f"{m} {res['all_layers'][m]['kl']:.4f}" for m in MODES))
    return res
