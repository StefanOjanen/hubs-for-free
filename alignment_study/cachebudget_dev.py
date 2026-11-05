# Development (exploratory, labelled, not preregistered): at a fixed key-value cache budget, is it better to merge
# key-value heads (the rebalanced fusion, kvfuse.py's recipe, half the heads) and keep more bits per cached value, or
# keep every head and quantize the cache harder? Cache quantization is simulated: per token and per key-value head,
# asymmetric min-max round-to-nearest over the head dimension, applied to the k_proj and v_proj outputs (keys before
# the rotary embedding, a simplification of KIVI-style schemes). Budgets: 16 bits all heads (1.0); half the heads at
# 16 against all heads at 8 (0.5); half at 8 against all at 4 (0.25); half at 4 against all at 2 (0.125). Perplexity
# on WikiText-2 windows of 2,048 tokens as they are and with the beginning-of-sequence token prepended.
#   python alignment_study/cachebudget_dev.py --model=TinyLlama/TinyLlama_v1.1 [--pplwin=8]
import json
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import fuse_balance as B
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME, N_PPL = arg("model", "TinyLlama/TinyLlama_v1.1"), int(arg("pplwin", 8))
# --quant=kivi: keys per channel over groups of 32 tokens, values per token (KIVI's layout, Liu et al. 2024, without
# its full-precision window of recent tokens; keys still before the rotary embedding)
QUANT = arg("quant", "token")
OUT = f"alignment_study/cachebudget_dev/{NAME.split('/')[-1]}{'' if QUANT == 'token' else '_' + QUANT}.json"
torch.set_grad_enabled(False)


def rtn_tokens(y, hd, bits):
    s = y.shape; z = y.reshape(*s[:-1], -1, hd).float()
    lo, hi = z.amin(-1, keepdim=True), z.amax(-1, keepdim=True)
    sc = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    return (torch.round((z - lo) / sc) * sc + lo).reshape(s).to(y.dtype)


def rtn_channels(y, bits, g=32):
    B_, T, D = y.shape; pad = (-T) % g
    z = torch.nn.functional.pad(y.float(), (0, 0, 0, pad), mode="replicate").reshape(B_, -1, g, D)
    lo, hi = z.amin(2, keepdim=True), z.amax(2, keepdim=True)
    sc = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    return (torch.round((z - lo) / sc) * sc + lo).reshape(B_, -1, D)[:, :T].to(y.dtype)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)
    hooks = []

    def set_bits(bits):
        for h in hooks:
            h.remove()
        hooks.clear()
        if bits >= 16:
            return
        for l in range(kv.L):
            a = kv.attn(l)
            kq = (lambda mod, i, o: rtn_channels(o, bits)) if QUANT == "kivi" else (lambda mod, i, o: rtn_tokens(o, kv.hd, bits))
            hooks.append(a.k_proj.register_forward_hook(kq))
            hooks.append(a.v_proj.register_forward_hook(lambda mod, i, o: rtn_tokens(o, kv.hd, bits)))

    def ev(bits):
        set_bits(bits)
        r = {"as_is": Z.perplexity(model, win, DEV, bs=2), "bos_prepended": Z.perplexity(model, win_bos, DEV, bs=2)}
        set_bits(16)
        return r

    res = {"model": NAME, "exploratory": True, "quant": QUANT, "ppl_windows": N_PPL, "kv_heads": kv.nkv if hasattr(kv, "nkv") else None, "full": {}, "fused": {}}
    for b in (16, 8, 4, 3, 2):
        res["full"][b] = ev(b); log(f"all heads, {b} bits: {res['full'][b]}")
        json.dump(res, open(OUT, "w"), indent=1)
    cal = C.windows_c4(tok, 32, 1024)
    S, _ = K.second_moments(model, kv, cal, DEV)
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    K.convert(kv, S, groups, "fuse"); del S
    B.balance(kv, groups)
    for b in (16, 8, 4, 3, 2):
        res["fused"][b] = ev(b); log(f"half heads, {b} bits: {res['fused'][b]}")
        json.dump(res, open(OUT, "w"), indent=1)
    log("CACHEBUDGET_DEV_DONE")
