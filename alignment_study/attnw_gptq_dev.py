# Development (exploratory, labelled): do the attention-derived metrics help quantization? GPTQ (Frantar et al. 2023)
# rounds each projection's rows against one input Hessian, E[x x^T] of the attention input for q_proj, k_proj and
# v_proj alike. With the attention pattern held fixed, the value error is exactly the S_bar-weighted one (BoA's value
# Hessian, Kim et al. ICML 2025), and to first order a key error at token s is weighted by w_s = sum_t p_ts
# |z_s - z_bar_t|^2 and a query error at t by u_t = sum_s p_ts |z_s - z_bar_t|^2; BoA's relaxed key and query Hessians
# drop these weights, which leaves plain GPTQ's for q_proj and k_proj. Arms, input Hessians of q / k / v (o_proj always
# E[x x^T] of its own input): plain S/S/S; boa S/S/S_bar; sens S_u/S_w/S_bar; sensqk S_u/S_w/S. The unconverted model's
# attention projections are quantized per output row at the given bits (GPTQ as in quant_balance_dev.py, Hessians
# from the unquantized model); WikiText-2 perplexity.
#   python alignment_study/attnw_gptq_dev.py --model=TinyLlama/TinyLlama_v1.1 [--calib=32] [--calib-len=1024] [--pplwin=8]
#       [--bits=3,2] [--arms=plain,boa,sens,sensqk]
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "TinyLlama/TinyLlama_v1.1")
N_CAL, T_CAL, N_PPL = int(arg("calib", 32)), int(arg("calib-len", 1024)), int(arg("pplwin", 8))
BITS = [int(b) for b in arg("bits", "3,2").split(",")]
ARMS = {"plain": ("S", "S", "S"), "boa": ("S", "S", "Sbar"), "sens": ("Su", "Sw", "Sbar"), "sensqk": ("Su", "Sw", "S")}
ARMS = {a: ARMS[a] for a in arg("arms", ",".join(ARMS)).split(",")}
OUT = f"alignment_study/attnw_gptq_dev/{NAME.split('/')[-1]}.json"
sys.argv = [sys.argv[0], f"--model={NAME}"]                    # attnw_mix_dev reads its own arguments at import
import attnw_mix_dev as M
from quant_balance_dev import gptq
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    Mx = M.moments(model, kv, cal)
    Ho = {}
    def mk(l):
        def pre(m, i):
            x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
            Ho[l] = Ho.get(l, 0) + (x.T @ x).cpu()
        return pre
    hooks = [kv.attn(l).o_proj.register_forward_pre_hook(mk(l)) for l in range(kv.L)]
    for w in range(cal.shape[0]):
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
    for h in hooks:
        h.remove()
    orig = {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.update({"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "query_stride": M.QSTRIDE,
                "base": Z.perplexity(model, win, DEV, bs=2)})
    res.setdefault("arms", {})
    for bits in BITS:
        for arm, (hq, hk, hv) in ARMS.items():
            for l in range(kv.L):
                a = kv.attn(l)
                for m, key in ((a.q_proj, hq), (a.k_proj, hk), (a.v_proj, hv)):
                    m.weight.copy_(gptq(m.weight, Mx[key][l], bits))
                a.o_proj.weight.copy_(gptq(a.o_proj.weight, Ho[l], bits))
            res["arms"][f"{arm}_w{bits}"] = {"hessians_qkv": [hq, hk, hv], "ppl": Z.perplexity(model, win, DEV, bs=2)}
            log(f"{bits} bits, {arm} ({hq}/{hk}/{hv}): ppl {res['arms'][f'{arm}_w{bits}']['ppl']:.3f}")
            json.dump(res, open(OUT, "w"), indent=1)
            for l in range(kv.L):
                for p, t in zip(kv.params(l), orig[l]):
                    p.copy_(t)
    log(f"unconverted {res['base']:.3f}")
    log("ATTNW_GPTQ_DEV_DONE")
