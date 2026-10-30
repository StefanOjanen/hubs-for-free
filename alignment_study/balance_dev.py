# Development check for fuse_balance.py (exploratory, labelled). For a model converted to half its key-value heads
# by the weighted fusion on matched pairs: (1) rebalancing leaves the logits unchanged (float32); (2) the spread of
# the attention projections' row and column norms before and after; (3) WikiText-2 perplexity after rounding the
# attention projections to bfloat16, and to 4 and 3 bits per output channel (round to nearest, asymmetric), with
# and without rebalancing.
#   python alignment_study/balance_dev.py --model=JackFram/llama-160m [--calib=32] [--calib-len=1024] [--pplwin=40]
import json
import os
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

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "JackFram/llama-160m")
N_CAL, T_CAL, N_PPL = int(arg("calib", 32)), int(arg("calib-len", 1024)), int(arg("pplwin", 40))
OUT = f"alignment_study/balance_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
    res = {"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "base": ppl()}
    S, _ = K.second_moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL), DEV)
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    K.convert(kv, S, groups, "fuse"); del S
    x = win[:1, :512].to(DEV)
    logits = lambda: model(input_ids=x).logits.float()

    def spread():
        out = {}
        for nm in ("q_proj", "k_proj", "v_proj"):
            r = torch.cat([getattr(kv.attn(l), nm).weight.float().norm(dim=1) for l in range(kv.L)])
            out[nm + "_rows"] = [float(r.min()), float(r.median()), float(r.max())]
        c = torch.cat([kv.attn(l).o_proj.weight.float().norm(dim=0) for l in range(kv.L)])
        out["o_proj_cols"] = [float(c.min()), float(c.median()), float(c.max())]
        return out
    snap = lambda: {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
    put = lambda s: [p.copy_(t) for l in range(kv.L) for p, t in zip(kv.params(l), s[l])]
    L0 = logits(); fused = snap()
    res["fuse"] = {"ppl": ppl(), "spread": spread()}
    B.balance(kv, groups)
    L1 = logits(); balanced = snap()
    res["balanced"] = {"ppl": ppl(), "spread": spread(), "max_logit_change_relative": float((L1 - L0).abs().max() / L0.abs().max())}
    log(f"fused ppl {res['fuse']['ppl']:.3f}, balanced {res['balanced']['ppl']:.3f} (largest logit change {res['balanced']['max_logit_change_relative']:.1e} of the largest logit)")
    for k in ("fuse", "balanced"):
        log(f"  {k}: " + ", ".join(f"{n} {v[0]:.3g}/{v[1]:.3g}/{v[2]:.3g}" for n, v in res[k]["spread"].items()))

    def rounded(kind):
        for l in range(kv.L):
            a = kv.attn(l)
            for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                m.weight.copy_(m.weight.to(torch.bfloat16).float() if kind == "bf16" else P.rtn(m.weight, int(kind[1:])))
        return ppl()
    for k, s in (("fuse", fused), ("balanced", balanced)):
        for kind in ("bf16", "w8", "w4", "w3"):
            put(s); res[k][kind] = rounded(kind)
        log(f"  {k}: attention weights rounded: bf16 {res[k]['bf16']:.3f}, 8 bits {res[k]['w8']:.3f}, 4 bits {res[k]['w4']:.3f}, 3 bits {res[k]['w3']:.3f}")
    json.dump(res, open(OUT, "w"), indent=1)
    log("BALANCE_DEV_DONE")
