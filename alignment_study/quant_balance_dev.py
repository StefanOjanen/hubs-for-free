# Development (exploratory, labelled): does rebalancing the fused heads (fuse_balance.py) still help under a
# stronger weight quantizer and for the key-value cache itself? For a model converted to half its key-value heads by
# the weighted fusion, unbalanced and rebalanced: WikiText-2 perplexity with (a) the attention projections quantized
# to 4 and 3 bits per output channel by GPTQ (Frantar et al. 2023; calibration Hessians of each projection's input,
# damping 1 percent of the mean diagonal, columns in order, blocks of 128) against round to nearest; (b) the key and
# value cache quantized to 4 and 2 bits per token and head (asymmetric, before the rotary embedding), weights intact.
#   python alignment_study/quant_balance_dev.py --model=TinyLlama/TinyLlama_v1.1 [--calib=32] [--calib-len=1024] [--pplwin=8]
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


NAME = arg("model", "TinyLlama/TinyLlama_v1.1")
N_CAL, T_CAL, N_PPL = int(arg("calib", 32)), int(arg("calib-len", 1024)), int(arg("pplwin", 8))
OUT = f"alignment_study/balance_dev/quant_{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def gptq(W, H, bits, block=128, damp=0.01):
    """GPTQ with one asymmetric min-max scale per output row, computed from the original weights."""
    out_dev, out_dtype = W.device, W.dtype
    W = W.detach().cpu().float().clone(); n = W.shape[1]       # the column loop runs on the host: many small steps
    H = H.detach().cpu().double().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += damp * H.diagonal().mean() * torch.eye(n, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    lo, hi = W.amin(1, keepdim=True), W.amax(1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8); zero = torch.round(-lo / scale)
    q = lambda w: (torch.clamp(torch.round(w / scale[:, 0]) + zero[:, 0], 0, 2 ** bits - 1) - zero[:, 0]) * scale[:, 0]
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            w = W1[:, i]; qi = q(w)
            Q[:, i1 + i] = qi
            e = (w - qi) / Hi1[i, i]
            W1[:, i:] -= e[:, None] * Hi1[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q.to(out_dtype).to(out_dev)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    S, _ = K.second_moments(model, kv, cal, DEV)
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    K.convert(kv, S, groups, "fuse"); del S
    res = {"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL}
    snap = lambda: {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
    put = lambda s: [p.copy_(t) for l in range(kv.L) for p, t in zip(kv.params(l), s[l])]

    def hessians():
        """E[x x^T] of the attention input (q, k, v) and of the output projection's input, on the calibration text."""
        H = {}
        def mk(key):
            def pre(m, i):
                x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
                H[key] = H.get(key, 0) + x.T @ x
            return pre
        hooks = []
        for l in range(kv.L):
            a = kv.attn(l)
            hooks += [a.q_proj.register_forward_pre_hook(mk((l, "in"))), a.o_proj.register_forward_pre_hook(mk((l, "o")))]
        for w in range(cal.shape[0]):
            model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        return H

    def kv_quant(bits):
        def hook(m, i, o):
            x = o.view(*o.shape[:-1], -1, kv.hd)
            lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
            sc = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8); z = torch.round(-lo / sc)
            return ((torch.clamp(torch.round(x / sc) + z, 0, 2 ** bits - 1) - z) * sc).view_as(o)
        return [m.register_forward_hook(hook) for l in range(kv.L) for m in (kv.attn(l).k_proj, kv.attn(l).v_proj)]

    for key in ("fused", "balanced"):
        if key == "balanced":
            B.balance(kv, groups)
        r = {"ppl": ppl()}
        s = snap()
        H = hessians()
        for bits in (4, 3):
            for method in ("rtn", "gptq"):
                put(s)
                for l in range(kv.L):
                    a = kv.attn(l)
                    for m in (a.q_proj, a.k_proj, a.v_proj):
                        m.weight.copy_(P.rtn(m.weight, bits) if method == "rtn" else gptq(m.weight, H[(l, "in")], bits))
                    a.o_proj.weight.copy_(P.rtn(a.o_proj.weight, bits) if method == "rtn" else gptq(a.o_proj.weight, H[(l, "o")], bits))
                r[f"{method}_w{bits}"] = ppl()
        put(s)
        for bits in (4, 2):
            hs = kv_quant(bits); r[f"kv{bits}"] = ppl()
            for h in hs:
                h.remove()
        res[key] = r
        log(f"{key}: ppl {r['ppl']:.3f}; 4 bits rtn {r['rtn_w4']:.3f} gptq {r['gptq_w4']:.3f}; 3 bits rtn {r['rtn_w3']:.3f} gptq {r['gptq_w3']:.3f}; "
            f"KV cache 4 bits {r['kv4']:.3f}, 2 bits {r['kv2']:.3f}")
        del H
        json.dump(res, open(OUT, "w"), indent=1)
    log("QUANT_BALANCE_DEV_DONE")
