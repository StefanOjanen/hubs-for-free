# Development (exploratory, labelled): quantization-aware value scaling for the fused model. Under per-output-channel
# rounding a row scale of q_proj, k_proj or v_proj is absorbed by that row's quantization scale, so after the
# fusion and rebalancing (fuse_balance.py) only the value-row factors d_r act on rounding, through the columns of
# o_proj that they divide. Here d_r = d_bal_r * a_r^alpha, a_r the root mean square input of the o_proj columns of
# value row r (AWQ's activation-aware scale search, Lin et al. 2024, applied to the fused value/output pair), with
# alpha chosen per layer from a grid to minimize tr(F H F^T), F = Q(W D^-1) D - W, H the o_proj input second
# moment. The function is unchanged before rounding. Compared with fuse+rebalance under RTN and GPTQ at 4 and 3 bits.
#   python alignment_study/qscale_dev.py --model=TinyLlama/TinyLlama_v1.1 [--calib=32] [--calib-len=1024] [--pplwin=8]
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
from quant_balance_dev import gptq

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "TinyLlama/TinyLlama_v1.1")
N_CAL, T_CAL, N_PPL = int(arg("calib", 32)), int(arg("calib-len", 1024)), int(arg("pplwin", 8))
ALPHAS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
OUT = f"alignment_study/balance_dev/qscale_{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


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
    B.balance(kv, groups)
    res = {"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "alphas": ALPHAS}
    snap = lambda: {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
    put = lambda s: [p.copy_(t) for l in range(kv.L) for p, t in zip(kv.params(l), s[l])]

    def hessians():
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

    def quantize(H, bits, method):
        for l in range(kv.L):
            a = kv.attn(l)
            for m in (a.q_proj, a.k_proj, a.v_proj):
                m.weight.copy_(P.rtn(m.weight, bits) if method == "rtn" else gptq(m.weight, H[(l, "in")], bits))
            a.o_proj.weight.copy_(P.rtn(a.o_proj.weight, bits) if method == "rtn" else gptq(a.o_proj.weight, H[(l, "o")], bits))

    def column_sets(l):
        """For every value row r of every fused group: the o_proj columns it feeds (one per query head of the group)."""
        hd, out = kv.hd, []
        for G in groups[l]:
            G = list(G)
            Q = [h for i in G for h in range(i * kv.rep, (i + 1) * kv.rep)]
            out += [(G, r, torch.tensor([h * hd + r for h in Q], device=DEV)) for r in range(hd)]
        return out

    def scale_search(H, bits):
        """Per layer: the alpha minimizing tr(F H F^T); applies d = a^alpha in place, returns the chosen alphas."""
        chosen = []
        for l in range(kv.L):
            a, hd = kv.attn(l), kv.hd
            W, Hl = a.o_proj.weight.float(), H[(l, "o")].to(DEV).float()
            cs = column_sets(l)
            act = torch.stack([Hl.diagonal()[c].mean().clamp_min(1e-12).sqrt() for _, _, c in cs])
            act = act / act.log().mean().exp()
            cols = torch.cat([c for _, _, c in cs])
            rep = torch.tensor([len(c) for _, _, c in cs], device=DEV)

            def cost(alpha):
                d = torch.ones(W.shape[1], device=DEV)
                d[cols] = (act ** alpha).repeat_interleave(rep)
                F = P.rtn(W / d[None, :], bits).float() * d[None, :] - W
                return torch.einsum("ij,jk,ik->", F, Hl, F).item()

            costs = {al: cost(al) for al in ALPHAS}
            best = min(costs, key=costs.get); chosen.append({"alpha": best, "cost": costs[best], "cost_alpha0": costs[0.0]})
            f = act ** best
            for (G, r, c), fr in zip(cs, f):
                for i in G:
                    row = i * hd + r
                    a.v_proj.weight[row] *= fr
                    if a.v_proj.bias is not None:
                        a.v_proj.bias[row] *= fr
                a.o_proj.weight[:, c] /= fr
        return chosen

    res["ppl"] = ppl(); s0 = snap()
    H0 = hessians()
    for bits in (4, 3):
        r = {}
        for method in ("rtn", "gptq"):
            put(s0); quantize(H0, bits, method); r[f"balanced_{method}"] = ppl()
        put(s0)
        r["search"] = scale_search(H0, bits)
        r["ppl_after_scaling"] = ppl()
        s1 = snap(); H1 = hessians()
        for method in ("rtn", "gptq"):
            put(s1); quantize(H1, bits, method); r[f"scaled_{method}"] = ppl()
        del H1
        res[f"w{bits}"] = r
        log(f"{bits} bits: balanced rtn {r['balanced_rtn']:.3f} gptq {r['balanced_gptq']:.3f}; scaled rtn {r['scaled_rtn']:.3f} "
            f"gptq {r['scaled_gptq']:.3f}; unquantized after scaling {r['ppl_after_scaling']:.3f}; alphas {[c['alpha'] for c in r['search']]}")
        json.dump(res, open(OUT, "w"), indent=1)
    log("QSCALE_DEV_DONE")
