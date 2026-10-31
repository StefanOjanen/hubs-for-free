# Development (exploratory, labelled): fitting the shared heads where attention looks. The weighted fusion fits each
# group's shared key and value head in the metric of the attention input's second moment S = E[x x^T], every token
# counted alike. A key or value error at token s matters through the attention that token receives, and the first
# token alone receives a large share. This replaces S by a mixture (1 - lam) S + lam S_att, where S_att weights each
# token by the attention it receives (averaged over the layer's heads and the query positions, scaled so that
# uniform attention gives S), with the pairs fixed at the fusion's matching under S; WikiText-2 perplexity.
#   python alignment_study/attnw_fuse_dev.py --model=JackFram/llama-160m [--calib=32] [--calib-len=1024] [--pplwin=20]
#       [--lams=0,0.25,0.5,1] [--first=1]   (--first=0 leaves the first position out of the attention weighting)
import gc
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "JackFram/llama-160m")
N_CAL, T_CAL, N_PPL = int(arg("calib", 32)), int(arg("calib-len", 1024)), int(arg("pplwin", 20))
LAMS = [float(x) for x in arg("lams", "0,0.25,0.5,1").split(",")]
FIRST = arg("first", "1") == "1"
WEIGHTING = arg("weighting", "attention")          # attention: received attention; position: what uniform attention would give
QSTD = arg("qstd", "0") == "1"                     # 1: query energies from the plain second moment (keys and values attention weighted)
REMATCH = arg("rematch", "0") == "1"               # 1: pairs matched under the mixed second moment
QSTRIDE = int(arg("qstride", 1))                   # received attention estimated from every QSTRIDE-th query position
OUT = (f"alignment_study/attnw_fuse_dev/{NAME.split('/')[-1]}{'' if FIRST else '_nofirst'}{'' if WEIGHTING == 'attention' else '_' + WEIGHTING}"
       f"{'_qstd' if QSTD else ''}{'_rematch' if REMATCH else ''}{'' if QSTRIDE == 1 else f'_qstride{QSTRIDE}'}.json")
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    load = lambda: AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    model = load(); kv = K.KV(model)
    d, L, hd, n = kv.d, kv.L, kv.hd, kv.n
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    S = {l: torch.zeros(d, d, dtype=torch.float64) for l in range(L)}
    Sa = {l: torch.zeros(d, d, dtype=torch.float64) for l in range(L)}
    recv_first = []
    for w in range(cal.shape[0]):
        T = cal.shape[1]
        pos = torch.arange(T, device=DEV)[None]
        mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril()

        def mk(l):
            def pre(mod, inp):
                x = inp[0][0].float()                                          # (T, d)
                a = kv.attn(l)
                cos, sin = model.model.rotary_emb(x[None], position_ids=pos)
                lin = torch.nn.functional.linear                               # not the modules: that would re-enter this hook
                q = lin(x, a.q_proj.weight, a.q_proj.bias).view(1, T, n, hd).transpose(1, 2)
                k = lin(x, a.k_proj.weight, a.k_proj.bias).view(1, T, kv.nkv, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                recv = torch.zeros(T, device=x.device)
                rows = torch.arange(QSTRIDE - 1, T, QSTRIDE, device=x.device)
                for h in range(n):
                    p = torch.softmax(((q[0, h][rows] @ k[0, h // kv.rep].T) / hd ** 0.5).masked_fill(~mask[rows], float("-inf")), -1)
                    recv += p.sum(0)                                           # attention received by each key position
                if WEIGHTING == "position":                                     # uniform causal attention: sum over t >= s of 1/(t+1)
                    recv = torch.flip(torch.cumsum(torch.flip(1.0 / torch.arange(1, T + 1, device=x.device, dtype=torch.float32), [0]), 0), [0])
                wgt = recv / recv.sum() * T                                     # uniform attention gives weight 1
                if l == 0:
                    recv_first.append(float(recv[0] / recv.sum()))
                if not FIRST:
                    wgt[0] = 1.0
                S[l] += F.host64(x.T @ x)
                Sa[l] += F.host64((x * wgt[:, None]).T @ x)
            return pre
        hooks = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l)) for l in range(L)]
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
    ntok = cal.shape[0] * cal.shape[1]
    S = {l: (v / ntok).float() for l, v in S.items()}; Sa = {l: (v / ntok).float() for l, v in Sa.items()}
    log(f"{NAME}: moments from {ntok} tokens; the first position receives {sum(recv_first) / len(recv_first):.3f} of layer 0's attention")
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(L)]
    res = {"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "first_included": FIRST, "weighting": WEIGHTING, "query_energy_plain": QSTD, "rematched": REMATCH, "query_stride": QSTRIDE,
           "first_position_share_layer0": sum(recv_first) / len(recv_first), "base": Z.perplexity(model, win, DEV, bs=2), "arms": {}}
    for lam in LAMS:
        Sm = {l: (1 - lam) * S[l] + lam * Sa[l] for l in range(L)}
        g = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, Sm[l].double())) for l in range(L)] if REMATCH else groups
        for l in range(L):                                   # fuse_kv.convert's body, with the query energies optionally from S
            S64 = Sm[l].double(); roots = F.layer_roots(S64, d)
            Lsq = S[l].double() if QSTD else roots[0] @ roots[0]
            for G in g[l]:
                K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)
        res["arms"][str(lam)] = Z.perplexity(model, win, DEV, bs=2)
        log(f"lambda {lam}: ppl {res['arms'][str(lam)]:.3f}")
        json.dump(res, open(OUT, "w"), indent=1)
        del model, kv; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        model = load(); kv = K.KV(model)
    log(f"unconverted {res['base']:.3f}")
    log("ATTNW_DEV_DONE")
