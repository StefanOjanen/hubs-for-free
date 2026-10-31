# PREREGISTRATION31: fitting the shared heads where attention looks. The weighted fusion (fuse_kv.py) fits each
# group's shared key and value head in the metric of the attention input's second moment S = E[x x^T], every token
# counted alike. This round replaces S by S_att, which weights each token by the attention it receives (averaged over
# the layer's heads and, every 8th position, over the query positions; scaled so that uniform attention would give
# S), the full weighting chosen by the rule committed before the last development value (lambda = 1,
# PREREGISTRATION31_lambda_rule.md), with the pairs fixed at the fusion's own matching under S (neighbouring pairs
# where the pair costs do not apply). Each model is converted to half its key-value heads twice, with S and with
# S_att; WikiText-2 perplexity on the first 8 windows of 2,048 tokens, and for Llama-2-7B on the whole test set at
# 2,048 tokens as in round 28.
#   python alignment_study/attnw_round.py --registered=<freeze permalink> [--only=<model>] | --dev=<model>
import gc
import json
import math
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
from hubsfree.adapters import device_label
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED, DEV_MODEL = arg("registered"), arg("dev")
if not REGISTERED and not DEV_MODEL:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION31.md>; use --dev=<model>")
MODELS = [DEV_MODEL] if DEV_MODEL else ["allenai/OLMo-1B-hf", "h2oai/h2o-danube-1.8b-base", "mistralai/Mistral-7B-Instruct-v0.2",
                                       "NousResearch/Llama-2-7b-hf", "Qwen/Qwen2.5-7B"]
REVISIONS = {"NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"}
FULL_TEST = {"NousResearch/Llama-2-7b-hf"}
OUT = "alignment_study/attnw_dev" if DEV_MODEL else "alignment_study/attnw"
BF16_ABOVE, N_CAL, T_CAL, N_PPL, QSTRIDE, LAM = 4e9, 128, 2048, 8, 8, 1.0
torch.set_grad_enabled(False)


def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, revision=REVISIONS.get(name), dtype=torch.bfloat16, attn_implementation="sdpa")
    n = sum(p.numel() for p in m.parameters())
    if n < BF16_ABOVE:
        m = m.float()
    return m.eval().to(DEV), n


def moments(model, kv, cal):
    """S and S_att of every layer's attention input (float32, host), and per layer the received attention's
    normalized entropy and the first position's share (averaged over windows)."""
    d, L, hd, n = kv.d, kv.L, kv.hd, kv.n
    S = {l: torch.zeros(d, d) for l in range(L)}                                # float32 on the host, as fuse_kv.second_moments
    Sa = {l: torch.zeros(d, d) for l in range(L)}
    conc = {l: {"entropy": 0.0, "first": 0.0} for l in range(L)}
    lin = torch.nn.functional.linear
    for w in range(cal.shape[0]):
        T = cal.shape[1]
        pos = torch.arange(T, device=DEV)[None]
        rows = torch.arange(QSTRIDE - 1, T, QSTRIDE, device=DEV)
        mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril()[rows]

        def mk(l):
            def pre(mod, inp):
                x = inp[0][0].float()
                a = kv.attn(l)
                cos, sin = model.model.rotary_emb(x[None], position_ids=pos)
                fb = lambda m: m.bias.float() if m.bias is not None else None
                q = lin(x, a.q_proj.weight.float(), fb(a.q_proj)).view(1, T, n, hd).transpose(1, 2)
                k = lin(x, a.k_proj.weight.float(), fb(a.k_proj)).view(1, T, kv.nkv, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                recv = torch.zeros(T, device=x.device)
                for h in range(n):
                    p = torch.softmax(((q[0, h][rows] @ k[0, h // kv.rep].T) / hd ** 0.5).masked_fill(~mask, float("-inf")), -1)
                    recv += p.sum(0)
                share = recv / recv.sum()
                conc[l]["entropy"] += float(-(share * share.clamp_min(1e-30).log()).sum() / math.log(T))
                conc[l]["first"] += float(share[0])
                wgt = share * T
                S[l] += (x.T @ x).cpu()
                Sa[l] += ((x * wgt[:, None]).T @ x).cpu()
            return pre
        hooks = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l)) for l in range(L)]
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
    ntok = cal.shape[0] * cal.shape[1]
    S = {l: v / ntok for l, v in S.items()}
    Sa = {l: v / ntok for l, v in Sa.items()}
    conc = {l: {k_: v / cal.shape[0] for k_, v in c.items()} for l, c in conc.items()}
    return S, Sa, conc


def run_model(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS.get(name))
    model, nparam = load(name)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)
    w8 = win[:N_PPL]
    res = {"model": name, "params": nparam, "dtype": str(next(model.parameters()).dtype), "lambda": LAM, "query_stride": QSTRIDE,
           "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "base": Z.perplexity(model, w8, DEV, bs=2)}
    S, Sa, conc = moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL))
    res["concentration"] = conc
    log(f"  {name}: unconverted {res['base']:.3f}; moments ({time.time() - t0:.0f}s); first-position share, layer mean "
        f"{sum(c['first'] for c in conc.values()) / kv.L:.3f}; received-attention entropy {sum(c['entropy'] for c in conc.values()) / kv.L:.3f}")
    if kv.has_bias(0) or kv.r != kv.hd:
        groups, kind = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)], "neighbour"
    else:
        groups, kind = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)], "fusion-residual matching"
    res["pairing"], res["groups"] = kind, groups
    mixed = Sa if LAM == 1.0 else {l: (1 - LAM) * S[l] + LAM * Sa[l] for l in range(kv.L)}
    for arm, M in (("plain", S), ("attention", mixed)):
        t = time.time()
        K.convert(kv, M, groups, "fuse")
        res[arm] = {"ppl": Z.perplexity(model, w8, DEV, bs=2)}
        if name in FULL_TEST:
            res[arm]["ppl_full_2048"] = Z.perplexity(model, win, DEV, bs=2)
        log(f"  {name}, {arm}: ppl {res[arm]['ppl']:.3f}" + (f", whole test set {res[arm]['ppl_full_2048']:.3f}" if name in FULL_TEST else "")
            + f" ({time.time() - t:.0f}s)")
        del model, kv; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        if arm == "plain":
            model, _ = load(name); kv = K.KV(model)
    del S, Sa, mixed; gc.collect()
    res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = arg("only")
    for name in MODELS:
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if (only and name != only) or os.path.exists(path):
            continue
        res = run_model(name, log)
        res.update({"registered": REGISTERED, "dev": bool(DEV_MODEL)})
        json.dump(res, open(path, "w"), indent=1)
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
    log("ATTNW_ROUND_DONE")
