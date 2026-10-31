# Post hoc diagnostic for round 32 (exploratory, labelled, not preregistered). On Llama-2-7B the sensitivity metrics
# raised the fused perplexity (8 windows 16.00 to 18.37; whole test 19.31 against round 28's 16.78), more than round
# 31's received-attention metric did (16.62). Where does the damage come from? With the round's calibration, pairs and
# instruments: the plain fusion everywhere (a check against 16.000), the sensitivity fit in layers 0 and 1 only, in
# layers 2 and up only, and everywhere (a check against 18.373). For each, WikiText-2 perplexity on 8 windows of 2,048
# tokens and round 29's one-window check: each position's largest absolute residual entry after layers 0 to 5, median
# over positions after the first, as a ratio to the unconverted model's (round 29 froze 2.35 as its threshold).
#   python alignment_study/attnw2_llama2_diag.py
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
import attnw2_common as W
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, REV = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
N_CAL, T_CAL, N_PPL, CHECK, EARLY = 128, 2048, 8, 6, 2
OUT = "alignment_study/attnw2_diag/Llama-2-7b-hf.json"
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


def load():
    return AutoModelForCausalLM.from_pretrained(NAME, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)


def free():
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


def check(model, kv, gw):
    H = {}
    def mk(l):
        def hook(m, i, o):
            H[l] = (o[0] if isinstance(o, tuple) else o)[0]
            if l == CHECK - 1:
                raise Stop
        return hook
    hooks = [kv.blocks[l].register_forward_hook(mk(l)) for l in range(CHECK)]
    try:
        model.model(input_ids=gw, use_cache=False)
    except Stop:
        pass
    for h in hooks:
        h.remove()
    return [float(H[l][1:].abs().amax(1).float().median()) for l in range(CHECK)]


def fuse_plain_layer(kv, l, groups_l, S_l):
    """fuse_kv.convert's 'fuse' branch for one layer."""
    S64 = S_l.double(); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
    for G in groups_l:
        K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    gw = C.windows_c4(tok, N_CAL + 1, T_CAL)[-1:].to(DEV)            # round 29's check window
    model = load(); kv = K.KV(model)
    res = {"model": NAME, "exploratory": True, "post_hoc_for": "PREREGISTRATION32", "ppl_windows": N_PPL, "early_layers": EARLY,
           "base": Z.perplexity(model, win, DEV, bs=2), "base_check": check(model, kv, gw), "variants": {}}
    ratio = lambda c: [a / b for a, b in zip(c, res["base_check"])]

    def record(key, model, kv):
        c = check(model, kv, gw)
        res["variants"][key] = {"ppl": Z.perplexity(model, win, DEV, bs=2), "check": c, "check_ratio": ratio(c), "max_ratio": max(ratio(c))}
        log(f"{key}: ppl {res['variants'][key]['ppl']:.3f}, one-window check largest ratio {res['variants'][key]['max_ratio']:.2f}")
        json.dump(res, open(OUT, "w"), indent=1)
    log(f"unconverted {res['base']:.3f}")
    S, _ = K.second_moments(model, kv, cal, DEV)
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    early = list(range(EARLY)); late = list(range(EARLY, kv.L))
    # plain everywhere, then layers 0 and 1 refitted with the sensitivity metrics (from the original weights)
    Sw0, Sb0, E0 = W.metrics(model, kv, cal, early, DEV)                 # unconverted model, layers 0 and 1 only
    orig_early = {l: [p.detach().cpu().clone() for p in kv.params(l)] for l in early}
    for l in range(kv.L):
        fuse_plain_layer(kv, l, groups[l], S[l])
    record("plain everywhere", model, kv)
    for l in early:
        for p, t in zip(kv.params(l), orig_early[l]):
            p.copy_(t.to(p.device))
        W.fuse_layer(kv, l, groups[l], Sw0[l], E0[l], Sb0[l])
    record("sensitivity in layers 0-1 only", model, kv)
    S = {l: S[l] for l in early}; free()
    del model, kv; free()
    # sensitivity in layers 2 and up, plain in 0 and 1
    model = load(); kv = K.KV(model)
    Sw, Sb, E = W.metrics(model, kv, cal, late, DEV)
    for l in early:
        fuse_plain_layer(kv, l, groups[l], S[l])
    for l in late:
        W.fuse_layer(kv, l, groups[l], Sw[l], E[l], Sb[l])
    record("sensitivity in layers 2 and up only", model, kv)
    del model, kv; free()
    # sensitivity everywhere (round 32's arm)
    model = load(); kv = K.KV(model)
    for l in early:
        W.fuse_layer(kv, l, groups[l], Sw0[l], E0[l], Sb0[l])
    for l in late:
        W.fuse_layer(kv, l, groups[l], Sw.pop(l), E.pop(l), Sb.pop(l))
    record("sensitivity everywhere", model, kv)
    del model, kv; free()
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log("ATTNW2_DIAG_DONE")
