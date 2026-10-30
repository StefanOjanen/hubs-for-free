# Mechanism check for the exploratory follow-up to round 28 (not preregistered). The bisection
# (llama2_followup_bisect.py) found that on Llama-2-7B the weighted fusion on pairs matched by CAGA's residual
# stays harmless in layer 0 alone and becomes catastrophic as soon as layer 1 is converted as well (perplexity in
# the thousands), while each layer alone is harmless (llama2_followup_diag.py). Llama-2-7B writes very large
# activations into a few residual dimensions in its second layer (Sun et al. 2024, "Massive activations in large
# language models"), at the first position and at some delimiter tokens; later layers use them as attention sinks.
# This records, for layers 0 and 1 converted on either pairing and in mixed combinations, the largest activation of
# every position after the first layers (where it sits and on which tokens), and finds which of layer 1's pairs
# (with layer 0 converted) and of layer 0's pairs (with layer 1 converted) set off the damage, one pair at a time.
#   python alignment_study/llama2_followup_mechanism.py
import json
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_followup_mechanism.json"
N_WIN, LAYERS_SEEN, BIG = 4, (0, 1, 2, 3, 7, 15, 31), 100.0
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pairings = {"fuse": json.load(open("alignment_study/llama2/Llama-2-7b-hf.json"))["pairing"]["paired"],
                "caga": json.load(open("alignment_study/llama2_followup.json"))["groups"]}
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    body, _ = Z._parts(model)

    # second moments of layers 0 and 1 only (the calibration of round 28; the forward pass stops after layer 1)
    S, n = {0: None, 1: None}, 0
    def mk(l):
        def pre(mod, inp):
            x = inp[0].detach().reshape(-1, kv.d).float()
            m = (x.T @ x).float().cpu()
            S[l] = m if S[l] is None else S[l] + m
        return pre
    def stop(mod, i, o):
        raise Stop
    hs = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l)) for l in (0, 1)] + [kv.blocks[1].register_forward_hook(stop)]
    cal = C.windows_c4(tok, 128, 2048)
    for w in range(cal.shape[0]):
        try:
            model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        except Stop:
            pass
        n += cal.shape[1]
    for h in hs:
        h.remove()
    S = {l: S[l] / n for l in S}
    log(f"second moments of layers 0 and 1 from {n} tokens")
    snap = {l: [p.detach().cpu().clone() for p in kv.params(l)] for l in (0, 1)}

    def restore(l):
        for p, s_ in zip(kv.params(l), snap[l]):
            p.copy_(s_.to(p.device))
    fit = {}
    def fuse(l, groups):
        if l not in fit:
            S64 = S[l].double(); roots = F.layer_roots(S64, kv.d); fit[l] = (S64, roots, roots[0] @ roots[0])
        S64, roots, Lsq = fit[l]
        for G in groups:
            K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)

    win = Z.wikitext2_windows(tok, T=2048)[:N_WIN]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
    x = win[:1]
    toks = tok.convert_ids_to_tokens(x[0].tolist())

    def activations():
        H = {}
        def mk2(l):
            def hook(m, i, o):
                H[l] = (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
            return hook
        hooks = [kv.blocks[l].register_forward_hook(mk2(l)) for l in LAYERS_SEEN]
        body(input_ids=x.to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        out = {}
        for l in LAYERS_SEEN:
            mx, arg = H[l].abs().max(1)
            top = mx.topk(8)
            out[l] = {"first_position_max": float(mx[0]), "first_position_dim": int(arg[0]),
                      "positions_above_100": int((mx > BIG).sum()), "median_position_max": float(mx.median()),
                      "top": [{"pos": int(p), "token": toks[int(p)], "max": float(v), "dim": int(arg[int(p)])} for v, p in zip(top.values, top.indices)]}
        return out

    res = {"model": NAME, "revision": REVISION, "exploratory": True, "windows": N_WIN, "configs": {}}
    configs = {"unconverted": {}, "fuse_0_1": {0: "fuse", 1: "fuse"}, "caga_0_1": {0: "caga", 1: "caga"},
               "caga_0_fuse_1": {0: "caga", 1: "fuse"}, "fuse_0_caga_1": {0: "fuse", 1: "caga"},
               "caga_0": {0: "caga"}, "caga_1": {1: "caga"}}
    for name, cfg in configs.items():
        for l, p in cfg.items():
            fuse(l, pairings[p][l])
        res["configs"][name] = {"ppl": ppl(), "activations": activations()}
        a1 = res["configs"][name]["activations"][1]
        log(f"{name}: ppl {res['configs'][name]['ppl']:.3f}; after layer 1: first position max {a1['first_position_max']:.0f} (dim {a1['first_position_dim']}), "
            f"{a1['positions_above_100']} positions above {BIG:.0f}, top {[(t['pos'], t['token'], round(t['max'])) for t in a1['top'][:4]]}")
        for l in cfg:
            restore(l)
        json.dump(res, open(OUT, "w"), indent=1)

    # one pair at a time: layer 1's CAGA pairs with layer 0 converted on CAGA's pairs, then the reverse
    for l, other in ((1, 0), (0, 1)):
        fuse(other, pairings["caga"][other])
        rows = []
        for G in pairings["caga"][l]:
            shared = sorted(G) in [sorted(H) for H in pairings["fuse"][l]]
            fuse(l, [G])
            r = {"pair": list(G), "in_round28_pairs": shared, "ppl": ppl()}
            a = activations()[1]
            r.update({"first_position_max_after_layer1": a["first_position_max"], "positions_above_100_after_layer1": a["positions_above_100"]})
            rows.append(r); restore(l)
            log(f"layer {l} pair {G} alone (layer {other} on CAGA's pairs): ppl {r['ppl']:.3f}, after layer 1 first position max "
                f"{r['first_position_max_after_layer1']:.0f}, {r['positions_above_100_after_layer1']} positions above {BIG:.0f}")
            res[f"single_pairs_layer{l}_with_layer{other}_caga"] = rows
            json.dump(res, open(OUT, "w"), indent=1)
        restore(other)
    log("MECHANISM_DONE")
