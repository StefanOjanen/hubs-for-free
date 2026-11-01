# PREREGISTRATION34: does the one-window check choose the early layers' ratio on models it has not seen? Round 29's
# design at a quarter of the key-value heads: every layer from 2 on is fixed at the residual's groups of four, and
# layers 0 and 1 each take one of three settings, unmerged, the residual's pairs, or the residual's groups of four,
# nine members per model. Per member: round 29's one-window check on the converted model (largest ratio over layers
# 0 to 5; threshold 2.35, frozen in round 29) and WikiText-2 perplexity on the first 8 windows of 2,048 tokens.
# Ground truth, in the evaluator, is each member's perplexity relative to the member with both early layers
# unmerged, which isolates the early-layer event from the cost of quartering the other layers. Also recorded: the
# half-heads conversion (pairs everywhere) and the full quartering under neighbouring groups (reported). The model is
# reloaded for every member. Writes alignment_study/quad_check/<model>.json per model as it completes.
#   python alignment_study/quad_check_round.py --registered=<freeze permalink> [--only=<model>] | --dev=<model>
import gc
import itertools
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import device_label
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED, DEV_MODEL = arg("registered"), arg("dev")
if not REGISTERED and not DEV_MODEL:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION34.md>; use --dev=<model>")
MODELS = [DEV_MODEL] if DEV_MODEL else ["NousResearch/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen2.5-7B"]
DEVELOPMENT = {"NousResearch/Llama-2-7b-hf"}             # the model the threshold and the fixed rule came from: reported, not scored
REVISIONS = {"NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d", "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
             "mistralai/Mistral-7B-Instruct-v0.2": "63a8b081895390a26e140280378bc85ec8bce07a", "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796"}
OUT = "alignment_study/quad_check_dev" if DEV_MODEL else "alignment_study/quad_check"
N_CAL, T_CAL = (int(arg("ncal", 16)), 1024) if DEV_MODEL else (128, 2048)
N_PPL, CHECK, TAU, BF16_ABOVE = 8, 6, 2.35, 4e9
SETTINGS = ("none", "pairs", "quads")
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, revision=REVISIONS.get(name), dtype=torch.bfloat16, attn_implementation="sdpa")
    if sum(p.numel() for p in m.parameters()) < BF16_ABOVE:
        m = m.float()
    return m.eval().to(DEV)


def free():
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


def run_model(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS.get(name))
    model = load(name); kv = K.KV(model)
    assert kv.nkv % 4 == 0 and kv.r == kv.hd, f"{name}: {kv.nkv} KV heads, rotary {kv.r} of {kv.hd}"
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    gw = C.windows_c4(tok, N_CAL + 1, T_CAL)[-1:].to(DEV)

    def check():
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
    base_check = check()
    res = {"model": name, "development": name in DEVELOPMENT, "dtype": str(next(model.parameters()).dtype), "layers": kv.L, "kv_heads": kv.nkv,
           "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "check_layers": CHECK, "tau": TAU, "settings": list(SETTINGS),
           "base": Z.perplexity(model, win, DEV, bs=2), "base_check": base_check, "members": []}
    S, ntok = K.second_moments(model, kv, cal, DEV)
    n = kv.nkv
    pairs = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    quads = [kvfuse.quad_groups(kv, l, S[l].double()) if n >= 8 else [list(range(n))] for l in range(kv.L)]
    none = [[i] for i in range(n)]
    neigh4 = [[list(range(s, s + 4)) for s in range(0, n, 4)] for _ in range(kv.L)]
    res.update({"pairs": pairs, "quads": quads})
    log(f"  {name}: unconverted {res['base']:.3f}; moments from {ntok} tokens ({time.time() - t0:.0f}s)")
    del model, kv; free()
    jobs = [("half", "pairs", "pairs", pairs), ("neighbour quarter", "quads", "quads", neigh4)]
    for s0, s1 in itertools.product(SETTINGS, SETTINGS):
        groups = [{"none": none, "pairs": pairs[0], "quads": quads[0]}[s0], {"none": none, "pairs": pairs[1], "quads": quads[1]}[s1]] + quads[2:]
        jobs.append(("member", s0, s1, groups))
    for kind, s0, s1, groups in jobs:
        model = load(name); kv = K.KV(model)
        t1 = time.time(); K.convert(kv, S, groups, "fuse")
        c = check(); ratios = [x / b for x, b in zip(c, base_check)]
        m = {"kind": kind, "layer0": s0, "layer1": s1, "check": c, "check_ratios": ratios, "max_ratio": max(ratios), "flag": max(ratios) > TAU,
             "ppl": Z.perplexity(model, win, DEV, bs=2), "seconds": round(time.time() - t1, 1)}
        res["members"].append(m)
        log(f"  {name}, {kind}, layer 0 {s0}, layer 1 {s1}: ppl {m['ppl']:.3f}, check {m['max_ratio']:.2f}" + (" FLAG" if m["flag"] else ""))
        del model, kv; free()
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
        free()
    log("QUAD_CHECK_ROUND_DONE")
