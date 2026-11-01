# Development (exploratory, labelled): a per-layer ratio for Llama-2-7B chosen by round 29's one-window check. At a
# quarter of the key-value heads every grouping of layer 1 tried collapses the model, and keeping layer 1 at its
# matched pairs while quartering layer 0 still reads 360 on the check (quad_repair_dev.py), as round 28's follow-up
# found that the pair fires once layer 0 is converted. This script keeps the first k layers at the matched pairs
# (round 28's groups, safe at half the heads) and quarters the rest, for k = 1 to CHECK, measuring the check with
# layers 0 to CHECK-1 converted; the smallest k under the frozen threshold is evaluated in full (whole-test
# perplexity at 2,048 tokens, ZS-7 on 500 examples, the check) and its cache fraction reported.
#   python alignment_study/quad_fallback_dev.py
import gc
import json
import os
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
sys.argv = [argv0[0], "--model=NousResearch/Llama-2-7b-hf"]
import quad_dev as Q
sys.argv = argv0
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, REV = "NousResearch/Llama-2-7b-hf", Q.REVISIONS["NousResearch/Llama-2-7b-hf"]
CHECK, TAU, SUBSET = 6, 2.35, 500
SRC = "alignment_study/quad_dev/Llama-2-7b-hf.json"
OUT = "alignment_study/quad_dev/Llama-2-7b-hf_fallback.json"
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


def load():
    return AutoModelForCausalLM.from_pretrained(NAME, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)


def fuse_plain_layer(kv, l, groups_l, S_l):
    S64 = S_l.double(); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
    for G in groups_l:
        K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    srcj = json.load(open(SRC)); quads, pairs = srcj["quads"], srcj["pairs"]
    tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
    cal = C.windows_c4(tok, Q.N_CAL, Q.T_CAL)
    gw = C.windows_c4(tok, Q.N_CAL + 1, Q.T_CAL)[-1:].to(DEV)
    model = load(); kv = K.KV(model)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    log(f"{NAME}: moments from {ntok} tokens")

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
    base = check()
    ratios = lambda c: [x / b for x, b in zip(c, base)]
    orig = {l: [p.detach().cpu().clone() for p in kv.params(l)] for l in range(CHECK)}

    def put_all():
        for l in range(CHECK):
            for p, t in zip(kv.params(l), orig[l]):
                p.copy_(t.to(p.device))
    res = {"model": NAME, "exploratory": True, "source": SRC, "tau": TAU, "check_layers": CHECK, "base_check": base, "sweep": []}
    chosen = None
    for k in range(0, CHECK + 1):                                   # k = 0 is the full quartering (reproduces quad_dev's check)
        put_all()
        for l in range(CHECK):
            fuse_plain_layer(kv, l, pairs[l] if l < k else quads[l], S[l])
        r = ratios(check()); res["sweep"].append({"layers_at_pairs": k, "ratios": r, "max_ratio": max(r)})
        log(f"first {k} layers at pairs, layers {k}-{CHECK - 1} at quads: check ratios {[round(x, 2) for x in r]}")
        if chosen is None and k >= 1 and max(r) <= TAU:
            chosen = k
    res["chosen_layers_at_pairs"] = chosen
    json.dump(res, open(OUT, "w"), indent=1)
    if chosen is None:
        log(f"no k up to {CHECK} passes the check; stopping")
        sys.exit(0)
    kvh = kv.nkv
    groups = [pairs[l] if l < chosen else quads[l] for l in range(kv.L)]
    res["cache_fraction"] = (chosen * (kvh // 2) + (kv.L - chosen) * (kvh // 4)) / (kv.L * kvh)
    log(f"evaluating the first {chosen} layers at pairs and the rest at quads: cache fraction {res['cache_fraction']:.4f}")
    del model, kv, orig; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    model = load(); kv = K.KV(model)
    t1 = time.time(); K.convert(kv, S, groups, "fuse"); log(f"converted in {time.time() - t1:.0f}s")
    r = ratios(check())
    tasks = {**Z.load_tasks(subset=SUBSET), **Q.R.extra_tasks(SUBSET)}; tasks = {k: tasks[k] for k in Q.R.ZS7}
    reqs = Z.build_requests(tok, tasks)
    ev = Z.evaluate_tasks(model, tok, tasks, reqs, DEV)
    ev["zs7"] = float(np.mean([ev[t][m] for t, m in Q.R.ZS7.items()]))
    ev["ppl_2048"] = Z.perplexity(model, Z.wikitext2_windows(tok, T=2048), DEV, bs=2)
    ev["check"] = r; ev["check_max_ratio"] = max(r)
    res["evals"] = {f"fuse/pairs_first_{chosen}_layers_quads_rest": ev}; res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"mixed ratio: ppl {ev['ppl_2048']:.3f}; ZS-7 {ev['zs7']:.4f}; one-window check {max(r):.2f}")
    log("QUAD_FALLBACK_DONE")
