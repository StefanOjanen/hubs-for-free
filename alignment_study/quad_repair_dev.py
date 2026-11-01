# Development (exploratory, labelled): repairing the quarter-heads conversion of Llama-2-7B. quad_dev.py's matched
# groups of four collapse the model (WikiText-2 perplexity 14,918, ZS-7 0.405) and round 29's one-window check reads
# 576 at layer 1 (the layer whose super-weight neuron one merged pair switched on in round 28's follow-up), although
# heads 8 and 18 are in different groups. This script (1) reproduces the check with layers 0 to 5 converted, (2) finds
# the group of layer 1 whose restoration to unmerged heads lowers the check most, and whether that group alone
# raises it, (3) repairs by round 29's procedure at the four-head level: the group is forbidden in the two-level
# matching, the layer is re-matched and re-fused, up to REPAIR_STEPS times until the check is under round 29's
# frozen threshold, and (4) evaluates the repaired conversion as quad_dev does (whole-test perplexity at 2,048
# tokens, ZS-7 on 500 examples per task, the check). The model is reloaded for the full conversion.
#   python alignment_study/quad_repair_dev.py
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
import quad_dev as Q                                   # REVISIONS, the ZS-7 task loader (Q.R) and the calibration constants
sys.argv = argv0
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, REV = "NousResearch/Llama-2-7b-hf", Q.REVISIONS["NousResearch/Llama-2-7b-hf"]
CHECK, REPAIR_STEPS, TAU, SUBSET = 6, 3, 2.35, 500
SRC = "alignment_study/quad_dev/Llama-2-7b-hf.json"
OUT = "alignment_study/quad_dev/Llama-2-7b-hf_repair.json"
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


def load():
    return AutoModelForCausalLM.from_pretrained(NAME, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)


def fuse_plain_layer(kv, l, groups_l, S_l):
    S64 = S_l.double(); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
    for G in groups_l:
        K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)


def quads_avoiding(kv, l, S64, forbidden):
    """kvfuse.quad_groups with the forbidden four-head groups priced out of the second level."""
    st = kvfuse.layer_stats(kv, l, S64)
    pairs = kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S64))
    cost = torch.zeros(len(pairs), len(pairs), dtype=torch.float64)
    for p in range(len(pairs)):
        for q in range(p + 1, len(pairs)):
            G = sorted(pairs[p] + pairs[q])
            cost[p, q] = cost[q, p] = 1e9 if tuple(G) in forbidden else kvfuse.group_cost(st, G)
    return sorted(sorted(pairs[p] + pairs[q]) for p, q in kvfuse.min_cost_pairs(cost))


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    src = SRC if os.path.exists(SRC) else SRC + ".partial"
    srcj = json.load(open(src)); quads, pairs_all = srcj["quads"], srcj["pairs"]
    tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
    cal = C.windows_c4(tok, Q.N_CAL, Q.T_CAL)
    gw = C.windows_c4(tok, Q.N_CAL + 1, Q.T_CAL)[-1:].to(DEV)
    model = load(); kv = K.KV(model)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    log(f"{NAME}: moments from {ntok} tokens; groups of four from {src}")

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

    def put(l):
        for p, t in zip(kv.params(l), orig[l]):
            p.copy_(t.to(p.device))
    res = {"model": NAME, "exploratory": True, "source": src, "tau": TAU, "check_layers": CHECK, "base_check": base}
    # (1) the early layers converted with the matched groups of four
    for l in range(CHECK):
        fuse_plain_layer(kv, l, quads[l], S[l])
    r = ratios(check()); res["quads_early_check"] = r
    log(f"layers 0-{CHECK - 1} converted with the matched groups of four: check ratios {[round(x, 2) for x in r]}")
    # (2) which group of layer 1 carries it: each group left unmerged in turn (the other layers stay converted)
    trials = []
    for G in quads[1]:
        put(1); fuse_plain_layer(kv, 1, [g for g in quads[1] if g != G], S[1])
        r = ratios(check()); trials.append({"group_left_unmerged": G, "max_ratio": max(r[1:]), "ratios": r})
        log(f"  layer 1 without group {G}: largest ratio {max(r[1:]):.2f}")
    worst = min(trials, key=lambda t: t["max_ratio"])["group_left_unmerged"]
    put(1); fuse_plain_layer(kv, 1, [worst], S[1]); r = ratios(check())
    res.update({"layer1_trials": trials, "worst_group": worst, "worst_group_alone_ratios": r})
    log(f"  group {worst} alone in layer 1: largest ratio {max(r[1:]):.2f}")
    # (3) repair: forbid the group, re-match, re-fuse, until the check is under tau
    forbidden, steps, quads1 = {tuple(worst)}, [], None
    for step in range(REPAIR_STEPS):
        put(1); quads1 = quads_avoiding(kv, 1, S[1].double(), forbidden)
        fuse_plain_layer(kv, 1, quads1, S[1]); r = ratios(check())
        steps.append({"forbidden": [list(g) for g in sorted(forbidden)], "groups": quads1, "ratios": r, "max_ratio": max(r[1:])})
        log(f"  repair step {step + 1}: forbidding {sorted(forbidden)}; new groups {quads1}; largest ratio {max(r[1:]):.2f}")
        if max(r[1:]) <= TAU:
            break
        trials2 = []
        for G in quads1:
            put(1); fuse_plain_layer(kv, 1, [g for g in quads1 if g != G], S[1]); trials2.append((max(ratios(check())[1:]), G))
        forbidden.add(tuple(min(trials2)[1]))
    res["repair_steps"] = steps; res["repaired_layer1_groups"] = quads1; res["repaired"] = steps[-1]["max_ratio"] <= TAU
    if not res["repaired"]:                                 # fallback: layer 1 kept at its matched pairs (round 28's groups)
        put(1); fuse_plain_layer(kv, 1, pairs_all[1], S[1]); r = ratios(check())
        res["fallback_layer1_pairs_ratios"] = r
        log(f"  no grouping of four passed within {REPAIR_STEPS} steps; layer 1 at its matched pairs: largest ratio {max(r[1:]):.2f}")
        quads1 = pairs_all[1]; res["evaluated_layer1_groups"] = "matched pairs (fallback)"
    else:
        res["evaluated_layer1_groups"] = "repaired groups of four"
    json.dump(res, open(OUT, "w"), indent=1)
    # (4) the full conversion with the repaired layer 1, evaluated as quad_dev
    del model, kv, orig; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    model = load(); kv = K.KV(model)
    groups = [quads1 if l == 1 else quads[l] for l in range(kv.L)]
    t1 = time.time(); K.convert(kv, S, groups, "fuse"); log(f"full conversion with the repaired layer 1 in {time.time() - t1:.0f}s")
    r = ratios(check())
    tasks = {**Z.load_tasks(subset=SUBSET), **Q.R.extra_tasks(SUBSET)}; tasks = {k: tasks[k] for k in Q.R.ZS7}
    reqs = Z.build_requests(tok, tasks)
    ev = Z.evaluate_tasks(model, tok, tasks, reqs, DEV)
    ev["zs7"] = float(np.mean([ev[t][m] for t, m in Q.R.ZS7.items()]))
    ev["ppl_2048"] = Z.perplexity(model, Z.wikitext2_windows(tok, T=2048), DEV, bs=2)
    ev["check"] = r; ev["check_max_ratio"] = max(r)
    res["evals"] = {"fuse/quad_repaired" if res["repaired"] else "fuse/quad_layer1_pairs": ev}; res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"repaired conversion: ppl {ev['ppl_2048']:.3f}; ZS-7 {ev['zs7']:.4f}; one-window check {max(r):.2f}")
    log("QUAD_REPAIR_DONE")
