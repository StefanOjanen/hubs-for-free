# Development (exploratory, labelled): does keeping the retrieval heads' key-value heads unmerged restore retrieval
# from context? needle_dev.py found Mistral-7B-Instruct-v0.2 retrieves a planted code 15 of 15 times unconverted and
# 2 of 15 after the plain fusion at half the key-value heads (perplexity 16.7 against 5.5 on 8 windows), answering
# in the right form with the wrong digits. Retrieval heads (Wu et al. 2024) are scored for this model in
# audits/retrieval_heads/<model>_scores.npy (layers by query heads; 35 query heads at 0.1 or more, 15 at 0.2). Arms,
# each a fresh load of the model under the same calibration moment (128 C4 documents of 2,048 tokens): unconverted;
# the plain fusion (every layer's residual matching); the key-value heads serving a query head scored at or above a
# threshold kept unmerged (groups of one are exact) with the rest matched by the residual (an odd remainder leaves
# the head whose exclusion gives the cheapest matching unmerged); the same number of key-value heads protected at
# random (seeded); and the same number protected by the fusion's own criterion (the heads whose cheapest pairing
# costs most). Per arm: needle_dev's 15 instances (five codes at three depths in a 2,048-token haystack), perplexity
# on the first 8 WikiText-2 windows of 2,048 tokens, the cache fraction, the protected heads. Writes
# alignment_study/retrieval_protect_dev/<model>.json after every arm.
# The two base arms (unconverted 15 of 15, plain fusion 2 of 15 in needle_dev.py, same haystack) run only with
# --with-base; the controls are drawn at the first threshold's count. Under paging an arm takes about 30 minutes.
# Follow-up arms: --protect=NAME:l-i,l-i;NAME2:... keeps the listed key-value heads unmerged; --random=K:SEED:EXCL;...
# protects K random heads drawn from those whose retrieval score is below EXCL (a control that cannot touch the
# retrieval heads); --thresholds= (empty) skips the threshold arms; --tag=NAME writes <model>_NAME.json.
# Registered mode (PREREGISTRATION35, draft): --registered=<freeze permalink> runs the registered population (ten codes at
# five depths, the arms of the preregistration, the unconverted model once, the exclusion rule) on the listed models and
# writes alignment_study/retrieval_round/<model>.json; --only=<model> restricts it.
#   python alignment_study/retrieval_protect_dev.py [--model=mistralai/Mistral-7B-Instruct-v0.2] [--thresholds=0.1,0.3] [--seed=0] [--with-base]
#   python alignment_study/retrieval_protect_dev.py --registered=<permalink> [--only=<model>]
import gc
import json
import os
import random
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
sys.argv = [argv0[0]]
import needle_dev as ND                                # the haystack, the planted facts, the prompt and the greedy answer
sys.argv = argv0
import gqa_common as C
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED = arg("registered")
REG_MODELS = ["Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"]      # the third is the development model, reported
REG_REVISIONS = {"Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796", "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
                 "mistralai/Mistral-7B-Instruct-v0.2": "63a8b081895390a26e140280378bc85ec8bce07a"}
REG_FACTS = [("blue", "48213"), ("green", "90457"), ("red", "27365"), ("yellow", "61928"), ("white", "35074"),
             ("black", "73506"), ("orange", "18649"), ("purple", "52937"), ("grey", "84071"), ("brown", "39615")]
REG_DEPTHS, REG_MIN_BASE = (10, 30, 50, 70, 90), 45
NAME = arg("model", "mistralai/Mistral-7B-Instruct-v0.2")
REV = ND.REVISIONS.get(NAME)
THRESHOLDS = [float(t) for t in arg("thresholds", "0.1,0.3").split(",") if t]
PROTECT, RANDOM, TAG = arg("protect", ""), arg("random", ""), arg("tag", "")
WITH_BASE = "--with-base" in sys.argv
SEED, N_PPL = int(arg("seed", 0)), 8
SCORES = f"audits/retrieval_heads/{NAME.split('/')[-1]}_scores.npy"
OUT = f"alignment_study/retrieval_protect_dev/{NAME.split('/')[-1]}{'_' + TAG if TAG else ''}.json"
torch.set_grad_enabled(False)


def load():
    return AutoModelForCausalLM.from_pretrained(NAME, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)


def free():
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


def match_rest(cost, rest, allowed_single=None):
    """Minimum-cost pairs among the heads in rest; an odd count leaves out the head whose exclusion is cheapest,
    chosen among allowed_single when given (the score-excluded random arms keep retrieval heads out of the leftovers)."""
    if len(rest) < 2:
        return [], list(rest)
    if len(rest) % 2 == 0:
        sub = cost[rest][:, rest]
        return [[rest[a], rest[b]] for a, b in kvfuse.min_cost_pairs(sub)], []
    best = None
    cands = [i for i in rest if allowed_single is None or i in allowed_single] or list(rest)
    for out in cands:
        keep = [i for i in rest if i != out]; sub = cost[keep][:, keep]
        pairs = kvfuse.min_cost_pairs(sub); total = float(sum(sub[a, b] for a, b in pairs))
        if best is None or total < best[0]:
            best = (total, [[keep[a], keep[b]] for a, b in pairs], [out])
    return best[1], best[2]


def groups_with(kv, costs, protected, allowed_single=None):
    """Per layer: the protected heads as groups of one, the rest matched by the residual; allowed_single, when given,
    is the set of (layer, head) that may be left unmerged by an odd remainder."""
    groups, n = [], kv.nkv
    for l in range(kv.L):
        prot = sorted(i for (ll, i) in protected if ll == l)
        rest = [i for i in range(n) if i not in prot]
        pairs, left = match_rest(costs[l], rest, None if allowed_single is None else {i for (ll, i) in allowed_single if ll == l})
        groups.append(sorted([[i] for i in prot + left] + [sorted(p) for p in pairs]))
    return groups


def needle_eval(model, tok, context_ids):
    inst = []
    for colour, code in ND.FACTS:
        needle = f" The secret code for the {colour} door is {code}. "
        question = f" Based on the content of the book, Question: What is the secret code for the {colour} door?\nAnswer:"
        for depth in ND.DEPTHS:
            ids = ND.build(tok, context_ids, needle, question, depth)
            a = ND.answer(model, tok, ids)
            inst.append({"colour": colour, "code": code, "depth": depth, "tokens": int(ids.shape[1]), "answer": a, "hit": code in a})
    return {"instances": inst, "hits": sum(i["hit"] for i in inst), "of": len(inst),
            "by_depth": {str(d): sum(i["hit"] for i in inst if i["depth"] == d) for d in ND.DEPTHS}}


def main_dev():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    rng = random.Random(SEED)
    scores = np.load(SCORES)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
    context_ids = tok.encode(ND.haystack_text(3 * ND.CTX), add_special_tokens=False)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, ND.N_CAL, ND.T_CAL)
    model = load(); kv = K.KV(model); n, L, rep = kv.nkv, kv.L, kv.rep
    assert scores.shape == (L, kv.n), f"{SCORES}: {scores.shape} against {L} layers and {kv.n} heads"
    S, ntok = K.second_moments(model, kv, cal, DEV)
    costs = {l: kvfuse.pair_costs(kv, l, S[l].double()) for l in range(L)}
    log(f"{NAME}: {L} layers, {kv.n} query heads over {n} key-value heads; moments from {ntok} tokens; retrieval scores from {SCORES}")
    res = {"model": NAME, "revision": REV, "exploratory": True, "scores": SCORES, "thresholds": THRESHOLDS, "seed": SEED, "calibration": [ND.N_CAL, ND.T_CAL],
           "needle": {"context_tokens": ND.CTX, "facts": len(ND.FACTS), "depths": list(ND.DEPTHS)}, "ppl_windows": N_PPL, "kv_heads_total": L * n, "arms": {}}
    arms = [("unconverted", None), ("fusion", set())] if WITH_BASE else []
    ALLOWED = {}
    for th in THRESHOLDS:
        P = sorted({(int(l), int(h) // rep) for l, h in np.argwhere(scores >= th)})
        arms.append((f"protect_retrieval_{th}", set(P)))
        k = len(P)
        if th != THRESHOLDS[0]:
            continue
        allheads = [(l, i) for l in range(L) for i in range(n)]
        arms.append((f"protect_random_{k}", set(rng.sample(allheads, k))))
        cheapest = {(l, i): float(min(costs[l][i, j] for j in range(n) if j != i)) for l in range(L) for i in range(n)}
        arms.append((f"protect_residual_{k}", set(sorted(cheapest, key=lambda x: -cheapest[x])[:k])))
    for spec in [x for x in PROTECT.split(";") if x]:
        nm, heads = spec.split(":")
        arms.append((f"protect_{nm}", {(int(h.split("-")[0]), int(h.split("-")[1])) for h in heads.split(",")}))
    for spec in [x for x in RANDOM.split(";") if x]:
        k, sd, ex = int(spec.split(":")[0]), int(spec.split(":")[1]), float(spec.split(":")[2])
        pool = [(l, i) for l in range(L) for i in range(n) if scores[l, i * rep:(i + 1) * rep].max() < ex]
        arms.append((f"protect_random_{k}_below{ex}_seed{sd}", set(random.Random(sd).sample(pool, k))))
        ALLOWED[f"protect_random_{k}_below{ex}_seed{sd}"] = set(pool)                   # leftovers below the score too
    res["arm_specs"] = {"protect": PROTECT, "random": RANDOM, "leftovers_restricted_for": sorted(ALLOWED)}
    for name, protected in arms:
        t1 = time.time()
        if name != "unconverted":
            if name != arms[0][0]:                              # the first arm converts the model already loaded for the moments
                del model, kv; free(); model = load(); kv = K.KV(model)
            groups = groups_with(kv, costs, protected, ALLOWED.get(name))
            K.convert(kv, S, groups, "fuse")
            kept = sum(len(g) for g in groups)
        else:
            groups, kept = None, L * n
        ne = needle_eval(model, tok, context_ids)
        ppl = Z.perplexity(model, win, DEV, bs=2)
        res["arms"][name] = {"protected": sorted(protected) if protected else [], "n_protected": len(protected) if protected else 0, "kv_heads_kept": kept,
                             "cache_fraction": kept / (L * n), "groups": groups, "ppl_8_windows": ppl, "needle": ne, "seconds": round(time.time() - t1, 1)}
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"{NAME}, {name}: {ne['hits']} of {ne['of']} retrieved; by depth {ne['by_depth']}; ppl {ppl:.3f}; cache {kept / (L * n):.3f} ({kept} of {L * n} key-value heads)")
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log("RETRIEVAL_PROTECT_DONE")


def run_registered(name, rev, log):
    """PREREGISTRATION35's population on one model: the unconverted model, then the arms (a) to (f), 50 instances each."""
    ND.FACTS, ND.DEPTHS = REG_FACTS, REG_DEPTHS
    scores_path = f"audits/retrieval_heads/{name.split('/')[-1]}_scores.npy"
    out = f"alignment_study/retrieval_round/{name.split('/')[-1]}.json"
    if not os.path.exists(scores_path):
        log(f"{name}: no retrieval scores at {scores_path}; skipped"); return None
    scores = np.load(scores_path)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    context_ids = tok.encode(ND.haystack_text(3 * ND.CTX), add_special_tokens=False)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, ND.N_CAL, ND.T_CAL)
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model); n, L, rep = kv.nkv, kv.L, kv.rep
    assert scores.shape == (L, kv.n), f"{scores_path}: {scores.shape} against {L} layers and {kv.n} heads"
    res = {"model": name, "revision": rev, "registered": REGISTERED, "development": name == "mistralai/Mistral-7B-Instruct-v0.2", "scores": scores_path,
           "needle": {"context_tokens": ND.CTX, "facts": len(REG_FACTS), "depths": list(REG_DEPTHS)}, "ppl_windows": N_PPL, "kv_heads_total": L * n, "arms": {}}
    ne = needle_eval(model, tok, context_ids); ppl = Z.perplexity(model, win, DEV, bs=2)
    res["arms"]["unconverted"] = {"n_protected": 0, "kv_heads_kept": L * n, "cache_fraction": 1.0, "ppl_8_windows": ppl, "needle": ne}
    log(f"{name}, unconverted: {ne['hits']} of {ne['of']} retrieved; ppl {ppl:.3f}")
    res["excluded"] = ne["hits"] < REG_MIN_BASE
    json.dump(res, open(out, "w"), indent=1)
    if res["excluded"]:
        log(f"{name}: unconverted retrieval {ne['hits']} of {ne['of']} is below {REG_MIN_BASE}; excluded before any converted arm"); return res
    S, ntok = K.second_moments(model, kv, cal, DEV)
    costs = {l: kvfuse.pair_costs(kv, l, S[l].double()) for l in range(L)}
    P3 = sorted({(int(l), int(h) // rep) for l, h in np.argwhere(scores >= 0.3)}); P1 = sorted({(int(l), int(h) // rep) for l, h in np.argwhere(scores >= 0.1)})
    pool = [(l, i) for l in range(L) for i in range(n) if scores[l, i * rep:(i + 1) * rep].max() < 0.1]
    cheapest = {(l, i): float(min(costs[l][i, j] for j in range(n) if j != i)) for l in range(L) for i in range(n)}
    arms = [("a_fusion", set(), None), ("b_retrieval_0.3", set(P3), None), ("c_retrieval_0.1", set(P1), None)]
    for sd in (0, 1):
        arms.append((f"d_random_{len(P3)}_seed{sd}", set(random.Random(sd).sample(pool, min(len(P3), len(pool)))), set(pool)))
        arms.append((f"e_random_{len(P1)}_seed{sd}", set(random.Random(100 + sd).sample(pool, min(len(P1), len(pool)))), set(pool)))
    arms.append(("f_residual", set(sorted(cheapest, key=lambda x: -cheapest[x])[:len(P3)]), None))
    res["counts"] = {"retrieval_0.3": len(P3), "retrieval_0.1": len(P1), "pool_below_0.1": len(pool)}
    for aname, protected, allowed in arms:
        t1 = time.time()
        del model, kv; free(); model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV); kv = K.KV(model)
        groups = groups_with(kv, costs, protected, allowed); K.convert(kv, S, groups, "fuse"); kept = sum(len(g) for g in groups)
        ne = needle_eval(model, tok, context_ids); ppl = Z.perplexity(model, win, DEV, bs=2)
        res["arms"][aname] = {"protected": sorted(protected), "n_protected": len(protected), "kv_heads_kept": kept, "cache_fraction": kept / (L * n), "groups": groups,
                              "ppl_8_windows": ppl, "needle": ne, "seconds": round(time.time() - t1, 1)}
        json.dump(res, open(out, "w"), indent=1)
        log(f"{name}, {aname}: {ne['hits']} of {ne['of']} retrieved; by depth {ne['by_depth']}; ppl {ppl:.3f}; cache {kept / (L * n):.3f}")
    del model, kv, S; free()
    return res


if __name__ == "__main__":
    if REGISTERED:
        os.makedirs("alignment_study/retrieval_round", exist_ok=True)
        t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
        only = arg("only")
        for name in REG_MODELS:
            if (only and name != only) or os.path.exists(f"alignment_study/retrieval_round/{name.split('/')[-1]}.json"):
                continue
            run_registered(name, REG_REVISIONS[name], log)
        log("RETRIEVAL_ROUND_DONE")
    else:
        main_dev()
