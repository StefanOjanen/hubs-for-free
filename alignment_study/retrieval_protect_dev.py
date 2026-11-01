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
#   python alignment_study/retrieval_protect_dev.py [--model=mistralai/Mistral-7B-Instruct-v0.2] [--thresholds=0.1,0.3] [--seed=0] [--with-base]
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


NAME = arg("model", "mistralai/Mistral-7B-Instruct-v0.2")
REV = ND.REVISIONS.get(NAME)
THRESHOLDS = [float(t) for t in arg("thresholds", "0.1,0.3").split(",")]
WITH_BASE = "--with-base" in sys.argv
SEED, N_PPL = int(arg("seed", 0)), 8
SCORES = f"audits/retrieval_heads/{NAME.split('/')[-1]}_scores.npy"
OUT = f"alignment_study/retrieval_protect_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def load():
    return AutoModelForCausalLM.from_pretrained(NAME, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)


def free():
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


def match_rest(cost, rest):
    """Minimum-cost pairs among the heads in rest; an odd count leaves out the head whose exclusion is cheapest."""
    if len(rest) < 2:
        return [], list(rest)
    if len(rest) % 2 == 0:
        sub = cost[rest][:, rest]
        return [[rest[a], rest[b]] for a, b in kvfuse.min_cost_pairs(sub)], []
    best = None
    for out in rest:
        keep = [i for i in rest if i != out]; sub = cost[keep][:, keep]
        pairs = kvfuse.min_cost_pairs(sub); total = float(sum(sub[a, b] for a, b in pairs))
        if best is None or total < best[0]:
            best = (total, [[keep[a], keep[b]] for a, b in pairs], [out])
    return best[1], best[2]


def groups_with(kv, costs, protected):
    """Per layer: the protected heads as groups of one, the rest matched by the residual."""
    groups, n = [], kv.nkv
    for l in range(kv.L):
        prot = sorted(i for (ll, i) in protected if ll == l)
        rest = [i for i in range(n) if i not in prot]
        pairs, left = match_rest(costs[l], rest)
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


if __name__ == "__main__":
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
    for name, protected in arms:
        t1 = time.time()
        if name != "unconverted":
            if name != arms[0][0]:                              # the first arm converts the model already loaded for the moments
                del model, kv; free(); model = load(); kv = K.KV(model)
            groups = groups_with(kv, costs, protected)
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
