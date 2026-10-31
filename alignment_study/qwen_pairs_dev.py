# Development (exploratory, labelled): Qwen2.5-7B converted to half its key-value heads with pairs matched by the
# fusion's residual (kvfuse.pair_costs accepts projections with biases since 72b7752) against the neighbouring
# pairs rounds 27, 30, 31 and 32 used; round 28's calibration (128 C4 validation documents of 2,048 tokens);
# WikiText-2 perplexity on 8 windows of 2,048 tokens; bfloat16 as in those rounds (their neighbour value: 23.873).
# Qwen2.5-7B has 4 KV heads per layer, so the matching chooses among three pairings per layer.
#   python alignment_study/qwen_pairs_dev.py
import gc
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
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, REV = "Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"
N_CAL, T_CAL, N_PPL = 128, 2048, 8
OUT = "alignment_study/qwen_pairs_dev.json"
torch.set_grad_enabled(False)


def load():
    return AutoModelForCausalLM.from_pretrained(NAME, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    model = load(); kv = K.KV(model)
    res = {"model": NAME, "revision": REV, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "base": Z.perplexity(model, win, DEV, bs=2)}
    S, ntok = K.second_moments(model, kv, cal, DEV)
    assert S[0].shape[0] == kv.d + 1
    costs = [kvfuse.pair_costs(kv, l, S[l].double()) for l in range(kv.L)]
    matched = [kvfuse.min_cost_pairs(c) for c in costs]
    neighbour = [[[0, 1], [2, 3]] for _ in range(kv.L)]
    res.update({"matched": matched, "layers_where_matched_is_neighbour": sum(m == n for m, n in zip(matched, neighbour)), "layers": kv.L,
                "cost_matched": [float(sum(c[i, j] for i, j in m)) for c, m in zip(costs, matched)],
                "cost_neighbour": [float(sum(c[i, j] for i, j in n)) for c, n in zip(costs, neighbour)]})
    log(f"moments from {ntok} tokens; matched pairing equals neighbours in {res['layers_where_matched_is_neighbour']} of {kv.L} layers; "
        f"summed residual {sum(res['cost_matched']):.3f} against {sum(res['cost_neighbour']):.3f}")
    for arm, groups in (("matched", matched), ("neighbour", neighbour)):
        t1 = time.time()
        K.convert(kv, S, groups, "fuse")
        res[arm] = {"ppl": Z.perplexity(model, win, DEV, bs=2), "seconds": round(time.time() - t1, 1)}
        log(f"{arm}: ppl {res[arm]['ppl']:.3f}")
        json.dump(res, open(OUT, "w"), indent=1)
        del model, kv; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        if arm == "matched":
            model = load(); kv = K.KV(model)
    log("QWEN_PAIRS_DONE")
