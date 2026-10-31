# Development (exploratory, labelled): the fusion at a quarter of the key-value heads in 7B models, where no
# training-free result is published (CAGA reports 8 KV heads only after 100M tokens of continued training and calls
# 4 heads a failure for every method). Round 28's setting: 128 C4 validation documents of 2,048 tokens, WikiText-2
# perplexity on the whole test set at 2,048 tokens, ZS-7 (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy,
# ARC-Challenge, OpenBookQA) on the first SUBSET examples of each task. Arms: fuse/quad (kvfuse.quad_groups:
# minimum-cost pairs, then pairs of pairs by the four-head residual), fuse/neighbour (four neighbouring heads),
# mean/neighbour (averaging, the uptraining start of Ainslie et al.); fuse/paired (32 to 16, round 28's arm) on
# request. The model is converted in its own dtype, as in round 28 (no rebalancing: same function).
#   python alignment_study/quad_dev.py [--model=NousResearch/Llama-2-7b-hf] [--subset=500] [--pplwin=0 (whole test)]
#       [--arms=fuse/quad,fuse/neighbour,mean/neighbour]
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import numpy as np
import torch

argv0 = list(sys.argv)


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in argv0 if a.startswith(f"--{name}=")), default)


NAME = arg("model", "NousResearch/Llama-2-7b-hf")
REVISIONS = {"NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
             "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da"}
N_CAL, T_CAL, SUBSET = 128, 2048, int(arg("subset", 500))
N_PPL = int(arg("pplwin", 0)) or None
ARMS = tuple(arg("arms", "fuse/quad,fuse/neighbour,mean/neighbour").split(","))
OUT = "alignment_study/quad_dev"
sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import device_label
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.argv = [argv0[0], "--dev=" + NAME]                 # llama2_round reads its mode at import; only ZS7 and extra_tasks are used
import llama2_round as R
sys.argv = argv0
torch.set_grad_enabled(False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    path = f"{OUT}/{NAME.split('/')[-1]}.json"; partial = path + ".partial"
    if os.path.exists(path):
        sys.exit(f"exists: {path}")
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISIONS.get(NAME))
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISIONS.get(NAME), dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    assert kv.nkv % 4 == 0
    res = json.load(open(partial)) if os.path.exists(partial) else {"evals": {}}
    res.update({"model": NAME, "revision": REVISIONS.get(NAME), "exploratory": True, "dtype": "bfloat16", "layers": kv.L, "heads": kv.n,
                "kv_heads": kv.nkv, "calibration": [N_CAL, T_CAL], "subset": SUBSET, "ppl_windows": N_PPL, "zs7_metrics": R.ZS7})
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    log(f"{NAME}: second moments from {ntok} tokens")
    def load():
        return AutoModelForCausalLM.from_pretrained(NAME, revision=REVISIONS.get(NAME), dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)

    def restore():                                                       # reload from the cache: no 4 GB snapshot on a full disk
        global model, kv
        del model, kv; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        model = load(); kv = K.KV(model)

    class Stop(Exception):
        pass
    gw = C.windows_c4(tok, N_CAL + 1, T_CAL)[-1:].to(DEV)                # round 29's one-window check: a document after the calibration

    def check():
        """Each position's largest absolute residual entry after layers 0 to 5, median over positions after the first."""
        H = {}
        def mk(l):
            def hook(m, i, o):
                H[l] = (o[0] if isinstance(o, tuple) else o)[0]
                if l == 5:
                    raise Stop
            return hook
        hooks = [kv.blocks[l].register_forward_hook(mk(l)) for l in range(6)]
        try:
            model.model(input_ids=gw, use_cache=False)
        except Stop:
            pass
        for h in hooks:
            h.remove()
        return [float(H[l][1:].abs().amax(1).float().median()) for l in range(6)]
    if "base_check" not in res:
        res["base_check"] = check()
    neighbour4 = [[list(range(s, s + 4)) for s in range(0, kv.nkv, 4)] for _ in range(kv.L)]
    if "quads" not in res:
        t1 = time.time(); quads, pairs, cq, cn = [], [], [], []
        for l in range(kv.L):
            S64 = S[l].double()
            st = kvfuse.layer_stats(kv, l, S64)
            q = kvfuse.quad_groups(kv, l, S64); quads.append(q)
            pairs.append(kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S64)))
            cq.append(sum(kvfuse.group_cost(st, G) for G in q)); cn.append(sum(kvfuse.group_cost(st, G) for G in neighbour4[l]))
            del st
        res.update({"quads": quads, "pairs": pairs, "cost_quad": cq, "cost_neighbour4": cn})
        json.dump(res, open(partial, "w"))
        log(f"  groups of four: summed relative residual {sum(cq):.3f} matched against {sum(cn):.3f} neighbouring ({time.time() - t1:.0f}s)")
    groups = {"quad": res["quads"], "neighbour": neighbour4, "paired": res["pairs"]}
    tasks = {**Z.load_tasks(subset=SUBSET), **R.extra_tasks(SUBSET)}
    tasks = {k: tasks[k] for k in R.ZS7}
    reqs = Z.build_requests(tok, tasks)
    win = Z.wikitext2_windows(tok, T=2048)
    win = win[:N_PPL] if N_PPL else win

    def evaluate(key):
        if key in res["evals"]:
            log(f"  {key}: cached"); return
        t2 = time.time()
        r = Z.evaluate_tasks(model, tok, tasks, reqs, DEV)
        r["zs7"] = float(np.mean([r[t][m] for t, m in R.ZS7.items()]))
        r["ppl_2048"] = Z.perplexity(model, win, DEV, bs=2)
        r["check"] = check(); r["check_max_ratio"] = max(c / b for c, b in zip(r["check"], res["base_check"]))
        r["seconds"] = round(time.time() - t2, 1); res["evals"][key] = r
        json.dump(res, open(partial, "w"))
        log(f"  {key}: ppl {r['ppl_2048']:.3f}; ZS-7 {r['zs7']:.4f}; one-window check {r['check_max_ratio']:.2f} ({r['seconds']:.0f}s)")
    evaluate("base")
    for arm in ARMS:
        construction, grouping = arm.split("/")
        if arm not in res["evals"]:
            t3 = time.time(); K.convert(kv, S, groups[grouping], construction)
            log(f"  {arm}: converted in {time.time() - t3:.0f}s")
        evaluate(arm)
        restore()
    res["restore_check_ppl"] = Z.perplexity(model, win[:8], DEV, bs=2)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(path, "w"), indent=1)
    if os.path.exists(partial):
        os.remove(partial)
    del model, kv, S; gc.collect()
    log("QUAD_DEV_DONE")
