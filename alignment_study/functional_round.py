# FUNCTIONAL ROUND (PREREGISTRATION11.md): attention weighted by what it
# moves. Confirmatory models, all held out from the development calibration
# (functional_dev.py on Qwen2.5-0.5B), all with grouped-query attention so
# the round-7 merge is defined. Per model: F1 and F2 on every layer, and the
# merge outcome (KL and change in NLL) for the pairs fixed below. One JSON
# per model in alignment_study/functional/ as it completes; finished models
# are skipped on rerun.
#   python alignment_study/functional_round.py [--only=name1,name2]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import functional_common as F
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

# Merge plan fixed before any confirmatory outcome exists. Layers spread over
# depth; pairs per layer either all within-group pairs (None) or a seeded
# sample without replacement; n_eval evaluation windows of 512 tokens.
PLAN = {
    "Qwen/Qwen2.5-1.5B":        {"merge_layers": [1, 5, 9, 13, 17, 21, 24, 27], "pairs": None, "n_eval": 8},
    "TinyLlama/TinyLlama_v1.1": {"merge_layers": [1, 4, 7, 10, 13, 16, 19, 21], "pairs": 48, "n_eval": 8},
    "mistralai/Mistral-7B-v0.1": {"merge_layers": [2, 8, 14, 20, 26, 31], "pairs": None, "n_eval": 4},
    "Qwen/Qwen2.5-7B":          {"merge_layers": [2, 8, 14, 20, 26], "pairs": 48, "n_eval": 4},
}
OUT = "alignment_study/functional"
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)


def run_model(name, plan):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L, n, nkv, hd, group = F.geometry(model)
    print(name, "DEVICE", DEV, "DTYPE", dt, f"layers {L} heads {n} kv {nkv} hd {hd} group {group}", flush=True)
    M.N_SEL, M.T_SEL, M.N_EVAL, M.T_EVAL = F.N_SEL, F.T_SEL, plan["n_eval"], F.T_EVAL
    sel = M.selection_windows(tok); X = M.eval_windows(tok)
    meas = F.selection_measures(model, sel, list(range(L)), n, hd, group)
    print(f"selection measures done ({time.time()-t0:.0f}s)", flush=True)
    base = F.base_cache(model, X)
    pairs = {}
    for l in plan["merge_layers"]:
        allp = [tuple(map(int, k.split("-"))) for k in meas[l]["pairs"]]
        if plan["pairs"] is not None and plan["pairs"] < len(allp):
            rg = np.random.default_rng(1000 + l)
            allp = [allp[i] for i in sorted(rg.choice(len(allp), plan["pairs"], replace=False))]
        pairs[l] = allp
    sw = F.sweep(model, X, base, plan["merge_layers"], pairs, hd, log=lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True))
    check_kl, _ = F.kl_eval(model, X, base)
    del model; release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "layers": L, "heads": n, "kv_heads": nkv, "head_dim": hd,
            "plan": plan, "eval_tokens": int(X.numel()), "restore_check_kl": float(np.mean(check_kl)),
            "measures": {str(l): meas[l] for l in range(L)}, "sweep": {str(l): sw[l] for l in plan["merge_layers"]},
            "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name, plan in PLAN.items():
        if ONLY and name not in ONLY:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            res = run_model(name, plan)
            json.dump(res, open(path, "w"), indent=1)
            print(f"MODEL_DONE {name} {res['runtime_s']}s", flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True); print(traceback.format_exc()[-1200:], flush=True)
    print("FUNCTIONAL_ROUND_DONE", flush=True)
